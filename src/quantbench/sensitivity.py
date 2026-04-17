"""Per-layer sensitivity: which layers can be compressed and which cannot.

The question a practitioner actually asks is not "what happens at INT8" but
"where in the network does the damage happen".  This module answers it three
ways, because they disagree in informative ways:

1. **Weight-space sensitivity** quantises one layer at a time, keeping every
   other layer at full precision, and measures the drop in test accuracy.  It
   is cheap and it isolates the layer, but it ignores error accumulation.
2. **Activation-space sensitivity** compares the layer's output before and
   after quantisation of its weights.  It catches a layer whose weights look
   harmless but whose outputs feed a nonlinearity near a decision boundary.
3. **Cumulative sensitivity** quantises layers 0..k and reports accuracy at
   each prefix.  Error accumulation is superadditive in practice, and this is
   the curve that shows it.

All three are run on a fixed test split so the numbers are comparable between
layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtypes import NumericFormat, simulate_format
from .model import MLPParams, forward
from .quantize import quantize_tensor, quantization_error


@dataclass(frozen=True)
class LayerSensitivity:
    """Result for a single layer at a single format."""

    layer: str
    index: int
    fmt: NumericFormat
    accuracy: float
    accuracy_drop: float
    output_relative_error: float
    weight_rmse: float
    weight_cosine_distance: float
    n_parameters: int

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly view used directly in the report."""
        return {
            "layer": self.layer,
            "index": self.index,
            "format": self.fmt.value,
            "accuracy": round(self.accuracy, 6),
            "accuracy_drop": round(self.accuracy_drop, 6),
            "output_relative_error": round(self.output_relative_error, 8),
            "weight_rmse": round(self.weight_rmse, 8),
            "weight_cosine_distance": round(self.weight_cosine_distance, 8),
            "n_parameters": self.n_parameters,
        }


def _accuracy_of(params: MLPParams, x: np.ndarray, y: np.ndarray, activation: str) -> float:
    """Top-1 accuracy of a parameter set on a fixed split."""
    probs, _ = forward(params, x, activation)
    return float(np.mean(np.argmax(probs, axis=1) == y))


def quantise_one_layer(
    params: MLPParams,
    layer_index: int,
    fmt: NumericFormat,
    scheme: str = "symmetric",
    granularity: str = "per_channel",
) -> MLPParams:
    """Return a copy of ``params`` with exactly one weight matrix quantised.

    Biases are left in float on purpose: the sensitivity question is about the
    weight matrices, and quantising biases at the same time would confound the
    attribution.
    """
    if not 0 <= layer_index < params.n_layers:
        raise IndexError(
            f"layer_index {layer_index} out of range for {params.n_layers} layers"
        )
    out = params.copy()
    quantised, _ = quantize_tensor(
        params.weights[layer_index],
        fmt=fmt,
        scheme=scheme,
        granularity=granularity,
        axis=0,
    )
    out.weights[layer_index] = quantised
    return out


def layer_output_error(
    params: MLPParams,
    quantised: MLPParams,
    x: np.ndarray,
    layer_index: int,
    activation: str = "relu",
) -> float:
    """Relative L2 change in one layer's *input* representation.

    The layer's input is what the previous layer produced; comparing the same
    layer's input under both parameter sets would be identical, so the
    meaningful comparison is the layer's own output, computed from the shared
    input.  ``output_relative_error`` therefore isolates the effect of
    quantising this layer's weights on the values the next layer receives.
    """
    _, ref_cache = forward(params, x, activation, record=True)
    _, cand_cache = forward(quantised, x, activation, record=True)
    assert ref_cache is not None and cand_cache is not None
    if layer_index < len(ref_cache.post_activations):
        ref = ref_cache.post_activations[layer_index]
        cand = cand_cache.post_activations[layer_index]
    else:
        ref = ref_cache.probabilities
        cand = cand_cache.probabilities
    assert ref is not None and cand is not None
    denom = float(np.linalg.norm(ref)) or 1.0
    return float(np.linalg.norm(cand - ref) / denom)


def single_layer_sensitivity(
    params: MLPParams,
    x_test: np.ndarray,
    y_test: np.ndarray,
    fmt: NumericFormat,
    activation: str = "relu",
    scheme: str = "symmetric",
    granularity: str = "per_channel",
) -> list[LayerSensitivity]:
    """Quantise each layer alone and record the resulting accuracy drop.

    This is the O(n_layers) sweep that produces the ranking.  Layers are
    reported in index order; ranking is a separate function because callers
    sometimes want the original order.
    """
    baseline = _accuracy_of(params, x_test, y_test, activation)
    results: list[LayerSensitivity] = []
    for i in range(params.n_layers):
        quantised = quantise_one_layer(params, i, fmt, scheme, granularity)
        acc = _accuracy_of(quantised, x_test, y_test, activation)
        w_err = quantization_error(params.weights[i], quantised.weights[i])
        results.append(
            LayerSensitivity(
                layer=f"layers.{i}.weight",
                index=i,
                fmt=fmt,
                accuracy=acc,
                accuracy_drop=baseline - acc,
                output_relative_error=layer_output_error(
                    params, quantised, x_test, i, activation
                ),
                weight_rmse=w_err["rmse"],
                weight_cosine_distance=w_err["cosine_distance"],
                n_parameters=int(params.weights[i].size),
            )
        )
    return results


def rank_layers(
    results: list[LayerSensitivity],
    by: str = "output_relative_error",
) -> list[LayerSensitivity]:
    """Sort layers most-sensitive first.

    ``output_relative_error`` is the default ranking key because accuracy drop
    is quantised into steps of ``1/n_test`` and often ties across layers, while
    the output error is continuous and therefore ranks stably.
    """
    if not results:
        return []
    key = by if by in results[0].as_dict() else "output_relative_error"
    return sorted(results, key=lambda r: float(getattr(r, key)), reverse=True)


def cumulative_sensitivity(
    params: MLPParams,
    x_test: np.ndarray,
    y_test: np.ndarray,
    fmt: NumericFormat,
    activation: str = "relu",
    scheme: str = "symmetric",
    granularity: str = "per_channel",
) -> list[dict[str, float | int | str]]:
    """Accuracy after quantising layers ``0..k`` for every ``k``.

    The increments between consecutive points are what reveal accumulation: if
    quantising layer 3 alone costs 0.5 points but quantising 0..3 costs 4
    points, the damage is interaction between layers, not any single layer.
    """
    baseline = _accuracy_of(params, x_test, y_test, activation)
    curve: list[dict[str, float | int | str]] = [
        {"n_layers_quantised": 0, "accuracy": baseline, "incremental_drop": 0.0}
    ]
    running = params.copy()
    previous = baseline
    for i in range(params.n_layers):
        quantised, _ = quantize_tensor(
            running.weights[i], fmt=fmt, scheme=scheme, granularity=granularity, axis=0
        )
        running.weights[i] = quantised
        acc = _accuracy_of(running, x_test, y_test, activation)
        curve.append(
            {
                "n_layers_quantised": i + 1,
                "layer": f"layers.{i}.weight",
                "accuracy": acc,
                "incremental_drop": previous - acc,
                "cumulative_drop": baseline - acc,
            }
        )
        previous = acc
    return curve


def format_sensitivity(
    params: MLPParams,
    x_test: np.ndarray,
    y_test: np.ndarray,
    formats: tuple[NumericFormat, ...],
    activation: str = "relu",
    layer_index: int = 0,
) -> list[dict[str, object]]:
    """Hold one layer fixed and vary the storage format.

    This is the "what does width alone cost" probe: scheme and granularity are
    fixed at their simplest settings so the only variable is the number of bits.
    """
    if not 0 <= layer_index < params.n_layers:
        raise IndexError(f"layer_index {layer_index} out of range")
    baseline = _accuracy_of(params, x_test, y_test, activation)
    rows: list[dict[str, object]] = []
    for fmt in formats:
        quantised = params.copy()
        quantised.weights[layer_index] = simulate_format(params.weights[layer_index], fmt)
        acc = _accuracy_of(quantised, x_test, y_test, activation)
        rows.append(
            {
                "format": fmt.value,
                "accuracy": round(acc, 6),
                "accuracy_drop": round(baseline - acc, 6),
                "weight_rmse": round(
                    quantization_error(
                        params.weights[layer_index], quantised.weights[layer_index]
                    )["rmse"],
                    8,
                ),
            }
        )
    return rows


def parameter_share(params: MLPParams) -> list[dict[str, object]]:
    """Fraction of *all* parameters held by each weight matrix.

    The denominator includes the bias vectors, so the shares sum to just under
    1.0 and the remainder is the biases.  That is the honest accounting: a
    sensitivity ranking is only actionable next to the size of each layer, and a
    sensitive layer holding 2% of the parameters can stay in float while the
    rest is compressed.
    """
    total = max(params.total_parameters(), 1)
    rows: list[dict[str, object]] = []
    for i, w in enumerate(params.weights):
        n = int(w.size)
        rows.append(
            {
                "layer": f"layers.{i}.weight",
                "index": i,
                "n_parameters": n,
                "share": round(n / total, 6),
                "share_of_weights": round(n / max(sum(w.size for w in params.weights), 1), 6),
                "shape": list(w.shape),
            }
        )
    return rows


def recommend_mixed_precision(
    ranked: list[LayerSensitivity],
    top_k: int,
    sensitive_format: NumericFormat = NumericFormat.BF16,
    default_format: NumericFormat = NumericFormat.INT8,
) -> dict[str, object]:
    """Suggest keeping the ``top_k`` most sensitive layers at a wider format.

    This is a recommendation derived from measured sensitivity, not a claim
    that the mixed-precision assignment is optimal — an exhaustive search over
    assignments is exponential in the number of layers.  The caller can evaluate
    the suggestion with :func:`apply_mixed_precision`.
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    keep = {r.index for r in ranked[:top_k]}
    return {
        "sensitive_layers": sorted(keep),
        "sensitive_format": sensitive_format.value,
        "default_format": default_format.value,
        "n_layers_wide": len(keep),
    }
