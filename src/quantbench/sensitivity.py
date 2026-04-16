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
