"""The sweep driver: train once, evaluate the whole quantisation matrix.

The cost model here is deliberate.  Training is expensive relative to
evaluating a quantised copy, so the reference model is trained exactly once and
every configuration is measured against the same weights.  That makes the rows
of the report directly comparable: any difference between two rows is caused by
the quantisation settings, not by a different training run.

Each configuration produces a :class:`SweepRow` — a flat, serialisable record
carrying the compression cost, the weight-space error, and the behavioural
metrics.  Rows are the unit the reporter and the CLI consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibration import CalibrationResult, calibrate
from .config import QuantConfig, SweepConfig
from .data import Dataset, build_dataset, calibration_subset
from .dtypes import NumericFormat, compression_ratio, quant_error_stats
from .metrics import compare_predictions
from .model import MLPParams, TrainHistory, accuracy, build_and_train, forward
from .quantize import QuantParams, quantize_layer_weights
from .sensitivity import (
    apply_mixed_precision,
    cumulative_sensitivity,
    parameter_share,
    rank_layers,
    recommend_mixed_precision,
    sensitivity_summary,
    single_layer_sensitivity,
)


@dataclass
class QuantizedModel:
    """A compressed copy of the reference weights plus the parameters used.

    The per-layer :class:`QuantParams` are retained because a report reader
    needs to know the scale spread: a layer whose per-channel scales differ by
    orders of magnitude is a layer where per-tensor quantisation must lose
    range.
    """

    params: MLPParams
    quant_params: dict[str, QuantParams]
    layer_stats: dict[str, dict[str, float]]
    format: NumericFormat
    scheme: str
    granularity: str

    def scale_spread(self) -> dict[str, float]:
        """Per-layer ratio between the largest and smallest channel scale."""
        out: dict[str, float] = {}
        for name, qp in self.quant_params.items():
            scale = np.asarray(qp.scale, dtype=np.float64)
            lo = float(scale.min())
            hi = float(scale.max())
            out[name] = float(hi / lo) if lo > 0 else float("inf")
        return out

    def total_storage_bytes(self) -> int:
        """Bytes the quantised weights would occupy, scales excluded."""
        from .dtypes import storage_bytes

        return sum(storage_bytes(w.shape, self.format) for w in self.params.weights)


@dataclass
class SweepRow:
    """One configuration's measured result, ready to serialise."""

    label: str
    format: str
    scheme: str
    granularity: str
    compression_ratio: float
    mean_weight_rmse: float
    max_weight_rmse: float
    accuracy: dict[str, float]
    churn: dict[str, float]
    confidence: dict[str, float]
    margin: dict[str, float]
    rank: dict[str, float]
    calibration_error: dict[str, float]
    layer_errors: dict[str, float]
    scale_spread: dict[str, float]
    per_class_candidate: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Flatten to a JSON-friendly dict, rounding floats for readability."""
        return {
            "label": self.label,
            "format": self.format,
            "scheme": self.scheme,
            "granularity": self.granularity,
            "compression_ratio": round(self.compression_ratio, 4),
            "weight_error": {
                "mean_rmse": round(self.mean_weight_rmse, 8),
                "max_rmse": round(self.max_weight_rmse, 8),
                "per_layer": {k: round(v, 8) for k, v in self.layer_errors.items()},
            },
            "accuracy": {k: _round(v) for k, v in self.accuracy.items()},
            "churn": {k: _round(v) for k, v in self.churn.items()},
            "confidence": {k: _round(v) for k, v in self.confidence.items()},
            "margin": {k: _round(v) for k, v in self.margin.items()},
            "rank": {k: _round(v) for k, v in self.rank.items()},
            "calibration_error": {k: _round(v) for k, v in self.calibration_error.items()},
            "scale_spread": {k: _round(v) for k, v in self.scale_spread.items()},
            "per_class_candidate": {k: _round(v) for k, v in self.per_class_candidate.items()},
        }


def _round(value: float, digits: int = 6) -> float:
    """Round a float, passing NaN and infinities through unchanged."""
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return value
    return round(float(value), digits)


@dataclass
class SweepResult:
    """Everything one sweep produced: rows, baseline, and diagnostics."""

    rows: list[SweepRow]
    baseline_accuracy: float
    history: TrainHistory
    dataset: Dataset
    layer_parameter_share: list[dict[str, object]]
    sensitivity: dict[str, Any]
    cumulative: list[dict[str, Any]]
    mixed_precision: dict[str, Any] | None
    calibration: dict[str, Any]
    reference_accuracy: float

    def worst_row(self, by: str = "accuracy_delta") -> dict[str, Any] | None:
        """The row with the largest accuracy regression."""
        if not self.rows:
            return None
        worst = min(self.rows, key=lambda r: r.accuracy.get(by, 0.0))
        return worst.as_dict()

    def best_row(self, by: str = "accuracy_delta") -> dict[str, Any] | None:
        """The row with the best accuracy, breaking ties on compression ratio."""
        if not self.rows:
            return None
        best = max(self.rows, key=lambda r: (r.accuracy.get(by, 0.0), r.compression_ratio))
        return best.as_dict()

    def churn_outliers(self, threshold: float = 0.02) -> list[dict[str, Any]]:
        """Rows where inputs flip far more often than the accuracy delta suggests.

        These are the interesting cases: ``|accuracy_delta|`` is small but
        ``flip_rate`` is large, meaning the model is trading correct answers for
        correct answers while changing its behaviour.
        """
        out: list[dict[str, Any]] = []
        for row in self.rows:
            acc_delta = abs(row.accuracy.get("accuracy_delta", 0.0))
            flip_rate = row.churn.get("flip_rate", 0.0)
            if flip_rate > threshold and flip_rate > 5.0 * max(acc_delta, 1e-9):
                out.append(
                    {
                        "label": row.label,
                        "flip_rate": round(flip_rate, 6),
                        "accuracy_delta": round(acc_delta, 6),
                        "ratio": round(flip_rate / max(acc_delta, 1e-9), 2),
                    }
                )
        return sorted(out, key=lambda r: r["ratio"], reverse=True)


def quantize_model(
    params: MLPParams,
    fmt: NumericFormat,
    scheme: str,
    granularity: str,
    seed: int | None = None,
    quantize_biases: bool = False,
) -> QuantizedModel:
    """Quantise every weight matrix, optionally including the bias vectors.

    Biases are left in float by default.  They are one vector per layer rather
    than a matrix, so per-channel quantisation does not apply to them, and
    including them confounds the attribution of damage to *weight* precision.
    ``config.yaml``'s ``weight_layers_only`` controls this.
    """
    out = params.copy()
    quant_params: dict[str, QuantParams] = {}
    layer_stats: dict[str, dict[str, float]] = {}
    for i, w in enumerate(params.weights):
        quantised, qp, stats = quantize_layer_weights(
            w, fmt=fmt, scheme=scheme, granularity=granularity, seed=seed
        )
        out.weights[i] = quantised
        name = f"layers.{i}.weight"
        quant_params[name] = qp
        layer_stats[name] = stats
    if quantize_biases:
        for i, b in enumerate(params.biases):
            quantised, qp, stats = quantize_layer_weights(
                b, fmt=fmt, scheme=scheme, granularity="per_tensor", seed=seed
            )
            out.biases[i] = quantised
            quant_params[f"layers.{i}.bias"] = qp
            layer_stats[f"layers.{i}.bias"] = stats
    return QuantizedModel(
        params=out,
        quant_params=quant_params,
        layer_stats=layer_stats,
        format=fmt,
        scheme=scheme,
        granularity=granularity,
    )


def evaluate_model(
    params: MLPParams,
    dataset: Dataset,
    activation: str,
) -> tuple[np.ndarray, float]:
    """Return test probabilities and accuracy for a parameter set."""
    probs, _ = forward(params, dataset.x_test, activation)
    return probs, accuracy(probs, dataset.y_test)
