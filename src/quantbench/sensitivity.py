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
