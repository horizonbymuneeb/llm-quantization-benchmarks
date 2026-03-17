"""Quantisation of weight matrices, per-tensor and per-channel.

The module produces *dequantised* float32 tensors rather than packed integers.
That choice is deliberate: the benchmark is about how compression changes model
behaviour, not about writing a fast kernel, so every quantised layer still runs
through the same numpy forward pass and the only thing that differs is the
numbers it sees.  Packing is simulated separately by
:func:`quantbench.dtypes.storage_bytes` when compression ratios are reported.

Symmetric quantisation maps ``[-a, a]`` onto a signed grid with a representable
zero.  Asymmetric quantisation maps ``[lo, hi]`` onto the full unsigned grid and
carries a zero point, which recovers range when the distribution is not centred
(such as post-ReLU activations) at the cost of one extra term per dot product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtypes import (
    FORMAT_SPECS,
    NumericFormat,
    quantize_to_grid,
    storage_bytes,
)


@dataclass(frozen=True)
class QuantParams:
    """Scale (and zero point) for one tensor or one channel.

    ``scale`` is broadcastable against the axis the parameters were derived
    along, which lets the same object serve per-tensor and per-channel without
    the caller reshaping anything.
    """

    scale: np.ndarray
    zero_point: np.ndarray
    axis: int | None
    fmt: NumericFormat

    @property
    def is_per_channel(self) -> bool:
        """True when there is more than one scale."""
        return self.scale.size > 1

    @property
    def n_channels(self) -> int:
        """Number of independent scales held by this object."""
        return int(self.scale.size)

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly summary; full arrays would bloat the report."""
        return {
            "format": self.fmt.value,
            "axis": self.axis,
            "n_channels": self.n_channels,
            "scale_min": float(np.min(self.scale)),
            "scale_max": float(np.max(self.scale)),
            "zero_point_min": int(np.min(self.zero_point)),
            "zero_point_max": int(np.max(self.zero_point)),
        }


def _check_axis(weights: np.ndarray, axis: int | None) -> int:
    """Validate that ``axis`` indexes a real dimension of ``weights``."""
    if weights.ndim == 0:
        raise ValueError("cannot quantise a scalar tensor")
    if axis is None:
        return 0
    if not -weights.ndim <= axis < weights.ndim:
        raise ValueError(f"axis {axis} out of range for tensor with {weights.ndim} dims")
    return axis % weights.ndim
