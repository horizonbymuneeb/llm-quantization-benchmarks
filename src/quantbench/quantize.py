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


def _reduce_axes(weights: np.ndarray, axis: int) -> tuple[int, ...]:
    """All axes except ``axis``; the reduction set for per-channel statistics."""
    return tuple(i for i in range(weights.ndim) if i != axis)


def symmetric_params(
    weights: np.ndarray,
    fmt: NumericFormat,
    granularity: str = "per_channel",
    axis: int = 0,
) -> QuantParams:
    """Derive symmetric scale(s): ``scale = max(|w|) / qmax``.

    The zero point is always 0 because a symmetric grid is centred on zero.
    ``qmax`` comes from the format's real integer range so INT4 and INT8 do not
    share a magic constant.
    """
    arr = np.asarray(weights, dtype=np.float32)
    if granularity not in ("per_tensor", "per_channel"):
        raise ValueError(f"unknown granularity {granularity!r}")
    spec = FORMAT_SPECS[fmt]
    if not spec.fmt.is_integer:
        raise TypeError(f"symmetric_params needs an integer format, got {fmt.value}")
    _, qmax = spec.integer_range

    if granularity == "per_tensor":
        amax = float(np.max(np.abs(arr))) if arr.size else 0.0
        scale = np.array([amax / qmax if amax > 0 else 1.0], dtype=np.float32)
        return QuantParams(scale=scale, zero_point=np.zeros(1, np.int32), axis=None, fmt=fmt)

    ax = _check_axis(arr, axis)
    reduce_axes = _reduce_axes(arr, ax)
    amax = np.max(np.abs(arr), axis=reduce_axes) if reduce_axes else np.abs(arr)
    scale = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)
    return QuantParams(
        scale=np.atleast_1d(scale),
        zero_point=np.zeros(np.atleast_1d(scale).shape, np.int32),
        axis=ax,
        fmt=fmt,
    )
