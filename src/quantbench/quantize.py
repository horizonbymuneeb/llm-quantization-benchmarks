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


def asymmetric_params(
    weights: np.ndarray,
    fmt: NumericFormat,
    granularity: str = "per_channel",
    axis: int = 0,
) -> QuantParams:
    """Derive asymmetric scale(s) and zero point(s) from the observed range.

    ``scale = (hi - lo) / (qmax - qmin)`` and the zero point is chosen so that
    float ``0.0`` maps exactly onto an integer code.  That exactness is the
    reason asymmetric mode is worth its extra arithmetic: zero padding and
    zero-padded convolutions stay bit-exact.
    """
    arr = np.asarray(weights, dtype=np.float32)
    if granularity not in ("per_tensor", "per_channel"):
        raise ValueError(f"unknown granularity {granularity!r}")
    spec = FORMAT_SPECS[fmt]
    if not spec.fmt.is_integer:
        raise TypeError(f"asymmetric_params needs an integer format, got {fmt.value}")
    qmin, qmax = spec.integer_range

    if granularity == "per_tensor":
        lo, hi = float(arr.min()), float(arr.max())
        scale, zp = _range_to_params(lo, hi, qmin, qmax)
        return QuantParams(
            scale=np.array([scale], np.float32),
            zero_point=np.array([zp], np.int32),
            axis=None,
            fmt=fmt,
        )

    ax = _check_axis(arr, axis)
    reduce_axes = _reduce_axes(arr, ax)
    lo = np.min(arr, axis=reduce_axes) if reduce_axes else arr
    hi = np.max(arr, axis=reduce_axes) if reduce_axes else arr
    lo = np.atleast_1d(lo)
    hi = np.atleast_1d(hi)
    # A constant channel has zero span.  Give it a small but *absolute* scale
    # rather than a relative epsilon: dividing by a near-zero span produces an
    # enormous zero point that overflows the int32 cast.
    span = np.where(hi > lo, hi - lo, 1.0).astype(np.float64)
    scale = (span / float(qmax - qmin)).astype(np.float32)
    safe_scale = np.where(scale > 0.0, scale, 1.0).astype(np.float64)
    zero_point = np.rint(qmin - lo.astype(np.float64) / safe_scale)
    zero_point = np.clip(zero_point, qmin, qmax).astype(np.int32)
    return QuantParams(scale=scale, zero_point=zero_point, axis=ax, fmt=fmt)


def _range_to_params(lo: float, hi: float, qmin: int, qmax: int) -> tuple[float, int]:
    """Convert a float range into (scale, zero_point), guarding a degenerate range."""
    if hi <= lo:
        hi = lo + 1e-8
    scale = (hi - lo) / float(qmax - qmin)
    zp = int(round(qmin - lo / scale))
    return float(scale), int(min(max(zp, qmin), qmax))


def derive_params(
    weights: np.ndarray,
    fmt: NumericFormat,
    scheme: str = "symmetric",
    granularity: str = "per_channel",
    axis: int = 0,
) -> QuantParams:
    """Dispatch to the symmetric or asymmetric parameter derivation."""
    if scheme == "symmetric":
        return symmetric_params(weights, fmt, granularity, axis)
    if scheme == "asymmetric":
        return asymmetric_params(weights, fmt, granularity, axis)
    raise ValueError(f"unknown scheme {scheme!r}")


def apply_params(
    weights: np.ndarray,
    params: QuantParams,
    round_mode: str = "nearest",
    seed: int | None = None,
) -> np.ndarray:
    """Quantise ``weights`` with precomputed parameters and dequantise back.

    Per-channel parameters are broadcast along the stored axis.  Stochastic
    rounding is offered because it is unbiased in expectation: the round-trip
    error stops being a systematic drift toward zero, which matters when many
    layers are stacked.
    """
    arr = np.asarray(weights, dtype=np.float32)
    if params.axis is None:
        scale = float(params.scale[0])
        zp = int(params.zero_point[0])
        if round_mode == "stochastic":
            codes = _stochastic_codes(arr, scale, zp, params.fmt, seed)
            return codes
        return quantize_to_grid(arr, params.fmt, scale, zp)

    shape = [1] * arr.ndim
    shape[params.axis] = params.scale.size
    scale = params.scale.reshape(shape)
    zp = params.zero_point.reshape(shape)
    if round_mode == "stochastic":
        return _stochastic_apply(arr, scale, zp, params.fmt, seed)

    low, high = FORMAT_SPECS[params.fmt].integer_range
    codes = np.rint(arr / scale).astype(np.int32) + zp
    codes = np.clip(codes, low, high)
    return ((codes - zp) * scale).astype(np.float32)


def _stochastic_codes(
    arr: np.ndarray, scale: float, zp: int, fmt: NumericFormat, seed: int | None
) -> np.ndarray:
    """Stochastic rounding for a per-tensor parameter set."""
    low, high = FORMAT_SPECS[fmt].integer_range
    rng = np.random.default_rng(seed)
    exact = arr / np.float32(scale)
    floor = np.floor(exact)
    frac = exact - floor
    bump = (rng.random(arr.shape) < frac).astype(np.float32)
    codes = np.clip((floor + bump).astype(np.int32) + zp, low, high)
    return ((codes - zp).astype(np.float32) * np.float32(scale)).astype(np.float32)
