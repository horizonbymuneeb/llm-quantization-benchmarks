"""Activation range selection ("calibration") for integer quantisation.

An INT8 kernel needs to know the float range it must cover before it can pick a
scale.  Getting that range wrong is the single largest source of accuracy loss
in post-training quantisation, and the four methods here fail in different ways:

* ``minmax`` uses the observed extremes.  One outlier then stretches the grid
  over a range that is mostly empty, so typical values land on few codes.
* ``percentile`` clips a fixed fraction of the tail.  Cheap, but the clip
  fraction is arbitrary and does not know what the layer actually does with
  those values.
* ``entropy`` (the KL method from the TensorRT/NVIDIA calibration literature)
  searches a clip point that minimises the information lost when the original
  distribution is squeezed into the reduced code count.
* ``mse`` minimises the reconstruction error of the quantise/dequantise round
  trip directly, which is what the model actually experiences.

The functions return a *range*, not a scale, so the caller can pair the range
with either symmetric or asymmetric scale derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtypes import NumericFormat

MIN_BINS = 16
ENTROPY_SEARCH_STEPS = 96


@dataclass(frozen=True)
class CalibrationResult:
    """The chosen clip range plus the diagnostics that justify it."""

    lower: float
    upper: float
    method: str
    metric: float
    n_observations: int

    @property
    def scale(self) -> float:
        """Half-range, i.e. the symmetric scale this range implies."""
        return (self.upper - self.lower) / 2.0

    @property
    def midpoint(self) -> float:
        """Centre of the calibrated range.

        This is the float value that maps onto the middle code of an unsigned
        grid.  The integer zero point is derived from it and the scale by the
        quantiser, not here, because that derivation depends on the target
        format's code count.
        """
        return (self.lower + self.upper) / 2.0

    def clip(self, values: np.ndarray) -> np.ndarray:
        """Clip an array into the calibrated range."""
        return np.clip(np.asarray(values, dtype=np.float32), self.lower, self.upper)

    def as_dict(self) -> dict[str, float | str | int]:
        """JSON-friendly view, embedded in the report."""
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
            "method": self.method,
            "metric": float(self.metric),
            "n_observations": int(self.n_observations),
        }


def flatten_observations(values: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Flatten one array or a batch of arrays into a single 1-D float32 vector."""
    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("cannot calibrate on an empty list of arrays")
        parts = [np.asarray(v, dtype=np.float32).ravel() for v in values]
        return np.concatenate(parts)
    arr = np.asarray(values, dtype=np.float32).ravel()
    if arr.size == 0:
        raise ValueError("cannot calibrate on an empty array")
    return arr


def minmax_range(observations: np.ndarray) -> CalibrationResult:
    """Use the full observed extent; no clipping at all."""
    obs = flatten_observations(observations)
    lo = float(obs.min())
    hi = float(obs.max())
    if lo == hi:
        # A constant activation still needs a non-degenerate scale.
        pad = abs(lo) * 0.05 if lo != 0.0 else 0.5
        lo, hi = lo - pad, hi + pad
    return CalibrationResult(lo, hi, "minmax", float(np.abs(obs).max()), int(obs.size))


def percentile_range(observations: np.ndarray, percentile: float = 99.9) -> CalibrationResult:
    """Clip the symmetric tail so ``percentile`` percent of mass is retained."""
    if not 50.0 <= percentile < 100.0:
        raise ValueError(f"percentile must lie in [50, 100), got {percentile!r}")
    obs = flatten_observations(observations)
    tail = (100.0 - percentile) / 2.0
    lo = float(np.percentile(obs, tail))
    hi = float(np.percentile(obs, 100.0 - tail))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0.0 else 0.5
        lo, hi = lo - pad, hi + pad
    # Report the fraction of samples actually retained, so a caller can see
    # how aggressive the clip was instead of trusting the parameter name.
    inside = float(np.mean((obs >= lo) & (obs <= hi)))
    return CalibrationResult(lo, hi, "percentile", inside, int(obs.size))


def _padded_extent(obs: np.ndarray) -> tuple[float, float]:
    """Observed extent, widened when every observation is identical.

    A calibration range of zero width would make every downstream scale zero,
    so a degenerate input is padded the same way :func:`minmax_range` pads it.
    """
    lo = float(obs.min())
    hi = float(obs.max())
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0.0 else 0.5
        return lo - pad, hi + pad
    return lo, hi
