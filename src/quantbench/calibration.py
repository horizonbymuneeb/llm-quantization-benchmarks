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


def build_histogram(observations: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Histogram the absolute values, which is what the clip search needs.

    Quantisation error depends on the magnitude of a value, not its sign, so the
    search runs on ``|x|`` and the resulting threshold is mirrored at the end.
    """
    if n_bins < MIN_BINS:
        raise ValueError(f"n_bins must be at least {MIN_BINS}, got {n_bins}")
    obs = flatten_observations(observations)
    absvals = np.abs(obs)
    hi = float(absvals.max())
    if hi <= 0.0:
        return np.array([float(absvals.size)]), np.array([0.0, 0.0])
    hist, edges = np.histogram(absvals, bins=n_bins, range=(0.0, hi))
    return hist.astype(np.float64), edges.astype(np.float64)


def _normalised_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a count vector, in bits, zero-safe."""
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def _level_assignment(n_bins: int, n_levels: int) -> np.ndarray:
    """Map each histogram bin onto one of ``n_levels`` codes, no clipping."""
    if n_bins < 1 or n_levels < 1:
        raise ValueError("n_bins and n_levels must be positive")
    return np.minimum((np.arange(n_bins) * n_levels) // n_bins, n_levels - 1)


def _saturating_assignment(n_bins: int, n_levels: int, cut: int) -> np.ndarray:
    """Map bins onto codes as a real kernel does when clipped at bin ``cut``.

    The retained range ``[0, cut)`` is divided into ``n_levels`` equal steps, so
    a small ``cut`` gives *finer* resolution inside the dense region.  Every bin
    at or beyond ``cut`` saturates onto the last code.  This is the detail that
    makes the entropy search non-degenerate: without the saturation term, a tiny
    threshold would look lossless because the surviving bins would each get
    their own code, and the search would always clip as hard as allowed.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be positive")
    if cut < 1:
        raise ValueError("cut must be positive")
    codes = (np.arange(n_bins) * n_levels) // max(cut, 1)
    return np.minimum(codes, n_levels - 1)


def _expand_codes(hist: np.ndarray, assignment: np.ndarray, n_levels: int) -> np.ndarray:
    """Spread each code's mass uniformly over the bins that map onto it.

    The quantised model cannot tell bins inside one code apart, so its best
    estimate for each of them is the code's mean probability.  Comparing that
    estimate against the reference on the *same* support is what makes the KL
    divergence meaningful.
    """
    hist = np.asarray(hist, dtype=np.float64)
    code_mass = np.zeros(n_levels, dtype=np.float64)
    np.add.at(code_mass, assignment, hist)
    counts = np.bincount(assignment, minlength=n_levels).astype(np.float64)
    per_bin = np.divide(code_mass, counts, out=np.zeros_like(code_mass), where=counts > 0)
    return per_bin[assignment]
