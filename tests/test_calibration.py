"""Tests for calibration: range selection and the entropy search."""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.calibration import (
    CalibrationResult,
    _kl_divergence,
    build_histogram,
    calibrate,
    calibrate_layer,
    entropy_calibration,
    flatten_observations,
    levels_for_format,
    minmax_range,
    mse_calibration,
    percentile_range,
)
from quantbench.dtypes import NumericFormat


@pytest.fixture()
def gaussian_with_outliers() -> np.ndarray:
    """A unit Gaussian bulk with 40 extreme outliers injected on top."""
    rng = np.random.default_rng(5)
    obs = rng.normal(0.0, 1.0, 20000)
    obs[:40] = rng.normal(0.0, 12.0, 40)
    return obs


def test_flatten_observations_concatenates_a_list_of_arrays() -> None:
    out = flatten_observations([np.array([[1.0, 2.0]]), np.array([3.0])])
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])
    assert out.dtype == np.float32


def test_flatten_observations_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty list"):
        flatten_observations([])
    with pytest.raises(ValueError, match="empty array"):
        flatten_observations(np.array([], dtype=np.float32))


def test_minmax_uses_the_observed_extremes_exactly() -> None:
    obs = np.array([-3.0, 0.0, 5.0], dtype=np.float32)
    result = minmax_range(obs)
    assert result.lower == pytest.approx(-3.0)
    assert result.upper == pytest.approx(5.0)
    assert result.method == "minmax"
    assert result.n_observations == 3


def test_minmax_pads_a_constant_observation_into_a_real_range() -> None:
    result = minmax_range(np.full(10, 4.0, dtype=np.float32))
    assert result.upper > result.lower
    assert result.scale > 0.0


def test_minmax_of_constant_zero_uses_a_fixed_pad() -> None:
    result = minmax_range(np.zeros(5, dtype=np.float32))
    assert result.lower == pytest.approx(-0.5)
    assert result.upper == pytest.approx(0.5)
