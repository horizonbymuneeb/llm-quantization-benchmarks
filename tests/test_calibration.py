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


def test_percentile_clips_the_tail_and_reports_the_retained_fraction(
    gaussian_with_outliers: np.ndarray,
) -> None:
    result = percentile_range(gaussian_with_outliers, 99.0)
    assert result.upper < float(gaussian_with_outliers.max())
    assert result.metric == pytest.approx(0.99, abs=0.01)
    assert result.method == "percentile"


def test_percentile_retention_rises_with_the_percentile(gaussian_with_outliers: np.ndarray) -> None:
    tight = percentile_range(gaussian_with_outliers, 90.0)
    loose = percentile_range(gaussian_with_outliers, 99.9)
    assert tight.upper < loose.upper
    assert tight.metric < loose.metric


def test_percentile_rejects_an_out_of_range_percentile() -> None:
    with pytest.raises(ValueError, match="percentile must lie"):
        percentile_range(np.zeros(10, dtype=np.float32), 100.0)
    with pytest.raises(ValueError, match="percentile must lie"):
        percentile_range(np.zeros(10, dtype=np.float32), 49.0)


def test_build_histogram_counts_every_observation() -> None:
    obs = np.random.default_rng(1).normal(size=1000)
    hist, edges = build_histogram(obs, 64)
    assert hist.size == 64
    assert edges.size == 65
    assert float(hist.sum()) == pytest.approx(1000.0)
    assert edges[0] == 0.0


def test_build_histogram_rejects_too_few_bins() -> None:
    with pytest.raises(ValueError, match="n_bins must be at least"):
        build_histogram(np.zeros(10), 4)


def test_build_histogram_handles_all_zero_input() -> None:
    hist, edges = build_histogram(np.zeros(10), 32)
    assert float(hist.sum()) == pytest.approx(10.0)


def test_kl_divergence_of_identical_distributions_is_zero() -> None:
    p = np.array([0.25, 0.25, 0.5])
    assert _kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_divergence_matches_the_analytic_value() -> None:
    p = np.array([1.0, 0.0])
    q = np.array([0.5, 0.5])
    # KL((1,0) || (.5,.5)) = 1 * log2(1 / 0.5) = 1 bit.
    assert _kl_divergence(p, q) == pytest.approx(1.0)


def test_kl_divergence_is_asymmetric() -> None:
    p = np.array([0.9, 0.1])
    q = np.array([0.5, 0.5])
    assert _kl_divergence(p, q) != pytest.approx(_kl_divergence(q, p))


def test_kl_divergence_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        _kl_divergence(np.array([0.5, 0.5]), np.array([1.0]))


def test_entropy_clips_the_outlier_tail(gaussian_with_outliers: np.ndarray) -> None:
    result = entropy_calibration(gaussian_with_outliers)
    assert result.method == "entropy"
    assert result.upper < float(gaussian_with_outliers.max())
    # It must not clip so hard that the bulk of a unit Gaussian is lost.
    assert result.upper > 2.0
    assert result.metric >= 0.0


def test_entropy_kl_improves_with_more_levels(gaussian_with_outliers: np.ndarray) -> None:
    coarse = entropy_calibration(gaussian_with_outliers, n_levels=8)
    fine = entropy_calibration(gaussian_with_outliers, n_levels=256)
    assert fine.metric < coarse.metric


def test_entropy_threshold_is_roughly_stable_across_level_counts(
    gaussian_with_outliers: np.ndarray,
) -> None:
    """The optimal clip point depends mostly on the distribution, not the grid.

    This was measured, not assumed: on a Gaussian bulk with an outlier tail the
    KL minimum sits at about four standard deviations whether the grid has 4
    codes or 256.  What changes with the level count is how *much* information
    is lost at that optimum, which is asserted separately below.  The test pins
    the stability so a regression that makes the threshold drift with the code
    count is caught.
    """
    coarse = entropy_calibration(gaussian_with_outliers, n_levels=4)
    fine = entropy_calibration(gaussian_with_outliers, n_levels=256)
    ratio = coarse.upper / fine.upper
    assert 0.5 < ratio < 2.0, ratio


def test_entropy_rejects_degenerate_configuration() -> None:
    with pytest.raises(ValueError, match="n_levels must be at least 2"):
        entropy_calibration(np.zeros(100), n_levels=1)
    with pytest.raises(ValueError, match="steps must be at least 2"):
        entropy_calibration(np.random.default_rng(0).normal(size=100), steps=1)


def test_entropy_on_a_tiny_input_falls_back_to_the_full_range() -> None:
    result = entropy_calibration(np.array([-1.0, 1.0], dtype=np.float32), n_bins=16)
    assert result.lower == pytest.approx(-1.0)
    assert result.upper == pytest.approx(1.0)
    assert result.metric == 0.0


def test_entropy_on_a_constant_input_pads_instead_of_collapsing() -> None:
    result = entropy_calibration(np.full(100, 3.0, dtype=np.float32), n_bins=32)
    assert result.upper > result.lower
    assert result.scale > 0.0


def test_entropy_never_clips_below_the_bulk_of_a_gaussian() -> None:
    obs = np.random.default_rng(11).normal(0.0, 1.0, 5000)
    result = entropy_calibration(obs)
    # A correct entropy search keeps at least 3 standard deviations.
    assert result.upper > 3.0


def test_mse_error_shrinks_as_levels_increase(gaussian_with_outliers: np.ndarray) -> None:
    coarse = mse_calibration(gaussian_with_outliers, n_levels=16)
    fine = mse_calibration(gaussian_with_outliers, n_levels=255)
    assert fine.metric < coarse.metric
    assert fine.upper > coarse.upper


def test_mse_rejects_degenerate_configuration() -> None:
    with pytest.raises(ValueError, match="n_levels must be at least 2"):
        mse_calibration(np.zeros(100), n_levels=1)


def test_calibrate_dispatches_and_rejects_unknown_methods() -> None:
    obs = np.random.default_rng(2).normal(size=500)
    for method in ("minmax", "percentile", "entropy", "mse"):
        result = calibrate(obs, method=method)
        assert isinstance(result, CalibrationResult)
        assert result.upper > result.lower
    with pytest.raises(ValueError, match="unknown calibration method"):
        calibrate(obs, method="histogram-equalisation")


def test_calibration_result_clip_and_scale_agree() -> None:
    result = CalibrationResult(lower=-2.0, upper=4.0, method="minmax", metric=0.0, n_observations=10)
    assert result.scale == pytest.approx(3.0)
    assert result.midpoint == pytest.approx(1.0)
    clipped = result.clip(np.array([-10.0, 0.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(clipped, [-2.0, 0.0, 4.0])


def test_calibration_result_as_dict_is_json_safe() -> None:
    result = minmax_range(np.array([-1.0, 1.0], dtype=np.float32))
    payload = result.as_dict()
    assert payload["method"] == "minmax"
    assert payload["n_observations"] == 2
    assert isinstance(payload["lower"], float)


def test_levels_for_format_matches_the_grid() -> None:
    assert levels_for_format(NumericFormat.INT8) == 255
    assert levels_for_format(NumericFormat.INT8, asymmetric=True) == 256
    assert levels_for_format(NumericFormat.INT4) == 15
    assert levels_for_format(NumericFormat.INT4, asymmetric=True) == 16
    assert levels_for_format(NumericFormat.BF16) == 0


def test_calibrate_layer_uses_the_format_level_count(gaussian_with_outliers: np.ndarray) -> None:
    int8 = calibrate_layer([gaussian_with_outliers], NumericFormat.INT8, method="entropy")
    int4 = calibrate_layer([gaussian_with_outliers], NumericFormat.INT4, method="entropy")
    # INT4 has 15 codes against INT8's 255, so at a comparable clip point it
    # loses far more information.  That is the measurable effect of the grid
    # size; the chosen threshold itself barely moves.
    assert int4.metric > int8.metric
    assert int4.upper == pytest.approx(int8.upper, rel=0.5)


def test_calibrate_layer_supports_every_method(gaussian_with_outliers: np.ndarray) -> None:
    for method in ("minmax", "percentile", "entropy", "mse"):
        result = calibrate_layer([gaussian_with_outliers], NumericFormat.INT8, method=method)
        assert result.upper > result.lower
    with pytest.raises(ValueError, match="unknown calibration method"):
        calibrate_layer([gaussian_with_outliers], NumericFormat.INT8, method="nope")


def test_calibration_is_deterministic() -> None:
    obs = np.random.default_rng(3).normal(size=2000)
    first = entropy_calibration(obs).upper
    second = entropy_calibration(obs).upper
    assert first == second
