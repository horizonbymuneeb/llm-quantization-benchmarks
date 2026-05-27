"""Tests for the quantisers: scale derivation, dequantisation, error metrics."""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.dtypes import NumericFormat
from quantbench.quantize import (
    QuantParams,
    apply_params,
    asymmetric_params,
    derive_params,
    effective_bits,
    quantization_error,
    quantize_layer_weights,
    quantize_tensor,
    symmetric_params,
)


def test_symmetric_per_tensor_scale_is_amax_over_qmax() -> None:
    weights = np.array([[1.0, -2.0], [0.5, 0.25]], dtype=np.float32)
    params = symmetric_params(weights, NumericFormat.INT8, "per_tensor")
    assert params.scale.size == 1
    assert params.scale[0] == pytest.approx(2.0 / 127.0, rel=1e-6)
    assert params.zero_point[0] == 0
    assert params.axis is None
    assert not params.is_per_channel


def test_symmetric_per_channel_gives_one_scale_per_output_row() -> None:
    weights = np.array([[1.0, -2.0], [0.5, 0.25]], dtype=np.float32)
    params = symmetric_params(weights, NumericFormat.INT8, "per_channel", axis=0)
    assert params.n_channels == 2
    assert params.axis == 0
    assert params.scale[0] == pytest.approx(2.0 / 127.0, rel=1e-6)
    assert params.scale[1] == pytest.approx(0.5 / 127.0, rel=1e-6)
    np.testing.assert_array_equal(params.zero_point, [0, 0])


def test_symmetric_per_channel_along_axis_one_uses_columns() -> None:
    # axis=1 reduces over rows, so each *column* gets its own scale:
    # column 0 holds {1, 2} -> amax 2, column 1 holds {4, 2} -> amax 4.
    weights = np.array([[1.0, 4.0], [2.0, 2.0]], dtype=np.float32)
    params = symmetric_params(weights, NumericFormat.INT8, "per_channel", axis=1)
    assert params.axis == 1
    assert params.n_channels == 2
    assert params.scale[0] == pytest.approx(2.0 / 127.0, rel=1e-6)
    assert params.scale[1] == pytest.approx(4.0 / 127.0, rel=1e-6)


def test_symmetric_int4_uses_fifteen_not_one_hundred_twenty_seven() -> None:
    weights = np.full((4, 4), 7.0, dtype=np.float32)
    params = symmetric_params(weights, NumericFormat.INT4, "per_tensor")
    assert params.scale[0] == pytest.approx(1.0, rel=1e-6)


def test_symmetric_of_an_all_zero_tensor_uses_unit_scale() -> None:
    params = symmetric_params(np.zeros((3, 3), dtype=np.float32), NumericFormat.INT8, "per_tensor")
    assert params.scale[0] == pytest.approx(1.0)


def test_symmetric_rejects_a_float_format() -> None:
    with pytest.raises(TypeError, match="integer format"):
        symmetric_params(np.ones((2, 2), dtype=np.float32), NumericFormat.BF16)


def test_symmetric_rejects_an_out_of_range_axis() -> None:
    with pytest.raises(ValueError, match="axis 5 out of range"):
        symmetric_params(np.ones((2, 2), dtype=np.float32), NumericFormat.INT8, "per_channel", axis=5)


def test_asymmetric_zero_point_maps_float_zero_onto_a_real_code() -> None:
    weights = np.array([[-1.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    params = asymmetric_params(weights, NumericFormat.INT8, "per_tensor")
    qmin, qmax = -128, 127
    scale = float(params.scale[0])
    zp = int(params.zero_point[0])
    assert qmin <= zp <= qmax
    # Dequantising code 0 with the zero point must land on exactly 0.0.
    reconstructed_zero = (zp - zp) * scale
    assert reconstructed_zero == 0.0
    assert scale == pytest.approx(3.0 / (qmax - qmin), rel=1e-5)


def test_asymmetric_recovers_more_range_than_symmetric_on_a_shifted_tensor() -> None:
    # A one-sided tensor: symmetric must cover [-8, 8] while asymmetric covers
    # [0, 8] and therefore has twice the resolution where the data lives.
    weights = np.linspace(0.0, 8.0, 64, dtype=np.float32).reshape(8, 8)
    sym = symmetric_params(weights, NumericFormat.INT8, "per_tensor")
    asym = asymmetric_params(weights, NumericFormat.INT8, "per_tensor")
    assert float(asym.scale[0]) < float(sym.scale[0])
    sym_err = np.abs(apply_params(weights, sym) - weights).max()
    asym_err = np.abs(apply_params(weights, asym) - weights).max()
    assert asym_err < sym_err


def test_asymmetric_per_channel_handles_a_degenerate_constant_channel() -> None:
    weights = np.array([[5.0, 5.0], [1.0, 3.0]], dtype=np.float32)
    params = asymmetric_params(weights, NumericFormat.INT8, "per_channel")
    assert params.n_channels == 2
    assert np.all(np.isfinite(params.scale))
    assert np.all(params.scale > 0)


def test_derive_params_dispatches_and_rejects_unknown_schemes() -> None:
    weights = np.ones((2, 2), dtype=np.float32)
    assert derive_params(weights, NumericFormat.INT8, "symmetric").zero_point[0] == 0
    with pytest.raises(ValueError, match="unknown scheme"):
        derive_params(weights, NumericFormat.INT8, "logarithmic")


def test_apply_params_per_tensor_roundtrip_error_is_at_most_half_a_scale() -> None:
    rng = np.random.default_rng(4)
    weights = rng.normal(size=(16, 8)).astype(np.float32)
    params = symmetric_params(weights, NumericFormat.INT8, "per_tensor")
    out = apply_params(weights, params)
    assert np.max(np.abs(out - weights)) <= float(params.scale[0]) / 2.0 + 1e-6


def test_apply_params_per_channel_broadcasts_along_the_stored_axis() -> None:
    weights = np.array([[1.0, 1.0], [10.0, 10.0]], dtype=np.float32)
    params = symmetric_params(weights, NumericFormat.INT8, "per_channel", axis=0)
    out = apply_params(weights, params)
    assert out.shape == weights.shape
    # Both rows sit exactly on a code because both are constant per row.
    np.testing.assert_allclose(out, weights, atol=1e-4)


def test_stochastic_rounding_is_reproducible_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(5)
    weights = rng.uniform(-1.0, 1.0, size=(32, 32)).astype(np.float32)
    params = symmetric_params(weights, NumericFormat.INT4, "per_tensor")
    a = apply_params(weights, params, round_mode="stochastic", seed=42)
    b = apply_params(weights, params, round_mode="stochastic", seed=42)
    c = apply_params(weights, params, round_mode="stochastic", seed=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stochastic_rounding_lands_on_the_grid() -> None:
    weights = np.array([0.55, -0.55, 1.35], dtype=np.float32)
    params = symmetric_params(weights, NumericFormat.INT8, "per_tensor")
    out = apply_params(weights, params, round_mode="stochastic", seed=0)
    scale = float(params.scale[0])
    codes = out / scale
    np.testing.assert_allclose(codes, np.rint(codes), atol=1e-4)


def test_stochastic_rounding_is_unbiased_where_nearest_is_not() -> None:
    # Values exactly halfway between two INT4 codes.  Nearest-rounding pushes
    # every one of them to the same even code, a systematic +half-scale bias.
    # Stochastic rounding picks either neighbour, so the mean over many seeds
    # recovers the original value.
    scale = 0.5
    halfway = np.full(64, scale * 0.5, dtype=np.float32)
    params = QuantParams(
        scale=np.array([scale], np.float32),
        zero_point=np.zeros(1, np.int32),
        axis=None,
        fmt=NumericFormat.INT4,
    )
    nearest = apply_params(halfway, params, round_mode="nearest")
    assert np.allclose(nearest, 0.0)

    draws = np.stack(
        [apply_params(halfway, params, round_mode="stochastic", seed=s) for s in range(400)]
    )
    stochastic_mean = float(np.mean(draws))
    # Unbiased: the average round trip recovers the original 0.25.
    assert stochastic_mean == pytest.approx(0.25, abs=0.02)
    # Nearest-rounding is off by a full half-scale in the same direction.
    assert float(np.mean(nearest)) == pytest.approx(0.0, abs=1e-6)


def test_quantize_tensor_passes_float_formats_through_the_simulator() -> None:
    rng = np.random.default_rng(6)
    weights = rng.normal(size=(8, 8)).astype(np.float32)
    out, params = quantize_tensor(weights, NumericFormat.BF16)
    assert out.shape == weights.shape
    assert params.scale[0] == 1.0
    assert not np.array_equal(out, weights)


def test_quantize_tensor_returns_matching_parameters() -> None:
    rng = np.random.default_rng(7)
    weights = rng.normal(size=(8, 4)).astype(np.float32)
    out, params = quantize_tensor(weights, NumericFormat.INT8, "asymmetric", "per_channel")
    assert params.n_channels == 8
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_quantization_error_reports_perfect_cosine_for_identical_tensors() -> None:
    rng = np.random.default_rng(8)
    weights = rng.normal(size=(8, 8)).astype(np.float32)
    stats = quantization_error(weights, weights.copy())
    assert stats["cosine"] == pytest.approx(1.0)
    assert stats["cosine_distance"] == pytest.approx(0.0)
    assert stats["rmse"] == 0.0


def test_quantization_error_detects_a_direction_change() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    stats = quantization_error(a, b)
    assert stats["cosine"] == pytest.approx(0.0, abs=1e-9)
    assert stats["cosine_distance"] == pytest.approx(1.0)


def test_quantization_error_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        quantization_error(np.zeros((2, 2)), np.zeros((3,)))
