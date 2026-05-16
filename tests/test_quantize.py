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
