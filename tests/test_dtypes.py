"""Tests for storage-format simulation and the byte accounting."""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.dtypes import (
    FORMAT_SPECS,
    FormatSpec,
    NumericFormat,
    cast_bf16,
    cast_fp16,
    cast_fp32,
    compression_ratio,
    format_spec,
    quant_error_stats,
    quantize_to_grid,
    simulate_format,
    storage_bytes,
)


def test_format_bits_and_kind_are_consistent() -> None:
    assert NumericFormat.FP32.bits == 32
    assert NumericFormat.BF16.bits == 16
    assert NumericFormat.INT4.bits == 4
    assert NumericFormat.INT8.is_integer
    assert not NumericFormat.BF16.is_integer
    assert NumericFormat.FP16.is_float


def test_from_string_accepts_case_and_whitespace_and_rejects_garbage() -> None:
    assert NumericFormat.from_string(" INT4 ") is NumericFormat.INT4
    assert NumericFormat.from_string("Bf16") is NumericFormat.BF16
    with pytest.raises(ValueError, match="unknown numeric format"):
        NumericFormat.from_string("int3")


def test_integer_ranges_match_the_bit_width() -> None:
    int8 = format_spec(NumericFormat.INT8)
    int4 = format_spec(NumericFormat.INT4)
    assert int8.integer_range == (-128, 127)
    assert int4.integer_range == (-8, 7)
    with pytest.raises(TypeError, match="not an integer format"):
        format_spec(NumericFormat.BF16).integer_range


def test_format_spec_total_bits_for_floats_includes_sign() -> None:
    assert FORMAT_SPECS[NumericFormat.FP32].total_bits == 32
    assert FORMAT_SPECS[NumericFormat.BF16].total_bits == 16
    assert FORMAT_SPECS[NumericFormat.FP16].total_bits == 16
    spec = FormatSpec(NumericFormat.INT8, 0, 7, True)
    assert spec.total_bits == 8
    assert not spec.is_float


def test_bf16_is_exact_on_representable_values() -> None:
    values = np.array([1.0, 0.5, 2.0, -0.25, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(cast_bf16(values), values)


def test_bf16_rounds_to_nearest_even_not_toward_zero() -> None:
    # 1 + 2^-8 is exactly halfway between BF16 codes 1.0 and 1.0078125;
    # round-half-to-even picks 1.0 (the even code).
    halfway_low = np.array([1.0 + 2.0**-8], dtype=np.float32)
    assert cast_bf16(halfway_low)[0] == np.float32(1.0)
    # 1 + 3*2^-8 rounds up to 1.015625, above the halfway point.
    above = np.array([1.0 + 3 * 2.0**-8], dtype=np.float32)
    assert cast_bf16(above)[0] == np.float32(1.015625)


def test_bf16_truncation_would_bias_downward_but_rounding_does_not() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, size=5000).astype(np.float32)
    rounded = cast_bf16(values)
    mean_error = float(np.mean(rounded.astype(np.float64) - values.astype(np.float64)))
    assert abs(mean_error) < 1e-4, mean_error


def test_bf16_error_bound_is_bounded_by_the_mantissa_width() -> None:
    rng = np.random.default_rng(1)
    values = rng.uniform(-4.0, 4.0, size=2000).astype(np.float32)
    error = np.abs(cast_bf16(values) - values)
    # 7 explicit mantissa bits -> relative error <= 2^-8.
    relative = error / np.abs(values)
    assert float(np.max(relative)) <= 2.0**-8 + 1e-7


def test_fp16_saturates_to_infinity_above_its_range() -> None:
    out = cast_fp16(np.array([70000.0], dtype=np.float32))
    assert np.isinf(out[0])
    assert cast_fp16(np.array([65504.0], dtype=np.float32))[0] == np.float32(65504.0)


def test_fp16_has_finer_resolution_than_bf16_near_one() -> None:
    probe = np.array([1.0 + 2.0**-9], dtype=np.float32)
    fp16_err = abs(float(cast_fp16(probe)[0]) - float(probe[0]))
    bf16_err = abs(float(cast_bf16(probe)[0]) - float(probe[0]))
    assert fp16_err < bf16_err


def test_fp32_cast_is_a_copy_not_a_view() -> None:
    src = np.array([1.5, 2.5], dtype=np.float32)
    out = cast_fp32(src)
    out[0] = 9.0
    assert src[0] == np.float32(1.5)


def test_quantize_to_grid_snaps_and_clamps() -> None:
    values = np.array([0.0, 0.1, 0.2, 5.0, -5.0], dtype=np.float32)
    out = quantize_to_grid(values, NumericFormat.INT8, scale=0.2)
    # 0.1 / 0.2 is exactly 0.5, so it lands on the even code 0, not code 1.
    np.testing.assert_allclose(out, [0.0, 0.0, 0.2, 5.0, -5.0], atol=1e-6)


def test_quantize_to_grid_rounds_half_to_even() -> None:
    # Codes 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, 3.5 -> 4 under round-half-to-even.
    values = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    out = quantize_to_grid(values, NumericFormat.INT8, scale=1.0)
    np.testing.assert_array_equal(out, [0.0, 2.0, 2.0, 4.0])


def test_quantize_to_grid_truncate_mode_rounds_toward_zero() -> None:
    values = np.array([0.9, 1.9, -1.9], dtype=np.float32)
    out = quantize_to_grid(values, NumericFormat.INT8, scale=1.0, round_mode="truncate")
    np.testing.assert_array_equal(out, [0.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="unknown round_mode"):
        quantize_to_grid(values, NumericFormat.INT8, 1.0, round_mode="ceil")


def test_quantize_to_grid_clamps_to_int4_range() -> None:
    values = np.array([0.0, 100.0], dtype=np.float32)
    out = quantize_to_grid(values, NumericFormat.INT4, scale=1.0)
    assert out[1] == 7.0
    assert quantize_to_grid(np.array([-100.0], dtype=np.float32), NumericFormat.INT4, 1.0)[0] == -8.0


def test_quantize_to_grid_applies_the_zero_point() -> None:
    values = np.array([0.0, 1.0], dtype=np.float32)
    plain = quantize_to_grid(values, NumericFormat.INT8, 1.0, zero_point=0)
    shifted = quantize_to_grid(values, NumericFormat.INT8, 1.0, zero_point=10)
    np.testing.assert_allclose(plain, values, atol=1e-6)
    np.testing.assert_allclose(shifted, values, atol=1e-6)


def test_quantize_to_grid_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        quantize_to_grid(np.zeros(3, dtype=np.float32), NumericFormat.INT8, 0.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        quantize_to_grid(np.zeros(3, dtype=np.float32), NumericFormat.INT8, -1.0)


def test_storage_bytes_handles_int4_nibble_packing() -> None:
    assert storage_bytes((10,), NumericFormat.INT4) == 5
    assert storage_bytes((7,), NumericFormat.INT4) == 4
    assert storage_bytes((10,), NumericFormat.INT8) == 10
    assert storage_bytes((10,), NumericFormat.BF16) == 20
    assert storage_bytes((10,), NumericFormat.FP32) == 40
    assert storage_bytes((), NumericFormat.FP32) == 4
