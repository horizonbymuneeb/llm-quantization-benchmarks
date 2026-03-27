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
