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
