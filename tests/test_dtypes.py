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
