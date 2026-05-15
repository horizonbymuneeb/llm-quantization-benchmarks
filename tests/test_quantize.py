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
