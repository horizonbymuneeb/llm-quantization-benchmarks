"""Tests for per-layer sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.config import DataConfig, ModelConfig
from quantbench.data import build_dataset
from quantbench.dtypes import NumericFormat
from quantbench.model import build_and_train
from quantbench.sensitivity import (
    LayerSensitivity,
    apply_mixed_precision,
    cumulative_sensitivity,
    format_sensitivity,
    layer_output_error,
    parameter_share,
    quantise_one_layer,
    rank_layers,
    recommend_mixed_precision,
    sensitivity_summary,
    single_layer_sensitivity,
)
