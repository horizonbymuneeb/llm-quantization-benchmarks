"""Tests for the sweep driver: quantisation of a model and row assembly."""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.config import CalibrationConfig, DataConfig, ModelConfig, SweepConfig
from quantbench.data import build_dataset
from quantbench.dtypes import NumericFormat
from quantbench.model import build_and_train
from quantbench.sweep import (
    QuantizedModel,
    SweepRow,
    aggregate_by_format,
    evaluate_model,
    quantize_model,
    row_delta_table,
    weight_error_report,
)
