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
