"""Tests for the behavioural metrics: churn, rank correlation, calibration error."""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.metrics import (
    PredictionRecord,
    accuracy_delta,
    accuracy_from_probs,
    compare_predictions,
    confidence_delta,
    confidence_rank_agreement,
    confusion_matrix,
    decision_churn,
    expected_calibration_error,
    kendall_tau,
    margin,
    margin_shift,
    per_class_accuracy,
    spearman_correlation,
)
