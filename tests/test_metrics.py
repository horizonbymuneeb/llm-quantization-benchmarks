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


def _onehot_predictions(classes: list[int], n_classes: int = 3) -> np.ndarray:
    """Build a probability matrix that predicts ``classes`` with certainty."""
    out = np.full((len(classes), n_classes), 0.01, dtype=np.float64)
    for row, cls in enumerate(classes):
        out[row, cls] = 0.98
    return out


def test_accuracy_from_probs_counts_the_argmax() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    labels = np.array([0, 1, 1])
    assert accuracy_from_probs(probs, labels) == pytest.approx(2.0 / 3.0)


def test_accuracy_from_probs_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="cannot score an empty batch"):
        accuracy_from_probs(np.zeros((0, 3)), np.zeros(0, dtype=np.int64))
