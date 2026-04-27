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


def test_accuracy_from_probs_rejects_a_row_mismatch() -> None:
    with pytest.raises(ValueError, match="row count mismatch"):
        accuracy_from_probs(np.zeros((4, 3)), np.zeros(5, dtype=np.int64))


def test_per_class_accuracy_marks_absent_classes_as_nan() -> None:
    probs = _onehot_predictions([0, 0, 1])
    labels = np.array([0, 1, 1])
    out = per_class_accuracy(probs, labels, 3)
    assert out["class_0"] == pytest.approx(1.0)
    assert out["class_1"] == pytest.approx(0.5)
    assert np.isnan(out["class_2"])


def test_confusion_matrix_places_counts_on_true_then_predicted() -> None:
    predictions = np.array([0, 1, 1, 2])
    labels = np.array([0, 0, 1, 2])
    matrix = confusion_matrix(predictions, labels, 3)
    assert matrix[0, 0] == 1  # true 0, predicted 0
    assert matrix[0, 1] == 1  # true 0, predicted 1
    assert matrix[1, 1] == 1
    assert matrix[2, 2] == 1
    assert int(matrix.sum()) == 4


def test_confusion_matrix_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        confusion_matrix(np.zeros(3, dtype=np.int64), np.zeros(4, dtype=np.int64), 2)


def test_accuracy_delta_is_signed_candidate_minus_reference() -> None:
    ref = _onehot_predictions([0, 1, 2, 0])
    cand = _onehot_predictions([0, 1, 2, 1])
    labels = np.array([0, 1, 2, 0])
    delta = accuracy_delta(ref, cand, labels)
    assert delta["reference_accuracy"] == pytest.approx(1.0)
    assert delta["candidate_accuracy"] == pytest.approx(0.75)
    assert delta["accuracy_delta"] == pytest.approx(-0.25)


def test_accuracy_delta_relative_is_nan_when_reference_is_zero() -> None:
    ref = _onehot_predictions([1, 1])
    cand = _onehot_predictions([0, 1])
    labels = np.array([0, 0])
    delta = accuracy_delta(ref, cand, labels)
    assert delta["reference_accuracy"] == 0.0
    assert np.isnan(delta["relative_delta"])


def test_decision_churn_counts_every_flip() -> None:
    ref = _onehot_predictions([0, 1, 2, 0, 1])
    cand = _onehot_predictions([0, 1, 1, 2, 1])
    labels = np.array([0, 1, 2, 0, 1])
    record = PredictionRecord(ref, cand, labels)
    churn = decision_churn(record)
    assert churn["flip_count"] == 2
    assert churn["flip_rate"] == pytest.approx(0.4)
    assert churn["agreement_rate"] == pytest.approx(0.6)


def test_decision_churn_separates_gained_lost_and_lateral_flips() -> None:
    # labels      = [1, 1, 2, 0]
    # row 0: ref 0 -> cand 1, wrong -> right   (gained)
    # row 1: ref 1 -> cand 0, right -> wrong   (lost)
    # row 2: ref 2 -> cand 2, right, no flip
    # row 3: ref 0 -> cand 2, right -> wrong   (lost)
    # A lateral flip (wrong -> differently wrong) is covered separately below.
    ref = _onehot_predictions([0, 1, 2, 0])
    cand = _onehot_predictions([1, 0, 2, 2])
    labels = np.array([1, 1, 2, 0])
    churn = decision_churn(PredictionRecord(ref, cand, labels))
    assert churn["gained"] == 1
    assert churn["lost"] == 2
    assert churn["lateral"] == 0
    assert churn["net_flips"] == -1
    assert churn["flip_count"] == 3


def test_decision_churn_counts_a_wrong_to_differently_wrong_flip_as_lateral() -> None:
    # label is 2, ref says 0, cand says 1: still wrong, but a different answer.
    ref = _onehot_predictions([0])
    cand = _onehot_predictions([1])
    churn = decision_churn(PredictionRecord(ref, cand, np.array([2])))
    assert churn["lateral"] == 1
    assert churn["gained"] == 0
    assert churn["lost"] == 0
    assert churn["net_flips"] == 0
