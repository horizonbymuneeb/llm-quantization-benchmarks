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


def test_churn_exposes_flips_hidden_by_a_zero_accuracy_delta() -> None:
    """The headline case: same accuracy, different behaviour.

    This is the whole reason the metric exists.  Two flips cancel in accuracy
    and the churn rate still reports them.
    """
    # labels [0, 0]: ref gets one right (row 0) and one wrong (row 1); cand
    # swaps the pair, so exactly one answer is gained and one lost.  Accuracy is
    # unchanged at 0.5 while every input changed its prediction.
    ref = _onehot_predictions([0, 1])
    cand = _onehot_predictions([1, 0])
    labels = np.array([0, 0])
    record = PredictionRecord(ref, cand, labels)
    churn = decision_churn(record)
    assert churn["accuracy_delta"] == pytest.approx(0.0)
    assert churn["flip_count"] == 2
    assert churn["flip_rate"] == pytest.approx(1.0)
    assert churn["gained"] == 1
    assert churn["lost"] == 1


def test_identical_models_have_no_churn() -> None:
    probs = _onehot_predictions([0, 1, 2])
    record = PredictionRecord(probs, probs.copy(), np.array([0, 1, 2]))
    churn = decision_churn(record)
    assert churn["flip_rate"] == 0.0
    assert churn["agreement_rate"] == 1.0


def test_confidence_delta_reports_the_flipped_subset_separately() -> None:
    ref = np.array([[0.9, 0.1], [0.55, 0.45], [0.7, 0.3]])
    cand = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    labels = np.array([0, 1, 0])
    stats = confidence_delta(PredictionRecord(ref, cand, labels))
    # Reference top-1 confidences are 0.90, 0.55, 0.70; candidate ones are
    # 0.90, 0.60, 0.80, so the signed deltas are 0.00, +0.05, +0.10.
    assert stats["n_flipped"] == 2
    assert stats["mean_delta"] == pytest.approx((0.0 + 0.05 + 0.10) / 3)
    # Only rows 1 and 2 flipped; their absolute deltas are 0.05 and 0.10.
    assert stats["flipped_mean_abs_delta"] == pytest.approx(0.075)


def test_margin_is_the_gap_between_the_top_two() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1]])
    np.testing.assert_allclose(margin(probs), [0.5, 0.1], atol=1e-9)


def test_margin_of_a_single_class_problem_is_zero() -> None:
    np.testing.assert_array_equal(margin(np.array([[1.0], [1.0]])), [0.0, 0.0])


def test_margin_shift_reports_shrinkage() -> None:
    ref = np.array([[0.9, 0.1], [0.8, 0.2]])
    cand = np.array([[0.55, 0.45], [0.51, 0.49]])
    record = PredictionRecord(ref, cand, np.array([0, 0]))
    stats = margin_shift(record)
    # Reference margins are 0.8 and 0.6; candidate margins are 0.10 and 0.02.
    assert stats["reference_mean"] == pytest.approx(0.7)
    assert stats["candidate_mean"] == pytest.approx(0.06)
    # Per-row margin changes are -0.70 and -0.58; the worst is -0.70.
    assert stats["max_shrinkage"] == pytest.approx(-0.70)
    assert stats["n_below_zero"] == 0.0


def test_spearman_of_a_perfectly_monotone_pair_is_one() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_correlation(a, a * 3.0 + 1.0) == pytest.approx(1.0)


def test_spearman_of_a_reversed_pair_is_minus_one() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_correlation(a, -a) == pytest.approx(-1.0)


def test_spearman_handles_ties_with_average_ranks() -> None:
    a = np.array([1.0, 1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 2.0, 3.0])
    value = spearman_correlation(a, b)
    assert -1.0 <= value <= 1.0
    assert value > 0.0


def test_spearman_returns_nan_for_a_constant_series() -> None:
    assert np.isnan(spearman_correlation(np.ones(5), np.arange(5.0)))


def test_spearman_returns_nan_below_two_points() -> None:
    assert np.isnan(spearman_correlation(np.array([1.0]), np.array([2.0])))


def test_spearman_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        spearman_correlation(np.arange(3.0), np.arange(4.0))


def test_kendall_tau_of_a_perfectly_concordant_pair_is_one() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert kendall_tau(a, a) == pytest.approx(1.0)


def test_kendall_tau_of_a_reversed_pair_is_minus_one() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert kendall_tau(a, -a) == pytest.approx(-1.0)


def test_kendall_tau_matches_a_hand_counted_example() -> None:
    # Pairs: (1,2)->concordant, (1,3)->discordant, (2,3)->concordant.
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 3.0, 2.0])
    # concordant 2, discordant 1 -> (2-1)/3
    assert kendall_tau(a, b) == pytest.approx(1.0 / 3.0)


def test_rank_agreement_reports_global_and_per_class_numbers() -> None:
    rng = np.random.default_rng(0)
    n = 60
    labels = rng.integers(0, 3, size=n)
    ref = rng.dirichlet(np.ones(3), size=n)
    # A slightly perturbed candidate: the ordering mostly survives.
    cand = ref + rng.normal(0.0, 0.01, size=ref.shape)
    cand = np.clip(cand, 1e-6, None)
    cand /= cand.sum(axis=1, keepdims=True)
    stats = confidence_rank_agreement(PredictionRecord(ref, cand, labels))
    assert stats["spearman_global"] > 0.5
    assert stats["kendall_global"] > 0.3
    assert 0.0 <= stats["spearman_argmax_agreement"] <= 1.0
    assert -1.0 <= stats["spearman_class_mean"] <= 1.0


def test_expected_calibration_error_is_zero_for_a_perfectly_calibrated_model() -> None:
    # One row per bin, each predicting its bin's midpoint confidence and being
    # right exactly that often is hard to construct, so use the trivial case:
    # all rows correct with confidence 1.0.
    probs = np.array([[1.0, 0.0], [1.0, 0.0]])
    labels = np.array([0, 0])
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0)


def test_expected_calibration_error_is_large_for_a_confidently_wrong_model() -> None:
    probs = np.array([[0.95, 0.05], [0.95, 0.05]])
    labels = np.array([1, 1])
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.95, abs=1e-6)
