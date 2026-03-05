"""Behavioural metrics: accuracy, decision churn, and confidence rank agreement.

Aggregate accuracy is a lossy summary of what compression did.  A model can keep
its accuracy to three decimal places while changing its mind on a sixth of the
inputs, and those flips are what a user of the model actually experiences.  This
module therefore reports three families of metric:

* **accuracy** — the classic aggregate, plus the per-class breakdown so a
  regression concentrated in one class is visible.
* **decision churn** — per-input comparison of the compressed model's argmax
  against the reference argmax: flips, net flips, confidence deltas, and the
  margin change for the inputs that flipped.
* **rank agreement** — Spearman and Kendall correlation between the reference
  and compressed confidence orderings.  Two models can agree on every argmax
  while ordering the rest of the distribution completely differently, and the
  rank metrics catch that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictionRecord:
    """Reference and candidate predictions for the same inputs."""

    reference_probs: np.ndarray
    candidate_probs: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        ref = np.asarray(self.reference_probs, dtype=np.float64)
        cand = np.asarray(self.candidate_probs, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.int64)
        if ref.ndim != 2 or cand.ndim != 2:
            raise ValueError("probability matrices must be 2-D")
        if ref.shape != cand.shape:
            raise ValueError(f"shape mismatch: {ref.shape} vs {cand.shape}")
        if labels.shape[0] != ref.shape[0]:
            raise ValueError("label count must match the number of rows")
        if labels.size and (labels.min() < 0 or labels.max() >= ref.shape[1]):
            raise ValueError("labels out of range for the number of classes")

    @property
    def n_samples(self) -> int:
        """Number of compared rows."""
        return int(self.labels.size)

    @property
    def reference_pred(self) -> np.ndarray:
        """Reference argmax predictions."""
        return np.argmax(self.reference_probs, axis=1)

    @property
    def candidate_pred(self) -> np.ndarray:
        """Candidate argmax predictions."""
        return np.argmax(self.candidate_probs, axis=1)

    @property
    def reference_confidence(self) -> np.ndarray:
        """Reference top-1 probability."""
        return np.max(self.reference_probs, axis=1)

    @property
    def candidate_confidence(self) -> np.ndarray:
        """Candidate top-1 probability."""
        return np.max(self.candidate_probs, axis=1)


def accuracy_from_probs(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy of a probability matrix."""
    probs = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("row count mismatch between probabilities and labels")
    if labels.size == 0:
        raise ValueError("cannot score an empty batch")
    return float(np.mean(np.argmax(probs, axis=1) == labels))


def per_class_accuracy(probabilities: np.ndarray, labels: np.ndarray, n_classes: int) -> dict[str, float]:
    """Accuracy within each class, with ``nan`` for classes absent from the split."""
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.argmax(np.asarray(probabilities), axis=1)
    out: dict[str, float] = {}
    for cls in range(n_classes):
        mask = labels == cls
        if not np.any(mask):
            out[f"class_{cls}"] = float("nan")
        else:
            out[f"class_{cls}"] = float(np.mean(preds[mask] == labels[mask]))
    return out


def confusion_matrix(predictions: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def accuracy_delta(
    reference_probs: np.ndarray,
    candidate_probs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Absolute and relative accuracy change from reference to candidate.

    The sign convention is ``candidate - reference``, so a negative delta is a
    regression.  ``relative`` is guarded: when reference accuracy is zero there
    is no meaningful ratio and the field is reported as ``nan``.
    """
    ref_acc = accuracy_from_probs(reference_probs, labels)
    cand_acc = accuracy_from_probs(candidate_probs, labels)
    delta = cand_acc - ref_acc
    relative = float(delta / ref_acc) if ref_acc > 0 else float("nan")
    return {
        "reference_accuracy": ref_acc,
        "candidate_accuracy": cand_acc,
        "accuracy_delta": delta,
        "relative_delta": relative,
    }


def decision_churn(record: PredictionRecord) -> dict[str, float]:
    """Per-input prediction changes between reference and candidate.

    The headline number is ``churn_rate``: the fraction of inputs whose argmax
    moved.  It is *not* the same as the accuracy delta.  A model can flip one
    input from correct to wrong and another from wrong to correct: churn 2/1000
    with an accuracy delta of exactly zero.  That is the case this metric exists
    to expose.

    ``net_flips`` counts flips that changed correctness (signed), while
    ``flip_rate`` counts all flips regardless of whether correctness changed.
    """
    ref_pred = record.reference_pred
    cand_pred = record.candidate_pred
    labels = record.labels
    n = record.n_samples

    flips = ref_pred != cand_pred
    ref_correct = ref_pred == labels
    cand_correct = cand_pred == labels
    gained = int(np.sum(~ref_correct & cand_correct))
    lost = int(np.sum(ref_correct & ~cand_correct))
    # Flips that do not change correctness: both wrong, but differently wrong.
    lateral = int(np.sum(~ref_correct & ~cand_correct & flips))

    return {
        "n_samples": float(n),
        "flip_count": float(np.sum(flips)),
        "flip_rate": float(np.mean(flips)),
        "gained": float(gained),
        "lost": float(lost),
        "lateral": float(lateral),
        "net_flips": float(gained - lost),
        "accuracy_delta": float((cand_correct.sum() - ref_correct.sum()) / n),
        "agreement_rate": float(np.mean(~flips)),
    }


def confidence_delta(record: PredictionRecord) -> dict[str, float]:
    """Change in top-1 confidence, including on the inputs that flipped.

    ``mean_abs`` over all rows is reported next to the mean over flipped rows
    only, because a small average change can hide a large change on exactly the
    inputs that crossed the decision boundary.
    """
    ref_conf = record.reference_confidence
    cand_conf = record.candidate_confidence
    diff = cand_conf - ref_conf
    flips = record.reference_pred != record.candidate_pred
    flipped_diff = diff[flips]
    return {
        "mean_delta": float(np.mean(diff)),
        "mean_abs_delta": float(np.mean(np.abs(diff))),
        "max_abs_delta": float(np.max(np.abs(diff))),
        "flipped_mean_abs_delta": float(np.mean(np.abs(flipped_diff))) if flips.any() else 0.0,
        "n_flipped": float(np.sum(flips)),
    }


def margin(probabilities: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 probability per row.

    A small margin means the row sits near the decision boundary, which is where
    quantisation noise is most likely to flip it.  Returns an all-zero array for
    a single-class problem.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape[1] < 2:
        return np.zeros(probs.shape[0], dtype=np.float64)
    part = np.partition(probs, -2, axis=1)
    return part[:, -1] - part[:, -2]


def margin_shift(record: PredictionRecord) -> dict[str, float]:
    """How much the decision margin shrank or grew under compression."""
    ref_margin = margin(record.reference_probs)
    cand_margin = margin(record.candidate_probs)
    diff = cand_margin - ref_margin
    return {
        "reference_mean": float(np.mean(ref_margin)),
        "candidate_mean": float(np.mean(cand_margin)),
        "mean_delta": float(np.mean(diff)),
        "max_shrinkage": float(np.min(diff)),
        "n_below_zero": float(np.sum(cand_margin <= 0.0)),
    }


def _rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks, 1-based, with ties receiving the mean of their span."""
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    ranks[order] = np.arange(1, arr.size + 1, dtype=np.float64)
    # Average ranks within runs of equal values.
    sorted_vals = arr[order]
    start = 0
    for i in range(1, arr.size + 1):
        if i == arr.size or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranks[order[start:i]] = np.mean(ranks[order[start:i]])
            start = i
    return ranks


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation with tie-averaged ranks.

    Implemented directly rather than imported so the tie handling is explicit
    and testable: ties are common here because quantised confidences repeat.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"length mismatch: {x.size} vs {y.size}")
    if x.size < 2:
        return float("nan")
    rx = _rank_average(x)
    ry = _rank_average(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx**2) * np.sum(ry**2)))
    if denom == 0.0:
        # One side is constant: correlation is undefined, not zero.
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b, handling tied pairs.

    O(n^2) is fine for the test splits used here and avoids a scipy round trip
    inside the metric loop.  tau-b divides by the geometric mean of the
    non-tied pair counts, so heavy tie blocks (INT4 confidences) do not produce
    a misleadingly small correlation.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"length mismatch: {x.size} vs {y.size}")
    n = x.size
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        prod = dx * dy
        concordant += int(np.sum(prod > 0))
        discordant += int(np.sum(prod < 0))
        tie_x += int(np.sum((dx == 0) & (dy != 0)))
        tie_y += int(np.sum((dy == 0) & (dx != 0)))
    denom = np.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def confidence_rank_agreement(record: PredictionRecord) -> dict[str, float]:
    """Rank correlation of confidence orderings between the two models.

    Both global (across all rows) and per-class correlations are reported.  The
    global number is dominated by the easy majority of the data; the per-class
    numbers show whether a specific class's confidence ordering collapsed.
    """
    ref_conf = record.reference_confidence
    cand_conf = record.candidate_confidence
    out = {
        "spearman_global": spearman_correlation(ref_conf, cand_conf),
        "kendall_global": kendall_tau(ref_conf, cand_conf),
        "spearman_argmax_agreement": float(
            np.mean(record.reference_pred == record.candidate_pred)
        ),
    }
    n_classes = record.reference_probs.shape[1]
    per_class: list[float] = []
    for cls in range(n_classes):
        mask = record.labels == cls
        if int(mask.sum()) < 3:
            continue
        value = spearman_correlation(ref_conf[mask], cand_conf[mask])
        if not np.isnan(value):
            per_class.append(value)
    out["spearman_class_mean"] = float(np.mean(per_class)) if per_class else float("nan")
    out["spearman_class_min"] = float(np.min(per_class)) if per_class else float("nan")
    return out


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Standard binned ECE of the top-1 confidence.

    Included because quantisation often makes a model *more* confidently wrong:
    the grid clips logits into fewer distinct values, so the confidence
    histogram changes shape even when accuracy does not.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = labels.size
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(conf[mask]))
        bin_acc = float(np.mean(correct[mask]))
        ece += abs(bin_conf - bin_acc) * (float(np.sum(mask)) / n)
    return float(ece)


def compare_predictions(
    reference_probs: np.ndarray,
    candidate_probs: np.ndarray,
    labels: np.ndarray,
    n_classes: int | None = None,
) -> dict[str, object]:
    """All behavioural metrics for one reference/candidate pair.

    This is the single function the sweep calls per configuration; keeping the
    assembly here means a report row is always the same set of keys.
    """
    record = PredictionRecord(reference_probs, candidate_probs, labels)
    n_cls = n_classes or int(np.asarray(reference_probs).shape[1])
    return {
        "accuracy": accuracy_delta(reference_probs, candidate_probs, labels),
        "churn": decision_churn(record),
        "confidence": confidence_delta(record),
        "margin": margin_shift(record),
        "rank": confidence_rank_agreement(record),
        "calibration_error": {
            "reference_ece": expected_calibration_error(reference_probs, labels),
            "candidate_ece": expected_calibration_error(candidate_probs, labels),
        },
        "per_class_reference": per_class_accuracy(reference_probs, labels, n_cls),
        "per_class_candidate": per_class_accuracy(candidate_probs, labels, n_cls),
        "confusion_reference": confusion_matrix(
            record.reference_pred, labels, n_cls
        ).tolist(),
        "confusion_candidate": confusion_matrix(
            record.candidate_pred, labels, n_cls
        ).tolist(),
    }
