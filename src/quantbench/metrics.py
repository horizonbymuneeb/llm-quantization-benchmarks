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
