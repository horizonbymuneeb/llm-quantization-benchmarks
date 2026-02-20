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
