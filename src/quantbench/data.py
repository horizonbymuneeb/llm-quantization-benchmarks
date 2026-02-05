"""Synthetic datasets for the benchmark, generated with numpy only.

No dataset downloads and no network: every split is produced by a seeded
generator, which means a report is reproducible from its config file alone.

Three generators are provided because they stress different parts of the
classifier.  ``clusters`` is separable Gaussians and is the main workload.
``moons`` has a curved decision boundary, so quantisation error in the first
layer shows up as churn rather than as an accuracy drop.  ``ring`` is
concentric and deliberately *hard* near the boundary, which is where a
compressed model changes its mind most often.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DataConfig


@dataclass(frozen=True)
class Dataset:
    """A train/test split plus the bookkeeping a report needs."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    n_features: int
    generator: str

    @property
    def n_train(self) -> int:
        """Number of training rows."""
        return int(self.x_train.shape[0])

    @property
    def n_test(self) -> int:
        """Number of test rows."""
        return int(self.x_test.shape[0])

    def class_counts(self) -> dict[str, int]:
        """Per-class row counts for the test split."""
        counts = np.bincount(self.y_test, minlength=self.n_classes)
        return {f"class_{i}": int(c) for i, c in enumerate(counts)}

    def summary(self) -> dict[str, object]:
        """JSON-friendly description used in the report header."""
        return {
            "generator": self.generator,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "test_class_counts": self.class_counts(),
        }


def _standardise(x: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance standardisation, guarding constant columns."""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((x - mean) / std).astype(np.float32)
