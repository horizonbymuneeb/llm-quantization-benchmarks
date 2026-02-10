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


def make_clusters(
    n_samples: int,
    n_features: int,
    n_classes: int,
    n_informative: int,
    class_sep: float,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic Gaussian blobs with ``class_sep`` between centres.

    Only ``n_informative`` dimensions carry class signal; the rest are pure
    noise.  That matters for quantisation: a weight column that is essentially
    noise can be crushed to a few levels without hurting accuracy, while a
    signal column cannot, so the per-channel scales end up very uneven — which
    is exactly the situation where per-channel beats per-tensor.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, class_sep, size=(n_classes, n_informative))
    per_class = _split_counts(n_samples, n_classes)
    labels = np.repeat(np.arange(n_classes), per_class)
    rng.shuffle(labels)

    informative = np.empty((n_samples, n_informative), dtype=np.float32)
    for cls in range(n_classes):
        mask = labels == cls
        count = int(mask.sum())
        if count == 0:
            continue
        informative[mask] = centres[cls] + rng.normal(
            0.0, noise, size=(count, n_informative)
        )

    x = np.empty((n_samples, n_features), dtype=np.float32)
    x[:, :n_informative] = informative
    if n_features > n_informative:
        x[:, n_informative:] = rng.normal(0.0, noise, size=(n_samples, n_features - n_informative))
    return _standardise(x), labels.astype(np.int64)


def _split_counts(n_samples: int, n_classes: int) -> np.ndarray:
    """Split ``n_samples`` across classes as evenly as possible."""
    base = n_samples // n_classes
    counts = np.full(n_classes, base, dtype=np.int64)
    counts[: n_samples % n_classes] += 1
    return counts
