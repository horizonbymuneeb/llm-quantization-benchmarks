"""Tests for the synthetic dataset generators and the stratified split."""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.config import DataConfig
from quantbench.data import (
    Dataset,
    build_dataset,
    calibration_subset,
    class_weights,
    make_clusters,
    make_moons,
    make_rings,
    one_hot,
    stratified_split,
)


def test_clusters_shapes_and_label_range() -> None:
    x, y = make_clusters(500, 10, 4, 6, 1.5, 0.5, seed=0)
    assert x.shape == (500, 10)
    assert y.shape == (500,)
    assert x.dtype == np.float32
    assert set(np.unique(y)) == {0, 1, 2, 3}


def test_clusters_are_standardised() -> None:
    x, _ = make_clusters(2000, 8, 3, 5, 1.5, 0.5, seed=1)
    np.testing.assert_allclose(x.mean(axis=0), np.zeros(8), atol=1e-5)
    np.testing.assert_allclose(x.std(axis=0), np.ones(8), atol=1e-4)


def test_clusters_are_deterministic_for_a_fixed_seed() -> None:
    a, ya = make_clusters(100, 6, 3, 4, 1.0, 0.5, seed=7)
    b, yb = make_clusters(100, 6, 3, 4, 1.0, 0.5, seed=7)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ya, yb)


def test_clusters_change_with_the_seed() -> None:
    a, _ = make_clusters(100, 6, 3, 4, 1.0, 0.5, seed=7)
    b, _ = make_clusters(100, 6, 3, 4, 1.0, 0.5, seed=8)
    assert not np.array_equal(a, b)


def test_clusters_class_separation_moves_the_centroids_apart() -> None:
    # Measure the gap between the two class centroids, not between two rows:
    # rows are shuffled and individually noisy, so single-row distances say
    # nothing about separation.
    def centroid_gap(sep: float) -> float:
        x, y = make_clusters(600, 4, 2, 4, sep, 0.3, seed=2)
        return float(np.linalg.norm(x[y == 0].mean(axis=0) - x[y == 1].mean(axis=0)))

    assert centroid_gap(4.0) > centroid_gap(0.5)


def test_clusters_split_the_sample_count_evenly_across_classes() -> None:
    _, y = make_clusters(1000, 5, 4, 3, 1.0, 0.5, seed=3)
    counts = np.bincount(y, minlength=4)
    assert counts.sum() == 1000
    assert int(counts.max() - counts.min()) <= 1


def test_moons_produces_the_requested_shape() -> None:
    x, y = make_moons(300, 7, 3, 0.1, seed=4)
    assert x.shape == (300, 7)
    assert set(np.unique(y)) == {0, 1, 2}


def test_moons_rejects_a_single_class() -> None:
    with pytest.raises(ValueError, match="at least 2 classes"):
        make_moons(100, 4, 1, 0.1, seed=0)


def test_rings_are_ordered_by_radius() -> None:
    x, y = make_rings(900, 2, 3, 0.05, seed=5)
    radii = np.linalg.norm(x, axis=1)
    # Class 0 is the innermost ring and class 2 the outermost.
    assert radii[y == 0].mean() < radii[y == 1].mean() < radii[y == 2].mean()
