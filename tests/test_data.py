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


def test_padding_a_generator_to_more_features_adds_noise_only() -> None:
    x, _ = make_moons(200, 10, 2, 0.1, seed=6)
    # The first two columns carry structure; the rest are standardised noise and
    # must not be constant.
    assert x.shape[1] == 10
    assert np.all(x[:, 2:].std(axis=0) > 0.0)


def test_stratified_split_keeps_every_class_on_both_sides() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3)).astype(np.float32)
    y = np.repeat(np.arange(4), 50)
    x_tr, y_tr, x_te, y_te = stratified_split(x, y, 0.25, seed=1)
    assert set(np.unique(y_tr)) == {0, 1, 2, 3}
    assert set(np.unique(y_te)) == {0, 1, 2, 3}
    assert x_tr.shape[0] + x_te.shape[0] == 200
    # Each class contributes round(50 * 0.25) = 12 test rows.
    counts = np.bincount(y_te, minlength=4)
    np.testing.assert_array_equal(counts, [12, 12, 12, 12])


def test_stratified_split_rejects_a_bad_fraction() -> None:
    x = np.zeros((10, 2), dtype=np.float32)
    y = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="test_fraction must lie"):
        stratified_split(x, y, 1.0, seed=0)
    with pytest.raises(ValueError, match="test_fraction must lie"):
        stratified_split(x, y, 0.0, seed=0)
