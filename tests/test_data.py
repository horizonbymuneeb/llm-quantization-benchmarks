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
