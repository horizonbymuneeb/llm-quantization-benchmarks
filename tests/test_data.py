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
