"""Tests for config loading, validation, and the sweep matrix expansion."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from quantbench.config import (
    CalibrationConfig,
    DataConfig,
    ModelConfig,
    QuantConfig,
    SweepConfig,
    config_to_dict,
    load_config,
)
from quantbench.dtypes import NumericFormat


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_default_model_config_is_valid() -> None:
    ModelConfig().validate()


def test_model_config_rejects_empty_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        ModelConfig(hidden_sizes=()).validate()


def test_model_config_rejects_non_positive_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden sizes must be positive"):
        ModelConfig(hidden_sizes=(16, 0)).validate()


def test_model_config_rejects_an_unknown_activation() -> None:
    with pytest.raises(ValueError, match="unsupported activation"):
        ModelConfig(activation="gelu").validate()


def test_model_config_rejects_a_non_positive_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        ModelConfig(learning_rate=0.0).validate()
