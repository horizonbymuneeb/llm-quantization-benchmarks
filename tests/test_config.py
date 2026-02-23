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


def test_model_config_layers_returns_a_mutable_list() -> None:
    config = ModelConfig(hidden_sizes=(32, 16))
    layers = config.layers
    layers.append(8)
    assert config.hidden_sizes == (32, 16)


def test_data_config_rejects_a_too_small_sample_count() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        DataConfig(n_samples=10).validate()


def test_data_config_rejects_an_out_of_range_informative_count() -> None:
    with pytest.raises(ValueError, match="n_informative must be within"):
        DataConfig(n_features=5, n_informative=6).validate()


def test_data_config_rejects_an_unknown_generator() -> None:
    with pytest.raises(ValueError, match="unknown generator"):
        DataConfig(generator="spirals").validate()


def test_data_config_rejects_a_degenerate_test_fraction() -> None:
    with pytest.raises(ValueError, match="test_fraction must lie"):
        DataConfig(test_fraction=1.0).validate()


def test_calibration_config_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="calibration method must be one of"):
        CalibrationConfig(method="histogram").validate()


def test_calibration_config_rejects_a_bad_percentile() -> None:
    with pytest.raises(ValueError, match="percentile must lie"):
        CalibrationConfig(percentile=100.0).validate()


def test_calibration_config_rejects_too_few_bins() -> None:
    with pytest.raises(ValueError, match="n_bins must be at least"):
        CalibrationConfig(n_bins=8).validate()


def test_quant_config_label_is_stable() -> None:
    point = QuantConfig(format=NumericFormat.INT4, scheme="asymmetric", granularity="per_channel")
    assert point.label == "int4-asymmetric-per_channel"


def test_quant_config_rejects_an_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="scheme must be symmetric or asymmetric"):
        QuantConfig(scheme="log").validate()


def test_quant_config_rejects_asymmetric_on_a_float_format() -> None:
    with pytest.raises(ValueError, match="requires an integer format"):
        QuantConfig(format=NumericFormat.BF16, scheme="asymmetric").validate()


def test_quant_config_rejects_an_unknown_round_mode() -> None:
    with pytest.raises(ValueError, match="unknown round_mode"):
        QuantConfig(round_mode="floor").validate()


def test_quant_grid_expands_integer_formats_only() -> None:
    config = SweepConfig(
        formats=(NumericFormat.BF16, NumericFormat.INT8),
        schemes=("symmetric", "asymmetric"),
        granularities=("per_tensor", "per_channel"),
    )
    grid = config.quant_grid()
    # BF16 contributes exactly one point; INT8 contributes 2 schemes x 2 granularities.
    assert len(grid) == 5
    assert sum(1 for p in grid if p.format is NumericFormat.BF16) == 1
    assert sum(1 for p in grid if p.format is NumericFormat.INT8) == 4


def test_quant_grid_of_the_default_config_is_ten_points() -> None:
    config = SweepConfig()
    assert len(config.quant_grid()) == 10


def test_sweep_config_validate_checks_the_matrix() -> None:
    with pytest.raises(ValueError, match="at least one numeric format"):
        SweepConfig(formats=()).validate()
    with pytest.raises(ValueError, match="at least one scheme"):
        SweepConfig(schemes=()).validate()
    with pytest.raises(ValueError, match="sensitivity_top_k must be positive"):
        SweepConfig(sensitivity_top_k=0).validate()


def test_load_config_reads_the_shipped_file() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert config.model.hidden_sizes == (64, 32)
    assert NumericFormat.INT4 in config.formats
    assert config.calibration.method == "entropy"


def test_load_config_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_parses_a_minimal_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        formats: [int8]
        schemes: [symmetric]
        granularities: [per_tensor]
        model:
          hidden_sizes: [8]
          epochs: 2
        data:
          n_samples: 40
          n_features: 4
          n_classes: 2
          n_informative: 3
        """,
    )
    config = load_config(path)
    assert config.formats == (NumericFormat.INT8,)
    assert config.model.hidden_sizes == (8,)
    assert config.model.epochs == 2
    assert config.data.n_samples == 40


def test_load_config_rejects_an_unknown_key(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        model:
          hidden_sizes: [8]
          learning_rat: 0.1
        """,
    )
    with pytest.raises(ValueError, match="unknown keys for ModelConfig"):
        load_config(path)


def test_load_config_rejects_an_unknown_format_name(tmp_path: Path) -> None:
    path = _write(tmp_path, "formats: [int3]\n")
    with pytest.raises(ValueError, match="unknown numeric format"):
        load_config(path)


def test_load_config_rejects_a_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- a list, not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config root must be a mapping"):
        load_config(path)


def test_load_config_validates_the_loaded_values(tmp_path: Path) -> None:
    path = _write(tmp_path, "model:\n  learning_rate: -1.0\n")
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        load_config(path)


def test_config_to_dict_serialises_enums_and_tuples() -> None:
    payload = config_to_dict(SweepConfig())
    assert payload["formats"] == ["fp32", "bf16", "int8", "int4"]
    assert payload["model"]["hidden_sizes"] == [64, 32]
    assert isinstance(payload["schemes"], list)
    assert payload["calibration"]["method"] == "entropy"


def test_config_to_dict_is_json_serialisable() -> None:
    import json

    json.dumps(config_to_dict(SweepConfig()))
