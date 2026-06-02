"""End-to-end tests: run_sweep, report writing, and the CLI.

These are the slowest tests in the suite because they train a model and run the
whole configuration matrix.  The config is deliberately small (three formats,
one scheme, ten epochs) so the whole module finishes in a few seconds while
still exercising every code path a real run takes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantbench.cli import build_parser, main
from quantbench.config import CalibrationConfig, DataConfig, ModelConfig, SweepConfig
from quantbench.dtypes import NumericFormat
from quantbench.report import (
    REPORT_SCHEMA_VERSION,
    ReportWriter,
    build_report,
    compare_reports,
    export_csv,
    load_report,
    summarise_report,
    write_report,
)
from quantbench.sweep import run_sweep


@pytest.fixture(scope="module")
def config() -> SweepConfig:
    """A small but complete sweep configuration."""
    return SweepConfig(
        model=ModelConfig(hidden_sizes=(16, 8), epochs=8, batch_size=32, learning_rate=0.05, seed=4),
        data=DataConfig(n_samples=400, n_features=6, n_classes=3, n_informative=5, seed=6),
        calibration=CalibrationConfig(method="entropy", n_bins=256, n_calibration_samples=64),
        formats=(NumericFormat.FP32, NumericFormat.INT8),
        schemes=("symmetric",),
        granularities=("per_tensor", "per_channel"),
        include_sensitivity=True,
        sensitivity_top_k=1,
    )


@pytest.fixture(scope="module")
def result(config: SweepConfig):
    """One completed sweep, shared by every test in the module."""
    return run_sweep(config)


def test_run_sweep_produces_one_row_per_configuration(config: SweepConfig, result) -> None:
    assert len(result.rows) == len(config.quant_grid())
    assert [row.label for row in result.rows] == [p.label for p in config.quant_grid()]


def test_run_sweep_reference_row_is_the_unquantised_model(result) -> None:
    fp32 = next(row for row in result.rows if row.format == "fp32")
    # FP32 quantisation is exact, so the behavioural metrics must be neutral.
    assert fp32.accuracy["accuracy_delta"] == pytest.approx(0.0, abs=1e-12)
    assert fp32.churn["flip_rate"] == pytest.approx(0.0)
    assert fp32.compression_ratio == pytest.approx(1.0)


def test_run_sweep_records_a_training_curve(config: SweepConfig, result) -> None:
    assert len(result.history.train_loss) == config.model.epochs
    assert len(result.history.test_accuracy) == config.model.epochs
    assert result.history.train_loss[-1] < result.history.train_loss[0]


def test_run_sweep_includes_sensitivity_when_requested(result) -> None:
    assert result.sensitivity["format"] == "int8"
    assert len(result.sensitivity["ranking"]) == 3
    assert result.mixed_precision is not None
    assert result.mixed_precision["reference_accuracy"] == pytest.approx(result.reference_accuracy)


def test_run_sweep_cumulative_curve_has_one_step_per_layer(result) -> None:
    assert [step["n_layers_quantised"] for step in result.cumulative] == [0, 1, 2, 3]


def test_run_sweep_calibrates_every_activation_site(result) -> None:
    assert result.calibration["method"] == "entropy"
    assert "input" in result.calibration["layers"]
    assert "hidden_0" in result.calibration["layers"]


def test_run_sweep_is_deterministic(config: SweepConfig, result) -> None:
    again = run_sweep(config)
    assert again.reference_accuracy == pytest.approx(result.reference_accuracy)
    assert [row.accuracy["accuracy_delta"] for row in again.rows] == [
        row.accuracy["accuracy_delta"] for row in result.rows
    ]
