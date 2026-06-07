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


def test_worst_row_is_at_least_as_bad_as_the_best_row(result) -> None:
    worst = result.worst_row()
    best = result.best_row()
    assert worst is not None and best is not None
    assert worst["accuracy"]["accuracy_delta"] <= best["accuracy"]["accuracy_delta"]


def test_best_row_breaks_ties_on_compression_ratio(result) -> None:
    best = result.best_row()
    assert best is not None
    # Among equally accurate rows the widest compression must win.
    same_delta = [
        row for row in result.rows
        if row.accuracy["accuracy_delta"] == best["accuracy"]["accuracy_delta"]
    ]
    assert best["compression_ratio"] == max(row.compression_ratio for row in same_delta)


def test_churn_outliers_only_lists_rows_above_the_threshold(result) -> None:
    for item in result.churn_outliers(threshold=0.0):
        assert item["flip_rate"] > 0.0
        assert item["ratio"] > 5.0


def test_build_report_has_the_documented_schema(config: SweepConfig, result) -> None:
    report = build_report(config, result)
    assert set(report) == {
        "meta",
        "config",
        "dataset",
        "training",
        "calibration",
        "reference_accuracy",
        "rows",
        "delta_table",
        "format_aggregates",
        "sensitivity",
        "cumulative_sensitivity",
        "mixed_precision",
        "parameter_share",
        "summary",
    }
    assert report["meta"]["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["meta"]["tool"] == "quantbench"


def test_build_report_is_strict_json_serialisable(config: SweepConfig, result) -> None:
    report = build_report(config, result)
    # allow_nan=False is the point: the report must not contain NaN literals.
    text = json.dumps(report, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text


def test_write_and_reload_a_report(config: SweepConfig, result, tmp_path: Path) -> None:
    from dataclasses import replace

    local = replace(config, output_dir=str(tmp_path / "reports"), report_name="out.json")
    path = write_report(local, result)
    assert path.exists()
    reloaded = load_report(path)
    assert reloaded["reference_accuracy"] == pytest.approx(round(result.reference_accuracy, 6))
    assert len(reloaded["rows"]) == len(result.rows)


def test_report_writer_creates_missing_directories(config: SweepConfig, result, tmp_path: Path) -> None:
    writer = ReportWriter(output_dir=tmp_path / "a" / "b" / "c", filename="r.json")
    path = writer.write(build_report(config, result))
    assert path.exists()
    assert path.parent.is_dir()


def test_load_report_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="report not found"):
        load_report(tmp_path / "nope.json")


def test_load_report_rejects_a_file_without_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"meta": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing the 'rows' key"):
        load_report(path)


def test_load_report_rejects_a_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="report root must be a mapping"):
        load_report(path)


def test_summarise_report_mentions_the_headline_numbers(config: SweepConfig, result) -> None:
    text = summarise_report(build_report(config, result))
    assert "reference accuracy" in text
    assert "configurations: 3" in text
    assert "fp32" in text


def test_compare_reports_of_identical_runs_is_all_zero(config: SweepConfig, result) -> None:
    report = build_report(config, result)
    diff = compare_reports(report, report)
    assert diff["n_shared_configs"] == len(result.rows)
    assert diff["only_in_baseline"] == []
    assert diff["only_in_candidate"] == []
    assert diff["reference_accuracy_diff"] == pytest.approx(0.0)
    for row in diff["rows"]:
        assert row["accuracy_delta_diff"] == pytest.approx(0.0)
        assert row["flip_rate_diff"] == pytest.approx(0.0)


def test_compare_reports_detects_a_missing_configuration(config: SweepConfig, result) -> None:
    full = build_report(config, result)
    trimmed = json.loads(json.dumps(full))
    trimmed["rows"] = trimmed["rows"][1:]
    diff = compare_reports(full, trimmed)
    assert diff["n_shared_configs"] == len(full["rows"]) - 1
    assert diff["only_in_baseline"] == [full["rows"][0]["label"]]


def test_export_csv_writes_a_header_and_one_line_per_row(
    config: SweepConfig, result, tmp_path: Path
) -> None:
    report = build_report(config, result)
    path = export_csv(report, tmp_path / "rows.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(result.rows) + 1
    assert lines[0].startswith("label,format,scheme")
    assert "fp32" in lines[1]


def test_cli_parser_exposes_every_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["sweep", "--config", "config.yaml", "--csv", "out.csv"])
    assert args.command == "sweep"
    assert args.config == "config.yaml"
    assert args.csv == "out.csv"
    assert parser.parse_args(["inspect", "--format", "int4"]).granularity == "per_channel"
    assert parser.parse_args(["compare", "a.json", "b.json"]).baseline == "a.json"
    assert parser.parse_args(["calibrate"]).command == "calibrate"


def test_cli_without_a_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_version_flag_prints_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert "quantbench" in capsys.readouterr().out


def test_cli_sweep_writes_a_report(config: SweepConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
formats: [fp32, int8]
schemes: [symmetric]
granularities: [per_tensor]
include_sensitivity: false
output_dir: {tmp_path / 'reports'}
report_name: cli.json
model:
  hidden_sizes: [8]
  epochs: 2
  batch_size: 32
data:
  n_samples: 60
  n_features: 4
  n_classes: 2
  n_informative: 3
calibration:
  method: minmax
  n_bins: 64
  n_calibration_samples: 16
""",
        encoding="utf-8",
    )
    code = main(["sweep", "--config", str(config_path), "--quiet"])
    assert code == 0
    assert (tmp_path / "reports" / "cli.json").exists()


def test_cli_sweep_with_a_missing_config_returns_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["sweep", "--config", str(tmp_path / "absent.yaml")])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_cli_inspect_emits_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  hidden_sizes: [8]
  epochs: 2
data:
  n_samples: 60
  n_features: 4
  n_classes: 2
  n_informative: 3
""",
        encoding="utf-8",
    )
    code = main(["inspect", "--config", str(config_path), "--format", "int8", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "int8"
    assert "behaviour" in payload
    assert payload["behaviour"]["churn"]["flip_rate"] >= 0.0


def test_cli_inspect_rejects_an_unknown_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  epochs: 1\n", encoding="utf-8")
    code = main(["inspect", "--config", str(config_path), "--format", "int3"])
    assert code == 1
    assert "unknown numeric format" in capsys.readouterr().err


def test_cli_compare_prints_the_shared_count(config: SweepConfig, result, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_report(
        SweepConfig(output_dir=str(tmp_path), report_name="r.json"), result
    )
    code = main(["compare", str(path), str(path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "shared configurations:" in out


def test_cli_calibrate_prints_each_activation_site(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  hidden_sizes: [8]
  epochs: 2
data:
  n_samples: 60
  n_features: 4
  n_classes: 2
  n_informative: 3
calibration:
  method: percentile
  percentile: 99.0
  n_bins: 128
  n_calibration_samples: 16
""",
        encoding="utf-8",
    )
    code = main(["calibrate", "--config", str(config_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "method=percentile" in out
    assert "hidden_0" in out
