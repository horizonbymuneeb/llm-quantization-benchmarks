"""Tests for the sweep driver: quantisation of a model and row assembly."""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.config import CalibrationConfig, DataConfig, ModelConfig, SweepConfig
from quantbench.data import build_dataset
from quantbench.dtypes import NumericFormat
from quantbench.model import build_and_train
from quantbench.sweep import (
    QuantizedModel,
    SweepRow,
    aggregate_by_format,
    evaluate_model,
    quantize_model,
    row_delta_table,
    weight_error_report,
)


@pytest.fixture(scope="module")
def small_config() -> SweepConfig:
    """A deliberately tiny sweep so the module's tests stay fast."""
    return SweepConfig(
        model=ModelConfig(hidden_sizes=(16, 8), epochs=10, batch_size=32, learning_rate=0.05, seed=3),
        data=DataConfig(n_samples=400, n_features=6, n_classes=3, n_informative=5, seed=2),
        calibration=CalibrationConfig(method="entropy", n_bins=256, n_calibration_samples=64),
        formats=(NumericFormat.FP32, NumericFormat.INT8),
        schemes=("symmetric",),
        granularities=("per_tensor", "per_channel"),
        include_sensitivity=False,
    )


@pytest.fixture(scope="module")
def trained(small_config: SweepConfig) -> tuple:
    dataset = build_dataset(small_config.data)
    params, history = build_and_train(
        small_config.model,
        dataset.n_features,
        dataset.n_classes,
        dataset.x_train,
        dataset.y_train,
        dataset.x_test,
        dataset.y_test,
    )
    return params, dataset, history


def test_quantize_model_returns_a_quantized_model(trained: tuple) -> None:
    params, _, _ = trained
    model = quantize_model(params, NumericFormat.INT8, "symmetric", "per_channel")
    assert isinstance(model, QuantizedModel)
    assert model.format is NumericFormat.INT8
    assert len(model.quant_params) == params.n_layers
    assert len(model.layer_stats) == params.n_layers


def test_quantize_model_leaves_the_source_model_untouched(trained: tuple) -> None:
    params, _, _ = trained
    before = [w.copy() for w in params.weights]
    quantize_model(params, NumericFormat.INT4, "symmetric", "per_tensor")
    for original, current in zip(before, params.weights):
        np.testing.assert_array_equal(original, current)


def test_quantize_model_fp32_is_a_no_op(trained: tuple) -> None:
    params, _, _ = trained
    model = quantize_model(params, NumericFormat.FP32, "symmetric", "per_tensor")
    for original, quantised in zip(params.weights, model.params.weights):
        np.testing.assert_array_equal(original, quantised)


def test_quantize_model_can_include_biases(trained: tuple) -> None:
    params, _, _ = trained
    model = quantize_model(
        params, NumericFormat.INT4, "symmetric", "per_tensor", quantize_biases=True
    )
    assert any(name.endswith(".bias") for name in model.quant_params)
    assert any(name.endswith(".bias") for name in model.layer_stats)


def test_quantized_model_scale_spread_is_at_least_one(trained: tuple) -> None:
    params, _, _ = trained
    model = quantize_model(params, NumericFormat.INT8, "symmetric", "per_channel")
    for spread in model.scale_spread().values():
        assert spread >= 1.0


def test_quantized_model_total_storage_shrinks_with_the_format(trained: tuple) -> None:
    params, _, _ = trained
    fp32 = quantize_model(params, NumericFormat.FP32, "symmetric", "per_tensor")
    int8 = quantize_model(params, NumericFormat.INT8, "symmetric", "per_tensor")
    int4 = quantize_model(params, NumericFormat.INT4, "symmetric", "per_tensor")
    assert int8.total_storage_bytes() == fp32.total_storage_bytes() // 4
    assert int4.total_storage_bytes() == fp32.total_storage_bytes() // 8


def test_evaluate_model_returns_probabilities_and_accuracy(trained: tuple, small_config: SweepConfig) -> None:
    params, dataset, _ = trained
    probs, acc = evaluate_model(params, dataset, small_config.model.activation)
    assert probs.shape == (dataset.n_test, dataset.n_classes)
    assert 0.0 <= acc <= 1.0
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(dataset.n_test), rtol=1e-5)


def test_weight_error_report_is_zero_for_identical_models(trained: tuple) -> None:
    params, _, _ = trained
    stats = weight_error_report(params, params.copy())
    assert stats["rmse"] == pytest.approx(0.0)
    assert stats["max_abs"] == pytest.approx(0.0)


def test_weight_error_report_grows_as_precision_drops(trained: tuple) -> None:
    params, _, _ = trained
    bf16 = weight_error_report(
        params, quantize_model(params, NumericFormat.BF16, "symmetric", "per_tensor").params
    )
    int4 = weight_error_report(
        params, quantize_model(params, NumericFormat.INT4, "symmetric", "per_tensor").params
    )
    assert bf16["rmse"] < int4["rmse"]


def test_row_delta_table_is_sorted_by_compression() -> None:
    rows = [
        SweepRow(
            label=f"row{i}",
            format="int8",
            scheme="symmetric",
            granularity="per_tensor",
            compression_ratio=ratio,
            mean_weight_rmse=0.1,
            max_weight_rmse=0.2,
            accuracy={"candidate_accuracy": 0.5, "accuracy_delta": 0.0},
            churn={"flip_rate": 0.1},
            confidence={},
            margin={},
            rank={"spearman_global": 0.9},
            calibration_error={},
            layer_errors={},
            scale_spread={},
        )
        for i, ratio in enumerate([8.0, 1.0, 4.0])
    ]
    table = row_delta_table(rows)
    assert [row["compression"] for row in table] == [1.0, 4.0, 8.0]
    assert set(table[0]) == {"label", "compression", "accuracy", "delta", "flip_rate", "spearman"}


def test_aggregate_by_format_groups_the_rows() -> None:
    def make_row(fmt: str, delta: float, flips: float, ratio: float) -> SweepRow:
        return SweepRow(
            label=f"{fmt}-{delta}",
            format=fmt,
            scheme="symmetric",
            granularity="per_tensor",
            compression_ratio=ratio,
            mean_weight_rmse=0.0,
            max_weight_rmse=0.0,
            accuracy={"accuracy_delta": delta},
            churn={"flip_rate": flips},
            confidence={},
            margin={},
            rank={},
            calibration_error={},
            layer_errors={},
            scale_spread={},
        )

    rows = [
        make_row("int8", 0.0, 0.01, 4.0),
        make_row("int8", -0.02, 0.03, 4.0),
        make_row("int4", -0.05, 0.08, 8.0),
    ]
    aggregates = aggregate_by_format(rows)
    assert set(aggregates) == {"int8", "int4"}
    assert aggregates["int8"]["n_configs"] == 2.0
    assert aggregates["int8"]["mean_accuracy_delta"] == pytest.approx(-0.01)
    assert aggregates["int8"]["worst_accuracy_delta"] == pytest.approx(-0.02)
    assert aggregates["int4"]["mean_flip_rate"] == pytest.approx(0.08)


def test_sweep_row_as_dict_rounds_and_keeps_nan(trained: tuple) -> None:
    params, dataset, _ = trained
    model = quantize_model(params, NumericFormat.INT8, "symmetric", "per_channel")
    probs, _ = evaluate_model(model.params, dataset, "relu")
    ref, _ = evaluate_model(params, dataset, "relu")
    from quantbench.metrics import compare_predictions

    behaviour = compare_predictions(ref, probs, dataset.y_test, dataset.n_classes)
    row = SweepRow(
        label="int8-test",
        format="int8",
        scheme="symmetric",
        granularity="per_channel",
        compression_ratio=4.0,
        mean_weight_rmse=0.01,
        max_weight_rmse=0.02,
        accuracy=behaviour["accuracy"],  # type: ignore[arg-type]
        churn=behaviour["churn"],  # type: ignore[arg-type]
        confidence=behaviour["confidence"],  # type: ignore[arg-type]
        margin=behaviour["margin"],  # type: ignore[arg-type]
        rank=behaviour["rank"],  # type: ignore[arg-type]
        calibration_error=behaviour["calibration_error"],  # type: ignore[arg-type]
        layer_errors={"layers.0.weight": 0.01},
        scale_spread={"layers.0.weight": 3.0},
    )
    payload = row.as_dict()
    assert payload["label"] == "int8-test"
    assert payload["compression_ratio"] == 4.0
    assert "accuracy_delta" in payload["accuracy"]
    # NaN must survive as NaN so the report writer can convert it to null.
    payload["rank"]["spearman_global"] = float("nan")
    assert np.isnan(payload["rank"]["spearman_global"])
