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
