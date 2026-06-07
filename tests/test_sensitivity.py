"""Tests for per-layer sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.config import DataConfig, ModelConfig
from quantbench.data import build_dataset
from quantbench.dtypes import NumericFormat
from quantbench.model import build_and_train
from quantbench.sensitivity import (
    LayerSensitivity,
    apply_mixed_precision,
    cumulative_sensitivity,
    format_sensitivity,
    layer_output_error,
    parameter_share,
    quantise_one_layer,
    rank_layers,
    recommend_mixed_precision,
    sensitivity_summary,
    single_layer_sensitivity,
)


@pytest.fixture(scope="module")
def trained() -> tuple:
    """A small trained model plus its test split, shared across the module."""
    config = ModelConfig(hidden_sizes=(16, 8), epochs=12, batch_size=32, learning_rate=0.05, seed=5)
    data_config = DataConfig(n_samples=500, n_features=6, n_classes=3, n_informative=5, seed=1)
    dataset = build_dataset(data_config)
    params, history = build_and_train(
        config,
        dataset.n_features,
        dataset.n_classes,
        dataset.x_train,
        dataset.y_train,
        dataset.x_test,
        dataset.y_test,
    )
    return params, dataset, history


def test_quantise_one_layer_changes_only_that_layer(trained: tuple) -> None:
    params, _, _ = trained
    out = quantise_one_layer(params, 1, NumericFormat.INT8)
    # Layer 1 changed; layers 0 and 2 are byte-identical copies.
    assert not np.array_equal(out.weights[1], params.weights[1])
    np.testing.assert_array_equal(out.weights[0], params.weights[0])
    np.testing.assert_array_equal(out.weights[2], params.weights[2])


def test_quantise_one_layer_leaves_the_source_untouched(trained: tuple) -> None:
    params, _, _ = trained
    before = params.weights[0].copy()
    quantise_one_layer(params, 0, NumericFormat.INT4)
    np.testing.assert_array_equal(params.weights[0], before)


def test_quantise_one_layer_leaves_biases_in_float(trained: tuple) -> None:
    params, _, _ = trained
    out = quantise_one_layer(params, 0, NumericFormat.INT4)
    for a, b in zip(out.biases, params.biases):
        np.testing.assert_array_equal(a, b)


def test_quantise_one_layer_rejects_a_bad_index(trained: tuple) -> None:
    params, _, _ = trained
    with pytest.raises(IndexError, match="out of range"):
        quantise_one_layer(params, 9, NumericFormat.INT8)
    with pytest.raises(IndexError, match="out of range"):
        quantise_one_layer(params, -1, NumericFormat.INT8)


def test_layer_output_error_is_zero_for_an_identical_model(trained: tuple) -> None:
    params, dataset, _ = trained
    assert layer_output_error(params, params.copy(), dataset.x_test, 0) == pytest.approx(0.0)


def test_layer_output_error_grows_with_coarser_formats(trained: tuple) -> None:
    params, dataset, _ = trained
    bf16 = layer_output_error(
        params, quantise_one_layer(params, 0, NumericFormat.BF16), dataset.x_test, 0
    )
    int4 = layer_output_error(
        params, quantise_one_layer(params, 0, NumericFormat.INT4), dataset.x_test, 0
    )
    assert bf16 < int4


def test_single_layer_sensitivity_covers_every_layer(trained: tuple) -> None:
    params, dataset, _ = trained
    results = single_layer_sensitivity(
        params, dataset.x_test, dataset.y_test, NumericFormat.INT8
    )
    assert len(results) == params.n_layers
    assert [r.index for r in results] == [0, 1, 2]
    assert all(isinstance(r, LayerSensitivity) for r in results)
    assert all(r.fmt is NumericFormat.INT8 for r in results)


def test_single_layer_sensitivity_accuracy_is_within_the_unit_interval(trained: tuple) -> None:
    params, dataset, _ = trained
    results = single_layer_sensitivity(
        params, dataset.x_test, dataset.y_test, NumericFormat.INT8
    )
    for result in results:
        assert 0.0 <= result.accuracy <= 1.0
        # The drop is measured against the unquantised baseline, so it can be
        # negative when quantisation happens to help a single row.
        assert -1.0 <= result.accuracy_drop <= 1.0


def test_single_layer_sensitivity_int4_damages_more_than_int8(trained: tuple) -> None:
    params, dataset, _ = trained
    int8 = single_layer_sensitivity(
        params, dataset.x_test, dataset.y_test, NumericFormat.INT8
    )
    int4 = single_layer_sensitivity(
        params, dataset.x_test, dataset.y_test, NumericFormat.INT4
    )
    assert max(r.accuracy_drop for r in int4) >= max(r.accuracy_drop for r in int8)
    assert max(r.output_relative_error for r in int4) > max(
        r.output_relative_error for r in int8
    )


def test_single_layer_sensitivity_records_parameter_counts(trained: tuple) -> None:
    params, dataset, _ = trained
    results = single_layer_sensitivity(
        params, dataset.x_test, dataset.y_test, NumericFormat.INT8
    )
    assert [r.n_parameters for r in results] == [int(w.size) for w in params.weights]


def test_rank_layers_orders_by_output_error_descending(trained: tuple) -> None:
    params, dataset, _ = trained
    results = single_layer_sensitivity(
        params, dataset.x_test, dataset.y_test, NumericFormat.INT4
    )
    ranked = rank_layers(results)
    errors = [r.output_relative_error for r in ranked]
    assert errors == sorted(errors, reverse=True)
