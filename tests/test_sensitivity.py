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


def test_rank_layers_can_sort_by_accuracy_drop(trained: tuple) -> None:
    params, dataset, _ = trained
    results = single_layer_sensitivity(
        params, dataset.x_test, dataset.y_test, NumericFormat.INT4
    )
    ranked = rank_layers(results, by="accuracy_drop")
    drops = [r.accuracy_drop for r in ranked]
    assert drops == sorted(drops, reverse=True)


def test_rank_layers_handles_an_empty_list() -> None:
    assert rank_layers([]) == []


def test_cumulative_sensitivity_starts_at_the_baseline(trained: tuple) -> None:
    params, dataset, _ = trained
    curve = cumulative_sensitivity(
        params, dataset.x_test, dataset.y_test, NumericFormat.INT8
    )
    assert curve[0]["n_layers_quantised"] == 0
    assert curve[0]["incremental_drop"] == 0.0
    assert len(curve) == params.n_layers + 1


def test_cumulative_sensitivity_is_monotone_in_layer_count(trained: tuple) -> None:
    params, dataset, _ = trained
    curve = cumulative_sensitivity(
        params, dataset.x_test, dataset.y_test, NumericFormat.INT8
    )
    counts = [int(step["n_layers_quantised"]) for step in curve]
    assert counts == list(range(params.n_layers + 1))
    # The cumulative drop is the sum of the incremental drops by construction.
    total = sum(float(step["incremental_drop"]) for step in curve[1:])
    assert total == pytest.approx(float(curve[-1]["cumulative_drop"]), abs=1e-9)


def test_format_sensitivity_orders_errors_by_width(trained: tuple) -> None:
    params, dataset, _ = trained
    rows = format_sensitivity(
        params,
        dataset.x_test,
        dataset.y_test,
        (NumericFormat.FP32, NumericFormat.BF16, NumericFormat.INT8, NumericFormat.INT4),
    )
    by_format = {row["format"]: row for row in rows}
    assert by_format["fp32"]["weight_rmse"] == pytest.approx(0.0)
    assert by_format["bf16"]["weight_rmse"] < by_format["int8"]["weight_rmse"]
    assert by_format["int8"]["weight_rmse"] < by_format["int4"]["weight_rmse"]


def test_format_sensitivity_rejects_a_bad_index(trained: tuple) -> None:
    params, dataset, _ = trained
    with pytest.raises(IndexError, match="out of range"):
        format_sensitivity(params, dataset.x_test, dataset.y_test, (NumericFormat.INT8,), layer_index=7)


def test_parameter_share_accounts_for_biases_in_the_denominator(trained: tuple) -> None:
    params, _, _ = trained
    rows = parameter_share(params)
    n_weights = sum(int(w.size) for w in params.weights)
    n_biases = sum(int(b.size) for b in params.biases)
    # Weight shares leave the bias vectors' share unaccounted for.
    assert sum(float(r["share"]) for r in rows) == pytest.approx(
        n_weights / (n_weights + n_biases), abs=1e-6
    )
    # Shares relative to the weight matrices alone sum to exactly one.
    assert sum(float(r["share_of_weights"]) for r in rows) == pytest.approx(1.0, abs=1e-6)
    assert all(r["layer"].startswith("layers.") for r in rows)
    assert [r["shape"] for r in rows] == [list(w.shape) for w in params.weights]


def test_recommend_mixed_precision_selects_the_top_k(trained: tuple) -> None:
    params, dataset, _ = trained
    ranked = rank_layers(
        single_layer_sensitivity(params, dataset.x_test, dataset.y_test, NumericFormat.INT8)
    )
    assignment = recommend_mixed_precision(ranked, top_k=1)
    assert assignment["n_layers_wide"] == 1
    assert assignment["sensitive_layers"] == [ranked[0].index]
    assert assignment["sensitive_format"] == "bf16"


def test_recommend_mixed_precision_with_zero_top_k_keeps_everything_narrow(trained: tuple) -> None:
    params, dataset, _ = trained
    ranked = rank_layers(
        single_layer_sensitivity(params, dataset.x_test, dataset.y_test, NumericFormat.INT8)
    )
    assignment = recommend_mixed_precision(ranked, top_k=0)
    assert assignment["sensitive_layers"] == []
    assert assignment["n_layers_wide"] == 0


def test_recommend_mixed_precision_rejects_a_negative_top_k(trained: tuple) -> None:
    params, dataset, _ = trained
    ranked = rank_layers(
        single_layer_sensitivity(params, dataset.x_test, dataset.y_test, NumericFormat.INT8)
    )
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        recommend_mixed_precision(ranked, top_k=-1)


def test_apply_mixed_precision_uses_the_requested_formats(trained: tuple) -> None:
    params, _, _ = trained
    assignment = {
        "sensitive_layers": [0],
        "sensitive_format": "bf16",
        "default_format": "int4",
        "n_layers_wide": 1,
    }
    mixed = apply_mixed_precision(params, assignment)
    # Layer 0 keeps BF16 precision, so its error must be far smaller than the
    # INT4 error on layer 1.
    err0 = float(np.max(np.abs(mixed.weights[0] - params.weights[0])))
    err1 = float(np.max(np.abs(mixed.weights[1] - params.weights[1])))
    assert err0 < err1


def test_apply_mixed_precision_of_an_all_float_assignment_is_identity(trained: tuple) -> None:
    params, _, _ = trained
    assignment = {
        "sensitive_layers": [],
        "sensitive_format": "fp32",
        "default_format": "fp32",
        "n_layers_wide": 0,
    }
    mixed = apply_mixed_precision(params, assignment)
    for a, b in zip(mixed.weights, params.weights):
        np.testing.assert_array_equal(a, b)


def test_sensitivity_summary_picks_the_extremes(trained: tuple) -> None:
    params, dataset, _ = trained
    ranked = rank_layers(
        single_layer_sensitivity(params, dataset.x_test, dataset.y_test, NumericFormat.INT4)
    )
    summary = sensitivity_summary(ranked)
    assert summary["n_layers"] == params.n_layers
    assert summary["most_sensitive"]["index"] == ranked[0].index  # type: ignore[index]
    assert summary["least_sensitive"]["index"] == ranked[-1].index  # type: ignore[index]
    assert summary["max_drop"] >= summary["mean_drop"] >= summary["min_drop"]


def test_sensitivity_summary_of_an_empty_ranking() -> None:
    summary = sensitivity_summary([])
    assert summary["n_layers"] == 0
    assert summary["most_sensitive"] is None


def test_layer_sensitivity_as_dict_has_the_report_keys(trained: tuple) -> None:
    params, dataset, _ = trained
    result = single_layer_sensitivity(
        params, dataset.x_test, dataset.y_test, NumericFormat.INT8
    )[0]
    payload = result.as_dict()
    assert set(payload) == {
        "layer",
        "index",
        "format",
        "accuracy",
        "accuracy_drop",
        "output_relative_error",
        "weight_rmse",
        "weight_cosine_distance",
        "n_parameters",
    }
    assert payload["layer"] == "layers.0.weight"
