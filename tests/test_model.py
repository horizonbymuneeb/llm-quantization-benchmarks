"""Tests for the numpy MLP: forward shapes, gradients, and training."""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.config import ModelConfig
from quantbench.model import (
    AdamState,
    MLPParams,
    accuracy,
    activation_grad,
    activate,
    backward,
    build_and_train,
    cross_entropy,
    forward,
    init_params,
    layer_names,
    layer_sizes,
    predict,
    predict_proba,
    softmax,
    train,
)


def test_layer_sizes_brackets_the_hidden_widths() -> None:
    assert layer_sizes(12, [64, 32], 5) == [12, 64, 32, 5]
    assert layer_sizes(4, [], 2) == [4, 2]


def test_layer_sizes_rejects_degenerate_arguments() -> None:
    with pytest.raises(ValueError, match="n_classes at least 2"):
        layer_sizes(4, [8], 1)
    with pytest.raises(ValueError, match="hidden sizes must be positive"):
        layer_sizes(4, [0], 3)


def test_init_params_shapes_follow_the_size_list() -> None:
    params = init_params([12, 16, 5], "relu", seed=1)
    assert params.n_layers == 2
    assert params.weights[0].shape == (16, 12)
    assert params.weights[1].shape == (5, 16)
    assert params.biases[0].shape == (16,)
    assert params.biases[1].shape == (5,)
    assert params.total_parameters() == 16 * 12 + 16 + 5 * 16 + 5


def test_init_params_is_deterministic_for_a_fixed_seed() -> None:
    a = init_params([8, 8, 3], seed=11)
    b = init_params([8, 8, 3], seed=11)
    np.testing.assert_array_equal(a.weights[0], b.weights[0])


def test_he_initialisation_scale_matches_the_fan_in() -> None:
    params = init_params([400, 200, 5], "relu", seed=3)
    # He normal: std = sqrt(2 / fan_in) = sqrt(2 / 400).
    assert float(np.std(params.weights[0])) == pytest.approx(np.sqrt(2.0 / 400.0), rel=0.08)


def test_init_params_rejects_an_unknown_activation() -> None:
    with pytest.raises(ValueError, match="unsupported activation"):
        init_params([4, 3], "gelu")


def test_mlp_params_copy_is_deep() -> None:
    params = init_params([4, 3], seed=0)
    clone = params.copy()
    clone.weights[0][0, 0] = 99.0
    assert params.weights[0][0, 0] != 99.0


def test_named_tensors_lists_weights_and_biases_per_layer() -> None:
    params = init_params([4, 6, 3], seed=0)
    names = [n for n, _ in params.named_tensors()]
    assert names == [
        "layers.0.weight",
        "layers.0.bias",
        "layers.1.weight",
        "layers.1.bias",
    ]
    assert params.shapes() == [(6, 4), (3, 6)]


def test_relu_and_its_gradient() -> None:
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(activate(x, "relu"), [0.0, 0.0, 2.0])
    np.testing.assert_array_equal(activation_grad(x, "relu"), [0.0, 0.0, 1.0])


def test_tanh_and_its_gradient() -> None:
    x = np.array([0.0], dtype=np.float32)
    assert activate(x, "tanh")[0] == pytest.approx(0.0)
    assert activation_grad(x, "tanh")[0] == pytest.approx(1.0)
    big = np.array([2.0], dtype=np.float32)
    assert activation_grad(big, "tanh")[0] == pytest.approx(1.0 - np.tanh(2.0) ** 2, rel=1e-5)


def test_activation_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unsupported activation"):
        activate(np.zeros(3), "swish")
    with pytest.raises(ValueError, match="unsupported activation"):
        activation_grad(np.zeros(3), "swish")


def test_softmax_rows_sum_to_one_and_are_shift_invariant() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], rtol=1e-6)
    shifted = softmax(logits + 1000.0)
    np.testing.assert_allclose(probs, shifted, rtol=1e-5)


def test_softmax_does_not_overflow_on_large_logits() -> None:
    probs = softmax(np.array([[1000.0, 1001.0]], dtype=np.float32))
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-6)


def test_cross_entropy_matches_a_hand_computed_value() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    expected = -0.5 * (np.log(0.9) + np.log(0.8))
    assert cross_entropy(probs, labels) == pytest.approx(float(expected), rel=1e-6)


def test_cross_entropy_clips_zero_probability() -> None:
    probs = np.array([[1.0, 0.0]], dtype=np.float32)
    value = cross_entropy(probs, np.array([1], dtype=np.int64))
    assert np.isfinite(value)
    assert value > 20.0


def test_forward_returns_probabilities_of_the_right_shape() -> None:
    params = init_params([6, 5, 3], seed=0)
    x = np.random.default_rng(0).normal(size=(7, 6)).astype(np.float32)
    probs, cache = forward(params, x)
    assert probs.shape == (7, 3)
    assert cache is None
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(7), rtol=1e-6)


def test_forward_with_record_fills_the_cache() -> None:
    params = init_params([6, 5, 3], seed=0)
    x = np.random.default_rng(0).normal(size=(7, 6)).astype(np.float32)
    _, cache = forward(params, x, record=True)
    assert cache is not None
    assert len(cache.pre_activations) == 2
    assert len(cache.post_activations) == 1
    assert cache.probabilities is not None
    assert cache.probabilities.shape == (7, 3)


def test_forward_rejects_a_feature_mismatch() -> None:
    params = init_params([6, 3], seed=0)
    with pytest.raises(ValueError, match="features but the model expects"):
        forward(params, np.zeros((4, 5), dtype=np.float32))
