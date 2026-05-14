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


def test_forward_rejects_a_one_dimensional_input() -> None:
    params = init_params([6, 3], seed=0)
    with pytest.raises(ValueError, match="expected a 2-D input"):
        forward(params, np.zeros(6, dtype=np.float32))


def test_backward_matches_a_finite_difference_gradient() -> None:
    rng = np.random.default_rng(4)
    params = init_params([5, 4, 3], "tanh", seed=2)
    x = rng.normal(size=(6, 5)).astype(np.float32)
    y = rng.integers(0, 3, size=6)

    probs, cache = forward(params, x, "tanh", record=True)
    assert cache is not None
    grad_w, grad_b = backward(params, cache, y, "tanh")

    def loss_of(p: MLPParams) -> float:
        out, _ = forward(p, x, "tanh")
        return cross_entropy(out, y)

    eps = 1e-3
    for i, j in [(0, 0), (0, 2), (1, 1)]:
        plus = params.copy()
        minus = params.copy()
        plus.weights[i][j, 0] += eps
        minus.weights[i][j, 0] -= eps
        numeric = (loss_of(plus) - loss_of(minus)) / (2 * eps)
        assert grad_w[i][j, 0] == pytest.approx(numeric, rel=0.05, abs=1e-3)

    plus = params.copy()
    minus = params.copy()
    plus.biases[1][0] += eps
    minus.biases[1][0] -= eps
    numeric_b = (loss_of(plus) - loss_of(minus)) / (2 * eps)
    assert grad_b[1][0] == pytest.approx(numeric_b, rel=0.05, abs=1e-3)


def test_backward_requires_a_recorded_forward() -> None:
    from quantbench.model import ForwardCache

    params = init_params([4, 3], seed=0)
    with pytest.raises(ValueError, match="missing probabilities"):
        backward(params, ForwardCache(), np.array([0]))


def test_adam_state_takes_a_step_toward_the_gradient() -> None:
    params = init_params([4, 3], seed=0)
    optimiser = AdamState(params)
    before = params.weights[0].copy()
    grad_w = [np.ones_like(w) for w in params.weights]
    grad_b = [np.ones_like(b) for b in params.biases]
    optimiser.step(params, grad_w, grad_b, lr=0.1)
    assert np.all(params.weights[0] < before)


def test_adam_weight_decay_shrinks_weights_without_a_gradient() -> None:
    params = init_params([4, 3], seed=0)
    optimiser = AdamState(params)
    before = params.weights[0].copy()
    zero_w = [np.zeros_like(w) for w in params.weights]
    zero_b = [np.zeros_like(b) for b in params.biases]
    optimiser.step(params, zero_w, zero_b, lr=0.1, weight_decay=0.5)
    assert np.all(np.abs(params.weights[0]) <= np.abs(before))


def test_accuracy_counts_matches() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
    labels = np.array([0, 1, 1], dtype=np.int64)
    assert accuracy(probs, labels) == pytest.approx(2.0 / 3.0)


def test_train_reduces_the_loss_on_a_separable_problem() -> None:
    rng = np.random.default_rng(6)
    x = np.concatenate([rng.normal(-2.0, 0.4, size=(200, 3)), rng.normal(2.0, 0.4, size=(200, 3))]).astype(np.float32)
    y = np.concatenate([np.zeros(200, dtype=np.int64), np.ones(200, dtype=np.int64)])
    params = init_params([3, 16, 2], "relu", seed=0)
    config = ModelConfig(hidden_sizes=(16,), epochs=30, batch_size=32, learning_rate=0.05, seed=0)
    history = train(params, x, y, config)
    assert len(history.train_loss) == 30
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.train_accuracy[-1] > 0.95


def test_train_records_test_accuracy_when_given_a_test_split() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100, 4)).astype(np.float32)
    y = rng.integers(0, 2, size=100)
    params = init_params([4, 8, 2], "relu", seed=0)
    config = ModelConfig(hidden_sizes=(8,), epochs=3, batch_size=32, learning_rate=0.05, seed=0)
    history = train(params, x, y, config, x[:20], y[:20])
    assert len(history.test_accuracy) == 3
    assert all(0.0 <= a <= 1.0 for a in history.test_accuracy)


def test_train_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(64, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=64)
    config = ModelConfig(hidden_sizes=(8,), epochs=5, batch_size=16, seed=3)
    a, hist_a = build_and_train(config, 4, 3, x, y, x, y)
    b, hist_b = build_and_train(config, 4, 3, x, y, x, y)
    np.testing.assert_array_equal(a.weights[0], b.weights[0])
    assert hist_a.train_loss == hist_b.train_loss


def test_train_validates_its_config() -> None:
    params = init_params([4, 3], seed=0)
    x = np.zeros((8, 4), dtype=np.float32)
    y = np.zeros(8, dtype=np.int64)
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        train(params, x, y, ModelConfig(hidden_sizes=(4,), learning_rate=0.0))


def test_predict_and_predict_proba_agree() -> None:
    params = init_params([4, 3], seed=0)
    x = np.random.default_rng(9).normal(size=(10, 4)).astype(np.float32)
    preds = predict(params, x)
    probs = predict_proba(params, x)
    np.testing.assert_array_equal(preds, np.argmax(probs, axis=1))


def test_layer_names_describe_the_shapes() -> None:
    params = init_params([12, 8, 3], seed=0)
    assert layer_names(params) == ["layer0_8x12", "layer1_3x8"]
