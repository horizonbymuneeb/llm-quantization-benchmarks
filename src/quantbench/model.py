"""A small multi-layer perceptron written in numpy.

The model exists so the benchmark has something with real layers to compress.
It is deliberately small and fully in numpy: no GPU, no framework version
pinning, and — most usefully — every weight matrix is reachable by name, which
is what per-layer sensitivity analysis needs.

Training uses Adam with cross-entropy loss.  Weights are stored in a
:class:`MLPParams` object rather than on the model, so a quantised copy of the
weights can be swapped in without cloning the model or mutating it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ModelConfig


@dataclass
class MLPParams:
    """Weights and biases, one matrix per linear layer.

    ``weights[i]`` has shape ``(out_features, in_features)`` and
    ``biases[i]`` has shape ``(out_features,)``.  Row-major with output units
    first is the layout the per-channel quantiser assumes.
    """

    weights: list[np.ndarray]
    biases: list[np.ndarray]

    @property
    def n_layers(self) -> int:
        """Number of linear layers."""
        return len(self.weights)

    def copy(self) -> "MLPParams":
        """Deep copy, so a quantisation experiment cannot mutate the source."""
        return MLPParams(
            weights=[w.copy() for w in self.weights],
            biases=[b.copy() for b in self.biases],
        )

    def named_tensors(self) -> list[tuple[str, np.ndarray]]:
        """Flat ``(name, array)`` list; the unit of per-layer analysis."""
        out: list[tuple[str, np.ndarray]] = []
        for i in range(self.n_layers):
            out.append((f"layers.{i}.weight", self.weights[i]))
            out.append((f"layers.{i}.bias", self.biases[i]))
        return out

    def total_parameters(self) -> int:
        """Total scalar count across weights and biases."""
        return sum(int(w.size) for w in self.weights) + sum(int(b.size) for b in self.biases)

    def shapes(self) -> list[tuple[int, int]]:
        """Weight matrix shapes, in layer order."""
        return [tuple(w.shape) for w in self.weights]  # type: ignore[misc]


@dataclass
class ForwardCache:
    """Per-layer intermediates kept for backpropagation and calibration.

    ``pre_activations[i]`` is the affine output of layer ``i`` and
    ``post_activations[i]`` is that after the nonlinearity; the last entry is
    the softmax probability matrix.
    """

    inputs: list[np.ndarray] = field(default_factory=list)
    pre_activations: list[np.ndarray] = field(default_factory=list)
    post_activations: list[np.ndarray] = field(default_factory=list)
    probabilities: np.ndarray | None = None


def layer_sizes(n_features: int, hidden_sizes: list[int] | tuple[int, ...], n_classes: int) -> list[int]:
    """Full width list including input and output dimensions."""
    if n_features <= 0 or n_classes < 2:
        raise ValueError("n_features must be positive and n_classes at least 2")
    if any(h <= 0 for h in hidden_sizes):
        raise ValueError(f"hidden sizes must be positive, got {hidden_sizes}")
    return [int(n_features), *[int(h) for h in hidden_sizes], int(n_classes)]


def init_params(
    sizes: list[int],
    activation: str = "relu",
    seed: int = 0,
) -> MLPParams:
    """He (ReLU) or Xavier (tanh) initialisation.

    The choice matters for a *quantisation* benchmark: a badly scaled
    initialisation leaves tiny weights that quantise to zero, so the INT4 arm
    would report catastrophic damage that is an artefact of the init rather
    than of the format.
    """
    if len(sizes) < 2:
        raise ValueError("need at least input and output sizes")
    if activation not in ("relu", "tanh"):
        raise ValueError(f"unsupported activation {activation!r}")
    rng = np.random.default_rng(seed)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        if activation == "relu":
            std = np.sqrt(2.0 / fan_in)
        else:
            std = np.sqrt(2.0 / (fan_in + fan_out))
        w = rng.normal(0.0, std, size=(fan_out, fan_in)).astype(np.float32)
        weights.append(w)
        biases.append(np.zeros(fan_out, dtype=np.float32))
    return MLPParams(weights=weights, biases=biases)


def activate(x: np.ndarray, activation: str) -> np.ndarray:
    """Apply the nonlinearity."""
    if activation == "relu":
        return np.maximum(x, 0.0).astype(np.float32)
    if activation == "tanh":
        return np.tanh(x).astype(np.float32)
    raise ValueError(f"unsupported activation {activation!r}")


def activation_grad(x: np.ndarray, activation: str) -> np.ndarray:
    """Derivative of the nonlinearity, evaluated at the pre-activation ``x``."""
    if activation == "relu":
        return (x > 0.0).astype(np.float32)
    if activation == "tanh":
        t = np.tanh(x)
        return (1.0 - t * t).astype(np.float32)
    raise ValueError(f"unsupported activation {activation!r}")


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax.

    Subtracting the row max before exponentiating is not optional here: the
    affine outputs of the last layer reach the hundreds during early training,
    and ``exp`` of that overflows float32.
    """
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted.astype(np.float32))
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)


def cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy in nats, clipped to avoid ``log(0)``."""
    labels = np.asarray(labels, dtype=np.int64)
    eps = 1e-12
    picked = probabilities[np.arange(labels.size), labels]
    return float(-np.mean(np.log(np.clip(picked, eps, 1.0))))


def forward(
    params: MLPParams,
    x: np.ndarray,
    activation: str = "relu",
    record: bool = False,
) -> tuple[np.ndarray, ForwardCache | None]:
    """Run the network and return softmax probabilities.

    When ``record`` is true the returned cache holds every intermediate array,
    which both backpropagation and activation-range calibration consume.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D input, got shape {x.shape}")
    if x.shape[1] != params.weights[0].shape[1]:
        raise ValueError(
            f"input has {x.shape[1]} features but the model expects {params.weights[0].shape[1]}"
        )
    cache = ForwardCache() if record else None
    current = np.asarray(x, dtype=np.float32)
    n = params.n_layers
    for i in range(n):
        if cache is not None:
            cache.inputs.append(current)
        pre = current @ params.weights[i].T + params.biases[i]
        if i == n - 1:
            if cache is not None:
                cache.pre_activations.append(pre)
            current = pre
            break
        post = activate(pre, activation)
        if cache is not None:
            cache.pre_activations.append(pre)
            cache.post_activations.append(post)
        current = post
    probs = softmax(current)
    if cache is not None:
        cache.probabilities = probs
    return probs, cache


def backward(
    params: MLPParams,
    cache: ForwardCache,
    labels: np.ndarray,
    activation: str = "relu",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Backpropagate the cross-entropy gradient.

    The softmax/cross-entropy pair has the well-known ``p - onehot`` gradient,
    so the output layer needs no separate Jacobian.
    """
    if cache.probabilities is None:
        raise ValueError("cache is missing probabilities; call forward(record=True)")
    labels = np.asarray(labels, dtype=np.int64)
    batch = labels.size
    n_classes = cache.probabilities.shape[1]
    grad_logits = cache.probabilities.copy()
    grad_logits[np.arange(batch), labels] -= 1.0
    grad_logits /= float(batch)

    grad_w: list[np.ndarray] = [np.zeros(0)] * params.n_layers
    grad_b: list[np.ndarray] = [np.zeros(0)] * params.n_layers
    delta = grad_logits
    for i in range(params.n_layers - 1, -1, -1):
        layer_in = cache.inputs[i]
        grad_w[i] = (delta.T @ layer_in).astype(np.float32)
        grad_b[i] = np.sum(delta, axis=0).astype(np.float32)
        if i > 0:
            delta = (delta @ params.weights[i]) * activation_grad(
                cache.pre_activations[i - 1], activation
            )
    return grad_w, grad_b


class AdamState:
    """First and second moment estimates for the Adam optimiser."""

    def __init__(self, params: MLPParams, beta1: float = 0.9, beta2: float = 0.999) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_w = [np.zeros_like(w) for w in params.weights]
        self.v_w = [np.zeros_like(w) for w in params.weights]
        self.m_b = [np.zeros_like(b) for b in params.biases]
        self.v_b = [np.zeros_like(b) for b in params.biases]
        self.t = 0

    def step(
        self,
        params: MLPParams,
        grad_w: list[np.ndarray],
        grad_b: list[np.ndarray],
        lr: float,
        weight_decay: float = 0.0,
    ) -> None:
        """Apply one Adam update in place on ``params``.

        Decoupled weight decay (AdamW form) is used rather than adding
        ``wd * w`` to the gradient, because L2-in-the-gradient interacts badly
        with adaptive scaling and shrinks large weights less than small ones.
        """
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        eps = 1e-8
        for i in range(params.n_layers):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grad_w[i] ** 2)
            m_hat = self.m_w[i] / bc1
            v_hat = self.v_w[i] / bc2
            params.weights[i] -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)
            if weight_decay:
                params.weights[i] -= (lr * weight_decay * params.weights[i]).astype(np.float32)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grad_b[i] ** 2)
            m_hat_b = self.m_b[i] / bc1
            v_hat_b = self.v_b[i] / bc2
            params.biases[i] -= (lr * m_hat_b / (np.sqrt(v_hat_b) + eps)).astype(np.float32)


@dataclass
class TrainHistory:
    """Per-epoch loss and accuracy, kept for the report."""

    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    test_accuracy: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[float]]:
        """JSON-friendly view."""
        return {
            "train_loss": [round(v, 6) for v in self.train_loss],
            "train_accuracy": [round(v, 6) for v in self.train_accuracy],
            "test_accuracy": [round(v, 6) for v in self.test_accuracy],
        }


def accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of rows whose argmax matches the label."""
    preds = np.argmax(probabilities, axis=1)
    return float(np.mean(preds == np.asarray(labels, dtype=np.int64)))


def train(
    params: MLPParams,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: ModelConfig,
    x_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> TrainHistory:
    """Fit ``params`` in place and return the training curve.

    Full-batch shuffling per epoch with mini-batch Adam.  The last batch of an
    epoch may be smaller than ``batch_size`` and is used as-is rather than
    dropped, so a dataset that is not a multiple of the batch size still
    contributes all of its rows.
    """
    config.validate()
    rng = np.random.default_rng(config.seed)
    optimiser = AdamState(params)
    history = TrainHistory()
    n = x_train.shape[0]
    for epoch in range(config.epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, n, config.batch_size):
            idx = order[start : start + config.batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            probs, cache = forward(params, xb, config.activation, record=True)
            assert cache is not None
            epoch_loss += cross_entropy(probs, yb) * idx.size
            seen += idx.size
            grad_w, grad_b = backward(params, cache, yb, config.activation)
            optimiser.step(params, grad_w, grad_b, config.learning_rate, config.weight_decay)
        history.train_loss.append(epoch_loss / max(seen, 1))
        train_probs, _ = forward(params, x_train, config.activation)
        history.train_accuracy.append(accuracy(train_probs, y_train))
        if x_test is not None and y_test is not None:
            test_probs, _ = forward(params, x_test, config.activation)
            history.test_accuracy.append(accuracy(test_probs, y_test))
    return history


def predict(params: MLPParams, x: np.ndarray, activation: str = "relu") -> np.ndarray:
    """Predicted class indices."""
    probs, _ = forward(params, x, activation)
    return np.argmax(probs, axis=1)


def predict_proba(params: MLPParams, x: np.ndarray, activation: str = "relu") -> np.ndarray:
    """Predicted class probabilities."""
    probs, _ = forward(params, x, activation)
    return probs
