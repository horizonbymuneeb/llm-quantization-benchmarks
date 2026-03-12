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
