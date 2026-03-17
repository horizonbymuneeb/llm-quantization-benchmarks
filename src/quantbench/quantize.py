"""Quantisation of weight matrices, per-tensor and per-channel.

The module produces *dequantised* float32 tensors rather than packed integers.
That choice is deliberate: the benchmark is about how compression changes model
behaviour, not about writing a fast kernel, so every quantised layer still runs
through the same numpy forward pass and the only thing that differs is the
numbers it sees.  Packing is simulated separately by
:func:`quantbench.dtypes.storage_bytes` when compression ratios are reported.

Symmetric quantisation maps ``[-a, a]`` onto a signed grid with a representable
zero.  Asymmetric quantisation maps ``[lo, hi]`` onto the full unsigned grid and
carries a zero point, which recovers range when the distribution is not centred
(such as post-ReLU activations) at the cost of one extra term per dot product.
"""
