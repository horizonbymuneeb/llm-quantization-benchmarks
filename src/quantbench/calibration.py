"""Activation range selection ("calibration") for integer quantisation.

An INT8 kernel needs to know the float range it must cover before it can pick a
scale.  Getting that range wrong is the single largest source of accuracy loss
in post-training quantisation, and the four methods here fail in different ways:

* ``minmax`` uses the observed extremes.  One outlier then stretches the grid
  over a range that is mostly empty, so typical values land on few codes.
* ``percentile`` clips a fixed fraction of the tail.  Cheap, but the clip
  fraction is arbitrary and does not know what the layer actually does with
  those values.
* ``entropy`` (the KL method from the TensorRT/NVIDIA calibration literature)
  searches a clip point that minimises the information lost when the original
  distribution is squeezed into the reduced code count.
* ``mse`` minimises the reconstruction error of the quantise/dequantise round
  trip directly, which is what the model actually experiences.

The functions return a *range*, not a scale, so the caller can pair the range
with either symmetric or asymmetric scale derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtypes import NumericFormat
