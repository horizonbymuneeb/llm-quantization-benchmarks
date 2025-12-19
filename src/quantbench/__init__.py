"""quantbench — a quantisation benchmark harness for how compression changes
model behaviour.

The package is organised around one idea: an accuracy number is not enough to
describe what compression did to a model.  The modules are:

``dtypes``       storage-format simulation (FP32/FP16/BF16/INT8/INT4)
``config``       typed, validated config objects backed by config.yaml
``data``         seeded synthetic dataset generators
``model``        the numpy MLP under test, with a hand-written Adam
``calibration``  activation range selection (minmax/percentile/entropy/mse)
``quantize``     symmetric/asymmetric, per-tensor/per-channel weight quantisation
``metrics``      accuracy delta, decision churn, confidence rank agreement
``sensitivity``  per-layer, cumulative, and mixed-precision analysis
``sweep``        the experiment driver that ties the above together
``report``       JSON report assembly, diffing, and CSV export
``cli``          command-line interface
"""

from __future__ import annotations

__version__ = "0.3.1"
