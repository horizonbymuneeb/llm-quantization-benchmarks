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

__all__ = [
    "__version__",
    "NumericFormat",
    "SweepConfig",
    "load_config",
    "run_sweep",
    "build_dataset",
    "MLPParams",
    "quantize_model",
    "compare_predictions",
    "calibrate",
    "single_layer_sensitivity",
    "write_report",
]


def __getattr__(name: str):
    """Lazily re-export the public API so ``import quantbench`` stays cheap.

    Importing every submodule eagerly would pull numpy into any consumer that
    only wants ``__version__``, and it would also make the import order matter
    during development.
    """
    mapping = {
        "NumericFormat": ("dtypes", "NumericFormat"),
        "SweepConfig": ("config", "SweepConfig"),
        "load_config": ("config", "load_config"),
        "run_sweep": ("sweep", "run_sweep"),
        "build_dataset": ("data", "build_dataset"),
        "MLPParams": ("model", "MLPParams"),
        "quantize_model": ("sweep", "quantize_model"),
        "compare_predictions": ("metrics", "compare_predictions"),
        "calibrate": ("calibration", "calibrate"),
        "single_layer_sensitivity": ("sensitivity", "single_layer_sensitivity"),
        "write_report": ("report", "write_report"),
    }
    if name in mapping:
        module_name, attr = mapping[name]
        import importlib

        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
