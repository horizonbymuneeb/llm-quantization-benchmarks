"""Typed configuration objects backed by ``config.yaml``.

The harness is driven by a single YAML file so a sweep can be reproduced from
the command line alone.  Rather than reaching into a dict with string keys all
over the code, every section is materialised into a frozen dataclass here, which
means a typo in the YAML fails loudly at load time instead of silently
defaulting to something else three modules later.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .dtypes import NumericFormat

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass(frozen=True)
class ModelConfig:
    """Shape and training schedule of the numpy MLP under test."""

    hidden_sizes: tuple[int, ...] = (64, 32)
    activation: str = "relu"
    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 0.05
    weight_decay: float = 1e-4
    seed: int = 7

    def validate(self) -> None:
        """Reject configurations that cannot train."""
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        if any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden sizes must be positive, got {self.hidden_sizes}")
        if self.activation not in ("relu", "tanh"):
            raise ValueError(f"unsupported activation {self.activation!r}")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @property
    def layers(self) -> list[int]:
        """Hidden layer widths as a mutable list, for building the model."""
        return list(self.hidden_sizes)


@dataclass(frozen=True)
class DataConfig:
    """Synthetic dataset generation parameters."""

    n_samples: int = 2400
    n_features: int = 12
    n_classes: int = 5
    n_informative: int = 8
    class_sep: float = 1.6
    noise: float = 0.55
    test_fraction: float = 0.3
    seed: int = 11
    generator: str = "clusters"

    def validate(self) -> None:
        """Check the knobs that would otherwise produce a broken dataset."""
        if self.n_samples < 20:
            raise ValueError("n_samples must be at least 20")
        if not 2 <= self.n_classes <= 20:
            raise ValueError("n_classes must be between 2 and 20")
        if not 1 <= self.n_informative <= self.n_features:
            raise ValueError("n_informative must be within [1, n_features]")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must lie strictly between 0 and 1")
        if self.generator not in ("clusters", "moons", "ring"):
            raise ValueError(f"unknown generator {self.generator!r}")


@dataclass(frozen=True)
class CalibrationConfig:
    """How activation ranges are chosen before quantising."""

    method: str = "entropy"
    percentile: float = 99.9
    n_bins: int = 2048
    n_calibration_samples: int = 256
    seed: int = 3

    def validate(self) -> None:
        """Validate method-specific parameters."""
        allowed = ("minmax", "percentile", "entropy", "mse")
        if self.method not in allowed:
            raise ValueError(f"calibration method must be one of {allowed}, got {self.method!r}")
        if not 50.0 <= self.percentile < 100.0:
            raise ValueError("percentile must lie in [50, 100)")
        if self.n_bins < 16:
            raise ValueError("n_bins must be at least 16")
        if self.n_calibration_samples < 1:
            raise ValueError("n_calibration_samples must be positive")


@dataclass(frozen=True)
class QuantConfig:
    """One point in quantisation design space."""

    format: NumericFormat = NumericFormat.INT8
    scheme: str = "symmetric"
    granularity: str = "per_channel"
    round_mode: str = "nearest"

    def validate(self) -> None:
        """Validate scheme/granularity combinations."""
        if self.scheme not in ("symmetric", "asymmetric"):
            raise ValueError(f"scheme must be symmetric or asymmetric, got {self.scheme!r}")
        if self.granularity not in ("per_tensor", "per_channel"):
            raise ValueError(
                f"granularity must be per_tensor or per_channel, got {self.granularity!r}"
            )
        if self.round_mode not in ("nearest", "stochastic"):
            raise ValueError(f"unknown round_mode {self.round_mode!r}")
        if self.scheme == "asymmetric" and self.format.is_float:
            # Float formats have no zero point; asymmetric is meaningless there.
            raise ValueError(
                f"asymmetric quantisation requires an integer format, got {self.format.value}"
            )

    @property
    def label(self) -> str:
        """Short stable identifier used as a report key."""
        return f"{self.format.value}-{self.scheme}-{self.granularity}"


@dataclass(frozen=True)
class SweepConfig:
    """The full experiment matrix read from ``config.yaml``."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    formats: tuple[NumericFormat, ...] = (
        NumericFormat.FP32,
        NumericFormat.BF16,
        NumericFormat.INT8,
        NumericFormat.INT4,
    )
    schemes: tuple[str, ...] = ("symmetric", "asymmetric")
    granularities: tuple[str, ...] = ("per_tensor", "per_channel")
    weight_layers_only: bool = True
    include_sensitivity: bool = True
    sensitivity_top_k: int = 4
    output_dir: str = "reports"
    report_name: str = "sweep_report.json"

    def validate(self) -> None:
        """Validate every nested section plus the matrix itself."""
        self.model.validate()
        self.data.validate()
        self.calibration.validate()
        if not self.formats:
            raise ValueError("at least one numeric format is required")
        if not self.schemes:
            raise ValueError("at least one scheme is required")
        if not self.granularities:
            raise ValueError("at least one granularity is required")
        if self.sensitivity_top_k < 1:
            raise ValueError("sensitivity_top_k must be positive")

    def quant_grid(self) -> list[QuantConfig]:
        """Expand the matrix into concrete :class:`QuantConfig` points.

        Float formats have no scheme or granularity, so they contribute exactly
        one point each instead of the full cross product.  That keeps the report
        from filling with twelve identical BF16 rows.
        """
        points: list[QuantConfig] = []
        for fmt in self.formats:
            if fmt.is_float:
                points.append(QuantConfig(format=fmt, scheme="symmetric", granularity="per_tensor"))
                continue
            for scheme in self.schemes:
                for gran in self.granularities:
                    points.append(QuantConfig(format=fmt, scheme=scheme, granularity=gran))
        return points


def _as_tuple(value: Any) -> tuple[Any, ...]:
    """Coerce a YAML list (or scalar) into a tuple."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _build(cls: type, payload: dict[str, Any] | None) -> Any:
    """Instantiate a dataclass from a partial dict, ignoring unknown keys loudly."""
    payload = payload or {}
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key in ("hidden_sizes", "formats", "schemes", "granularities"):
            kwargs[key] = _as_tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> SweepConfig:
    """Read a YAML file and return a validated :class:`SweepConfig`.

    A missing file is an error: silently running with defaults would make a
    report that does not correspond to the config the user thinks they ran.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")

    formats = tuple(
        NumericFormat.from_string(str(f)) for f in _as_tuple(raw.get("formats"))
    ) or SweepConfig.formats

    config = SweepConfig(
        model=_build(ModelConfig, raw.get("model")),
        data=_build(DataConfig, raw.get("data")),
        calibration=_build(CalibrationConfig, raw.get("calibration")),
        formats=formats,
        schemes=_as_tuple(raw.get("schemes")) or SweepConfig.schemes,
        granularities=_as_tuple(raw.get("granularities")) or SweepConfig.granularities,
        weight_layers_only=bool(raw.get("weight_layers_only", True)),
        include_sensitivity=bool(raw.get("include_sensitivity", True)),
        sensitivity_top_k=int(raw.get("sensitivity_top_k", 4)),
        output_dir=str(raw.get("output_dir", "reports")),
        report_name=str(raw.get("report_name", "sweep_report.json")),
    )
    config.validate()
    return config


def config_to_dict(config: SweepConfig) -> dict[str, Any]:
    """Serialise a config back to plain JSON-compatible data.

    Written into every report so a result can be traced to the exact settings
    that produced it without keeping the YAML around.
    """
    def conv(value: Any) -> Any:
        if isinstance(value, NumericFormat):
            return value.value
        if isinstance(value, tuple):
            return [conv(v) for v in value]
        if isinstance(value, dict):
            return {k: conv(v) for k, v in value.items()}
        return value

    return {
        "model": conv(dataclasses.asdict(config.model)),
        "data": conv(dataclasses.asdict(config.data)),
        "calibration": conv(dataclasses.asdict(config.calibration)),
        "formats": [f.value for f in config.formats],
        "schemes": list(config.schemes),
        "granularities": list(config.granularities),
        "weight_layers_only": config.weight_layers_only,
        "include_sensitivity": config.include_sensitivity,
        "sensitivity_top_k": config.sensitivity_top_k,
        "output_dir": config.output_dir,
        "report_name": config.report_name,
    }
