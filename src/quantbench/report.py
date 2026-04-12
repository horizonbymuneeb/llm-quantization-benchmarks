"""JSON report assembly and on-disk output.

A report is written as a single JSON document with a stable top-level schema:
``meta``, ``config``, ``training``, ``dataset``, ``calibration``, ``rows``,
``sensitivity``, ``summary``.  The schema is stable so a downstream script can
diff two runs without special-casing missing keys.

NaN and infinity are legal in Python's ``json`` output but not in strict JSON,
and several editors reject them, so they are converted to ``null`` here rather
than left to the writer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import SweepConfig, config_to_dict
from .sweep import (
    SweepResult,
    aggregate_by_format,
    row_delta_table,
)

REPORT_SCHEMA_VERSION = "1.0"


def _sanitise(value: Any) -> Any:
    """Recursively replace NaN/inf with ``None`` and tuples with lists."""
    if isinstance(value, dict):
        return {str(k): _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def build_report(config: SweepConfig, result: SweepResult) -> dict[str, Any]:
    """Assemble the full report dictionary from a completed sweep."""
    return {
        "meta": {
            "schema_version": REPORT_SCHEMA_VERSION,
            "tool": "quantbench",
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "config": config_to_dict(config),
        "dataset": result.dataset.summary(),
        "training": {
            "epochs": len(result.history.train_loss),
            "final_train_loss": round(result.history.train_loss[-1], 6)
            if result.history.train_loss
            else None,
            "final_train_accuracy": round(result.history.train_accuracy[-1], 6)
            if result.history.train_accuracy
            else None,
            "final_test_accuracy": round(result.history.test_accuracy[-1], 6)
            if result.history.test_accuracy
            else None,
            "history": result.history.as_dict(),
        },
        "calibration": result.calibration,
        "reference_accuracy": round(result.reference_accuracy, 6),
        "rows": [row.as_dict() for row in result.rows],
        "delta_table": row_delta_table(result.rows),
        "format_aggregates": aggregate_by_format(result.rows),
        "sensitivity": result.sensitivity,
        "cumulative_sensitivity": result.cumulative,
        "mixed_precision": result.mixed_precision,
        "parameter_share": result.layer_parameter_share,
        "summary": {
            "n_configurations": len(result.rows),
            "best": result.best_row(),
            "worst": result.worst_row(),
            "churn_outliers": result.churn_outliers(),
            "reference_accuracy": round(result.baseline_accuracy, 6),
        },
    }


@dataclass(frozen=True)
class ReportWriter:
    """Writes a report to disk, creating the output directory if needed."""

    output_dir: Path
    filename: str = "sweep_report.json"

    @property
    def path(self) -> Path:
        """Full path of the report file."""
        return self.output_dir / self.filename

    def write(self, report: dict[str, Any]) -> Path:
        """Serialise ``report`` as strict JSON and return the written path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_sanitise(report), indent=2, allow_nan=False)
        self.path.write_text(payload + "\n", encoding="utf-8")
        return self.path


def write_report(config: SweepConfig, result: SweepResult) -> Path:
    """Build and write the report described by ``config``."""
    writer = ReportWriter(
        output_dir=Path(config.output_dir),
        filename=config.report_name,
    )
    return writer.write(build_report(config, result))


def load_report(path: str | Path) -> dict[str, Any]:
    """Read a previously written report.

    Used by the CLI's ``--compare`` mode and by tests that round-trip a report.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"report not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"report root must be a mapping, got {type(data).__name__}")
    if "rows" not in data:
        raise ValueError("report is missing the 'rows' key")
    return data


def summarise_report(report: dict[str, Any]) -> str:
    """Render a short human-readable digest of a report.

    Kept textual (not a table) so it can be pasted into a log or a commit
    message without alignment artefacts.
    """
    lines: list[str] = []
    meta = report.get("meta", {})
    lines.append(f"quantbench report {meta.get('version', '?')} ({meta.get('generated_at', '?')})")
    ref = report.get("reference_accuracy")
    lines.append(f"reference accuracy: {ref}")
    rows = report.get("rows", [])
    lines.append(f"configurations: {len(rows)}")
    for entry in report.get("delta_table", [])[:10]:
        lines.append(
            f"  {entry['label']:<34s} x{entry['compression']:<6.2f} "
            f"acc={entry['accuracy']:<7.4f} delta={entry['delta']:+.4f} "
            f"flips={entry['flip_rate']:.4f}"
        )
    outliers = report.get("summary", {}).get("churn_outliers", [])
    if outliers:
        lines.append(f"churn outliers ({len(outliers)}):")
        for item in outliers[:5]:
            lines.append(
                f"  {item['label']}: flip_rate={item['flip_rate']:.4f} "
                f"vs accuracy_delta={item['accuracy_delta']:.4f}"
            )
    return "\n".join(lines)


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Diff two reports row by row, keyed on the configuration label.

    Only the headline numbers are compared: the purpose is to see whether a
    change to the harness moved the conclusions, not to diff every float.
    """
    base_rows = {row["label"]: row for row in baseline.get("rows", [])}
    cand_rows = {row["label"]: row for row in candidate.get("rows", [])}
    shared = sorted(set(base_rows) & set(cand_rows))
    rows: list[dict[str, Any]] = []
    for label in shared:
        b = base_rows[label]
        c = cand_rows[label]
        rows.append(
            {
                "label": label,
                "accuracy_delta_diff": round(
                    c["accuracy"]["accuracy_delta"] - b["accuracy"]["accuracy_delta"], 6
                ),
                "flip_rate_diff": round(
                    c["churn"]["flip_rate"] - b["churn"]["flip_rate"], 6
                ),
                "spearman_diff": _diff(
                    c["rank"]["spearman_global"], b["rank"]["spearman_global"]
                ),
            }
        )
    return {
        "n_shared_configs": len(shared),
        "only_in_baseline": sorted(set(base_rows) - set(cand_rows)),
        "only_in_candidate": sorted(set(cand_rows) - set(base_rows)),
        "reference_accuracy_diff": _diff(
            candidate.get("reference_accuracy"), baseline.get("reference_accuracy")
        ),
        "rows": rows,
    }


def _diff(a: Any, b: Any) -> float | None:
    """Numeric difference that tolerates missing or ``None`` values."""
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 6)


def export_csv(report: dict[str, Any], path: str | Path) -> Path:
    """Flatten the report rows into a CSV file for spreadsheet inspection.

    Written by hand rather than with a dataframe dependency so the project keeps
    its minimal requirements list.
    """
    out = Path(path)
    columns = [
        "label",
        "format",
        "scheme",
        "granularity",
        "compression_ratio",
        "accuracy",
        "accuracy_delta",
        "flip_rate",
        "net_flips",
        "spearman_global",
        "kendall_global",
        "mean_weight_rmse",
    ]
    lines = [",".join(columns)]
    for row in report.get("rows", []):
        values = [
            row.get("label", ""),
            row.get("format", ""),
            row.get("scheme", ""),
            row.get("granularity", ""),
            row.get("compression_ratio", ""),
            row.get("accuracy", {}).get("candidate_accuracy", ""),
            row.get("accuracy", {}).get("accuracy_delta", ""),
            row.get("churn", {}).get("flip_rate", ""),
            row.get("churn", {}).get("net_flips", ""),
            row.get("rank", {}).get("spearman_global", ""),
            row.get("rank", {}).get("kendall_global", ""),
            row.get("weight_error", {}).get("mean_rmse", ""),
        ]
        lines.append(",".join(str(v) for v in values))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
