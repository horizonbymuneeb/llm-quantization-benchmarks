"""Command-line entry point for the quantisation benchmark.

Subcommands exist rather than flags because the three modes have genuinely
different outputs: ``sweep`` writes a report, ``inspect`` explains one
configuration, and ``compare`` diffs two reports.  Every command takes
``--config`` so nothing depends on the working directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import SweepConfig, config_to_dict, load_config
from .data import build_dataset, calibration_subset
from .dtypes import NumericFormat
from .model import build_and_train, forward
from .report import (
    compare_reports,
    export_csv,
    load_report,
    summarise_report,
    write_report,
)
from .sweep import (
    aggregate_by_format,
    evaluate_model,
    quantize_model,
    row_delta_table,
    run_sweep,
    weight_error_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="quantbench",
        description="Quantisation benchmark harness for a small numpy MLP.",
    )
    parser.add_argument(
        "--version", action="store_true", help="print the package version and exit"
    )
    sub = parser.add_subparsers(dest="command")

    sweep_p = sub.add_parser("sweep", help="run the full sweep and write a JSON report")
    sweep_p.add_argument("--config", default=None, help="path to config.yaml")
    sweep_p.add_argument(
        "--output", default=None, help="override the report output directory"
    )
    sweep_p.add_argument(
        "--csv", default=None, help="also write a flat CSV of the rows to this path"
    )
    sweep_p.add_argument(
        "--quiet", action="store_true", help="suppress the printed digest"
    )

    inspect_p = sub.add_parser("inspect", help="evaluate a single quantisation point")
    inspect_p.add_argument("--config", default=None)
    inspect_p.add_argument("--format", required=True, help="fp32|fp16|bf16|int8|int4")
    inspect_p.add_argument("--scheme", default="symmetric", choices=("symmetric", "asymmetric"))
    inspect_p.add_argument(
        "--granularity", default="per_channel", choices=("per_tensor", "per_channel")
    )
    inspect_p.add_argument("--json", action="store_true", help="emit JSON instead of text")

    compare_p = sub.add_parser("compare", help="diff two reports")
    compare_p.add_argument("baseline", help="path to the baseline report")
    compare_p.add_argument("candidate", help="path to the candidate report")
    compare_p.add_argument("--json", action="store_true")

    cal_p = sub.add_parser("calibrate", help="print activation calibration ranges")
    cal_p.add_argument("--config", default=None)

    return parser


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run the sweep, write the report, and print a digest."""
    config = load_config(args.config)
    if args.output:
        config = _replace_output(config, args.output)
    result = run_sweep(config)
    path = write_report(config, result)
    if args.csv:
        report = load_report(path)
        csv_path = export_csv(report, args.csv)
        print(f"wrote CSV to {csv_path}")
    if not args.quiet:
        report = load_report(path)
        print(summarise_report(report))
    print(f"wrote report to {path}")
    return 0


def _replace_output(config: SweepConfig, output: str) -> SweepConfig:
    """Return a copy of ``config`` with a different output directory."""
    from dataclasses import replace

    return replace(config, output_dir=output)


def cmd_inspect(args: argparse.Namespace) -> int:
    """Train the reference model and evaluate one quantisation configuration."""
    config = load_config(args.config)
    fmt = NumericFormat.from_string(args.format)
    dataset = build_dataset(config.data)
    params, history = build_and_train(
        config.model,
        dataset.n_features,
        dataset.n_classes,
        dataset.x_train,
        dataset.y_train,
        dataset.x_test,
        dataset.y_test,
    )
    ref_probs, ref_acc = evaluate_model(params, dataset, config.model.activation)
    model = quantize_model(params, fmt, args.scheme, args.granularity, seed=config.model.seed)
    cand_probs, cand_acc = evaluate_model(model.params, dataset, config.model.activation)

    from .metrics import compare_predictions

    behaviour = compare_predictions(
        ref_probs, cand_probs, dataset.y_test, dataset.n_classes
    )
    payload = {
        "format": fmt.value,
        "scheme": args.scheme,
        "granularity": args.granularity,
        "reference_accuracy": ref_acc,
        "candidate_accuracy": cand_acc,
        "training_epochs": len(history.train_loss),
        "weight_error": weight_error_report(params, model.params),
        "layer_rmse": {k: round(v["rmse"], 8) for k, v in model.layer_stats.items()},
        "scale_spread": {k: round(v, 4) for k, v in model.scale_spread().items()},
        "behaviour": behaviour,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"format={fmt.value} scheme={args.scheme} granularity={args.granularity}")
    print(f"reference accuracy {ref_acc:.4f} -> candidate {cand_acc:.4f}")
    print(f"flip rate {behaviour['churn']['flip_rate']:.4f} "
          f"(gained {behaviour['churn']['gained']:.0f}, lost {behaviour['churn']['lost']:.0f})")
    print(f"spearman {behaviour['rank']['spearman_global']:.4f}")
    for name, rmse in payload["layer_rmse"].items():
        print(f"  {name:<20s} rmse={rmse:.6f}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Diff two report files and print the differences."""
    diff = compare_reports(load_report(args.baseline), load_report(args.candidate))
    if args.json:
        print(json.dumps(diff, indent=2))
        return 0
    print(f"shared configurations: {diff['n_shared_configs']}")
    if diff["only_in_baseline"]:
        print(f"only in baseline: {', '.join(diff['only_in_baseline'])}")
    if diff["only_in_candidate"]:
        print(f"only in candidate: {', '.join(diff['only_in_candidate'])}")
    ref_diff = diff["reference_accuracy_diff"]
    if ref_diff is not None:
        print(f"reference accuracy difference: {ref_diff:+.6f}")
    for row in diff["rows"]:
        print(
            f"  {row['label']:<34s} d_acc={row['accuracy_delta_diff']:+.6f} "
            f"d_flip={row['flip_rate_diff']:+.6f}"
        )
    return 0
