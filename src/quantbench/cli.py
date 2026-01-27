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
