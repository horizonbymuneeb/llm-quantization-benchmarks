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
