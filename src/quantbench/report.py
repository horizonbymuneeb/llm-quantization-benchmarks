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
