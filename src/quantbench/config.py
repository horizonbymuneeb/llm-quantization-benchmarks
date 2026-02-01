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
