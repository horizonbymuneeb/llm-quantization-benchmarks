"""Low-precision number formats used to simulate compressed model weights.

The benchmark needs to reason about *storage* formats independently of the
*quantisation scheme* applied inside them.  A BF16 tensor and an INT8 tensor are
different kinds of object: the first still has a floating exponent, the second
is an integer grid plus a scale.  Keeping both concepts in one module lets the
rest of the code say "give me this weight matrix at BF16 precision" without
caring which mechanism produced it.

Everything here is implemented with numpy so the whole benchmark runs on a CPU
with no accelerator and no vendor runtime.
"""

from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass

import numpy as np


class NumericFormat(enum.Enum):
    """Storage formats the harness can simulate."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def bits(self) -> int:
        """Nominal bit width of one stored value."""
        return _FORMAT_BITS[self]

    @property
    def is_integer(self) -> bool:
        """True for integer grids that need an explicit scale."""
        return self in (NumericFormat.INT8, NumericFormat.INT4)

    @property
    def is_float(self) -> bool:
        """True for formats that keep a floating exponent."""
        return not self.is_integer

    @classmethod
    def from_string(cls, name: str) -> "NumericFormat":
        """Parse a config string such as ``"int4"`` or ``"BF16"``."""
        key = name.strip().lower()
        for fmt in cls:
            if fmt.value == key:
                return fmt
        raise ValueError(
            f"unknown numeric format {name!r}; expected one of "
            f"{sorted(f.value for f in cls)}"
        )
