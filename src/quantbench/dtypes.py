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


_FORMAT_BITS: dict[NumericFormat, int] = {
    NumericFormat.FP32: 32,
    NumericFormat.FP16: 16,
    NumericFormat.BF16: 16,
    NumericFormat.INT8: 8,
    NumericFormat.INT4: 4,
}


@dataclass(frozen=True)
class FormatSpec:
    """Static description of a numeric format: bit layout and range."""

    fmt: NumericFormat
    exponent_bits: int
    mantissa_bits: int
    signed: bool

    @property
    def total_bits(self) -> int:
        """Bits actually stored per value, including the sign bit."""
        return 1 + self.exponent_bits + self.mantissa_bits if self.is_float else self.fmt.bits

    @property
    def is_float(self) -> bool:
        """Mirror of :attr:`NumericFormat.is_float` for convenience."""
        return self.fmt.is_float

    @property
    def integer_range(self) -> tuple[int, int]:
        """Inclusive ``(low, high)`` integer range for integer formats."""
        if not self.fmt.is_integer:
            raise TypeError(f"{self.fmt.value} is not an integer format")
        if self.signed:
            half = self.fmt.bits - 1
            return (-(1 << half), (1 << half) - 1)
        return (0, (1 << self.fmt.bits) - 1)


FORMAT_SPECS: dict[NumericFormat, FormatSpec] = {
    NumericFormat.FP32: FormatSpec(NumericFormat.FP32, 8, 23, True),
    NumericFormat.FP16: FormatSpec(NumericFormat.FP16, 5, 10, True),
    NumericFormat.BF16: FormatSpec(NumericFormat.BF16, 8, 7, True),
    NumericFormat.INT8: FormatSpec(NumericFormat.INT8, 0, 7, True),
    NumericFormat.INT4: FormatSpec(NumericFormat.INT4, 0, 3, True),
}


def format_spec(fmt: NumericFormat) -> FormatSpec:
    """Return the static :class:`FormatSpec` for a format enum member."""
    return FORMAT_SPECS[fmt]


def _round_to_even_shift(mantissa_dropped: int) -> int:
    """Round-to-nearest-even for a dropped bit pattern of ``mantissa_dropped``."""
    if mantissa_dropped == 0:
        return 0
    return 1 if mantissa_dropped > 1 else 0
