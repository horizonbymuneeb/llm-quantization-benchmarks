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


def cast_bf16(values: np.ndarray) -> np.ndarray:
    """Round an array to BF16 precision and return it as float32.

    BF16 keeps the FP32 exponent and truncates the mantissa from 23 to 7 bits.
    Naive truncation biases every value toward zero, which compounds over many
    layers, so the low 16 bits are rounded to nearest with ties to even before
    being masked off.
    """
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32).astype(np.uint64)
    # A tie is exactly 0x8000 in the discarded half: bump only when the
    # retained part is odd, which is the round-half-to-even rule.
    low = bits & np.uint64(0xFFFF)
    retained = bits & np.uint64(0xFFFF0000)
    tie = low == np.uint64(0x8000)
    odd = (retained >> np.uint64(16)) & np.uint64(1)
    bump = (low > np.uint64(0x8000)) | (tie & (odd == np.uint64(1)))
    rounded = retained + np.where(bump, np.uint64(0x10000), np.uint64(0))
    rounded = np.minimum(rounded, np.uint64(0xFFFF0000))
    return rounded.astype(np.uint32).view(np.float32).astype(np.float32)


def cast_fp16(values: np.ndarray) -> np.ndarray:
    """Round an array through IEEE-754 binary16 and back to float32.

    Values beyond binary16's range (about 65504) saturate to infinity.  That is
    correct IEEE behaviour and the caller may well be measuring it, so the cast
    overflow warning is suppressed here rather than escalated.
    """
    arr = np.asarray(values, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return arr.astype(np.float16).astype(np.float32)
