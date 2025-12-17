"""Source root for the quantbench package.

The project is laid out as ``src/quantbench`` so that ``import src.quantbench``
works from a plain checkout without installing anything, which is what the
tests and the Makefile rely on.  Installing via ``pip install -e .`` instead
exposes the package as plain ``quantbench``; both paths import the same modules.
"""

from __future__ import annotations

from .quantbench import __version__

__all__ = ["__version__", "quantbench"]
