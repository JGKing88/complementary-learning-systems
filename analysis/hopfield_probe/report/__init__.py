"""Report layer: static, self-contained HTML pages from result JSON.

Spec: ``docs/ENCODER_HOPFIELD_EVAL_VIZ.md``.

Imports nothing from the test modules. The tests run headless on a compute
node; the pages are built afterwards, from JSON, and recompute nothing -- so
restyling a figure never costs a recall.
"""
from __future__ import annotations

__all__ = ["build", "figures", "page", "theme"]
