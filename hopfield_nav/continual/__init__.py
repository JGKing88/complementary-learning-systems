"""Continual-learning methods that plug into the RNN baseline's BC update.

Not to be confused with `analysis/continual/`, which is the *figure* pipeline
for the same experiment. This package holds the algorithms; that one draws what
they produced.

See `docs/CONTINUAL_CONTROLS_PLAN.md` for what is in the suite and why. Every
method here is a modification of exactly two things -- what loss the update
adds, and what happens at a block boundary -- which is what `ContinualMethod`
in `base.py` formalises.
"""
from .base import CONTINUAL_METHODS, ContinualMethod, NoMethod, build_method

__all__ = ["CONTINUAL_METHODS", "ContinualMethod", "NoMethod", "build_method"]
