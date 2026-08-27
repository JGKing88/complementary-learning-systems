"""Evaluating an encoder through its Hopfield readout.

Spec: ``docs/ENCODER_HOPFIELD_EVAL.md``. Reporting layer:
``docs/ENCODER_HOPFIELD_EVAL_VIZ.md``.

Four tests, in the order the errors compound:

    A  attractor / basin      is a stored goal a fixed point, and over what
                              real-space disc does a cue relax to it
    B  q accuracy, grid       how far off q points, per cell
    C  q accuracy, continuous the same, at positions that reach the encoder
                              only through the env's round() snap
    D  flow                   whether the field the first three describe
                              actually carries trajectories to the goal

Lives in ``analysis/`` because it needs both ``encoder_training`` and
``hopfield_nav.world``, and ``encoder_training`` may not import upward.
"""
from __future__ import annotations

__all__ = ["attractor", "controls", "encode", "flow", "harness", "qfield",
           "stats"]
