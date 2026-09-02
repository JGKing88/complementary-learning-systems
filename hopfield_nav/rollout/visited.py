"""Per-episode visitation probe: "have I been to the cell over there?"

Used for two different things, which is why it lives on its own:

  * the AUXILIARY HEAD's target (`aux_visited_weight`, P2 doc §24.2 lever B),
    which shapes the trunk and is never seen by the policy; and
  * the DIAGNOSTIC INPUT CHANNEL (`input_visited`, §27.5), which hands the same
    vector straight to the policy.

Both need identical arithmetic or the diagnostic would not be testing what the
aux head was trained on, so there is one implementation.

The probe reads at the DECISION position, before the step, and marks the
current cell visited afterwards. That ordering matters: the vector has to
describe what the agent knew when it CHOSE, not what it learns by moving.
"""
from __future__ import annotations

import numpy as np

N_DIR = 8


class VisitedProbe:
    """Tracks a per-trial visited set and reads 8 compass cells around it."""

    def __init__(self, size: int, radius: float, batch: int):
        self.size = int(size)
        self.B = int(batch)
        ang = np.arange(N_DIR) * (np.pi / 4.0)
        self.offsets = np.stack(
            [np.rint(float(radius) * np.cos(ang)),
             np.rint(float(radius) * np.sin(ang))], axis=1).astype(int)
        self.seen = np.zeros((self.B, self.size, self.size), dtype=bool)

    def read(self, positions: np.ndarray) -> np.ndarray:
        """(B, 8) float 0/1 for the 8 probed cells, then mark where we stand.

        ``positions`` is the SNAPPED (B, 2) integer position.
        """
        p = np.asarray(positions, dtype=int).reshape(self.B, 2)
        q = np.clip(p[:, None, :] + self.offsets[None, :, :], 0, self.size - 1)
        out = self.seen[np.arange(self.B)[:, None], q[:, :, 0], q[:, :, 1]]
        self.seen[np.arange(self.B), p[:, 0], p[:, 1]] = True
        return out.astype(np.float32)


def probe_for(cfg, batch: int) -> "VisitedProbe | None":
    """A probe when either consumer is enabled, else None."""
    a = cfg.agent
    if (getattr(a, "aux_visited_weight", 0.0) > 0
            or getattr(a, "input_visited", False)):
        return VisitedProbe(cfg.env.size,
                            getattr(a, "aux_visited_radius", 3.0), batch)
    return None


def abs_position_channel(vec, size: int) -> np.ndarray:
    """(B, 2) absolute position, normalised to [-1, 1]. DIAGNOSTIC ONLY.

    An oracle at test time (§29.4). Uses the CONTINUOUS position where the env
    has one, so it is the same quantity `at_goal` and the swept metric use.
    """
    getter = getattr(vec, "positions_continuous", None)
    p = np.asarray(getter() if getter is not None else vec.positions(),
                   dtype=np.float32)
    return (2.0 * p / max(size - 1, 1) - 1.0).astype(np.float32)


__all__ = ["VisitedProbe", "probe_for", "N_DIR", "abs_position_channel"]
