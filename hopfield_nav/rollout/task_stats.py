"""Per-trajectory bookkeeping for the task-faithful regime.

`docs/TASK_FAITHFUL_PLAN.md` §4. A task rollout has two phases per trajectory
-- search until the first goal touch (the oracle writes the goal then), and
navigation from that touch on -- and a trajectory may also *arrive* with the
goal already stored (a revisit, `visits > 1`). The tracker records the step
of the first touch, how much of the arena had been visited by then, and how
the touches after it are spaced; `merge` pools rollouts into the scalars the
trainer logs and the eval prints.

Touches are counted at decision time (`at_goal_mask` at the top of a step),
which is also when the reward is paid and the teleport happens, so a touch
at step t means the agent stood on the goal when step t began.
"""
from __future__ import annotations

import numpy as np

_NAN = float("nan")


class TaskTracker:
    """Observe one rollout step at a time; read the arrays out at the end."""

    def __init__(self, started_stored: np.ndarray):
        B = int(started_stored.shape[0])
        self.started_stored = started_stored.astype(bool).copy()
        self.first_step = np.full(B, -1, dtype=np.int64)
        self.last_step = np.full(B, -1, dtype=np.int64)
        self.n_touch = np.zeros(B, dtype=np.int64)
        self.cov_first = np.full(B, _NAN, dtype=np.float32)

    def observe(self, t: int, at_goal: np.ndarray,
                visited_count: np.ndarray | None) -> None:
        """`at_goal` (B,) bool at decision time of step `t`; `visited_count`
        (B,) cells visited so far, or None when the rollout keeps no
        visited-cell buffer (novelty off)."""
        new_first = at_goal & (self.first_step < 0)
        if new_first.any():
            self.first_step[new_first] = t
            if visited_count is not None:
                self.cov_first[new_first] = visited_count[new_first]
        self.last_step[at_goal] = t
        self.n_touch += at_goal.astype(np.int64)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "started_stored": self.started_stored.copy(),
            "first_step": self.first_step.copy(),
            "last_step": self.last_step.copy(),
            "n_touch": self.n_touch.copy(),
            "cov_first": self.cov_first.copy(),
        }


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else _NAN


def _phase_stats(a: dict[str, np.ndarray], rows: np.ndarray,
                 n_cells: float | None) -> dict[str, float]:
    found = rows & (a["first_step"] >= 0)
    n_rows = int(rows.sum())
    out = {
        "n": float(n_rows),
        "found_frac": (float(found.sum()) / n_rows) if n_rows else _NAN,
        "steps_first": _mean(a["first_step"][found].astype(np.float64)),
    }
    cov = a["cov_first"][found]
    cov = cov[~np.isnan(cov)]
    out["cov_first"] = (_mean(cov) / n_cells) if (n_cells and cov.size) else _NAN
    post = (a["n_touch"][found] - 1).astype(np.float64)
    out["reaches_post"] = _mean(post)
    has_post = post > 0
    if has_post.any():
        span = (a["last_step"][found] - a["first_step"][found]).astype(np.float64)
        out["steps_per_reach"] = float((span[has_post] / post[has_post]).mean())
    else:
        out["steps_per_reach"] = _NAN
    return out


def merge(records: list[dict[str, np.ndarray]],
          n_cells: float | None = None) -> dict[str, float]:
    """Pool rollouts (each a `TaskTracker.arrays()` dict) into scalars.

    Visit-1 trajectories (arrived without the goal) give `found_frac`,
    `steps_first`, `cov_first` (fraction of `n_cells`, NaN when unknown),
    `reaches_post`, `steps_per_reach`; trajectories that arrived with the goal
    stored give the same under a `revisit_` prefix (`revisit_steps_first` is
    the revisit's steps to the goal from a fresh state -- the continual
    protocol's number). Means are over trajectories, so every trajectory
    weighs the same whichever rollout it came from.
    """
    if not records:
        return {}
    a = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
    out: dict[str, float] = {}
    search = ~a["started_stored"]
    for k, v in _phase_stats(a, search, n_cells).items():
        out[k] = v
    if a["started_stored"].any():
        for k, v in _phase_stats(a, a["started_stored"], n_cells).items():
            if k != "cov_first":
                out[f"revisit_{k}"] = v
    return out


def format_line(stats: dict[str, float]) -> str:
    """One compact fragment for the trainer's per-update print."""
    if not stats:
        return ""
    s = (f"found={stats['found_frac']:.2f} first={stats['steps_first']:.1f} "
         f"cov={stats['cov_first']:.2f} post={stats['reaches_post']:.1f}"
         f"@{stats['steps_per_reach']:.1f}")
    if "revisit_found_frac" in stats:
        s += (f" rv={stats['revisit_found_frac']:.2f}"
              f"@{stats['revisit_steps_first']:.1f}"
              f"/{stats['revisit_steps_per_reach']:.1f}")
    return s


__all__ = ["TaskTracker", "merge", "format_line"]
