"""Swept-area coverage: the fraction of the arena the agent could have detected.

THE HEADLINE EXPLORE METRIC from 2026-09-01. See EXPERIMENTS_NAV_P2 §19.

At every step the agent occupies a continuous position and would detect a goal
anywhere within ``goal_radius`` of it -- that is what ``at_goal`` tests, an L2
ball on the *continuous* position. ``swept`` is the fraction of the arena
covered by the UNION of those discs along the whole path. For a uniformly
placed goal it is exactly **P(found by the end of the episode)**.

Why this and not ``mean_coverage``
----------------------------------
``mean_coverage`` counts unique *snapped cells* the agent's position landed on.
That has a detection radius too -- it is just an accidental one, the half-width
of a grid cell -- and it disagrees with the real one wherever the stride is
long: a fast agent sweeps the same corridor but lands on fewer cell CENTRES, so
cell-counting charges it for ground it actually swept.

Measured on billiards at T=200, r=1.0 (2026-09-01):

    speed   cell cov   swept cov
    0.50      0.246      0.391
    1.00      0.383      0.633
    2.00      0.384      0.839
    3.00      0.397      0.881

Cell coverage says speed barely matters and peaks near 1.25; swept area is
monotone increasing. §2.1's "billiard coverage peaks at |a| ~ 1.25 and falls
above it, so the [0.5, 1.0] band costs explore nothing" was measured on the
cell version and does not survive the swept one.

Why the ENDPOINT and not the time integral
------------------------------------------
Expected discovery time is ``E[min(T, T_max)] = sum_t (1 - swept(t))`` -- the
area above the curve, which uses the whole trajectory rather than its end. That
is the more fundamental quantity, but it was measured to rank every policy and
every checkpoint we have identically to the endpoint (§19), because our
policies all share one curve shape and differ only in rate. The endpoint is
simpler and strictly cheaper -- it needs only the final union mask.

The two come apart only if something bends the curve's shape without moving its
end. ``novelty_scale_remaining`` is exactly such a thing: it pays up to 10x
more for late cells than early ones. If that knob changes, re-check against
``E[T]`` before trusting the endpoint.

Why it is not radius-free
-------------------------
It cannot be. "How much did you observe" has no scale-free form, and an
attempt at a radius-invariant behavioural statistic
(``swept / (2r * path_length)``) was measured to vary by 0.25-0.40 across r --
boundary waste scales with ``r * perimeter / area``, and the ratio saturates
once ``2rL`` exceeds the arena. The radius here is the task's own
``goal_radius``, which is the honest choice: a bigger goal really is easier to
find.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class SweptResult(NamedTuple):
    """What one env's trials swept.

    ``per_trial``  (B,) fraction each trial swept on its own.
    ``union``      fraction ANY trial swept -- the swept analogue of
                   ``union_coverage``. Answers "given B independent attempts,
                   what share of the arena was reachable at all", which is a
                   diagnostic of the policy's spread rather than of one
                   episode's search. A policy collapsed onto a single route
                   has union == per_trial; a well-spread one has union much
                   larger.
    """

    per_trial: np.ndarray
    union: float


class SweptArea:
    """Per-trial union of detection discs along a path.

    ``grid`` sub-cells per arena unit sets the raster resolution; 8 gives a
    0.125-unit quantisation, comfortably finer than any goal_radius in use.
    """

    def __init__(self, size: int, radius: float, batch: int, *, grid: int = 8):
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.size = int(size)
        self.radius = float(radius)
        self.grid = int(grid)
        self.B = int(batch)
        self.res = self.size * self.grid
        self._mask = np.zeros((self.B, self.res * self.res), dtype=bool)

        # Disc stencil in raster offsets, built once.
        k = int(np.ceil(self.radius * self.grid))
        oy, ox = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1),
                             indexing="ij")
        keep = (ox ** 2 + oy ** 2) <= (self.radius * self.grid) ** 2
        self._sten = np.stack([ox[keep], oy[keep]], axis=1).astype(np.int64)
        self._prev: np.ndarray | None = None

    def add(self, pos: np.ndarray) -> None:
        """Sweep from the previous position to ``pos`` (B, 2), inclusive.

        The segment is sub-sampled so consecutive samples sit no more than
        0.4*radius apart. Without that a long stride leaves phantom gaps
        between discs and re-introduces the very stride penalty this metric
        exists to remove.
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(self.B, 2)
        if self._prev is None:
            qs = pos[:, None, :]
        else:
            stride = float(np.abs(pos - self._prev).max())
            n_sub = max(1, int(np.ceil(stride / (0.4 * self.radius))))
            ts = np.linspace(0.0, 1.0, n_sub + 1)[1:]
            qs = self._prev[:, None, :] + (pos - self._prev)[:, None, :] * ts[None, :, None]
        self._prev = pos.copy()

        gi = np.rint(qs * self.grid).astype(np.int64)          # (B, n, 2)
        xs = gi[:, :, None, 0] + self._sten[None, None, :, 0]  # (B, n, S)
        ys = gi[:, :, None, 1] + self._sten[None, None, :, 1]
        ok = (xs >= 0) & (xs < self.res) & (ys >= 0) & (ys < self.res)
        b = np.broadcast_to(
            np.arange(self.B)[:, None, None], xs.shape)
        flat = (b[ok] * (self.res * self.res) + xs[ok] * self.res + ys[ok])
        self._mask.reshape(-1)[flat] = True

    def fraction(self) -> np.ndarray:
        """(B,) fraction of the arena swept so far."""
        return self._mask.mean(axis=1)

    def union_fraction(self) -> float:
        """Fraction swept by AT LEAST ONE trial."""
        return float(self._mask.any(axis=0).mean())

    def result(self) -> SweptResult:
        return SweptResult(self.fraction(), self.union_fraction())


def swept_positions(vec) -> np.ndarray:
    """Continuous positions when the env has them, snapped ones otherwise.

    Discrete envs have no sub-cell position, so the snap IS the position and
    the two coincide.
    """
    getter = getattr(vec, "positions_continuous", None)
    if getter is not None:
        return np.asarray(getter(), dtype=np.float64)
    return np.asarray(vec.positions(), dtype=np.float64)


__all__ = ["SweptArea", "SweptResult", "swept_positions"]
