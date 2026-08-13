"""Memoryless walkers on this project's arena, and the coverage they earn.

Pure numpy, no repo imports: this is a leaf, so both the reference CLI
(`coverage_reference`) and the policy diagnostic (`policy_motion`) can use it
without either importing the other -- which they must not, being entry points.

The step semantics are copied from `world/vec_env.ContinuousVecEnv.step_batch`
and are the whole reason these numbers are not textbook random-walk numbers:
position is continuous, the action is *added* to it, and the result is
**clipped** to the arena rather than reflected. A step into a wall is absorbed,
so the agent stays put and loses the step. Coverage counts distinct *snapped*
cells, exactly as `batched_exploration_trials` does.
"""
from __future__ import annotations

import numpy as np


def simulate_coverage(pos_f: np.ndarray, size: int, steps: int,
                      direction_fn, rng: np.random.RandomState) -> np.ndarray:
    """Run B walkers for `steps` and return each one's covered-cell fraction.

    `direction_fn(t, blocked, pos)` returns the (B, 2) displacement for step t.
    `blocked` says which walkers had their previous step absorbed by the clip,
    and `pos` is the current continuous position -- passed rather than closed
    over, because this function owns the position and a closure over the
    caller's copy silently reads the *initial* one forever. That is not a
    hypothetical: it made the closed-loop spiral below score 0.24 instead of
    1.00, which reads as "the strategy does not work" rather than "the
    simulator did not tell it where it was".
    """
    B = pos_f.shape[0]
    visited = np.zeros((B, size, size), dtype=bool)
    idx = np.arange(B)
    snapped = np.clip(np.rint(pos_f).astype(int), 0, size - 1)
    visited[idx, snapped[:, 0], snapped[:, 1]] = True
    blocked = np.zeros(B, dtype=bool)
    for _t in range(steps):
        before = pos_f.copy()
        pos_f = np.clip(pos_f + direction_fn(_t, blocked, pos_f), 0.0,
                        float(size - 1))
        blocked = np.linalg.norm(pos_f - before, axis=1) < 1e-9
        snapped = np.clip(np.rint(pos_f).astype(int), 0, size - 1)
        visited[idx, snapped[:, 0], snapped[:, 1]] = True
    return visited.reshape(B, -1).sum(1) / float(size * size)


def random_starts(B: int, size: int, rng: np.random.RandomState) -> np.ndarray:
    """Uniform integer cells, matching `random_start` / `reset_all`."""
    return rng.randint(0, size, size=(B, 2)).astype(np.float64)


def unit_vectors(theta: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def correlated_walk(B: int, size: int, steps: int, stride: float,
                    turn_sigma: float, rng: np.random.RandomState
                    ) -> np.ndarray:
    """Heading does a wrapped random walk of width `turn_sigma` per step.

    `turn_sigma -> 0` is ballistic and `-> large` is diffusive, so this one
    family spans the whole memoryless range. It is also the family
    `persistence_bonus` moves a policy along, that bonus being exactly the
    cosine of successive headings.

    A blocked step redraws the heading. Without it the walker pushes into the
    wall forever, which is the dead fixed point a collapsed policy falls into
    and not a movement statistic anyone would choose.
    """
    pos = random_starts(B, size, rng)
    theta = rng.uniform(0, 2 * np.pi, B)

    def fn(_t, blocked, _pos):
        nonlocal theta
        theta = theta + rng.normal(0.0, turn_sigma, B)
        if blocked.any():
            theta[blocked] = rng.uniform(0, 2 * np.pi, int(blocked.sum()))
        return unit_vectors(theta) * stride

    return simulate_coverage(pos, size, steps, fn, rng)


def diffusive_walk(B: int, size: int, steps: int, stride: float,
                   rng: np.random.RandomState) -> np.ndarray:
    """Direction redrawn uniformly every step."""
    pos = random_starts(B, size, rng)

    def fn(_t, _blocked, _pos):
        return unit_vectors(rng.uniform(0, 2 * np.pi, B)) * stride

    return simulate_coverage(pos, size, steps, fn, rng)


def bounce_walk(B: int, size: int, steps: int, stride: float,
                rng: np.random.RandomState) -> np.ndarray:
    """Straight until the step is clipped, then a new direction."""
    pos = random_starts(B, size, rng)
    theta = rng.uniform(0, 2 * np.pi, B)

    def fn(_t, blocked, _pos):
        nonlocal theta
        if blocked.any():
            theta[blocked] = rng.uniform(0, 2 * np.pi, int(blocked.sum()))
        return unit_vectors(theta) * stride

    return simulate_coverage(pos, size, steps, fn, rng)


def wall_aware_walk(B: int, size: int, steps: int, stride: float,
                    turn_sigma: float, rng: np.random.RandomState,
                    margin: float = 1.5) -> np.ndarray:
    """A correlated walk that turns *before* it hits the wall.

    Still memoryless -- it knows nothing about where it has been -- but it uses
    the one thing the foveal cone reports directly: how close the wall ahead
    is. When the next step would land within `margin` of the boundary the
    heading is redrawn to point back inside.

    This is the control that `correlated_walk` is not. A policy can beat a
    plain correlated walk purely by not wasting steps on the wall, which is
    reactive and needs no memory at all; the gap that survives *this* baseline
    is the part that does. Reporting only the plain-walk excess conflates the
    two, and they call for completely different next steps.
    """
    pos = random_starts(B, size, rng)
    theta = rng.uniform(0, 2 * np.pi, B)
    lo, hi = margin, size - 1 - margin

    def fn(_t, blocked, p):
        nonlocal theta
        step = unit_vectors(theta) * stride
        nxt = p + step
        bad = ((nxt[:, 0] < lo) | (nxt[:, 0] > hi)
               | (nxt[:, 1] < lo) | (nxt[:, 1] > hi) | blocked)
        if bad.any():
            # Point back toward the middle, plus a wide jitter so this is a
            # turn rather than a deterministic bounce off the centre.
            to_mid = (size - 1) / 2.0 - p[bad]
            theta[bad] = (np.arctan2(to_mid[:, 1], to_mid[:, 0])
                          + rng.normal(0.0, 0.8, int(bad.sum())))
        theta = theta + rng.normal(0.0, turn_sigma, B)
        return unit_vectors(theta) * stride

    return simulate_coverage(pos, size, steps, fn, rng)


def ring_path(size: int) -> np.ndarray:
    """Cells of an inward spiral: the perimeter, then the next ring, inward.

    Every cell of the arena appears exactly once, so the path is a complete
    sweep of length `size * size`. It is the cheapest *stateful* strategy this
    policy could plausibly run: it needs a ring index and the distance to the
    wall ahead, and the foveal cone reports the latter directly.
    """
    cells: list[tuple[int, int]] = []
    for k in range((size + 1) // 2):
        lo, hi = k, size - 1 - k
        if lo == hi:
            cells.append((lo, lo))
            continue
        cells += [(lo, y) for y in range(lo, hi)]
        cells += [(x, hi) for x in range(lo, hi)]
        cells += [(hi, y) for y in range(hi, lo, -1)]
        cells += [(x, lo) for x in range(hi, lo, -1)]
    return np.array(cells, dtype=np.float64)


def spiral_walk(B: int, size: int, steps: int, stride: float,
                exec_sigma: float, rng: np.random.RandomState) -> np.ndarray:
    """The inward spiral, executed imperfectly.

    Closed-loop: each step aims from where the walker actually *is* toward the
    next cell of the plan, at `stride`, plus per-component noise of width
    `exec_sigma`. So this answers the question a policy designer needs -- how
    accurately would the strategy have to be executed for it to beat the
    memoryless ceiling -- rather than merely restating that a perfect sweep
    covers everything.

    Walkers start at different phases of the path, so they do not all trace one
    trajectory.
    """
    path = ring_path(size)
    n = len(path)
    phase = rng.randint(0, n, size=B)
    pos = path[phase].copy()
    cursor = phase.copy()

    def fn(_t, _blocked, pos):
        nonlocal cursor
        cursor = (cursor + 1) % n
        want = path[cursor] - pos
        norm = np.linalg.norm(want, axis=1, keepdims=True).clip(1e-9)
        return want / norm * stride + rng.normal(0.0, exec_sigma, (B, 2))

    return simulate_coverage(pos, size, steps, fn, rng)


def lawnmower_coverage(size: int, steps: int) -> float:
    """The ideal boustrophedon sweep: one fresh cell per step.

    An upper bound, and not reachable without knowing where you have been --
    which is exactly what makes it the right thing to compare against.
    """
    return min(steps + 1, size * size) / float(size * size)


__all__ = [
    "bounce_walk", "correlated_walk", "diffusive_walk", "lawnmower_coverage",
    "random_starts", "ring_path", "simulate_coverage", "spiral_walk",
    "unit_vectors", "wall_aware_walk",
]
