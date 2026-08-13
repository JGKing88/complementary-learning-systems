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

    `direction_fn(t, blocked)` returns the (B, 2) displacement for step t;
    `blocked` says which walkers had their previous step absorbed by the clip.
    """
    B = pos_f.shape[0]
    visited = np.zeros((B, size, size), dtype=bool)
    idx = np.arange(B)
    snapped = np.clip(np.rint(pos_f).astype(int), 0, size - 1)
    visited[idx, snapped[:, 0], snapped[:, 1]] = True
    blocked = np.zeros(B, dtype=bool)
    for _t in range(steps):
        before = pos_f.copy()
        pos_f = np.clip(pos_f + direction_fn(_t, blocked), 0.0,
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

    def fn(_t, blocked):
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

    def fn(_t, _blocked):
        return unit_vectors(rng.uniform(0, 2 * np.pi, B)) * stride

    return simulate_coverage(pos, size, steps, fn, rng)


def bounce_walk(B: int, size: int, steps: int, stride: float,
                rng: np.random.RandomState) -> np.ndarray:
    """Straight until the step is clipped, then a new direction."""
    pos = random_starts(B, size, rng)
    theta = rng.uniform(0, 2 * np.pi, B)

    def fn(_t, blocked):
        nonlocal theta
        if blocked.any():
            theta[blocked] = rng.uniform(0, 2 * np.pi, int(blocked.sum()))
        return unit_vectors(theta) * stride

    return simulate_coverage(pos, size, steps, fn, rng)


def lawnmower_coverage(size: int, steps: int) -> float:
    """The ideal boustrophedon sweep: one fresh cell per step.

    An upper bound, and not reachable without knowing where you have been --
    which is exactly what makes it the right thing to compare against.
    """
    return min(steps + 1, size * size) / float(size * size)


__all__ = [
    "bounce_walk", "correlated_walk", "diffusive_walk", "lawnmower_coverage",
    "random_starts", "simulate_coverage", "unit_vectors",
]
