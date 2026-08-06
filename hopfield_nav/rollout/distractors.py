"""Distractor patterns and goal patterns, sampled one way.

`sample_distractors` draws encoded patterns from grid cells *outside* the test
env's footprint, so a recall that lands on one is unambiguously a false memory
rather than a neighbouring cell of the same arena.

It existed as `eval._sample_distractor_goals` plus three inline copies of the
same rejection-sampling loop, in `train.py` and twice in `train_phase_a_only.py`.
The copies had to agree with the eval one, or the claim that training and eval
see the same distractor distribution -- the point of the training-distractor
curriculum -- would have been false with nothing to detect it.

The draw order is load-bearing: two `rng.randint(0, Npos)` calls per candidate,
rejected candidates included. Changing it reshuffles every seeded run.
"""
from __future__ import annotations

import numpy as np

from ..world.scaffold import VectorHash


def sample_distractors(
    vectorhash: VectorHash,
    test_env_offset: tuple[int, int],
    env_size: int,
    n_distractors: int,
    rng: np.random.RandomState,
) -> list[np.ndarray]:
    """Sample encoded patterns from grid positions NOT in the test env's region.

    Returns list of (embed_dim,) numpy arrays.
    """
    Npos = vectorhash.Npos
    cx, cy = test_env_offset
    patterns = []
    while len(patterns) < n_distractors:
        gx = rng.randint(0, Npos)
        gy = rng.randint(0, Npos)
        if cx <= gx < cx + env_size and cy <= gy < cy + env_size:
            continue
        patterns.append(vectorhash.encoded_Phi[gx, gy].copy())
    return patterns


def goal_encoding(
    vectorhash: VectorHash,
    env_offset: tuple[int, int],
    goal: tuple[int, int],
) -> np.ndarray:
    """Look up the encoded pattern for this env's goal."""
    gx = min(max(goal[0] + env_offset[0], 0), vectorhash.Npos - 1)
    gy = min(max(goal[1] + env_offset[1], 0), vectorhash.Npos - 1)
    return vectorhash.encoded_Phi[gx, gy]

__all__ = ["goal_encoding", "sample_distractors"]
