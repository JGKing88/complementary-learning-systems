"""The reference walkers, and the two claims the explore line rests on.

`world/walks.py` is what makes `mean_coverage` interpretable: it says which
coverage a given movement statistic earns under this env's step rule. Two
results are load-bearing enough to pin, because a regression in either would
quietly invalidate every conclusion drawn from the table.
"""
from __future__ import annotations

import numpy as np

from hopfield_nav.world.walks import (
    correlated_walk, diffusive_walk, lawnmower_coverage, ring_path,
    simulate_coverage, spiral_walk,
)

SIZE, STEPS, TRIALS = 20, 400, 256


def _rng():
    return np.random.RandomState(0)


def test_the_ring_path_is_a_complete_sweep():
    """Every cell exactly once, in unit steps -- so a walker that tracks it
    perfectly covers the arena in `size * size` steps and nothing else about
    the spiral's score is an artifact of the path."""
    for size in (4, 5, 20):
        path = ring_path(size)
        assert len(path) == size * size
        assert len({tuple(c) for c in path}) == size * size
        hops = np.linalg.norm(np.diff(path, axis=0), axis=1)
        assert hops.max() <= 1.0 + 1e-9


def test_nothing_memoryless_beats_about_point_five_six():
    """The ceiling the whole explore lineage has been sitting at.

    Swept over the two parameters that set a memoryless walker's coverage --
    stride and turn width -- with the diffusive limit included.
    """
    best = 0.0
    for stride in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        best = max(best, diffusive_walk(TRIALS, SIZE, STEPS, stride,
                                        _rng()).mean())
        for turn in (0.1, 0.2, 0.4, 0.8, 1.6, 3.0):
            best = max(best, correlated_walk(TRIALS, SIZE, STEPS, stride,
                                             turn, _rng()).mean())
    assert best < 0.60, f"a memoryless walker reached {best:.3f}"
    assert best > 0.50, f"the ceiling should be near 0.56, got {best:.3f}"


def test_a_stateful_sweep_clears_that_ceiling_without_precision():
    """The prize, and how much execution error it tolerates.

    If this ever needed near-exact execution the strategy would not be a
    realistic target for a policy with a frozen sigma of 0.165.
    """
    exact = spiral_walk(TRIALS, SIZE, STEPS, 1.0, 0.0, _rng()).mean()
    sloppy = spiral_walk(TRIALS, SIZE, STEPS, 1.0, 0.5, _rng()).mean()
    assert exact > 0.90, exact
    # Half a cell of noise per step, three times the policy's own sigma, and
    # it still clears the memoryless ceiling.
    assert sloppy > 0.60, sloppy


def test_a_blocked_step_is_lost():
    """The env clips rather than reflecting, which is why straight-line motion
    scores badly here and why a collapsed policy can sit against a wall
    forever."""
    pos = np.array([[0.0, 5.0]])
    # Due west, into the wall, every step.
    cov = simulate_coverage(pos, SIZE, 50,
                            lambda t, blocked, p: np.array([[-1.0, 0.0]]),
                            _rng())
    assert cov[0] == 1.0 / (SIZE * SIZE)


def test_lawnmower_is_one_fresh_cell_per_step():
    assert lawnmower_coverage(SIZE, STEPS) == 1.0
    assert lawnmower_coverage(SIZE, 99) == 100 / 400
