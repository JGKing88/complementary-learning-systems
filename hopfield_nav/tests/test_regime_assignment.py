"""`--regime_assignment` decides WHICH envs are exploit, not how many.

`index`, the default, is the historical behaviour: the first `n_pre` envs in
order. That means at a fixed `empty_frac` an env keeps its regime for the whole
run, which lets a policy gate on env identity -- learnable from the env's own
wall codebook -- instead of on the recall signal. The shortcut does not transfer
to a held-out env, so it is a real confound for any interleaved schedule.

`shuffle` re-draws the assignment every update. These pin the two properties
that make it a fix rather than a change: the *count* is preserved exactly (so
the schedule still means what it says), and the assignment actually moves
between updates.
"""
from __future__ import annotations

import numpy as np
import pytest

from hopfield_nav.config import TrainConfig


def _assign(mode, n_envs, n_pre, seed=0):
    """The three lines of run_navigate that choose the regime, in isolation."""
    np.random.seed(seed)
    if mode == "shuffle":
        is_pre = np.zeros(n_envs, dtype=bool)
        is_pre[np.random.permutation(n_envs)[:n_pre]] = True
    else:
        is_pre = np.arange(n_envs) < n_pre
    return is_pre


@pytest.mark.parametrize("mode", ["index", "shuffle"])
@pytest.mark.parametrize("n_pre", [0, 1, 40, 79, 80])
def test_count_is_exactly_preserved(mode, n_pre):
    """Whatever the schedule asked for, that many envs are exploit."""
    assert _assign(mode, 80, n_pre).sum() == n_pre


def test_index_is_the_historical_prefix():
    is_pre = _assign("index", 80, 40)
    assert is_pre[:40].all() and not is_pre[40:].any()


def test_index_never_moves_between_updates():
    """The confound, stated as a property: same envs, every update."""
    a = _assign("index", 80, 40, seed=1)
    b = _assign("index", 80, 40, seed=2)
    assert np.array_equal(a, b)


def test_shuffle_moves_between_updates():
    """Consecutive draws differ, so env identity carries no regime information."""
    np.random.seed(0)
    draws = []
    for _ in range(8):
        is_pre = np.zeros(80, dtype=bool)
        is_pre[np.random.permutation(80)[:40]] = True
        draws.append(is_pre)
    assert any(not np.array_equal(draws[0], d) for d in draws[1:])
    # And over enough updates every env has been in both regimes.
    stacked = np.stack(draws)
    assert stacked.any(axis=0).all(), "some env was never exploit"
    assert (~stacked).any(axis=0).all(), "some env was never explore"


def test_default_is_index_so_existing_runs_are_unchanged():
    assert TrainConfig().regime_assignment == "index"


def test_reward_split_follows_the_flags_not_a_slice():
    """The pre/emp logging split must key off the per-rollout flag.

    The old code sliced `rollouts[:n_pre * n_worlds]`, which is world-major --
    so with more than one world it mixed regimes even under `index`.
    """
    n_worlds, n_envs, n_pre = 3, 4, 2
    is_pre = _assign("index", n_envs, n_pre)
    flags, rollouts = [], []
    for w in range(n_worlds):
        for i in range(n_envs):
            flags.append(bool(is_pre[i]))
            rollouts.append((w, i))

    by_flag = [r for r, pre in zip(rollouts, flags) if pre]
    by_slice = rollouts[:n_pre * n_worlds]

    assert all(i < n_pre for _, i in by_flag)
    assert len(by_flag) == n_pre * n_worlds
    assert by_flag != by_slice, "the slice was wrong for num_worlds > 1"
