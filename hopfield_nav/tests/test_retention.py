"""Retention arithmetic: the primary/revisit split and the dead-env exclusion.

Both exist because pooling gets the answer wrong in a specific, measured way.
On d0_base u725 the pooled revisit rate is 0.833 and the live-env revisit rate
is 1.0000 -- two envs that were never learned in the first place drag a
zero-forgetting result down by seventeen points. And a primary episode is not
comparable to a revisit at all: the primary has nothing in memory and must be
explored, the revisit is a pure exploit episode.

The trace format is `[iteration, block, {env_idx_str: record}]`, and the
protocol's rule is that only env == block may store.
"""
from __future__ import annotations

import numpy as np

from analysis.continual import retention


def hist(rows, n, ipb=2, name="t"):
    return ({"trace": rows, "metadata": {"n_envs": n, "iters_per_block": ipb,
                                         "run_name": name}},
            {"n_envs": n, "iters_per_block": ipb, "run_name": name})


def rec(reached, steps=None):
    return {"reached": int(reached), "steps_to_goal": steps}


class TestSplit:

    def test_primary_is_the_diagonal_and_revisit_is_left_of_it(self):
        rows = [
            [0, 0, {"0": rec(1, 10)}],
            [1, 1, {"0": rec(1, 4), "1": rec(0)}],
        ]
        h, md = hist(rows, 2)
        r, _ = retention.summarise(h, md, thresh=0.4)
        # env0 own block: 1.0 -> live. env1 own block: 0.0 -> dead.
        assert r["live"] == [0] and r["dead"] == [1]
        # the only live revisit is env0 in block1
        assert r["revisit_success"] == 1.0
        assert r["revisit_episodes"] == 1
        assert r["primary_episodes"] == 1

    def test_dead_env_excluded_from_retention_but_kept_in_the_all_figure(self):
        """The seventeen-point case, in miniature."""
        rows = [
            [0, 0, {"0": rec(1, 5)}],
            [1, 1, {"0": rec(1, 5), "1": rec(0)}],
            [2, 2, {"0": rec(1, 5), "1": rec(0), "2": rec(1, 5)}],
        ]
        h, md = hist(rows, 3)
        r, _ = retention.summarise(h, md)
        assert r["dead"] == [1]
        assert r["revisit_success"] == 1.0          # env0 only
        assert r["revisit_success_all"] < 1.0       # env1's zeros included
        assert np.isclose(r["revisit_success_all"], 2 / 3)

    def test_steps_use_successful_episodes_only(self):
        rows = [
            [0, 0, {"0": rec(1, 10)}],
            [1, 1, {"0": rec(0, None), "1": rec(1, 8)}],
            [2, 2, {"0": rec(1, 6), "1": rec(1, 8), "2": rec(1, 8)}],
        ]
        h, md = hist(rows, 3)
        r, _ = retention.summarise(h, md, thresh=0.4)
        # env0's revisits: a failure (no steps) and a success at 6
        assert np.isclose(r["revisit_steps"], (6 + 8) / 2)


class TestMatrix:

    def test_unrun_pairs_are_nan_not_zero(self):
        """Env 1 does not exist during block 0. Zero would read as a failure."""
        rows = [[0, 0, {"0": rec(1, 3)}], [1, 1, {"0": rec(1, 3), "1": rec(1, 3)}]]
        h, md = hist(rows, 2)
        _, m = retention.summarise(h, md)
        assert np.isnan(m[0, 1])
        assert m[1, 0] == 1.0

    def test_retention_delta_is_final_minus_own(self):
        rows = [
            [0, 0, {"0": rec(0, None)}],
            [0, 0, {"0": rec(1, 3)}],           # own block: 0.5
            [1, 1, {"0": rec(1, 3), "1": rec(1, 3)}],
            [1, 1, {"0": rec(1, 3), "1": rec(1, 3)}],   # final block: 1.0
        ]
        h, md = hist(rows, 2)
        r, _ = retention.summarise(h, md, thresh=0.4)
        assert np.isclose(r["retention_delta"][0], +0.5)
        assert r["worst_retention_delta"] >= 0.0

    def test_forgetting_shows_as_a_negative_delta(self):
        rows = [
            [0, 0, {"0": rec(1, 3)}],
            [1, 1, {"0": rec(0, None), "1": rec(1, 3)}],
        ]
        h, md = hist(rows, 2)
        r, _ = retention.summarise(h, md, thresh=0.4)
        assert r["retention_delta"][0] == -1.0
        assert r["worst_retention_delta"] == -1.0


class TestLiveEnvs:

    def test_threshold_is_on_the_envs_own_block(self):
        rows = [
            [0, 0, {"0": rec(1, 3)}],
            [0, 0, {"0": rec(0)}],              # own-block rate 0.5
            [1, 1, {"0": rec(1, 3), "1": rec(1, 3)}],
        ]
        h, md = hist(rows, 2)
        reach, _ = retention.cells(h, 2)
        assert retention.live_envs(reach, 2, 0.4) == [0, 1]
        assert retention.live_envs(reach, 2, 0.6) == [1]

    def test_episode_count_covers_every_record(self):
        rows = [
            [0, 0, {"0": rec(1, 3)}],
            [1, 1, {"0": rec(1, 3), "1": rec(1, 3)}],
            [2, 2, {"0": rec(1, 3), "1": rec(1, 3), "2": rec(1, 3)}],
        ]
        h, md = hist(rows, 3)
        r, _ = retention.summarise(h, md)
        assert r["episodes"] == 6
