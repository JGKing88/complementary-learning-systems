"""Per-update, per-regime rollout diagnostics.

The point of these is stated in `docs/DUAL_TRAINING.md` §7 and D2: the corner
trap has only ever been diagnosed after a run finished, and the prediction on
record -- that `chase_q` rises BEFORE `edge_frac` -- has never been testable
because nothing logged either per update.

The tests below pin the three things most able to go quietly wrong:

  * `cos_aq` must EXCLUDE epsilon steps, or a policy that never follows the
    recall reads as partially following it purely from random actions;
  * `pin_frac` must be a PER-ROW statistic, because a rollout with half its
    rows pinned and one with every row half-pinned have identical pooled
    `clip_frac` and only the first is §18.7's basin;
  * `merge` must weight by live steps, or a rollout that died at step 3 counts
    as much as one that ran 200.
"""
from __future__ import annotations

import numpy as np
import pytest

from hopfield_nav.rollout.diagnostics import (
    PIN_CLIP, PIN_SPEED, RegimeDiagnostics, merge, on_perimeter,
)


def _step(d, *, q, action, realized, at_edge=None, alive=None, pol=None):
    n = len(q)
    d.observe(
        q=np.asarray(q, dtype=float),
        action=np.asarray(action, dtype=float),
        realized=np.asarray(realized, dtype=float),
        at_edge=np.zeros(n, bool) if at_edge is None else np.asarray(at_edge),
        alive=alive, from_policy=pol,
    )


class TestPerimeter:

    def test_one_definition_for_reward_and_diagnostic(self):
        """The wall penalty and the logged edge_frac read the same function,
        so the number charged for and the number reported cannot drift."""
        import inspect
        from hopfield_nav.rollout import collector
        src = inspect.getsource(collector)
        assert src.count("on_perimeter(") >= 2

    def test_corners_edges_and_interior(self):
        pos = np.array([[0, 0], [19, 19], [0, 7], [7, 0], [7, 19], [10, 10]])
        got = on_perimeter(pos, 20)
        assert got.tolist() == [True, True, True, True, True, False]

    def test_uniform_occupancy_is_not_zero(self):
        """19% of a 20x20 arena is perimeter and a good explorer MUST go
        there -- so edge_frac is a number to read, never one to minimise."""
        all_cells = np.array([[x, y] for x in range(20) for y in range(20)])
        assert on_perimeter(all_cells, 20).mean() == pytest.approx(0.19)


class TestCosAQ:

    def test_perfect_following_is_one(self):
        d = RegimeDiagnostics(2)
        _step(d, q=[[1, 0], [0, 2]], action=[[3, 0], [0, 1]],
              realized=[[1, 0], [0, 1]])
        assert d.summary()["cos_aq"] == pytest.approx(1.0)

    def test_anti_following_is_minus_one(self):
        d = RegimeDiagnostics(1)
        _step(d, q=[[1, 0]], action=[[-1, 0]], realized=[[-1, 0]])
        assert d.summary()["cos_aq"] == pytest.approx(-1.0)

    def test_epsilon_steps_are_excluded(self):
        """A policy that perfectly ANTI-follows on its own steps, with a
        random step mixed in, must still read -1. Including the override
        would report a policy that ignores q as one that partly follows it."""
        d = RegimeDiagnostics(2)
        _step(d, q=[[1, 0], [1, 0]], action=[[-1, 0], [1, 0]],
              realized=[[-1, 0], [1, 0]], pol=[True, False])
        assert d.summary()["cos_aq"] == pytest.approx(-1.0)

    def test_a_null_recall_contributes_no_angle(self):
        """||q|| below Q_EPS is a numerical accident, not a direction."""
        d = RegimeDiagnostics(2)
        _step(d, q=[[1, 0], [1e-9, 0]], action=[[1, 0], [-1, 0]],
              realized=[[1, 0], [-1, 0]])
        s = d.summary()
        assert s["cos_aq"] == pytest.approx(1.0)
        assert s["cos_aq_frac"] == pytest.approx(0.5)

    def test_dead_rows_are_excluded(self):
        d = RegimeDiagnostics(2)
        _step(d, q=[[1, 0], [1, 0]], action=[[1, 0], [-1, 0]],
              realized=[[1, 0], [-1, 0]], alive=[True, False])
        assert d.summary()["cos_aq"] == pytest.approx(1.0)


class TestClipAndPin:

    def test_clip_matches_behavior_probe_threshold(self):
        """behavior_probe: `clipped = realized < 0.9 * want`. 0.95 of the
        commanded magnitude is NOT clipped; 0.5 is."""
        d = RegimeDiagnostics(2)
        _step(d, q=[[1, 0], [1, 0]], action=[[1, 0], [1, 0]],
              realized=[[0.95, 0], [0.5, 0]])
        assert d.summary()["clip_frac"] == pytest.approx(0.5)

    def test_the_wall_pin_signature(self):
        """§18.7: commanding ~0.79 and realizing ~0.09. Both conditions are
        required -- clip_frac > 0.5 AND realized speed < 0.5."""
        d = RegimeDiagnostics(1)
        for _ in range(10):
            _step(d, q=[[0.1, 0]], action=[[0.79, 0]], realized=[[0.09, 0]])
        s = d.summary()
        assert s["pin_frac"] == pytest.approx(1.0)
        assert s["clip_frac"] == pytest.approx(1.0)
        assert s["realized_mag"] == pytest.approx(0.09)
        assert s["cmd_mag"] == pytest.approx(0.79)

    def test_a_policy_parked_at_the_clamp_is_clipped_but_NOT_pinned(self):
        """The trap the probe's own docstring warns about: a policy commanding
        8.18 against a cap of 2.0 reads clip_frac 1.000 with no wall involved.
        pin_frac is the clamp-immune statistic and must stay 0 here."""
        d = RegimeDiagnostics(1)
        for _ in range(10):
            _step(d, q=[[0.1, 0]], action=[[8.18, 0]], realized=[[1.98, 0]])
        s = d.summary()
        assert s["clip_frac"] == pytest.approx(1.0)
        assert s["pin_frac"] == pytest.approx(0.0)

    def test_pin_is_per_row_not_pooled(self):
        """Half the rows fully pinned vs every row half-pinned: identical
        pooled clip_frac, and only the first is the basin."""
        half_pinned = RegimeDiagnostics(2)
        for _ in range(10):
            _step(half_pinned, q=[[.1, 0]] * 2,
                  action=[[0.8, 0]] * 2, realized=[[0.09, 0], [0.8, 0]])
        every_row_half = RegimeDiagnostics(2)
        for t in range(10):
            r = [[0.09, 0]] * 2 if t < 5 else [[0.8, 0]] * 2
            _step(every_row_half, q=[[.1, 0]] * 2,
                  action=[[0.8, 0]] * 2, realized=r)
        a, b = half_pinned.summary(), every_row_half.summary()
        assert a["clip_frac"] == pytest.approx(b["clip_frac"])
        assert a["pin_frac"] == pytest.approx(0.5)
        assert b["pin_frac"] == pytest.approx(0.0)

    def test_thresholds_are_the_documented_ones(self):
        assert (PIN_CLIP, PIN_SPEED) == (0.5, 0.5)


class TestSummaryShape:

    def test_an_empty_rollout_gives_zeros_not_nan(self):
        """tri finding 8's convention. A NaN would poison the wandb series and
        hide exactly the updates worth looking at."""
        s = RegimeDiagnostics(4).summary()
        assert all(np.isfinite(v) for v in s.values())
        assert s["pin_frac"] == 0.0 and s["cos_aq"] == 0.0

    def test_edge_frac_is_a_live_step_mean(self):
        d = RegimeDiagnostics(2)
        _step(d, q=[[1, 0]] * 2, action=[[1, 0]] * 2, realized=[[1, 0]] * 2,
              at_edge=[True, False])
        assert d.summary()["edge_frac"] == pytest.approx(0.5)


class TestMerge:

    def test_empty(self):
        assert merge([]) == {}

    def test_weights_by_live_steps_not_by_rollout(self):
        """A rollout that died at step 1 must not count as much as one that
        ran 99 steps."""
        long_run = RegimeDiagnostics(1)
        for _ in range(99):
            _step(long_run, q=[[1, 0]], action=[[1, 0]], realized=[[1, 0]],
                  at_edge=[False])
        short = RegimeDiagnostics(1)
        _step(short, q=[[1, 0]], action=[[1, 0]], realized=[[1, 0]],
              at_edge=[True])
        m = merge([long_run.summary(), short.summary()])
        assert m["edge_frac"] == pytest.approx(0.01)
        assert m["steps"] == pytest.approx(100.0)

    def test_cos_is_weighted_by_the_steps_that_entered_it(self):
        """cos_aq is conditioned on a usable recall, so it must be pooled on
        its own count -- not on total steps, which would let a rollout with
        no usable recall drag the mean toward zero."""
        follows = RegimeDiagnostics(1)
        for _ in range(10):
            _step(follows, q=[[1, 0]], action=[[1, 0]], realized=[[1, 0]])
        no_recall = RegimeDiagnostics(1)
        for _ in range(90):
            _step(no_recall, q=[[0.0, 0.0]], action=[[1, 0]],
                  realized=[[1, 0]])
        m = merge([follows.summary(), no_recall.summary()])
        assert m["cos_aq"] == pytest.approx(1.0)
        assert m["cos_aq_frac"] == pytest.approx(0.10)

    def test_merging_one_is_the_identity(self):
        d = RegimeDiagnostics(1)
        _step(d, q=[[1, 0]], action=[[0, 1]], realized=[[0, 1]])
        s = d.summary()
        m = merge([s])
        for k in s:
            assert m[k] == pytest.approx(s[k]), k


class TestWiring:

    def test_the_trainer_logs_both_regimes_and_the_gap(self):
        src = open("hopfield_nav/train_navigate.py").read()
        assert 'log[f"train/{name}/{k}"]' in src
        assert 'log["train/regime_gap"]' in src
        # Split by the recorded per-rollout flag, never by a list slice --
        # the rollout list is world-major and a slice mixes regimes as soon
        # as num_worlds > 1.
        assert "for r, pre in zip(rollouts, pre_flags)" in src

    def test_the_collector_attaches_it(self):
        src = open("hopfield_nav/rollout/collector.py").read()
        assert "diag=_diag.summary() if _diag is not None else None" in src
        # Continuous only: cos(a, q) needs an action vector.
        assert 'if cfg.agent.movement_mode == "continuous" else None' in src

    def test_regime_gap_is_guarded_on_a_usable_recall_in_BOTH_regimes(self):
        """Caught by running it, not by a unit test: with n_distractors = 0 the
        goal-absent memory is EMPTY, so q = 0 exactly and `chase_q` is
        *undefined*, not zero -- §7.5's degenerate condition. Ungated, the gap
        would report exploit's own cos_aq as if it were a discrimination."""
        src = open("hopfield_nav/train_navigate.py").read()
        assert 'min(a["cos_aq_frac"], b["cos_aq_frac"]) >= MIN_COS_FRAC' in src
        from hopfield_nav.train_navigate import MIN_COS_FRAC
        assert 0.0 < MIN_COS_FRAC <= 1.0

    def test_an_empty_memory_reports_no_recall_rather_than_a_zero_angle(self):
        """The statistic that makes the guard above possible."""
        d = RegimeDiagnostics(1)
        for _ in range(20):
            _step(d, q=[[0.0, 0.0]], action=[[1, 0]], realized=[[1, 0]])
        s = d.summary()
        assert s["cos_aq_frac"] == 0.0
        assert s["q_mag"] == 0.0

    def test_diag_read_before_done_is_updated(self):
        """`alive` must mean 'was stepped this iteration'. Reading it after
        `done |= goal_reached` would drop every row that arrived, which is
        exactly the population an exploit diagnostic is about."""
        src = open("hopfield_nav/rollout/collector.py").read()
        assert src.index("_diag.observe(") < src.index("done = done | goal_reached")
