"""`chart_frac` — the one genuinely missing input channel P3 found.

The policy receives `q`, the recalled displacement projected into a local 2-D
Gram-Schmidt frame: about 8 of its 74 input dims. The recall itself is
1024-dimensional, so **1022 dimensions are projected away and never reach the
policy**. `chart_frac = ‖q‖ / ‖recall − x‖` is the single scalar that says how
much of the recall the local chart explains, and §7.7.2 measured it at AUC
**0.974 / 0.988** at ten distractors against ‖q‖'s **0.698 / 0.930** — +0.276
on the encoder §7 was measured on, and BETTER than the env-fitted 64-dim basis
while needing no fit at all.

§7.7.1 predicted this compression would fail. It did not. That is why the
channel exists.

**Not an oracle.** Unlike `input_visited` and `input_abs_position`, every term
is already computed in the rollout, so a model trained with it is shippable.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav.config import AgentConfig
from hopfield_nav.policy import channels
from hopfield_nav.rollout.signal import chart_fraction


class TestTheStatistic:

    def test_a_recall_fully_inside_the_chart_reads_one(self):
        """If the whole displacement lies in the 2-D frame, q captures it."""
        q = np.array([[3.0, 4.0]])
        # ‖recall − x‖ = 5 == ‖q‖
        rec = np.array([[3.0, 4.0, 0.0]])
        x = np.array([[0.0, 0.0, 0.0]])
        assert chart_fraction(q, rec, x)[0] == pytest.approx(1.0)

    def test_a_recall_mostly_outside_the_chart_reads_small(self):
        """The goal-absent case: a foreign pattern points out of the chart."""
        q = np.array([[1.0, 0.0]])
        rec = np.array([[1.0, 0.0, 10.0]])
        x = np.zeros((1, 3))
        got = chart_fraction(q, rec, x)[0]
        assert got == pytest.approx(1.0 / np.sqrt(101.0), rel=1e-5)
        assert got < 0.11

    def test_it_reproduces_the_measured_separation_in_shape(self):
        """§7.7.2: frac_goal 0.638 vs frac_dist 0.125 at ten distractors, a 5×
        gap. Not the exact numbers — those need the real encoder — but the
        statistic must be able to express that separation at all."""
        D = 64
        rng = np.random.default_rng(0)
        x = np.zeros((2, D))
        # goal-like: displacement mostly inside the 2-D chart
        goal = np.zeros(D)
        goal[:2] = [0.64, 0.0]
        goal[2:] += rng.normal(0, 0.01, D - 2)
        # distractor-like: an unrelated direction in D dimensions
        dist = rng.normal(0, 1.0, D)
        dist[:2] = [0.125, 0.0]
        rec = np.stack([goal, dist])
        q = rec[:, :2].copy()
        got = chart_fraction(q, rec, x)
        assert got[0] > 0.9
        assert got[1] < 0.3
        assert got[0] / got[1] > 3.0

    def test_no_recall_is_zero_not_a_ratio(self):
        """A row with no memory has recalled == 0, so the displacement is −x
        and the ratio is a perfectly finite number that means NOTHING. The
        same distinction cos_aq_frac needed in the regime diagnostics."""
        q = np.array([[0.0, 0.0], [1.0, 0.0]])
        rec = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        x = np.array([[5.0, 5.0, 5.0], [0.0, 0.0, 0.0]])
        got = chart_fraction(q, rec, x, valid=np.array([False, True]))
        assert got[0] == 0.0
        assert got[1] == pytest.approx(1.0)

    def test_without_the_valid_mask_it_would_invent_a_number(self):
        """Documents why the mask is not optional in practice."""
        q = np.array([[0.0, 0.0]])
        rec = np.zeros((1, 3))
        x = np.array([[5.0, 0.0, 0.0]])
        assert chart_fraction(q, rec, x)[0] == 0.0  # q is 0 here, so 0/5
        q2 = np.array([[1.0, 0.0]])
        assert chart_fraction(q2, rec, x)[0] > 0.0  # a meaningless 1/5

    def test_it_is_bounded_and_finite(self):
        rng = np.random.default_rng(1)
        rec = rng.normal(size=(64, 32))
        x = rng.normal(size=(64, 32))
        q = (rec - x)[:, :2]
        got = chart_fraction(q, rec, x)
        assert np.all(np.isfinite(got))
        assert np.all(got >= 0.0) and np.all(got <= 1.0 + 1e-6)


class TestChannel:

    def _cfg(self, **kw):
        # encoded_state off: it is embed_dim wide and this class is about the
        # one-dim channel, not about the rest of the layout.
        return AgentConfig(hopfield_mode="continuous",
                           movement_mode="continuous",
                           input_encoded_state=False, **kw)

    def test_off_by_default(self):
        assert AgentConfig().input_chart_frac is False
        names = [c.name for c in channels.channel_specs(self._cfg(), 8)]
        assert "chart_frac" not in names

    def test_on_it_adds_exactly_one_dim(self):
        base = channels.input_dim(self._cfg(), 8)
        on = channels.input_dim(self._cfg(input_chart_frac=True), 8)
        assert on == base + 1

    def test_it_appends_rather_than_reordering(self):
        """Channel order is a compatibility surface: every saved checkpoint's
        first layer was trained against it. A new channel must not move an
        existing one."""
        off = [c.name for c in channels.channel_specs(self._cfg(), 8)]
        on = [c.name for c in
              channels.channel_specs(self._cfg(input_chart_frac=True), 8)]
        assert [n for n in on if n != "chart_frac"] == off

    def test_an_enabled_but_unsupplied_channel_raises(self):
        """build_policy_input is strict on purpose -- the failure mode it
        prevents is a silent layout shift, where the tensor keeps its shape
        while a channel moves and the only symptom is worse behaviour."""
        specs = channels.channel_specs(self._cfg(input_chart_frac=True), 8)
        values = {"current_reward": torch.zeros(2, 1),
                  "hopfield_signal": torch.zeros(2, 2)}
        with pytest.raises(KeyError, match="chart_frac"):
            channels.build_policy_input(specs, values, batch_size=2)

    def test_supplied_at_the_right_width_it_builds(self):
        specs = channels.channel_specs(self._cfg(input_chart_frac=True), 8)
        values = {"current_reward": torch.zeros(2, 1),
                  "hopfield_signal": torch.zeros(2, 2),
                  "chart_frac": torch.ones(2, 1)}
        out = channels.build_policy_input(specs, values, batch_size=2)
        assert out.shape == (2, 4)
        assert out[0, -1].item() == pytest.approx(1.0)


class TestWiring:

    def test_every_policy_input_site_supplies_it(self):
        """Four assembly sites -- the rollout main loop, the rollout
        bootstrap, and both batched evaluators -- plus metrics' B=1 path."""
        for path, n in (("hopfield_nav/rollout/collector.py", 2),
                        ("hopfield_nav/evaluation/batched.py", 2),
                        ("hopfield_nav/evaluation/metrics.py", 1)):
            src = open(path).read()
            assert src.count('"chart_frac"') >= n, path

    def test_the_signal_helper_is_opt_in(self):
        """Fifteen call sites unpack the four-tuple; the fifth element is
        opt-in so none of them had to change."""
        src = open("hopfield_nav/rollout/signal.py").read()
        assert "return_chart: bool = False" in src

    def test_chart_is_initialised_before_the_branch(self):
        """It is bound only inside the hopfield_signal branch, so a config
        with that channel off would NameError without a default."""
        for path in ("hopfield_nav/evaluation/batched.py",
                     "hopfield_nav/evaluation/metrics.py"):
            src = open(path).read()
            assert "_chart = None" in src, path
