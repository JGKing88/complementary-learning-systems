"""Ramping the kappa CEILING over training (P2 doc §24.2, lever A).

The cap is a training-time device: kappa does not affect a deterministic action
at all (§20.1, measured -- a 4x kappa change moved every behavioural statistic
by <0.001). §17.9 needed it ON early, for exploit's policy-space exploration.
§24 wants it OFF late, so the MEAN policy -- the one actually deployed -- is
optimized in something closer to its deployment regime.

Default is no ramp, so every run before 2026-09-01 reproduces unchanged.
"""
from __future__ import annotations

import pytest
import torch

from hopfield_nav.config import AgentConfig
from hopfield_nav.policy.polar_head import PolarHead
from hopfield_nav.train_navigate import _compute_log_kappa_max as ramp


class TestTheRamp:

    def test_no_end_is_constant(self):
        """The default. Every historical run takes this path."""
        for u in (1, 50, 500, 5000):
            assert ramp(u, 2.5, None, 400) == 2.5

    def test_zero_updates_is_constant(self):
        for u in (1, 50, 500):
            assert ramp(u, 2.5, 5.0, 0) == 2.5

    def test_linear_between_the_endpoints(self):
        assert ramp(1, 2.5, 5.0, 400) == pytest.approx(2.5)
        assert ramp(201, 2.5, 5.0, 400) == pytest.approx(3.75)
        assert ramp(401, 2.5, 5.0, 400) == pytest.approx(5.0)

    def test_clamped_past_the_end(self):
        """A run longer than the anneal window sits at the end value, not past
        it -- otherwise a 700-update run on a 400-update ramp would reach 7.2
        and uncap kappa entirely by accident."""
        assert ramp(700, 2.5, 5.0, 400) == pytest.approx(5.0)
        assert ramp(10_000, 2.5, 5.0, 400) == pytest.approx(5.0)

    def test_ramps_downward_too(self):
        """Nothing in the helper assumes end > start."""
        assert ramp(201, 5.0, 2.5, 400) == pytest.approx(3.75)


class TestTheHeadHonoursIt:

    @staticmethod
    def _head(**kw):
        cfg = AgentConfig(movement_mode="continuous",
                          hopfield_mode="continuous", **kw)
        return PolarHead(cfg, 16, 0.5, 1.0)

    def test_the_cap_is_mutable_and_read_at_forward_time(self):
        """The ramp assigns onto the head between updates, so the value must be
        read when the distribution is built, not captured at construction."""
        h = self._head(log_kappa_max=2.5)
        assert h.log_kappa_max == 2.5
        h.log_kappa_max = 5.0
        assert h.log_kappa_max == 5.0

    def test_a_lower_cap_gives_a_lower_kappa(self):
        """The whole point: the ceiling has to bind."""
        torch.manual_seed(0)
        feats = torch.randn(8, 16)
        mean = torch.randn(8, 2)
        h = self._head(log_kappa_max=5.0)
        # drive log_kappa well above both ceilings so the clamp is what binds
        with torch.no_grad():
            if h.log_kappa_head is not None:
                h.log_kappa_head.bias.fill_(9.0)
                h.log_kappa_head.weight.zero_()
            else:
                h.log_kappa.fill_(9.0)
        hi = h(feats, mean).kappa.mean().item()
        h.log_kappa_max = 2.5
        lo = h(feats, mean).kappa.mean().item()
        assert lo < hi, (lo, hi)
        # and the ceiling really is exp(cap), since shrink is in [0, 1) and
        # can only pull kappa further down
        ceiling = float(torch.tensor(2.5).exp())          # 12.182
        assert lo <= ceiling + 1e-4, (lo, ceiling)
