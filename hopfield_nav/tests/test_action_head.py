"""The radial squash and the state-dependent std.

Both replace a piece of the action parameterization, so the tests here pin the
properties whose failure would be *silent*: a squash that changes direction, a
log-prob that no longer matches the distribution, a state_dict whose keys moved,
and a std head that does not start where the policy it replaces sat.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from hopfield_nav.config import AgentConfig
from hopfield_nav.policy.action_head import (
    action_bounds_from, build_log_std, movement_std, squash_mean,
)

LO, HI = 0.5, 2.0


def _cfg(**kw):
    base = dict(movement_mode="continuous", hopfield_mode="continuous",
                hidden_size=16, init_log_std=-0.7, freeze_log_std=True)
    base.update(kw)
    return AgentConfig(**base)


class TestSquashBounds:

    def test_maps_everything_into_the_band(self):
        x = torch.randn(4096, 2) * 20.0
        r = squash_mean(x, LO, HI).norm(dim=-1)
        assert float(r.min()) >= LO - 1e-6
        assert float(r.max()) <= HI + 1e-6

    def test_zero_maps_to_zero_not_the_floor(self):
        """Direction is undefined at the origin, so the floor is not applied.
        Harmless -- the sample is mean+noise and the env floors the result --
        but it is the behaviour, so it is the test."""
        r = squash_mean(torch.zeros(8, 2), LO, HI).norm(dim=-1)
        assert torch.allclose(r, torch.zeros(8), atol=1e-6)

    def test_direction_is_exactly_preserved(self):
        """The reason it is radial. An elementwise tanh fails this test."""
        x = torch.randn(2048, 2) * 5.0
        y = squash_mean(x, LO, HI)
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        assert float(cos.min()) > 1 - 1e-6

    def test_no_diagonal_bias(self):
        """Elementwise squashing rotates axis-aligned and diagonal vectors by
        different amounts; radial squashing rotates neither."""
        axis = torch.tensor([[5.0, 0.0]])
        diag = torch.tensor([[5.0, 5.0]]) / (2 ** 0.5)
        for v in (axis, diag):
            out = squash_mean(v, LO, HI)
            cos = torch.nn.functional.cosine_similarity(v, out, dim=-1)
            assert float(cos) > 1 - 1e-6

    def test_monotone_in_magnitude(self):
        rs = torch.linspace(0.0, 40.0, 200).unsqueeze(-1) * torch.tensor([[1.0, 0.0]])
        out = squash_mean(rs, LO, HI).norm(dim=-1)
        assert bool((out[1:] - out[:-1] >= -1e-6).all())   # float32 noise at saturation

    def test_gradient_is_alive_well_past_the_bound(self):
        """Not "never zero" -- float32 tanh saturates. Measured: 6.6e-01 at
        r_raw=1, 5.1e-03 at 5, 6.4e-06 at 10, exactly 0.0 by 15. So the dead
        region moves from hi=2.0 out to ~15 rather than disappearing, which is
        what the module docstring now says."""
        for scale, floor in ((0.1, 1e-2), (1.0, 1e-2), (5.0, 1e-4)):
            x = torch.tensor([[scale, 0.0]], requires_grad=True)
            squash_mean(x, LO, HI).norm().backward()
            assert float(x.grad.abs().max()) > floor, scale

    def test_the_dead_region_is_far_outside_the_bound(self):
        """Pins where saturation actually bites, so a future change that moves
        it back toward hi fails here rather than silently."""
        x = torch.tensor([[2.5, 0.0]], requires_grad=True)   # just past hi
        squash_mean(x, LO, HI).norm().backward()
        assert float(x.grad.abs().max()) > 1e-3

    def test_effective_magnitude_is_pinned_however_far_the_raw_drifts(self):
        """The property that actually neutralizes the pathology: raw drift
        stops mattering because the effective magnitude cannot follow it."""
        for scale in (8.18, 50.0, 1000.0):    # 8.18 is the measured drift
            r = squash_mean(torch.tensor([[scale, 0.0]]), LO, HI).norm()
            assert float(r) == pytest.approx(HI, abs=1e-4)

    def test_small_inputs_are_near_the_identity_offset(self):
        x = torch.tensor([[0.01, 0.0]])
        r = float(squash_mean(x, LO, HI).norm())
        assert r == pytest.approx(LO + 0.01, abs=1e-4)


class TestStateDependentStd:

    def test_head_starts_at_the_global_value(self):
        """Zero weights and init_log_std in the bias, so step one reproduces
        the policy it replaces rather than starting from noise."""
        cfg = _cfg(state_dependent_std=True)
        param, head = build_log_std(cfg, cfg.hidden_size)
        assert param is None and head is not None
        feats = torch.randn(32, cfg.hidden_size) * 10.0
        std = movement_std(cfg, feats, torch.zeros(32, 2), None, head)
        assert torch.allclose(std, torch.full((32, 2), 0.4966), atol=1e-3)

    def test_head_is_clamped(self):
        cfg = _cfg(state_dependent_std=True, log_std_min=-2.5, log_std_max=0.5)
        _, head = build_log_std(cfg, cfg.hidden_size)
        torch.nn.init.normal_(head.weight, std=5.0)      # force wild outputs
        with torch.no_grad():
            std = movement_std(cfg, torch.randn(512, cfg.hidden_size),
                               torch.zeros(512, 2), None, head)
        import math
        assert float(std.min()) >= math.exp(-2.5) - 1e-6
        assert float(std.max()) <= math.exp(0.5) + 1e-6

    def test_global_path_is_unchanged(self):
        cfg = _cfg(state_dependent_std=False)
        param, head = build_log_std(cfg, cfg.hidden_size)
        assert head is None and param is not None
        assert param.requires_grad is False              # freeze_log_std=True
        std = movement_std(cfg, torch.randn(8, cfg.hidden_size),
                           torch.zeros(8, 2), param, None)
        assert torch.allclose(std, torch.full((8, 2), 0.4966), atol=1e-3)

    def test_unfrozen_global_keeps_gradient(self):
        cfg = _cfg(state_dependent_std=False, freeze_log_std=False)
        param, _ = build_log_std(cfg, cfg.hidden_size)
        assert param.requires_grad is True


class TestAgentIntegration:
    """The properties that make this safe to ship: keys, log-prob, defaults."""

    def _agent(self, **kw):
        from hopfield_nav.policy.agent import NavAgent
        cfg = _cfg(**kw)
        bounds = (LO, HI) if kw.get("action_squash") else None
        return NavAgent(cfg, input_dim=8, action_bounds=bounds), cfg

    def test_state_dict_keys_unchanged_when_flags_are_off(self):
        """Renaming a parameter would break loading every checkpoint in the
        project, so the default path must keep the original names."""
        agent, _ = self._agent()
        keys = set(agent.state_dict())
        assert "movement_log_std" in keys
        assert "movement_mean.weight" in keys
        assert not any(k.startswith("movement_log_std_head") for k in keys)

    def test_state_head_adds_keys_only_when_enabled(self):
        agent, _ = self._agent(state_dependent_std=True)
        keys = set(agent.state_dict())
        assert "movement_log_std_head.weight" in keys
        assert "movement_log_std" not in keys

    def test_squash_bounds_the_distribution_mean(self):
        agent, _ = self._agent(action_squash=True)
        x = torch.randn(4, 6, 8) * 50.0
        dist, *_ = agent(x) if not isinstance(agent(x), tuple) else (agent(x)[0],)
        r = dist.mean.norm(dim=-1)
        assert float(r.max()) <= HI + 1e-5
        assert float(r.min()) >= LO - 1e-5

    def test_log_prob_still_matches_the_distribution(self):
        """Squashing the MEAN needs no Jacobian; squashing the sample would.
        If someone later moves the squash onto the sample, this fails."""
        agent, _ = self._agent(action_squash=True)
        x = torch.randn(2, 3, 8)
        out = agent(x)
        dist = out[0] if isinstance(out, tuple) else out
        a = dist.sample()
        manual = torch.distributions.Normal(dist.mean, dist.stddev).log_prob(a)
        assert torch.allclose(dist.log_prob(a), manual, atol=1e-6)

    def test_squash_without_bounds_raises(self):
        from hopfield_nav.policy.agent import NavAgent
        with pytest.raises(ValueError, match="action_bounds"):
            NavAgent(_cfg(action_squash=True), input_dim=8)


class TestBoundsHelper:

    def test_returns_none_when_either_is_unset(self):
        from hopfield_nav.config import EnvConfig
        assert action_bounds_from(EnvConfig()) is None
        assert action_bounds_from(EnvConfig(min_action_norm=0.5)) is None
        assert action_bounds_from(
            EnvConfig(min_action_norm=0.5, max_action_norm=2.0)) == (0.5, 2.0)
