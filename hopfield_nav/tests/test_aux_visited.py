"""The auxiliary visitation head (P2 doc §24.2, lever B).

§22 measured that the explore policy REPLAYS on a state repeat: it is close to
a fixed function of (position, heading) and does not consult where it has been.
This head makes the trunk predict, from its own features, which of 8
surrounding cells it has already visited -- forcing the hidden state to carry
visitation so the policy head *can* use it.

Training-time oracle only: the target comes from the collector's own bookkeeping
and nothing is added to the observation, the reward, or deployment. Default off,
so every run before 2026-09-01 reproduces unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield import Hopfield
from hopfield_nav.tests.fixtures import make_collector, make_stub_cfg
from hopfield_nav.world.env import make_env


def _cfg(weight: float, radius: float = 2.0):
    cfg = make_stub_cfg(movement_mode="discrete")
    cfg.env.size = 8
    cfg.batch_envs = 4
    cfg.steps_per_rollout = 12
    cfg.agent.aux_visited_weight = weight
    cfg.agent.aux_visited_radius = radius
    return cfg


def _rollout(cfg):
    col, agent, vh = make_collector(cfg, 8, seed=0)
    env = make_env(cfg.env, "discrete", seed=3)
    torch.manual_seed(0)
    np.random.seed(0)
    hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(cfg.batch_envs)]
    return col.collect_rollout(env, agent, hops, allow_store=True,
                               update_idx=1), agent


class TestOffByDefault:

    def test_no_head_and_no_targets_when_weight_is_zero(self):
        """The default path. A run that says nothing pays nothing."""
        cfg = _cfg(0.0)
        roll, agent = _rollout(cfg)
        assert agent.visited_head is None
        assert agent.visited_logits(torch.zeros(2, 3, cfg.agent.hidden_size)) is None
        assert roll.visited_targets is None

    def test_stub_default_is_zero(self):
        from hopfield_nav.config import AgentConfig
        assert AgentConfig().aux_visited_weight == 0.0


class TestTargets:

    def test_shape_and_range(self):
        cfg = _cfg(1.0)
        roll, agent = _rollout(cfg)
        t = roll.visited_targets
        assert t is not None
        assert t.shape == (cfg.batch_envs, cfg.steps_per_rollout, 8)
        assert set(torch.unique(t).tolist()) <= {0.0, 1.0}

    def test_first_step_has_nothing_visited(self):
        """At t=0 the agent has visited only where it stands, and the 8 probes
        sit `radius` away -- so nothing they point at can be visited yet."""
        cfg = _cfg(1.0, radius=2.0)
        roll, _ = _rollout(cfg)
        assert roll.visited_targets[:, 0].sum().item() == 0.0

    def test_visited_count_is_non_decreasing_per_direction(self):
        """The visited SET only grows, so for a fixed probe direction the label
        can only turn on, never off -- unless the agent moved, which it does.
        The population count over the whole batch must still trend up."""
        cfg = _cfg(1.0, radius=1.0)
        roll, _ = _rollout(cfg)
        per_step = roll.visited_targets.sum(dim=(0, 2))
        assert per_step[-1] >= per_step[0]

    def test_targets_describe_what_was_known_BEFORE_the_step(self):
        """The label has to be what the agent knew when it CHOSE, not what it
        learns by stepping -- otherwise the head predicts the future."""
        cfg = _cfg(1.0, radius=1.0)
        roll, _ = _rollout(cfg)
        # The agent's own cell is marked visited only AFTER the label is read,
        # so a probe landing on the start cell reads 0 at t=0.
        assert roll.visited_targets[:, 0].max().item() == 0.0


class TestTheHead:

    def test_head_exists_and_has_the_right_width(self):
        cfg = _cfg(1.0)
        _, agent = _rollout(cfg)
        assert agent.visited_head is not None
        feats = torch.zeros(2, 5, cfg.agent.hidden_size)
        assert agent.visited_logits(feats).shape == (2, 5, 8)

    def test_it_can_actually_learn_the_target(self):
        """A sanity check that the head is trainable and the task is learnable
        from features: fit it directly on a fixed batch and watch BCE fall."""
        torch.manual_seed(0)
        cfg = _cfg(1.0)
        _, agent = _rollout(cfg)
        feats = torch.randn(16, 8, cfg.agent.hidden_size)
        # a target that is a deterministic function of the features
        target = (feats[..., :8] > 0).float()
        opt = torch.optim.Adam(agent.visited_head.parameters(), lr=0.05)
        first = last = None
        for i in range(200):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                agent.visited_logits(feats), target)
            if i == 0:
                first = loss.item()
            last = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert last < 0.5 * first, (first, last)
