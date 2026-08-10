"""Freezing the store head has to remove its objective, not just pin its weights.

`set_phase_freeze(freeze_store=True)` clears `requires_grad` on `store_head`'s
weights. That stops those weights updating -- but `store_head` is a Linear on
the shared GRU features, and backprop through a frozen Linear still reaches its
input. So `store_loss` and the store entropy bonus went on shaping the RNN trunk
every update of every run that believed the store head was out of the picture:
all of `train_navigate` (which freezes it at startup and never unfreezes) and
`train_phased` phases 2 and 3 (`phase{2,3}_freeze_store` both default True).

Invisible for the same reason `freeze_log_std` was: the assertion everyone would
have written -- "the frozen weights did not move" -- passes either way. The
property that actually distinguishes the two is whether a *store-only* knob can
still move the trunk.
"""
from __future__ import annotations

import copy

import torch

from hopfield_nav.config import AgentConfig, PPOConfig
from hopfield_nav.policy.agent import NavAgent
from hopfield_nav.rollout.types import RolloutBatch
from hopfield_nav.training.world_setup import set_phase_freeze
from hopfield_nav.updates.ppo import ppo_update

B, T = 4, 6


def _rollout() -> RolloutBatch:
    torch.manual_seed(0)
    return RolloutBatch(
        obs=torch.randn(B, T, 1),
        move_actions=torch.zeros(B, T, dtype=torch.long),
        store_actions=torch.randint(0, 2, (B, T)).float(),
        move_log_probs=torch.zeros(B, T),
        store_log_probs=torch.zeros(B, T),
        values=torch.zeros(B, T),
        rewards=torch.randn(B, T),
        bootstrap_value=torch.zeros(B),
        goal_reached=torch.zeros(B, T),
        # All-ones is what train_navigate collects: `explore_steps` is None
        # there, so nothing masks the store terms out on its behalf.
        explore_mask=torch.ones(B, T),
    )


def _trunk_after_one_update(store_ent_coef: float, freeze_store: bool) -> dict:
    """RNN weights after one PPO update, with everything but the knob fixed."""
    torch.manual_seed(1)
    agent_cfg = AgentConfig(
        hidden_size=8, num_rnn_layers=1,
        input_encoded_state=False, input_hopfield_signal=False,
    )
    agent = NavAgent(agent_cfg, input_dim=1)
    set_phase_freeze(agent, freeze_move=False, freeze_store=freeze_store,
                     freeze_value=False, freeze_rnn=False)
    opt = torch.optim.Adam(
        [p for p in agent.parameters() if p.requires_grad], lr=1e-2)

    cfg = PPOConfig(store_ent_coef=store_ent_coef)
    torch.manual_seed(2)          # ppo_update shuffles minibatches
    ppo_update(agent, [_rollout()], cfg, opt)
    return copy.deepcopy(agent.rnn.state_dict())


def _same(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def test_frozen_store_head_cannot_reach_the_trunk():
    """The regression: a store-only coefficient moving the shared RNN."""
    lo = _trunk_after_one_update(store_ent_coef=0.0, freeze_store=True)
    hi = _trunk_after_one_update(store_ent_coef=0.5, freeze_store=True)
    assert _same(lo, hi), (
        "store_ent_coef changed the RNN trunk while the store head was frozen: "
        "the store objective is still backpropagating through the frozen "
        "Linear into the shared features")


def test_the_guard_is_not_vacuous():
    """Unfrozen, the same knob must still reach the trunk."""
    lo = _trunk_after_one_update(store_ent_coef=0.0, freeze_store=False)
    hi = _trunk_after_one_update(store_ent_coef=0.5, freeze_store=False)
    assert not _same(lo, hi), (
        "store_ent_coef did not move the trunk even with the store head "
        "trainable, so the frozen-case assertion proves nothing")


def test_frozen_store_weights_do_not_move():
    """True before the fix as well -- kept so the weaker claim stays covered."""
    torch.manual_seed(1)
    agent_cfg = AgentConfig(
        hidden_size=8, num_rnn_layers=1,
        input_encoded_state=False, input_hopfield_signal=False,
    )
    agent = NavAgent(agent_cfg, input_dim=1)
    set_phase_freeze(agent, freeze_move=False, freeze_store=True,
                     freeze_value=False, freeze_rnn=False)
    before = copy.deepcopy(agent.store_head.state_dict())

    opt = torch.optim.Adam(
        [p for p in agent.parameters() if p.requires_grad], lr=1e-2)
    ppo_update(agent, [_rollout()], PPOConfig(), opt)

    assert _same(before, agent.store_head.state_dict())


def test_store_diagnostics_are_still_reported_when_frozen():
    """Dropping the terms from the loss must not empty the run's logs."""
    torch.manual_seed(1)
    agent_cfg = AgentConfig(
        hidden_size=8, num_rnn_layers=1,
        input_encoded_state=False, input_hopfield_signal=False,
    )
    agent = NavAgent(agent_cfg, input_dim=1)
    set_phase_freeze(agent, freeze_move=False, freeze_store=True,
                     freeze_value=False, freeze_rnn=False)
    opt = torch.optim.Adam(
        [p for p in agent.parameters() if p.requires_grad], lr=1e-2)

    losses = ppo_update(agent, [_rollout()], PPOConfig(), opt)
    assert losses["store_entropy"] > 0.0
    assert losses["store_loss"] != 0.0
