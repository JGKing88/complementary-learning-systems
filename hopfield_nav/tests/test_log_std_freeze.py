"""`--freeze_log_std` must survive the phase-freeze mask.

`movement_log_std` is a movement parameter, so `set_phase_freeze` with
`freeze_move=False` -- which is what every trainer passes, since no phase
freezes movement -- used to hand its gradient straight back and silently undo
the flag. That is not a cosmetic bug: a learnable log_std is the exact failure
V10 was built to avoid, where PPO's loss on sampled actions leaves the policy
*mean* under-pressured, the mean drifts, and deterministic-eval coverage falls
away from its own training-time peak.

It was invisible because nothing asserted on `requires_grad` and the only
symptom was a slow drift in a printed `std=` field.
"""
from __future__ import annotations

import torch

from hopfield_nav.config import AgentConfig
from hopfield_nav.policy.agent import NavAgent, compute_input_dim
from hopfield_nav.training.world_setup import move_params, set_phase_freeze

EMBED_DIM = 16
OBS_SIZE = 12


def _agent(freeze: bool) -> NavAgent:
    cfg = AgentConfig(movement_mode="continuous", hidden_size=32,
                      init_log_std=-1.8, freeze_log_std=freeze)
    return NavAgent(cfg, compute_input_dim(cfg, EMBED_DIM, OBS_SIZE))


def test_freeze_survives_the_phase_mask():
    agent = _agent(freeze=True)
    assert agent.movement_log_std.requires_grad is False

    # What train_navigate does at startup: movement is never a frozen head.
    set_phase_freeze(agent, freeze_move=False, freeze_store=True,
                     freeze_value=False, freeze_rnn=False)

    assert agent.movement_log_std.requires_grad is False, (
        "unfreezing the movement head re-enabled log_std, which is the bug: "
        "--freeze_log_std became a no-op on every train_navigate run")
    # The rest of the movement head is unfrozen, or the mask did nothing.
    assert any(p.requires_grad for p in agent.movement_mean.parameters())


def test_unfrozen_log_std_is_left_alone():
    agent = _agent(freeze=False)
    set_phase_freeze(agent, freeze_move=False, freeze_store=True,
                     freeze_value=False, freeze_rnn=False)
    assert agent.movement_log_std.requires_grad is True


def test_a_frozen_log_std_does_not_move_under_an_optimizer_step():
    """The property that actually matters, end to end."""
    agent = _agent(freeze=True)
    set_phase_freeze(agent, freeze_move=False, freeze_store=True,
                     freeze_value=False, freeze_rnn=False)
    before = agent.movement_log_std.detach().clone()

    opt = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad],
                           lr=1e-1)
    # Any loss that would push log_std up if it were trainable.
    loss = -agent.movement_log_std.sum() + sum(
        p.sum() for p in agent.movement_mean.parameters())
    loss.backward()
    opt.step()

    assert agent.movement_log_std.grad is None
    torch.testing.assert_close(agent.movement_log_std.detach(), before)


def test_move_params_still_reports_log_std():
    """The freeze is enforced in set_phase_freeze, not by hiding the parameter.

    `move_params` is also what builds optimizer groups, so dropping log_std
    from it would break `--no-freeze_log_std` instead.
    """
    agent = _agent(freeze=True)
    assert any(p is agent.movement_log_std for p in move_params(agent))
