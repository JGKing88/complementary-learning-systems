"""The optional `input_prev_action` / `input_prev_reward` channels must be
present on step 0, not just from step 1.

Both the DAgger collector and the RNN evaluator used to assemble the previous-
action channel only when a previous action existed -- which it does not at
``t=0``. With ``input_prev_action`` on, the first step therefore fed the trunk
an input two columns narrower (continuous) or four narrower (discrete) than
``compute_rnn_input_dim`` had sized it for, and torch raised

    RuntimeError: input.size(-1) must be equal to input_size. Expected 62, got 60

on the very first forward. The flags were unusable, which is why every recorded
continual history has them off. ``prev_action_channel`` now returns an all-zero
"no previous action" row at t=0, distinct from every one-hot and from any real
displacement, and ``prev_reward`` starts at a genuine zero.

These tests fail loudly on the old code in both movement modes and on both
paths, so the regression cannot come back silently.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav.config import RNNAgentConfig
from hopfield_nav.evaluation.rnn import evaluate_nav_one_env
from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from hopfield_nav.rollout.rnn import (
    action_to_prev_channel, collect_rollout_rnn, prev_action_channel)
from hopfield_nav.world.env import GridEnv
from hopfield_nav.world.vec_env import make_vec

OBS = 24
SIZE = 6


def _agent_cfg(movement_mode: str, **kw) -> RNNAgentConfig:
    return RNNAgentConfig(
        hidden_size=16, num_rnn_layers=1, movement_mode=movement_mode,
        init_log_std=-1.0, freeze_log_std=True, **kw,
    )


def _build(movement_mode: str, **kw):
    cfg = _agent_cfg(movement_mode, **kw)
    input_dim = compute_rnn_input_dim(cfg, OBS)
    agent = RNNAgent(cfg, input_dim)
    env = GridEnv(size=SIZE, observation_size=OBS, seed=3)
    return cfg, agent, env, input_dim


# --------------------------------------------------------------------------
# the helper itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("movement_mode,width", [("discrete", 4), ("continuous", 2)])
def test_channel_is_zeros_at_t0(movement_mode, width):
    """No previous action -> an all-zero row of the correct width."""
    ch = prev_action_channel(None, movement_mode, batch=5)
    assert ch.shape == (5, width)
    assert ch.dtype == np.float32
    assert not ch.any(), "t=0 channel must be all zeros, not a spurious action"


@pytest.mark.parametrize("movement_mode", ["discrete", "continuous"])
def test_channel_matches_action_to_prev_channel_when_action_exists(movement_mode):
    """With an action in hand the helper must not change existing behavior."""
    action = (np.array([0, 3, 1]) if movement_mode == "discrete"
              else np.array([[1.0, 0.0], [0.0, -1.0], [0.7, 0.7]]))
    np.testing.assert_array_equal(
        prev_action_channel(action, movement_mode, batch=3),
        action_to_prev_channel(action, movement_mode),
    )


def test_zero_row_is_not_a_valid_one_hot():
    """The 'no action yet' code must be distinguishable from every real action."""
    zero = prev_action_channel(None, "discrete", batch=1)
    for a in range(4):
        onehot = action_to_prev_channel(np.array([a]), "discrete")
        assert not np.array_equal(zero, onehot)


# --------------------------------------------------------------------------
# the two paths that were broken
# --------------------------------------------------------------------------

@pytest.mark.parametrize("movement_mode", ["discrete", "continuous"])
@pytest.mark.parametrize("prev_action,prev_reward", [
    (True, False), (False, True), (True, True),
])
def test_collector_runs_with_optional_channels(movement_mode, prev_action, prev_reward):
    """collect_rollout_rnn must survive step 0 with either channel enabled."""
    cfg, agent, env, input_dim = _build(
        movement_mode,
        input_prev_action=prev_action, input_prev_reward=prev_reward,
    )
    vec = make_vec(env, 3, movement_mode, 1.0, False)
    rollout = collect_rollout_rnn(
        vec, agent, cfg, steps=4, device=torch.device("cpu"),
    )
    # The recorded obs must be the full promised width on *every* step,
    # including the first -- a narrower t=0 was the bug.
    assert rollout.obs.shape == (3, 4, input_dim)


@pytest.mark.parametrize("movement_mode", ["discrete", "continuous"])
@pytest.mark.parametrize("prev_action,prev_reward", [
    (True, False), (False, True), (True, True),
])
def test_evaluator_runs_with_optional_channels(movement_mode, prev_action, prev_reward):
    """evaluate_nav_one_env must survive step 0 with either channel enabled."""
    cfg, agent, env, _ = _build(
        movement_mode,
        input_prev_action=prev_action, input_prev_reward=prev_reward,
    )
    m = evaluate_nav_one_env(
        env, agent, n_trials=3, max_steps=4, device=torch.device("cpu"),
    )
    assert 0.0 <= m["nav_det"] <= 1.0


@pytest.mark.parametrize("movement_mode", ["discrete", "continuous"])
def test_prev_action_channel_actually_carries_the_last_action(movement_mode):
    """Beyond not crashing: step t's channel must hold step t-1's action.

    Guards the other direction -- a "fix" that always passed zeros would make
    every test above pass while rendering the channel useless.
    """
    cfg, agent, env, _ = _build(movement_mode, input_prev_action=True)
    vec = make_vec(env, 2, movement_mode, 1.0, False)
    rollout = collect_rollout_rnn(
        vec, agent, cfg, steps=5, device=torch.device("cpu"),
    )
    width = 4 if movement_mode == "discrete" else 2
    # The prev_action block is appended straight after the sensory vector.
    ch = rollout.obs[:, :, OBS:OBS + width].numpy()
    expected = action_to_prev_channel(
        rollout.student_move_action[:, 0].numpy(), movement_mode)

    assert not ch[:, 0].any(), "step 0 must still be the zero row"
    np.testing.assert_allclose(ch[:, 1], expected, rtol=1e-5, atol=1e-6)
