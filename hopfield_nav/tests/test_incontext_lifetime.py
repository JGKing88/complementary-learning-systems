"""A lifetime has to be longer than one backprop window, and it has to have
episode boundaries in it.

Both of these were wrong for the whole first in-context measurement, and
neither raised anything. The collector reset the hidden state at the start of
every rollout, so a training "lifetime" was 200 steps while the evaluation ran
2000-step ones -- a tenfold extrapolation. And an episode ended only on a
goal-reach, so a row that never found the goal spent the entire rollout in a
single episode and never crossed a boundary at all, which on a fresh
environment with a weak policy is the usual case. The regime the experiment
exists to train was therefore barely present in its own training data.

Neither shows up as a failure. Both produce a clean run, a plausible loss curve
and a flat result. So the tests here check the two structural facts directly:
that state crosses a rollout boundary and is used by the gradient, and that
episodes end when they run out of steps.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav.config import RNNAgentConfig, RNNBCConfig
from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from hopfield_nav.rollout.rnn import collect_rollout_rnn
from hopfield_nav.updates.bc_rnn import bc_rnn_update
from hopfield_nav.world.env import GridEnv
from hopfield_nav.world.vec_env import make_vec

OBS, SIZE, HID = 16, 8, 8


def _agent():
    cfg = RNNAgentConfig(hidden_size=HID, movement_mode="continuous",
                         init_log_std=-1.0)
    return RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS))


def _vec(batch=3, seed=0):
    env = GridEnv(size=SIZE, observation_size=OBS, seed=seed)
    v = make_vec(env, batch, "continuous", 1.0)
    v.reset_all()
    return v


def _roll(agent, vec, steps=12, **kw):
    return collect_rollout_rnn(vec, agent, agent.cfg, steps,
                               torch.device("cpu"), **kw)


# ===========================================================================
# state across a rollout boundary
# ===========================================================================

def test_rollout_returns_a_final_state_to_continue_from():
    r = _roll(_agent(), _vec(), carry_across_episodes=True)
    assert r.final_h is not None
    assert r.final_h.shape == (1, 3, HID)
    assert not r.final_h.requires_grad, "the carried state must be detached"


def test_final_state_is_not_the_initial_one():
    """If they matched, the recurrence would be carrying nothing."""
    a = _agent()
    h0 = torch.randn(1, 3, HID) * 0.1
    r = _roll(a, _vec(), carry_across_episodes=True, initial_h=h0)
    assert not torch.allclose(r.initial_h, r.final_h)


def test_the_rollout_actually_starts_from_the_state_it_was_given():
    """The defect: `h` was reset at the top of every rollout, so a lifetime
    could never be longer than one rollout however the caller chained them."""
    a = _agent()
    torch.manual_seed(0)
    zero = _roll(a, _vec(seed=1), carry_across_episodes=True, initial_h=None)
    torch.manual_seed(0)
    big = _roll(a, _vec(seed=1), carry_across_episodes=True,
                initial_h=torch.full((1, 3, HID), 0.9))
    assert not torch.allclose(zero.obs, big.obs) or \
        not torch.allclose(zero.final_h, big.final_h)


def test_initial_state_is_recorded_on_the_batch():
    a = _agent()
    h0 = torch.randn(1, 3, HID) * 0.1
    r = _roll(a, _vec(), carry_across_episodes=True, initial_h=h0)
    assert torch.allclose(r.initial_h, h0)
    r0 = _roll(a, _vec(), carry_across_episodes=True)
    assert r0.initial_h is None, "a fresh lifetime must not claim a state"


# ===========================================================================
# episode boundaries
# ===========================================================================

def test_without_a_timeout_a_row_that_misses_the_goal_never_leaves_episode_one():
    """The second half of the defect, measured directly.

    A row that does not find the goal was never teleported, so it never
    experienced an episode boundary -- and on an unseen environment with a weak
    policy that is the usual case, which is why the cross-episode regime was
    barely present in its own training data.
    """
    torch.manual_seed(0)
    a = _agent()
    r = _roll(a, _vec(batch=4, seed=3), steps=40, carry_across_episodes=True)
    finished = r.episodes_completed
    reached = (r.goal_reached.sum(dim=1) > 0)
    # Every boundary must be attributable to a goal-reach; rows that never
    # reached are stuck in episode 1 for all forty steps.
    assert int(finished[~reached].sum()) == 0, (
        "a row completed an episode without reaching the goal and without a "
        "timeout, so something else is ending episodes")


def test_a_timeout_creates_the_boundaries_a_goal_reach_never_would():
    torch.manual_seed(0)
    a = _agent()
    r = _roll(a, _vec(batch=4, seed=3), steps=20, carry_across_episodes=True,
              episode_max_steps=5)
    # Twenty steps at a five-step cap is four episodes per row, whether or not
    # the goal was ever found.
    assert int(r.episodes_completed.min()) >= 4, r.episodes_completed


def test_the_timeout_is_what_makes_lifetimes_multi_episode():
    """The mutation check: same seed, same policy, one flag."""
    torch.manual_seed(0)
    a = _agent()
    without = _roll(a, _vec(batch=4, seed=3), steps=30,
                    carry_across_episodes=True)
    torch.manual_seed(0)
    with_to = _roll(a, _vec(batch=4, seed=3), steps=30,
                    carry_across_episodes=True, episode_max_steps=6)
    assert int(with_to.episodes_completed.sum()) > int(
        without.episodes_completed.sum())


def test_episodic_mode_completes_no_episodes():
    """Episodic mode freezes a reacher instead of teleporting, so a row never
    continues into a second episode and the count is zero by construction."""
    a = _agent()
    r = _roll(a, _vec(), steps=12, carry_across_episodes=False,
              episode_max_steps=2)
    assert int(r.episodes_completed.sum()) == 0


# ===========================================================================
# the gradient has to use the carried state
# ===========================================================================

def test_the_update_starts_from_the_carried_state():
    """The subtlest half. Even with the collector carrying state, the BC update
    re-ran the agent from zero -- so the gradient treated every chunk as the
    start of its own lifetime, capping the horizon the network could learn to
    use at `steps_per_rollout` no matter how long the lifetime was."""
    def grads(h0):
        torch.manual_seed(0)
        a = _agent()
        vec = _vec(batch=3, seed=5)
        torch.manual_seed(1)
        r = _roll(a, vec, steps=10, carry_across_episodes=True, initial_h=h0)
        opt = torch.optim.SGD(a.parameters(), lr=0.0)   # measure, do not move
        bc_rnn_update(a, [r], RNNBCConfig(epochs=1, n_minibatches=1), opt,
                      "continuous")
        return torch.cat([p.grad.reshape(-1) for p in a.parameters()
                          if p.grad is not None])

    g_zero = grads(None)
    g_big = grads(torch.full((1, 3, HID), 0.9))
    assert not torch.allclose(g_zero, g_big), \
        "the update ignored the rollout's initial hidden state"


def test_update_refuses_a_mix_of_continuations_and_fresh_starts():
    a = _agent()
    fresh = _roll(a, _vec(seed=1), steps=8, carry_across_episodes=True)
    cont = _roll(a, _vec(seed=2), steps=8, carry_across_episodes=True,
                 initial_h=torch.zeros(1, 3, HID))
    opt = torch.optim.SGD(a.parameters(), lr=0.0)
    with pytest.raises(ValueError, match="initial hidden state"):
        bc_rnn_update(a, [fresh, cont], RNNBCConfig(epochs=1, n_minibatches=1),
                      opt, "continuous")


def test_update_without_any_carried_state_is_unchanged():
    """Every existing caller passes no initial state, and must be unaffected."""
    a = _agent()
    r = _roll(a, _vec(), steps=8)
    opt = torch.optim.SGD(a.parameters(), lr=1e-3)
    out = bc_rnn_update(a, [r], RNNBCConfig(epochs=1, n_minibatches=1), opt,
                        "continuous")
    assert np.isfinite(out["move_loss"])
