"""Explore rollouts end when the agent reaches the goal.

Before 2026-08-12 they did not: with the goal active, an explore env ran the
TRAINING contract, so reaching the goal paid +1, discarded the move, teleported
the agent to a fresh start, and the rollout carried on for its full
`steps_per_rollout`.

The hard part is not the freezing, it is the value function. `compute_gae`
documented "No terminal states within a rollout -- only truncation at the end",
and a reached goal *is* terminal: there is no return past it. Bootstrapping
`gamma * V(next)` across that boundary teaches the value head that reward keeps
flowing after the episode is over, and no loss curve says so. Half of the tests
here are about that.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield import Hopfield
from hopfield_nav.config import TrainConfig
from hopfield_nav.tests.fixtures import make_collector, make_stub_cfg
from hopfield_nav.updates.ppo import compute_gae
from hopfield_nav.world.env import make_env


# ---------------------------------------------------------------------------
# GAE across a terminal state
# ---------------------------------------------------------------------------

def _gae_inputs(B=3, T=6, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, T), torch.randn(B, T), torch.randn(B)


def test_no_terminations_is_the_historical_arithmetic_exactly():
    """Every run that does not end episodes early must be untouched -- not
    'equivalent up to a mask of ones', identical."""
    r, v, boot = _gae_inputs()
    a0, g0 = compute_gae(r, v, boot, 0.99, 0.95)
    a1, g1 = compute_gae(r, v, boot, 0.99, 0.95, alive=torch.ones_like(r))
    assert torch.equal(a0, a1) and torch.equal(g0, g1)


def test_a_terminal_step_has_no_future_term():
    """The whole point. Advantage at the last alive step is r - V, with no
    `gamma * V(next)` -- because there is no next."""
    r, v, boot = _gae_inputs()
    alive = torch.ones_like(r)
    alive[1, 3:] = 0
    a, _ = compute_gae(r, v, boot, 0.99, 0.95, alive=alive)
    assert torch.allclose(a[1, 2], r[1, 2] - v[1, 2])


def test_bootstrapping_across_the_ending_is_what_this_rules_out():
    """Stated as a difference, so the test fails if the `cont` factor is
    dropped: treating the ending as truncation gives a different answer."""
    r, v, boot = _gae_inputs()
    alive = torch.ones_like(r)
    alive[1, 3:] = 0
    terminal, _ = compute_gae(r, v, boot, 0.99, 0.95, alive=alive)
    truncated, _ = compute_gae(r, v, boot, 0.99, 0.95)   # no termination at all
    assert not torch.allclose(terminal[1, 2], truncated[1, 2]), (
        "ending an episode made no difference to its last advantage, so the "
        "terminal boundary is not being cut")


def test_dead_steps_carry_no_advantage_and_no_return():
    r, v, boot = _gae_inputs()
    alive = torch.ones_like(r)
    alive[1, 3:] = 0
    a, g = compute_gae(r, v, boot, 0.99, 0.95, alive=alive)
    assert bool((a[1, 3:] == 0).all()) and bool((g[1, 3:] == 0).all())


def test_one_row_ending_does_not_disturb_the_others():
    """B trajectories share an env; they must not share an ending."""
    r, v, boot = _gae_inputs()
    base, _ = compute_gae(r, v, boot, 0.99, 0.95)
    alive = torch.ones_like(r)
    alive[1, 3:] = 0
    got, _ = compute_gae(r, v, boot, 0.99, 0.95, alive=alive)
    assert torch.allclose(got[0], base[0]) and torch.allclose(got[2], base[2])


def test_a_row_alive_to_the_horizon_still_bootstraps():
    """Ending early is terminal; running out of steps is truncation. The two
    must stay distinguishable or a long episode is scored as a failed one."""
    r, v, boot = _gae_inputs()
    alive = torch.ones_like(r)
    alive[1, 3:] = 0
    a, _ = compute_gae(r, v, boot, 0.99, 0.95, alive=alive)
    base, _ = compute_gae(r, v, boot, 0.99, 0.95)
    assert torch.allclose(a[0], base[0])


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------

def _rollout(ends: bool, steps: int = 30, B: int = 4):
    cfg = make_stub_cfg(movement_mode="discrete")
    cfg.env.size = 4
    cfg.env.goal_radius = 1.5
    cfg.env.goals_active = True
    cfg.steps_per_rollout = steps
    cfg.batch_envs = B
    col, agent, vh = make_collector(cfg, 8, seed=0)
    env = make_env(cfg.env, "discrete", seed=100)
    torch.manual_seed(0)
    np.random.seed(0)
    hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(B)]
    return col.collect_rollout(env, agent, hops, allow_store=True,
                               update_idx=1, ends_on_goal=ends)


def test_off_by_default_leaves_no_mask_at_all():
    """None rather than ones, so PPO takes its historical reduction."""
    assert _rollout(False).alive_mask is None


def test_each_trajectory_ends_at_its_own_goal_step():
    out = _rollout(True)
    am = out.alive_mask
    assert am is not None and int(am.sum()) < am.numel(), (
        "nothing terminated; this fixture cannot show the feature working")
    for b in range(am.shape[0]):
        n = int(am[b].sum())
        assert bool((am[b].diff() <= 0).all()), "a finished row came back to life"
        assert int(out.goal_reached[b, n - 1]) == 1, (
            "the last alive step was not the goal step")


def test_finished_rows_stop_accruing_anything():
    out = _rollout(True)
    dead = out.alive_mask == 0
    assert bool(dead.any())
    assert bool((out.rewards[dead] == 0).all()), "a frozen row was still paid"


def test_a_finished_row_bootstraps_nothing():
    """Its episode is over, so the value past the horizon is zero -- not
    V(wherever it happens to be frozen)."""
    out = _rollout(True)
    finished = out.alive_mask[:, -1] == 0
    assert bool(finished.any())
    assert bool((out.bootstrap_value[finished] == 0).all())


def test_the_rows_that_end_are_the_ones_that_reached_the_goal():
    """Not merely 'some rows stopped': the ending is caused by the goal."""
    out = _rollout(True)
    am = out.alive_mask
    for b in range(am.shape[0]):
        n = int(am[b].sum())
        # No goal contact before the final alive step, or it would have ended
        # there instead.
        assert int(out.goal_reached[b, :n - 1].sum()) == 0


# ---------------------------------------------------------------------------
# The regimes
# ---------------------------------------------------------------------------

def test_explore_declares_it_and_exploit_does_not():
    """Both regimes run with the goal active, so `goals_active` cannot be the
    signal -- exploit still teleports and keeps navigating."""
    from hopfield_nav.training.exploit import ExploitRegime
    from hopfield_nav.training.explore import ExploreRegime

    cfg = make_stub_cfg(movement_mode="discrete")
    dev, rng = torch.device("cpu"), np.random.RandomState(0)
    ex = ExploreRegime(cfg, 8, dev, rng, goals_off=False, ends_on_goal=True)
    xp = ExploitRegime(cfg, 8, dev, rng)
    assert ex.ends_on_goal is True
    assert getattr(xp, "ends_on_goal", False) is False


def test_it_is_vacuous_when_the_goal_is_off():
    """With `--explore_goals_off` there is no goal event to end on, so the
    clause must not claim there is one."""
    from hopfield_nav.training.explore import ExploreRegime
    from hopfield_nav.training.stages import Knobs

    cfg = make_stub_cfg(movement_mode="discrete")
    reg = ExploreRegime(cfg, 8, torch.device("cpu"), np.random.RandomState(0),
                        goals_off=True, ends_on_goal=True)
    assert reg.ends_on_goal is True          # asked for
    # ...but the resolved spec says no, because there is nothing to end on.
    assert (not reg.goals_off) is False


def test_the_default_is_on():
    assert TrainConfig().explore_ends_on_goal is True


# ---------------------------------------------------------------------------
# End to end through PPO
# ---------------------------------------------------------------------------

def test_ppo_consumes_a_terminating_rollout():
    """The masks reach the losses and nothing degenerates -- a pool where most
    steps are dead still produces finite gradients."""
    from hopfield_nav.updates.ppo import ppo_update

    cfg = make_stub_cfg(movement_mode="discrete")
    cfg.env.size = 4
    cfg.env.goal_radius = 1.5
    cfg.env.goals_active = True
    cfg.steps_per_rollout = 30
    cfg.batch_envs = 4
    col, agent, vh = make_collector(cfg, 8, seed=0)
    env = make_env(cfg.env, "discrete", seed=100)
    torch.manual_seed(0)
    np.random.seed(0)
    hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(4)]
    r = col.collect_rollout(env, agent, hops, allow_store=True, update_idx=1,
                            ends_on_goal=True)
    assert int(r.alive_mask.sum()) < r.alive_mask.numel()

    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
    losses = ppo_update(agent, [r], cfg.ppo, opt)
    for k, v in losses.items():
        assert np.isfinite(v), f"{k} came back {v}"
