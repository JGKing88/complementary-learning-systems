"""C5 of the at-goal contract, under `--reset_state_on_teleport`.

Whether the RNN hidden state survives a post-goal teleport is a claim about what
the agent's recurrence *means*: reset, each goal starts a fresh episode and
memory cannot span them; carried, one rollout is one continuous experience.
Until 2026-08-12 the answer was hardcoded to "reset" in five places across four
files, three of them routed through the contract and two deriving the rule for
themselves.

The golden fixtures do not discriminate here, which is why these tests exist.
Measured while writing them: the collector's golden rollouts reach a goal on
**zero** of 88 recorded steps, so the training reset had no coverage at all; and
the long-horizon evaluators teleport 18 times but their aggregate numbers are
identical either way -- a randomly-initialised `hidden_size=16` GRU does not
shift a 4-way argmax. So these look at the hidden state itself.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield import Hopfield
from hopfield_nav.config import EnvConfig
from hopfield_nav.evaluation import metrics as ev
from hopfield_nav.tests.fixtures import make_collector, make_stub_cfg
from hopfield_nav.world import episode
from hopfield_nav.world.env import make_env


def _cfg(flag: bool):
    """A world an untrained policy actually reaches the goal in."""
    cfg = make_stub_cfg(movement_mode="discrete")
    cfg.env.size = 4
    cfg.env.goal_radius = 1.5
    cfg.env.goals_active = True
    cfg.env.reset_state_on_teleport = flag
    return cfg


# ---------------------------------------------------------------------------
# The switch itself
# ---------------------------------------------------------------------------

def test_the_default_carries_state_across_a_teleport():
    """The 2026-08-12 change. A run that says nothing gets the new behaviour."""
    assert EnvConfig().reset_state_on_teleport is False


def test_the_flag_reaches_the_contract_for_sites_that_teleport():
    on = episode.contract_for("training_rollout", reset_state=True)
    off = episode.contract_for("training_rollout", reset_state=False)
    assert on.teleport and on.reset_state is True
    assert off.teleport and off.reset_state is False
    # Everything else about the contract is untouched.
    for clause in ("reward", "store_target_is_goal", "move_ignored", "teleport"):
        assert getattr(on, clause) == getattr(off, clause)


def test_a_site_that_never_teleports_is_unaffected():
    """C5 is unreachable without C4, and the contract forbids the combination
    outright -- so the switch must not be able to create it."""
    for site in ("evaluate_navigation", "evaluate_exploration"):
        for flag in (True, False):
            c = episode.contract_for(site, reset_state=flag)
            assert c.reset_state is False, f"{site} gained a reset without a teleport"


def test_reset_rows_is_the_same_rule_resolve_at_goal_applies():
    """Two derivations of one contract clause is how they drift. The collector
    holds only the pre-step mask, so it needs the rule on its own -- but the
    same rule."""
    mask = np.array([True, False, True, True])
    for site in ("training_rollout", "evaluate_navigation"):
        for flag in (True, False):
            c = episode.contract_for(site, reset_state=flag)
            full = episode.resolve_at_goal(
                mask, c, goal_reward=1.0, time_penalty=0.01).reset_state
            assert np.array_equal(episode.reset_rows(mask, c), full)


# ---------------------------------------------------------------------------
# Training: the path the goldens never reached
# ---------------------------------------------------------------------------

def _rollout(flag: bool):
    cfg = _cfg(flag)
    cfg.steps_per_rollout = 40
    cfg.batch_envs = 4
    col, agent, vh = make_collector(cfg, 8, seed=0)
    env = make_env(cfg.env, "discrete", seed=100)
    torch.manual_seed(0)
    np.random.seed(0)
    hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(cfg.batch_envs)]
    return col.collect_rollout(env, agent, hops, allow_store=True, update_idx=1)


def test_the_training_rollout_honours_the_switch():
    """The collector derived its reset mask from `goal_reached` directly, which
    happened to agree with TRAINING and would have ignored the flag entirely."""
    on, off = _rollout(True), _rollout(False)
    assert int(on.goal_reached.sum()) > 0, (
        "no teleport occurred, so this cannot show the switch doing anything")
    assert not torch.equal(on.move_actions, off.move_actions), (
        "the training rollout is identical either way -- the flag is not "
        "reaching the collector")


def test_the_training_rollout_is_identical_when_nothing_teleports(monkeypatch):
    """The switch is about the episode boundary and nothing else.

    With `goals_active=False` the agent still stands on the goal cell -- the
    at-goal indicator the store head keys on is recorded either way -- but the
    contract makes it a non-event, so no row is ever reset and the two settings
    must produce the same rollout.
    """
    import hopfield_nav.rollout.collector as collector_mod

    def go(flag):
        cfg = _cfg(flag)
        cfg.env.goals_active = False
        cfg.steps_per_rollout = 20
        cfg.batch_envs = 2
        col, agent, vh = make_collector(cfg, 8, seed=0)
        env = make_env(cfg.env, "discrete", seed=100)

        reset_rows_seen = []
        real = episode.reset_rows
        monkeypatch.setattr(
            collector_mod.episode, "reset_rows",
            lambda m, c: (lambda r: (reset_rows_seen.append(int(r.sum())), r)[1])(
                real(m, c)))
        torch.manual_seed(0)
        np.random.seed(0)
        hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(2)]
        out = col.collect_rollout(env, agent, hops, allow_store=True,
                                  update_idx=1)
        monkeypatch.undo()
        return out, sum(reset_rows_seen)

    (on, on_resets), (off, off_resets) = go(True), go(False)
    assert int(on.goal_reached.sum()) > 0, (
        "the agent never stood on the goal, so 'no teleport' is trivially true "
        "and this proves nothing")
    assert on_resets == off_resets == 0, "a row reset with goals inactive"
    assert torch.equal(on.move_actions, off.move_actions)
    assert torch.equal(on.rewards, off.rewards)


# ---------------------------------------------------------------------------
# Evaluation: observed on the hidden state, since the metrics do not move
# ---------------------------------------------------------------------------

def _h_after_each_goal(flag: bool) -> list[float]:
    """|h| on the step following each goal-reach, through `evaluate_realistic`."""
    cfg = _cfg(flag)
    _c, agent, vh = make_collector(cfg, 8, seed=0)
    vh.env_offsets = [(0, 0), (8, 0)]
    envs = [make_env(cfg.env, "discrete", seed=100 + i) for i in range(2)]

    seen: list[float] = []
    state = {"armed": False}
    real_step, real_at_goal = ev.agent_step, ev.at_goal

    def step_spy(agent_, env, off, vh_, hop, h_rnn, cfg_, dev, **kw):
        if state["armed"]:
            seen.append(0.0 if h_rnn is None else float(h_rnn.abs().sum()))
            state["armed"] = False
        return real_step(agent_, env, off, vh_, hop, h_rnn, cfg_, dev, **kw)

    def at_goal_spy(e):
        r = real_at_goal(e)
        if bool(np.asarray(r).any()):
            state["armed"] = True
        return r

    ev.agent_step, ev.at_goal = step_spy, at_goal_spy
    try:
        torch.manual_seed(0)
        np.random.seed(0)
        ev.evaluate_realistic(agent, envs, vh, vh.env_offsets, cfg,
                              torch.device("cpu"), steps_per_env=120, seed=13)
    finally:
        ev.agent_step, ev.at_goal = real_step, real_at_goal
    return seen


def test_evaluation_zeroes_the_hidden_state_only_when_asked():
    on = _h_after_each_goal(True)
    off = _h_after_each_goal(False)
    assert on and len(on) == len(off), "no goal was reached; the test is vacuous"
    assert all(v == 0.0 for v in on), (
        "reset_state_on_teleport=True left the hidden state standing")
    assert any(v > 0.0 for v in off), (
        "reset_state_on_teleport=False zeroed the hidden state anyway")
