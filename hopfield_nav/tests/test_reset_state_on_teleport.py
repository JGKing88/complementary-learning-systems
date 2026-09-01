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


# ---------------------------------------------------------------------------
# The enrichment buffers, which must share the hidden state's fate
# ---------------------------------------------------------------------------

def _rollout_capturing_inputs(flag: bool):
    """Run a teleporting rollout and capture what the policy was shown.

    `prev_displacement` is an observation channel, so the only way to see
    whether it survived a teleport is to read the policy input on the step
    after one.
    """
    cfg = _cfg(flag)
    cfg.steps_per_rollout = 40
    cfg.batch_envs = 4
    cfg.agent.input_prev_reward = True
    cfg.agent.input_prev_action = True
    cfg.agent.input_prev_displacement = True
    col, agent, vh = make_collector(cfg, 8, seed=0)
    env = make_env(cfg.env, "discrete", seed=100)
    torch.manual_seed(0)
    np.random.seed(0)
    hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(cfg.batch_envs)]

    seen = []
    real = agent.get_action_and_value

    def _spy(x, h=None, **kw):
        seen.append(x.detach().clone())
        return real(x, h, **kw)

    agent.get_action_and_value = _spy
    roll = col.collect_rollout(env, agent, hops, allow_store=True,
                               update_idx=1)
    # (T, B, D) -- one (B, 1, D) capture per step
    return roll, torch.cat(seen, dim=1).permute(1, 0, 2), cfg


def _slice_of(cfg, name):
    from hopfield_nav.policy import channels
    off = 0
    for spec in channels.channel_specs(cfg.agent, embed_dim=8,
                                       sensory_dim=cfg.env.observation_size):
        if spec.name == name:
            return slice(off, off + spec.width)
        off += spec.width
    raise AssertionError(f"{name} not in the channel layout")


def test_a_teleporting_step_realizes_no_displacement():
    """Why prev_disp_t's reset is belt-and-braces rather than a fix.

    The TRAINING contract sets `move_ignored`, so an at-goal row is skipped by
    the move loop (`vec_env.py:199`) and the teleport that follows never writes
    to `moved`. Its realized displacement is therefore *already* exactly zero,
    which is why removing prev_disp_t from the reset block was unobservable
    even with the switch on -- there was nothing non-zero to carry.

    This is the invariant that makes it so. If `move_ignored` is ever dropped
    from the contract, a teleport starts producing a displacement, and the
    reset in the collector stops being redundant -- so this test failing is
    the signal to go look at it.
    """
    cfg = _cfg(True)
    cfg.steps_per_rollout = 40
    cfg.batch_envs = 4
    col, agent, vh = make_collector(cfg, 8, seed=0)
    env = make_env(cfg.env, "discrete", seed=100)
    torch.manual_seed(0)
    np.random.seed(0)
    hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(cfg.batch_envs)]

    from hopfield_nav.world.vec_env import VecEnv
    vec = VecEnv(env, batch_size=cfg.batch_envs)
    vec.reset_all()
    contract = episode.contract_for("training_rollout", reset_state=True)
    assert contract.move_ignored and contract.teleport

    rng = np.random.RandomState(0)
    seen_teleport = 0
    for _ in range(60):
        acts = rng.randint(0, 4, size=cfg.batch_envs)
        _, reached, _ = vec.step_batch(
            acts, indices=np.arange(cfg.batch_envs), contract=contract)
        disp = vec.last_displacement()
        for b in np.nonzero(np.asarray(reached, dtype=bool))[0]:
            assert not disp[b].any(), (
                f"a teleporting row realized displacement {disp[b].tolist()}; "
                f"prev_disp_t's reset is now load-bearing, not redundant")
            seen_teleport += 1
    assert seen_teleport > 0, "no teleport occurred; this test proves nothing"



def test_all_three_enrichment_buffers_agree_after_a_teleport():
    """Whatever the rule is, it must be the same rule for all of them.

    Discriminating for `prev_reward` and `prev_action` -- removing either
    reset from the collector fails this. NOT discriminating for
    `prev_displacement`, which is already zero after a teleport for the
    reason `test_a_teleporting_step_realizes_no_displacement` pins; it is
    included so the three stay coupled if that ever changes.
    """
    roll, obs, cfg = _rollout_capturing_inputs(True)
    reached = roll.goal_reached.cpu().numpy().astype(bool)
    slices = {n: _slice_of(cfg, n)
              for n in ("prev_reward", "prev_action", "prev_displacement")}
    for b in range(reached.shape[0]):
        for t in np.nonzero(reached[b])[0]:
            if t + 1 >= obs.shape[0]:
                continue
            for name, sl in slices.items():
                v = obs[t + 1, b, sl]
                assert torch.allclose(v, torch.zeros_like(v)), (
                    f"{name} survived a teleport at row {b} step {t}")


def test_evaluation_zeroes_the_hidden_state_only_when_asked():
    on = _h_after_each_goal(True)
    off = _h_after_each_goal(False)
    assert on and len(on) == len(off), "no goal was reached; the test is vacuous"
    assert all(v == 0.0 for v in on), (
        "reset_state_on_teleport=True left the hidden state standing")
    assert any(v > 0.0 for v in off), (
        "reset_state_on_teleport=False zeroed the hidden state anyway")
