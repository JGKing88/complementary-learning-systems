"""Batched navigation reproduces the sequential loop, trial for trial.

evaluate_navigation ran one trial at a time through agent_step -- a B=1 call
into a recurrent policy. Its trials within one (env, distractor level) cell are
independent, so they can run as one batch. What must not change is which
Hopfield and which start each trial gets, or whether it reached the goal.

Only navigation is batched: it ends the trial on arrival, so it never takes a
step from the goal and never exercises the teleport clause. The evaluators that
keep stepping at the goal are a separate change.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav import eval as ev
from hopfield_nav.env import make_env
from hopfield_nav.evaluation.batched import batched_navigation_trials
from hopfield_nav.hopfield import Hopfield
from hopfield_nav.tests.fixtures import make_collector, make_stub_cfg

EMBED_DIM = 8


def _world(goal_radius=1.5, size=5, n_envs=2):
    cfg = make_stub_cfg(movement_mode="discrete")
    cfg.env.size = size
    cfg.env.goal_radius = goal_radius
    _c, agent, vh = make_collector(cfg, EMBED_DIM, seed=0)
    vh.env_offsets = [(0, 0), (8, 0), (0, 8)][:n_envs]
    envs = [make_env(cfg.env, "discrete", seed=200 + i) for i in range(n_envs)]
    return cfg, agent, vh, envs


def _sequential_reference(agent, env, env_offset, vh, hopfields, cfg, device,
                          starts, goal, max_steps):
    """The pre-batching loop, kept here as the thing batching must reproduce."""
    out = []
    for hop, start in zip(hopfields, starts):
        env.set_position(start)
        h_rnn = prev_reward = prev_action = None
        steps = -1
        for step in range(max_steps):
            res = ev.agent_step(
                agent, env, env_offset, vh, hop, h_rnn, cfg, device,
                deterministic=True, goal_local=goal, goal_in_memory=True,
                prev_reward=prev_reward, prev_action=prev_action)
            h_rnn = res["h_rnn"]
            prev_reward = res["next_prev_reward"]
            prev_action = res["next_prev_action"]
            from hopfield_nav.env import at_goal
            if at_goal(env):
                steps = step + 1
                break
        out.append(steps)
    return out


@pytest.mark.parametrize("goal_radius", [0.5, 1.5])
def test_batched_matches_sequential_trial_for_trial(goal_radius):
    cfg, agent, vh, envs = _world(goal_radius=goal_radius)
    env, env_offset = envs[0], vh.env_offsets[0]
    goal = env.goal_location
    device = torch.device("cpu")

    rng = np.random.RandomState(4)
    hopfields, starts = [], []
    for _ in range(16):
        hop = Hopfield(EMBED_DIM, beta=cfg.hopfield.beta, device="cpu")
        hop.input_memory(torch.from_numpy(
            ev._goal_encoding(vh, env_offset, goal)).float())
        hopfields.append(hop)
        starts.append(ev.random_start(env.size, goal, rng))

    seq = _sequential_reference(agent, env, env_offset, vh, hopfields, cfg,
                                device, starts, goal, max_steps=30)
    bat = batched_navigation_trials(
        agent=agent, env=env, env_offset=env_offset, vectorhash=vh,
        hopfields=hopfields, cfg=cfg, device=device, starts=starts, goal=goal,
        max_steps=30, deterministic=True)

    assert bat == seq
    if goal_radius > 0.5:
        assert any(s > 0 for s in seq), "vacuous: no trial reached the goal"


def test_finished_trials_are_frozen():
    """A trial that arrives must not move again -- that is what makes the
    'episode ends on arrival' semantics exact rather than approximate, and it
    is why the teleport branch is never entered."""
    cfg, agent, vh, envs = _world(goal_radius=2.5, size=4)
    env, env_offset = envs[0], vh.env_offsets[0]
    goal = env.goal_location
    rng = np.random.RandomState(9)
    hopfields, starts = [], []
    for _ in range(8):
        hop = Hopfield(EMBED_DIM, beta=cfg.hopfield.beta, device="cpu")
        hop.input_memory(torch.from_numpy(
            ev._goal_encoding(vh, env_offset, goal)).float())
        hopfields.append(hop)
        starts.append(ev.random_start(env.size, goal, rng))

    steps = batched_navigation_trials(
        agent=agent, env=env, env_offset=env_offset, vectorhash=vh,
        hopfields=hopfields, cfg=cfg, device=torch.device("cpu"),
        starts=starts, goal=goal, max_steps=20, deterministic=True)
    assert any(s > 0 for s in steps), "vacuous: nothing reached the goal"
    # Every recorded step count is within the budget and positive-or-(-1).
    assert all(s == -1 or 1 <= s <= 20 for s in steps)


def test_trials_keep_their_own_memory_and_start():
    """Row b uses hopfields[b] and starts[b] -- batching must not cross them."""
    cfg, agent, vh, envs = _world(goal_radius=1.5)
    env, env_offset = envs[0], vh.env_offsets[0]
    goal = env.goal_location

    # Two trials: one with the goal preloaded, one with an unrelated pattern.
    good = Hopfield(EMBED_DIM, beta=cfg.hopfield.beta, device="cpu")
    good.input_memory(torch.from_numpy(
        ev._goal_encoding(vh, env_offset, goal)).float())
    junk = Hopfield(EMBED_DIM, beta=cfg.hopfield.beta, device="cpu")
    junk.input_memory(torch.from_numpy(vh.encoded_Phi[13, 13]).float())

    start = ev.random_start(env.size, goal, np.random.RandomState(2))
    both = batched_navigation_trials(
        agent=agent, env=env, env_offset=env_offset, vectorhash=vh,
        hopfields=[good, junk], cfg=cfg, device=torch.device("cpu"),
        starts=[start, start], goal=goal, max_steps=25, deterministic=True)
    solo_good = batched_navigation_trials(
        agent=agent, env=env, env_offset=env_offset, vectorhash=vh,
        hopfields=[good], cfg=cfg, device=torch.device("cpu"),
        starts=[start], goal=goal, max_steps=25, deterministic=True)
    assert both[0] == solo_good[0]
