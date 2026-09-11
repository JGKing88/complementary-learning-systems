"""Readout 2 for Experiment B: direction quality against experience (plan §4.4).

Run lifetimes on one env -- `n_lifetimes` independent rows, each a sequence
of episodes with the hidden state carried across them and a fresh start AND
a fresh goal at every goal-reach -- and at every step score the policy's own
action against the teacher, `normalize(g - p)`. Every score lands at an
(episode index, step index) coordinate. The two marginals:

  by episode   mean(score | episode = k)   -- with a fresh goal every episode
               this is exactly *goals seen so far in this env*: the
               map-learning curve, the one that decides B.
  by step      mean(score | step = t)      -- within-episode, episode-local.

The full 2-D table is kept too, with counts, because the marginals blur the
three shapes worth telling apart: flat in both (memoryless-equivalent);
rises with step and resets each episode (episode-local history); rises with
episode (a map of the env accumulating across goals).

The teacher scores only. The student acts on its own policy every step, and
the action is **sampled** (`deterministic=False`) -- BC fits a Gaussian's
mean to the teacher's conditional mean, and in an env where the policy is
uncertain that mean collapses toward zero and the mean action scores a
policy that barely moves (§5.2 measured 2-4x from this). Readout 1 stays
deterministic because it is a function evaluation, not a rollout.

Episode 0, step 0 is readout 1's state by construction -- same h = 0, same
prev_action = 0, same first observation -- so that cell of the table has to
agree with the static table's entry for the matching quadrant (gate C16).
"""
from __future__ import annotations

import numpy as np
import torch

from ..evaluation.goal_pairs import angular_error_deg, optimal_action_set
from ..rollout.rnn import (
    build_rnn_input, goal_channel_vec, goal_sensory_vec, grid_state_vec,
    prev_action_channel, sensory_vec, xy_vec)
from ..world.env import at_goal
from ..world.spec import CellSets
from ..world.vec_env import make_vec


def _cell_ids(cells, size):
    return np.array(sorted(x * size + y for x, y in cells), dtype=np.int64)


@torch.no_grad()
def evaluate_lifetime_direction(
    env, agent, *, cells: CellSets, n_lifetimes: int, n_episodes: int,
    max_steps: int, device, sgb=None, env_offset=None,
    continuous_scale: float = 1.0, continuous_normalize: bool = True,
    deterministic: bool = False, seed: int = 0,
) -> dict:
    """The (episode x step) score table for one env, plus its marginals."""
    cfg = agent.cfg
    mm = cfg.movement_mode
    S = env.size
    vec = make_vec(env, n_lifetimes, mm, continuous_scale, continuous_normalize, reset=False)
    vec._rng = np.random.RandomState(seed)
    # Starts from the training start set, goals from the training goal set --
    # the same pair distribution B trained on, so the curve measures
    # experience of the ENV, not of an unfamiliar pair distribution.
    vec.set_goal_pool(cells.goal_train)
    start_ids = _cell_ids(cells.start_train, S)
    vec.reset_all()
    starts = start_ids[vec._rng.randint(len(start_ids), size=n_lifetimes)]
    vec.set_positions(np.stack([starts // S, starts % S], 1).astype(
        np.float64 if mm == "continuous" else np.int32))
    # set_positions may have landed a row on its goal; re-roll those starts.
    for _ in range(10):
        on = at_goal(vec)
        if not on.any():
            break
        idx = np.where(on)[0]
        st = start_ids[vec._rng.randint(len(start_ids), size=len(idx))]
        pos = vec.positions().astype(np.float64 if mm == "continuous" else np.int32)
        pos[idx] = np.stack([st // S, st % S], 1)
        vec.set_positions(pos)

    B = n_lifetimes
    ep_idx = np.zeros(B, dtype=np.int64)
    steps_in_ep = np.zeros(B, dtype=np.int64)
    table = np.zeros((n_episodes, max_steps + 1), dtype=np.float64)
    count = np.zeros((n_episodes, max_steps + 1), dtype=np.int64)

    want_sensory = getattr(cfg, "input_sensory", True)
    sensory_mode = getattr(cfg, "sensory_mode", "ego")
    want_goal_gs = getattr(cfg, "input_goal_grid_state", False)
    goal_sens_mode = getattr(cfg, "goal_sensory", "none")
    want_xy = getattr(cfg, "input_xy_state", False)

    h = None
    prev_action_np = None
    prev_reward_np = np.zeros(B, dtype=np.float32)
    budget = n_episodes * (max_steps + 1)
    for _ in range(budget):
        live = ep_idx < n_episodes
        if not live.any():
            break
        reached = at_goal(vec) & live
        timed_out = (steps_in_ep >= max_steps) & live & ~reached
        closing = np.where(reached | timed_out)[0]
        if len(closing) > 0:
            ep_idx[closing] += 1
            steps_in_ep[closing] = 0
            still = closing[ep_idx[closing] < n_episodes]
            if len(still) > 0:
                vec.reset_indices(still)       # fresh start AND fresh goal; h kept
        live = ep_idx < n_episodes
        if not live.any():
            break

        positions = vec.positions()
        goals = vec._goals
        if not want_sensory:
            sensory = None
        elif sensory_mode == "omni":
            sensory = sensory_vec(vec, positions, "omni")
        else:
            sensory = vec.obs_batch().astype(np.float32)
        prev_act_ch = (prev_action_channel(prev_action_np, mm, B)
                       if cfg.input_prev_action else None)
        grid_state = (grid_state_vec(positions, env_offset, sgb)
                      if (cfg.input_grid_state and sgb is not None and env_offset is not None) else None)
        goal_gs = (grid_state_vec(goals, env_offset, sgb)
                   if (want_goal_gs and sgb is not None and env_offset is not None) else None)
        goal_sens = goal_sensory_vec(vec, goals, goal_sens_mode) if goal_sens_mode != "none" else None
        xy_state = xy_vec(positions, S) if want_xy else None
        goal_vec = (goal_channel_vec(positions, goals, S, cfg.goal_channel)
                    if getattr(cfg, "goal_channel", "none") != "none" else None)
        x = build_rnn_input(sensory, prev_act_ch, prev_reward_np, grid_state, cfg, device,
                            goal_vec=goal_vec, xy_state=xy_state,
                            goal_grid_state=goal_gs, goal_sensory=goal_sens)
        out = agent.act(x, h, deterministic=deterministic)
        h_next = out["h_next"]
        # Dead rows: do not advance h (the existing in-context evaluator does,
        # on a stale observation; harmless there, wrong here).
        if h is not None and h_next is not None:
            live_t = torch.from_numpy(live).to(h_next.device).view(1, -1, 1)
            h = torch.where(live_t, h_next, h)
        else:
            h = h_next
        action = out["move_action"].cpu().numpy()

        # Score the policy's own action against the teacher, live rows only.
        p = positions[:, 0] * S + positions[:, 1]
        g = goals[:, 0] * S + goals[:, 1]
        if mm == "continuous":
            d = np.stack([goals[:, 0] - positions[:, 0], goals[:, 1] - positions[:, 1]], 1).astype(np.float32)
            n = np.linalg.norm(d, axis=1, keepdims=True)
            u = d / np.maximum(n, 1e-8)
            score = angular_error_deg(action.astype(np.float32), u)
        else:
            opt = optimal_action_set(p, g, S)
            score = opt[np.arange(B), action.astype(np.int64)].astype(np.float64)
        for b in np.where(live & ~reached)[0]:      # at-goal steps are not scored
            e, t = ep_idx[b], min(steps_in_ep[b], max_steps)
            table[e, t] += score[b]
            count[e, t] += 1

        idx = np.where(live)[0]
        rewards_full = np.zeros(B, dtype=np.float32)
        if len(idx) > 0:
            r, _, _ = vec.step_batch(action[idx], indices=idx)
            rewards_full[idx] = r
        steps_in_ep[live] += 1
        prev_action_np = action
        prev_reward_np = rewards_full

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_table = np.where(count > 0, table / np.maximum(count, 1), np.nan)
        by_episode = np.where(count.sum(1) > 0, table.sum(1) / np.maximum(count.sum(1), 1), np.nan)
        by_step = np.where(count.sum(0) > 0, table.sum(0) / np.maximum(count.sum(0), 1), np.nan)
    return {
        "by_episode": by_episode.tolist(),
        "by_step": by_step.tolist(),
        "table": mean_table.tolist(),
        "count": count.tolist(),
        "ep0_step0": float(mean_table[0, 0]) if count[0, 0] > 0 else None,
        "n_lifetimes": int(n_lifetimes), "n_episodes": int(n_episodes),
        "max_steps": int(max_steps), "metric": "deg" if mm == "continuous" else "acc",
    }


def aggregate_lifetimes(results: list[dict]) -> dict:
    """Mean of by_episode / by_step over envs (nan-aware), plus the spread."""
    be = np.array([r["by_episode"] for r in results], dtype=np.float64)
    bs = np.array([r["by_step"] for r in results], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return {
            "by_episode": np.nanmean(be, axis=0).tolist(),
            "by_episode_std": np.nanstd(be, axis=0).tolist(),
            "by_step": np.nanmean(bs, axis=0).tolist(),
            "ep0_step0": float(np.nanmean([r["ep0_step0"] for r in results if r["ep0_step0"] is not None])),
            "n_envs": len(results), "metric": results[0]["metric"],
        }
