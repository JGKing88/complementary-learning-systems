"""Does a frozen recurrent policy adapt to a new environment in-context?

This is plan section 5.2, and it is the only control that competes with the
Hopfield store on the store's own terms: **no weight updates at all**. Both
models get one environment they have never seen, and neither is allowed to
learn from it in the ordinary sense. The store writes one Hebbian outer product
and is done. The RNN gets nothing but its own recurrent activity.

The measurement is a *lifetime*: one environment, N episodes back to back, the
hidden state carried across every episode boundary and zeroed only at the start
of the lifetime. The goal is never observed and never moves. So an agent that
solves episode 5 faster than episode 1 can only be doing it by remembering
where the goal was, in activations. Episode 1 is the floor -- nothing has been
seen yet, so it measures blind search -- and the *slope* across episodes is the
adaptation.

Reading the result:

- **Rising curve.** The RNN can hold an environment in activations. Forgetting
  is then impossible by construction for both models, the retention comparison
  stops being the interesting one, and the paper's framing has to change.
- **Flat curve.** Activation memory cannot do this job, which is the strongest
  positive result available -- it is the one comparison a referee cannot answer
  with "you just needed a bigger buffer."

But a flat *mean* curve is ambiguous on its own, because it pools lifetimes
that found the goal with lifetimes that never did. If the agent only stumbles
onto the goal in a tenth of first episodes, nine tenths of the average is
measuring blind search and the flatness says nothing about memory. So the
sharper statistic is `memory_lift`: among consecutive episode pairs, the
success rate when the previous episode *found* the goal minus the rate when it
did not. An agent holding the goal in its activations must do better having
just been there; one searching blind cannot tell the difference. That is the
number to read.

Either way it has to be run, which is why it is here rather than in future work.

**One confound, named rather than hidden.** The two arms do not receive equal
supervision. A lifetime rollout teleports its reachers and keeps collecting, so
more of its steps are supervised than in an episodic rollout, where a reacher
is frozen and masked out for the remainder. The lifetime arm therefore gets
more gradient signal per update and tends to be the better policy outright --
which is visible in pretraining, not subtle.

So **the comparison between arms is the adaptation slope, not the absolute
level**. `adaptation` (last episode minus first) is scale-relative and is the
number to read; comparing the arms' raw success rates would partly be comparing
how much supervision each received. A cleaner design would match total
supervised steps rather than updates, and that is the right thing to do if this
measurement ends up load-bearing.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from ..policy.agent_rnn import RNNAgent
from ..rollout.rnn import (
    build_rnn_input, goal_channel_vec, grid_state_vec, prev_action_channel)
from ..world.env import GridEnv, at_goal
from ..world.vec_env import make_vec


@torch.no_grad()
def evaluate_in_context(
    env: GridEnv,
    agent: RNNAgent,
    n_lifetimes: int,
    n_episodes: int,
    max_steps: int,
    device: torch.device,
    deterministic: bool = True,
    continuous_scale: float = 1.0,
    continuous_normalize: bool = False,
    sgb: np.ndarray | None = None,
    env_offset: tuple[int, int] | None = None,
    goal_visible_episodes: int = -1,
) -> dict:
    """Run `n_lifetimes` independent lifetimes of `n_episodes` episodes each.

    Weights are never touched. Within a lifetime the hidden state persists
    across episode boundaries; across lifetimes it is reset, so the lifetimes
    are independent samples of "meeting this environment for the first time".

    Returns per-episode-index success rates and mean steps-to-goal, plus the
    slope from the first episode to the last -- the adaptation, if any.
    """
    movement_mode = agent.cfg.movement_mode
    vec = make_vec(env, n_lifetimes, movement_mode, continuous_scale,
                   continuous_normalize, reset=False)
    vec.reset_all()

    B = n_lifetimes
    goal = (int(vec._goal[0]), int(vec._goal[1]))
    ep_idx = np.zeros(B, dtype=np.int64)        # which episode each lifetime is on
    steps_in_ep = np.zeros(B, dtype=np.int64)
    success = np.zeros((B, n_episodes), dtype=bool)
    steps_to = np.full((B, n_episodes), np.nan)

    h = None
    prev_action_np: np.ndarray | None = None
    prev_reward_np = np.zeros(B, dtype=np.float32)

    # A lifetime needs at most n_episodes * max_steps ticks; the +n_episodes
    # covers the tick each episode spends *standing on* the goal before it is
    # recorded and teleported.
    budget = n_episodes * (max_steps + 1)

    for _ in range(budget):
        live = ep_idx < n_episodes
        if not live.any():
            break

        reached = at_goal(vec) & live
        timed_out = (steps_in_ep >= max_steps) & live & ~reached

        for b in np.where(reached)[0]:
            success[b, ep_idx[b]] = True
            steps_to[b, ep_idx[b]] = steps_in_ep[b]
        # A closing episode -- reached or timed out -- rolls the lifetime on to
        # the next one from a fresh start. `h` is deliberately untouched: it is
        # the only channel by which anything can carry over, and zeroing it here
        # would silently turn this back into the ordinary episodic eval.
        closing = np.where(reached | timed_out)[0]
        if len(closing) > 0:
            ep_idx[closing] += 1
            steps_in_ep[closing] = 0
            still = closing[ep_idx[closing] < n_episodes]
            if len(still) > 0:
                vec.reset_indices(still)

        live = ep_idx < n_episodes
        if not live.any():
            break

        sensory = vec.obs_batch().astype(np.float32)
        positions = vec.positions()
        prev_act_ch = (prev_action_channel(prev_action_np, movement_mode, B)
                       if agent.cfg.input_prev_action else None)
        grid_state = (grid_state_vec(positions, env_offset, sgb)
                      if (agent.cfg.input_grid_state and sgb is not None
                          and env_offset is not None) else None)
        # The oracle channel, if this is a ceiling arm. `visible` is what makes
        # the episode-1-only control possible: show the goal while the lifetime
        # is on its first episode and withhold it afterwards, so the agent has
        # to *carry* the fact rather than rediscover it.
        goal_vec = None
        if getattr(agent.cfg, "goal_channel", "none") != "none":
            vis = (ep_idx < goal_visible_episodes
                   if goal_visible_episodes >= 0 else None)
            goal_vec = goal_channel_vec(positions, goal, env.size,
                                        agent.cfg.goal_channel, visible=vis)
        x = build_rnn_input(sensory, prev_act_ch, prev_reward_np, grid_state,
                            agent.cfg, device, goal_vec=goal_vec)
        out = agent.act(x, h, deterministic=deterministic)
        h = out["h_next"]
        action = out["move_action"].cpu().numpy()

        idx = np.where(live)[0]
        rewards_full = np.zeros(B, dtype=np.float32)
        if len(idx) > 0:
            r, _, _ = vec.step_batch(action[idx], indices=idx)
            rewards_full[idx] = r
        steps_in_ep[live] += 1

        prev_action_np = action
        prev_reward_np = rewards_full

    # --- the conditional test, which is far sharper than the mean curve ----
    #
    # The mean success-vs-episode curve pools lifetimes that found the goal
    # with lifetimes that never did, and if the agent only stumbles onto the
    # goal in a tenth of first episodes then nine tenths of the average is
    # measuring blind search. That makes a flat curve ambiguous: it could mean
    # "no memory", or it could mean "rarely had anything to remember".
    #
    # Conditioning removes the ambiguity. Among consecutive episode pairs,
    # compare the success rate when the *previous* episode found the goal
    # against when it did not. An agent carrying the goal in its activations
    # must do better in the first case; one that is searching blind cannot tell
    # the difference. `memory_lift` is that gap.
    nxt_given_hit, nxt_given_miss = [], []
    for b in range(B):
        for k in range(n_episodes - 1):
            (nxt_given_hit if success[b, k] else nxt_given_miss).append(
                bool(success[b, k + 1]))
    p_hit = float(np.mean(nxt_given_hit)) if nxt_given_hit else float("nan")
    p_miss = float(np.mean(nxt_given_miss)) if nxt_given_miss else float("nan")

    per_ep = success.mean(axis=0)
    # An episode index nobody solved is an all-NaN slice; nanmean warns and
    # returns nan, which is the right answer -- "no successful trial to average"
    # is not the same as zero steps. Suppress the warning, keep the nan.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_steps = np.nanmean(np.where(success, steps_to, np.nan), axis=0)
    return {
        "success_by_episode": [float(v) for v in per_ep],
        "mean_steps_by_episode": [None if np.isnan(v) else float(v)
                                  for v in mean_steps],
        "first_episode": float(per_ep[0]),
        "last_episode": float(per_ep[-1]),
        # The headline: how much better the agent gets at an environment it is
        # not allowed to learn. Zero means activation memory did nothing.
        "adaptation": float(per_ep[-1] - per_ep[0]),
        # The conditional test. `memory_lift` is the headline: how much more
        # often the next episode succeeds when the previous one found the goal.
        # Zero means the agent gets nothing from having just been there.
        "p_next_given_hit": p_hit,
        "p_next_given_miss": p_miss,
        "memory_lift": (float("nan") if (np.isnan(p_hit) or np.isnan(p_miss))
                        else p_hit - p_miss),
        "n_pairs_after_hit": len(nxt_given_hit),
        "n_pairs_after_miss": len(nxt_given_miss),
        "n_lifetimes": int(n_lifetimes),
        "n_episodes": int(n_episodes),
    }


def evaluate_in_context_all(
    envs: list[GridEnv], agent: RNNAgent, **kw
) -> dict[int, dict]:
    """`evaluate_in_context` over a list of held-out envs."""
    return {i: evaluate_in_context(e, agent, **kw) for i, e in enumerate(envs)}


__all__ = ["evaluate_in_context", "evaluate_in_context_all"]
