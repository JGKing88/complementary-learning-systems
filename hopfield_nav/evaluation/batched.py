"""Batched evaluation rollouts: one forward pass per step for all trials.

The evaluators ran one trial at a time through ``eval.agent_step``, which is a
B=1 call into a recurrent policy. With the stock ``run_eval_all.sh`` settings --
5 val envs x 3 distractor levels x 32 trials x up to 200 steps -- navigation
alone is up to ~96,000 batch-1 forward passes per checkpoint. Running the
trials of one (env, distractor level) cell together turns that into ~600.

Only ``evaluate_navigation`` is batched here, and only because its contract
allows it: it ends the trial the moment the agent arrives, so it never takes a
step *from* the goal and never exercises C3/C4 (see
``world/episode.SITE_CONTRACTS``). The evaluators that keep stepping at the
goal need reward-without-teleport, which ``VecEnv`` could not express until
phase 5a; batching those is a separate change with its own equivalence work,
because it alters what they measure if done carelessly.

Exactness
---------
Under a deterministic policy this reproduces the sequential loop trial for
trial: the setup draws from the caller's RNG in the original order, and each
row's per-trial Hopfield, start position and step budget are unchanged. Values
can differ in the last float32 ULP, because batched matmuls accumulate in a
different order -- which is why the goldens pin per-trial outcomes rather than
aggregate floats.

Under a *stochastic* policy the equivalence is distributional, not exact: one
batched sample draws differently from B sequential ones. That is inherent to
batching, not a defect here.
"""
from __future__ import annotations

import numpy as np
import torch

from ..policy import channels
from ..rollout import signal as signal_ops
from ..world.env import GridEnv, at_goal
from hopfield import Hopfield
from ..world.vec_env import make_vec
from ..world import episode


@torch.no_grad()
def batched_navigation_trials(
    *,
    agent,
    env: GridEnv,
    env_offset: tuple[int, int],
    vectorhash,
    hopfields: list[Hopfield],
    cfg,
    device: torch.device,
    starts: list[tuple[int, int]],
    goal: tuple[int, int],
    max_steps: int,
    deterministic: bool = True,
    action_temperature: float = 1.0,
) -> list[int]:
    """Run one navigation trial per entry of ``hopfields``, in parallel.

    Each trial starts at its own position with its own preloaded Hopfield, and
    finishes the first step after which it stands on the goal. Finished trials
    are frozen: they are excluded from the environment step, so the at-goal
    branch of the contract is never entered and the sequential semantics -- the
    episode simply ends -- are preserved exactly.

    Returns, per trial, the number of steps taken to reach the goal, or -1 if
    the step budget ran out.
    """
    B = len(hopfields)
    assert len(starts) == B

    contract = episode.contract_for("evaluate_navigation")
    episode.require_single_env_support(contract, "batched_navigation_trials")

    vec = make_vec(env, B, cfg.agent.movement_mode, cfg.env.continuous_scale,
                   continuous_normalize=cfg.env.continuous_normalize,
                   reset=False)
    vec.set_positions(starts)

    input_specs = channels.channel_specs(
        cfg.agent, vectorhash.encoded_Phi.shape[2], cfg.env.observation_size)
    signal_dim = channels.signal_width(cfg.agent)
    prev_action_dim = channels.prev_action_width(cfg.agent)

    h_rnn = None
    prev_reward_t = torch.zeros(B, 1, device=device)
    prev_action_t = torch.zeros(B, prev_action_dim, device=device)
    steps_to_goal = [-1] * B
    active = np.ones(B, dtype=bool)

    for step in range(max_steps):
        positions = vec.positions()
        at_goal_mask = at_goal(vec)
        current_reward = (
            np.where(at_goal_mask, cfg.env.goal_reward, -cfg.env.time_penalty)
            if vec.goals_active
            else np.full(B, -cfg.env.time_penalty, dtype=np.float32)
        ).astype(np.float32)

        embeddings_np = vectorhash.get_encoded_state(positions, env_offset)
        embeddings = torch.from_numpy(embeddings_np).float().to(device)

        if cfg.agent.input_hopfield_signal:
            sig_t, q, _mask, _W = signal_ops.hopfield_signal_at(
                vectorhash, cfg, embeddings_np, embeddings, positions,
                env_offset, hopfields, False, device, embeddings.shape[1])
            if (cfg.agent.input_hopfield_raw
                    and cfg.agent.hopfield_mode != "discrete"):
                hop_signal = torch.from_numpy(q.astype(np.float32)).to(device)
            else:
                hop_signal = sig_t
        else:
            hop_signal = torch.zeros(B, signal_dim, device=device)

        values = {
            "current_reward": torch.from_numpy(current_reward).to(device).unsqueeze(1),
            "prev_reward": prev_reward_t,
            "encoded_state": embeddings,
            "hopfield_signal": hop_signal,
            "prev_action": prev_action_t,
            # Navigation preloads the goal into every trial's Hopfield, so the
            # bit is True from step zero -- matching agent_step's hardcoded
            # goal_in_memory=True on this path.
            "goal_in_memory": torch.ones(B, 1, device=device),
        }
        if cfg.agent.input_sensory:
            values["sensory"] = torch.from_numpy(
                vec.obs_batch()).float().to(device)
        if (cfg.agent.input_hopfield_multistep
                and cfg.agent.hopfield_mode == "continuous"):
            W = vectorhash.gram_schmidt_projection(positions, env_offset)
            msq = signal_ops.multistep_q(
                vectorhash, cfg, embeddings_np, embeddings, hopfields, False,
                W, cfg.agent.input_hopfield_multistep, embeddings.shape[1],
                device)
            for s, q_s in msq.items():
                values[channels.multistep_name(s)] = torch.from_numpy(
                    q_s.astype(np.float32)).to(device)

        rnn_input = channels.build_policy_input(
            input_specs, values, batch_size=B).unsqueeze(1)
        result = agent.get_action_and_value(
            rnn_input, h_rnn, deterministic=deterministic,
            action_temperature=action_temperature)
        h_rnn = result["h_next"]

        if cfg.agent.movement_mode == "discrete":
            actions = result["move_action"].cpu().numpy().astype(int)
            prev_action_t = torch.nn.functional.one_hot(
                result["move_action"].long().view(-1), num_classes=4).float()
        else:
            actions = result["move_action"].cpu().numpy()
            prev_action_t = result["move_action"].float().view(B, -1)
        prev_reward_t = torch.from_numpy(current_reward).to(device).unsqueeze(1)

        # Step only the live trials. A finished one must not move again --
        # that is what makes "the episode ends on arrival" exact rather than
        # approximate, and it is why the teleport branch is never reached.
        active_idx = np.nonzero(active)[0]
        if active_idx.size == 0:
            break
        vec.step_batch(actions[active_idx], indices=active_idx,
                       contract=contract)

        reached = at_goal(vec)
        for b in active_idx:
            if reached[b]:
                steps_to_goal[b] = step + 1
                active[b] = False

    return steps_to_goal


__all__ = ["batched_navigation_trials"]
