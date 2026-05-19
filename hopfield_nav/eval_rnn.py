"""Per-env nav eval for the vanilla-RNN baseline.

Standalone: does not call anything in eval.py / eval_all.py (those are
Hopfield-coupled). For each env, runs K parallel trials from random starts;
each trial succeeds the first step at which the agent's pre-step position
equals the goal. Reports nav_det = fraction of trials that reach goal within
max_steps.
"""
from __future__ import annotations

import numpy as np
import torch

from .agent_rnn import RNNAgent, compute_rnn_input_dim
from .config import RNNAgentConfig
from .env import GridEnv, at_goal
from .rollout_rnn import _action_to_prev_channel, _build_rnn_input, _grid_state_vec
from .vec_env import ContinuousVecEnv, VecEnv


def _make_vec(env: GridEnv, batch: int, movement_mode: str,
              continuous_scale: float = 1.0,
              continuous_normalize: bool = False) -> VecEnv | ContinuousVecEnv:
    if movement_mode == "continuous":
        return ContinuousVecEnv(
            env, batch_size=batch, scale=continuous_scale,
            normalize=continuous_normalize,
        )
    return VecEnv(env, batch_size=batch)


@torch.no_grad()
def evaluate_nav_one_env(
    env: GridEnv,
    agent: RNNAgent,
    n_trials: int,
    max_steps: int,
    device: torch.device,
    deterministic: bool = True,
    continuous_scale: float = 1.0,
    continuous_normalize: bool = False,
    sgb: np.ndarray | None = None,
    env_offset: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Run n_trials parallel trials from random starts.

    Returns nav_det (success rate), mean_steps_to_goal (over successful
    trials; nan if zero), mean_episode_return.
    """
    movement_mode = agent.cfg.movement_mode
    vec = _make_vec(env, n_trials, movement_mode, continuous_scale,
                    continuous_normalize=continuous_normalize)
    vec.reset_all()

    def _positions_float() -> np.ndarray:
        if movement_mode == "continuous":
            return vec.positions_continuous()
        return vec.positions().astype(np.float64)

    goal = (int(vec._goal[0]), int(vec._goal[1]))
    success = np.zeros(n_trials, dtype=bool)
    steps_to_goal = np.full(n_trials, fill_value=max_steps, dtype=np.int64)
    path_to_goal = np.full(n_trials, fill_value=np.nan, dtype=np.float64)
    path_acc = np.zeros(n_trials, dtype=np.float64)
    prev_positions_f = _positions_float()
    returns = np.zeros(n_trials, dtype=np.float32)

    h = None
    prev_action_np: np.ndarray | None = None
    prev_reward_np: np.ndarray | None = None

    for t in range(max_steps):
        sensory = vec.obs_batch().astype(np.float32)
        positions = vec.positions()
        positions_f = _positions_float()

        # Accumulate distance traveled in the prior step (zero at t=0). Frozen
        # for trials that have already succeeded so teleports don't pollute.
        disp = np.linalg.norm(positions_f - prev_positions_f, axis=1)
        not_yet = ~success
        path_acc[not_yet] += disp[not_yet]
        prev_positions_f = positions_f

        # Pre-step at-goal: count as success and freeze this trial's record.
        at_goal_mask = at_goal(vec)
        first_hit = at_goal_mask & ~success
        if first_hit.any():
            steps_to_goal[first_hit] = t
            path_to_goal[first_hit] = path_acc[first_hit]
        success |= at_goal_mask

        prev_act_ch = (
            _action_to_prev_channel(prev_action_np, movement_mode)
            if (agent.cfg.input_prev_action and prev_action_np is not None) else None
        )
        grid_state = (
            _grid_state_vec(positions, env_offset, sgb)
            if (agent.cfg.input_grid_state and sgb is not None
                and env_offset is not None) else None
        )
        x = _build_rnn_input(sensory, prev_act_ch, prev_reward_np, grid_state,
                             agent.cfg, device)
        out = agent.act(x, h, deterministic=deterministic)
        h = out["h_next"]
        student_action = out["move_action"].cpu().numpy()

        rewards, goal_reached, _ = vec.step_batch(student_action)
        returns += rewards

        # Reset RNN h on teleport so trials that already succeeded don't
        # carry stale memory into their next (wasted) steps.
        if h is not None and goal_reached.any():
            reset_mask = torch.from_numpy(goal_reached).to(device)
            h = h * (~reset_mask).view(1, -1, 1).float()

        prev_action_np = student_action
        prev_reward_np = rewards

    succ_idx = success.nonzero()[0]
    return {
        "nav_det": float(success.mean()),
        "mean_steps_to_goal": (
            float(steps_to_goal[succ_idx].mean()) if len(succ_idx) > 0 else float("nan")
        ),
        "mean_path_to_goal": (
            float(np.nanmean(path_to_goal[succ_idx])) if len(succ_idx) > 0 else float("nan")
        ),
        "mean_episode_return": float(returns.mean()),
    }


@torch.no_grad()
def evaluate_nav_all(
    envs: list[GridEnv],
    agent: RNNAgent,
    n_trials: int,
    max_steps: int,
    device: torch.device,
    deterministic: bool = True,
    continuous_scale: float = 1.0,
    continuous_normalize: bool = False,
    sgb: np.ndarray | None = None,
    env_offsets: list[tuple[int, int]] | None = None,
) -> dict[int, dict[str, float]]:
    """Eval the same agent on each env. Returns {env_idx: metrics}.

    ``env_offsets[i]`` is env i's offset into the global scaffold; required
    when ``agent.cfg.input_grid_state`` is True (passed to the gbook lookup).
    """
    out: dict[int, dict[str, float]] = {}
    for i, env in enumerate(envs):
        out[i] = evaluate_nav_one_env(
            env, agent, n_trials, max_steps, device,
            deterministic=deterministic, continuous_scale=continuous_scale,
            continuous_normalize=continuous_normalize,
            sgb=sgb,
            env_offset=env_offsets[i] if env_offsets is not None else None,
        )
    return out
