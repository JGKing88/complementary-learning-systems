"""Slim DAgger rollout collector for the vanilla-RNN BC baseline.

Inputs: a VecEnv (B parallel rollouts on one env), an RNNAgent, an oracle.
Per-step: read positions/goal -> oracle teacher action; agent samples its own
action (DAgger); env steps with the student action; record (obs, teacher,
mask). On goal-reach, env teleports and the RNN's hidden state for that env
is zeroed (each per-env trajectory is an independent navigation episode).

No dependencies on RolloutCollector, VectorHash, encoder, or Hopfield.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .agent_rnn import RNNAgent, compute_rnn_input_dim
from .config import RNNAgentConfig
from .env import CARDINAL_ACTIONS, GridEnv, at_goal
from .oracle_bfs import bfs_action_batch_continuous, bfs_action_batch_discrete
from .vec_env import ContinuousVecEnv, VecEnv


@dataclass
class RNNRolloutBatch:
    """Move-only rollout for BC training. Shapes: (B, T, ...) unless noted."""
    obs: torch.Tensor                       # (B, T, input_dim) — assembled RNN input
    teacher_move_action: torch.Tensor       # (B, T) int64 (discrete) or (B, T, 2) float
    move_label_mask: torch.Tensor           # (B, T) float — 0 at at-goal steps
    rewards: torch.Tensor                   # (B, T) float — for logging only
    goal_reached: torch.Tensor              # (B, T) float — 1 if env was at goal pre-step
    student_move_action: torch.Tensor       # (B, T) int64 or (B, T, 2) float — for logging


def build_rnn_input(
    sensory: np.ndarray,                    # (B, obs_size) float32
    prev_action: np.ndarray | None,         # (B, 4) or (B, 2) float32 or None
    prev_reward: np.ndarray | None,         # (B,) float32 or None
    grid_state: np.ndarray | None,          # (B, Ng) smoothed-gbook lookup or None
    cfg: RNNAgentConfig,
    device: torch.device,
) -> torch.Tensor:
    """Assemble per-step RNN input -> (B, 1, input_dim) tensor on device.

    (The ``grid_state`` argument is what ``grid_state_vec`` below returns; the
    parameter is named differently so it does not shadow that function.)
    """
    parts = [sensory]
    if cfg.input_prev_action and prev_action is not None:
        parts.append(prev_action)
    if cfg.input_prev_reward and prev_reward is not None:
        parts.append(prev_reward.reshape(-1, 1))
    if cfg.input_grid_state and grid_state is not None:
        parts.append(grid_state)
    x = np.concatenate(parts, axis=1)                # (B, input_dim)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(1).to(device)  # (B, 1, D)


def grid_state_vec(
    positions: np.ndarray,                  # (B, 2) int — local env coords
    env_offset: tuple[int, int],            # (ox, oy) — env's offset in the global scaffold
    sgb: np.ndarray,                        # (Ng, Npos, Npos) — smooth_gbook(vh.gbook, lambdas, fwhm_ratio)
) -> np.ndarray:
    """Smoothed-gbook column at the agent's GLOBAL position. Returns (B, Ng) float32."""
    Npos = sgb.shape[1]
    gx = np.clip(positions[:, 0] + env_offset[0], 0, Npos - 1)
    gy = np.clip(positions[:, 1] + env_offset[1], 0, Npos - 1)
    return sgb[:, gx, gy].T.astype(np.float32)


def action_to_prev_channel(
    action: np.ndarray, movement_mode: str
) -> np.ndarray:
    """One-hot (discrete) or pass-through (continuous) for prev_action input."""
    if movement_mode == "discrete":
        B = action.shape[0]
        out = np.zeros((B, 4), dtype=np.float32)
        out[np.arange(B), action.astype(np.int64)] = 1.0
        return out
    return action.astype(np.float32)


@torch.no_grad()
def collect_rollout_rnn(
    vec: VecEnv | ContinuousVecEnv,
    agent: RNNAgent,
    cfg: RNNAgentConfig,
    steps: int,
    device: torch.device,
    deterministic: bool = False,
    teacher_force: bool = False,
    sgb: np.ndarray | None = None,
    env_offset: tuple[int, int] | None = None,
) -> RNNRolloutBatch:
    """Collect a single (B, T) rollout in DAgger style.

    teacher_force=True overrides the student action with the oracle action when
    stepping the env. Used by the oracle-sanity smoke test (an in-distribution
    upper bound on nav success); irrelevant for normal training.
    """
    B = vec.B
    movement_mode = agent.cfg.movement_mode
    gbook_dim = int(sgb.shape[0]) if (sgb is not None and agent.cfg.input_grid_state) else 0
    input_dim = compute_rnn_input_dim(agent.cfg, vec._obs_size, gbook_dim)

    obs_buf = torch.zeros((B, steps, input_dim), dtype=torch.float32, device=device)
    if movement_mode == "discrete":
        teacher_buf = torch.zeros((B, steps), dtype=torch.int64, device=device)
        student_buf = torch.zeros((B, steps), dtype=torch.int64, device=device)
    else:
        teacher_buf = torch.zeros((B, steps, 2), dtype=torch.float32, device=device)
        student_buf = torch.zeros((B, steps, 2), dtype=torch.float32, device=device)
    mask_buf = torch.zeros((B, steps), dtype=torch.float32, device=device)
    reward_buf = torch.zeros((B, steps), dtype=torch.float32, device=device)
    goal_buf = torch.zeros((B, steps), dtype=torch.float32, device=device)

    h = None
    prev_action_np: np.ndarray | None = None
    prev_reward_np: np.ndarray | None = None
    goal = (int(vec._goal[0]), int(vec._goal[1]))

    # Per-env "done" flag. Once an env's pre-step position is the goal, the
    # navigation episode ends: that env is frozen for the rest of the rollout
    # (no teleport, no further env steps), and all remaining steps for that
    # env are masked out of the BC loss.
    done = np.zeros(B, dtype=bool)

    for t in range(steps):
        sensory = vec.obs_batch().astype(np.float32)             # (B, obs_size)
        positions = vec.positions()                              # (B, 2) int

        at_goal_mask = at_goal(vec)

        # Oracle teacher action.
        if movement_mode == "discrete":
            teacher_np = bfs_action_batch_discrete(positions, goal, vec.size, vec._rng)
        else:
            teacher_np = bfs_action_batch_continuous(positions, goal, vec._rng)

        # Build RNN input and step the agent.
        prev_act_ch = (
            action_to_prev_channel(prev_action_np, movement_mode)
            if (agent.cfg.input_prev_action and prev_action_np is not None) else None
        )
        grid_state = (
            grid_state_vec(positions, env_offset, sgb)
            if (agent.cfg.input_grid_state and sgb is not None
                and env_offset is not None) else None
        )
        x = build_rnn_input(sensory, prev_act_ch, prev_reward_np, grid_state,
                             agent.cfg, device)
        out = agent.act(x, h, deterministic=deterministic)
        h = out["h_next"]
        student_action = out["move_action"].cpu().numpy()

        # Mask=0 for steps where the env is at-goal or already finished.
        step_mask = ~done & ~at_goal_mask

        # Record into buffers (post-input, pre-step).
        obs_buf[:, t] = x.squeeze(1)
        if movement_mode == "discrete":
            teacher_buf[:, t] = torch.from_numpy(teacher_np.astype(np.int64)).to(device)
            student_buf[:, t] = torch.from_numpy(student_action.astype(np.int64)).to(device)
        else:
            teacher_buf[:, t] = torch.from_numpy(teacher_np).to(device)
            student_buf[:, t] = torch.from_numpy(student_action.astype(np.float32)).to(device)
        mask_buf[:, t] = torch.from_numpy(step_mask.astype(np.float32)).to(device)
        goal_buf[:, t] = torch.from_numpy(at_goal_mask.astype(np.float32)).to(device)

        # Mark newly-at-goal envs as done (after recording the at-goal step's
        # mask=0 so the agent's choice at the goal is never supervised).
        done = done | at_goal_mask

        # Step only the still-active envs. Done envs are frozen at their
        # current position (the goal) for the rest of the rollout — no
        # teleport, no further env transitions.
        action_to_step = teacher_np if teacher_force else student_action
        active_idx = np.where(~done)[0]
        rewards_full = np.zeros(B, dtype=np.float32)
        if len(active_idx) > 0:
            rewards_active, _, _ = vec.step_batch(
                action_to_step[active_idx], indices=active_idx,
            )
            rewards_full[active_idx] = rewards_active
        reward_buf[:, t] = torch.from_numpy(rewards_full).to(device)

        # Update prev_action / prev_reward inputs.
        prev_action_np = student_action
        prev_reward_np = rewards_full

    return RNNRolloutBatch(
        obs=obs_buf,
        teacher_move_action=teacher_buf,
        move_label_mask=mask_buf,
        rewards=reward_buf,
        goal_reached=goal_buf,
        student_move_action=student_buf,
    )
