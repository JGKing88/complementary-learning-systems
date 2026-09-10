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

from ..policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from ..config import RNNAgentConfig
from ..world.env import CARDINAL_ACTIONS, GridEnv, at_goal
from .oracles import bfs_action_batch_continuous, bfs_action_batch_discrete
from ..world.vec_env import ContinuousVecEnv, VecEnv


@dataclass
class RNNRolloutBatch:
    """Move-only rollout for BC training. Shapes: (B, T, ...) unless noted."""
    obs: torch.Tensor                       # (B, T, input_dim) — assembled RNN input
    teacher_move_action: torch.Tensor       # (B, T) int64 (discrete) or (B, T, 2) float
    move_label_mask: torch.Tensor           # (B, T) float — 0 at at-goal steps
    rewards: torch.Tensor                   # (B, T) float — for logging only
    goal_reached: torch.Tensor              # (B, T) float — 1 if env was at goal pre-step
    student_move_action: torch.Tensor       # (B, T) int64 or (B, T, 2) float — for logging
    # The recurrent state at the START of this chunk, detached. `None` means
    # "the lifetime began here", which is what every non-lifetime caller wants
    # and what the field defaults to.
    #
    # This is load-bearing for the in-context measurement rather than
    # bookkeeping. `bc_rnn_update` re-runs the agent over `obs` to get
    # gradients, and it starts from whatever is here -- so a chunk whose
    # initial state is dropped is trained as though its lifetime started at
    # that chunk, which caps the horizon the network can ever learn to use at
    # `steps_per_rollout` no matter how long the lifetime actually is.
    initial_h: torch.Tensor | None = None    # (num_layers, B, hidden)
    #: The state at the end, detached, for the caller to feed into the next
    #: chunk of the same lifetime.
    final_h: torch.Tensor | None = None      # (num_layers, B, hidden)
    #: Episodes each row finished during this chunk, by reaching the goal or by
    #: running out of steps. Zero under `carry_across_episodes=False`, where an
    #: episode ends the row rather than continuing it.
    #:
    #: Worth recording rather than deriving: this is the quantity that was
    #: silently 1 for the whole first in-context measurement, and a training
    #: regime whose lifetimes contain one episode cannot teach anything about
    #: crossing an episode boundary.
    episodes_completed: torch.Tensor | None = None   # (B,) int64


def build_rnn_input(
    sensory: np.ndarray,                    # (B, obs_size) float32
    prev_action: np.ndarray | None,         # (B, 4) or (B, 2) float32 or None
    prev_reward: np.ndarray | None,         # (B,) float32 or None
    grid_state: np.ndarray | None,          # (B, Ng) smoothed-gbook lookup or None
    cfg: RNNAgentConfig,
    device: torch.device,
    goal_vec: np.ndarray | None = None,     # (B, 2) oracle goal channel or None
) -> torch.Tensor:
    """Assemble per-step RNN input -> (B, 1, input_dim) tensor on device.

    (The ``grid_state`` argument is what ``grid_state_vec`` below returns; the
    parameter is named differently so it does not shadow that function.)

    ``goal_vec`` is what ``goal_channel_vec`` returns and is an **oracle**: the
    goal is normally unobservable, and this exists only to put a ceiling under
    the in-context measurement. It is appended last, so a checkpoint trained
    without it keeps the same layout for every other channel.
    """
    parts = [sensory]
    if cfg.input_prev_action and prev_action is not None:
        parts.append(prev_action)
    if cfg.input_prev_reward and prev_reward is not None:
        parts.append(prev_reward.reshape(-1, 1))
    if cfg.input_grid_state and grid_state is not None:
        parts.append(grid_state)
    if getattr(cfg, "goal_channel", "none") != "none" and goal_vec is not None:
        parts.append(goal_vec)
    x = np.concatenate(parts, axis=1)                # (B, input_dim)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(1).to(device)  # (B, 1, D)


def goal_channel_vec(
    positions: np.ndarray,                  # (B, 2) agent positions
    goal: tuple[int, int],                  # this env's goal cell
    size: int,                              # arena size, for normalisation
    mode: str,                              # "abs" | "rel"
    visible: np.ndarray | None = None,      # (B,) bool; False rows are zeroed
) -> np.ndarray:
    """The oracle goal channel. Returns (B, 2) float32.

    ``abs`` gives the goal's normalised coordinates and leaves the agent to
    work out where it is; ``rel`` gives the displacement to it, which is the
    answer. The two bracket the ceiling: no in-context memory could beat
    ``abs``, because remembering where the goal is does not tell you where you
    are, and nothing can beat ``rel`` at all.

    ``visible`` masks the channel per row, so it can be shown for the first
    episode of a lifetime and withheld afterwards -- which turns the ceiling
    arm into a test of whether the network can *carry* a fact across an episode
    boundary, separately from whether it can discover one.
    """
    B = positions.shape[0]
    if mode == "abs":
        out = np.tile(np.asarray(goal, dtype=np.float32) / max(size, 1), (B, 1))
    elif mode == "rel":
        out = (np.asarray(goal, dtype=np.float32)[None, :]
               - positions.astype(np.float32)) / max(size, 1)
    else:
        raise ValueError(f"unknown goal_channel mode {mode!r}; use 'abs' or 'rel'")
    if visible is not None:
        out = out * visible.astype(np.float32).reshape(-1, 1)
    return out.astype(np.float32)


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


def prev_action_channel(
    action: np.ndarray | None, movement_mode: str, batch: int
) -> np.ndarray:
    """`action_to_prev_channel`, but defined at t=0 where there is no action yet.

    Both the collector and the evaluator used to build the channel only when
    ``prev_action is not None``, which is false on the first step -- so with
    ``input_prev_action`` on, step 0 fed the trunk an input two (continuous) or
    four (discrete) columns narrower than ``compute_rnn_input_dim`` promised and
    torch raised ``input.size(-1) must be equal to input_size``. The flag could
    therefore never be used at all. "No previous action" is the all-zero row,
    which is distinct from every one-hot and from any real displacement.
    """
    width = 4 if movement_mode == "discrete" else 2
    if action is None:
        return np.zeros((batch, width), dtype=np.float32)
    return action_to_prev_channel(action, movement_mode)


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
    carry_across_episodes: bool = False,
    initial_h: torch.Tensor | None = None,
    episode_max_steps: int | None = None,
) -> RNNRolloutBatch:
    """Collect a single (B, T) rollout in DAgger style.

    teacher_force=True overrides the student action with the oracle action when
    stepping the env. Used by the oracle-sanity smoke test (an in-distribution
    upper bound on nav success); irrelevant for normal training.

    ``carry_across_episodes`` switches the rollout from *one episode* to *a
    lifetime*. Normally an env that reaches its goal is frozen for the rest of
    the rollout, so each row is a single independent navigation episode and the
    recurrent state never has to carry anything between them. With this on, a
    reaching env is instead **teleported to a fresh start and the hidden state
    is kept**, so one row becomes a sequence of episodes in the same
    environment and the only thing linking them is recurrent activity.

    That is the whole in-context control (plan section 5.2): the goal is never
    observed, so an agent that solves episode 2 faster than episode 1 can only
    be doing it by remembering where the goal was -- in activations, with no
    weight change. It is the one comparison that runs on the Hopfield store's
    own terms, and the one whose positive result would force the framing to
    change.
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

    h = initial_h
    # Steps taken in the current episode, per row. Only used under
    # `carry_across_episodes`, to end an episode that is going nowhere.
    steps_in_ep = np.zeros(B, dtype=np.int64)
    # Which episode of the lifetime each row is on. Only meaningful under
    # `carry_across_episodes`, where a goal-reach teleports rather than ending
    # the row -- and only used by the oracle goal channel's visibility mask, so
    # a run without one is bit-identical to before.
    ep_index = np.zeros(B, dtype=np.int64)
    prev_action_np: np.ndarray | None = None
    # Zero at t=0 for the same reason `prev_action_channel` exists: the channel
    # has to be present on every step or the input width does not match the
    # trunk. Zero reward is also the truthful value -- nothing has happened yet.
    prev_reward_np: np.ndarray = np.zeros(B, dtype=np.float32)
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
            prev_action_channel(prev_action_np, movement_mode, B)
            if agent.cfg.input_prev_action else None
        )
        grid_state = (
            grid_state_vec(positions, env_offset, sgb)
            if (agent.cfg.input_grid_state and sgb is not None
                and env_offset is not None) else None
        )
        goal_vec = None
        if getattr(agent.cfg, "goal_channel", "none") != "none":
            # `goal_visible_episodes >= 0` hides the oracle after the first N
            # episodes of a lifetime, so the network has to *carry* the goal
            # across an episode boundary rather than be reminded of it. That is
            # the architecture-level positive control: if it cannot do this,
            # the failure is in carrying information, not in discovering it,
            # and that is a legible failure mode rather than a bare null.
            vis = (ep_index < agent.cfg.goal_visible_episodes
                   if getattr(agent.cfg, "goal_visible_episodes", -1) >= 0
                   else None)
            goal_vec = goal_channel_vec(positions, goal, vec.size,
                                        agent.cfg.goal_channel, visible=vis)
        x = build_rnn_input(sensory, prev_act_ch, prev_reward_np, grid_state,
                             agent.cfg, device, goal_vec=goal_vec)
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

        if carry_across_episodes:
            # A lifetime, not an episode: teleport the reachers and leave `h`
            # alone. Nothing is marked done, so every row keeps collecting and
            # the recurrent state is the only thing carrying the goal forward.
            # An episode ends on a goal-reach *or* on running out of steps.
            # The timeout matters more than it looks: without one, a row that
            # never finds the goal spends the whole rollout in a single episode
            # and never crosses an episode boundary at all -- which on a fresh
            # environment with a weak policy is the common case, so the
            # cross-episode structure this whole regime exists to train would
            # barely appear in the data. The evaluator has always ended
            # episodes both ways; the collector did not, and the two therefore
            # described different things.
            steps_in_ep += 1
            ended = at_goal_mask.copy()
            if episode_max_steps is not None:
                ended |= (steps_in_ep >= episode_max_steps)
            closing = np.where(ended)[0]
            if len(closing) > 0:
                vec.reset_indices(closing)
                ep_index[closing] += 1
                steps_in_ep[closing] = 0
        else:
            # Mark newly-at-goal envs as done (after recording the at-goal
            # step's mask=0 so the agent's choice at the goal is never
            # supervised).
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
        initial_h=(initial_h.detach().clone() if initial_h is not None else None),
        final_h=(h.detach().clone() if h is not None else None),
        episodes_completed=torch.from_numpy(ep_index.copy()),
        obs=obs_buf,
        teacher_move_action=teacher_buf,
        move_label_mask=mask_buf,
        rewards=reward_buf,
        goal_reached=goal_buf,
        student_move_action=student_buf,
    )
