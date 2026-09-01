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

from ..policy.agent_rnn import RNNAgent, compute_rnn_input_dim, set_agent_task
from ..config import RNNAgentConfig
from ..world.env import GridEnv, at_goal
from ..rollout.rnn import (
    build_rnn_input, goal_channel_vec, grid_state_vec, prev_action_channel)
from ..world.vec_env import ContinuousVecEnv, VecEnv, make_vec


def _make_vec(env: GridEnv, batch: int, movement_mode: str,
              continuous_scale: float = 1.0,
              continuous_normalize: bool = False) -> VecEnv | ContinuousVecEnv:
    """Deprecated alias for vec_env.make_vec.

    reset=False: this evaluator places the agent at seeded start positions
    itself, so resetting here would consume the env RNG and move them.
    """
    return make_vec(env, batch, movement_mode, continuous_scale,
                    continuous_normalize, reset=False)


def optimal_path_to_goal(
    starts: np.ndarray,
    goal: tuple[int, int],
    goal_radius: float,
    movement_mode: str,
) -> np.ndarray:
    """Shortest *attainable* path from each start to the goal, per trial.

    This arena has no interior obstacles -- the four walls are the boundary,
    and they carry the barcode rather than blocking anything -- so the box is
    convex and the straight segment between any two interior points is
    traversable. That makes this an exact optimum rather than the loose lower
    bound a geodesic-on-an-obstacle-map would give, which is unusual and worth
    relying on: the denominator of an SPL built from it is not an estimate.

    Units are deliberately *distance*, not steps, under continuous movement.
    "Optimal steps" is not a property of the environment there: with
    ``continuous_normalize=False`` the displacement is the raw action vector,
    so the policy chooses its own step magnitude and the step count that a
    route costs depends on how far the agent commits to moving each time. The
    teacher always emits a unit vector, so 1.0 is the natural conversion --
    but it is an assumption about the policy, and baking it into a recorded
    number would hide it. Divide by the step size at analysis time if you want
    steps.

    Under discrete movement the actions are the four cardinals, so one action
    is one cell and the Manhattan distance *is* both the distance and the
    optimal step count.
    """
    starts = np.asarray(starts, dtype=np.float64).reshape(-1, 2)
    gx, gy = float(goal[0]), float(goal[1])
    if movement_mode == "continuous":
        d = np.sqrt((gx - starts[:, 0]) ** 2 + (gy - starts[:, 1]) ** 2)
        # The agent only has to reach the edge of the at-goal ball, not its
        # centre -- `at_goal` is an L2 ball of radius `goal_radius`.
        return np.maximum(d - float(goal_radius), 0.0)
    return np.abs(gx - starts[:, 0]) + np.abs(gy - starts[:, 1])


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
    reset_state_on_teleport: bool = False,
) -> dict[str, float]:
    """Run n_trials parallel trials from random starts.

    Returns nav_det (success rate), mean_steps_to_goal and mean_path_to_goal
    and mean_optimal_to_goal (all over successful trials; nan if zero),
    mean_optimal_all (over every trial), and mean_episode_return.

    The three path quantities are recorded together because on their own none
    of them is comparable across arms. `mean_steps_to_goal` mixes route
    quality with step magnitude, since a continuous-mode policy picks how far
    it moves each step. `mean_path_to_goal` isolates distance travelled but is
    still conditioned on success, so an arm that only solves the nearby goals
    is scored on a nearer subpopulation than one that solves the far ones too.
    `mean_optimal_to_goal` is what removes that: the ratio path/optimal is a
    route-efficiency figure that does not care how hard the trial was, and
    `optimal / max(path, optimal)`, zeroed on failures, is SPL.
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
    # Recorded at reset because reaching the goal teleports the agent:
    # after the first hit there is no start position left to read.
    optimal = optimal_path_to_goal(
        prev_positions_f, goal, getattr(vec, "goal_radius", 0.5),
        movement_mode)
    returns = np.zeros(n_trials, dtype=np.float32)

    h = None
    prev_action_np: np.ndarray | None = None
    prev_reward_np: np.ndarray = np.zeros(n_trials, dtype=np.float32)

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
            prev_action_channel(prev_action_np, movement_mode, n_trials)
            if agent.cfg.input_prev_action else None
        )
        grid_state = (
            grid_state_vec(positions, env_offset, sgb)
            if (agent.cfg.input_grid_state and sgb is not None
                and env_offset is not None) else None
        )
        goal_vec = (
            goal_channel_vec(positions, goal, env.size, agent.cfg.goal_channel)
            if getattr(agent.cfg, "goal_channel", "none") != "none" else None
        )
        x = build_rnn_input(sensory, prev_act_ch, prev_reward_np, grid_state,
                             agent.cfg, device, goal_vec=goal_vec)
        out = agent.act(x, h, deterministic=deterministic)
        h = out["h_next"]
        student_action = out["move_action"].cpu().numpy()

        rewards, goal_reached, _ = vec.step_batch(student_action)
        returns += rewards

        # C5 of the at-goal contract, under `--reset_state_on_teleport`. Every
        # trial's record is frozen at its first hit, so this changes no reported
        # number here -- but it must follow the same switch as training, or the
        # RNN stack would evaluate a recurrence the trainer never produced.
        if reset_state_on_teleport and h is not None and goal_reached.any():
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
        # Over successes only, so it pairs cell-for-cell with mean_path_to_goal
        # and their ratio is a route-efficiency figure for the same trials.
        "mean_optimal_to_goal": (
            float(optimal[succ_idx].mean()) if len(succ_idx) > 0 else float("nan")
        ),
        # Over every trial, so it says how hard the attempted trials were --
        # which is what tells you whether an arm's successes are its easy ones.
        "mean_optimal_all": float(optimal.mean()),
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
    reset_state_on_teleport: bool = False,
) -> dict[int, dict[str, float]]:
    """Eval the same agent on each env. Returns {env_idx: metrics}.

    ``env_offsets[i]`` is env i's offset into the global scaffold; required
    when ``agent.cfg.input_grid_state`` is True (passed to the gbook lookup).

    A task-conditioned policy is told which env it is being evaluated on before
    each one. That is an oracle task id and it is a real advantage -- the whole
    point of recording `needs_task_id` on the method is that these arms are
    upper bounds rather than peers. It is set here rather than by the caller
    because the caller does not always know the agent needs it, and an agent
    evaluated on five envs under one task's parameters produces a curve
    indistinguishable from catastrophic forgetting.
    """
    out: dict[int, dict[str, float]] = {}
    for i, env in enumerate(envs):
        set_agent_task(agent, i)
        out[i] = evaluate_nav_one_env(
            env, agent, n_trials, max_steps, device,
            deterministic=deterministic, continuous_scale=continuous_scale,
            continuous_normalize=continuous_normalize,
            sgb=sgb,
            env_offset=env_offsets[i] if env_offsets is not None else None,
            reset_state_on_teleport=reset_state_on_teleport,
        )
    return out


__all__ = [
    "optimal_path_to_goal",
    "evaluate_nav_one_env",
    "evaluate_nav_all",
]
