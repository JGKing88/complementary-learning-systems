"""Evaluation functions for Hopfield navigation.

All evals are independent of training setup.

- evaluate_navigation: can the agent follow pre-stored Hopfield signal to goal?
- evaluate_goal_discovery: how fast does the agent find and store a new goal?
- evaluate_exploration: how efficiently does the agent explore without a goal?
"""
from __future__ import annotations

import numpy as np
import torch

from .config import TrainConfig
from .env import GridEnv, CARDINAL_ACTIONS
from .vectorhash import VectorHash
from .hopfield import Hopfield
from .agent import NavAgent
from .utils import classify_direction_batch, direction_to_onehot


def _snap_position(pos_f: np.ndarray, grid_size: int) -> np.ndarray:
    """Snap float position(s) to valid integer grid indices."""
    return np.clip(np.round(pos_f).astype(np.int32), 0, grid_size - 1)


# ---------------------------------------------------------------------------
# Shared: single-step agent logic
# ---------------------------------------------------------------------------

def _agent_step(
    agent: NavAgent,
    pos: np.ndarray,
    pos_f: np.ndarray,
    goal: tuple[int, int],
    env_offset: tuple[int, int],
    vectorhash: VectorHash,
    hopfield: Hopfield,
    h_rnn: torch.Tensor | None,
    cfg: TrainConfig,
    device: torch.device,
    deterministic: bool = True,
) -> dict:
    """Run one agent step. Reward is computed from current position before acting."""
    signal_dim = 4 if cfg.agent.hopfield_mode == "discrete" else 2
    env_size = cfg.env.size

    # Current reward from current position (before acting)
    at_goal_now = _at_goal(pos, goal)
    current_reward = torch.tensor([[1.0 if at_goal_now else -cfg.env.time_penalty]], device=device)

    embeddings_np = vectorhash.get_encoded_state(pos, env_offset)
    embeddings = torch.from_numpy(embeddings_np).float().to(device)

    # Hopfield signal
    if hopfield.num_memories > 0:
        recalled = hopfield.recall_batch(
            embeddings, steps=cfg.hopfield.steps,
            beta=cfg.hopfield.beta, alpha=cfg.hopfield.alpha,
        )
        W = vectorhash.gram_schmidt_projection(pos, env_offset)
        q = vectorhash.project_displacement(embeddings_np, recalled.cpu().numpy(), W)
        if cfg.agent.hopfield_mode == "discrete":
            hop_signal = torch.from_numpy(direction_to_onehot(
                classify_direction_batch(q))).float().to(device)
        else:
            mag = np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
            hop_signal = torch.from_numpy((q / mag).astype(np.float32)).to(device)
    else:
        hop_signal = torch.zeros(1, signal_dim, device=device)

    # Build RNN input
    parts = [current_reward]
    if cfg.agent.input_encoded_state:
        parts.append(embeddings)
    if cfg.agent.input_hopfield_signal:
        parts.append(hop_signal)
    rnn_input = torch.cat(parts, dim=-1).unsqueeze(1)

    # Agent forward
    result = agent.get_action_and_value(rnn_input, h_rnn, deterministic=deterministic)

    # Step
    if cfg.agent.movement_mode == "discrete":
        action_idx = int(result["move_action"].item())
        dx, dy = CARDINAL_ACTIONS[action_idx]
        nx = max(0, min(env_size - 1, pos[0, 0] + dx))
        ny = max(0, min(env_size - 1, pos[0, 1] + dy))
        pos_f[0] = [nx, ny]
    else:
        action = result["move_action"].cpu().numpy()[0]
        pos_f[0] = np.clip(pos_f[0] + action * cfg.env.continuous_scale, 0.0, env_size - 1)
    pos[0] = _snap_position(pos_f[0], env_size)

    return {
        "pos": pos,
        "pos_f": pos_f,
        "h_rnn": result["h_next"],
        "embedding": embeddings,
        "store_action": result["store_action"].item(),
    }


@torch.no_grad()
def evaluate_navigation(
    agent: NavAgent,
    envs: list[GridEnv],
    vectorhash: VectorHash,
    env_global_indices: list[int],
    cfg: TrainConfig,
    device: torch.device,
    num_trials: int = 32,
    max_steps: int = 20,
) -> dict[str, float]:
    """Evaluate navigation with pre-stored Hopfield goals, no agent store.

    For each env: store that env's goal in a shared Hopfield (val goals only),
    run num_trials independent trials from random starts, measure success and speed.

    Args:
        envs: Val environments to evaluate on.
        env_global_indices: Global indices into vectorhash.env_offsets for each env.
        num_trials: Independent start positions per env.
        max_steps: Max steps per trial.

    Returns dict with:
        success_rate: fraction of trials reaching goal
        mean_speed: mean(manhattan_distance / steps_taken) for successes
        mean_steps: mean steps for successes
    """
    agent.eval()
    embed_dim = vectorhash.encoded_Phi.shape[2]
    signal_dim = 4 if cfg.agent.hopfield_mode == "discrete" else 2

    # Build shared Hopfield with val env goals only
    hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
    for local_idx, env in enumerate(envs):
        global_idx = env_global_indices[local_idx]
        offset = vectorhash.env_offsets[global_idx]
        goal = env.goal_location
        gx = min(max(goal[0] + offset[0], 0), vectorhash.Npos - 1)
        gy = min(max(goal[1] + offset[1], 0), vectorhash.Npos - 1)
        goal_enc = vectorhash.encoded_Phi[gx, gy]
        hopfield.input_memory(torch.from_numpy(goal_enc).float())

    total_successes = 0
    total_trials = 0
    speed_sum = 0.0
    steps_sum = 0

    for local_idx, env in enumerate(envs):
        global_idx = env_global_indices[local_idx]
        env_offset = vectorhash.env_offsets[global_idx]
        goal = env.goal_location

        for trial in range(num_trials):
            # Fresh start, fresh RNN
            start = goal
            while start == goal:
                start = (int(np.random.randint(0, env.size)),
                         int(np.random.randint(0, env.size)))

            manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            pos = np.array([start], dtype=np.int32)
            pos_f = np.array([start], dtype=np.float64)
            h_rnn = None

            reached = False
            for step in range(max_steps):
                out = _agent_step(
                    agent, pos, pos_f, goal, env_offset, vectorhash, hopfield,
                    h_rnn, cfg, device, deterministic=True,
                )
                pos, pos_f = out["pos"], out["pos_f"]
                h_rnn = out["h_rnn"]

                if _at_goal(pos, goal):
                    reached = True
                    total_successes += 1
                    steps_taken = step + 1
                    speed_sum += manhattan / steps_taken
                    steps_sum += steps_taken
                    break

            total_trials += 1

    success_rate = total_successes / max(total_trials, 1)
    mean_speed = speed_sum / max(total_successes, 1)
    mean_steps = steps_sum / max(total_successes, 1)

    return {
        "success_rate": float(success_rate),
        "mean_speed": float(mean_speed),
        "mean_steps": float(mean_steps),
        "total_trials": total_trials,
        "total_successes": total_successes,
    }


# ---------------------------------------------------------------------------
# Distractor goal sampling
# ---------------------------------------------------------------------------

def _sample_distractor_goals(
    vectorhash: VectorHash,
    test_env_offset: tuple[int, int],
    env_size: int,
    n_distractors: int,
    rng: np.random.RandomState,
) -> list[np.ndarray]:
    """Sample encoded goal patterns from positions NOT in the test env's region.

    Returns list of (embed_dim,) numpy arrays.
    """
    Npos = vectorhash.Npos
    cx, cy = test_env_offset
    patterns = []
    while len(patterns) < n_distractors:
        gx = rng.randint(0, Npos)
        gy = rng.randint(0, Npos)
        # Reject if inside the test env's region
        if cx <= gx < cx + env_size and cy <= gy < cy + env_size:
            continue
        patterns.append(vectorhash.encoded_Phi[gx, gy].copy())
    return patterns


def _at_goal(pos: np.ndarray, goal: tuple[int, int]) -> bool:
    """Check if snapped position equals goal."""
    return (int(pos[0, 0]), int(pos[0, 1])) == goal


# ---------------------------------------------------------------------------
# Eval 2: Goal discovery speed
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_goal_discovery(
    agent: NavAgent,
    env: GridEnv,
    vectorhash: VectorHash,
    env_global_idx: int,
    cfg: TrainConfig,
    device: torch.device,
    num_trials: int = 32,
    max_steps: int = 100,
    n_distractors_list: list[int] | None = None,
) -> dict[str, list]:
    """Evaluate whether the agent stores at the goal when it finds it.

    For each n_distractors in n_distractors_list, pre-load that many goal
    patterns from other env regions into the Hopfield, then run trials where
    the agent explores with store enabled. The agent walks freely — no
    teleportation on goal reach. Trial ends when agent stores at goal, or
    after max_steps.

    Returns dict mapping n_distractors -> {store_success_rate, store_efficiency, ...}
    """
    if n_distractors_list is None:
        n_distractors_list = [0, 1, 3, 5, 10]

    agent.eval()
    embed_dim = vectorhash.encoded_Phi.shape[2]
    env_offset = vectorhash.env_offsets[env_global_idx]
    goal = env.goal_location
    goal_gx = min(max(goal[0] + env_offset[0], 0), vectorhash.Npos - 1)
    goal_gy = min(max(goal[1] + env_offset[1], 0), vectorhash.Npos - 1)
    goal_encoding = vectorhash.encoded_Phi[goal_gx, goal_gy]

    rng = np.random.RandomState(42)
    results = {}

    for n_dist in n_distractors_list:
        distractors = _sample_distractor_goals(
            vectorhash, env_offset, cfg.env.size, n_dist, rng)

        trial_steps = []
        trial_reached = []
        trial_stored = []

        for trial in range(num_trials):
            # Build Hopfield with distractors only
            hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            for pat in distractors:
                hopfield.input_memory(torch.from_numpy(pat).float())

            # Random start
            start = goal
            while start == goal:
                start = (int(rng.randint(0, env.size)), int(rng.randint(0, env.size)))

            pos = np.array([start], dtype=np.int32)
            pos_f = np.array([start], dtype=np.float64)
            h_rnn = None
            reached_goal = False
            stored_goal = False
            steps_to_store = max_steps

            for step in range(max_steps):
                out = _agent_step(
                    agent, pos, pos_f, goal, env_offset, vectorhash, hopfield,
                    h_rnn, cfg, device, deterministic=False,
                )
                pos, pos_f = out["pos"], out["pos_f"]
                h_rnn = out["h_rnn"]

                at_g = _at_goal(pos, goal)

                if at_g:
                    reached_goal = True

                # Check if agent stores while at the goal
                if out["store_action"] > 0.5:
                    if at_g:
                        stored_goal = True
                        steps_to_store = step + 1
                    hopfield.input_memory(out["embedding"][0])

                if stored_goal:
                    break

            trial_steps.append(steps_to_store)
            trial_reached.append(reached_goal)
            trial_stored.append(stored_goal)

        n_stored = sum(trial_stored)
        n_reached = sum(trial_reached)
        reach_rate = n_reached / num_trials
        store_rate = n_stored / num_trials
        results[n_dist] = {
            "store_success_rate": store_rate,
            "reach_success_rate": reach_rate,
            "store_efficiency": store_rate / max(reach_rate, 1e-8),
            "mean_steps_to_store": float(np.mean([s for s, ok in zip(trial_steps, trial_stored) if ok])) if n_stored > 0 else float("nan"),
            "mean_steps_all": float(np.mean(trial_steps)),
        }

    return results


# ---------------------------------------------------------------------------
# Eval 3: Exploration efficiency
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_exploration(
    agent: NavAgent,
    env: GridEnv,
    vectorhash: VectorHash,
    env_global_idx: int,
    cfg: TrainConfig,
    device: torch.device,
    num_trials: int = 32,
    max_steps: int = 100,
    n_distractors_list: list[int] | None = None,
) -> dict[str, dict]:
    """Evaluate exploration efficiency without the test env's goal in Hopfield.

    Measures: grid coverage (fraction of unique positions visited),
    steps to first goal reach (by chance through exploration).

    Sweep over n_distractors pre-loaded from other env regions.
    """
    if n_distractors_list is None:
        n_distractors_list = [0, 1, 3, 5, 10]

    agent.eval()
    embed_dim = vectorhash.encoded_Phi.shape[2]
    env_offset = vectorhash.env_offsets[env_global_idx]
    goal = env.goal_location
    grid_size = cfg.env.size
    total_positions = grid_size * grid_size

    rng = np.random.RandomState(42)
    results = {}

    for n_dist in n_distractors_list:
        distractors = _sample_distractor_goals(
            vectorhash, env_offset, grid_size, n_dist, rng)

        trial_coverage = []
        trial_steps_to_goal = []
        trial_found_goal = []

        for trial in range(num_trials):
            hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            for pat in distractors:
                hopfield.input_memory(torch.from_numpy(pat).float())

            start = goal
            while start == goal:
                start = (int(rng.randint(0, grid_size)), int(rng.randint(0, grid_size)))

            pos = np.array([start], dtype=np.int32)
            pos_f = np.array([start], dtype=np.float64)
            h_rnn = None

            visited = set()
            visited.add((int(pos[0, 0]), int(pos[0, 1])))
            found_goal = False
            steps_to_goal = max_steps

            for step in range(max_steps):
                out = _agent_step(
                    agent, pos, pos_f, goal, env_offset, vectorhash, hopfield,
                    h_rnn, cfg, device, deterministic=False,
                )
                pos, pos_f = out["pos"], out["pos_f"]
                h_rnn = out["h_rnn"]
                visited.add((int(pos[0, 0]), int(pos[0, 1])))

                # Allow agent to store if it wants
                if out["store_action"] > 0.5:
                    hopfield.input_memory(out["embedding"][0])

                at_g = _at_goal(pos, goal)

                if at_g and not found_goal:
                    found_goal = True
                    steps_to_goal = step + 1

            trial_coverage.append(len(visited) / total_positions)
            trial_found_goal.append(found_goal)
            trial_steps_to_goal.append(steps_to_goal)

        n_found = sum(trial_found_goal)
        results[n_dist] = {
            "mean_coverage": float(np.mean(trial_coverage)),
            "goal_find_rate": n_found / num_trials,
            "mean_steps_to_goal": float(np.mean([s for s, ok in zip(trial_steps_to_goal, trial_found_goal) if ok])) if n_found > 0 else float("nan"),
        }

    return results
