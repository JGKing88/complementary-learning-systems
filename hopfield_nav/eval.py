"""Evaluation functions for Hopfield navigation.

All three evals share a unified structure:
    for each val_env in val_envs:
        for each n_distractors in n_distractors_list:
            for each trial in range(num_trials):
                build a fresh Hopfield
                (Eval 1: preload this env's goal + distractors)
                (Eval 2/3: preload distractors only)
                run trial from a random start
            aggregate metrics across trials for this (env, n_dist)

Metrics are pooled across val envs per n_distractors. Returns
    dict[n_dist] -> dict[metric_name] -> float

- evaluate_navigation: can the agent follow a preloaded Hopfield signal to goal?
- evaluate_goal_discovery: when the agent encounters the goal, does it store?
- evaluate_exploration: how efficiently does the agent explore?
"""
from __future__ import annotations

import numpy as np
import torch

from .config import TrainConfig
from .env import GridEnv, ContinuousGridEnv, CARDINAL_ACTIONS
from .vectorhash import VectorHash
from .hopfield import Hopfield
from .agent import NavAgent
from .utils import classify_direction_batch, direction_to_onehot


def _random_start(env_size: int, goal: tuple[int, int], rng: np.random.RandomState) -> tuple[int, int]:
    """Sample a random grid cell != goal."""
    while True:
        s = (int(rng.randint(0, env_size)), int(rng.randint(0, env_size)))
        if s != goal:
            return s


def _sample_distractor_goals(
    vectorhash: VectorHash,
    test_env_offset: tuple[int, int],
    env_size: int,
    n_distractors: int,
    rng: np.random.RandomState,
) -> list[np.ndarray]:
    """Sample encoded patterns from grid positions NOT in the test env's region.

    Returns list of (embed_dim,) numpy arrays.
    """
    Npos = vectorhash.Npos
    cx, cy = test_env_offset
    patterns = []
    while len(patterns) < n_distractors:
        gx = rng.randint(0, Npos)
        gy = rng.randint(0, Npos)
        if cx <= gx < cx + env_size and cy <= gy < cy + env_size:
            continue
        patterns.append(vectorhash.encoded_Phi[gx, gy].copy())
    return patterns


def _cardinal_tuple_to_idx(action: tuple[int, int]) -> int:
    for i, a in enumerate(CARDINAL_ACTIONS):
        if a == action:
            return i
    raise ValueError(f"not a cardinal action: {action}")


def _goal_encoding(
    vectorhash: VectorHash,
    env_offset: tuple[int, int],
    goal: tuple[int, int],
) -> np.ndarray:
    """Look up the encoded pattern for this env's goal."""
    gx = min(max(goal[0] + env_offset[0], 0), vectorhash.Npos - 1)
    gy = min(max(goal[1] + env_offset[1], 0), vectorhash.Npos - 1)
    return vectorhash.encoded_Phi[gx, gy]


# ---------------------------------------------------------------------------
# Shared single-step agent logic
# ---------------------------------------------------------------------------

def _agent_step(
    agent: NavAgent,
    env: GridEnv,
    env_offset: tuple[int, int],
    vectorhash: VectorHash,
    hopfield: Hopfield,
    h_rnn: torch.Tensor | None,
    cfg: TrainConfig,
    device: torch.device,
    deterministic: bool = True,
    goal_local: tuple[int, int] | None = None,
    goal_in_memory: bool = True,
) -> dict:
    """Run one agent step. Env owns all position/movement state.

    If ``cfg.hopfield_oracle`` is set, ``goal_in_memory`` is True, the
    Hopfield has at least one memory, and ``goal_local`` is the env goal,
    the Hopfield *signal* is the oracle: same Gram--Schmidt projection as real
    recall, but the displacement is ``goal_embedding - current_embedding``
    (no associative recall), testing whether a perfect directional cue fixes
    behavior.

    If ``cfg.action_oracle`` is set with the same gating, the *movement* action
    is overridden to a greedy best cardinal step (discrete) or a unit step
    toward the goal in continuous space, while the store head is unchanged.
    """
    signal_dim = 4 if cfg.agent.hopfield_mode == "discrete" else 2

    pos_tuple = env.current_location
    pos_arr = np.array([pos_tuple], dtype=np.int32)
    current_reward = torch.tensor([[env.reward()]], device=device, dtype=torch.float32)

    embeddings_np = vectorhash.get_encoded_state(pos_arr, env_offset)
    embeddings = torch.from_numpy(embeddings_np).float().to(device)

    use_oracle = (
        bool(getattr(cfg, "hopfield_oracle", False))
        and bool(cfg.agent.input_hopfield_signal)
        and goal_local is not None
        and goal_in_memory
        and hopfield.num_memories > 0
    )

    if not cfg.agent.input_hopfield_signal:
        hop_signal = torch.zeros(1, signal_dim, device=device)
    elif use_oracle:
        g_arr = np.array([goal_local], dtype=np.int32)
        goal_enc = vectorhash.get_encoded_state(g_arr, env_offset)
        W = vectorhash.gram_schmidt_projection(pos_arr, env_offset)
        q = vectorhash.project_displacement(embeddings_np, goal_enc, W)
        if cfg.agent.hopfield_mode == "discrete":
            hop_signal = torch.from_numpy(direction_to_onehot(
                classify_direction_batch(q))).float().to(device)
        else:
            mag = np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
            hop_signal = torch.from_numpy((q / mag).astype(np.float32)).to(device)
    elif hopfield.num_memories > 0:
        recalled = hopfield.recall_batch(
            embeddings, steps=cfg.hopfield.steps,
            beta=cfg.hopfield.beta, alpha=cfg.hopfield.alpha,
        )
        W = vectorhash.gram_schmidt_projection(pos_arr, env_offset)
        q = vectorhash.project_displacement(embeddings_np, recalled.cpu().numpy(), W)
        if cfg.agent.hopfield_mode == "discrete":
            hop_signal = torch.from_numpy(direction_to_onehot(
                classify_direction_batch(q))).float().to(device)
        else:
            mag = np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
            hop_signal = torch.from_numpy((q / mag).astype(np.float32)).to(device)
    else:
        hop_signal = torch.zeros(1, signal_dim, device=device)

    parts = [current_reward]
    if cfg.agent.input_encoded_state:
        parts.append(embeddings)
    if cfg.agent.input_hopfield_signal:
        parts.append(hop_signal)
    rnn_input = torch.cat(parts, dim=-1).unsqueeze(1)

    use_action_oracle = (
        bool(getattr(cfg, "action_oracle", False))
        and goal_local is not None
        and goal_in_memory
        and hopfield.num_memories > 0
    )
    move_action_override: torch.Tensor | None = None
    move_override_mask: torch.Tensor | None = None
    if use_action_oracle:
        if cfg.agent.movement_mode == "discrete":
            a = env.best_action_to_goal()
            idx = _cardinal_tuple_to_idx(a)
            move_action_override = torch.tensor([[idx]], device=device, dtype=torch.long)
        else:
            if not isinstance(env, ContinuousGridEnv):
                raise TypeError("action_oracle in continuous mode requires ContinuousGridEnv")
            v = env.oracle_unit_toward_goal()
            move_action_override = (
                torch.from_numpy(v).to(device).float().view(1, 1, 2)
            )
        move_override_mask = torch.tensor([True], device=device, dtype=torch.bool)

    result = agent.get_action_and_value(
        rnn_input,
        h_rnn,
        deterministic=deterministic,
        move_action_override=move_action_override,
        move_override_mask=move_override_mask,
    )

    if cfg.agent.movement_mode == "discrete":
        action_idx = int(result["move_action"].item())
        env.step(CARDINAL_ACTIONS[action_idx])
    else:
        action_vec = result["move_action"].cpu().numpy()[0]
        env.step(action_vec)

    return {
        "h_rnn": result["h_next"],
        "embedding": embeddings,
        "store_action": result["store_action"].item(),
    }


# ---------------------------------------------------------------------------
# Eval 1: Navigation — agent must follow the preloaded Hopfield signal to goal
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_navigation(
    agent: NavAgent,
    val_envs: list[GridEnv],
    vectorhash: VectorHash,
    env_global_indices: list[int],
    cfg: TrainConfig,
    device: torch.device,
    num_trials: int = 32,
    max_steps: int = 200,
    n_distractors_list: list[int] | None = None,
    deterministic: bool = True,
    seed: int = 42,
) -> dict[int, dict[str, float]]:
    """Evaluate whether the agent can follow a preloaded goal signal to the goal.

    For each val env and each distractor count, every trial gets a fresh Hopfield
    containing exactly (this env's goal) + (n_distractors random patterns from
    elsewhere in the VectorHash). No other val envs' goals are included.

    Returns {n_dist: {success_rate, mean_speed, mean_steps, total_trials,
                      total_successes}}.
    """
    if n_distractors_list is None:
        n_distractors_list = [0]

    agent.eval()
    embed_dim = vectorhash.encoded_Phi.shape[2]
    results: dict[int, dict[str, float]] = {}

    for n_dist in n_distractors_list:
        rng = np.random.RandomState(seed)  # deterministic per distractor level
        total_successes = 0
        total_trials = 0
        speed_sum = 0.0
        steps_sum = 0

        for local_idx, env in enumerate(val_envs):
            global_idx = env_global_indices[local_idx]
            env_offset = vectorhash.env_offsets[global_idx]
            goal = env.goal_location
            goal_enc = _goal_encoding(vectorhash, env_offset, goal)

            for _trial in range(num_trials):
                hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                hopfield.input_memory(torch.from_numpy(goal_enc).float())
                for pat in _sample_distractor_goals(
                    vectorhash, env_offset, cfg.env.size, n_dist, rng,
                ):
                    hopfield.input_memory(torch.from_numpy(pat).float())

                start = _random_start(env.size, goal, rng)
                dy = start[0] - goal[0]
                dx = start[1] - goal[1]
                if cfg.env.movement_mode == "continuous":
                    start_dist = float(np.hypot(dy, dx))
                else:
                    start_dist = float(abs(dy) + abs(dx))
                env.set_position(start)
                h_rnn = None

                for step in range(max_steps):
                    out = _agent_step(
                        agent, env, env_offset, vectorhash, hopfield,
                        h_rnn, cfg, device, deterministic=deterministic,
                        goal_local=goal, goal_in_memory=True,
                    )
                    h_rnn = out["h_rnn"]

                    if env.current_location == goal:
                        total_successes += 1
                        steps_taken = step + 1
                        speed_sum += start_dist / steps_taken
                        steps_sum += steps_taken
                        break

                total_trials += 1

        results[n_dist] = {
            "success_rate": float(total_successes / max(total_trials, 1)),
            "mean_speed": float(speed_sum / max(total_successes, 1)),
            "mean_steps": float(steps_sum / max(total_successes, 1)),
            "total_trials": total_trials,
            "total_successes": total_successes,
        }

    return results


# ---------------------------------------------------------------------------
# Eval 2: Goal discovery — does the agent store when it reaches the goal?
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_goal_discovery(
    agent: NavAgent,
    val_envs: list[GridEnv],
    vectorhash: VectorHash,
    env_global_indices: list[int],
    cfg: TrainConfig,
    device: torch.device,
    num_trials: int = 32,
    max_steps: int = 200,
    n_distractors_list: list[int] | None = None,
    seed: int = 42,
    deterministic: bool = True,
) -> dict[int, dict[str, float]]:
    """For each val env + distractor count, run trials where the agent explores
    with store enabled. Each trial gets a fresh Hopfield pre-loaded with ONLY
    distractors (not this env's goal, and not other val envs' goals).

    Trial ends when agent stores at goal or after max_steps.
    When ``deterministic`` is True (default), actions follow the policy mean /
    argmax and store uses prob > 0.5, matching the navigation eval. Set False to
    sample from the action and store distributions.
    """
    if n_distractors_list is None:
        n_distractors_list = [0]

    agent.eval()
    embed_dim = vectorhash.encoded_Phi.shape[2]
    results: dict[int, dict[str, float]] = {}

    for n_dist in n_distractors_list:
        rng = np.random.RandomState(seed)
        trial_steps: list[int] = []
        trial_reached: list[bool] = []
        trial_stored: list[bool] = []

        for local_idx, env in enumerate(val_envs):
            global_idx = env_global_indices[local_idx]
            env_offset = vectorhash.env_offsets[global_idx]
            goal = env.goal_location

            for _trial in range(num_trials):
                distractors = _sample_distractor_goals(
                    vectorhash, env_offset, cfg.env.size, n_dist, rng,
                )
                hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                for pat in distractors:
                    hopfield.input_memory(torch.from_numpy(pat).float())

                start = _random_start(env.size, goal, rng)
                env.set_position(start)
                h_rnn = None
                reached_goal = False
                stored_goal = False
                steps_to_store = max_steps
                goal_in_mem = False  # this env's goal pattern written to Hopfield

                for step in range(max_steps):
                    at_g_pre = env.current_location == goal
                    if at_g_pre:
                        reached_goal = True

                    out = _agent_step(
                        agent, env, env_offset, vectorhash, hopfield,
                        h_rnn, cfg, device, deterministic=deterministic,
                        goal_local=goal, goal_in_memory=goal_in_mem,
                    )
                    h_rnn = out["h_rnn"]

                    if env.current_location == goal:
                        reached_goal = True

                    if out["store_action"] > 0.5:
                        if at_g_pre:
                            stored_goal = True
                            steps_to_store = step + 1
                            goal_in_mem = True
                        hopfield.input_memory(out["embedding"][0])

                    if stored_goal:
                        break

                trial_steps.append(steps_to_store)
                trial_reached.append(reached_goal)
                trial_stored.append(stored_goal)

        n = max(len(trial_stored), 1)
        n_stored = sum(trial_stored)
        n_reached = sum(trial_reached)
        reach_rate = n_reached / n
        store_rate = n_stored / n
        store_steps = [s for s, ok in zip(trial_steps, trial_stored) if ok]
        results[n_dist] = {
            "store_success_rate": float(store_rate),
            "reach_success_rate": float(reach_rate),
            "store_efficiency": float(store_rate / max(reach_rate, 1e-8)),
            "mean_steps_to_store": float(np.mean(store_steps)) if store_steps else float("nan"),
            "mean_steps_all": float(np.mean(trial_steps)) if trial_steps else float("nan"),
        }

    return results


# ---------------------------------------------------------------------------
# Eval 3: Exploration efficiency
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_exploration(
    agent: NavAgent,
    val_envs: list[GridEnv],
    vectorhash: VectorHash,
    env_global_indices: list[int],
    cfg: TrainConfig,
    device: torch.device,
    num_trials: int = 32,
    max_steps: int = 200,
    n_distractors_list: list[int] | None = None,
    seed: int = 42,
    deterministic: bool = True,
) -> dict[int, dict[str, float]]:
    """For each val env + distractor count, measure grid coverage and steps-to-goal
    for full-length (no early termination) rollouts. Each trial gets a fresh Hopfield
    pre-loaded with ONLY distractors. Default ``deterministic=True`` (see
    :func:`evaluate_goal_discovery`)."""
    if n_distractors_list is None:
        n_distractors_list = [0]

    agent.eval()
    embed_dim = vectorhash.encoded_Phi.shape[2]
    grid_size = cfg.env.size
    total_positions = grid_size * grid_size
    results: dict[int, dict[str, float]] = {}

    for n_dist in n_distractors_list:
        rng = np.random.RandomState(seed)
        trial_coverage: list[float] = []
        trial_found: list[bool] = []
        trial_steps_to_goal: list[int] = []

        for local_idx, env in enumerate(val_envs):
            global_idx = env_global_indices[local_idx]
            env_offset = vectorhash.env_offsets[global_idx]
            goal = env.goal_location

            for _trial in range(num_trials):
                distractors = _sample_distractor_goals(
                    vectorhash, env_offset, grid_size, n_dist, rng,
                )
                hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                for pat in distractors:
                    hopfield.input_memory(torch.from_numpy(pat).float())

                start = _random_start(grid_size, goal, rng)
                env.set_position(start)
                h_rnn = None
                goal_in_mem = False

                visited: set[tuple[int, int]] = set()
                visited.add(env.current_location)
                found_goal = False
                steps_to_goal = max_steps

                for step in range(max_steps):
                    at_g_pre = env.current_location == goal
                    out = _agent_step(
                        agent, env, env_offset, vectorhash, hopfield,
                        h_rnn, cfg, device, deterministic=deterministic,
                        goal_local=goal, goal_in_memory=goal_in_mem,
                    )
                    h_rnn = out["h_rnn"]
                    visited.add(env.current_location)

                    if out["store_action"] > 0.5:
                        hopfield.input_memory(out["embedding"][0])
                        if at_g_pre:
                            goal_in_mem = True

                    if env.current_location == goal and not found_goal:
                        found_goal = True
                        steps_to_goal = step + 1

                trial_coverage.append(len(visited) / total_positions)
                trial_found.append(found_goal)
                trial_steps_to_goal.append(steps_to_goal)

        n = max(len(trial_found), 1)
        n_found = sum(trial_found)
        found_steps = [s for s, ok in zip(trial_steps_to_goal, trial_found) if ok]
        results[n_dist] = {
            "mean_coverage": float(np.mean(trial_coverage)) if trial_coverage else 0.0,
            "goal_find_rate": float(n_found / n),
            "mean_steps_to_goal": float(np.mean(found_steps)) if found_steps else float("nan"),
        }

    return results


# ---------------------------------------------------------------------------
# Eval 4: Realistic — a single Hopfield accumulates memories across envs as
# the agent is introduced to them sequentially; after each new env we retest
# all prior envs to measure catastrophic interference.
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_realistic(
    agent: NavAgent,
    val_envs: list[GridEnv],
    vectorhash: VectorHash,
    env_global_indices: list[int],
    cfg: TrainConfig,
    device: torch.device,
    steps_per_env: int = 1000,
    seed: int = 42,
    deterministic: bool = True,
) -> dict:
    """End-of-training realistic eval with persistent Hopfield memory.

    Protocol (for env_i in val_envs, in order):
      - PRIMARY: random start, fresh RNN, run steps_per_env steps with the
        agent's own store head driving Hopfield writes. On every goal-reach,
        teleport to a random non-goal cell and reset RNN.
      - For each prior env_j (j < i): RETEST phase with identical mechanics
        but STORING DISABLED — the Hopfield is frozen during retests.

    The Hopfield is never reset: it accumulates memories across the entire
    eval.  Retests measure interference from later-stored patterns.

    Returns a dict with keys:
      - "primary": {env_idx: {n_reaches, intervals, mean_interval, stored_at_reach, ...}}
      - "retest":  {(visit_i, retest_j): {...}}
      - "drift":   {env_idx: [(gap, metrics), ...]}  # gap=0 is primary,
        gap=k>0 is the retest that happened after visiting env (env_idx + k)
      - "summary": {mean_primary_reaches, mean_final_retest_reaches,
                    interference_drop, hopfield_final_memories}
    """
    agent.eval()
    embed_dim = vectorhash.encoded_Phi.shape[2]
    hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
    rng = np.random.RandomState(seed)
    # Per-env: has this val env's goal ever been written into the (shared) Hopfield?
    shared_goal_stored: dict[int, bool] = {i: False for i in range(len(val_envs))}

    def _run_phase(
        env: GridEnv,
        env_offset: tuple[int, int],
        allow_store: bool,
        local_idx: int,
    ) -> dict:
        goal = env.goal_location
        start = _random_start(env.size, goal, rng)
        env.set_position(start)
        h_rnn = None
        last_reach_step = 0
        intervals: list[int] = []
        stored_at_reach: list[bool] = []

        for step in range(steps_per_env):
            goal_in_mem = shared_goal_stored[local_idx]
            out = _agent_step(
                agent, env, env_offset, vectorhash, hopfield,
                h_rnn, cfg, device, deterministic=deterministic,
                goal_local=goal, goal_in_memory=goal_in_mem,
            )
            h_rnn = out["h_rnn"]

            store_fired = bool(allow_store and (out["store_action"] > 0.5))
            if store_fired:
                hopfield.input_memory(out["embedding"][0])

            if env.current_location == goal:
                if allow_store and store_fired:
                    shared_goal_stored[local_idx] = True
                intervals.append(step + 1 - last_reach_step)
                # Count a "store at reach" only when a write actually happens on
                # this step (store_fired) while at goal. Retest: allow_store is
                # False, so this is always False.
                stored_at_reach.append(store_fired)
                new_start = _random_start(env.size, goal, rng)
                env.set_position(new_start)
                h_rnn = None
                last_reach_step = step + 1

        # Trailing partial segment: steps since last reach (or phase start) that
        # did NOT culminate in a goal reach. Kept separately so metric
        # aggregation still uses only completed intervals, but the plot can
        # show where the phase was cut off.
        tail_steps = steps_per_env - last_reach_step
        return {
            "n_reaches": len(intervals),
            "intervals": intervals,
            "stored_at_reach": stored_at_reach,
            "mean_interval": float(np.mean(intervals)) if intervals else float("nan"),
            "tail_steps": int(tail_steps),
        }

    N = len(val_envs)
    primary: dict[int, dict] = {}
    retest: dict[tuple[int, int], dict] = {}
    drift: dict[int, list[tuple[int, dict]]] = {j: [] for j in range(N)}

    for i, env in enumerate(val_envs):
        env_offset_i = vectorhash.env_offsets[env_global_indices[i]]
        prim = _run_phase(env, env_offset_i, allow_store=True, local_idx=i)
        primary[i] = prim
        drift[i].append((0, prim))
        print(f"  realistic env {i}: primary n_reaches={prim['n_reaches']}, "
              f"hopfield_mem={hopfield.num_memories}")

        for j in range(i):
            env_j = val_envs[j]
            env_offset_j = vectorhash.env_offsets[env_global_indices[j]]
            gap = i - j
            rm = _run_phase(env_j, env_offset_j, allow_store=False, local_idx=j)
            retest[(i, j)] = rm
            drift[j].append((gap, rm))
            print(f"    retest env {j} (gap={gap}): n_reaches={rm['n_reaches']}")

    primary_reaches = [primary[i]["n_reaches"] for i in range(N)]
    mean_primary = float(np.mean(primary_reaches)) if primary_reaches else 0.0

    # Final retest = each prior env tested after the LAST env was visited.
    final_retest_reaches: list[int] = []
    drops: list[float] = []
    for j in range(N - 1):
        entry = retest.get((N - 1, j))
        if entry is None:
            continue
        r = entry["n_reaches"]
        final_retest_reaches.append(r)
        p = primary[j]["n_reaches"]
        if p > 0:
            drops.append((p - r) / p)

    return {
        "primary": primary,
        "retest": retest,
        "drift": drift,
        "summary": {
            "mean_primary_reaches": mean_primary,
            "mean_final_retest_reaches": (
                float(np.mean(final_retest_reaches))
                if final_retest_reaches else float("nan")
            ),
            "interference_drop": float(np.mean(drops)) if drops else float("nan"),
            "hopfield_final_memories": int(hopfield.num_memories),
        },
    }


# ---------------------------------------------------------------------------
# Eval 5: Repeat — realistic primary phase only, isolated Hopfield per trial
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_repeat(
    agent: NavAgent,
    val_envs: list[GridEnv],
    vectorhash: VectorHash,
    env_global_indices: list[int],
    cfg: TrainConfig,
    device: torch.device,
    n_trials: int = 5,
    steps_per_env: int = 200,
    seed: int = 42,
    deterministic: bool = True,
) -> dict:
    """Per-env repeated single-env runs with a totally fresh Hopfield each time.

    For each val env, run ``n_trials`` independent trials of the realistic-eval
    primary phase: random start, empty Hopfield, fresh RNN, agent's own store
    head writes into the Hopfield, goal reaches teleport and reset the RNN.

    Returns {"trials": {env_idx: [trial_entry, ...]}, "summary": {...}} where
    each ``trial_entry`` has ``intervals``, ``stored_at_reach``, ``tail_steps``,
    ``n_reaches``, ``start``, ``trial_idx``.
    """
    agent.eval()
    embed_dim = vectorhash.encoded_Phi.shape[2]
    rng = np.random.RandomState(seed)

    def _run_trial(
        env: GridEnv,
        env_offset: tuple[int, int],
        hopfield: Hopfield,
    ) -> dict:
        goal = env.goal_location
        start = _random_start(env.size, goal, rng)
        env.set_position(start)
        h_rnn = None
        last_reach_step = 0
        intervals: list[int] = []
        stored_at_reach: list[bool] = []
        goal_in_mem = False

        for step in range(steps_per_env):
            out = _agent_step(
                agent, env, env_offset, vectorhash, hopfield,
                h_rnn, cfg, device, deterministic=deterministic,
                goal_local=goal, goal_in_memory=goal_in_mem,
            )
            h_rnn = out["h_rnn"]

            store_fired = bool(out["store_action"] > 0.5)
            if store_fired:
                hopfield.input_memory(out["embedding"][0])

            if env.current_location == goal:
                if store_fired:
                    goal_in_mem = True
                intervals.append(step + 1 - last_reach_step)
                stored_at_reach.append(store_fired)
                new_start = _random_start(env.size, goal, rng)
                env.set_position(new_start)
                h_rnn = None
                last_reach_step = step + 1

        tail_steps = steps_per_env - last_reach_step
        return {
            "n_reaches": len(intervals),
            "intervals": intervals,
            "stored_at_reach": stored_at_reach,
            "mean_interval": float(np.mean(intervals)) if intervals else float("nan"),
            "tail_steps": int(tail_steps),
            "start": [int(start[0]), int(start[1])],
        }

    N = len(val_envs)
    trials: dict[int, list[dict]] = {i: [] for i in range(N)}
    for i, env in enumerate(val_envs):
        env_offset = vectorhash.env_offsets[env_global_indices[i]]
        for t in range(n_trials):
            hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            entry = _run_trial(env, env_offset, hopfield)
            entry["trial_idx"] = int(t)
            trials[i].append(entry)
        mean_r = float(np.mean([e["n_reaches"] for e in trials[i]])) if trials[i] else 0.0
        print(f"  repeat env {i}: n_trials={n_trials} mean_reaches={mean_r:.2f}")

    flat_reaches = [e["n_reaches"] for lst in trials.values() for e in lst]
    return {
        "trials": trials,
        "summary": {
            "n_trials": int(n_trials),
            "steps_per_env": int(steps_per_env),
            "mean_reaches": float(np.mean(flat_reaches)) if flat_reaches else 0.0,
        },
    }
