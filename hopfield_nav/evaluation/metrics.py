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
import torch.nn.functional as F

from ..policy import channels
from ..rollout import signal as signal_ops
from . import protocols
from .batched import batched_exploration_trials, batched_navigation_trials
from ..world import episode
from ..config import TrainConfig
from ..world.env import GridEnv, ContinuousGridEnv, CARDINAL_ACTIONS, at_goal
from ..rollout.distractors import goal_encoding, sample_distractors
from ..world.scaffold import VectorHash
from hopfield import Hopfield
from ..policy.agent import NavAgent


def random_start(env_size: int, goal: tuple[int, int], rng: np.random.RandomState) -> tuple[int, int]:
    """Sample a random grid cell != goal."""
    while True:
        s = (int(rng.randint(0, env_size)), int(rng.randint(0, env_size)))
        if s != goal:
            return s


def _cardinal_tuple_to_idx(action: tuple[int, int]) -> int:
    for i, a in enumerate(CARDINAL_ACTIONS):
        if a == action:
            return i
    raise ValueError(f"not a cardinal action: {action}")


# ---------------------------------------------------------------------------
# Shared single-step agent logic
# ---------------------------------------------------------------------------

def agent_step(
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
    prev_reward: torch.Tensor | None = None,
    prev_action: torch.Tensor | None = None,
    action_temperature: float = 1.0,
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
    prev_action_dim = 4 if cfg.agent.movement_mode == "discrete" else 2

    # Enrichment defaults: if caller didn't thread prev_* state, fall back to
    # zeros (correct at start-of-episode, OOD for subsequent steps — callers
    # must pass these for accurate behavior once enrichment flags are on).
    if prev_reward is None:
        prev_reward = torch.zeros(1, 1, device=device)
    if prev_action is None:
        prev_action = torch.zeros(1, prev_action_dim, device=device)

    pos_tuple = env.current_location
    pos_arr = np.array([pos_tuple], dtype=np.int32)
    current_reward = torch.tensor([[env.reward()]], device=device, dtype=torch.float32)

    embeddings_np = vectorhash.get_encoded_state(pos_arr, env_offset)
    embeddings = torch.from_numpy(embeddings_np).float().to(device)

    # Which pattern a store on this step writes. Only differs from `embeddings`
    # when off-cell stores are suppressed and the agent is at goal on some other
    # cell -- see VectorHash.get_store_patterns.
    if getattr(cfg.env, "allow_offcell_store", True) or goal_local is None:
        store_embeddings = embeddings
    else:
        store_embeddings = torch.from_numpy(
            vectorhash.get_store_patterns(
                pos_arr, env_offset,
                at_goal_mask=np.array([at_goal(env)]),
                goal=goal_local,
                allow_offcell=False,
            )
        ).float().to(device)

    use_oracle = (
        bool(getattr(cfg, "hopfield_oracle", False))
        and bool(cfg.agent.input_hopfield_signal)
        and goal_local is not None
        and goal_in_memory
        and hopfield.num_memories > 0
    )

    def _to_channel(sig: torch.Tensor, q: np.ndarray) -> torch.Tensor:
        """Which of the two the policy is fed: the normalized signal, or raw q.

        ``input_hopfield_raw`` hands the policy the unnormalized displacement,
        so magnitude ("how far to the goal") survives alongside direction. Only
        defined in continuous mode; discrete stays a one-hot. The rollout
        collector makes the same choice at its own call site.
        """
        if cfg.agent.input_hopfield_raw and cfg.agent.hopfield_mode != "discrete":
            return torch.from_numpy(q.astype(np.float32)).to(device)
        return sig

    if not cfg.agent.input_hopfield_signal:
        hop_signal = torch.zeros(1, signal_dim, device=device)
    elif use_oracle:
        sig_np, q = signal_ops.oracle_signal_at(
            vectorhash, embeddings_np, pos_arr, env_offset, goal_local,
            cfg.agent,
        )
        hop_signal = _to_channel(
            torch.from_numpy(sig_np).float().to(device), q)
    elif hopfield.num_memories > 0:
        # B=1 through the same batched implementation the collector uses.
        sig_t, q, _mask, _W = signal_ops.hopfield_signal_at(
            vectorhash, cfg, embeddings_np, embeddings, pos_arr, env_offset,
            hopfield, True, device, embeddings.shape[1],
        )
        hop_signal = _to_channel(sig_t, q)
    else:
        hop_signal = torch.zeros(1, signal_dim, device=device)

    values = {
        "current_reward": current_reward,
        "prev_reward": prev_reward,
        "encoded_state": embeddings,
        "hopfield_signal": hop_signal,
        "prev_action": prev_action,
        # Single-env counterpart of VecEnv.last_displacement(); the env records
        # it on every step. Zero before the first move.
        "prev_displacement": torch.from_numpy(
            np.asarray(getattr(env, "_last_displacement", np.zeros(2)),
                       dtype=np.float32)).view(1, 2).to(device),
        "goal_in_memory": torch.tensor(
            [[1.0 if goal_in_memory else 0.0]], device=device),
    }
    if cfg.agent.input_sensory:
        # env.obs() reads at the env's own heading; pos_tuple IS env's current
        # cell, so this was the same call before headings existed and is the
        # heading-correct one now.
        values["sensory"] = torch.from_numpy(
            env.obs()[None, :]).float().to(device)               # (1, obs_size)
    if (cfg.agent.input_hopfield_multistep
            and cfg.agent.hopfield_mode == "continuous"):
        # Project the recall trajectory at each requested iteration count.
        # multistep_q returns zeros when there is no basis, which is exactly
        # the empty-Hopfield case: nothing recalled, no displacement to project.
        W = (vectorhash.gram_schmidt_projection(pos_arr, env_offset)
             if hopfield.num_memories > 0 else None)
        msq = signal_ops.multistep_q(
            vectorhash, cfg, embeddings_np, embeddings, hopfield, True, W,
            cfg.agent.input_hopfield_multistep, embeddings.shape[1], device,
        )
        for s, q_s in msq.items():
            values[channels.multistep_name(s)] = torch.from_numpy(
                q_s.astype(np.float32)).to(device)

    rnn_input = channels.build_policy_input(
        channels.channel_specs(cfg.agent, embeddings.shape[1],
                               cfg.env.observation_size),
        values, batch_size=1,
    ).unsqueeze(1)

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
        action_temperature=action_temperature,
    )

    if cfg.agent.movement_mode == "discrete":
        action_idx = int(result["move_action"].item())
        env.step(CARDINAL_ACTIONS[action_idx])
        next_prev_action = F.one_hot(
            result["move_action"].long().view(-1), num_classes=4
        ).float()  # (1, 4)
    else:
        action_vec = result["move_action"].cpu().numpy()[0]
        env.step(action_vec)
        next_prev_action = result["move_action"].float().view(1, -1)  # (1, 2)

    return {
        "h_rnn": result["h_next"],
        "embedding": embeddings,
        # The pattern a store fired on THIS step should write. Identical to
        # "embedding" unless cfg.env.allow_offcell_store is False and the agent
        # is at goal on a non-goal cell (reachable only at goal_radius > 0.5).
        # Callers that store must use this key, not "embedding".
        "store_embedding": store_embeddings,
        "store_action": result["store_action"].item(),
        # For the caller to thread into the next agent_step call:
        "next_prev_reward": current_reward,        # (1, 1)
        "next_prev_action": next_prev_action,      # (1, D)
    }


# ---------------------------------------------------------------------------
# Eval 1: Navigation — agent must follow the preloaded Hopfield signal to goal
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_navigation(
    agent: NavAgent,
    val_envs: list[GridEnv],
    vectorhash: VectorHash,
    env_offsets: list[tuple[int, int]],
    cfg: TrainConfig,
    device: torch.device,
    num_trials: int = 32,
    max_steps: int = 200,
    n_distractors_list: list[int] | None = None,
    deterministic: bool = True,
    seed: int = 42,
    action_temperature: float = 1.0,
    per_trial: list | None = None,
) -> dict[int, dict[str, float]]:
    """Evaluate whether the agent can follow a preloaded goal signal to the goal.

    For each val env and each distractor count, every trial gets a fresh Hopfield
    containing exactly (this env's goal) + (n_distractors random patterns from
    elsewhere in the VectorHash). No other val envs' goals are included.

    Returns {n_dist: {success_rate, mean_speed, mean_steps, total_trials,
                      total_successes}}.

    If ``per_trial`` is a list, one record per trial is appended to it:
    ``(n_dist, env_local_idx, trial_idx, success, steps_taken)`` with
    ``steps_taken = -1`` on failure. The aggregate floats above are averages
    over exactly these records. Per-trial outcomes are the stable contract --
    they survive changes to batching and RNG consumption order that would move
    the aggregates -- so this is what the golden fixtures pin.
    """
    if n_distractors_list is None:
        n_distractors_list = [0]

    agent.eval()
    embed_dim = vectorhash.encoded_Phi.shape[2]
    # Declared, not inherited. The guard fails loudly if this row
    # is ever changed to one a GridEnv-stepping loop cannot honour.
    episode.require_single_env_support(
        episode.contract_for("evaluate_navigation"), "evaluate_navigation")
    results: dict[int, dict[str, float]] = {}

    for n_dist in n_distractors_list:
        rng = np.random.RandomState(seed)  # deterministic per distractor level
        total_successes = 0
        total_trials = 0
        speed_sum = 0.0
        steps_sum = 0

        for local_idx, env in enumerate(val_envs):
            env_offset = env_offsets[local_idx]
            goal = env.goal_location
            goal_enc = goal_encoding(vectorhash, env_offset, goal)

            # Set up all num_trials trials first, drawing from `rng` in exactly
            # the sequential order -- distractors, shuffle, start, per trial.
            # Batching must not change which memories or which starts a trial
            # gets; only how many forward passes it takes to run them.
            hopfields: list[Hopfield] = []
            starts: list[tuple[int, int]] = []
            start_dists: list[float] = []
            for _trial_idx in range(num_trials):
                hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                # Goal + distractors, shuffled so storage order is random.
                patterns = [goal_enc]
                # `env.size`, not `cfg.env.size`: the exclusion box is this
                # arena's footprint, and a val set at a size the run never
                # trained on would otherwise draw "distractors" from inside it.
                patterns.extend(sample_distractors(
                    vectorhash, env_offset, env.size, n_dist, rng,
                ))
                rng.shuffle(patterns)
                for pat in patterns:
                    hopfield.input_memory(torch.from_numpy(pat).float())
                hopfields.append(hopfield)

                start = random_start(env.size, goal, rng)
                starts.append(start)
                dy = start[0] - goal[0]
                dx = start[1] - goal[1]
                start_dists.append(
                    float(np.hypot(dy, dx)) if cfg.env.movement_mode == "continuous"
                    else float(abs(dy) + abs(dx))
                )

            trial_steps_arr = batched_navigation_trials(
                agent=agent, env=env, env_offset=env_offset,
                vectorhash=vectorhash, hopfields=hopfields, cfg=cfg,
                device=device, starts=starts, goal=goal, max_steps=max_steps,
                deterministic=deterministic,
                action_temperature=action_temperature,
            )

            for _trial_idx, trial_steps in enumerate(trial_steps_arr):
                if trial_steps > 0:
                    total_successes += 1
                    speed_sum += start_dists[_trial_idx] / trial_steps
                    steps_sum += trial_steps
                total_trials += 1
                if per_trial is not None:
                    per_trial.append((n_dist, local_idx, _trial_idx,
                                      int(trial_steps > 0), trial_steps))

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
    env_offsets: list[tuple[int, int]],
    cfg: TrainConfig,
    device: torch.device,
    num_trials: int = 32,
    max_steps: int = 200,
    n_distractors_list: list[int] | None = None,
    seed: int = 42,
    deterministic: bool = True,
    action_temperature: float = 1.0,
    per_trial: list | None = None,
) -> dict[int, dict[str, float]]:
    """Explore a world holding only distractors; store the goal on arrival.

    Each trial gets a fresh Hopfield pre-loaded with ONLY distractors -- not
    this env's goal, and not any other val env's goal -- and a random non-goal
    start. The agent explores. Every step that *begins* at the goal is one
    arrival: the store head gets exactly that step to fire, and then the agent
    is teleported to a fresh non-goal cell with its recurrent state reset. The
    trial runs the full ``max_steps``, so it yields as many arrivals as the
    policy earns.

    The teleport is the point (contract `TRAINING`, see `world/episode.py`).
    Without it the agent could park inside the goal radius and take a fresh
    store decision every step until its probability drifted past 0.5 -- which
    measures "fires eventually", not "fires on arrival", and is not the deal
    training gives it. Adopted 2026-08-06; before that this site declared
    `OBSERVE` and the trial ended at the first successful store.

    Stores fire into the Hopfield wherever they happen; only those on an
    arrival step count as successes. After the first success `goal_in_memory`
    is True for the rest of the trial, because by then it is.

    When ``deterministic`` is True (default), actions follow the policy mean /
    argmax and store uses prob > 0.5, matching the navigation eval. Set False to
    sample from the action and store distributions.

    Metrics: ``store_success_rate`` and ``reach_success_rate`` are per *trial*
    (did it ever store / ever arrive); ``store_efficiency`` is per *arrival*
    (stores / arrivals), which is what the old ``store_rate / reach_rate``
    approximated and can now be counted exactly; ``mean_steps_to_store`` is to
    the first success; ``mean_arrivals`` is new.

    If ``per_trial`` is a list, one record per trial is appended to it:
    ``(n_dist, env_local_idx, trial_idx, reached, stored, steps_to_store,
    n_arrivals, n_stores)``. See evaluate_navigation for why per-trial records,
    not aggregates, are what the golden fixtures pin.
    """
    if n_distractors_list is None:
        n_distractors_list = [0]

    agent.eval()
    embed_dim = vectorhash.encoded_Phi.shape[2]
    # The full training contract, applied by hand below the way
    # evaluate_realistic does -- GridEnv.step implements neither C3 nor C4, so
    # this site cannot go through require_single_env_support.
    contract = episode.contract_for(
        "evaluate_goal_discovery",
        reset_state=cfg.env.reset_state_on_teleport)
    results: dict[int, dict[str, float]] = {}

    for n_dist in n_distractors_list:
        rng = np.random.RandomState(seed)
        trial_steps: list[int] = []
        trial_reached: list[bool] = []
        trial_stored: list[bool] = []
        trial_arrivals: list[int] = []
        trial_store_events: list[int] = []

        for local_idx, env in enumerate(val_envs):
            env_offset = env_offsets[local_idx]
            goal = env.goal_location

            for _trial_idx in range(num_trials):
                distractors = sample_distractors(
                    vectorhash, env_offset, env.size, n_dist, rng,
                )
                hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                for pat in distractors:
                    hopfield.input_memory(torch.from_numpy(pat).float())

                start = random_start(env.size, goal, rng)
                env.set_position(start)
                h_rnn = None
                prev_reward = None
                prev_action = None
                stored_goal = False
                steps_to_store = max_steps
                goal_in_mem = False  # this env's goal pattern written to Hopfield
                n_arrivals = 0       # steps that *began* at the goal
                n_stores = 0         # arrivals on which the store head fired

                for step in range(max_steps):
                    # An arrival is a step that begins at the goal: that is the
                    # step on which the agent can act on being there. A
                    # post-step arrival becomes the next iteration's at_g_pre.
                    at_g_pre = at_goal(env)
                    if at_g_pre:
                        n_arrivals += 1

                    out = agent_step(
                        agent, env, env_offset, vectorhash, hopfield,
                        h_rnn, cfg, device, deterministic=deterministic,
                        goal_local=goal, goal_in_memory=goal_in_mem,
                        prev_reward=prev_reward, prev_action=prev_action,
                        action_temperature=action_temperature,
                    )
                    h_rnn = out["h_rnn"]
                    prev_reward = out["next_prev_reward"]
                    prev_action = out["next_prev_action"]

                    if out["store_action"] > 0.5:
                        if at_g_pre:
                            n_stores += 1
                            if not stored_goal:
                                stored_goal = True
                                steps_to_store = step + 1
                            goal_in_mem = True
                        hopfield.input_memory(out["store_embedding"][0])

                    if at_g_pre:
                        # One opportunity per visit, then relocate -- the deal
                        # training gives it. Without this the agent could sit
                        # inside the goal radius accumulating chances until its
                        # store probability drifted past 0.5, which measures
                        # "fires eventually", not "fires on arrival".
                        res = episode.resolve_at_goal(
                            np.array([True]), contract,
                            goal_reward=cfg.env.goal_reward,
                            time_penalty=cfg.env.time_penalty,
                        )
                        if res.teleport[0]:
                            env.set_position(random_start(env.size, goal, rng))
                        if res.reset_state[0]:
                            h_rnn = None
                            prev_reward = None
                            prev_action = None

                reached_goal = n_arrivals > 0
                trial_steps.append(steps_to_store)
                trial_reached.append(reached_goal)
                trial_stored.append(stored_goal)
                trial_arrivals.append(n_arrivals)
                trial_store_events.append(n_stores)
                if per_trial is not None:
                    per_trial.append((n_dist, local_idx, _trial_idx,
                                      int(reached_goal), int(stored_goal),
                                      steps_to_store, n_arrivals, n_stores))

        n = max(len(trial_stored), 1)
        n_stored = sum(trial_stored)
        n_reached = sum(trial_reached)
        reach_rate = n_reached / n
        store_rate = n_stored / n
        store_steps = [s for s, ok in zip(trial_steps, trial_stored) if ok]
        # Per-arrival, now that a trial yields several. This is what
        # store_rate / reach_rate was approximating with per-trial rates; with
        # the teleport in place the exact version is available, and the ratio
        # of rates would be meaningless (a trial can arrive many times).
        total_arrivals = sum(trial_arrivals)
        total_stores = sum(trial_store_events)
        results[n_dist] = {
            "store_success_rate": float(store_rate),
            "reach_success_rate": float(reach_rate),
            "store_efficiency": float(total_stores / max(total_arrivals, 1e-8)),
            "mean_steps_to_store": float(np.mean(store_steps)) if store_steps else float("nan"),
            "mean_steps_all": float(np.mean(trial_steps)) if trial_steps else float("nan"),
            "mean_arrivals": float(np.mean(trial_arrivals)) if trial_arrivals else float("nan"),
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
    env_offsets: list[tuple[int, int]],
    cfg: TrainConfig,
    device: torch.device,
    num_trials: int = 32,
    max_steps: int = 200,
    n_distractors_list: list[int] | None = None,
    seed: int = 42,
    deterministic: bool = True,
    action_temperature: float = 1.0,
    per_trial: list | None = None,
) -> dict[int, dict[str, float]]:
    """How much ground the policy covers when there is nothing to find.

    Each trial gets a fresh Hopfield holding distractors and nothing else, a
    random non-goal start, and exactly ``max_steps`` steps. **No store fires**
    -- the store head's output is not read, so the goal can never enter memory
    -- and the goal is **inert**: contract `NO_GOALS`, so standing on it pays
    `-time_penalty` like any other cell. The policy has no signal that the goal
    exists, which is what makes this a measurement of walking rather than of
    goal-seeking. Reaching the goal is recorded anyway, as an incidental.

    Nothing terminates early. That is deliberate: with variable-length trials a
    per-step coverage rate is biased by trial length, and trial length would be
    set by goal-finding -- folding the thing being controlled for back into the
    number. Equal lengths also make the union across trials well defined.

    Metrics per distractor count. Only five are independent:

      mean_coverage        cells one rollout visits, / that **env's** cells
      cells_per_step       the same quantity / max_steps -- a rescale of
                           mean_coverage, kept because it is invariant to grid
                           size where the other is invariant to step budget
      union_coverage       cells ALL rollouts in an env visit between them, /
                           that env's cells, then averaged over envs
      union_per_rollout    union_coverage / num_trials -- a rescale
      redundancy           |union| / sum of per-trial counts, in [1/N, 1]. 1.0
                           means every rollout explored disjoint ground; 1/N
                           means they all retraced the same path. This is what
                           separates "each rollout covers little" from "every
                           rollout covers the same little", which neither
                           coverage number can distinguish on its own.
      goal_find_rate       trials that ever stood on the goal / trials
      mean_steps_to_goal   first-arrival step, over trials that found it; nan
                           if none did

    Absorbed ``evaluate_union_coverage`` on 2026-08-06: it ran its own
    independent rollouts to compute the same union, so the two now share one
    set. ``num_trials`` is the N the union is taken over, and the union
    saturates toward 1.0 as N grows -- at large N it stops discriminating.

    If ``per_trial`` is a list, one record per trial is appended to it:
    ``(n_dist, env_local_idx, trial_idx, n_cells, found_goal, steps_to_goal)``.
    """
    if n_distractors_list is None:
        n_distractors_list = [0]

    agent.eval()
    embed_dim = vectorhash.encoded_Phi.shape[2]
    # Coverage is a fraction of *this env's* cells, and `env.size` below is the
    # only place that is read. Taking it from the config is the §6.1 silent
    # failure: a size-12 val set scored under a size-6 config reported coverage
    # against 36 instead of 144 -- 4x the truth, no error.
    results: dict[int, dict[str, float]] = {}

    for n_dist in n_distractors_list:
        rng = np.random.RandomState(seed)
        trial_cells: list[int] = []
        trial_denom: list[int] = []
        trial_found: list[bool] = []
        trial_steps_to_goal: list[int] = []
        per_env_union: list[float] = []
        per_env_redundancy: list[float] = []

        for local_idx, env in enumerate(val_envs):
            env_offset = env_offsets[local_idx]
            goal = env.goal_location
            grid_size = int(env.size)
            total_positions = grid_size * grid_size

            # Setup draws from the caller's RNG in the original per-trial
            # order, so the batched run sees the same worlds the sequential
            # one would have.
            hopfields, starts = [], []
            for _trial_idx in range(num_trials):
                distractors = sample_distractors(
                    vectorhash, env_offset, grid_size, n_dist, rng,
                )
                hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta,
                                    device=str(device))
                for pat in distractors:
                    hopfield.input_memory(torch.from_numpy(pat).float())
                hopfields.append(hopfield)
                starts.append(random_start(grid_size, goal, rng))

            visited, found, steps_to_goal = batched_exploration_trials(
                agent=agent, env=env, env_offset=env_offset,
                vectorhash=vectorhash, hopfields=hopfields, cfg=cfg,
                device=device, starts=starts, max_steps=max_steps,
                deterministic=deterministic,
                action_temperature=action_temperature,
            )

            union: set = set()
            summed = 0
            for _trial_idx, (cells, hit, s) in enumerate(
                    zip(visited, found, steps_to_goal)):
                union |= cells
                summed += len(cells)
                trial_cells.append(len(cells))
                trial_denom.append(total_positions)
                trial_found.append(hit)
                trial_steps_to_goal.append(s if s >= 0 else max_steps)
                if per_trial is not None:
                    per_trial.append((n_dist, local_idx, _trial_idx,
                                      len(cells), int(hit),
                                      s if s >= 0 else max_steps))
            per_env_union.append(len(union) / total_positions)
            # |union| / sum of parts: 1.0 iff no two rollouts shared a cell.
            per_env_redundancy.append(len(union) / max(summed, 1))

        mean_cells = float(np.mean(trial_cells)) if trial_cells else 0.0
        union_cov = float(np.mean(per_env_union)) if per_env_union else 0.0
        reach_steps = [s for s, ok in zip(trial_steps_to_goal, trial_found) if ok]
        # One denominator when every env is the same size -- which is the
        # historical expression exactly, so a same-size run is unchanged to the
        # last bit. A mixed-size set has no single denominator and averages the
        # per-trial fractions instead.
        denoms = set(trial_denom)
        if len(denoms) == 1:
            mean_cov = mean_cells / float(denoms.pop())
        elif trial_cells:
            mean_cov = float(np.mean(np.asarray(trial_cells, dtype=float)
                                     / np.asarray(trial_denom, dtype=float)))
        else:
            mean_cov = 0.0
        results[n_dist] = {
            "mean_coverage": mean_cov,
            "cells_per_step": mean_cells / max(max_steps, 1),
            "union_coverage": union_cov,
            "union_per_rollout": union_cov / max(num_trials, 1),
            "redundancy": (float(np.mean(per_env_redundancy))
                           if per_env_redundancy else 0.0),
            "goal_find_rate": (float(np.mean(trial_found))
                               if trial_found else 0.0),
            "mean_steps_to_goal": (float(np.mean(reach_steps))
                                   if reach_steps else float("nan")),
            "num_trials_per_env": float(num_trials),
            "max_steps": float(max_steps),
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
    env_offsets: list[tuple[int, int]],
    cfg: TrainConfig,
    device: torch.device,
    steps_per_env: int = 1000,
    seed: int = 42,
    deterministic: bool = True,
    lock_store_after_goal: bool = False,
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

    If ``lock_store_after_goal`` is True, a per-env store lock is enforced:
    once an env's goal has been stored into the (shared) Hopfield, all further
    store actions in that env are suppressed for the rest of the eval. This
    isolates "store discipline" — an agent that fires store every step still
    only writes the goal once per env.

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
    # Declared, not inherited: this evaluator measures reach intervals against
    # the training distribution, so it takes the training contract at the goal.
    contract = episode.contract_for(
        "evaluate_realistic", reset_state=cfg.env.reset_state_on_teleport)
    # Per-env: has this val env's goal ever been written into the (shared) Hopfield?
    shared_goal_stored: dict[int, bool] = {i: False for i in range(len(val_envs))}

    def _run_phase(
        env: GridEnv,
        env_offset: tuple[int, int],
        allow_store: bool,
        local_idx: int,
    ) -> dict:
        """Training-matched semantics: a "reach" is the iter where the agent
        *sits* at the goal (pre-step ``at_g=True``), matching ``VecEnv.step_batch``
        which returns ``goal_reached`` from the pre-step check. On such an iter
        the agent reads ``reward=+1``, may fire store (writing the goal
        embedding), then teleports + RNN reset. The prior landing iter is a
        normal step (agent decides from non-goal state, writes a non-goal
        embedding if it fires store).
        """
        goal = env.goal_location
        start = random_start(env.size, goal, rng)
        env.set_position(start)
        h_rnn = None
        prev_reward = None
        prev_action = None
        last_reach_step = 0
        intervals: list[int] = []
        stored_at_reach: list[bool] = []

        for step in range(steps_per_env):
            at_g = at_goal(env)
            goal_in_mem = shared_goal_stored[local_idx]
            out = agent_step(
                agent, env, env_offset, vectorhash, hopfield,
                h_rnn, cfg, device, deterministic=deterministic,
                goal_local=goal, goal_in_memory=goal_in_mem,
                prev_reward=prev_reward, prev_action=prev_action,
            )
            h_rnn = out["h_rnn"]
            prev_reward = out["next_prev_reward"]
            prev_action = out["next_prev_action"]

            allow_store_now = allow_store and not (
                lock_store_after_goal and shared_goal_stored[local_idx]
            )
            store_fired = bool(allow_store_now and (out["store_action"] > 0.5))
            if store_fired:
                hopfield.input_memory(out["store_embedding"][0])

            if at_g:
                # Training-matching reach: agent sat on goal this iter.
                # ``out["store_embedding"][0]`` is the pre-step (= goal) pattern,
                # so ``store_fired`` here means the goal was actually written.
                if allow_store_now and store_fired:
                    shared_goal_stored[local_idx] = True
                intervals.append(step + 1 - last_reach_step)
                stored_at_reach.append(store_fired)
                # The declared contract decides what follows. Under TRAINING
                # that is: discard the move agent_step already applied by
                # relocating, and reset the recurrent state, because the new
                # episode segment has no valid "previous step".
                res = episode.resolve_at_goal(
                    np.array([True]), contract,
                    goal_reward=cfg.env.goal_reward,
                    time_penalty=cfg.env.time_penalty,
                )
                if res.teleport[0]:
                    env.set_position(random_start(env.size, goal, rng))
                if res.reset_state[0]:
                    h_rnn = None
                    prev_reward = None
                    prev_action = None
                last_reach_step = step + 1

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
        env_offset_i = env_offsets[i]
        prim = _run_phase(env, env_offset_i, allow_store=True, local_idx=i)
        primary[i] = prim
        drift[i].append((0, prim))
        print(f"  realistic env {i}: primary n_reaches={prim['n_reaches']}, "
              f"hopfield_mem={hopfield.num_memories}")

        for j in range(i):
            env_j = val_envs[j]
            env_offset_j = env_offsets[j]
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
    env_offsets: list[tuple[int, int]],
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
    contract = episode.contract_for(
        "evaluate_repeat", reset_state=cfg.env.reset_state_on_teleport)

    def _run_trial(
        env: GridEnv,
        env_offset: tuple[int, int],
        hopfield: Hopfield,
    ) -> dict:
        goal = env.goal_location
        start = random_start(env.size, goal, rng)
        env.set_position(start)
        h_rnn = None
        prev_reward = None
        prev_action = None
        last_reach_step = 0
        intervals: list[int] = []
        stored_at_reach: list[bool] = []
        goal_in_mem = False

        # Training-matched "reach" semantics (see ``evaluate_realistic._run_phase``).
        for step in range(steps_per_env):
            at_g = at_goal(env)
            out = agent_step(
                agent, env, env_offset, vectorhash, hopfield,
                h_rnn, cfg, device, deterministic=deterministic,
                goal_local=goal, goal_in_memory=goal_in_mem,
                prev_reward=prev_reward, prev_action=prev_action,
            )
            h_rnn = out["h_rnn"]
            prev_reward = out["next_prev_reward"]
            prev_action = out["next_prev_action"]

            store_fired = bool(out["store_action"] > 0.5)
            if store_fired:
                hopfield.input_memory(out["store_embedding"][0])

            if at_g:
                if store_fired:
                    goal_in_mem = True
                intervals.append(step + 1 - last_reach_step)
                stored_at_reach.append(store_fired)
                # Same declared contract as evaluate_realistic; see there.
                res = episode.resolve_at_goal(
                    np.array([True]), contract,
                    goal_reward=cfg.env.goal_reward,
                    time_penalty=cfg.env.time_penalty,
                )
                if res.teleport[0]:
                    env.set_position(random_start(env.size, goal, rng))
                if res.reset_state[0]:
                    h_rnn = None
                    prev_reward = None
                    prev_action = None
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
        env_offset = env_offsets[i]
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


# ---------------------------------------------------------------------------
# Eval 6: Sequential continual — episodic success, persistent Hopfield
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sequential_episodes(
    agent: NavAgent,
    val_envs: list[GridEnv],
    vectorhash: VectorHash,
    env_offsets: list[tuple[int, int]],
    cfg: TrainConfig,
    device: torch.device,
    iters_per_block: int = 50,
    max_steps: int = 32,
    seed: int = 42,
    deterministic: bool = True,
    lock_store_after_goal: bool = False,
    oracle_store_at_goal: bool = False,
) -> dict:
    """Sequential continual-learning eval, reproducing the paper-style figure.

    Protocol:
      - Introduce ``val_envs`` in order, producing one "block" per env.
      - At every outer iteration within block ``i``, run **one mini-episode in
        every already-introduced env** ``j <= i``:
          * random non-goal start, fresh RNN, run up to ``max_steps`` steps;
          * ``j == i`` (primary) → agent's store head may write into the
            shared Hopfield;
          * ``j < i``  (revisit) → stores disabled (Hopfield frozen for that
            mini-episode).
        Each mini-episode contributes a single 0/1 "success" bit (goal reached
        within ``max_steps`` from that random start).
      - The Hopfield is never reset — it accumulates across all blocks.

    If ``lock_store_after_goal`` is True, once an env's goal has been stored
    (``goal_in_mem[local_idx]`` flips True), all further store actions in that
    env are suppressed for the rest of the eval — both within the current
    mini-episode and in any subsequent primary mini-episode of the same env.

    If ``oracle_store_at_goal`` is True, the agent's store head is bypassed:
    a store fires automatically and only when the agent is on the goal cell
    of the current env. Off-goal stores are suppressed. Useful for evaluating
    Phase-A-only ckpts whose store policy is untrained.

    Returns a dict whose main payload is
    ``env_iters[j] = [(iter, success, stored_at_goal, stored_off_goal), ...]``:
    the per-env sequence of (outer-iteration, success-bit, at-goal-store-bit,
    off-goal-store-bit) tuples, ready to plot as a moving-average line plus
    store-event markers. Also returns per-block boundaries and summary
    statistics.
    """
    agent.eval()
    N = len(val_envs)
    embed_dim = vectorhash.encoded_Phi.shape[2]
    hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
    rng = np.random.RandomState(seed)
    goal_in_mem: dict[int, bool] = {i: False for i in range(N)}
    env_iters: dict[int, list[tuple[int, int]]] = {i: [] for i in range(N)}
    stored_at_goal_count: dict[int, int] = {i: 0 for i in range(N)}

    # The protocol itself lives in evaluation/protocols.py, shared with the
    # figure pipeline in final_plotting/agenthash.py. This function keeps its
    # signature, its accumulator and its summary block, so the eval_all JSON
    # schema is unchanged; only the ~85 lines of duplicated control flow moved.
    #
    # eval exposes one --oracle-store-at-goal flag meaning both "force a store
    # at the goal" and "suppress stores anywhere else". agenthash splits those.
    # Passing the same value for both preserves the combined meaning exactly.
    boundaries: list[int] = []

    def _record_block(block: int, cur_iter: int) -> None:
        boundaries.append(cur_iter)
        print(f"  sequential block {block} (env {block}): iters={iters_per_block} "
              f"hopfield_mem={hopfield.num_memories} "
              f"goal_in_mem={goal_in_mem[block]}", flush=True)

    for step in protocols.run_sequential_protocol(
        agent=agent, val_envs=val_envs, env_offsets=env_offsets,
        vectorhash=vectorhash, hopfield=hopfield, cfg=cfg, device=device,
        iters_per_block=iters_per_block, max_steps=max_steps, rng=rng,
        goal_in_mem=goal_in_mem, stored_at_goal_count=stored_at_goal_count,
        deterministic=deterministic,
        oracle_store_at_goal=oracle_store_at_goal,
        suppress_off_goal_stores=oracle_store_at_goal,
        lock_store_after_goal=lock_store_after_goal,
        on_block_end=_record_block,
    ):
        r = step.record
        env_iters[step.env_idx].append(
            (step.iteration, int(r.reached), int(r.stored_at_goal),
             int(r.stored_off_goal)))

    # Per-env primary success: mean success during own block.
    per_env_primary: list[float] = []
    per_env_final_revisit: list[float] = []
    for j in range(N):
        pts = env_iters[j]
        if not pts:
            per_env_primary.append(float("nan"))
            per_env_final_revisit.append(float("nan"))
            continue
        # own primary block: iters in [boundaries[j-1] (or 0), boundaries[j])
        b_lo = boundaries[j - 1] if j > 0 else 0
        b_hi = boundaries[j]
        prim = [p[1] for p in pts if b_lo <= p[0] < b_hi]
        per_env_primary.append(float(np.mean(prim)) if prim else float("nan"))
        # final revisit (only for j < N - 1): iters in last block
        if j < N - 1:
            f_lo = boundaries[N - 2]
            f_hi = boundaries[N - 1]
            final_rv = [p[1] for p in pts if f_lo <= p[0] < f_hi]
            per_env_final_revisit.append(
                float(np.mean(final_rv)) if final_rv else float("nan")
            )
        else:
            per_env_final_revisit.append(float("nan"))

    mean_prim = float(np.nanmean(per_env_primary)) if per_env_primary else float("nan")
    final_vals = [v for v in per_env_final_revisit if not np.isnan(v)]
    mean_final_rv = float(np.mean(final_vals)) if final_vals else float("nan")

    return {
        "params": {
            "iters_per_block": int(iters_per_block),
            "max_steps": int(max_steps),
            "seed": int(seed),
            "deterministic": bool(deterministic),
        },
        "env_iters": {int(j): env_iters[j] for j in range(N)},
        "boundaries": boundaries,
        "summary": {
            "per_env_primary_success": per_env_primary,
            "per_env_final_revisit_success": per_env_final_revisit,
            "mean_primary_success": mean_prim,
            "mean_final_revisit_success": mean_final_rv,
            "interference_drop": (mean_prim - mean_final_rv)
            if not np.isnan(mean_prim) and not np.isnan(mean_final_rv)
            else float("nan"),
            "hopfield_final_memories": int(hopfield.num_memories),
            "stored_at_goal_count": stored_at_goal_count,
        },
    }
