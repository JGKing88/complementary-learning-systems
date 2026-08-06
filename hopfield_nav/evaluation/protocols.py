"""The sequential continual-learning protocol, in one implementation.

Two copies of this existed. ``eval.evaluate_sequential_episodes`` drives the
in-training metric and the ``eval_all`` JSON; ``final_plotting.agenthash``
drives the paper figure. Same protocol, ~130 lines apart, and they had already
started to diverge -- agenthash grew a store-trainer hook, path-length
tracking, and a second oracle flag that eval expresses only as the conjunction
of one.

Divergence here is expensive in a specific way: the two produce numbers that get
compared to each other, so a drift shows up as a scientific result rather than
as a bug.

What varies between the callers, and is therefore a parameter:

    oracle_store_at_goal        force a store when the agent sits on the goal,
                                bypassing an untrained store head
    suppress_off_goal_stores    drop stores fired anywhere else. eval couples
                                this to oracle_store_at_goal (one flag, both
                                meanings); agenthash splits them. Passing the
                                same value for both reproduces eval's flag.
    lock_store_after_goal       once this env's goal is in memory, stop storing
    on_step                     per-step callback, for the figure pipeline's
                                online store-head training

What does not vary is the at-goal semantics, which mirror training: the success
iteration is the one where the agent *sits* on the goal at decision time, reads
``reward=+1`` and may fire store. Training would teleport at that point; a
mini-episode ends instead. That is the ENDS_ON_ARRIVAL row of
``world/episode.SITE_CONTRACTS``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import torch

from ..world.env import GridEnv, at_goal
from ..world import episode


@dataclass(frozen=True)
class EpisodeRecord:
    """Outcome of one mini-episode.

    ``steps_to_goal`` and ``path_to_goal`` are None on timeout. Both are always
    recorded now, including on the eval path, which previously discarded them --
    that is what lets ``eval_all --seq-iters-per-block`` emit a history the
    plotting code can render without re-running the protocol through agenthash.
    """
    reached: int
    steps_to_goal: int | None
    path_to_goal: float | None
    stored_at_goal: int
    stored_off_goal: int


def env_xy_float(env: GridEnv) -> np.ndarray:
    """Float (x, y). Continuous envs expose ``_continuous_pos``; discrete envs
    use the snapped integer position. Used to accumulate path length."""
    cp = getattr(env, "_continuous_pos", None)
    if cp is not None:
        return np.asarray(cp, dtype=np.float64).copy()
    return np.asarray(env._pos, dtype=np.float64)


def run_mini_episode(
    *,
    agent,
    env: GridEnv,
    env_offset: tuple[int, int],
    vectorhash,
    hopfield,
    cfg,
    device: torch.device,
    local_idx: int,
    allow_store: bool,
    max_steps: int,
    rng: np.random.RandomState,
    goal_in_mem: dict[int, bool],
    stored_at_goal_count: dict[int, int],
    deterministic: bool = True,
    oracle_store_at_goal: bool = False,
    suppress_off_goal_stores: bool = False,
    lock_store_after_goal: bool = False,
    on_step: Callable[[torch.Tensor, bool], None] | None = None,
) -> EpisodeRecord:
    """One mini-episode from a random non-goal start, up to ``max_steps``.

    Mutates ``hopfield``, ``goal_in_mem`` and ``stored_at_goal_count`` in place:
    the protocol's whole point is that memory accumulates across episodes and
    blocks, so these are shared state, not outputs.

    ``on_step(features, at_goal)`` is called after each agent step with the
    last-layer recurrent features -- the same trunk output the store head reads.
    """
    # Deferred import: agent_step still lives in eval.py, which imports this
    # module. Phase 6 moves it to evaluation/agent_step.py and this becomes a
    # normal top-level import.
    from .metrics import agent_step, random_start

    episode.require_single_env_support(
        episode.contract_for("evaluate_sequential_episodes"),
        "run_mini_episode")

    goal = env.goal_location
    env.set_position(random_start(env.size, goal, rng))
    h_rnn = None
    prev_reward = None
    prev_action = None
    reached = False
    steps_to_goal: int | None = None
    path_to_goal: float | None = None
    path_acc = 0.0
    prev_xy = env_xy_float(env)
    stored_at_goal = 0
    stored_off_goal = 0

    for step in range(max_steps):
        at_g = at_goal(env)
        out = agent_step(
            agent, env, env_offset, vectorhash, hopfield,
            h_rnn, cfg, device, deterministic=deterministic,
            goal_local=goal, goal_in_memory=goal_in_mem[local_idx],
            prev_reward=prev_reward, prev_action=prev_action,
        )
        h_rnn = out["h_rnn"]
        prev_reward = out["next_prev_reward"]
        prev_action = out["next_prev_action"]

        if on_step is not None:
            # h_rnn is (num_layers, B=1, hidden); the last layer's output is
            # the trunk feature the store head reads at this step.
            on_step(h_rnn[-1, 0, :], bool(at_g))

        allow_store_now = allow_store and not (
            lock_store_after_goal and goal_in_mem[local_idx]
        )
        agent_wants_store = out["store_action"] > 0.5

        if at_g:
            store_fired = True if oracle_store_at_goal else agent_wants_store
            if allow_store_now and store_fired:
                hopfield.input_memory(out["store_embedding"][0])
                goal_in_mem[local_idx] = True
                stored_at_goal_count[local_idx] += 1
                stored_at_goal = 1
            reached = True
            steps_to_goal = step
            path_to_goal = path_acc
            break

        # Off goal: accumulate the step just taken, then handle the store.
        new_xy = env_xy_float(env)
        path_acc += float(np.linalg.norm(new_xy - prev_xy))
        prev_xy = new_xy

        if not suppress_off_goal_stores:
            if allow_store_now and agent_wants_store:
                hopfield.input_memory(out["store_embedding"][0])
                stored_off_goal = 1

    return EpisodeRecord(
        reached=int(reached),
        steps_to_goal=steps_to_goal,
        path_to_goal=path_to_goal,
        stored_at_goal=stored_at_goal,
        stored_off_goal=stored_off_goal,
    )


@dataclass(frozen=True)
class ProtocolStep:
    """One mini-episode's place in the schedule, plus its outcome."""
    block: int          # index of the env whose block this is
    iteration: int      # global outer-iteration counter
    env_idx: int        # which env this mini-episode ran in (<= block)
    is_primary: bool    # env_idx == block, i.e. stores are allowed
    record: EpisodeRecord


def run_sequential_protocol(
    *,
    agent,
    val_envs: list[GridEnv],
    env_offsets: list[tuple[int, int]],
    vectorhash,
    hopfield,
    cfg,
    device: torch.device,
    iters_per_block: int,
    max_steps: int,
    rng: np.random.RandomState,
    goal_in_mem: dict[int, bool],
    stored_at_goal_count: dict[int, int],
    deterministic: bool = True,
    oracle_store_at_goal: bool = False,
    suppress_off_goal_stores: bool = False,
    lock_store_after_goal: bool = False,
    on_step: Callable[[torch.Tensor, bool], None] | None = None,
    on_episode_end: Callable[[ProtocolStep], None] | None = None,
    on_block_end: Callable[[int, int], None] | None = None,
) -> Iterator[ProtocolStep]:
    """Yield every mini-episode of the protocol, in order.

    Envs are introduced one at a time. Within block ``i``, each outer iteration
    runs one mini-episode in every env introduced so far (``j <= i``); only the
    primary (``j == i``) may store. The Hopfield is never reset -- it
    accumulates across blocks, which is what makes the revisit episodes a
    measure of interference.

    A generator, so the two callers can accumulate the results however their
    output schema requires without this function knowing about either. The
    control flow is the shared part; the bookkeeping is not.
    """
    cur_iter = 0
    for i in range(len(val_envs)):
        block_start = cur_iter
        for _k in range(iters_per_block):
            for j in range(i + 1):
                record = run_mini_episode(
                    agent=agent, env=val_envs[j], env_offset=env_offsets[j],
                    vectorhash=vectorhash, hopfield=hopfield, cfg=cfg,
                    device=device, local_idx=j, allow_store=(j == i),
                    max_steps=max_steps, rng=rng, goal_in_mem=goal_in_mem,
                    stored_at_goal_count=stored_at_goal_count,
                    deterministic=deterministic,
                    oracle_store_at_goal=oracle_store_at_goal,
                    suppress_off_goal_stores=suppress_off_goal_stores,
                    lock_store_after_goal=lock_store_after_goal,
                    on_step=on_step,
                )
                step = ProtocolStep(block=i, iteration=cur_iter, env_idx=j,
                                    is_primary=(j == i), record=record)
                if on_episode_end is not None:
                    on_episode_end(step)
                yield step
            cur_iter += 1
        if on_block_end is not None:
            on_block_end(i, cur_iter)


__all__ = [
    "EpisodeRecord",
    "ProtocolStep",
    "env_xy_float",
    "run_mini_episode",
    "run_sequential_protocol",
]
