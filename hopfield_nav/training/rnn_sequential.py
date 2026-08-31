"""The RNN baseline's continual-learning block loop, shared by its two drivers.

``train_rnn.train_sequential`` (the training entry point, which logs to wandb)
and ``final_plotting.baseline.run_sequential`` (the figure driver, which emits
a history JSON) ran the same loop from two copies: one block per env, and per
update collect a rollout, apply a BC update, then evaluate every env introduced
so far. What differed was how many eval trials, what gets recorded, and what
gets printed -- so those are the parameters, and the loop is not.

Deliberately NOT merged with the Hopfield sequential protocol in
``evaluation/protocols.py``. The RNN baseline is a control: its value comes
from sharing only the environment with the Hopfield stack, and its inner step
trains the agent, which the Hopfield protocol's never does.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from ..continual.base import ContinualMethod, NoMethod
from ..updates.bc_rnn import bc_rnn_update
from ..world.env import GridEnv
from ..evaluation.rnn import evaluate_nav_all
from ..rollout.rnn import collect_rollout_rnn
from ..world.vec_env import make_vec


@dataclass
class UpdateResult:
    """One training update and the evaluation that followed it."""
    global_step: int        # 1-indexed, across all blocks
    block: int              # index of the env being trained
    update: int             # 1-indexed within this block
    rollout: object         # RolloutBatchRNN
    losses: dict            # whatever bc_rnn_update returned
    metrics: dict           # {env_idx: metrics} for every env introduced so far


def run_sequential_blocks(
    *,
    cfg,
    agent,
    optimizer: torch.optim.Optimizer,
    envs: list[GridEnv],
    device: torch.device,
    n_eval_trials: int,
    sgb: np.ndarray | None = None,
    env_offsets: list[tuple[int, int]] | None = None,
    on_update: Callable[[UpdateResult], None] | None = None,
    on_block_start: Callable[[int, GridEnv], None] | None = None,
    method: ContinualMethod | None = None,
    reset_optimizer_each_block: bool = False,
) -> list[tuple[int, int, int]]:
    """Train each env in turn; evaluate every env introduced so far, every update.

    Untrained envs are excluded from the evaluation -- both to save compute and
    so the forgetting curve does not show pre-training noise for envs that have
    not been touched.

    ``on_update`` receives each update's rollout, losses and metrics; the caller
    decides what to record and what to print. Returns ``blocks`` as
    ``(start_step_inclusive, end_step_inclusive, env_idx)``.

    ``method`` is a `hopfield_nav.continual.ContinualMethod`; the default is the
    no-op, which reproduces naive sequential SGD exactly. The block loop is
    where every method in the suite intervenes, so the hooks live here rather
    than in the update: replay contributes extra batches, regularisers
    contribute a loss term, and both get told where the block boundaries are.
    Note the ordering -- `extra_batches` is asked for *before* `after_update`
    stores the new rollout, so a replayed trajectory is always genuinely older
    than the one driving the update.

    ``reset_optimizer_each_block`` clears Adam's moment estimates at every task
    boundary. Off by default, because that is what every recorded history did.
    It is worth sweeping (plan section 3.1, W2): the optimizer is built once and
    its second moments carry across boundaries, so the first steps in env `i`
    are scaled by statistics gathered on env `i-1` -- which is one of the
    mechanisms behind the stability gap. The state is cleared in place rather
    than by rebuilding the optimizer, so the parameter groups and the learning
    rate survive untouched.
    """
    method = method or NoMethod()
    movement_mode = cfg.agent.movement_mode
    blocks: list[tuple[int, int, int]] = []
    global_step = 0

    for i, env in enumerate(envs):
        if on_block_start is not None:
            on_block_start(i, env)
        method.on_block_start(i, agent, envs)
        if reset_optimizer_each_block and i > 0:
            # In place, so param_groups and the lr are untouched -- only the
            # per-parameter moment history goes. Skipped at i=0, where there is
            # nothing to carry over and clearing would only discard whatever a
            # pretraining checkpoint handed us.
            optimizer.state = defaultdict(dict)
        block_start = global_step + 1
        vec = make_vec(env, cfg.batch_envs, movement_mode,
                       cfg.env.continuous_scale,
                       continuous_normalize=cfg.env.continuous_normalize)
        env_offset_i = env_offsets[i] if env_offsets is not None else None

        for upd in range(1, cfg.updates_per_env + 1):
            vec.reset_all()
            rollout = collect_rollout_rnn(
                vec, agent, cfg.agent, cfg.steps_per_rollout, device,
                deterministic=False, teacher_force=False,
                sgb=sgb, env_offset=env_offset_i,
            )
            extra = method.extra_batches(rollout, i)
            losses = bc_rnn_update(
                agent, [rollout] + list(extra), cfg.bc, optimizer,
                movement_mode,
                penalty_fn=lambda: method.penalty(agent),
                aux_loss_fn=lambda: method.aux_loss(agent, rollout, extra),
            )
            method.after_update(rollout, i, agent)
            losses["n_replay_batches"] = float(len(extra))
            global_step += 1

            metrics = evaluate_nav_all(
                envs[: i + 1], agent, n_eval_trials, cfg.eval_max_steps,
                device, deterministic=True,
                continuous_scale=cfg.env.continuous_scale,
                continuous_normalize=cfg.env.continuous_normalize,
                sgb=sgb,
                env_offsets=(env_offsets[: i + 1]
                             if env_offsets is not None else None),
                reset_state_on_teleport=cfg.env.reset_state_on_teleport,
            )
            if on_update is not None:
                on_update(UpdateResult(
                    global_step=global_step, block=i, update=upd,
                    rollout=rollout, losses=losses, metrics=metrics))

        method.on_block_end(i, agent, envs)
        blocks.append((block_start, global_step, i))

    return blocks


__all__ = ["UpdateResult", "run_sequential_blocks"]
