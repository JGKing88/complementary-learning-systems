"""BC update for the vanilla-RNN baseline.

Pure CE on the movement teacher action, masked by `move_label_mask`. No store
head, no value head, no Hopfield. Mirrors `bc.py:bc_update`'s minibatched
trajectory-level structure.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn

from ..continual.cost import COUNTER

from ..policy.agent_rnn import RNNAgent
from ..config import RNNBCConfig
from ..rollout.rnn import RNNRolloutBatch


def bc_rnn_update(
    agent: RNNAgent,
    rollouts: list[RNNRolloutBatch],
    cfg: RNNBCConfig,
    optimizer: torch.optim.Optimizer,
    movement_mode: str,
    *,
    penalty_fn: Callable[[], torch.Tensor | None] | None = None,
    aux_loss_fn: Callable[[], torch.Tensor | None] | None = None,
    on_step: Callable[[], None] | None = None,
) -> dict[str, float]:
    """CE on movement teacher, masked. Returns averaged scalars per minibatch.

    The three optional hooks are what continual-learning methods need
    (`hopfield_nav/continual/`), and none of them changes the naive path:

    ``penalty_fn``   a parameter-space term (EWC, SI), recomputed every
                     minibatch step because it depends on the parameters as
                     they currently stand -- a value computed once outside the
                     loop would stop constraining after the first step.
    ``aux_loss_fn``  an output-space term (CLEAR, DER++ distillation), for
                     methods that regularise what the model *does* on specific
                     states rather than where its parameters are.
    ``on_step``      fires after each optimiser step, for methods that
                     accumulate along the optimisation path (SI).

    Replay needs no hook at all: ``rollouts`` is already a list and is already
    concatenated, so a method contributes replayed trajectories simply by
    having them appended. Loss weighting across new and replayed data is then
    by supervised-token count, which is the intended behaviour.
    """
    obs = torch.cat([r.obs for r in rollouts], dim=0)                     # (N, T, D)
    tm  = torch.cat([r.teacher_move_action for r in rollouts], dim=0)     # (N, T) or (N, T, 2)
    mm  = torch.cat([r.move_label_mask for r in rollouts], dim=0)         # (N, T)

    # The recurrent state each rollout *started* from, if it was a continuation
    # of a longer lifetime. Without this the forward pass below begins at zero,
    # which trains the network as though every chunk were the start of its own
    # lifetime -- and that silently caps the horizon it can learn to exploit at
    # `steps_per_rollout`, however long the lifetime really is. That is exactly
    # the defect that made the first in-context measurement meaningless: it was
    # trained on 200-step lifetimes and evaluated on 2000-step ones.
    #
    # All-or-nothing across the batch: a mix of continuations and fresh starts
    # would need per-row zeros, which is representable but has no caller, so it
    # is refused rather than guessed at.
    h0s = [r.initial_h for r in rollouts]
    if any(h is not None for h in h0s):
        if any(h is None for h in h0s):
            raise ValueError(
                "some rollouts carry an initial hidden state and some do not; "
                "concatenating them would train the continuations as fresh "
                "lifetimes. Pass initial_h for all of them or none.")
        h0 = torch.cat(h0s, dim=1)                # (num_layers, N, hidden)
    else:
        h0 = None

    if cfg.only_train_on_reached:
        gr = torch.cat([r.goal_reached for r in rollouts], dim=0)         # (N, T)
        reached = (gr.sum(dim=1) > 0)                                     # (N,) bool
        if not reached.any():
            # No trajectory reached the goal in this rollout — skip update.
            return {"move_loss": 0.0, "move_entropy": 0.0}
        obs = obs[reached]
        tm = tm[reached]
        mm = mm[reached]
        if h0 is not None:
            h0 = h0[:, reached]

    N = obs.shape[0]
    n_mb = max(1, min(cfg.n_minibatches, N))
    mb_size = max(1, N // n_mb)

    totals: dict[str, float] = defaultdict(float)
    n_steps = 0

    for _ in range(cfg.epochs):
        perm = torch.randperm(N, device=obs.device)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]
            if idx.numel() == 0:
                continue

            mb_obs = obs[idx]
            mb_tm = tm[idx]
            mb_mm = mm[idx]

            # Detached: the state is a boundary condition carried in from the
            # previous chunk, not something to backpropagate into it. This is
            # truncated BPTT -- the lifetime is longer than the window, and the
            # window is what fits in memory.
            mb_h0 = h0[:, idx].contiguous().detach() if h0 is not None else None
            # The replayed batches were concatenated into `obs` by the caller,
            # so this single count already covers them -- which is exactly the
            # thing that makes ER's gradient step cost more than naive SGD's.
            COUNTER.add(mb_obs, backward=True)
            move_dist, _ = agent(mb_obs, mb_h0)

            move_logp = move_dist.log_prob(mb_tm)
            if movement_mode == "continuous":
                move_logp = move_logp.sum(-1)
            denom = mb_mm.sum().clamp_min(1.0)
            move_loss = -(move_logp * mb_mm).sum() / denom

            move_entropy = move_dist.entropy()
            if move_entropy.dim() > 2:
                move_entropy = move_entropy.sum(-1)
            move_ent = (move_entropy * mb_mm).sum() / denom

            loss = move_loss - cfg.move_ent_coef * move_ent

            pen = penalty_fn() if penalty_fn is not None else None
            if pen is not None:
                loss = loss + pen
                totals["penalty"] += float(pen.item())
            aux = aux_loss_fn() if aux_loss_fn is not None else None
            if aux is not None:
                loss = loss + aux
                totals["aux_loss"] += float(aux.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()
            if on_step is not None:
                on_step()

            totals["move_loss"] += move_loss.item()
            totals["move_entropy"] += move_ent.item()
            n_steps += 1

    denom = max(n_steps, 1)
    return {k: v / denom for k, v in totals.items()}
