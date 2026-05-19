"""BC update for the vanilla-RNN baseline.

Pure CE on the movement teacher action, masked by `move_label_mask`. No store
head, no value head, no Hopfield. Mirrors `bc.py:bc_update`'s minibatched
trajectory-level structure.
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from .agent_rnn import RNNAgent
from .config import RNNBCConfig
from .rollout_rnn import RNNRolloutBatch


def bc_rnn_update(
    agent: RNNAgent,
    rollouts: list[RNNRolloutBatch],
    cfg: RNNBCConfig,
    optimizer: torch.optim.Optimizer,
    movement_mode: str,
) -> dict[str, float]:
    """CE on movement teacher, masked. Returns averaged scalars per minibatch."""
    obs = torch.cat([r.obs for r in rollouts], dim=0)                     # (N, T, D)
    tm  = torch.cat([r.teacher_move_action for r in rollouts], dim=0)     # (N, T) or (N, T, 2)
    mm  = torch.cat([r.move_label_mask for r in rollouts], dim=0)         # (N, T)

    if cfg.only_train_on_reached:
        gr = torch.cat([r.goal_reached for r in rollouts], dim=0)         # (N, T)
        reached = (gr.sum(dim=1) > 0)                                     # (N,) bool
        if not reached.any():
            # No trajectory reached the goal in this rollout — skip update.
            return {"move_loss": 0.0, "move_entropy": 0.0}
        obs = obs[reached]
        tm = tm[reached]
        mm = mm[reached]

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

            move_dist, _ = agent(mb_obs)

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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            totals["move_loss"] += move_loss.item()
            totals["move_entropy"] += move_ent.item()
            n_steps += 1

    denom = max(n_steps, 1)
    return {k: v / denom for k, v in totals.items()}
