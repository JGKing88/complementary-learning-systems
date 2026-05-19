"""DAgger-style supervised (behavior-cloning) training mode.

Two pieces:
  - novelty_action_batch_{discrete, continuous}: per-env oracle action used
    before any memory is stored; biases the agent to first-visit neighbors.
  - bc_update: pool rollouts and minimize CE(move) + BCE(store) against the
    teacher labels recorded in the RolloutBatch. Student acted on its own
    sample during rollout (DAgger); labels are what the oracle would have done.

Labels are generated inside RolloutCollector when cfg.training_mode == "bc";
this module consumes them.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import NavAgent
from .config import BCConfig
from .env import CARDINAL_ACTIONS
from .ppo import RolloutBatch


# ---------------------------------------------------------------------------
# Novelty oracles
# ---------------------------------------------------------------------------

def novelty_action_batch_discrete(
    positions: np.ndarray,
    visited_cells: np.ndarray,
    size: int,
    rng: np.random.RandomState,
    fallback: str = "random",
) -> np.ndarray:
    """Pick a cardinal action per env that lands on an unvisited cell.

    positions:     (B, 2) int — current cell.
    visited_cells: (B, size, size) bool — cells visited this rollout.
    fallback:      "random" → uniform over all 4 actions when no neighbor is
                   unvisited; "stay" → pick action 0 (agent still nudges).

    Returns (B,) int32 indices into CARDINAL_ACTIONS.
    """
    B = positions.shape[0]
    out = np.zeros(B, dtype=np.int32)
    for b in range(B):
        cx, cy = int(positions[b, 0]), int(positions[b, 1])
        unvisited_cands = []
        all_cands = []
        for a_idx, (dx, dy) in enumerate(CARDINAL_ACTIONS):
            nx = max(0, min(size - 1, cx + dx))
            ny = max(0, min(size - 1, cy + dy))
            all_cands.append(a_idx)
            if not visited_cells[b, nx, ny]:
                unvisited_cands.append(a_idx)
        if unvisited_cands:
            out[b] = unvisited_cands[rng.randint(len(unvisited_cands))]
        elif fallback == "stay":
            out[b] = 0
        else:  # "random"
            out[b] = all_cands[rng.randint(len(all_cands))]
    return out


def novelty_action_batch_continuous(
    positions: np.ndarray,
    visited_cells: np.ndarray,
    size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Unit vector toward the nearest unvisited cell per env.

    positions: (B, 2) float — continuous position (snapping tolerated).
    Returns (B, 2) float32 unit vectors. Falls back to a random unit vector
    if every cell has been visited.
    """
    B = positions.shape[0]
    out = np.zeros((B, 2), dtype=np.float32)
    # Precompute (X, Y) grid once.
    xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for b in range(B):
        mask = ~visited_cells[b]  # (size, size) bool
        if not mask.any():
            theta = rng.uniform(0.0, 2 * np.pi)
            out[b] = [np.sin(theta), np.cos(theta)]
            continue
        dx = xs[mask] - positions[b, 0]
        dy = ys[mask] - positions[b, 1]
        dist2 = dx * dx + dy * dy
        k = int(np.argmin(dist2))
        v = np.array([dx[k], dy[k]], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            theta = rng.uniform(0.0, 2 * np.pi)
            out[b] = [np.sin(theta), np.cos(theta)]
        else:
            out[b] = v / n
    return out


# ---------------------------------------------------------------------------
# BC update
# ---------------------------------------------------------------------------

def bc_update(
    agent: NavAgent,
    rollouts: list[RolloutBatch],
    cfg: BCConfig,
    movement_mode: str,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Supervised update: CE(move | teacher) + store_weight * BCE(store | teacher).

    Rollouts must carry teacher_move_action / teacher_store_action /
    move_label_mask / store_label_mask (populated by RolloutCollector in BC mode).
    Trajectory-level minibatching, same convention as ppo_update.
    """
    obs = torch.cat([r.obs for r in rollouts], dim=0)
    tm  = torch.cat([r.teacher_move_action for r in rollouts], dim=0)
    ts  = torch.cat([r.teacher_store_action for r in rollouts], dim=0)
    mm  = torch.cat([r.move_label_mask for r in rollouts], dim=0)
    sm  = torch.cat([r.store_label_mask for r in rollouts], dim=0)
    th  = torch.cat([r.trust_hop_mask for r in rollouts], dim=0)

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
            mb_ts = ts[idx]
            mb_mm = mm[idx]
            mb_sm = sm[idx]
            mb_th = th[idx]

            move_dist, store_dist, _, _ = agent(mb_obs)

            # Movement CE (teacher action log-prob, masked).
            # Per-step weight: trust_hop steps (post-store-at-goal Hopfield-
            # follow labels) get cfg.nav_weight, novelty-exploration steps get
            # 1.0. With nav_weight > 1 this counteracts the dilution from
            # abundant pre-memory novelty labels overwhelming rare nav labels.
            move_logp = move_dist.log_prob(mb_tm)
            if movement_mode == "continuous":
                move_logp = move_logp.sum(-1)
            step_w = 1.0 + (cfg.nav_weight - 1.0) * mb_th
            weighted_mask = mb_mm * step_w
            m_denom = weighted_mask.sum().clamp_min(1.0)
            move_loss = -(move_logp * weighted_mask).sum() / m_denom

            # Entropy on movement (masked, unweighted — entropy bonus is a
            # uniform regularizer, not a label-dependent signal).
            move_entropy = move_dist.entropy()
            if move_entropy.dim() > 2:
                move_entropy = move_entropy.sum(-1)
            ent_denom = mb_mm.sum().clamp_min(1.0)
            move_ent = (move_entropy * mb_mm).sum() / ent_denom

            # Store BCE against at-goal label, masked. Apply pos_weight to
            # compensate for class imbalance — at-goal is a small fraction of
            # steps, and unweighted BCE collapses the store head to always-zero
            # (BC-mode bug observed empirically: store_loss ~0.02 but
            # store_success at eval ~0%). Matches the PPO-mode scheme.
            store_logits = store_dist.logits
            masked_pos = (mb_ts * mb_sm).sum().clamp_min(1.0)
            masked_neg = ((1.0 - mb_ts) * mb_sm).sum().clamp_min(1.0)
            pos_weight = masked_neg / masked_pos
            if cfg.bce_pos_weight_cap > 0:
                pos_weight = pos_weight.clamp_max(cfg.bce_pos_weight_cap)
            bce = F.binary_cross_entropy_with_logits(
                store_logits, mb_ts, reduction="none",
                pos_weight=pos_weight,
            )
            s_denom = mb_sm.sum().clamp_min(1.0)
            store_loss = (bce * mb_sm).sum() / s_denom

            loss = move_loss + cfg.store_weight * store_loss - cfg.move_ent_coef * move_ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            totals["move_loss"] += move_loss.item()
            totals["store_loss"] += store_loss.item()
            totals["move_entropy"] += move_ent.item()
            n_steps += 1

    denom = max(n_steps, 1)
    return {k: v / denom for k, v in totals.items()}
