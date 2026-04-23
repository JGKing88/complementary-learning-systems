"""PPO: GAE computation and clipped policy update.

Handles both discrete and continuous movement actions + binary store action.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli, Normal

from .config import PPOConfig


# ---------------------------------------------------------------------------
# Rollout data
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    """Collected rollout data for PPO update."""
    obs: torch.Tensor               # (B, T, input_dim)
    move_actions: torch.Tensor      # (B, T) int for discrete, (B, T, 2) for continuous
    store_actions: torch.Tensor     # (B, T) float 0/1
    move_log_probs: torch.Tensor    # (B, T)
    store_log_probs: torch.Tensor   # (B, T)
    values: torch.Tensor            # (B, T)
    rewards: torch.Tensor           # (B, T)
    bootstrap_value: torch.Tensor   # (B,) — value at truncation point
    goal_reached: torch.Tensor           # (B, T) float 0/1 — BC label for store head
    explore_mask: torch.Tensor      # (B, T) float 0/1 — 1 during explore phase (store-eligible steps)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    No terminal states within a rollout — only truncation at the end.

    rewards, values: (B, T).  bootstrap_value: (B,).
    Returns (advantages, returns) both (B, T).
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B, device=rewards.device)
    last_value = bootstrap_value

    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * last_value - values[:, t]
        last_adv = delta + gamma * lam * last_adv
        advantages[:, t] = last_adv
        last_value = values[:, t]

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def _pool_rollouts(
    rollouts: list[RolloutBatch],
    gamma: float,
    gae_lambda: float,
) -> dict[str, torch.Tensor]:
    """Concatenate rollouts along the batch (trajectory) axis.

    GAE is computed PER rollout (each has its own bootstrap_value and its
    trajectories live on their own timeline), then advantages/returns are
    concatenated. All other per-trajectory tensors concat along dim 0.

    Returns a dict of (N, T, ...) tensors where N = sum_k B_k.
    """
    advs, rets = [], []
    for r in rollouts:
        a, g = compute_gae(r.rewards, r.values, r.bootstrap_value, gamma, gae_lambda)
        advs.append(a)
        rets.append(g)

    return {
        "obs": torch.cat([r.obs for r in rollouts], dim=0),
        "move_actions": torch.cat([r.move_actions for r in rollouts], dim=0),
        "store_actions": torch.cat([r.store_actions for r in rollouts], dim=0),
        "old_move_lp": torch.cat([r.move_log_probs for r in rollouts], dim=0),
        "old_store_lp": torch.cat([r.store_log_probs for r in rollouts], dim=0),
        "goal_reached": torch.cat([r.goal_reached for r in rollouts], dim=0),
        "explore_mask": torch.cat([r.explore_mask for r in rollouts], dim=0),
        "advantages": torch.cat(advs, dim=0),
        "returns": torch.cat(rets, dim=0),
    }


def ppo_update(
    agent: nn.Module,
    rollouts: list[RolloutBatch],
    cfg: PPOConfig,
    optimizer: torch.optim.Optimizer,
    aux_scale: float = 1.0,
) -> dict[str, float]:
    """Run PPO epochs on a pooled rollout buffer with minibatching.

    `rollouts` is the full set of rollouts collected this update (across all
    worlds + envs). They are concatenated along the trajectory axis; advantages
    are computed per-rollout (GAE is trajectory-local) and normalized globally
    across the pool. Each epoch shuffles trajectory indices and splits them
    into `cfg.n_minibatches` minibatches; each minibatch is one full gradient
    step.

    Minibatching is done over full trajectories (rows of length T), not over
    individual timesteps — this preserves each trajectory's RNN temporal
    structure when the agent re-forwards it.

    Args:
        aux_scale: Multiplier applied to auxiliary losses (store_bc_weight) this update,
            for linear annealing from 1.0 → 0.0 over training.

    Returns dict of loss components (averaged over all gradient steps) for logging.
    """
    pool = _pool_rollouts(rollouts, cfg.gamma, cfg.gae_lambda)

    obs = pool["obs"]                   # (N, T, D)
    move_actions = pool["move_actions"] # (N, T) or (N, T, 2)
    store_actions = pool["store_actions"]
    old_move_lp = pool["old_move_lp"]
    old_store_lp = pool["old_store_lp"]
    goal_reached = pool["goal_reached"]
    explore_mask = pool["explore_mask"]
    advantages = pool["advantages"]
    returns = pool["returns"]

    # Normalize advantages across the full pool (not per-minibatch — that would
    # inject minibatch-dependent bias into the policy gradient).
    adv_mean = advantages.mean()
    adv_std = advantages.std().clamp_min(1e-8)
    advantages = (advantages - adv_mean) / adv_std

    effective_bc_weight = cfg.store_bc_weight * aux_scale

    N = obs.shape[0]
    n_mb = max(1, min(cfg.n_minibatches, N))
    mb_size = max(1, N // n_mb)

    total_move_loss = 0.0
    total_store_loss = 0.0
    total_value_loss = 0.0
    total_move_ent = 0.0
    total_store_ent = 0.0
    total_store_bc = 0.0
    n_steps = 0

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(N, device=obs.device)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]
            if idx.numel() == 0:
                continue

            mb_obs = obs[idx]
            mb_move_act = move_actions[idx]
            mb_store_act = store_actions[idx]
            mb_old_move_lp = old_move_lp[idx]
            mb_old_store_lp = old_store_lp[idx]
            mb_adv = advantages[idx]
            mb_ret = returns[idx]
            mb_goal = goal_reached[idx]
            mb_mask = explore_mask[idx]

            move_dist, store_dist, new_values, _ = agent(mb_obs)

            # Movement policy loss
            new_move_lp = move_dist.log_prob(mb_move_act)
            if isinstance(move_dist, Normal):  # continuous: sum over action dims
                new_move_lp = new_move_lp.sum(-1)
            ratio_move = torch.exp(new_move_lp - mb_old_move_lp)
            surr1 = ratio_move * mb_adv
            surr2 = torch.clamp(ratio_move, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
            move_loss = -torch.min(surr1, surr2).mean()

            # Store policy loss — masked by explore_mask: during exploit the
            # store action is inert (rollout ignores it, no store_cost/
            # store_bonus apply, Hopfield is frozen), so those timesteps carry
            # zero causal signal for the store head. Including them just pumps
            # variance into the store logits and the shared RNN trunk.
            new_store_lp = store_dist.log_prob(mb_store_act)
            ratio_store = torch.exp(new_store_lp - mb_old_store_lp)
            surr1_s = ratio_store * mb_adv
            surr2_s = torch.clamp(ratio_store, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
            mask_sum = mb_mask.sum().clamp_min(1.0)
            store_loss = (-torch.min(surr1_s, surr2_s) * mb_mask).sum() / mask_sum

            # Value loss
            value_loss = ((mb_ret - new_values) ** 2).mean()

            # Entropy — move is always causal, store only during explore phase.
            move_entropy = move_dist.entropy()
            if move_entropy.dim() > 2:
                move_entropy = move_entropy.sum(-1)
            move_ent = move_entropy.mean()
            store_ent = (store_dist.entropy() * mb_mask).sum() / mask_sum

            # Auxiliary BCE loss on store head: directly teach "fire store at
            # goal". Only applied where the store action is eligible (explore
            # phase). pos_weight compensates for class imbalance — goal_reached
            # timesteps are a small fraction of the batch, so an unweighted BCE
            # drives the network to always predict 0 (never store).
            # pos_weight = n_neg / n_pos restores balance.
            if effective_bc_weight > 0:
                store_logits = store_dist.logits
                masked_goal = mb_goal * mb_mask
                n_pos = masked_goal.sum().clamp_min(1.0)
                n_neg = (mb_mask - masked_goal).sum().clamp_min(1.0)
                pos_weight = n_neg / n_pos
                bce_full = F.binary_cross_entropy_with_logits(
                    store_logits, mb_goal, reduction="none",
                    pos_weight=pos_weight,
                )
                store_bc_loss = (bce_full * mb_mask).sum() / mask_sum
            else:
                store_bc_loss = torch.zeros((), device=obs.device)

            loss = (
                move_loss
                + store_loss
                + cfg.vf_coef * value_loss
                - cfg.ent_coef * move_ent
                - cfg.store_ent_coef * store_ent
                + effective_bc_weight * store_bc_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_move_loss += move_loss.item()
            total_store_loss += store_loss.item()
            total_value_loss += value_loss.item()
            total_move_ent += move_ent.item()
            total_store_ent += store_ent.item()
            total_store_bc += store_bc_loss.item()
            n_steps += 1

    denom = max(n_steps, 1)
    return {
        "move_loss": total_move_loss / denom,
        "store_loss": total_store_loss / denom,
        "value_loss": total_value_loss / denom,
        "move_entropy": total_move_ent / denom,
        "store_entropy": total_store_ent / denom,
        "store_bc_loss": total_store_bc / denom,
    }
