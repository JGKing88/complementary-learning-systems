"""PPO: GAE computation and clipped policy update.

Handles both discrete and continuous movement actions + binary store action.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
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

def ppo_update(
    agent: nn.Module,
    rollout: RolloutBatch,
    cfg: PPOConfig,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Run PPO epochs on collected rollout.

    Returns dict of loss components for logging.
    """
    obs = rollout.obs
    move_actions = rollout.move_actions
    store_actions = rollout.store_actions
    old_move_lp = rollout.move_log_probs
    old_store_lp = rollout.store_log_probs

    # Compute advantages
    advantages, returns = compute_gae(
        rollout.rewards, rollout.values, rollout.bootstrap_value,
        cfg.gamma, cfg.gae_lambda,
    )
    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std().clamp_min(1e-8)
    advantages = (advantages - adv_mean) / adv_std

    total_move_loss = 0.0
    total_store_loss = 0.0
    total_value_loss = 0.0
    total_move_ent = 0.0
    total_store_ent = 0.0

    for _ in range(cfg.ppo_epochs):
        move_dist, store_dist, new_values, _ = agent(obs)

        # Movement policy loss
        new_move_lp = move_dist.log_prob(move_actions)
        if isinstance(move_dist, Normal):  # continuous: sum over action dims
            new_move_lp = new_move_lp.sum(-1)
        ratio_move = torch.exp(new_move_lp - old_move_lp)
        surr1 = ratio_move * advantages
        surr2 = torch.clamp(ratio_move, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * advantages
        move_loss = -torch.min(surr1, surr2).mean()

        # Store policy loss (same advantages)
        new_store_lp = store_dist.log_prob(store_actions)
        ratio_store = torch.exp(new_store_lp - old_store_lp)
        surr1_s = ratio_store * advantages
        surr2_s = torch.clamp(ratio_store, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * advantages
        store_loss = -torch.min(surr1_s, surr2_s).mean()

        # Value loss
        value_loss = ((returns - new_values) ** 2).mean()

        # Entropy
        move_entropy = move_dist.entropy()
        if move_entropy.dim() > 2:
            move_entropy = move_entropy.sum(-1)
        move_ent = move_entropy.mean()
        store_ent = store_dist.entropy().mean()

        # Total loss
        loss = (
            move_loss
            + store_loss
            + cfg.vf_coef * value_loss
            - cfg.ent_coef * move_ent
            - cfg.store_ent_coef * store_ent
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

    n = cfg.ppo_epochs
    return {
        "move_loss": total_move_loss / n,
        "store_loss": total_store_loss / n,
        "value_loss": total_value_loss / n,
        "move_entropy": total_move_ent / n,
        "store_entropy": total_store_ent / n,
    }
