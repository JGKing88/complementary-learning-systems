"""What a rollout produces.

`RolloutBatch` lived in `ppo.py`, which made `rollout.collector` -- the module
that *builds* one -- import from `updates`, a layer above it. The record is not
PPO's: `bc.py` consumes the same object, and its DAgger teacher-label fields are
`None` in PPO mode. It belongs beside the collector that fills it.

(The RNN baseline has its own `RNNRolloutBatch` in `rnn.py`. It is a control
model that deliberately shares only the environment, so the two are not merged.)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


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
    # (B, T) float 0/1 -- 1 while row b's episode was still running. None
    # means every row ran the whole rollout, which is every run that does
    # not end episodes on goal-reach; that path keeps the old arithmetic
    # exactly rather than multiplying by a mask of ones.
    alive_mask: torch.Tensor | None = None
    policy_action_mask: torch.Tensor | None = None  # (B, T) float 0/1 — 1 where executed action came from policy sample, 0 where ε / auto-nav override replaced it. ε actions are env-exploration only and including them in the PPO surrogate explodes ratios under narrow std (the action lies far from the policy mean → log_prob is large negative → tiny mean drift → huge ratio).
    # DAgger teacher labels — populated only in training_mode == "bc". All
    # None in PPO mode; PPO update ignores these fields entirely.
    teacher_move_action: torch.Tensor | None = None   # (B, T) long | (B, T, 2) float
    teacher_store_action: torch.Tensor | None = None  # (B, T) float
    move_label_mask: torch.Tensor | None = None       # (B, T) float 0/1
    store_label_mask: torch.Tensor | None = None      # (B, T) float 0/1
    # (B, T, 8) float 0/1 -- for each of 8 compass directions at
    # `aux_visited_radius`, had the agent already visited that cell at
    # that step? Target for the auxiliary visitation head (§24.2 lever
    # B). None unless aux_visited_weight > 0.
    visited_targets: torch.Tensor | None = None
    trust_hop_mask: torch.Tensor | None = None        # (B, T) float 0/1 — 1 when teacher's move label was a Hopfield-direction (post-store-at-goal) label, 0 when novelty. Used for upweighting nav labels in BC loss.

__all__ = ["RolloutBatch"]
