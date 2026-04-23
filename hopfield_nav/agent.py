"""NavAgent: RNN policy with movement, store, and value heads."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli, Normal

from .config import AgentConfig


def compute_input_dim(cfg: AgentConfig, embed_dim: int) -> int:
    """Compute RNN input dimension from config."""
    dim = 1  # prev_reward (always)
    if cfg.input_encoded_state:
        dim += embed_dim
    if cfg.input_hopfield_signal:
        dim += 4 if cfg.hopfield_mode == "discrete" else 2
    if cfg.input_prev_action:
        dim += 4 if cfg.movement_mode == "discrete" else 2
    return dim


class NavAgent(nn.Module):
    """Three-headed RNN policy for Hopfield navigation.

    Outputs:
        movement: Categorical(4) for discrete, DiagGaussian(2) for continuous
        store: Bernoulli(1) — whether to store current state in Hopfield
        value: scalar value estimate
    """

    def __init__(self, cfg: AgentConfig, input_dim: int) -> None:
        super().__init__()
        self.cfg = cfg

        self.rnn = nn.GRU(
            input_dim, cfg.hidden_size,
            num_layers=cfg.num_rnn_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_rnn_layers > 1 else 0.0,
        )

        # Movement head
        if cfg.movement_mode == "discrete":
            self.movement_head = nn.Linear(cfg.hidden_size, 4)
        else:
            self.movement_mean = nn.Linear(cfg.hidden_size, 2)
            # Init log_std negative so std < 1: deterministic eval (action = mean)
            # is meaningful. With zero-init, training entropy bonus drives std up
            # to 2-3 per dim, making the policy reliant on sampling noise for motion.
            self.movement_log_std = nn.Parameter(torch.full((2,), cfg.init_log_std))

        # Store head (binary)
        self.store_head = nn.Linear(cfg.hidden_size, 1)

        # Value head
        self.value_head = nn.Linear(cfg.hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple:
        """Forward pass.

        x: (B, T, input_dim)
        h: (num_layers, B, hidden_size) or None

        Returns (move_dist, store_dist, values, h_next):
            move_dist: Categorical or Normal distribution
            store_dist: Bernoulli distribution
            values: (B, T)
            h_next: (num_layers, B, hidden_size)
        """
        features, h_next = self.rnn(x, h)  # (B, T, hidden)

        # Movement distribution
        if self.cfg.movement_mode == "discrete":
            logits = self.movement_head(features)  # (B, T, 4)
            move_dist = Categorical(logits=logits)
        else:
            mean = self.movement_mean(features)  # (B, T, 2)
            std = self.movement_log_std.exp().expand_as(mean)
            move_dist = Normal(mean, std)

        # Store distribution
        store_logits = self.store_head(features).squeeze(-1)  # (B, T)
        store_dist = Bernoulli(logits=store_logits)

        # Value
        values = self.value_head(features).squeeze(-1)  # (B, T)

        return move_dist, store_dist, values, h_next

    @torch.no_grad()
    def get_action_and_value(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        deterministic: bool = False,
        move_action_override: torch.Tensor | None = None,
        move_override_mask: torch.Tensor | None = None,
    ) -> dict:
        """Single-step action selection for rollout collection.

        x: (B, 1, input_dim)

        Optional teacher forcing on the movement action:
            move_action_override: (B, 1) long for discrete, (B, 1, 2) float for continuous.
            move_override_mask:   (B,) bool — envs where the override should replace the sample.
        When the mask is True for env b, move_action[b] is set to move_action_override[b] and
        its log_prob is re-scored under the current policy so PPO's importance ratio is
        well-defined for the action that was actually executed.

        Returns dict with: move_action, store_action, move_log_prob,
                           store_log_prob, value, h_next
        """
        move_dist, store_dist, values, h_next = self.forward(x, h)

        if deterministic:
            if self.cfg.movement_mode == "discrete":
                move_action = move_dist.probs.argmax(-1)  # (B, 1)
            else:
                move_action = move_dist.mean  # (B, 1, 2)
            store_action = (store_dist.probs > 0.5).float()  # (B, 1)
        else:
            move_action = move_dist.sample()  # (B, 1) or (B, 1, 2)
            store_action = store_dist.sample()  # (B, 1)

        if move_action_override is not None and move_override_mask is not None:
            if self.cfg.movement_mode == "discrete":
                # move_action: (B, 1) long; override: (B, 1) long; mask: (B,) bool
                move_action = torch.where(
                    move_override_mask.unsqueeze(-1),
                    move_action_override.to(move_action.dtype),
                    move_action,
                )
            else:
                # move_action: (B, 1, 2) float; override: (B, 1, 2) float; mask: (B,) bool
                move_action = torch.where(
                    move_override_mask.view(-1, 1, 1),
                    move_action_override.to(move_action.dtype),
                    move_action,
                )

        move_log_prob = move_dist.log_prob(move_action)
        if self.cfg.movement_mode == "continuous":
            move_log_prob = move_log_prob.sum(-1)  # sum over action dims
        store_log_prob = store_dist.log_prob(store_action)

        return {
            "move_action": move_action.squeeze(1),           # (B,) or (B, 2)
            "store_action": store_action.squeeze(1),          # (B,)
            "move_log_prob": move_log_prob.squeeze(1),        # (B,)
            "store_log_prob": store_log_prob.squeeze(1),      # (B,)
            "value": values.squeeze(1),                       # (B,)
            "h_next": h_next,
        }
