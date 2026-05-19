"""RNNAgent: vanilla GRU policy with a single movement head.

Control baseline for hopfield_nav: no Hopfield, no store head, no value head.
Trained by behavior cloning against a shortest-path oracle (see oracle_bfs.py
and bc_rnn.py). Uses raw sensory observations only — no encoder, no VectorHash.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .config import RNNAgentConfig


def compute_rnn_input_dim(
    cfg: RNNAgentConfig, sensory_dim: int, gbook_dim: int = 0,
) -> int:
    """Sensory always on; optional prev_action / prev_reward / grid_state channels.

    ``gbook_dim`` is the smoothed-gbook channel width (== ``vectorhash.Ng``); it
    is added when ``cfg.input_grid_state`` is True. Caller is responsible for
    computing it from the VectorHash they built.
    """
    dim = sensory_dim
    if cfg.input_prev_action:
        dim += 4 if cfg.movement_mode == "discrete" else 2
    if cfg.input_prev_reward:
        dim += 1
    if cfg.input_grid_state:
        dim += gbook_dim
    return dim


class RNNAgent(nn.Module):
    """GRU + single linear move head. forward(x, h) -> (move_dist, h_next)."""

    def __init__(self, cfg: RNNAgentConfig, input_dim: int) -> None:
        super().__init__()
        self.cfg = cfg

        self.rnn = nn.GRU(
            input_dim, cfg.hidden_size,
            num_layers=cfg.num_rnn_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_rnn_layers > 1 else 0.0,
        )

        if cfg.movement_mode == "discrete":
            self.movement_head = nn.Linear(cfg.hidden_size, 4)
        else:
            self.movement_mean = nn.Linear(cfg.hidden_size, 2)
            self.movement_log_std = nn.Parameter(torch.full((2,), cfg.init_log_std))
            if cfg.freeze_log_std:
                self.movement_log_std.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple:
        """x: (B, T, input_dim), h: (num_layers, B, hidden) or None.

        Returns (move_dist, h_next).
        """
        features, h_next = self.rnn(x, h)
        if self.cfg.movement_mode == "discrete":
            logits = self.movement_head(features)
            move_dist = Categorical(logits=logits)
        else:
            mean = self.movement_mean(features)
            std = self.movement_log_std.exp().expand_as(mean)
            move_dist = Normal(mean, std)
        return move_dist, h_next

    @torch.no_grad()
    def act(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> dict:
        """Single-step action selection. x: (B, 1, input_dim).

        Returns dict with move_action ((B,) int or (B, 2) float),
        move_log_prob ((B,)), and h_next.
        """
        move_dist, h_next = self.forward(x, h)
        if deterministic:
            if self.cfg.movement_mode == "discrete":
                move_action = move_dist.probs.argmax(-1)
            else:
                move_action = move_dist.mean
        else:
            move_action = move_dist.sample()

        move_log_prob = move_dist.log_prob(move_action)
        if self.cfg.movement_mode == "continuous":
            move_log_prob = move_log_prob.sum(-1)

        return {
            "move_action": move_action.squeeze(1),
            "move_log_prob": move_log_prob.squeeze(1),
            "h_next": h_next,
        }
