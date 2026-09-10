"""RNNAgent: vanilla GRU policy with a single movement head.

Control baseline for hopfield_nav: no Hopfield, no store head, no value head.
Trained by behavior cloning against a shortest-path oracle (see oracle_bfs.py
and bc_rnn.py). Uses raw sensory observations only — no encoder, no VectorHash.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .recurrent import build_recurrent_core
from .action_head import build_log_std, movement_std, squash_mean
from .polar_head import PolarHead
from ..config import RNNAgentConfig


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

    def __init__(self, cfg: RNNAgentConfig, input_dim: int,
                 action_bounds: tuple[float, float] | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        # (min, max) action norm from the ENV config, passed in rather than
        # mirrored onto the agent config so the two cannot drift apart. Only
        # needed when cfg.action_squash is on.
        self.action_bounds = action_bounds

        self.rnn = build_recurrent_core(cfg, input_dim)

        if cfg.movement_mode == "discrete":
            self.movement_head = nn.Linear(cfg.hidden_size, 4)
        else:
            # Under polar this is the DIRECTION head; see agent.py.
            self.movement_mean = nn.Linear(cfg.hidden_size, 2)
            if getattr(cfg, "action_polar", False):
                if action_bounds is None:
                    raise ValueError(
                        "action_polar needs the env's min/max_action_norm "
                        "passed as action_bounds; the speed Beta is defined "
                        "on that interval")
                self.polar_head = PolarHead(cfg, cfg.hidden_size, *action_bounds)
            else:
                self.polar_head = None
                log_std, log_std_head = build_log_std(cfg, cfg.hidden_size)
                if log_std is not None:
                    self.movement_log_std = nn.Parameter(log_std)
                    self.movement_log_std.requires_grad = log_std.requires_grad
                    self.movement_log_std_head = None
                else:
                    self.movement_log_std = None
                    self.movement_log_std_head = log_std_head
                if getattr(cfg, "action_squash", False) and action_bounds is None:
                    raise ValueError(
                        "action_squash needs the env's min/max_action_norm passed "
                        "as action_bounds; without them there is no range")

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
        elif self.polar_head is not None:
            move_dist = self.polar_head(features, self.movement_mean(features))
        else:
            mean = self.movement_mean(features)
            if getattr(self.cfg, "action_squash", False):
                mean = squash_mean(mean, *self.action_bounds)
            std = movement_std(self.cfg, features, mean,
                               self.movement_log_std, self.movement_log_std_head)
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
            "move_mean": move_dist.mean.detach() if hasattr(move_dist, "mean") else None,
            "move_std": move_dist.stddev.detach() if hasattr(move_dist, "stddev") else None,
            "move_action": move_action.squeeze(1),
            "move_log_prob": move_log_prob.squeeze(1),
            "h_next": h_next,
        }
