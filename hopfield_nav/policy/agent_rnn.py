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
    if getattr(cfg, "goal_channel", "none") != "none":
        dim += 2
    return dim


class RNNAgent(nn.Module):
    """GRU + single linear move head. forward(x, h) -> (move_dist, h_next)."""

    def __init__(self, cfg: RNNAgentConfig, input_dim: int) -> None:
        super().__init__()
        self.cfg = cfg

        self.rnn = build_recurrent_core(cfg, input_dim)

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
        return act_from_forward(self, x, h, deterministic)


def set_agent_task(agent, task: int) -> bool:
    """Tell a task-conditioned policy which task it is in. Returns whether it is.

    The hypernetwork and the isolation policies select parameters by task id and
    have no sensible default; every other policy in the stack has one set of
    weights and no opinion. Rather than teach the driver and the evaluator which
    is which, they call this on every agent and act on what comes back -- so
    adding a task-conditioned policy needs no change to either, and a policy
    that is *not* task-conditioned cannot accidentally be treated as one.

    The return value is load-bearing, not informational: `run_sequential_blocks`
    uses it to refuse combinations that would be silently wrong, such as
    replaying one task's trajectories through another task's head.
    """
    fn = getattr(agent, "set_task", None)
    if fn is None:
        return False
    fn(task)
    return True


@torch.no_grad()
def act_from_forward(
    agent,
    x: torch.Tensor,
    h: torch.Tensor | None = None,
    deterministic: bool = False,
) -> dict:
    """One step of action selection, given anything with `forward` and `cfg`.

    Free function rather than a method because `RNNAgent` is no longer the only
    policy with this head. The hypernetwork agent and the multi-head agent
    generate or select their weights differently, but the step *after* the
    distribution exists -- sample or take the mode, sum the log-prob over a
    continuous action's two dimensions, drop the length-1 time axis -- is
    identical for all three, and duplicating it three times is how a
    `deterministic` flag comes to mean the mode in one agent and the mean in
    another.

    Duck-typed on `agent.forward(x, h) -> (dist, h_next)` and
    `agent.cfg.movement_mode`, so nothing here needs to import the agents.
    """
    move_dist, h_next = agent.forward(x, h)
    if deterministic:
        if agent.cfg.movement_mode == "discrete":
            move_action = move_dist.probs.argmax(-1)
        else:
            move_action = move_dist.mean
    else:
        move_action = move_dist.sample()

    move_log_prob = move_dist.log_prob(move_action)
    if agent.cfg.movement_mode == "continuous":
        move_log_prob = move_log_prob.sum(-1)

    return {
        "move_action": move_action.squeeze(1),
        "move_log_prob": move_log_prob.squeeze(1),
        "h_next": h_next,
    }
