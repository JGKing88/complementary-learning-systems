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


def rnn_input_layout(
    cfg: RNNAgentConfig, sensory_dim: int, gbook_dim: int = 0,
) -> list[tuple[str, int]]:
    """The RNN stack's input channels, in order, as ``(name, width)``.

    This is the one place the layout is written down; ``compute_rnn_input_dim``
    sums it and ``rollout.rnn.build_rnn_input`` assembles it. Before this
    function the two agreed by convention, in two files, which is how a
    channel can move while the tensor keeps its shape.

    **Order is a compatibility surface.** Every saved checkpoint's first layer
    was trained against ``sensory, prev_action, prev_reward, grid_state,
    goal_vec``. New channels append after those, in the order below;
    reordering silently invalidates every checkpoint. A channel that is off
    contributes nothing, so a checkpoint trained with the defaults sees
    exactly the layout it was trained with.

    ``sensory_dim`` is the width of ONE view (``observation_size``); the
    channel is that or four times it under ``sensory_mode="omni"``.
    ``gbook_dim`` is the smoothed-gbook width (``vectorhash.Ng``), used by both
    grid-state channels; the caller computes it from the VectorHash it built.
    """
    specs: list[tuple[str, int]] = []
    if getattr(cfg, "input_sensory", True):
        mode = getattr(cfg, "sensory_mode", "ego")
        specs.append(("sensory", 4 * sensory_dim if mode == "omni" else sensory_dim))
    if cfg.input_prev_action:
        specs.append(("prev_action", 4 if cfg.movement_mode == "discrete" else 2))
    if cfg.input_prev_reward:
        specs.append(("prev_reward", 1))
    if cfg.input_grid_state:
        specs.append(("grid_state", gbook_dim))
    if getattr(cfg, "goal_channel", "none") != "none":
        specs.append(("goal_vec", 2))
    # --- appended 2026-09-10 for the goal-conditioned control -------------
    if getattr(cfg, "input_xy_state", False):
        specs.append(("xy_state", 2))
    if getattr(cfg, "input_goal_grid_state", False):
        specs.append(("goal_grid_state", gbook_dim))
    gs = getattr(cfg, "goal_sensory", "none")
    if gs != "none":
        specs.append(("goal_sensory", 4 * sensory_dim if gs == "omni" else sensory_dim))
    return specs


def compute_rnn_input_dim(
    cfg: RNNAgentConfig, sensory_dim: int, gbook_dim: int = 0,
) -> int:
    """Total RNN input width: the sum over ``rnn_input_layout``."""
    return sum(w for _, w in rnn_input_layout(cfg, sensory_dim, gbook_dim))


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
        "move_mean": move_dist.mean.detach() if hasattr(move_dist, "mean") else None,
        "move_std": move_dist.stddev.detach() if hasattr(move_dist, "stddev") else None,
        "move_action": move_action.squeeze(1),
        "move_log_prob": move_log_prob.squeeze(1),
        "h_next": h_next,
    }
