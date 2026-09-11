"""PairRegressor: the memoryless goal-conditioned model (plan §3.1).

Experiment A asks whether a network with no memory can emit the direction to
a goal from an encoding of where it is and an encoding of where it is going.
That is a regression, not a policy: there is nothing to sample, nothing to
carry between steps, and no variance to fit. So this is `FeedForwardCore`
plus one linear layer, and none of `RNNAgent`'s apparatus -- no `Normal`
head, no `log_std`, no `act()`, no hidden state.

Keeping the distribution head out is deliberate, not a shortcut. The §5.2
in-context measurement was inverted once by evaluating a Gaussian's *mean*
where the fitted mean had collapsed toward zero; a model with no Gaussian
cannot have that defect. The continuous output here is the direction itself,
normalised, so that the loss (`1 - cos`) and the metric (angular error) are
the same function of the same vector.

The interface the static evaluator reads is two methods:

    predict_direction(x) -> (B, 2) unit vectors        continuous
    predict_logits(x)    -> (B, 4) cardinal logits     discrete

`RNNAgentAsPairModel` in `evaluation/goal_pairs.py` gives an `RNNAgent` the
same two methods, which is how Experiment B's weights get scored on the same
table (readout 1) without the evaluator knowing which model it holds.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recurrent import FeedForwardCore

DIRECTION_EPS = 1e-6


def normalize_direction(v: torch.Tensor, eps: float = DIRECTION_EPS) -> torch.Tensor:
    """Unit vectors along the last axis; the zero vector maps to zero."""
    return v / (v.norm(dim=-1, keepdim=True) + eps)


class PairRegressor(nn.Module):
    """`FeedForwardCore` + one linear output. `movement_mode` picks the head."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        movement_mode: str,
        nonlinearity: str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if movement_mode not in ("discrete", "continuous"):
            raise ValueError(f"movement_mode must be discrete|continuous, got {movement_mode!r}")
        self.movement_mode = movement_mode
        self.input_dim = int(input_dim)
        self.core = FeedForwardCore(input_dim, hidden_size, num_layers=num_layers,
                                    nonlinearity=nonlinearity, dropout=dropout)
        self.head = nn.Linear(hidden_size, 4 if movement_mode == "discrete" else 2)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) -> (B, H). The core wants (B, T, D); T = 1 here."""
        if x.dim() != 2:
            raise ValueError(f"PairRegressor takes (B, D) inputs, got {tuple(x.shape)}")
        out, _ = self.core(x.unsqueeze(1))
        return out[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw head output: (B, 4) logits or (B, 2) un-normalised direction."""
        return self.head(self.features(x))

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.movement_mode != "discrete":
            raise RuntimeError("predict_logits on a continuous PairRegressor")
        return self.forward(x)

    def predict_direction(self, x: torch.Tensor) -> torch.Tensor:
        if self.movement_mode != "continuous":
            raise RuntimeError("predict_direction on a discrete PairRegressor")
        return normalize_direction(self.forward(x))

    # ------------------------------------------------------------------
    # Loss (plan §3.3)
    # ------------------------------------------------------------------

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Discrete: CE against a distribution over the optimal set.
        Continuous: 1 - cos(direction, target unit vector).

        `target` is (B, 4) non-negative weights summing to 1 per row for
        discrete (uniform over the optimal actions), or (B, 2) unit vectors
        for continuous. Both are what `pair_targets` returns.
        """
        out = self.forward(x)
        if self.movement_mode == "discrete":
            logp = F.log_softmax(out, dim=-1)
            return -(target * logp).sum(dim=-1).mean()
        d = normalize_direction(out)
        return (1.0 - (d * target).sum(dim=-1)).mean()
