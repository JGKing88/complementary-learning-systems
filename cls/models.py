"""Neural network models for policies.

This module currently exposes a simple GRU-based policy head that produces
logits over 4 cardinal actions on a square grid. The actions are ordered as
[N, E, S, W] and are expected to be mapped upstream to environment vectors:
(0, 1), (1, 0), (0, -1), (-1, 0).

Inputs are flexible; the trainer constructs feature vectors that include
agent heading, positions, and coarse sensory statistics. The network treats
these as generic features and does not assume a particular semantics beyond
the input dimensionality.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

class GRU(nn.Module):
    """GRU backbone that outputs hidden features only (no policy head)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_rnn_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0.0,
        )

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, h_next = self.rnn(x, h)
        return out, h_next


class Agent(nn.Module):
    """Agent with encoder head, backbone model, and policy head.

    The agent applies an optional encoder projection to inputs, feeds them
    through the provided backbone model class (default `GRU`), and maps the
    resulting hidden features to action logits via a policy head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_rnn_layers: int = 1,
        num_actions: int = 4,
        dropout: float = 0.0,
        model_class: str = "GRU",
        encoder_dim: Optional[int] = None,
        num_encoder_layers: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.num_actions = num_actions

        # Encoder head (optional multi-layer with nonlinearity on hidden layers)
        model_input_size = encoder_dim if (encoder_dim is not None) else input_size
        if num_encoder_layers <= 0:
            if encoder_dim is not None and encoder_dim != input_size:
                self.encoder = nn.Linear(input_size, encoder_dim)
            else:
                self.encoder = nn.Identity()
        else:
            layers: list[nn.Module] = []
            in_dim = input_size
            out_dim = model_input_size
            for layer_idx in range(num_encoder_layers):
                layers.append(nn.Linear(in_dim, out_dim))
                if layer_idx < num_encoder_layers - 1:
                    layers.append(nn.ReLU())
                in_dim = out_dim
            self.encoder = nn.Sequential(*layers)

        if model_class == "GRU":
            model_class = GRU

        # Backbone model (feature extractor)
        self.model = model_class(
            input_size=model_input_size,
            hidden_size=hidden_size,
            num_rnn_layers=num_rnn_layers,
            dropout=dropout,
        )

        # Policy and value heads
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        features, h_next = self.model(enc, h)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values, h_next

    @torch.no_grad()
    def act(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, h_next = self.forward(x, h)
        action_idx = int(torch.argmax(logits[0, 0]).item())
        return action_idx, logits, values, h_next


