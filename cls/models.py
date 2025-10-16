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
        num_model_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_model_layers = num_model_layers

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_model_layers,
            batch_first=True,
            dropout=dropout if num_model_layers > 1 else 0.0,
        )

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, h_next = self.rnn(x, h)
        return out, h_next


class MLP(nn.Module):
    """Stateless MLP backbone. Returns features per timestep and no hidden state.

    Applies an MLP to each time step independently.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_model_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_model_layers = num_model_layers

        layers: list[nn.Module] = []
        in_dim = input_size
        for i in range(max(1, num_model_layers)):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        # x: (B, T, F)
        B, T, F = x.shape
        x_flat = x.reshape(B * T, F)
        y = self.net(x_flat)
        y = y.reshape(B, T, self.hidden_size)
        return y, None


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
        num_model_layers: int = 1,
        num_actions: int = 4,
        dropout: float = 0.0,
        model_class: str = "GRU",
        encoder_dim: Optional[int] = None,
        num_encoder_layers: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_model_layers = num_model_layers
        self.num_actions = num_actions

        # Encoder head (optional multi-layer with nonlinearity on hidden layers)
        model_input_size = encoder_dim if (encoder_dim is not None) else input_size
        if num_encoder_layers <= 0:
            self.encoder = nn.Identity()
        else:
            layers: list[nn.Module] = []
            in_dim = input_size
            out_dim = model_input_size
            for layer_idx in range(num_encoder_layers):
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                in_dim = out_dim
            self.encoder = nn.Sequential(*layers)

        if isinstance(model_class, str):
            if model_class.upper() == "GRU":
                model_class = GRU
            elif model_class.upper() == "MLP":
                model_class = MLP
            else:
                raise ValueError("model_class must be 'GRU' or 'MLP'")

        # Backbone model (feature extractor)
        self.model = model_class(
            input_size=model_input_size,
            hidden_size=hidden_size,
            num_model_layers=num_model_layers,
            dropout=dropout,
        )

        # Whether backbone is recurrent (expects/returns hidden state)
        self.is_recurrent = isinstance(self.model, GRU)

        # Policy and value heads
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        if self.is_recurrent:
            features, h_next = self.model(enc, h)
        else:
            features, h_next = self.model(enc, None)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values, h_next

    @torch.no_grad()
    def act(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, h_next = self.forward(x, h)
        action_idx = int(torch.argmax(logits[0, 0]).item())
        return action_idx, logits, values, h_next


