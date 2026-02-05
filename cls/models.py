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

import matplotlib.pyplot as plt

class GRU(nn.Module):
    """GRU backbone that outputs hidden features only (no policy head)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_model_layers: int = 1,
        dropout: float = 0.0,
        **kwargs,  # Ignore extra args for API compatibility
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
        **kwargs,  # Ignore extra args for API compatibility
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


class GridCNNBackbone(nn.Module):
    """CNN backbone for g_hot grid representations.
    
    Reshapes flattened g_hot into 2D channels and applies convolutions.
    Returns features per timestep (like MLP), no hidden state.
    
    Args:
        input_size: Total input dimension (used to detect concatenated mode)
        hidden_size: Output feature dimension
        num_model_layers: Number of conv layers (default 3)
        dropout: Dropout rate
        lambdas: List of module periods (e.g., [11, 12])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_model_layers: int = 3,
        dropout: float = 0.0,
        lambdas: Optional[list[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if lambdas is None:
            lambdas = [11, 12]  # Default
        
        self.lambdas = lambdas
        self.hidden_size = hidden_size
        self.num_modules = len(lambdas)
        self.max_lambda = max(lambdas)
        
        # Size of a single g_hot
        self.single_g_size = sum(l * l for l in lambdas)
        
        # Detect if input is concatenated (start + next) or single
        self.concatenated = (input_size == 2 * self.single_g_size)
        self.num_channels = 2 * self.num_modules if self.concatenated else self.num_modules
        
        # Module ranges for reshaping
        self.module_ranges = []
        start = 0
        for l in lambdas:
            end = start + l * l
            self.module_ranges.append((start, end, l))
            start = end
        
        # Build conv layers
        conv_layers = []
        in_ch = self.num_channels
        for i in range(max(1, num_model_layers)):
            conv_layers.append(nn.Conv2d(in_ch, hidden_size, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            in_ch = hidden_size
        self.convs = nn.Sequential(*conv_layers)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

    def _reshape_single_g(self, g: torch.Tensor, channel_offset: int, out: torch.Tensor) -> None:
        """Reshape a single g_hot into channels of the output tensor."""
        batch_size = g.shape[0]
        for m_idx, (start, end, l) in enumerate(self.module_ranges):
            module_flat = g[:, start:end]
            module_2d = module_flat.view(batch_size, l, l)
            out[:, channel_offset + m_idx, :l, :l] = module_2d

    def reshape_to_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flattened g_hot to (batch, num_channels, max_λ, max_λ)."""
        batch_size = x.shape[0]
        out = torch.zeros(batch_size, self.num_channels, self.max_lambda, self.max_lambda,
                         device=x.device, dtype=x.dtype)
        
        if self.concatenated:
            start_g = x[:, :self.single_g_size]
            next_g = x[:, self.single_g_size:]
            self._reshape_single_g(start_g, 0, out)
            self._reshape_single_g(next_g, self.num_modules, out)
        else:
            self._reshape_single_g(x, 0, out)
        
        return out

    _saved_heatmap = False  # Class-level flag to save only once

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        # x: (B, T, F)
        B, T, F = x.shape
        x_flat = x.reshape(B * T, F)
        
        # Reshape to 2D and apply convs
        x_2d = self.reshape_to_2d(x_flat)
        
        # Save heatmap once
        if not GridCNNBackbone._saved_heatmap:
            GridCNNBackbone._saved_heatmap = True
            sample = x_2d[0].detach().cpu().numpy()  # (C, H, W)
            nc = sample.shape[0]
            fig, axes = plt.subplots(1, nc, figsize=(3 * nc, 3))
            if nc == 1:
                axes = [axes]
            for i, ax in enumerate(axes):
                ax.imshow(sample[i], cmap='viridis')
                ax.set_title(f'Ch {i}')
                ax.axis('off')
            plt.tight_layout()
            plt.savefig('cnn_input_heatmap.png', dpi=100)
            plt.close()
            print(f"Saved CNN input heatmap to cnn_input_heatmap.png ({nc} channels)")
        
        features = self.convs(x_2d)
        features = self.pool(features)
        features = features.view(B * T, self.hidden_size)
        features = self.dropout(features)
        
        # Reshape back to (B, T, hidden_size)
        features = features.reshape(B, T, self.hidden_size)
        return features, None


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
        lambdas: Optional[list[int]] = None,  # For GridCNN backbone
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_model_layers = num_model_layers
        self.num_actions = num_actions

        # Resolve model_class string to class
        use_cnn = False
        if isinstance(model_class, str):
            if model_class.upper() == "GRU":
                model_class = GRU
            elif model_class.upper() == "MLP":
                model_class = MLP
            elif model_class.upper() == "CNN" or model_class.upper() == "GRIDCNN":
                model_class = GridCNNBackbone
                use_cnn = True
            else:
                raise ValueError("model_class must be 'GRU', 'MLP', or 'CNN'")

        # CNN needs raw spatial structure - disable encoder
        if use_cnn and num_encoder_layers > 0:
            print(f"Warning: CNN backbone requires raw g_hot, ignoring num_encoder_layers={num_encoder_layers}")
            num_encoder_layers = 0

        # Encoder head (optional multi-layer with nonlinearity on hidden layers)
        if num_encoder_layers <= 0:
            model_input_size = input_size
            self.encoder = nn.Identity()
        else:
            model_input_size = encoder_dim
            layers: list[nn.Module] = []
            in_dim = input_size
            out_dim = model_input_size
            for layer_idx in range(num_encoder_layers):
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                in_dim = out_dim
            self.encoder = nn.Sequential(*layers)

        # Backbone model (feature extractor)
        self.model = model_class(
            input_size=model_input_size,
            hidden_size=hidden_size,
            num_model_layers=num_model_layers,
            dropout=dropout,
            lambdas=lambdas,
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


