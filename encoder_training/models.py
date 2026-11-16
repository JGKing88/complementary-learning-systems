"""Grid cell encoder architectures: MLP and CNN.

Both produce L2-normalized embeddings on the unit sphere.
Input: flattened grid one-hot vectors (B, sum(l^2 for l in lambdas)).
Output: (B, out_dim), unit-norm.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EncoderModelConfig


class GridEncoder(nn.Module):
    """MLP encoder for flattened grid one-hot codes."""

    def __init__(self, lambdas: list[int], hidden_dim: int = 512,
                 num_hidden_layers: int = 2, out_dim: int = 128,
                 nonlinearity: str = "gelu", output_nonlinearity: str = "tanh",
                 gain: float = 1.0, in_dim: int | None = None,
                 sparse_active: float = 0.1):
        super().__init__()
        if in_dim is None:
            in_dim = sum(l * l for l in lambdas)
        # relu_sparse / relu_rank outputs, the forward pass of branch worktree-relu-sparse-encoder
        # (encoder_training/models.py there), copied so the probe can load those encoders.
        from statistics import NormalDist
        nd = NormalDist()
        self.sparse_active = float(sparse_active)
        self.sparse_thresh = nd.inv_cdf(1.0 - self.sparse_active)
        prof = [max(nd.inv_cdf(1.0 - (k + 0.5) / out_dim) - self.sparse_thresh, 0.0) for k in range(out_dim)]
        prof = torch.tensor(prof, dtype=torch.float32)
        self.register_buffer("rank_profile", prof / prof.norm(), persistent=False)
        self.homeo_eta = 0.01
        if "homeo" in output_nonlinearity:     # only here, so older checkpoints load
            self.register_buffer("homeo_bias", torch.zeros(out_dim))
        else:
            self.homeo_bias = None
        self.signed_thresh = nd.inv_cdf(1.0 - self.sparse_active / 2)
        sprof = [max(nd.inv_cdf(1.0 - (k + 0.5) / (2 * out_dim)) - self.signed_thresh, 0.0) for k in range(out_dim)]
        sprof = torch.tensor(sprof, dtype=torch.float32)
        self.register_buffer("signed_profile", sprof / sprof.norm(), persistent=False)
        self.out_dim = out_dim
        self.output_nonlinearity = output_nonlinearity
        self.gain = gain

        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(
            nonlinearity.lower(), nn.GELU)

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), act()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, gain: float | None = None) -> torch.Tensor:
        squeeze_time = x.dim() == 3
        if squeeze_time:
            B, T, D = x.shape
            x = x.reshape(B * T, D)

        z = self.net(x)
        g = gain if gain is not None else self.gain
        if self.output_nonlinearity == "tanh":
            z = torch.tanh(g * z)
        elif self.output_nonlinearity == "sigmoid":
            z = torch.sigmoid(g * z)
        elif self.output_nonlinearity == "signed_rank_homeo":
            z = (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            sgn, mag = torch.sign(z), z.abs() - self.homeo_bias
            soft = F.normalize(sgn * F.relu(mag - self.signed_thresh), p=2, dim=-1)
            order = mag.argsort(dim=-1, descending=True)
            hard = torch.zeros_like(z).scatter_(
                -1, order, self.signed_profile.to(z.dtype).expand_as(z).contiguous()) * sgn
            if self.training:
                with torch.no_grad():
                    self.homeo_bias += self.homeo_eta * ((hard != 0).float().mean(0) - self.sparse_active)
            z = soft + (hard - soft).detach()
        elif self.output_nonlinearity in ("relu_sparse", "relu_rank", "relu_rank_homeo"):
            z = (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            if self.output_nonlinearity == "relu_rank_homeo":
                z = z - self.homeo_bias
            soft = F.normalize(F.relu(z - self.sparse_thresh), p=2, dim=-1)
            if self.output_nonlinearity == "relu_sparse":
                z = soft
            else:
                order = z.argsort(dim=-1, descending=True)
                hard = torch.zeros_like(z).scatter_(
                    -1, order, self.rank_profile.to(z.dtype).expand_as(z).contiguous())
                if self.output_nonlinearity == "relu_rank_homeo" and self.training:
                    with torch.no_grad():
                        self.homeo_bias += self.homeo_eta * ((hard > 0).float().mean(0) - self.sparse_active)
                z = soft + (hard - soft).detach()
        z = F.normalize(z, p=2, dim=-1)

        if squeeze_time:
            z = z.reshape(B, T, -1)
        return z


class GridEncoderCNN(nn.Module):
    """CNN encoder that reshapes flat grid codes into 2D per-module channels."""

    def __init__(self, lambdas: list[int], hidden_channels: int = 128,
                 num_conv_layers: int = 3, kernel_size: int = 3,
                 num_hidden_layers: int = 1, hidden_dim: int = 128,
                 out_dim: int = 128, nonlinearity: str = "gelu",
                 output_nonlinearity: str = "tanh", gain: float = 1.0):
        super().__init__()
        self.lambdas = lambdas
        self.out_dim = out_dim
        self.output_nonlinearity = output_nonlinearity
        self.gain = gain
        self.num_modules = len(lambdas)
        self.max_lambda = max(lambdas)

        # Precompute module slicing ranges
        self.module_ranges: list[tuple[int, int, int]] = []
        start = 0
        for l in lambdas:
            end = start + l * l
            self.module_ranges.append((start, end, l))
            start = end

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(
            nonlinearity.lower(), nn.GELU)

        # Conv stack
        conv_layers: list[nn.Module] = []
        in_ch = self.num_modules
        for _ in range(num_conv_layers):
            conv_layers += [nn.Conv2d(in_ch, hidden_channels,
                                      kernel_size=kernel_size,
                                      padding=kernel_size // 2),
                            act_fn()]
            in_ch = hidden_channels
        self.convs = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # MLP head
        mlp: list[nn.Module] = []
        if num_hidden_layers == 0:
            mlp.append(nn.Linear(hidden_channels * 16, out_dim))
        else:
            mlp += [nn.Linear(hidden_channels * 16, hidden_dim), act_fn()]
            for _ in range(num_hidden_layers - 1):
                mlp += [nn.Linear(hidden_dim, hidden_dim), act_fn()]
            mlp.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*mlp)

    def reshape_to_2d(self, x: torch.Tensor) -> torch.Tensor:
        """(B, sum(l^2)) -> (B, num_modules, max_l, max_l), zero-padded."""
        B = x.shape[0]
        out = x.new_zeros(B, self.num_modules, self.max_lambda, self.max_lambda)
        for m_idx, (start, end, l) in enumerate(self.module_ranges):
            pad = self.max_lambda - l
            top = pad // 2
            left = pad // 2
            out[:, m_idx, top:top + l, left:left + l] = x[:, start:end].view(B, l, l)
        return out

    def forward(self, x: torch.Tensor, gain: Optional[float] = None) -> torch.Tensor:
        squeeze_time = x.dim() == 3
        if squeeze_time:
            B, T, D = x.shape
            x = x.reshape(B * T, D)

        features = self.pool(self.convs(self.reshape_to_2d(x))).flatten(1)
        z = self.mlp(features)

        g = gain if gain is not None else self.gain
        if self.output_nonlinearity == "tanh":
            z = torch.tanh(g * z)
        elif self.output_nonlinearity == "sigmoid":
            z = torch.sigmoid(g * z)
        z = F.normalize(z, p=2, dim=-1)

        if squeeze_time:
            z = z.reshape(B, T, -1)
        return z


def create_encoder(cfg: EncoderModelConfig, device: str | None = None) -> GridEncoder | GridEncoderCNN:
    """Factory: build encoder from config."""
    if cfg.encoder_type == "mlp":
        enc = GridEncoder(
            lambdas=cfg.lambdas, hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers, out_dim=cfg.out_dim,
            nonlinearity=cfg.nonlinearity, output_nonlinearity=cfg.output_nonlinearity,
            gain=cfg.gain, sparse_active=getattr(cfg, "sparse_active", 0.1),
        )
    elif cfg.encoder_type == "cnn":
        enc = GridEncoderCNN(
            lambdas=cfg.lambdas, hidden_channels=cfg.hidden_channels,
            num_conv_layers=cfg.num_conv_layers, kernel_size=cfg.kernel_size,
            num_hidden_layers=cfg.num_hidden_layers, hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_dim, nonlinearity=cfg.nonlinearity,
            output_nonlinearity=cfg.output_nonlinearity, gain=cfg.gain,
        )
    elif cfg.encoder_type == "equivariant":
        # §8.1. Exactly translation-equivariant by construction, so
        # cos(z(x), z(y)) depends only on the offset and r_min == r_median.
        # out_dim/hidden_dim/gain are ignored: the character table determines
        # the output size, and any pointwise nonlinearity would break the
        # equivariance (tanh does not commute with a rotation of the
        # (Re, Im) pair a character contributes).
        from encoder_training.equivariant import EquivariantCharacterEncoder
        enc = EquivariantCharacterEncoder(
            lambdas=cfg.lambdas, p_max=cfg.char_p_max, m_max=cfg.char_m_max,
        )
    elif cfg.encoder_type == "ideal_net":
        # The ideal encoder as a network with a trainable integer layer
        # (analysis/hopfield_probe/ideal_net.py). Layers 1-2 are fixed.
        from analysis.hopfield_probe.ideal_net import IdealNet

        class _IdealNetEncoder(IdealNet):
            def forward(self, x, gain=None):          # gain does not apply
                return super().forward(x).float()

        enc = _IdealNetEncoder(r=cfg.ideal_r, n_freq=cfg.ideal_n_freq,
                               seed=cfg.ideal_seed, lambdas=cfg.lambdas,
                               trainable_harmonics=True, init=cfg.ideal_init)
    else:
        raise ValueError(f"Unknown encoder_type: {cfg.encoder_type}")
    if device is not None:
        enc = enc.to(device)
    return enc
