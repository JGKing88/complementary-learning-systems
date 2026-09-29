"""The frozen grid MLP: ``(g_now, g_goal) -> unit direction``.

Trained on branch ``worktree-nn-generalization-control`` as a ``PairRegressor``
(``FeedForwardCore`` + linear head) by ``train_decode_walk.py`` -- the Phase-1
decode, random walks + odometry with displacement-balanced pairs. Its input is
``[smoothed gbook(p), smoothed gbook(g)]`` on the nav scaffold (``lambdas 11 12
13``, ``fwhm 0.25``). Rebuilt here from the state dict as the plain Linear/act
stack it is, rather than porting that branch's trunk module: the checkpoint's
``core.net.{0,2,..}`` / ``head`` keys fix the architecture.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn

DIRECTION_EPS = 1e-6   # PairRegressor.normalize_direction


class GridDirectionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int,
                 nonlinearity: str = "relu") -> None:
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[nonlinearity]
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_size), act()]
            d = hidden_size
        self.core = nn.Module()
        self.core.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, 2)
        self.input_dim = int(input_dim)

    @torch.no_grad()
    def direction(self, g_now: torch.Tensor, g_goal: torch.Tensor) -> torch.Tensor:
        """(B, Ng), (B, Ng) -> (B, 2) unit vectors (x, y) = (East, North)."""
        out = self.head(self.core.net(torch.cat([g_now, g_goal], dim=-1)))
        return out / (out.norm(dim=-1, keepdim=True) + DIRECTION_EPS)


def load_grid_mlp(path: str, device: str = "cpu") -> GridDirectionMLP:
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    argv = ck.get("argv", {})
    if argv.get("movement_mode", "continuous") != "continuous":
        raise ValueError(f"{path}: need a continuous (direction) grid MLP")
    if argv.get("dropout", 0.0) or argv.get("norm", False):
        raise ValueError(f"{path}: dropout/norm layers are not supported here")
    idx = sorted(int(m.group(1)) for k in sd
                 if (m := re.fullmatch(r"core\.net\.(\d+)\.weight", k)))
    if idx != list(range(0, 2 * len(idx), 2)):
        raise ValueError(f"{path}: unexpected layer layout {idx}")
    hidden, input_dim = sd["core.net.0.weight"].shape
    model = GridDirectionMLP(int(input_dim), int(hidden), len(idx),
                             argv.get("nonlinearity", "relu"))
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)
