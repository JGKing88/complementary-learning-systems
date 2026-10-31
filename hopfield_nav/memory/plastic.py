"""Idea 2's memory: a small network whose weights hold the goals (GRID_MLP_NAV_PLAN §6).

``M`` reads a direction-aligned view and outputs the goal's grid phases and a
confidence ``c``. Writing a goal is ``k`` gradient steps on ``M``'s weights.
Everything here is functional (weights passed explicitly) so that a write is
differentiable and the meta-learner can backpropagate through every write in a
lifetime.

Input ("canvas"). A live view of ``n`` rays at heading ψ is placed into a
``PANORAMA_BINS``-slot vector by each ray's absolute direction, with a mask of
which slots are filled. This is the alignment idea 1's key uses, so the
panorama, and with it the env's identity, sits at the same slots whatever the
heading.

Output. For each grid module (period λ) and each axis, the phase angle
2π·(position mod λ)/λ as a (cos, sin) pair: 3 modules x 2 axes x 2 = 12
numbers, plus one logit for ``c``. ``decode_cells`` turns predicted phases into
the best-matching cell of an arena, so a prediction is always a real grid
state.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..world.env import PANORAMA_BINS, cone_offsets, panorama_bins

N_PHASE = 12          # 3 modules x 2 axes x (cos, sin)


def canvas(views: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """(B, n) views cast at headings (B,) -> (B, 2 * PANORAMA_BINS) values + mask."""
    views = np.asarray(views, dtype=np.float32)
    B, n = views.shape
    bins = panorama_bins(np.asarray(psi, np.float64)[:, None] + cone_offsets(n)[None, :])
    out = np.zeros((B, 2 * PANORAMA_BINS), dtype=np.float32)
    rows = np.repeat(np.arange(B), n)
    out[rows, bins.ravel()] = views.ravel()
    out[rows, PANORAMA_BINS + bins.ravel()] = 1.0
    return out


def goal_phases(gx, gy, lambdas) -> np.ndarray:
    """Global cells -> (B, 12) unit (cos, sin) phase pairs, module-major."""
    gx = np.asarray(gx, np.float64).reshape(-1)
    gy = np.asarray(gy, np.float64).reshape(-1)
    cols = []
    for lam in lambdas:
        for coord in (gx, gy):
            a = 2 * np.pi * np.mod(coord, lam) / lam
            cols += [np.cos(a), np.sin(a)]
    return np.stack(cols, axis=1).astype(np.float32)


def decode_cells(pred: np.ndarray, cells_global: np.ndarray, lambdas) -> np.ndarray:
    """(B, 12) predicted phases -> index into ``cells_global`` (C, 2) of the
    cell whose phases match best (sum of cosines over the 6 angles)."""
    ref = goal_phases(cells_global[:, 0], cells_global[:, 1], lambdas)   # (C, 12)
    p = pred.reshape(len(pred), 6, 2)
    p = p / np.maximum(np.linalg.norm(p, axis=2, keepdims=True), 1e-8)
    return (p.reshape(len(pred), 12) @ ref.T).argmax(axis=1)


# ---------------------------------------------------------------------------
# The network, functionally
# ---------------------------------------------------------------------------

def init_params(in_dim: int, hidden: int, layers: int, gen: torch.Generator | None = None
                ) -> list[torch.Tensor]:
    """[W1, b1, ..., Wout, bout] with PyTorch's default Linear init."""
    dims = [in_dim] + [hidden] * layers + [N_PHASE + 1]
    params = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        bound = 1.0 / math.sqrt(d_in)
        params.append((torch.rand(d_in, d_out, generator=gen) * 2 - 1) * bound)
        params.append((torch.rand(d_out, generator=gen) * 2 - 1) * bound)
    return params


def forward(params: list[torch.Tensor], x: torch.Tensor):
    """x (B, in) -> (phases (B, 12) unit pairs, c logit (B,))."""
    h = x
    for i in range(0, len(params) - 2, 2):
        h = F.relu(h @ params[i] + params[i + 1])
    out = h @ params[-2] + params[-1]
    ph = out[:, :N_PHASE].reshape(-1, 6, 2)
    ph = ph / (ph.norm(dim=2, keepdim=True) + 1e-6)
    return ph.reshape(-1, N_PHASE), out[:, N_PHASE]


def phase_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - mean cosine between predicted and target angles (0 = exact)."""
    return 1.0 - (pred * target).reshape(-1, 6, 2).sum(2).mean()


def write_loss(params, x, target_phase) -> torch.Tensor:
    """What a write minimises: these views map to this goal, with c -> 1."""
    ph, c = forward(params, x)
    return phase_loss(ph, target_phase.expand_as(ph)) + F.binary_cross_entropy_with_logits(
        c, torch.ones_like(c))


def write(params, log_alpha, x, target_phase, k: int = 1, create_graph: bool = True):
    """``k`` gradient steps with per-weight learning rates exp(log_alpha).

    ``create_graph=True`` keeps the steps differentiable, so the outer loss can
    reach the initial weights and the learning rates through every write.
    """
    for _ in range(k):
        loss = write_loss(params, x, target_phase)
        grads = torch.autograd.grad(loss, params, create_graph=create_graph)
        params = [p - la.exp() * g for p, la, g in zip(params, log_alpha, grads)]
    return params


__all__ = ["N_PHASE", "canvas", "decode_cells", "forward", "goal_phases",
           "init_params", "phase_loss", "write", "write_loss"]
