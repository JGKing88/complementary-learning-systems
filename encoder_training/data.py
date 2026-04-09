"""Data generation and kernel computation for encoder training."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from cls.vectorhash.assoc_utils_np_2D import gen_gbook_2d

from .utils import smooth_gbook


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IndexDataset(Dataset):
    """Trivial dataset that just yields integer indices."""
    def __init__(self, N: int):
        self.N = N
    def __len__(self) -> int:
        return self.N
    def __getitem__(self, idx: int) -> int:
        return idx


# ---------------------------------------------------------------------------
# Grid code generation
# ---------------------------------------------------------------------------

def build_grid_data(
    lambdas: list[int],
    fwhm_ratio: float = 0.25,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Generate grid codes and coordinate tensors.

    Returns:
        Phi: (Npos*Npos, Ng) flattened grid codes (optionally smoothed)
        Xcoords: (Npos*Npos, 2) grid coordinates (y, x)
        Npos: grid side length = product of lambdas
    """
    Ng = sum(l * l for l in lambdas)
    Npos = int(np.prod(lambdas))

    gbook = gen_gbook_2d(lambdas, Ng, Npos)  # (Ng, Npos, Npos)

    if fwhm_ratio > 0:
        gbook = smooth_gbook(gbook, lambdas, fwhm_ratio)

    # Flatten to (N, Ng) where N = Npos*Npos
    Phi = torch.from_numpy(
        gbook.reshape(Ng, -1).T.astype(np.float32)
    ).to(device)  # (N, Ng)

    # Build coordinate tensor
    ys, xs = np.meshgrid(np.arange(Npos), np.arange(Npos), indexing='ij')
    Xcoords = torch.from_numpy(
        np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    ).to(device)  # (N, 2)

    return Phi, Xcoords, Npos


# ---------------------------------------------------------------------------
# Kernel computation
# ---------------------------------------------------------------------------

def rbf_kernel_batch(
    Xcoords: torch.Tensor,
    idx: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Build RBF kernel K[idx, idx] on the fly.

    Xcoords: (N, 2), idx: (B,).  Returns (B, B).
    """
    Xb = Xcoords.index_select(0, idx)
    diff = Xb[:, None, :] - Xb[None, :, :]
    dist2 = (diff * diff).sum(-1)
    return torch.exp(-dist2 / (2.0 * tau ** 2))


def rbf_kernel_local(
    Xcoords: torch.Tensor,
    idx: torch.Tensor,
    tau: float | None = None,
) -> tuple[torch.Tensor, float]:
    """RBF kernel with local (median-based) tau estimation."""
    Xi = Xcoords.index_select(0, idx)
    x2 = (Xi ** 2).sum(1, keepdim=True)
    D2 = (x2 + x2.T - 2 * (Xi @ Xi.T)).clamp_min(0.0)
    if tau is None:
        tri = torch.triu(torch.ones_like(D2, dtype=torch.bool), diagonal=1)
        tau = float(torch.quantile(torch.sqrt(D2[tri] + 1e-12), 0.5).clamp_min(1e-6).item())
    K = torch.exp(-D2 / (2.0 * tau ** 2))
    return K, tau


@torch.no_grad()
def estimate_tau_median(Xcoords: torch.Tensor, sample_pairs: int = 50000) -> float:
    """Estimate median pairwise distance from random pairs."""
    N = Xcoords.size(0)
    i = torch.randint(0, N, (sample_pairs,), device=Xcoords.device)
    j = torch.randint(0, N, (sample_pairs,), device=Xcoords.device)
    mask = i != j
    d = torch.linalg.norm(Xcoords[i[mask]] - Xcoords[j[mask]], dim=-1)
    return float(d.median().item())


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

def sample_random_patches(
    H: int, W: int,
    num_patches: int = 16,
    min_size: int = 4,
    max_size: int = 10,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Sample random rectangular patches, return lists of flat indices (y*W + x)."""
    device = device or torch.device("cpu")
    patches = []
    for _ in range(num_patches):
        ph = torch.randint(min_size, max_size + 1, (1,), device=device).item()
        pw = torch.randint(min_size, max_size + 1, (1,), device=device).item()
        ph, pw = min(ph, H), min(pw, W)
        y0 = torch.randint(0, max(1, H - ph + 1), (1,), device=device).item()
        x0 = torch.randint(0, max(1, W - pw + 1), (1,), device=device).item()
        rows = []
        for dy in range(ph):
            rows.append(torch.arange(x0, x0 + pw, device=device, dtype=torch.long) + (y0 + dy) * W)
        patches.append(torch.cat(rows))
    return patches


# ---------------------------------------------------------------------------
# Triple generation for coplanarity loss
# ---------------------------------------------------------------------------

def build_grid_triples(
    H: int, W: int,
    stride: int = 1,
    include_diagonals: bool = False,
    both_directions: bool = True,
) -> torch.Tensor:
    """Build (T, 3) LongTensor of consecutive triple indices for a H x W grid."""
    triples = []

    for y in range(H):
        for x in range(W - 2 * stride):
            i0, i1, i2 = y * W + x, y * W + x + stride, y * W + x + 2 * stride
            triples.append((i0, i1, i2))
            if both_directions:
                triples.append((i2, i1, i0))

    for x in range(W):
        for y in range(H - 2 * stride):
            i0, i1, i2 = y * W + x, (y + stride) * W + x, (y + 2 * stride) * W + x
            triples.append((i0, i1, i2))
            if both_directions:
                triples.append((i2, i1, i0))

    if include_diagonals:
        for y in range(H - 2 * stride):
            for x in range(W - 2 * stride):
                i0 = y * W + x
                i1 = (y + stride) * W + x + stride
                i2 = (y + 2 * stride) * W + x + 2 * stride
                triples.append((i0, i1, i2))
                if both_directions:
                    triples.append((i2, i1, i0))
            for x in range(2 * stride, W):
                i0 = y * W + x
                i1 = (y + stride) * W + x - stride
                i2 = (y + 2 * stride) * W + x - 2 * stride
                triples.append((i0, i1, i2))
                if both_directions:
                    triples.append((i2, i1, i0))

    return torch.tensor(triples, dtype=torch.long)
