"""Data generation and batch sampling for binary-method encoder training."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from gridcode.codebook import gen_gbook_2d

from gridcode.smoothing import smooth_gbook


class IndexDataset(Dataset):
    def __init__(self, n: int):
        self.n = n
    def __len__(self) -> int: return self.n
    def __getitem__(self, i: int) -> int: return i


# ---------------------------------------------------------------------------
# Full-grid code generation
# ---------------------------------------------------------------------------

def build_full_grid(
    lambdas: list[int],
    fwhm_ratio: float = 0.25,
) -> Tuple[torch.Tensor, int]:
    """Generate the full smoothed grid-code book.

    Returns:
        Phi_full: (code_dim, full_Npos, full_Npos) float32 CPU tensor.
        full_Npos: grid side length = product of lambdas.
    """
    full_Npos = int(np.prod(lambdas))
    Ng = int(np.sum(np.square(lambdas)))
    gbook = gen_gbook_2d(lambdas, Ng, full_Npos)
    Phi = torch.tensor(
        smooth_gbook(gbook, lambdas, fwhm_ratio) if fwhm_ratio > 0
        else gbook.astype(np.float32),
        dtype=torch.float32,
    )
    return Phi, full_Npos


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

def sample_nonoverlapping_patches(
    H: int, W: int,
    npos: int | list[int],
    nenv: int | None = None,
    max_attempts: int = 20_000,
) -> Tuple[List[int], List[int], List[int]]:
    """Sample non-overlapping square patches via rejection sampling.

    Args:
        H, W: grid bounds.
        npos: either an int (all patches that size) or a list of sizes.
        nenv: number of patches (required if npos is int; ignored otherwise).
        max_attempts: rejections tolerated, ``nenv * max_attempts`` in total
            across the whole layout — the budget is shared, not per patch, so
            the last patches spend what the earlier ones left.

    Returns (y0s, x0s, sizes) as parallel lists of length nenv.

    Raised from 1000 for headroom, but note what it does *not* buy: above about
    60% coverage the binding constraint is geometric, not the budget. A 65%
    layout of 200/150/100 patches fails at seed 44 with "Could only place
    49/68" and fails identically with twenty times the attempts, because once
    the large squares are scattered at random there is no 150-cell gap left
    anywhere. Placing largest-first (as every caller here does) delays that but
    does not avoid it, and no amount of rejection helps a layout that needs
    backtracking or a tiling.

    The practical consequence is that **the reachable coverage depends on the
    seed**, so a mix must be placement-checked at every seed a sweep will use.
    Checking two and launching four is how ``w15`` lost half its cells.
    """
    if isinstance(npos, int):
        assert nenv is not None, "nenv required when npos is an int"
        sizes = [npos] * nenv
    else:
        sizes = list(npos)
        nenv = len(sizes)

    def overlaps(y0a, x0a, sa, y0b, x0b, sb):
        return not (y0a + sa <= y0b or y0b + sb <= y0a or
                    x0a + sa <= x0b or x0b + sb <= x0a)

    y0s, x0s = [], []
    total_attempts = 0
    max_total = nenv * max_attempts
    for i in range(nenv):
        s = sizes[i]
        placed = False
        while total_attempts < max_total:
            y0 = int(torch.randint(0, H - s + 1, (1,)).item())
            x0 = int(torch.randint(0, W - s + 1, (1,)).item())
            if not any(overlaps(y0, x0, s, y0s[j], x0s[j], sizes[j])
                       for j in range(len(y0s))):
                y0s.append(y0); x0s.append(x0)
                placed = True
                break
            total_attempts += 1
        if not placed:
            raise RuntimeError(f"Could only place {len(y0s)}/{nenv} patches")
    return y0s, x0s, sizes


def build_patch_codes(
    lambdas: list[int],
    y0s: List[int],
    x0s: List[int],
    sizes: List[int],
    device: torch.device,
    fwhm_ratio: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``extract_patches`` without ever building the full codebook.

    ``build_full_grid`` materialises ``(Ng, Npos, Npos)`` float64 — 10.2 GB at
    lambdas (11, 12, 13) — and training then keeps only the ~20% of it that
    falls inside a patch. Since the code is one-hot per module and the smoothed
    block is a separable wrapped-Gaussian bump, each patch can be built
    directly, which drops the run's peak host memory from ~20 GB to ~1 GB.

    That is the difference between one training run per node and several, which
    is the binding constraint when the partition is full. Verified against the
    full-codebook path in ``tests/test_lazy_patch_codes.py``.
    """
    from encoder_training.eval_unique_radius import grid_code_batch

    all_phi, all_coords, all_env = [], [], []
    for i, s in enumerate(sizes):
        ys, xs = torch.meshgrid(
            torch.arange(y0s[i], y0s[i] + s),
            torch.arange(x0s[i], x0s[i] + s), indexing="ij")
        ys, xs = ys.reshape(-1), xs.reshape(-1)
        # grid_code_batch's first argument indexes the first spatial axis of
        # the codebook, which is the one extract_patches slices with y0.
        all_phi.append(torch.from_numpy(
            grid_code_batch(lambdas, ys.numpy(), xs.numpy(), fwhm_ratio)))
        all_coords.append(torch.stack([ys.float(), xs.float()], dim=-1))
        all_env.append(torch.full((s * s,), i, dtype=torch.long))
    return (
        torch.cat(all_phi).to(device),
        torch.cat(all_coords).to(device),
        torch.cat(all_env).to(device),
    )


def extract_patches(
    Phi_full: torch.Tensor,
    y0s: List[int],
    x0s: List[int],
    sizes: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract patches → (Phi_flat, coords, env_ids), all on `device`."""
    all_phi, all_coords, all_env = [], [], []
    for i, s in enumerate(sizes):
        patch = Phi_full[:, y0s[i]:y0s[i] + s, x0s[i]:x0s[i] + s]   # [C, s, s]
        all_phi.append(patch.reshape(patch.shape[0], -1).T)         # [s*s, C]
        ys, xs = torch.meshgrid(
            torch.arange(y0s[i], y0s[i] + s),
            torch.arange(x0s[i], x0s[i] + s), indexing="ij")
        all_coords.append(
            torch.stack([ys.reshape(-1).float(), xs.reshape(-1).float()], dim=-1))
        all_env.append(torch.full((s * s,), i, dtype=torch.long))
    return (
        torch.cat(all_phi).to(device),
        torch.cat(all_coords).to(device),
        torch.cat(all_env).to(device),
    )


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def mixed_batch_iterator(n_points: int, batch_size: int):
    """Yield random batches across the entire dataset (mixed envs)."""
    dl = DataLoader(IndexDataset(n_points), batch_size=batch_size,
                    shuffle=True, drop_last=True)
    for idx in dl:
        yield idx


def single_env_batch_iterator(
    env_indices: List[torch.Tensor],
    batch_size: int,
):
    """Yield one batch per env per call, drawn only from that env's points.

    env_indices: list of 1-D long tensors, where env_indices[e] is the set of
    global point indices belonging to env e.
    """
    order = torch.randperm(len(env_indices)).tolist()
    for e in order:
        ids = env_indices[e]
        if len(ids) >= batch_size:
            perm = torch.randperm(len(ids))[:batch_size]
            yield ids[perm]
        else:
            yield ids
