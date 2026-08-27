"""Data generation and batch sampling for binary-method encoder training."""
from __future__ import annotations

import math
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

def _grid_dims(nenv: int, H: int, W: int, min_cell: int = 0
               ) -> Tuple[int, int] | None:
    """Rows x cols for a stratifying grid: least waste, then squarest cells.

    Enumerates every ``(rows, cols)`` with at least ``nenv`` cells and scores
    ``(cells - nenv) + 3 * |log(cell_h / cell_w)|`` — one wasted cell is worth
    about a 1.4x cell aspect ratio. For a square arena this gives 3x3 for 7
    patches (over the tighter but 2:1 elongated 2x4), 3x4 for 11, 5x6 for 29.

    For a given ``rows``, ``cols = ceil(nenv / rows)`` is the fewest columns
    that still reach ``nenv`` cells and so the widest they can be, which is why
    a ``min_cell`` floor can be applied inside the same enumeration. Returns
    ``None`` when no grid of at least ``nenv`` cells has cells that big.
    """
    best, best_score = None, float("inf")
    for rows in range(1, nenv + 1):
        cols = -(-nenv // rows)                     # ceil
        cell_h, cell_w = H / rows, W / cols
        if min_cell and (cell_h < min_cell or cell_w < min_cell):
            continue
        score = (rows * cols - nenv) + 3.0 * abs(math.log(cell_h / cell_w))
        if score < best_score:
            best, best_score = (rows, cols), score
    return best


def _stratified_patches(
    H: int, W: int, sizes: List[int], max_attempts: int,
) -> Tuple[List[int], List[int]]:
    """Jittered lattice: patches spread over a coarse grid, offset at random.

    Motivation (§4.6): the references that hold ``r_min`` down are the ones no
    training patch got near — the correlation between a reference's distance to
    the nearest patch and its radius is -0.47. Rejection sampling makes that
    distance long in whatever holes it happens to leave; stratifying bounds it
    by construction. Measured over seeds 42-45 on the §5 mixes, it roughly
    halves the worst hole (``lo_mixtop``: 839 -> 461 cells). At the 50.8%
    coverage §4 won with, random sampling already leaves a worst hole of 192,
    so there is nothing there to fix — which is the control the claim needs.

    Two regimes, because a patch can be larger than its share of the arena:

    * **Sparse** — some grid of at least ``len(sizes)`` cells has cells the
      largest patch fits in. One patch per cell, on a random subset of cells,
      offset so it stays wholly inside. Placement cannot fail and no rejection
      is needed: the gap between neighbouring cells does the work rejection
      sampling has to search for.
    * **Dense** — no such grid (e.g. 54 patches up to 200 cells in a 1716
      arena). Use the finest grid whose cells still fit the largest patch,
      deal the patches round-robin largest-first so every cell gets a similar
      size mix, and rejection-sample within each cell. Patches in different
      cells still cannot collide, so the rejection problem is local and much
      easier than the global one — but at high enough coverage it can still
      fail, the same geometric wall documented for the random path.
    """
    nenv = len(sizes)
    smax = max(sizes)
    order = sorted(range(nenv), key=lambda i: -sizes[i])

    dims = _grid_dims(nenv, H, W, min_cell=smax)
    if dims is not None:
        rows, cols = dims
        cell_of = dict(zip(order, torch.randperm(rows * cols)[:nenv].tolist()))
    else:
        rows = max(1, H // smax)
        cols = max(1, W // smax)
        shuffled = torch.randperm(rows * cols).tolist()
        cell_of = {i: shuffled[k % len(shuffled)]
                   for k, i in enumerate(order)}

    y0s: List[int] = [0] * nenv
    x0s: List[int] = [0] * nenv
    in_cell: dict[int, List[Tuple[int, int, int]]] = {}
    attempts = 0
    for i in order:
        s = sizes[i]
        r, c = divmod(cell_of[i], cols)
        ylo, yhi = round(r * H / rows), round((r + 1) * H / rows)
        xlo, xhi = round(c * W / cols), round((c + 1) * W / cols)
        neighbours = in_cell.setdefault(cell_of[i], [])

        placed = False
        while True:
            y0 = ylo + int(torch.randint(0, yhi - ylo - s + 1, (1,)).item())
            x0 = xlo + int(torch.randint(0, xhi - xlo - s + 1, (1,)).item())
            if not any(_overlaps(y0, x0, s, *p) for p in neighbours):
                placed = True
                break
            attempts += 1
            if attempts >= nenv * max_attempts:
                break
        if not placed:
            raise RuntimeError(
                f"Could only place {sum(map(len, in_cell.values()))}/{nenv} "
                f"patches (stratified, {rows}x{cols} grid): no room left for a "
                f"{s}-cell patch in its {yhi - ylo}x{xhi - xlo} cell")
        y0s[i], x0s[i] = y0, x0
        neighbours.append((y0, x0, s))
    return y0s, x0s


def _overlaps(y0a, x0a, sa, y0b, x0b, sb):
    return not (y0a + sa <= y0b or y0b + sb <= y0a or
                x0a + sa <= x0b or x0b + sb <= x0a)


def sample_nonoverlapping_patches(
    H: int, W: int,
    npos: int | list[int],
    nenv: int | None = None,
    max_attempts: int = 20_000,
    placement: str = "random",
) -> Tuple[List[int], List[int], List[int]]:
    """Sample non-overlapping square patches.

    Args:
        H, W: grid bounds.
        npos: either an int (all patches that size) or a list of sizes.
        nenv: number of patches (required if npos is int; ignored otherwise).
        max_attempts: rejections tolerated, ``nenv * max_attempts`` in total
            across the whole layout — the budget is shared, not per patch, so
            the last patches spend what the earlier ones left.
        placement: ``"random"`` for uniform rejection sampling (everything
            through §4 used this), or ``"stratified"`` for a jittered lattice —
            see :func:`_stratified_patches`.

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

    if placement == "stratified":
        y0s, x0s = _stratified_patches(H, W, sizes, max_attempts)
        return y0s, x0s, sizes
    if placement != "random":
        raise ValueError(f"unknown placement {placement!r}")

    overlaps = _overlaps
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
