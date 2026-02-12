import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable, Tuple
import torch.nn.functional as F
from itertools import product

from scipy.stats import spearmanr, pearsonr

# Ensure project root on sys.path for local imports when running directly
try:
    HERE = os.path.dirname(os.path.abspath(__file__))  # type: ignore[name-defined]
except NameError:
    HERE = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cls import WMEnv
from cls.utils.GridUtils import *
from cls.envs.environments import GridWMEnv
from cls.vectorhash.seq_utils import *
from cls.vectorhash.assoc_utils_np import *
from cls.vectorhash.senstranspose_utils import *
from cls.vectorhash.assoc_utils_np_2D import gen_gbook_2d, path_integration_Wgg_2d, module_wise_NN_2d
from cls.encoder import GridEncoder, GridEncoderCNN

seed = 3
rng = np.random.default_rng(seed)

class SphericalMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, nonlinearity: str = "gelu", output_nonlinearity: str = "tanh", gain: float = 1):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(nonlinearity.lower(), nn.GELU)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act(),
            nn.Linear(hidden, out_dim),
        )
        self.output_nonlinearity = output_nonlinearity
        self.gain = gain

    def forward(self, x: torch.Tensor, gain = None) -> torch.Tensor:
        if gain is None:
            gain = self.gain
        z = self.net(x)                         # [B, out_dim]
        if self.output_nonlinearity == "tanh":
            z = torch.tanh(gain * z)
        elif self.output_nonlinearity == "sigmoid":
            z = torch.sigmoid(gain * z)
        z = F.normalize(z, p=2, dim=-1)          # project to unit sphere
        return z

def _center_kernel(K: torch.Tensor) -> torch.Tensor:
    # Double-centering: Kc = K - row_mean - col_mean + grand_mean
    mean_row = K.mean(dim=1, keepdim=True)
    mean_col = K.mean(dim=0, keepdim=True)
    mean_all = K.mean()
    return K - mean_row - mean_col + mean_all
    
# -----------------------------
# Kernel alignment losses
#   - uncentered (classic alignment)
#   - centered (CKA-style), more robust in practice
# -----------------------------
def kernel_alignment_loss(K_pred: torch.Tensor, K_tgt: torch.Tensor, centered: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """
    K_pred, K_tgt: [B, B] (same B). Returns 1 - alignment (minimize).
    """
    assert K_pred.ndim == 2 and K_tgt.ndim == 2, "kernels must be 2D"
    assert K_pred.shape == K_tgt.shape, "kernels must be same shape"

    if centered:
        Kp = _center_kernel(K_pred)
        Kt = _center_kernel(K_tgt)
    else:
        # mean subtraction makes it equivalent to Pearson-style correlation between matrices
        Kp = K_pred - K_pred.mean()
        Kt = K_tgt - K_tgt.mean()

    num = (Kp * Kt).sum()
    den = torch.sqrt(Kp.square().sum().clamp_min(eps)) * torch.sqrt(Kt.square().sum().clamp_min(eps))
    return 1.0 - num / den

def weighted_kernel_alignment_loss(
    K_pred: torch.Tensor,   # [B,B]
    K_tgt: torch.Tensor,    # [B,B]
    topk: int = 16,
    near_mult: float = 5.0,
    centered: bool = True,
    eps: float = 1e-8,
):
    assert K_pred.shape == K_tgt.shape
    B = K_pred.size(0)

    # center
    if centered:
        Kp = _center_kernel(K_pred)
        Kt = _center_kernel(K_tgt)
    else:
        Kp = K_pred - K_pred.mean()
        Kt = K_tgt  - K_tgt.mean()

    # build weights: base 1 everywhere off-diagonal
    W = torch.ones((B, B), device=K_pred.device, dtype=K_pred.dtype)
    W.fill_diagonal_(0.0)

    if topk is not None and topk < B - 1:
        # topk according to target similarity (exclude self)
        Kt_masked = K_tgt.clone()
        Kt_masked.fill_diagonal_(-1e9)
        _, nn = torch.topk(Kt_masked, k=topk, dim=1)  # [B, topk]
        W.scatter_(1, nn, near_mult)
        # symmetrize so pair (i,j) is "near" if either i picks j or j picks i
        W = torch.maximum(W, W.T)

    # apply weights
    A = W * Kp
    Bm = W * Kt

    num = (A * Bm).sum()
    den = torch.sqrt(A.square().sum().clamp_min(eps)) * torch.sqrt(Bm.square().sum().clamp_min(eps))
    return 1.0 - num / den


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Uniformity regularizer on the sphere.
    z : [B, d] embeddings (they can be raw; we L2-normalize inside)
    t : temperature (larger => stronger repulsion)

    Loss = log( mean_{i!=j} exp( -t * ||zi - zj||^2 ) )
    Lower is more uniform (more spread out).
    """
    # L2-normalize so distances are angular
    z = torch.nn.functional.normalize(z, dim=-1)  # [B, d]
    B = z.size(0)
    if B < 2:
        return z.new_zeros(())

    # Pairwise squared distances using Gram matrix
    # ||zi - zj||^2 = 2 - 2 * <zi, zj> for unit vectors
    S = z @ z.t()                                  # [B, B]
    dist2 = (2.0 - 2.0 * S).clamp_min(0.0)

    # Exclude diagonal
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    dist2_off = dist2[mask]

    # Uniformity: log mean exp(-t * d^2)
    loss = torch.logsumexp(-t * dist2_off, dim=0) - torch.log(torch.tensor(dist2_off.numel(), device=z.device, dtype=z.dtype))
    return loss

def coplanarity_loss_sphere(z_triples: torch.Tensor) -> torch.Tensor:
    """
    z_triples: [T, 3, d], assumed (or made) unit-norm.
    Loss = mean det(Gram([u,v,w])) = mean parallelepiped volume^2.
    det(G)= 1 + 2abc - (a^2 + b^2 + c^2) for a=<u,v>, b=<u,w>, c=<v,w>.
    """
    z = F.normalize(z_triples, dim=-1)
    u, v, w = z[:, 0, :], z[:, 1, :], z[:, 2, :]
    a = (u * v).sum(-1)
    b = (u * w).sum(-1)
    c = (v * w).sum(-1)
    detG = 1.0 + 2.0 * a * b * c - (a * a + b * b + c * c)
    return detG.mean()

def build_grid_triples(H: int, W: int, stride: int = 1, include_diagonals: bool = False, both_directions: bool = True) -> torch.Tensor:
    """
    Returns LongTensor [T, 3] of flattened indices (y*W + x) for consecutive triples.
    - Rows: (x, x+stride, x+2*stride) for each y
    - Cols: (y, y+stride, y+2*stride) for each x
    - Optionally diagonals (down-right, down-left)
    - Optionally both directions (also reversed triples)
    """
    triples = []

    # rows
    for y in range(H):
        for x in range(0, W - 2 * stride):
            i0 = y * W + x
            i1 = y * W + (x + stride)
            i2 = y * W + (x + 2 * stride)
            triples.append((i0, i1, i2))
            if both_directions:
                triples.append((i2, i1, i0))

    # cols
    for x in range(W):
        for y in range(0, H - 2 * stride):
            i0 = y * W + x
            i1 = (y + stride) * W + x
            i2 = (y + 2 * stride) * W + x
            triples.append((i0, i1, i2))
            if both_directions:
                triples.append((i2, i1, i0))

    if include_diagonals:
        # down-right
        for y in range(0, H - 2 * stride):
            for x in range(0, W - 2 * stride):
                i0 = y * W + x
                i1 = (y + stride) * W + (x + stride)
                i2 = (y + 2 * stride) * W + (x + 2 * stride)
                triples.append((i0, i1, i2))
                if both_directions:
                    triples.append((i2, i1, i0))
        # down-left
        for y in range(0, H - 2 * stride):
            for x in range(2 * stride, W):
                i0 = y * W + x
                i1 = (y + stride) * W + (x - stride)
                i2 = (y + 2 * stride) * W + (x - 2 * stride)
                triples.append((i0, i1, i2))
                if both_directions:
                    triples.append((i2, i1, i0))

    return torch.tensor(triples, dtype=torch.long)  # [T, 3]

def plane_loss_from_triples(
    encoder,
    Phi: torch.Tensor,          # [N, code_dim]
    triples_all: torch.Tensor,  # [T_all, 3] (CPU okay)
    T_batch: int,
    gain: float = 1.0,
) -> torch.Tensor:
    """
    Samples T_batch triples, encodes only unique indices once, computes spherical coplanarity loss.
    """
    device = Phi.device
    T_all = triples_all.size(0)
    sel = torch.randint(0, T_all, (min(T_batch, T_all),), device=triples_all.device)
    triples = triples_all[sel].to(device)                          # [Tb, 3]

    flat = triples.reshape(-1)                                     # [3*Tb]
    uniq, inv = torch.unique(flat, return_inverse=True)            # uniq: [U], inv: [3*Tb]
    Zuniq = encoder(Phi.index_select(0, uniq), gain)      # [U, d]
    z_triples = Zuniq.index_select(0, inv).view(-1, 3, Zuniq.size(-1))  # [Tb, 3, d]

    return coplanarity_loss_sphere(z_triples)


def sample_random_patches(H: int, W: int,
                          num_patches: int = 16,
                          max_ph: int = 10, max_pw: int = 10,
                          min_ph: int = 4,  min_pw: int = 4,
                          device=None) -> list[torch.Tensor]:
    """
    Returns a list of 1D LongTensors; each is the flattened indices (y*W+x)
    of a random H×W patch. Patch sizes are uniform in [min,max] per axis.
    """
    device = device or torch.device("cpu")
    patches = []
    for _ in range(num_patches):
        ph = torch.randint(min_ph, max_ph + 1, (1,), device=device).item()
        pw = torch.randint(min_pw, max_pw + 1, (1,), device=device).item()
        ph = min(ph, H); pw = min(pw, W)
        y0 = torch.randint(0, max(1, H - ph + 1), (1,), device=device).item()
        x0 = torch.randint(0, max(1, W - pw + 1), (1,), device=device).item()

        rows = []
        for dy in range(ph):
            y = y0 + dy
            row = torch.arange(x0, x0 + pw, device=device, dtype=torch.long) + y * W
            rows.append(row)
        patches.append(torch.stack(rows, dim=0).reshape(-1))  # [ph*pw]
    return patches  # list of [Pi]

def rbf_from_coords_local(Xcoords: torch.Tensor, idx: torch.Tensor, tau: float | None = None):
    """
    Builds RBF kernel for a small set of indices with a LOCAL tau (median distance).
    Returns K:[B,B], tau_local:scalar
    """
    Xi = Xcoords.index_select(0, idx)                  # [B, 2]
    x2 = (Xi**2).sum(dim=1, keepdim=True)
    D2 = (x2 + x2.T - 2 * (Xi @ Xi.T)).clamp_min(0.0)
    if tau is None:
        tri = torch.triu(torch.ones_like(D2, dtype=torch.bool), diagonal=1)
        D = torch.sqrt(D2 + 1e-12)
        tau = torch.quantile(D[tri], 0.5).clamp_min(1e-6)
    K = torch.exp(- D2 / (2.0 * (tau**2)))
    return K, tau

def _weighted_local_mse(Kp: torch.Tensor, Kt: torch.Tensor, alpha: float = 2.0, topk: int | None = 8) -> torch.Tensor:
    """
    Weighted MSE between Kp (cosine mapped to [0,1]) and Kt (RBF in [0,1]),
    emphasizing near pairs via weights w_ij = Kt^alpha. Optionally restrict to topk per row.
    """
    B = Kp.size(0)
    eye = torch.eye(B, device=Kp.device, dtype=Kp.dtype)
    Kp01 = (Kp + 1.0) * 0.5
    Kp01 = Kp01 * (1.0 - eye)
    Kt    = Kt   * (1.0 - eye)

    W = (Kt.clamp(0, 1)) ** alpha
    if topk is not None and topk < B - 1:
        vals, idx = torch.topk(Kt, k=min(topk, B-1), dim=1)
        mask = torch.zeros_like(Kt, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        mask = mask | mask.T  # symmetrize
        if mask.float().mean() < 0.02:  # fallback if too sparse
            mask = torch.ones_like(Kt, dtype=torch.bool); mask.fill_diagonal_(False)
        W = W * mask

    active = (W > 0).float().sum().clamp_min(1.0)
    W = W * (active / W.sum().clamp_min(1e-8))  # normalize to keep scale stable
    return ((W * (Kp01 - Kt))**2).sum() / active

def local_weighted_alignment_loss(
    encoder,
    Phi: torch.Tensor,          # [N, code_dim]
    Xcoords: torch.Tensor,      # [N, 2] (grid coords)
    H: int, W: int,
    num_patches: int = 16,
    max_ph: int = 10, max_pw: int = 10,
    min_ph: int = 4,  min_pw: int = 4,
    alpha: float = 2.0,
    topk: int | None = 8,
    gain: float = 1.0,
) -> torch.Tensor:
    """
    Samples multiple small patches across the grid (global coverage),
    encodes all unique points once, and averages weighted local MSE
    (with per-patch local τ) over patches.
    """
    device = Phi.device
    patches = sample_random_patches(H, W, num_patches, max_ph, max_pw, min_ph, min_pw, device=device)
    if len(patches) == 0:
        return Phi.new_zeros(())

    # Gather all unique indices once
    all_idx = torch.unique(torch.cat(patches, dim=0))
    Z_all = encoder(Phi.index_select(0, all_idx), gain)     # [U, d]
    Z_all = F.normalize(Z_all, dim=-1)

    # Map each patch to positions in Z_all for fast slicing
    # Build a lookup from original index -> position in all_idx
    # (torch.unique with return_inverse could also be used; we keep it simple)
    pos = {int(idx.item()): i for i, idx in enumerate(all_idx)}
    loss_sum = Phi.new_zeros(())
    count = 0.0

    for p in patches:
        if p.numel() < 4:
            continue
        # positions for this patch
        ids = torch.tensor([pos[int(ii.item())] for ii in p], device=device, dtype=torch.long)
        Zp = Z_all.index_select(0, ids)                                # [Bp, d]
        Kp = (Zp @ Zp.T).clamp(-1.0, 1.0)                              # [Bp, Bp]
        Kt, _tau_loc = rbf_from_coords_local(Xcoords, p)               # [Bp, Bp], local τ

        loss_sum = loss_sum + _weighted_local_mse(Kp, Kt, alpha=alpha, topk=topk)
        count += 1.0

    if count == 0.0:
        return Phi.new_zeros(())
    return loss_sum / count


    
@torch.no_grad()
def estimate_tau_median(X_coords: torch.Tensor, sample_pairs: int = 50000) -> float:
    """
    Estimate median pairwise distance from a random subset of pairs.
    X_coords: [N, 2] on CPU or GPU.
    """
    N = X_coords.size(0)
    device = X_coords.device
    i = torch.randint(0, N, (sample_pairs,), device=device)
    j = torch.randint(0, N, (sample_pairs,), device=device)
    mask = i != j
    i, j = i[mask], j[mask]
    d = torch.linalg.norm(X_coords[i] - X_coords[j], dim=-1)
    return float(d.median().item())

def rbf_kernel_from_coords_batch(X_coords: torch.Tensor, idx: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Build RBF kernel K[idx, idx] on-the-fly.
    X_coords: [N, 2] (or [N, d_xy]), idx: [B], returns [B, B].
    """
    Xb = X_coords.index_select(0, idx)               # [B, 2]
    diff = Xb[:, None, :] - Xb[None, :, :]           # [B, B, 2]
    dist2 = (diff * diff).sum(-1)                    # [B, B]
    K = torch.exp(- dist2 / (2.0 * (tau ** 2)))
    return K

# -----------------------------
# Target kernel builders g(D)
# -----------------------------
def rbf_from_dist(D: torch.Tensor, tau: float) -> torch.Tensor:
    # K_ij = exp( - D_ij^2 / (2 tau^2) )
    return torch.exp(- (D**2) / (2.0 * (tau**2)))

def linear_from_dist(D: torch.Tensor, scale: float) -> torch.Tensor:
    # clamp into [0, 1] then rescale to similarities (1 close, 0 far)
    K = torch.clamp(1.0 - D / scale, min=0.0, max=1.0)
    return K

# -----------------------------
# Utilities: pairwise Euclidean distances
# -----------------------------
def pairwise_dist(xy: torch.Tensor) -> torch.Tensor:
    # xy: [N, d]
    # returns D: [N, N]
    x2 = (xy**2).sum(dim=1, keepdim=True)
    D2 = x2 + x2.T - 2.0 * (xy @ xy.T)
    D2 = torch.clamp(D2, min=0.0)
    return torch.sqrt(D2 + 1e-12)

# -----------------------------
# Dataset that just yields indices (we build kernels from batch slices)
# -----------------------------
class IndexDataset(Dataset):
    def __init__(self, N: int):
        self.N = N
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        return idx

# -----------------------------
# Training step
# -----------------------------
def train_epoch(
    encoder,
    Phi,
    Xcoords,
    tau,
    optimizer,
    batch_size,
    triples_all,
    T_triple_batch,
    centered=True,
    gain=1,
    uniformity_lambda=0,
    lambda_plane=0,
    lambda_local=0,
    # Modulate target kernel before CKA: Kt_mod = clamp(Kt,0,1)^cka_alpha with optional top-k mask
    cka_alpha: float = 1.0,
    cka_topk: int | None = None,
    mod_loss_lambda: float = 1.0,
):
    device = next(encoder.parameters()).device
    # ds = IndexDataset(Phi.size(0))
    ds = IndexDataset(Xcoords.size(0))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder.train()
    running = 0.0

    for idx in dl:
        idx = idx.to(device).long()
        xb = Phi[idx]                          # [B, p]
        zb = encoder(xb, gain)                       # [B, k]; encoder already L2-normalizes
        assert zb.ndim == 2, f"zb ndim != 2, got {zb.shape}"

        # robust kernel
        K_pred = torch.einsum('ik,jk->ij', zb, zb).clamp(-1.0, 1.0)   # [B, B]
        # K_tgt  = Kt_full.index_select(0, idx).index_select(1, idx)    # [B, B]
        K_tgt  = rbf_kernel_from_coords_batch(Xcoords, idx, tau)

        # Optionally modulate target kernel to emphasize local pairs inside the CKA term
        K_tgt_mod = K_tgt
        if (cka_alpha is not None and cka_alpha != 1.0) or (cka_topk is not None):
            B = K_tgt.size(0)
            eye = torch.eye(B, device=K_tgt.device, dtype=K_tgt.dtype)
            K_tgt_mod = K_tgt.clamp(0, 1)
            if cka_alpha is not None and cka_alpha != 1.0:
                K_tgt_mod = K_tgt_mod.pow(cka_alpha)
            if cka_topk is not None and cka_topk < B - 1:
                vals, idx = torch.topk(K_tgt, k=min(cka_topk, B-1), dim=1)
                mask = torch.zeros_like(K_tgt, dtype=torch.bool)
                mask.scatter_(1, idx, True)
                mask = mask | mask.T
                mask = mask & (~torch.eye(B, dtype=torch.bool, device=K_tgt.device))
                K_tgt_mod = K_tgt_mod * mask + eye  # keep diagonal at 1

        # alignment loss (centered = CKA-like) with modulated target kernel
        loss = kernel_alignment_loss(K_pred, K_tgt, centered=centered)
        mod_loss = kernel_alignment_loss(K_pred, K_tgt_mod, centered=centered) 
        loss = loss + mod_loss * mod_loss_lambda
        # loss = weighted_kernel_alignment_loss(K_pred, K_tgt, topk=cka_topk, near_mult=10.0, centered=True)
        loss += uniformity_lambda * uniformity_loss(zb)

        # after global loss pieces (CKA, uniformity, plane, etc.)
        with torch.no_grad():
            H = int(Xcoords[:,0].max().item() + 1)
            W = int(Xcoords[:,1].max().item() + 1)

        # L_local = local_weighted_alignment_loss(
        #     encoder, Phi, Xcoords, H, W,
        #     num_patches=16,         # 8–32 is typical
        #     max_ph=10, max_pw=10,   # cap patch size to keep O(P^2) cheap
        #     min_ph=4,  min_pw=4,
        #     alpha=2.0, topk=8,
        #     gain=gain,
        # )
        # loss = loss + lambda_local * L_local

        # --- AUX curvature pass ---
        L_plane = plane_loss_from_triples(
            encoder,
            Phi,                # [N, code_dim], already on device
            triples_all,        # precomputed [T_all, 3] (CPU ok)
            T_batch=T_triple_batch,
            gain=gain,
        )
        # loss = loss + lambda_plane * L_plane
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += float(loss)
    return running / len(dl)

def upper_tri_vec(M: torch.Tensor) -> torch.Tensor:
    tri = torch.triu(torch.ones_like(M), diagonal=1).bool()
    return M[tri]

@torch.no_grad()
def eval_full(
    encoder,
    Phi,                # [N, p]
    # Kt_full,            # [N, N] target kernel (e.g., RBF of D)
    Xcoords,
    tau,
    subset: int = 2048,
    centered: bool = True,
    D_true: torch.Tensor | None = None,  # [N, N] optional true distances
    tau_for_rbf: float | None = None,    # optional, if you want to reconstruct RBF from D
    gain: float = 1.0,
    cka_alpha: float = 1.0,
    cka_topk: int | None = None,
):
    """
    Returns a dict of metrics on a random subset.
    If D_true is provided, computes distance-based R^2 and Shepard stats.
    Otherwise, computes kernel-based metrics only.
    """
    device = next(encoder.parameters()).device
    N = Phi.size(0)
    M = min(subset, N)
    idx = torch.randperm(N, device=device)[:M]

    Xb = Phi[idx]                       # [M, p]
    Z   = encoder(Xb, gain)                   # [M, k], unit sphere
    Kp  = (Z @ Z.T).clamp(-1, 1)        # predicted cosine kernel
    # Kt  = Kt_full[idx][:, idx]          # target kernel on this subset
    Kt = rbf_kernel_from_coords_batch(X_coords=Xcoords, idx=idx, tau=tau)
    # Apply same modulation as training for alignment metric
    Kt_mod = Kt
    if (cka_alpha is not None and cka_alpha != 1.0) or (cka_topk is not None):
        B = Kt.size(0)
        eye = torch.eye(B, device=Kt.device, dtype=Kt.dtype)
        Kt_mod = Kt.clamp(0, 1)
        if cka_alpha is not None and cka_alpha != 1.0:
            Kt_mod = Kt_mod.pow(cka_alpha)
        if cka_topk is not None and cka_topk < B - 1:
            vals, idx_top = torch.topk(Kt, k=min(cka_topk, B-1), dim=1)
            mask = torch.zeros_like(Kt, dtype=torch.bool)
            mask.scatter_(1, idx_top, True)
            mask = mask | mask.T
            mask = mask & (~torch.eye(B, dtype=torch.bool, device=Kt.device))
            Kt_mod = Kt_mod * mask + eye

    # ---- Kernel correlation metrics (shape-of-geometry)
    s_pred = upper_tri_vec(Kp).detach().cpu().numpy()
    s_tgt  = upper_tri_vec(Kt).detach().cpu().numpy()

    # Pearson / Spearman of similarities
    try:
        from scipy.stats import spearmanr, pearsonr
        sp, _ = spearmanr(s_pred, s_tgt)
        pe, _ = pearsonr(s_pred, s_tgt)
    except Exception:
        # fallback without SciPy (Pearson)
        import numpy as np
        s0 = (s_pred - s_pred.mean()) / (s_pred.std() + 1e-8)
        t0 = (s_tgt  - s_tgt.mean())  / (s_tgt.std()  + 1e-8)
        pe = float(np.mean(s0 * t0))
        sp = float("nan")

    # Centered kernel alignment loss re-using your function
    align_loss = kernel_alignment_loss(Kp, Kt_mod, centered=centered).item()
    
    # after global loss pieces (CKA, uniformity, plane, etc.)
    # Note: local_weighted_alignment_loss assumes contiguous H*W grid, skip if using patch sampling
    with torch.no_grad():
        H = int(Xcoords[:,0].max().item() + 1)
        W = int(Xcoords[:,1].max().item() + 1)
    
    # Only compute local loss if Phi is a contiguous grid (N == H*W)
    N_phi = Phi.size(0)
    if N_phi == H * W:
        L_local = local_weighted_alignment_loss(
            encoder, Phi, Xcoords, H, W,
            num_patches=8,         # 8–32 is typical
            max_ph=6, max_pw=6,   # cap patch size to keep O(P^2) cheap
            min_ph=4,  min_pw=4,
            alpha=2.0,
            gain=gain,
        )
        local_loss = L_local.item()
    else:
        # Skip local loss for non-contiguous (patch-sampled) data
        local_loss = float('nan')

    # ---- Triplet accuracy (kernel order)
    # sample triplets (i, a, b) and check that Kp[i,a] > Kp[i,b] whenever Kt[i,a] > Kt[i,b]
    T = min(200000, M * 200)  # ~200 triplets per anchor; cap to keep fast
    i  = torch.randint(0, M, (T,), device=device)
    a  = torch.randint(0, M, (T,), device=device)
    b  = torch.randint(0, M, (T,), device=device)
    # avoid degenerate a==b or equal targets by masking later
    tgt_cmp = (Kt[i, a] > Kt[i, b])
    pred_cmp = (Kp[i, a] > Kp[i, b])
    mask = (a != b) & (i != a) & (i != b) & (Kt[i, a] != Kt[i, b])
    triplet_acc = (pred_cmp[mask] == tgt_cmp[mask]).float().mean().item() if mask.any() else float("nan")

    # ---- Nearest-neighbor consistency (kernel argmax)
    # nearest neighbor in target (exclude self)
    Kt_masked = Kt - torch.eye(M, device=device) * 1e9
    Kp_masked = Kp - torch.eye(M, device=device) * 1e9
    nn_tgt = Kt_masked.argmax(dim=1)
    nn_pred = Kp_masked.argmax(dim=1)
    nn_acc = (nn_tgt == nn_pred).float().mean().item()

    out = {
        "align_loss": align_loss,
        "pearson_sim": float(pe),
        "spearman_sim": float(sp),
        "triplet_acc": float(triplet_acc),
        "nn_consistency": float(nn_acc),
        "local_loss": float(local_loss),
    }

    # ---- Distance-based diagnostics (optional, if D_true is available)
    if D_true is not None:
        Db = D_true[idx][:, idx]                  # [M, M]
        # Predicted "distance" on the sphere: 1 - cosine (in [0, 2])
        S_pred = 1.0 - Kp
        # Choose a monotone map of D_true to compare (linear or exp). Use exp if tau provided:
        if tau_for_rbf is not None:
            # Map distance to similarity, then to a distance-like quantity to compare fairly
            # e.g., S_tgt = 1 - exp(-D^2/(2 tau^2))  ∈ [0,1]
            S_tgt = 1.0 - torch.exp(-(Db**2) / (2.0 * (tau_for_rbf**2)))
        else:
            # Scale D_true into [0,1] by dividing by its 95th percentile (robust)
            with torch.no_grad():
                tri = upper_tri_vec(Db)
                scale = torch.quantile(tri, 0.95).clamp(min=1e-6)
            S_tgt = (Db / scale).clamp(0, 1)

        sp_d = upper_tri_vec(S_pred).detach()
        st_d = upper_tri_vec(S_tgt).detach()

        # R^2
        st_mean = st_d.mean()
        ss_res = ((sp_d - st_d)**2).sum()
        ss_tot = ((st_d - st_mean)**2).sum().clamp(min=1e-12)
        r2 = float(1.0 - (ss_res / ss_tot))

        # Shepard line fit (least squares y = a*x + b), report slope, intercept, RMSE
        A = torch.stack([st_d, torch.ones_like(st_d)], dim=1)         # [P, 2]
        sol = torch.linalg.lstsq(A, sp_d).solution
        a = float(sol[0].item()); b = float(sol[1].item())
        rmse = float(torch.sqrt(((sp_d - (a*st_d + b))**2).mean()).item())

        out.update({
            "R2_distproxy": r2,
            "shepard_slope": a,
            "shepard_intercept": b,
            "shepard_RMSE": rmse,
        })

    return out

# ---------- (A) build random row/col sequences of length L on the HxW grid ----------
def sample_sequences(H: int, W: int, L: int, stride: int = 1, num_per_axis: int = 128, device=None):
    """
    Returns indices for sequences along rows and cols:
      seqs: [S, L] LongTensor of flattened indices (y*W + x)
    """
    seqs = []

    # rows
    if W >= (L-1)*stride + 1:
        ys = torch.randint(0, H, (num_per_axis,), device=device)
        xs = torch.randint(0, W - (L-1)*stride, (num_per_axis,), device=device)
        for y, x0 in zip(ys.tolist(), xs.tolist()):
            seqs.append([y*W + (x0 + k*stride) for k in range(L)])

    # cols
    if H >= (L-1)*stride + 1:
        xs = torch.randint(0, W, (num_per_axis,), device=device)
        ys = torch.randint(0, H - (L-1)*stride, (num_per_axis,), device=device)
        for x, y0 in zip(xs.tolist(), ys.tolist()):
            seqs.append([(y0 + k*stride)*W + x for k in range(L)])

    if not seqs:
        return torch.empty(0, L, dtype=torch.long, device=device)
    return torch.tensor(seqs, dtype=torch.long, device=device)

# ---------- (B) geodesic curvature via determinant of Gram for sliding triples ----------
def det_gram_triple(u, v, w):
    # u,v,w: [d], unit-norm expected
    a = torch.dot(u, v)
    b = torch.dot(u, w)
    c = torch.dot(v, w)
    return 1.0 + 2.0*a*b*c - (a*a + b*b + c*c)  # = vol^2 >= 0

def mean_triple_det_along_sequence(Z_seq):
    """
    Z_seq: [L, d] unit vectors along a single trajectory
    returns mean det(Gram) over sliding triples (0,1,2), (1,2,3), ...
    """
    L = Z_seq.size(0)
    if L < 3:
        return torch.tensor(0.0, device=Z_seq.device)
    vals = []
    for k in range(L-2):
        vals.append(det_gram_triple(Z_seq[k], Z_seq[k+1], Z_seq[k+2]))
    return torch.stack(vals).mean()

# ---------- (C) plane deviation: distance from each point to span{start, end} ----------
def plane_deviation_stats(Z_seq):
    """
    Z_seq: [L, d], unit vectors on sphere.
    Returns mean and max squared residual from the great-circle plane spanned by endpoints.
    """
    u0 = F.normalize(Z_seq[0], dim=0)
    u1 = F.normalize(Z_seq[-1], dim=0)

    # Build orthonormal basis Q for the 2D subspace span{u0, u1}
    q1 = u0
    r = u1 - torch.dot(q1, u1) * q1
    if torch.linalg.norm(r) < 1e-8:  # endpoints almost parallel; pick any orthonormal complement
        # find a vector orthogonal to q1
        e = torch.zeros_like(q1); e[0] = 1.0
        if torch.allclose(e, q1, atol=1e-6):
            e = torch.roll(e, 1)  # avoid exact colinearity
        r = e - torch.dot(q1, e) * q1
    q2 = F.normalize(r, dim=0)
    Q = torch.stack([q1, q2], dim=1)       # [d, 2]

    # Projection onto plane: P = Q Q^T
    # Residual squared = ||z - P z||^2 = ||z||^2 - ||Q^T z||^2  (since Q columns orthonormal)
    ZT = Z_seq @ Q                          # [L, 2]
    res2 = 1.0 - (ZT**2).sum(dim=1).clamp_min(0.0)  # each z is unit norm
    return res2.mean(), res2.max()

# ---------- (D) angle progression linearity (R^2) + monotonicity ----------
def angle_progression_stats(Z_seq):
    """
    Angle from start vs step index should grow roughly linearly if path hugs a great circle.
    Returns (R2, monotonic_fraction).
    """
    L = Z_seq.size(0)
    u0 = Z_seq[0]
    dots = (Z_seq @ u0).clamp(-1.0, 1.0)
    theta = torch.arccos(dots)              # [L]
    k = torch.arange(L, device=Z_seq.device, dtype=Z_seq.dtype)
    # Fit theta ~ a*k + b (least squares)
    A = torch.stack([k, torch.ones_like(k)], dim=1)  # [L,2]
    sol = torch.linalg.lstsq(A, theta).solution      # [2]
    theta_hat = A @ sol
    ss_res = ((theta - theta_hat)**2).sum()
    ss_tot = ((theta - theta.mean())**2).sum().clamp_min(1e-12)
    R2 = 1.0 - (ss_res / ss_tot)
    # monotonicity (exclude step 0)
    mono = (theta[1:] > theta[:-1]).float().mean() if L > 1 else torch.tensor(1.0, device=Z_seq.device)
    return float(R2.item()), float(mono.item())

# ---------- (E) top-level evaluator ----------
@torch.no_grad()
def trajectory_straightness_metrics(
    encoder,
    Phi,                    # [N, code_dim]
    H: int, W: int,
    L_seq: int = 16,
    stride: int = 1,
    num_per_axis: int = 128,
    gain: float = 1.0,
):
    """
    Samples row/col sequences, encodes them once, and computes straightness metrics:
      - mean plane deviation^2
      - max plane deviation^2
      - mean geodesic curvature (det(Gram) over triples)
      - angle progression R^2 and monotonicity
    Returns dict of aggregated means over sequences.
    """
    device = Phi.device
    seqs = sample_sequences(H, W, L=L_seq, stride=stride, num_per_axis=num_per_axis, device=device)  # [S,L]
    if seqs.numel() == 0:
        return {"num_seqs": 0}

    # Encode unique indices once
    flat = seqs.reshape(-1)
    uniq, inv = torch.unique(flat, return_inverse=True)
    Zuniq = encoder(Phi.index_select(0, uniq), gain)      # [U, d]
    Zuniq = F.normalize(Zuniq, dim=-1)
    Z_all = Zuniq.index_select(0, inv).view(seqs.size(0), L_seq, -1)  # [S, L, d]

    plane_mean_list, plane_max_list, det_list, R2_list, mono_list = [], [], [], [], []

    for s in range(Z_all.size(0)):
        Z_seq = Z_all[s]  # [L, d]
        pm, pM = plane_deviation_stats(Z_seq)
        detm   = mean_triple_det_along_sequence(Z_seq)
        R2, mono = angle_progression_stats(Z_seq)
        plane_mean_list.append(pm.item())
        plane_max_list.append(pM.item())
        det_list.append(detm.item())
        R2_list.append(R2)
        mono_list.append(mono)

    return {
        "num_seqs": int(Z_all.size(0)),
        "plane_dev_mean": float(torch.tensor(plane_mean_list).mean().item()),
        "plane_dev_max_mean": float(torch.tensor(plane_max_list).mean().item()),
        "triple_det_mean": float(torch.tensor(det_list).mean().item()),
        "angle_R2_mean": float(torch.tensor(R2_list).mean().item()),
        "angle_monotonic_frac_mean": float(torch.tensor(mono_list).mean().item()),
    }

def save_encoder(encoder, config, filename, encoders_dir="/home/jackking/cls/encoders"):
    """Save encoder weights and config to a file.
    
    Args:
        encoder: GridEncoder instance
        filename: Name of the file (e.g., 'my_encoder.pt')
        encoders_dir: Directory to save to (default: 'encoders')
    """
    os.makedirs(encoders_dir, exist_ok=True)
    
    checkpoint = {
        "config": config,
        "state_dict": encoder.state_dict(),
    }
    
    path = os.path.join(encoders_dir, filename)
    torch.save(checkpoint, path)
    print(f"Saved encoder to {path}")

def load_encoder(encoder_name, encoders_dir=None, device=None):
    encoders_dir = "/home/jackking/cls/encoders"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder_path = os.path.join(encoders_dir, encoder_name)
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    
    config = checkpoint["config"]
    sweep_params = checkpoint.get("sweep_params", {})
    training_info = checkpoint.get("training_info", {})
    lambdas = config["model_params"]["lambdas"]
    hidden_channels = config["model_params"]["hidden_channels"]
    num_hidden_layers = config["model_params"].get("num_hidden_layers", 1)
    kernel_size = config["model_params"].get("kernel_size", 3)
    hidden_dim = config["model_params"]["hidden_dim"]
    out_dim = config["model_params"]["out_dim"]
    nonlinearity = config["model_params"]["nonlinearity"]
    output_nonlinearity = config["model_params"]["output_nonlinearity"]
    
    # Create encoder
    # encoder = SphericalMLP(
    #     in_dim=in_dim,
    #     hidden=config["hidden"],
    #     out_dim=config["out_dim"],
    #     nonlinearity=config.get("nonlinearity", "gelu"),
    #     output_nonlinearity=config.get("output_nonlinearity", "tanh"),
    # ).to(device)

    encoder = GridEncoderCNN(lambdas, 
        hidden_channels=hidden_channels,
        out_dim=out_dim, 
        hidden_dim=hidden_dim, 
        nonlinearity=nonlinearity,    
        output_nonlinearity=output_nonlinearity,
        num_hidden_layers=num_hidden_layers,
        kernel_size=kernel_size
    ).to(device)

    encoder.load_state_dict(checkpoint["state_dict"])
    encoder.eval()

    return encoder, config


def sample_nonoverlapping_patches(
    H_full: int, 
    W_full: int, 
    Npos: int, 
    Nenv: int, 
    max_attempts_per_patch: int = 1000
) -> Tuple[List[int], List[int]]:
    """
    Sample Nenv non-overlapping square patches of size Npos x Npos from a grid of size H_full x W_full.
    Uses rejection sampling to place patches at arbitrary positions.
    
    Args:
        H_full: Height of the full grid
        W_full: Width of the full grid
        Npos: Size of each square patch
        Nenv: Number of patches to sample
        max_attempts_per_patch: Max attempts per patch before giving up
        
    Returns:
        y0s: List of top-left y coordinates for each patch
        x0s: List of top-left x coordinates for each patch
    """
    assert H_full >= Npos and W_full >= Npos, \
        f"Grid dims ({H_full}, {W_full}) must be >= Npos ({Npos})"
    
    def patches_overlap(y0_a: int, x0_a: int, y0_b: int, x0_b: int, size: int) -> bool:
        """Check if two square patches of given size overlap."""
        return not (y0_a + size <= y0_b or y0_b + size <= y0_a or
                    x0_a + size <= x0_b or x0_b + size <= x0_a)
    
    y0s = []
    x0s = []
    total_attempts = 0
    max_total_attempts = Nenv * max_attempts_per_patch
    
    while len(y0s) < Nenv and total_attempts < max_total_attempts:
        # Sample random corner
        y0 = torch.randint(0, H_full - Npos + 1, (1,)).item()
        x0 = torch.randint(0, W_full - Npos + 1, (1,)).item()
        
        # Check overlap with all existing patches
        overlaps = False
        for i in range(len(y0s)):
            if patches_overlap(y0, x0, y0s[i], x0s[i], Npos):
                overlaps = True
                break
        
        if not overlaps:
            y0s.append(y0)
            x0s.append(x0)
        
        total_attempts += 1
    
    if len(y0s) < Nenv:
        raise RuntimeError(
            f"Could only fit {len(y0s)} non-overlapping patches after {total_attempts} attempts. "
            f"Requested {Nenv} patches of size {Npos}x{Npos} in {H_full}x{W_full} grid."
        )
    
    print(f"Sampled {Nenv} non-overlapping {Npos}x{Npos} patches (took {total_attempts} attempts)")
    return y0s, x0s


def extract_patches_to_flat(
    Phi: torch.Tensor,
    y0s: List[int],
    x0s: List[int],
    Npos: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract patches from Phi and return flattened Phi values and their original coordinates.
    
    Args:
        Phi: [code_dim, H_full, W_full] tensor
        y0s: List of top-left y coordinates
        x0s: List of top-left x coordinates
        Npos: Size of each square patch
        device: Device to put tensors on
        
    Returns:
        Phi_flat: [Nenv * Npos * Npos, code_dim] tensor of Phi values
        X: [Nenv * Npos * Npos, 2] tensor of original (y, x) coordinates
    """
    Nenv = len(y0s)
    
    all_ys = []
    all_xs = []
    for i in range(Nenv):
        ys_patch, xs_patch = torch.meshgrid(
            torch.arange(y0s[i], y0s[i] + Npos),
            torch.arange(x0s[i], x0s[i] + Npos),
            indexing="ij"
        )
        all_ys.append(ys_patch.reshape(-1))
        all_xs.append(xs_patch.reshape(-1))
    
    all_ys = torch.cat(all_ys)  # [Nenv * Npos * Npos]
    all_xs = torch.cat(all_xs)  # [Nenv * Npos * Npos]
    
    # Extract Phi values at sampled positions
    Phi_flat = Phi[:, all_ys, all_xs].T.to(device)  # [Nenv * Npos * Npos, code_dim]
    
    # X coordinates are the ORIGINAL positions in the full grid
    X = torch.stack([all_ys.float(), all_xs.float()], dim=-1).to(device)  # [Nenv * Npos * Npos, 2]
    
    return Phi_flat, X


def train(Phi, config, save_every = True):
    lambdas = config["model_params"]["lambdas"]
    full_Npos = np.prod(lambdas)
    Npos = config["model_params"]["Npos"]
    Nenv = config["model_params"].get("Nenv", 1)  # Default to 1 (full grid behavior if Nenv not specified)
    num_layers = config["model_params"]["num_layers"]
    hidden_channels = config["model_params"]["hidden_channels"]
    kernel_size = config["model_params"]["kernel_size"]
    embed_dim = config["model_params"]["out_dim"]
    hidden_dim = config["model_params"]["hidden_dim"]
    num_hidden_layers = config["model_params"]["num_hidden_layers"]
    batch_size = config["training_params"]["batch_size"]
    lr = config["training_params"]["lr"]
    epochs = config["training_params"]["epochs"]
    centered = True
    output_nonlinearity = "tanh"

    # Gain annealing
    gain_start = config["training_params"]["gain_start"]
    gain_end = config["training_params"]["gain_end"]
    gain_up_epochs = config["training_params"]["gain_up_epochs"]
    gains = np.linspace(gain_start, gain_end, epochs)
    gains = np.concatenate([gains, np.ones(epochs - gain_up_epochs) * gains[-1]])

    # Uniformity loss annealing
    uniformity_lambda_start = config["training_params"]["uniformity_lambda_start"]
    uniformity_lambda_end = config["training_params"]["uniformity_lambda_end"]
    uniformity_lambda_scale_up_epochs = config["training_params"]["uniformity_lambda_scale_up_epochs"]
    uniformity_lambda_scales = np.linspace(uniformity_lambda_start, uniformity_lambda_end, uniformity_lambda_scale_up_epochs)
    uniformity_lambda_scales = np.concatenate([uniformity_lambda_scales, np.ones(epochs - uniformity_lambda_scale_up_epochs) * uniformity_lambda_scales[-1]])

    # Parameters for modulating target kernel used by CKA
    cka_alpha = config["training_params"]["cka_alpha"]
    cka_topk = config["training_params"]["cka_topk"]
    mod_loss_lambda = config["training_params"]["mod_loss_lambda"]

    T_triple_batch = 4096
    triple_stride = 3
    include_diagonals = False
    plane_lambda_start = 0
    plane_lambda_end = 0
    plane_lambda_scale_up_epochs = 1
    plane_lambda_scales = np.linspace(plane_lambda_start, plane_lambda_end, plane_lambda_scale_up_epochs)
    plane_lambda_scales = np.concatenate([plane_lambda_scales, np.ones(epochs - plane_lambda_scale_up_epochs) * plane_lambda_scales[-1]])

    local_lambda_start = 0
    local_lambda_end = 0
    local_lambda_scale_up_epochs = 1
    local_lambda_scales = np.linspace(local_lambda_start, local_lambda_end, local_lambda_scale_up_epochs)
    local_lambda_scales = np.concatenate([local_lambda_scales, np.ones(epochs - local_lambda_scale_up_epochs) * local_lambda_scales[-1]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Phi: [code_dim, H_full, W_full]
    code_dim, H_full, W_full = Phi.shape

    # Sample Nenv non-overlapping patches of size Npos x Npos
    y0s, x0s = sample_nonoverlapping_patches(H_full, W_full, Npos, Nenv)
    
    # Visualize patch placements
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, W_full - 0.5)
    ax.set_ylim(H_full - 0.5, -0.5)  # Invert y-axis to match image coordinates
    ax.set_aspect('equal')
    ax.set_title(f"Sampled {Nenv} patches ({Npos}x{Npos}) on {H_full}x{W_full} grid")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Draw grid outline
    ax.add_patch(plt.Rectangle((0, 0), W_full, H_full, fill=False, edgecolor='black', linewidth=2))
    
    # Draw each patch with a different color
    colors = plt.cm.tab10(np.linspace(0, 1, max(Nenv, 10)))
    for i, (y0, x0) in enumerate(zip(y0s, x0s)):
        color = colors[i % len(colors)]
        rect = plt.Rectangle((x0, y0), Npos, Npos, fill=True, facecolor=color, 
                              edgecolor='white', linewidth=1.5, alpha=0.6)
        ax.add_patch(rect)
        # Add patch number label at center
        ax.text(x0 + Npos/2, y0 + Npos/2, str(i), ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.show()
    
    # Extract patches and get flattened Phi values with original coordinates
    Phi_flat, X = extract_patches_to_flat(Phi, y0s, x0s, Npos, device)
    
    N = Phi_flat.shape[0]  # Nenv * Npos * Npos
    H, W = Npos, Npos  # Patch dimensions for triples calculation
    
    tau = estimate_tau_median(X, sample_pairs=min(50000, N * (N - 1) // 2))

    triples_all = build_grid_triples(H, W, stride=triple_stride, include_diagonals=include_diagonals, both_directions=True)
    print(f"Precomputed {triples_all.size(0)} curvature triples.")

    # encoder = SphericalMLP(in_dim=Phi_flat.size(1), hidden=hidden, out_dim=embed_dim, nonlinearity="gelu", output_nonlinearity=output_nonlinearity).to(device)
    encoder = GridEncoderCNN(lambdas, hidden_channels=hidden_channels, out_dim=embed_dim, hidden_dim=hidden_dim, nonlinearity="gelu", output_nonlinearity=output_nonlinearity, num_conv_layers=num_layers, kernel_size=kernel_size, num_hidden_layers=num_hidden_layers).to(device)

    optim = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Using device: {device} | N={N} | code_dim={Phi_flat.size(1)} | embed_dim={embed_dim}")
    print(f"RBF tau (median dist) = {tau:.4f}")

    viz_idx_y, viz_idx_x = y0s[0] + Npos // 2, x0s[0] + Npos // 2

    #grab an idx_y and idx_x that are not in any of the patches
    def in_any_patch(y, x):
        return any(y0 <= y < y0 + Npos and x0 <= x < x0 + Npos for y0, x0 in zip(y0s, x0s))
    
    rand_idx_y, rand_idx_x = np.random.randint(0, full_Npos), np.random.randint(0, full_Npos)
    while in_any_patch(rand_idx_y, rand_idx_x):
        rand_idx_y, rand_idx_x = np.random.randint(0, full_Npos), np.random.randint(0, full_Npos)

    sims = []
    triplet_accs = []
    nn_consistencies = []
    losses = []

    for ep in range(1, epochs + 1):
        gain = gains[ep - 1]
        tr_loss = train_epoch(
            encoder,
            Phi_flat,
            X,
            tau,
            optim,
            batch_size,
            triples_all,
            T_triple_batch,
            centered=centered,
            gain=gain,
            uniformity_lambda=uniformity_lambda_scales[ep - 1],
            lambda_plane=plane_lambda_scales[ep - 1],
            lambda_local=local_lambda_scales[ep - 1],
            cka_alpha=cka_alpha,
            cka_topk=cka_topk,
            mod_loss_lambda=mod_loss_lambda,
        )

        metrics = eval_full(
            encoder,
            Phi_flat,
            Xcoords=X,
            tau=tau,
            subset=min(2048, N),
            centered=True,         # same as training
            D_true=None,
            tau_for_rbf=tau,
            cka_alpha=cka_alpha,
            cka_topk=cka_topk,
        )

        print(
            f"Epoch {ep:02d} | train_align_loss={tr_loss:.4f} | "
            f"align_loss={metrics['align_loss']:.4f} | "
            f"local_loss={metrics['local_loss']:.6f} | "
            f"Pearson(sim)={metrics['pearson_sim']:.4f} | "
            f"Spearman(sim)={metrics['spearman_sim']:.4f} | "
            f"TripletAcc={metrics['triplet_acc']:.3f} | "
            f"NNconsistency={metrics['nn_consistency']:.3f}"
            + (f" | R2={metrics['R2_distproxy']:.3f} | RMSE={metrics['shepard_RMSE']:.3f}" if 'R2_distproxy' in metrics else "")
        )

        sims.append(metrics['pearson_sim'])
        triplet_accs.append(metrics['triplet_acc'])
        nn_consistencies.append(metrics['nn_consistency'])
        losses.append(tr_loss)

        # Only compute trajectory metrics if Phi_flat is a contiguous grid (N == H*W)
        if N == H * W:
            metrics_traj = trajectory_straightness_metrics(
                encoder,
                Phi_flat,            # [N, code_dim] on same device
                H=H, W=W,
                L_seq=16,            # try 8, 16, 32
                stride=triple_stride,
                num_per_axis=256,    # more sequences → tighter estimates
                gain=gain,
            )
            print(
                f"plane_dev_mean={metrics_traj['plane_dev_mean']:.3f} | "
                f"plane_dev_max_mean={metrics_traj['plane_dev_max_mean']:.3f} | "
                f"triple_det_mean={metrics_traj['triple_det_mean']:.5f} | "
                f"angle_R2_mean={metrics_traj['angle_R2_mean']:.3f} | "
                f"angle_monotonic_frac_mean={metrics_traj['angle_monotonic_frac_mean']:.3f}"
            )

        if save_every:
            save_encoder(encoder, config, f"encoder_{ep}.pt")
        
        def plot_metrics(encoded_Phi_grid, idx_y, idx_x):
            phix = encoded_Phi_grid[idx_y, idx_x, :]

            #vectorized version
            cosine_sims = np.sum(encoded_Phi_grid * phix, axis=-1) / np.sqrt(np.sum(encoded_Phi_grid**2, axis=-1) * np.sum(phix**2, axis=-1))
            plt.imshow(cosine_sims)
            
            # Add circle marker at the reference point
            plt.scatter(idx_x, idx_y, s=100, facecolors='none', edgecolors='red', linewidths=2, marker='o')

            plt.title(f"cosine_sim(E(phi({idx_y},{idx_x})), E(phi(x,y))), lambdas = {lambdas}")
            plt.xlabel("delta(x)")
            plt.ylabel("delta(y)")
            plt.colorbar(label="cosine sim")
            ax = plt.gca()
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.yaxis.get_offset_text().set_visible(False)
            plt.show()

            plt.plot(cosine_sims[idx_y])
            plt.show()

        if ep % 10 == 0:
            # Batch processing: encode the FULL Phi grid for visualization (not just sampled patches)
            Phi_full_flat = Phi.reshape(code_dim, full_Npos * full_Npos).T.to(device)  # [full_Npos*full_Npos, code_dim]
            batch_size = 1000  # adjust as needed according to GPU/CPU RAM constraints
            encoded_Phi_chunks = []
            with torch.no_grad():
                for i in range(0, Phi_full_flat.shape[0], batch_size):
                    encoded_chunk = encoder(Phi_full_flat[i:i+batch_size], gain=gain).detach().cpu()
                    encoded_Phi_chunks.append(encoded_chunk)
            encoded_Phi = torch.cat(encoded_Phi_chunks, dim=0)
            encoded_Phi_grid = encoded_Phi.reshape(full_Npos, full_Npos, embed_dim).to("cpu").numpy()

            plt.clf()  # Clear the current figure
            plt.plot(sims, label='pearson_sim')
            plt.plot(triplet_accs, label='triplet_acc')
            plt.plot(nn_consistencies, label='nn_consistency')
            plt.plot(losses, label='loss')
            plt.legend()
            plt.draw()
            plt.pause(0.001)
            plt.show()

            plot_metrics(encoded_Phi_grid, viz_idx_y, viz_idx_x)
            plot_metrics(encoded_Phi_grid, rand_idx_y, rand_idx_x)


    plt.plot(sims, label='pearson_sim')
    plt.plot(triplet_accs, label='triplet_acc')
    plt.plot(losses, label='loss')
    plt.plot(nn_consistencies, label='nn_consistency')
    plt.legend()
    plt.show()

    return encoder


if __name__ == "__main__":
    lambdas = [11,12,13]
    Ng = np.sum(np.square(lambdas))
    Npos = np.prod(lambdas)
    gbook = gen_gbook_2d(lambdas, Ng, Npos)
    Phi_np = smooth_gbook(gbook, lambdas, 0.25)
    Phi = torch.tensor(Phi_np, dtype=torch.float32)

    # lower learning rate, larger model.
    config = {
        "model_params": {
            "lambdas": lambdas,
            "hidden_dim": 512,
            "hidden_channels": 128,
            "num_layers": 3,
            "out_dim": 128,
            "num_hidden_layers": 2,
            "kernel_size": 5,
            "nonlinearity": "gelu",
            "output_nonlinearity": "tanh",
            "gain": 5,
            "Npos": 50,
            "Nenv": 200
        },
        "training_params": {
            "lr": 0.0001,
            "batch_size": 8192,
            "epochs": 200,
            "gain_start": 1,
            "gain_end": 5,
            "gain_up_epochs": 50,
            "uniformity_lambda_start": 0,
            "uniformity_lambda_end": 0.1,
            "uniformity_lambda_scale_up_epochs": 25,
            "cka_alpha": 1,
            "cka_topk": 20,
            "mod_loss_lambda": 0.75,
        },
        "hopfield_params": {
            "alpha": 0.85
        }
    }

    cka_topks = [10, 20, 40, 80]
    mod_loss_lambdas = [0.5, 0.75, 1.0, 1.5]
    out_dims = [64, 128, 256]

    iterator = product(cka_topks, mod_loss_lambdas, out_dims)
    for cka_topk, mod_loss_lambda, out_dim in iterator:
        config["training_params"]["cka_topk"] = cka_topk
        config["training_params"]["mod_loss_lambda"] = mod_loss_lambda
        config["model_params"]["out_dim"] = out_dim
        print(f"Training with cka_topk={cka_topk}, mod_loss_lambda={mod_loss_lambda}, out_dim={out_dim}")
        encoder = train(Phi, config, save_every = False)
        save_encoder(encoder, config, f"encoder_cka_topk={cka_topk}_mod_loss_lambda={mod_loss_lambda}_out_dim={out_dim}.pt")
        print(f"Saved encoder_cka_topk={cka_topk}_mod_loss_lambda={mod_loss_lambda}_out_dim={out_dim}.pt")
        print(f"--------------------------------")