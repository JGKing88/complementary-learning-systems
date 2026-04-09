"""Evaluation metrics for encoder quality."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr

from .losses import kernel_alignment_loss, upper_tri_vec
from .data import rbf_kernel_batch


@torch.no_grad()
def eval_encoder(
    encoder,
    Phi: torch.Tensor,
    Xcoords: torch.Tensor,
    tau: float,
    subset: int = 2048,
    centered: bool = True,
    gain: float = 1.0,
    cka_alpha: float = 1.0,
    cka_topk: int | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics on a random subset.

    Returns dict with: align_loss, pearson_sim, spearman_sim, triplet_acc, nn_consistency.
    """
    device = next(encoder.parameters()).device
    N = Phi.size(0)
    M = min(subset, N)
    idx = torch.randperm(N, device=device)[:M]

    Z = encoder(Phi[idx], gain)
    Kp = (Z @ Z.T).clamp(-1, 1)
    Kt = rbf_kernel_batch(Xcoords, idx, tau)

    # Optionally modulate target kernel (match training)
    Kt_mod = Kt
    if (cka_alpha is not None and cka_alpha != 1.0) or (cka_topk is not None):
        B = Kt.size(0)
        eye = torch.eye(B, device=Kt.device, dtype=Kt.dtype)
        Kt_mod = Kt.clamp(0, 1)
        if cka_alpha is not None and cka_alpha != 1.0:
            Kt_mod = Kt_mod.pow(cka_alpha)
        if cka_topk is not None and cka_topk < B - 1:
            _, idx_top = torch.topk(Kt, k=min(cka_topk, B - 1), dim=1)
            mask = torch.zeros_like(Kt, dtype=torch.bool)
            mask.scatter_(1, idx_top, True)
            mask = (mask | mask.T) & ~eye.bool()
            Kt_mod = Kt_mod * mask.float() + eye

    align_loss = kernel_alignment_loss(Kp, Kt_mod, centered=centered).item()

    # Pearson / Spearman on upper-triangular entries
    s_pred = upper_tri_vec(Kp).cpu().numpy()
    s_tgt = upper_tri_vec(Kt).cpu().numpy()
    pe, _ = pearsonr(s_pred, s_tgt)
    sp, _ = spearmanr(s_pred, s_tgt)

    # Triplet accuracy
    T = min(200_000, M * 200)
    i = torch.randint(0, M, (T,), device=device)
    a = torch.randint(0, M, (T,), device=device)
    b = torch.randint(0, M, (T,), device=device)
    tgt_cmp = Kt[i, a] > Kt[i, b]
    pred_cmp = Kp[i, a] > Kp[i, b]
    valid = (a != b) & (i != a) & (i != b) & (Kt[i, a] != Kt[i, b])
    triplet_acc = float((pred_cmp[valid] == tgt_cmp[valid]).float().mean().item()) if valid.any() else float("nan")

    # Nearest-neighbor consistency
    Kt_m = Kt - torch.eye(M, device=device) * 1e9
    Kp_m = Kp - torch.eye(M, device=device) * 1e9
    nn_acc = float((Kt_m.argmax(1) == Kp_m.argmax(1)).float().mean().item())

    return {
        "align_loss": align_loss,
        "pearson_sim": float(pe),
        "spearman_sim": float(sp),
        "triplet_acc": triplet_acc,
        "nn_consistency": nn_acc,
    }
