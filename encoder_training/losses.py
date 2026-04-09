"""Loss functions for encoder training.

All operate on [B, B] kernel matrices (cosine similarity for predicted,
RBF for target) or [B, d] embeddings.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Double-centering: K_c = K - row_mean - col_mean + grand_mean."""
    return K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()


def upper_tri_vec(M: torch.Tensor) -> torch.Tensor:
    """Extract upper-triangular entries as a flat vector."""
    return M[torch.triu(torch.ones_like(M), diagonal=1).bool()]


# ---------------------------------------------------------------------------
# Core losses
# ---------------------------------------------------------------------------

def kernel_alignment_loss(
    K_pred: torch.Tensor,
    K_tgt: torch.Tensor,
    centered: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """CKA-style kernel alignment loss.  Returns 1 - alignment (minimize)."""
    if centered:
        Kp, Kt = _center_kernel(K_pred), _center_kernel(K_tgt)
    else:
        Kp = K_pred - K_pred.mean()
        Kt = K_tgt - K_tgt.mean()
    num = (Kp * Kt).sum()
    den = torch.sqrt(Kp.square().sum().clamp_min(eps)) * torch.sqrt(Kt.square().sum().clamp_min(eps))
    return 1.0 - num / den


def weighted_kernel_alignment_loss(
    K_pred: torch.Tensor,
    K_tgt: torch.Tensor,
    topk: int = 16,
    near_mult: float = 5.0,
    centered: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """CKA with up-weighted near-neighbor pairs."""
    B = K_pred.size(0)
    if centered:
        Kp, Kt = _center_kernel(K_pred), _center_kernel(K_tgt)
    else:
        Kp = K_pred - K_pred.mean()
        Kt = K_tgt - K_tgt.mean()

    W = torch.ones(B, B, device=K_pred.device, dtype=K_pred.dtype)
    W.fill_diagonal_(0.0)
    if topk is not None and topk < B - 1:
        Kt_masked = K_tgt.clone()
        Kt_masked.fill_diagonal_(-1e9)
        _, nn = torch.topk(Kt_masked, k=topk, dim=1)
        W.scatter_(1, nn, near_mult)
        W = torch.maximum(W, W.T)

    A = W * Kp
    Bm = W * Kt
    num = (A * Bm).sum()
    den = torch.sqrt(A.square().sum().clamp_min(eps)) * torch.sqrt(Bm.square().sum().clamp_min(eps))
    return 1.0 - num / den


def local_attract_far_repel_loss(
    K_pred: torch.Tensor,
    K_tgt: torch.Tensor,
    topk: int = 5,
    centered: bool = True,
    far_lambda: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Top-K local pairs get CKA alignment; remaining pairs get pushed apart."""
    B = K_pred.size(0)
    device = K_pred.device
    eye = torch.eye(B, device=device, dtype=torch.bool)

    K_tgt_noself = K_tgt.clone().masked_fill_(eye, -1.0)
    _, topk_idx = torch.topk(K_tgt_noself, k=min(topk, B - 1), dim=1)
    local_mask = torch.zeros(B, B, device=device, dtype=torch.bool)
    local_mask.scatter_(1, topk_idx, True)
    local_mask = (local_mask | local_mask.T) & ~eye
    far_mask = ~local_mask & ~eye

    # Local CKA
    eye_f = eye.float()
    K_pred_local = K_pred * local_mask.float() + eye_f * K_pred.diagonal()
    K_tgt_local = K_tgt * local_mask.float() + eye_f * K_tgt.diagonal()
    local_loss = kernel_alignment_loss(K_pred_local, K_tgt_local, centered=centered, eps=eps)

    # Far repulsion: push cosine similarity toward -1
    if far_mask.any():
        far_loss = ((K_pred[far_mask] + 1.0) / 2.0).mean()
    else:
        far_loss = K_pred.new_zeros(())

    return local_loss + far_lambda * far_loss


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Uniformity regularizer: encourages embeddings to spread on the sphere."""
    z = F.normalize(z, dim=-1)
    B = z.size(0)
    if B < 2:
        return z.new_zeros(())
    S = z @ z.t()
    dist2 = (2.0 - 2.0 * S).clamp_min(0.0)
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    dist2_off = dist2[mask]
    return torch.logsumexp(-t * dist2_off, dim=0) - torch.log(
        torch.tensor(dist2_off.numel(), device=z.device, dtype=z.dtype))


def coplanarity_loss_sphere(z_triples: torch.Tensor) -> torch.Tensor:
    """Encourage great-circle straightness for consecutive position triples.

    z_triples: (T, 3, d), assumed unit-norm.
    Loss = mean det(Gram matrix) = mean squared parallelepiped volume.
    """
    z = F.normalize(z_triples, dim=-1)
    u, v, w = z[:, 0], z[:, 1], z[:, 2]
    a = (u * v).sum(-1)
    b = (u * w).sum(-1)
    c = (v * w).sum(-1)
    return (1.0 + 2.0 * a * b * c - (a * a + b * b + c * c)).mean()
