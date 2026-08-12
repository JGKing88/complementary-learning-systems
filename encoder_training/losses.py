"""Loss functions for binary-method encoder training.

The core method is `mse_attract_repel`:
- "near" pairs → cos sim 1
- "far" pairs (within the same batch, outside near mask) → cos sim 0

`cka_loss` and `uniformity_loss` are kept as optional components.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Double-center a kernel: K - row_mean - col_mean + grand_mean."""
    return K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()


def cka_loss(
    K_pred: torch.Tensor,
    K_tgt: torch.Tensor,
    centered: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """1 - CKA(K_pred, K_tgt). Centered double-centering by default."""
    if centered:
        Kp, Kt = _center_kernel(K_pred), _center_kernel(K_tgt)
    else:
        Kp = K_pred - K_pred.mean()
        Kt = K_tgt - K_tgt.mean()
    num = (Kp * Kt).sum()
    den = torch.sqrt(Kp.square().sum().clamp_min(eps)) * torch.sqrt(
        Kt.square().sum().clamp_min(eps))
    return 1.0 - num / den


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """logsumexp(-t * ||zi - zj||^2) — push embeddings apart on the sphere."""
    z = F.normalize(z, dim=-1)
    B = z.size(0)
    if B < 2:
        return z.new_zeros(())
    S = z @ z.t()
    dist2 = (2.0 - 2.0 * S).clamp_min(0.0)
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    d = dist2[mask]
    return torch.logsumexp(-t * d, dim=0) - torch.log(
        torch.tensor(d.numel(), device=z.device, dtype=z.dtype))


def mse_attract_repel(
    K_pred: torch.Tensor,
    near_mask: torch.Tensor,
    attract_lambda: float = 2.0,
    repel_weight: float = 5.0,
    far_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary attract-repel MSE loss.

    Args:
        K_pred: [B, B] cosine similarity (must be clamped to [-1, 1]).
        near_mask: [B, B] boolean. True entries pull to 1; False (off-diagonal)
                   entries push to 0. The diagonal is excluded from both.
        attract_lambda: weight on near-pair MSE.
        repel_weight: weight on far-pair MSE.
        far_mask: optional [B, B] boolean naming exactly which pairs to repel.
            Defaults to "everything not near", which in a mixed batch includes
            every cross-environment pair — that is where cross-environment
            repulsion comes from, rather than from any dedicated term. Pass a
            narrower mask to withhold it.

    Both terms are means over their own pair sets, so the attract:repel balance
    does not shift when one set grows or shrinks.
    """
    B = K_pred.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=K_pred.device)

    near = near_mask & ~eye
    far = (~near_mask & ~eye) if far_mask is None else (far_mask & ~eye)

    if near.any():
        attract = ((K_pred[near] - 1.0) ** 2).mean()
    else:
        attract = K_pred.new_zeros(())
    if far.any():
        repel = (K_pred[far] ** 2).mean()
    else:
        repel = K_pred.new_zeros(())
    return attract_lambda * attract + repel_weight * repel
