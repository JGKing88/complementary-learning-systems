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


def uniformity_loss(z: torch.Tensor, t: float = 2.0,
                    pair_mask: torch.Tensor | None = None) -> torch.Tensor:
    """logsumexp(-t * ||zi - zj||^2) — push embeddings apart on the sphere.

    ``pair_mask`` restricts the term to a subset of the off-diagonal pairs.
    Left at None the term is *indiscriminate*: it never asks where a pair came
    from, which is what makes it the only spread term available when the
    cross-environment pairs are withheld from the repel term. The cost is that
    ``logsumexp`` is dominated by the pair with the smallest distance, and the
    smallest distances belong to exactly the pairs ``attract`` is holding at
    cosine 1 — pass a mask that drops them to separate the two effects.
    """
    z = F.normalize(z, dim=-1)
    B = z.size(0)
    if B < 2:
        return z.new_zeros(())
    S = z @ z.t()
    dist2 = (2.0 - 2.0 * S).clamp_min(0.0)
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    if pair_mask is not None:
        mask = mask & pair_mask
    n = mask.sum()
    if n == 0:
        return z.new_zeros(())
    # Masked-fill rather than `dist2[mask]`. The gather compacts ~67M elements
    # at B=8192 and, with the default mask being just "not the diagonal", it
    # copies essentially the whole matrix to drop 8192 entries -- which cost
    # 2.7x the step time and would have truncated every uniformity run at the
    # wall clock. Filling with -inf contributes exp(-inf)=0 to the sum instead.
    neg = (-t * dist2).masked_fill(~mask, float("-inf"))
    return torch.logsumexp(neg.reshape(-1), dim=0) - torch.log(n.to(z.dtype))


def vicreg_terms(z: torch.Tensor, gamma: float = 1.0,
                 eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg's variance and covariance regularizers, as (var_loss, cov_loss).

    Spread terms that are *pair-free*: both are statistics of the batch's
    second moment and neither looks at any individual pair, so unlike
    ``uniformity_loss`` neither can concentrate its gradient on the closest
    pair. What they prevent is dimensional collapse — a code that spends its
    1024 outputs on a handful of effective directions can only distinguish a
    handful of places.

    Being pair-free does not, however, keep them off the neighbourhood. At
    ``var_lambda=1, cov_lambda=0.1`` the profile sits at r_median 0 for the
    first few hundred epochs before recovering. It raises the strength at which
    the damage starts, and nothing more. Measured alongside
    ``coding_rate_loss``, which reaches a lower alias ceiling at a strength that
    never disturbs the decay — prefer that one.

    ``z`` is unit-norm, so a coordinate's natural scale is ``1/sqrt(D)``, not 1;
    it is rescaled by ``sqrt(D)`` here so that ``gamma=1`` asks for "each
    coordinate as variable as an isotropic unit vector's coordinate" rather
    than for something 32x out of reach.
    """
    B, D = z.shape
    if B < 2:
        zero = z.new_zeros(())
        return zero, zero
    zs = z * (D ** 0.5)
    zc = zs - zs.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()
    cov = (zc.T @ zc) / (B - 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return var_loss, off_diag / D


def participation_ratio(z: torch.Tensor) -> torch.Tensor:
    """Effective number of dimensions the batch's codes occupy.

    ``(tr C)^2 / ||C||_F^2`` for the covariance C — the participation ratio of
    its eigenvalues, 1 for a rank-one code and ``out_dim`` for an isotropic one.
    Computed from two traces rather than an eigendecomposition, so it is cheap
    enough to log every epoch.

    Worth logging because it is the quantity that separates the regimes:
    withholding the cross-environment pairs takes the trained code from ~202
    effective dimensions to 24–59 out of 1024, and the unique radius follows it.
    """
    B, D = z.shape
    if B < 2:
        return z.new_zeros(())
    zc = z - z.mean(dim=0, keepdim=True)
    C = (zc.T @ zc) / (B - 1)
    return C.diagonal().sum().square() / C.square().sum().clamp_min(1e-20)


def coding_rate_loss(z: torch.Tensor, eps: float = 0.5) -> torch.Tensor:
    """Negative log-det coding rate: ``-1/(2D) logdet(I + D/(B eps^2) Z^T Z)``.

    The MCR^2 rate term. Like the VICReg pair it is a statistic of the batch's
    second moment and never looks at an individual pair; unlike the VICReg pair
    it rewards *spectrum* rather than per-coordinate variance, since logdet is
    maximised by an even eigenvalue distribution.

    What it is for, empirically: it is the one term measured to move the alias
    ceiling without touching the decay width. At ``rate_lambda=0.3`` it took the
    ceiling from the binary baseline's 0.946 to 0.907 — the lowest reached under
    ``exclude_cross_env_pairs`` — with decay50 unchanged at 22.5, and r_min 4.5
    to 9.

    Being pair-free does NOT keep it off the neighbourhood, which I first
    assumed and then measured otherwise: at ``rate_lambda=3`` the profile spends
    hundreds of epochs with r_median 0. Pair-free only raises the strength at
    which the damage starts. Strength is the whole thing; 0.3 is the tested
    value.

    Normalised by D so the scale does not move when ``out_dim`` does.
    """
    B, D = z.shape
    if B < 2:
        return z.new_zeros(())
    scale = D / (B * eps * eps)
    gram = torch.eye(D, device=z.device, dtype=z.dtype) + scale * (z.T @ z)
    return -torch.linalg.cholesky(gram).diagonal().log().sum() / D


def mse_attract_repel(
    K_pred: torch.Tensor,
    near_mask: torch.Tensor,
    attract_lambda: float = 2.0,
    repel_weight: float = 5.0,
    far_mask: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
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
        target: optional [B, B] per-pair target similarity, replacing the
            binary 1-on-near / 0-on-far. The mask split still selects which
            pairs each weight applies to, so ``attract_lambda`` and
            ``repel_weight`` keep their meaning; only what each pair is asked
            for changes. A distance-graded target is what turns the flat "cosine
            1 out to the radius" plateau into a curve that actually decreases,
            which is what the unique radius measures.

    Both terms are means over their own pair sets, so the attract:repel balance
    does not shift when one set grows or shrinks.
    """
    B = K_pred.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=K_pred.device)

    near = near_mask & ~eye
    far = (~near_mask & ~eye) if far_mask is None else (far_mask & ~eye)

    if near.any():
        tgt = 1.0 if target is None else target[near]
        attract = ((K_pred[near] - tgt) ** 2).mean()
    else:
        attract = K_pred.new_zeros(())
    if far.any():
        tgt = 0.0 if target is None else target[far]
        repel = ((K_pred[far] - tgt) ** 2).mean()
    else:
        repel = K_pred.new_zeros(())
    return attract_lambda * attract + repel_weight * repel
