"""The spread terms and the graded target, checked where they could go wrong.

These exist because every one of them is a *weight on a term that is always
on*: a sign error or a scale error does not crash, it just quietly trains a
different objective for eight hours.
"""
from __future__ import annotations

import torch

from encoder_training.losses import (
    mse_attract_repel, uniformity_loss, vicreg_terms,
)


def _unit(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_uniformity_mask_changes_the_value():
    """A mask that drops pairs must actually drop them.

    The near pairs are the ones with the smallest distance and logsumexp is
    dominated by the smallest distance, so masking them is not a rounding
    difference -- if the two agree, the mask is not being applied.
    """
    z = _unit(64, 16)
    z[1] = z[0]                                   # a collapsed near pair
    mask = torch.ones(64, 64, dtype=torch.bool)
    mask[0, 1] = mask[1, 0] = False
    full = uniformity_loss(z, t=2.0)
    masked = uniformity_loss(z, t=2.0, pair_mask=mask)
    assert masked < full - 1e-6


def test_uniformity_rewards_spread():
    """Spread-out codes must score lower than collapsed ones."""
    d = 32
    spread = _unit(64, d)
    collapsed = _unit(64, d) * 0.01 + _unit(1, d, seed=1)
    collapsed = torch.nn.functional.normalize(collapsed, dim=-1)
    assert uniformity_loss(spread) < uniformity_loss(collapsed)


def test_vicreg_var_is_zero_only_for_maximal_spread():
    """gamma=1 is the ceiling for unit-norm rows, not an arbitrary constant.

    Rows are L2-normalised, so E[z_i^2] = 1/D and the rescaled per-coordinate
    variance cannot exceed 1. A sign-flipped constant code attains it; anything
    less structured does not.
    """
    d = 16
    signs = torch.where(torch.arange(64)[:, None] % 2 == 0, 1.0, -1.0)
    ideal = torch.nn.functional.normalize(signs.expand(64, d).contiguous(), dim=-1)
    var_ideal, _ = vicreg_terms(ideal)
    var_random, _ = vicreg_terms(_unit(64, d))
    assert var_ideal < 1e-3
    assert var_random > var_ideal


def test_vicreg_cov_penalises_correlated_coordinates():
    z = _unit(128, 8)
    dup = torch.nn.functional.normalize(
        torch.cat([z[:, :4], z[:, :4]], dim=1), dim=-1)   # coords repeated
    _, cov_dup = vicreg_terms(dup)
    _, cov_free = vicreg_terms(z)
    assert cov_dup > cov_free


def test_graded_target_reduces_to_binary_at_the_limit():
    """target=None and an explicit 1/0 target must give the same loss."""
    K = (_unit(32, 8) @ _unit(32, 8).T).clamp(-1, 1)
    near = torch.zeros(32, 32, dtype=torch.bool)
    near[:8, :8] = True
    binary = mse_attract_repel(K, near)
    explicit = mse_attract_repel(K, near, target=near.float())
    assert torch.allclose(binary, explicit)


def test_graded_target_is_satisfied_by_the_matching_decay():
    """A code whose similarity already equals the target scores ~0.

    Guards the sign and the argument order: with the target and the prediction
    swapped, or the Gaussian inverted, this is the assertion that fails.
    """
    dist = torch.cdist(torch.arange(64).float()[:, None],
                       torch.arange(64).float()[:, None])
    sigma = 10.0
    target = torch.exp(-dist.square() / (2 * sigma ** 2))
    near = dist < sigma
    loss = mse_attract_repel(target.clamp(-1, 1), near, target=target)
    assert loss.item() < 1e-10
