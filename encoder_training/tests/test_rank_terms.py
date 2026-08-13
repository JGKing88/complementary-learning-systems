"""The rank statistics, on codes whose rank is known by construction.

These back the central claim of §4: withholding the cross-environment pairs
collapses the code's effective dimensionality, and the rank terms are what
push it back. If ``participation_ratio`` mis-scales, the diagnostic that
claim rests on is wrong, and if ``coding_rate_loss`` has its sign flipped the
runs optimise *for* collapse.
"""
from __future__ import annotations

import torch

from encoder_training.losses import coding_rate_loss, participation_ratio


def _unit(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_participation_ratio_counts_the_directions_used():
    """Rank-k Gaussian codes should read back roughly k, not 1 and not d."""
    d, n = 64, 4096
    g = torch.Generator().manual_seed(0)
    for k in (1, 4, 16):
        basis = torch.linalg.qr(torch.randn(d, d, generator=g))[0][:, :k]
        z = torch.nn.functional.normalize(
            torch.randn(n, k, generator=g) @ basis.T, dim=-1)
        pr = float(participation_ratio(z))
        assert 0.5 * k <= pr <= 2 * k + 1, f"k={k} gave pr={pr}"


def test_participation_ratio_is_near_out_dim_for_isotropic_codes():
    d = 64
    pr = float(participation_ratio(_unit(8192, d)))
    assert pr > 0.8 * d


def test_coding_rate_prefers_spread_to_collapse():
    """Lower loss for the isotropic code -- the sign that decides the runs."""
    d = 32
    spread = _unit(2048, d)
    g = torch.Generator().manual_seed(1)
    direction = torch.nn.functional.normalize(torch.randn(1, d, generator=g), dim=-1)
    collapsed = torch.nn.functional.normalize(
        direction + 0.01 * torch.randn(2048, d, generator=g), dim=-1)
    assert coding_rate_loss(spread) < coding_rate_loss(collapsed)


def test_coding_rate_gradient_raises_the_participation_ratio():
    """Optimising the term alone must move the statistic it targets.

    The sign test above passes for a term that is merely *correlated* with
    rank; this is the one that says descending on it actually un-collapses a
    code. Adam rather than SGD because the loss carries a 1/D and the raw
    gradient is small enough that a plain step moves the ratio by 1e-5.
    """
    d = 32
    g = torch.Generator().manual_seed(2)
    raw = torch.randn(512, 4, generator=g) @ torch.randn(4, d, generator=g)
    z = torch.nn.Parameter(raw)

    def pr_now() -> float:
        with torch.no_grad():
            return float(participation_ratio(
                torch.nn.functional.normalize(z, dim=-1)))

    before = pr_now()
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        coding_rate_loss(torch.nn.functional.normalize(z, dim=-1)).backward()
        opt.step()
    after = pr_now()
    assert after > before + 1.0, f"{before:.2f} -> {after:.2f}"
