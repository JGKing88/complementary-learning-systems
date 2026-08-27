"""Saturation has to be read before normalisation, or it reads the wrong thing.

`forward` is `z = tanh(g * net(x))` then `z = z / ||z||`. The divide is a scalar,
so it preserves a saturated +-1 pattern exactly (as +-1/sqrt(D)) -- but it also
rescales an *un*saturated code up to the same magnitudes, which is how the
original `frac_saturated` reported 0.34 for encoders whose true tanh saturation
is 0.000.
"""
from __future__ import annotations

import torch


def _post(pre):
    return torch.nn.functional.normalize(pre, p=2, dim=-1)


def test_normalisation_preserves_a_binary_pattern():
    """+-1 in, +-1/sqrt(D) out: the sign structure survives the divide."""
    D = 64
    pre = torch.sign(torch.randn(32, D))
    post = _post(pre)
    assert torch.allclose(post.abs(), torch.full_like(post, D ** -0.5),
                          atol=1e-6)
    assert torch.equal(torch.sign(post), torch.sign(pre))


def test_post_norm_threshold_cannot_tell_saturated_from_unsaturated():
    """The failure the old diagnostic had, pinned so it cannot come back.

    A code scaled down by 100x is not saturated at all, yet after
    normalisation it is indistinguishable from the saturated one.
    """
    D = 64
    pre_sat = torch.sign(torch.randn(32, D))
    pre_unsat = pre_sat * 0.01                      # nowhere near tanh's rails
    a = (_post(pre_sat).abs() * D ** 0.5 > 0.95).float().mean()
    b = (_post(pre_unsat).abs() * D ** 0.5 > 0.95).float().mean()
    assert torch.allclose(a, b)                     # post-norm: identical
    # pre-norm the two are opposites, which is the reading that means something
    assert (pre_sat.abs() > 0.95).float().mean() == 1.0
    assert (pre_unsat.abs() > 0.95).float().mean() == 0.0


def test_gain_cancels_when_tanh_is_linear():
    """Why the gain ramp is close to inert: normalisation cancels a scalar.

    For pre-activations small enough that tanh(u) ~ u, `normalize(tanh(g*u))`
    is independent of g. Measured medians in this campaign are |g*net(x)| ~
    0.05, which is squarely in that regime.
    """
    u = torch.randn(16, 32) * 0.01
    z1 = _post(torch.tanh(1.0 * u))
    z5 = _post(torch.tanh(5.0 * u))
    assert float((z1 * z5).sum(-1).min()) > 0.9999
