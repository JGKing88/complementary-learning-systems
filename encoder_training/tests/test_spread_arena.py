"""The out-of-brief arena-spread diagnostic must touch the spread terms only.

`spread_arena_frac` exists to size one number (§5.6k): what the aliases living
outside every training patch would cost if the loss could see them. That answer
is only interpretable if the extra positions reach the spread terms and nothing
else -- if they leaked into attract/repel the run would be a different
experiment wearing the same name.
"""
from __future__ import annotations

import numpy as np
import torch

from encoder_training.eval_unique_radius import grid_code_batch
from encoder_training.losses import coding_rate_loss, uniformity_loss


def test_arena_codes_match_the_training_builder():
    """Extra positions must be built the same way as training patches."""
    from encoder_training.data import build_patch_codes

    lambdas = [3, 4, 5]
    dev = torch.device("cpu")
    # One 3x3 patch at (2, 7): build it both ways and compare.
    phi_patch, coords, _ = build_patch_codes(lambdas, [2], [7], [3], dev,
                                             fwhm_ratio=0.25)
    ys = coords[:, 0].numpy()
    xs = coords[:, 1].numpy()
    # Argument order is (first spatial axis, second) despite the parameter
    # names -- build_patch_codes passes (ys, xs) and train.py must match. This
    # test failed on the swap when it was written, which is why it exists.
    phi_lazy = torch.as_tensor(grid_code_batch(lambdas, ys, xs, 0.25))
    assert torch.allclose(phi_patch.cpu().float(), phi_lazy.float(), atol=1e-6)


def test_spread_terms_see_the_extra_rows():
    """Concatenating extra rows must change the spread terms, not silently pass."""
    torch.manual_seed(0)
    zb = torch.nn.functional.normalize(torch.randn(64, 32), dim=-1)
    extra = torch.nn.functional.normalize(torch.randn(16, 32), dim=-1)
    z_spread = torch.cat([zb, extra], dim=0)

    assert not np.isclose(float(uniformity_loss(zb)),
                          float(uniformity_loss(z_spread)))
    assert not np.isclose(float(coding_rate_loss(zb)),
                          float(coding_rate_loss(z_spread)))


def test_pair_masks_stay_batch_shaped():
    """The pair terms are indexed by the batch, so extra rows must not reach them.

    This is the property that keeps the diagnostic honest: a B+E row block fed
    to a B x B mask would raise, and silently truncating instead would make the
    extra positions into training pairs.
    """
    B, E, D = 32, 8, 16
    zb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    z_spread = torch.cat(
        [zb, torch.nn.functional.normalize(torch.randn(E, D), dim=-1)], dim=0)
    near = torch.zeros(B, B, dtype=torch.bool)

    K_pred = (zb @ zb.T)
    assert K_pred.shape == near.shape           # pair term uses zb alone
    assert z_spread.shape[0] == B + E           # spread term sees more
    try:
        uniformity_loss(z_spread, pair_mask=near)
    except RuntimeError:
        return                                   # shape mismatch, as intended
    raise AssertionError("a batch-shaped pair mask must not accept extra rows")
