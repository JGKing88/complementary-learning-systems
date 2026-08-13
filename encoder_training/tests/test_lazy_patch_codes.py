"""The lazy patch builder must reproduce the full-codebook path exactly.

``build_full_grid`` materialises ``(Ng, Npos, Npos)`` float64 -- 10.2 GB at
lambdas (11, 12, 13) -- for a codebook that is one-hot per module and
trivially recomputable. ``eval_unique_radius.grid_code_batch`` already builds it
a batch at a time; ``build_patch_codes`` uses that for training too, which is
what lets several runs share a node.

Exact agreement is the whole point: if the two paths differ even slightly, every
number in the log stops being comparable to the runs that came before.
"""
from __future__ import annotations

import numpy as np
import torch

from encoder_training.data import build_full_grid, build_patch_codes


def test_lazy_matches_full_codebook():
    lambdas = [3, 4, 5]                      # Npos = 60, small enough to build
    Phi_full, npos = build_full_grid(lambdas, fwhm_ratio=0.25)
    y0s, x0s, sizes = [2, 30], [7, 41], [11, 9]

    from encoder_training.data import extract_patches
    dev = torch.device("cpu")
    phi_a, coords_a, env_a = extract_patches(Phi_full, y0s, x0s, sizes, dev)
    phi_b, coords_b, env_b = build_patch_codes(lambdas, y0s, x0s, sizes, dev,
                                               fwhm_ratio=0.25)

    assert torch.equal(env_a, env_b)
    assert torch.equal(coords_a, coords_b)
    assert phi_a.shape == phi_b.shape
    np.testing.assert_allclose(phi_a.numpy(), phi_b.numpy(), atol=1e-6)


def test_lazy_unsmoothed_matches_too():
    """fwhm_ratio=0 takes a different branch in both builders."""
    lambdas = [3, 4, 5]
    Phi_full, _ = build_full_grid(lambdas, fwhm_ratio=0.0)
    from encoder_training.data import extract_patches
    dev = torch.device("cpu")
    phi_a, _, _ = extract_patches(Phi_full, [1], [5], [8], dev)
    phi_b, _, _ = build_patch_codes(lambdas, [1], [5], [8], dev, fwhm_ratio=0.0)
    np.testing.assert_allclose(phi_a.numpy(), phi_b.numpy(), atol=1e-6)
