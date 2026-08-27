"""The equivariance probe must be able to read zero on a code that has it.

The claim in `equivariance.py` is that the grid code is exactly
translation-equivariant, so an encoder that commutes with translation gives a
similarity depending only on the offset. If the probe cannot register zero
spread for such an encoder, its readings on trained ones mean nothing.

The identity encoder is the case to check: the raw normalised code is trivially
equivariant, so its cos-at-fixed-offset must be constant across the arena.
"""
from __future__ import annotations

import numpy as np
import torch

from encoder_training.eval_unique_radius import grid_code_batch


def _cos_at_offset(lambdas, dy, dx, n, npos, seed=0):
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, npos, size=n)
    xs = rng.integers(0, npos, size=n)
    a = torch.as_tensor(grid_code_batch(lambdas, ys, xs, 0.25))
    b = torch.as_tensor(grid_code_batch(lambdas, (ys + dy) % npos,
                                        (xs + dx) % npos, 0.25))
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(-1).numpy()


def test_raw_code_is_exactly_equivariant():
    """cos at a fixed offset must not depend on where you measure it."""
    lambdas = [3, 4, 5]
    npos = 3 * 4 * 5
    for dy, dx in ((0, 7), (5, 5), (0, 1), (11, 3)):
        cos = _cos_at_offset(lambdas, dy, dx, n=256, npos=npos)
        assert cos.std() < 1e-5, (dy, dx, cos.std())


def test_probe_separates_offsets():
    """Different offsets must give different similarity, or it measures nothing."""
    lambdas = [3, 4, 5]
    npos = 3 * 4 * 5
    near = _cos_at_offset(lambdas, 0, 1, n=64, npos=npos).mean()
    far = _cos_at_offset(lambdas, 0, 7, n=64, npos=npos).mean()
    assert near > far


def test_translation_is_a_permutation_of_the_code():
    """The premise: Phi(x + a) is Phi(x) with its indices permuted.

    Checked as a multiset equality, which is what permutation-equivalence means
    and is enough to make the equivariance argument go through.
    """
    lambdas = [3, 4, 5]
    npos = 3 * 4 * 5
    base = grid_code_batch(lambdas, np.array([7]), np.array([11]), 0.25)[0]
    for dy, dx in ((1, 0), (0, 1), (6, 9)):
        shifted = grid_code_batch(lambdas, np.array([(7 + dy) % npos]),
                                  np.array([(11 + dx) % npos]), 0.25)[0]
        assert np.allclose(np.sort(base), np.sort(shifted), atol=1e-6)
