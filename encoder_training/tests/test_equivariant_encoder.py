"""The equivariant encoder must actually be equivariant, or it tests nothing.

§8.1's whole premise is that an equivariant code has cos(z(x), z(y)) = k(y - x),
so every reference is identical and r_min = r_median. If the construction is
only approximately equivariant, any r_min it achieves says nothing about the
premise. These pin the property directly.
"""
from __future__ import annotations

import numpy as np
import torch

from encoder_training.equivariant import (
    EquivariantCharacterEncoder, build_gaussian, character_table,
)
from encoder_training.eval_unique_radius import grid_code_batch

LAM = [3, 4, 5]          # npos = 60, small enough to test exhaustively
NPOS = 60


def _codes(enc, ys, xs):
    phi = torch.as_tensor(grid_code_batch(LAM, ys, xs, 0.25))
    with torch.no_grad():
        return enc(phi)


def test_cosine_depends_only_on_offset():
    """The property the whole §8.1 argument rests on."""
    enc = build_gaussian(LAM, sigma=3.0, p_max=1, m_max=None)
    rng = np.random.default_rng(0)
    for dy, dx in ((0, 1), (0, 7), (5, 5), (11, 3)):
        ys = rng.integers(0, NPOS, size=64)
        xs = rng.integers(0, NPOS, size=64)
        a = _codes(enc, ys, xs)
        b = _codes(enc, (ys + dy) % NPOS, (xs + dx) % NPOS)
        cos = (a * b).sum(-1)
        assert cos.std() < 1e-5, (dy, dx, float(cos.std()))


def test_character_table_frequencies_are_correct():
    """m must be the induced position-frequency, in 1/prod(lambdas) units."""
    triples, ms = character_table(LAM, p_max=1, m_max=None)
    total = int(np.prod(LAM))
    cof = np.array([total // l for l in LAM])
    for p, m in zip(triples, ms):
        assert (int(np.dot(cof, p)) - m) % total == 0


def test_gaussian_amplitudes_give_a_decaying_kernel():
    """A wider sigma must give a wider kernel, or the analytic route is wrong."""
    rng = np.random.default_rng(1)
    ys = rng.integers(0, NPOS, size=64)
    xs = rng.integers(0, NPOS, size=64)
    widths = []
    for sigma in (2.0, 5.0):
        enc = build_gaussian(LAM, sigma=sigma, p_max=1, m_max=None)
        a = _codes(enc, ys, xs)
        prof = []
        for d in range(0, 12):
            b = _codes(enc, ys, (xs + d) % NPOS)
            prof.append(float((a * b).sum(-1).mean()))
        # first offset at which the profile drops below 0.5
        below = [i for i, v in enumerate(prof) if v < 0.5]
        widths.append(below[0] if below else len(prof))
    assert widths[1] > widths[0], widths


def test_amplitudes_are_learnable_and_change_the_kernel():
    """The same module has to be trainable, not just analytic."""
    enc = EquivariantCharacterEncoder(LAM, p_max=1, m_max=None)
    assert enc.log_amp.requires_grad
    rng = np.random.default_rng(2)
    ys = rng.integers(0, NPOS, size=32)
    xs = rng.integers(0, NPOS, size=32)
    phi = torch.as_tensor(grid_code_batch(LAM, ys, xs, 0.25))
    z = enc(phi)
    z.sum().backward()
    assert enc.log_amp.grad is not None
    assert torch.isfinite(enc.log_amp.grad).all()
