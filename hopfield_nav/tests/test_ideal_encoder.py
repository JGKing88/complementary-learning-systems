"""``analysis.hopfield_probe.ideal_encoder``: the grid-code -> RFF module.

Pinned: the module, fed real grid codes (``encode.grid_codes``), equals the
closed form ``cos/sin(2 pi n . p / 1716) / sqrt(512)`` on random scaffold
positions, rows are unit norm, the CRT harmonics reconstruct ``n``, and the
probe loader resolves an ``ideal:`` spec. A few hundred positions only.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from analysis.hopfield_probe.encode import Field, grid_codes
from analysis.hopfield_probe.harness import load_probe_encoder
from analysis.hopfield_probe.ideal_encoder import (
    IdealEncoder, crt_harmonics, draw_frequencies,
)

LAMBDAS = [11, 12, 13]
NPOS = 1716


def test_crt_harmonics_are_6n_minus_n_7n_and_reconstruct_n():
    n = np.arange(-858, 859)
    j = crt_harmonics(n, LAMBDAS)
    assert np.array_equal(j[0], np.mod(6 * n, 11))
    assert np.array_equal(j[1], np.mod(-n, 12))
    assert np.array_equal(j[2], np.mod(7 * n, 13))
    assert np.array_equal(np.mod(156 * j[0] + 143 * j[1] + 132 * j[2], NPOS),
                          np.mod(n, NPOS))


def test_frequencies_are_reduced_to_the_half_open_torus():
    for r in (0.3, 2, 16):
        n = draw_frequencies(r, 4096, 0, NPOS)
        assert n.min() > -858 and n.max() <= 858


@pytest.mark.parametrize("r", [2, 4, 8, 16])
@pytest.mark.parametrize("fwhm", [0.25, 0.0])
def test_module_equals_closed_form_on_scaffold_positions(r, fwhm):
    enc = IdealEncoder(r=r, n_freq=512, seed=0)
    rng = np.random.RandomState(r)
    xs = rng.randint(0, NPOS, 300)
    ys = rng.randint(0, NPOS, 300)
    codes = torch.from_numpy(grid_codes(LAMBDAS, xs, ys, fwhm))
    with torch.no_grad():
        z = enc(codes).numpy().astype(np.float64)
    ref = enc.closed_form(xs, ys)
    assert z.shape == (300, 1024)
    np.testing.assert_allclose(z, ref, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)


def test_kernel_is_gaussian_at_short_range():
    """Mean over the draw is exp(-d^2/2r^2); one 512-draw is within ~4/sqrt(1024)."""
    r = 8
    enc = IdealEncoder(r=r, seed=0)
    field = Field(enc, LAMBDAS, 0.25, enc.gain, NPOS)
    x0, y0 = 800, 900
    d = np.array([0, 2, 4, 8, 12, 16, 30])
    z0 = field.encode(np.array([x0]), np.array([y0]))[0]
    z = field.encode(x0 + d, np.full_like(d, y0))
    np.testing.assert_allclose(z @ z0, np.exp(-d ** 2 / (2 * r * r)),
                               atol=4 / np.sqrt(1024))


def test_loader_resolves_an_ideal_spec():
    enc, cfg, gain, fwhm, header = load_probe_encoder(
        "ideal:r=4,n_freq=64,seed=3,gain=100")
    assert isinstance(enc, IdealEncoder)
    assert (enc.r, enc.n_freq, enc.seed) == (4.0, 64, 3)
    assert cfg.out_dim == 128 and list(cfg.lambdas) == LAMBDAS
    assert gain == 100.0 and fwhm == 0.25
    assert header["ideal"] == {"r": 4.0, "n_freq": 64, "seed": 3}
