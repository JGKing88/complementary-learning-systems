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
    assert header["ideal"] == {"r": 4.0, "n_freq": 64, "seed": 3,
                               "weights": None}


# --- IdealNet: the same computation as explicit layers ----------------------

def test_ideal_net_matches_ideal_encoder_and_closed_form():
    import numpy as np
    import torch
    from analysis.hopfield_probe.encode import grid_codes
    from analysis.hopfield_probe.ideal_encoder import IdealEncoder
    from analysis.hopfield_probe.ideal_net import IdealNet

    rng = np.random.RandomState(3)
    xs, ys = rng.randint(0, 1716, 300), rng.randint(0, 1716, 300)
    for fwhm in (0.25, 0.0):
        codes = torch.from_numpy(grid_codes([11, 12, 13], xs, ys, fwhm))
        for r in (4.0, 16.0, 48.0):
            net, enc = IdealNet(r=r), IdealEncoder(r=r)
            a = net(codes).double().numpy()
            assert np.abs(a - enc(codes).double().numpy()).max() < 2e-6
            assert np.abs(a - net_closed(enc, xs, ys)).max() < 2e-6
            assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def net_closed(enc, xs, ys):
    return enc.closed_form(xs, ys)


def test_ideal_net_weights_are_integer_and_shaped():
    from analysis.hopfield_probe.ideal_net import IdealNet

    net = IdealNet(r=16.0)
    assert tuple(net.readout.weight.shape) == (12, 434)
    assert tuple(net.harmonics.weight.shape) == (512, 6)
    assert net.integer_distance() == 0.0
    assert not net.harmonics.weight.requires_grad
    assert IdealNet(r=16.0, trainable_harmonics=True,
                    init="random").integer_distance() > 0.0


def test_non_integer_harmonics_break_at_module_wraps():
    """With the integer weights nudged, the code jumps where atan2 wraps.

    atan2 returns (-pi, pi], so module 11's x phase 2 pi (x mod 11)/11 jumps
    by -2 pi between x mod 11 = 5 and 6. Modules 12 and 13 do not wrap there.
    """
    import numpy as np
    import torch
    from analysis.hopfield_probe.encode import grid_codes
    from analysis.hopfield_probe.ideal_net import IdealNet

    net = IdealNet(r=16.0)
    xs = np.array([4, 5, 6]); ys = np.zeros(3, dtype=int)
    codes = torch.from_numpy(grid_codes([11, 12, 13], xs, ys, 0.25))
    z = net(codes).double().numpy()
    ordinary, across = np.linalg.norm(z[1] - z[0]), np.linalg.norm(z[2] - z[1])
    assert abs(across - ordinary) < 0.02 * ordinary     # integers: no seam
    with torch.no_grad():
        net.harmonics.weight[:, 0].add_(0.3)            # nudge the 11-x weight
    z2 = net(codes).double().numpy()
    ordinary2, across2 = np.linalg.norm(z2[1] - z2[0]), np.linalg.norm(z2[2] - z2[1])
    assert across2 > 2 * ordinary2                       # a seam at the wrap
