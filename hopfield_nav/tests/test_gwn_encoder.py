"""``analysis.hopfield_probe.gwn_encoder``: the grid wave network.

Pinned: exact L1 turns real grid codes into cos/sin of each module pattern, a
wave unit is the plane wave n = 156 j11 + 143 j12 + 132 j13 (mod 1716), the
quarter-squares product is exact at p = 2, and a saved network reloads through
the probe's ``gwn:`` spec. A few hundred positions only.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from analysis.hopfield_probe.encode import grid_codes
from analysis.hopfield_probe.gwn_encoder import (
    GWNEncoder, bump_strength, exact_l1, random_triples, save_gwn, triple_frequencies,
    unit_waves,
)
from analysis.hopfield_probe.harness import load_probe_encoder

LAMBDAS = [11, 12, 13]
rng = np.random.default_rng(0)
XS, YS = rng.integers(0, 1716, 300), rng.integers(0, 1716, 300)


@pytest.mark.parametrize("fwhm", [0.25, 0.10])
def test_exact_l1_gives_module_waves(fwhm):
    l1 = exact_l1(fwhm)
    g = torch.from_numpy(grid_codes(LAMBDAS, XS, YS, fwhm)).double()
    c, s = g @ l1.Wc, g @ l1.Ws
    for m, l in enumerate(LAMBDAS):
        idx = l1.pairs_of(m)
        j = l1.pattern[idx].numpy()
        th = 2 * np.pi * (np.outer(XS, j[:, 0]) + np.outer(YS, j[:, 1])) / l
        # the float32 grid code's rounding (~6e-8 per cell) is divided by A(j),
        # down to ~1e-7 of A(0) for the finest 2D pattern at bump 0.25
        A = bump_strength(l, fwhm, j[:, 0]) * bump_strength(l, fwhm, j[:, 1])
        tol = 1e-9 + 2e-7 * l * l / np.abs(A)
        assert (np.abs(c[:, idx].numpy() - np.cos(th)) <= tol[None, :]).all()
        assert (np.abs(s[:, idx].numpy() - np.sin(th)) <= tol[None, :]).all()


@pytest.mark.parametrize("mult,param", [("exact", None), ("quarter", 2.0)])
def test_wave_unit_is_plane_wave(mult, param):
    l1 = exact_l1(0.10)
    tr = random_triples(l1, 500, seed=1)
    n = triple_frequencies(l1, tr).numpy()
    g = torch.from_numpy(grid_codes(LAMBDAS, XS, YS, 0.10)).double()
    C, S = unit_waves(g @ l1.Wc, g @ l1.Ws, tr, mult, param)
    th = 2 * np.pi * (np.outer(XS, n[:, 0]) + np.outer(YS, n[:, 1])) / 1716
    np.testing.assert_allclose(C.numpy(), np.cos(th), atol=1e-4)
    np.testing.assert_allclose(S.numpy(), np.sin(th), atol=1e-4)


def test_save_load_probe_spec(tmp_path):
    l1 = exact_l1(0.10)
    tr = random_triples(l1, 40, seed=2)
    enc = GWNEncoder(l1, tr, torch.rand(40, dtype=torch.float64), fwhm_ratio=0.10,
                     meta={"coverage": 0.05})
    p = tmp_path / "g.pt"
    save_gwn(enc, str(p))
    g = torch.from_numpy(grid_codes(LAMBDAS, XS[:50], YS[:50], 0.10))
    out = enc(g)
    torch.testing.assert_close(out.norm(dim=1), torch.ones(50))
    enc2, cfg, gain, fwhm, header = load_probe_encoder(f"gwn:{p}")
    assert fwhm == 0.10 and cfg.out_dim == 80 and header["gwn"]["units"] == 40
    torch.testing.assert_close(enc2(g), out)
