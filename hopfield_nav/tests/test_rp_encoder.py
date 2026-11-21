"""``analysis.hopfield_probe.rp_encoder``: random projection + threshold + readout.

Pinned: the torch grid code equals ``encode.grid_codes`` at several bump
widths, the calibrated threshold gives the requested activity, a saved encoder
reloads to the same outputs, and the probe loader resolves ``rpnet:`` with the
stored bump width. A few thousand positions only.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from analysis.hopfield_probe.encode import grid_codes
from analysis.hopfield_probe.harness import load_probe_encoder
from analysis.hopfield_probe.rp_encoder import (
    RPEncoder, grid_codes_torch, load_rp_encoder, make_random_layer, save_rp_encoder,
)

LAMBDAS = [11, 12, 13]


@pytest.mark.parametrize("fwhm", [0.25, 0.1, 0.0])
def test_torch_grid_codes_match_encode(fwhm):
    rng = np.random.default_rng(0)
    xs, ys = rng.integers(0, 1716, 500), rng.integers(0, 1716, 500)
    ref = grid_codes(LAMBDAS, xs, ys, fwhm)
    got = grid_codes_torch(torch.from_numpy(xs), torch.from_numpy(ys), LAMBDAS, fwhm).numpy()
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("active", [0.5, 0.1])
def test_threshold_sets_activity(active):
    layer = make_random_layer(256, seed=1, nonlin="hard", active=active, n_calib=20_000)
    xy = torch.randint(0, 1716, (2, 4000), generator=torch.Generator().manual_seed(2))
    h = layer(grid_codes_torch(xy[0], xy[1]))
    assert abs(h.mean().item() - active) < 0.03


def test_save_load_and_probe_spec(tmp_path):
    layer = make_random_layer(128, seed=3, nonlin="sigmoid", gain=4.0, fwhm_ratio=0.15,
                              n_calib=5000)
    A = torch.randn(128, 16, generator=torch.Generator().manual_seed(4))
    enc = RPEncoder(layer, torch.zeros(128), A, fwhm_ratio=0.15, meta={"coverage": 0.05})
    p = tmp_path / "rp.pt"
    save_rp_encoder(enc, str(p))
    codes = grid_codes_torch(torch.arange(50), torch.arange(50) * 7, fwhm_ratio=0.15)
    out = enc(codes)
    torch.testing.assert_close(out.norm(dim=1), torch.ones(50))
    torch.testing.assert_close(load_rp_encoder(str(p))(codes), out)
    enc2, cfg, gain, fwhm, header = load_probe_encoder(f"rpnet:{p}")
    assert fwhm == 0.15 and header["hidden_dim"] == 128 and cfg.out_dim == 16
    torch.testing.assert_close(enc2(codes), out)
