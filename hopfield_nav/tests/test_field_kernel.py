"""Two-scale options of the ideal encoder: module against closed form, and the original draw."""
import numpy as np
import pytest
import torch

from analysis.hopfield_probe.encode import grid_codes
from analysis.hopfield_probe.ideal_encoder import IdealEncoder, draw_frequencies, parse_ideal_spec

LAM = [11, 12, 13]


@pytest.mark.parametrize("kw", [
    dict(grid_fast=0.3, active=0.1),
    dict(grid_units=0.5, active=0.1),
    dict(fast=0.2, active=0.1),
    dict(grid_fast=0.1, center_global=True),
])
def test_module_equals_closed_form(kw):
    rng = np.random.default_rng(0)
    xs, ys = rng.integers(0, 1716, 200), rng.integers(0, 1716, 200)
    g = torch.from_numpy(grid_codes(LAM, xs, ys, 0.25))
    e = IdealEncoder(r=23.2, relu_mix=True, **kw)
    a, b = e(g).numpy(), e.closed_form(xs, ys)
    assert np.abs(a - b).max() < 1e-5
    if not kw.get("center_global"):
        assert b.min() >= 0
        assert abs((b > 0).mean() - kw["active"]) < 0.02


def test_fast_zero_is_the_original_draw():
    base = draw_frequencies(16, 512, 0, 1716)
    assert np.array_equal(base, draw_frequencies(16, 512, 0, 1716, fast=0.0))
    f = draw_frequencies(16, 512, 0, 1716, fast=0.25, r_fast=2)
    assert np.array_equal(f[:384], draw_frequencies(16, 384, 0, 1716))
    # fast waves are about 8 times faster
    assert np.abs(f[384:]).mean() > 5 * np.abs(f[:384]).mean()


def test_grid_input_has_unit_variance_share():
    e = IdealEncoder(r=23.2, relu_mix=True, grid_fast=0.3, active=0.1)
    rng = np.random.default_rng(1)
    xs, ys = rng.integers(0, 1716, 4000), rng.integers(0, 1716, 4000)
    y = e.drive(e._waves_np(xs, ys), e._grid_np(xs, ys)) * np.sqrt(e.out_dim)
    assert abs(y.var() - 1.0) < 0.05
    assert abs(y.mean()) < 0.02


def test_spec_keys():
    kw = parse_ideal_spec("ideal:r=16,relu_mix=1,grid_fast=0.2,grid_seed=3,fast=0.1,r_fast=2")
    assert kw["grid_fast"] == 0.2 and kw["grid_seed"] == 3 and kw["fast"] == 0.1
