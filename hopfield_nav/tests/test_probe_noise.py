"""``Field`` test-time grid noise (``run.py --grid_noise``).

Pinned: noise 0 is the old encode exactly; noise > 0 changes every encode with
a fresh draw; the local basis of the direction readout stays clean unless
``noise_basis``.
"""
from __future__ import annotations

import numpy as np
import torch

from analysis.hopfield_probe.encode import Field, encode_positions

LAM = [3, 4, 5]
XS, YS = np.arange(10), np.arange(10) * 2


def _field(**kw):
    return Field(encoder=torch.nn.Identity(), lambdas=LAM, fwhm_ratio=0.25, gain=1.0, Npos=60, **kw)


def test_zero_noise_is_the_old_encode():
    ref = encode_positions(torch.nn.Identity(), LAM, 0.25, XS, YS)
    assert np.array_equal(_field().encode(XS, YS), ref)


def test_noise_changes_every_draw():
    f = _field(noise=0.05)
    a, b = f.encode(XS, YS), f.encode(XS, YS)
    clean = _field().encode(XS, YS)
    assert not np.allclose(a, clean) and not np.allclose(a, b)
    assert abs(float(np.std(a - clean)) - 0.05) < 0.01


def test_local_basis_clean_unless_asked():
    pos = np.stack([XS + 5, YS + 5], 1)
    clean = _field().local_basis(pos, (0, 0))
    assert np.allclose(_field(noise=0.05).local_basis(pos, (0, 0)), clean)
    assert not np.allclose(_field(noise=0.05, noise_basis=True).local_basis(pos, (0, 0)), clean)
