"""``analysis.hopfield_probe.gwn_online``: the streaming pieces of Sec 22.

Pinned: ``LockIn`` and ``ShiftLMS`` are step for step the Sec 21 functions they
replace; ``L2Stream``'s online waves are the waves of the L1 it exports; and
``depression_l1`` without noise is the Sec 21.1 depression rule.
"""
from __future__ import annotations

import numpy as np
import torch

from analysis.hopfield_probe import gwn_l1_noisefix as NF
from analysis.hopfield_probe import gwn_l1_online as G
from analysis.hopfield_probe import gwn_online as O
from analysis.hopfield_probe.gwn_encoder import exact_l1, random_triples, unit_waves
from analysis.hopfield_probe.gwn_l1l3_local import depression_gains
from analysis.hopfield_probe.gwn_local_gain import walk_in_env


class _NoWorld:
    rects = None


def _steps(T=600, noise=0.0):
    return G.make_steps(20, 0.10, noise, T, _NoWorld, 5)


def test_lockin_matches_lock_in(monkeypatch):
    monkeypatch.setattr(G, "ETAS", (0.2, 0.05))
    monkeypatch.setattr(G, "KPER", 12)
    IN, OUT, ax, _ = _steps()
    for rule, ratio in (("2c", 0.001), ("2b", 1.0), ("2e", 0.1)):
        ref = G.lock_in(IN, OUT, ax, rule, seed=3, checks={len(IN)}, phi_ratio=ratio)[len(IN)][0]
        lk = O.LockIn((0.2, 0.05), rule=rule, phi_ratio=ratio, seed=3, kper=12)
        lk.run(IN, OUT, ax)
        assert np.array_equal(lk.snapshot()[0], ref), rule


def test_shiftlms_matches_noisefix(monkeypatch):
    monkeypatch.setattr(NF, "RATES", (0.1, 0.02))
    IN, OUT, ax, _ = _steps(noise=0.01)
    ref = NF.lms(IN, OUT, ax, {len(IN)}, 1e-4)[len(IN)]
    sh = O.ShiftLMS((0.1, 0.02), sig2=1e-4)
    sh.run(IN, OUT, ax)
    got = sh.snapshot()
    for v in ref:
        assert np.allclose(got[v], ref[v], atol=1e-6), v


def test_l2stream_waves_are_its_l1s_waves(monkeypatch):
    monkeypatch.setattr(G, "QMAX", 1)
    lk = O.LockIn((0.2,), rule="2c", seed=1, kper=G.LP, track=True)
    st = O.L2Stream(lk, 0, 300, seed=2, T_total=10, variants={"a": ("always", 0.005)})
    pos = np.array([[701, 905], [702, 905]])
    IN, OUT, ax, _ = O.session_steps(pos, 0.10, 0.0, torch.Generator().manual_seed(0))
    z0, z1 = G._proj(IN[0], lk.W), G._proj(OUT[0], lk.W)
    st.step(0, z0, z1)
    l1 = st.l1()
    c, s = (t.float() for t in O.l1_out(l1, pos, 0.10))
    C, S = unit_waves(c, s, st.tr)
    Z = st._waves(st._pairs(z0)), st._waves(st._pairs(z1))
    assert torch.allclose(torch.complex(C[0], S[0]), Z[0], atol=1e-4)
    assert torch.allclose(torch.complex(C[1], S[1]), Z[1], atol=1e-4)


def test_depression_l1_without_noise_is_sec_21_1():
    l1 = exact_l1(0.10).to(O.DEV)
    tr = random_triples(l1, 2000, seed=0)
    pos = walk_in_env(700, 900, 14, 300, 9)
    assert torch.allclose(O.depression_l1(l1, tr, pos, 0.10, 0.0, seed=0), depression_gains(l1, tr, pos),
                          rtol=1e-4, atol=1e-6)
