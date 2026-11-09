"""``analysis.hopfield_probe.gwn_l1_online``: learning L1 inside one box.

Pinned: the exact shift is the permutation the walk data obey; a box of side 14
gives every module's shift exactly (routes 1b and 1c); a box of side 13 never
makes module 13's wrap step, so that module alone fails; an exact pattern with
its own turn is a fixed point of the adaptive-turn rule 2c, step by step.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from analysis.hopfield_probe import gwn_l1_online as G


class _NoWorld:
    rects = None


def _walk(side, T, fwhm=0.10, seed=5):
    return G.make_steps(side, fwhm, 0.0, T, _NoWorld, seed)


def _pure_by_module(side, T):
    IN, OUT, ax, _ = _walk(side, T)
    lms = G.lms_shift(IN, OUT, ax, {T})
    r1 = G.route1(IN, OUT, ax, T, lms[T])
    return {k: G.pairs_stats([p for p, _ in mods], 0.10)["modules"] for k, mods in r1.items()}


def test_exact_shift_matches_walk_data():
    IN, OUT, ax, _ = _walk(20, 500)
    for m, l in enumerate(G.LAMBDAS):
        L = l * l
        for e in (0, 1):
            s = ax == e
            X0, X1 = IN[s, m, :L].double().numpy(), OUT[s, m, :L].double().numpy()
            assert np.abs(X1 - X0 @ G.exact_shift(l, e).T).max() < 1e-6


@pytest.mark.parametrize("route", ["1b", "1c"])
def test_box_of_14_gives_every_module(route):
    mods = _pure_by_module(14, 20_000)[route]
    for l in G.LAMBDAS:
        assert mods[str(l)]["frac_pure_0.99"] == 1.0
        assert mods[str(l)]["cov_conj"] == 1.0


def test_box_of_13_misses_module_13_only():
    mods = _pure_by_module(13, 20_000)["1b"]
    assert mods["11"]["frac_pure_0.99"] == 1.0 and mods["12"]["frac_pure_0.99"] == 1.0
    assert mods["13"]["frac_pure_0.99"] < 0.1


def test_pattern_with_its_own_turn_is_a_fixed_point_of_2c(monkeypatch):
    monkeypatch.setattr(G, "ETAS", (0.2,))
    monkeypatch.setattr(G, "KPER", 1)
    IN, OUT, ax, _ = _walk(20, 2_000)
    # start the lone unit of each module at pattern j = (2, 5), with phi = its turns
    # and the predictor u_e = T_e^T w = e^{i theta_e} w already learned
    W0 = torch.zeros(3, G.LP, 1, dtype=torch.complex64)
    phi0 = torch.zeros(2, 3, 1)
    for m, l in enumerate(G.LAMBDAS):
        a, b = np.meshgrid(np.arange(l), np.arange(l), indexing="ij")
        W0[m, :l * l, 0] = torch.tensor(np.exp(2j * np.pi * (2 * a + 5 * b).ravel() / l) / l)
        phi0[0, m, 0], phi0[1, m, 0] = 2 * np.pi * 2 / l, 2 * np.pi * 5 / l
    U0 = torch.stack([torch.exp(1j * phi0[e])[:, None, :] * W0 for e in (0, 1)])
    monkeypatch.setattr(G, "_init_units", lambda M, Lp, K, gen: (W0.clone(), phi0.clone(), U0.clone()))
    W = G.lock_in(IN, OUT, ax, "2c", seed=0, checks={2_000})[2_000][0]
    for m, l in enumerate(G.LAMBDAS):
        pur, j = G.unit_purity(W[m][:l * l], l)
        # exactly stationary: every error term is zero, so purity stays 1 to
        # rounding (a wrong turn drifts to ~0.996-0.9999 in the same steps)
        assert pur[0] > 1 - 1e-6 and j[0] == 2 * l + 5


def test_noisy_slowness_keeps_the_slowest_units():
    """Under noise the at-rest-corrected change of a slow unit can come out
    negative from sampling error; it must stay slow (finite v, gain ~1), not be
    dropped as constant (the bug fixed 2026-11-08)."""
    from analysis.hopfield_probe.gwn_encoder import exact_l1, random_triples, triple_frequencies
    l1 = exact_l1(0.10).to(G.DEV)
    tr = random_triples(l1, 20_000, seed=0)
    n = triple_frequencies(l1, tr).abs().max(1).values
    from analysis.hopfield_probe.gwn_local_gain import walk_in_env
    pos = walk_in_env(700, 900, 14, 600, 3)
    v, _ = G.local_slowness(l1, tr, (pos[:-1], pos[1:]), 0.10, noise=0.01)
    slow = (n <= 12).to(v.device)
    assert int(slow.sum()) > 0
    assert torch.isfinite(v[slow]).all()
