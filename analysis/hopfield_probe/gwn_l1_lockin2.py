"""Sec 21 follow-up: can a lock-in rule keep its coverage? (fully online L1)

Rule 2c (predictive lock-in with a phase-locked turn) purifies every unit but
collapses onto a few coarse patterns: the turn follows the unit's output, which
the strongest-input pattern dominates long before it dominates the weights
(27-44% of units on the constant pattern, no weak pattern covered;
``gwn_l1_online``). Two fixes, run side by side:

* 2c with a slower turn: eta_phi = ratio * eta, ratio 0.1, 0.01, 0.001;
* 2e: the turn read from the weights, angle(w^H u_e) = angle(sum_j |a_j|^2
  e^{i theta_j}), which weights each pattern by its share of the weights, not
  by its input power; ratio 1 and 0.1.

Bump 0.10; boxes 14 and 20 and the scattered steps, and box 14 at cell noise
0.01; eta 0.05 and 0.2; 6 l^2 units per module; up to 300k steps. The best
variant per condition (all units pure above 0.99 at >= 95%, then most
coverage) goes through the network: every unit (learned and exact twin) and
one unit per pattern (dedup).

    python -m analysis.hopfield_probe.gwn_l1_lockin2 --out DIR
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from . import gwn_l1_online as G
from .gwn_check import SMOKE, World, _dump
from .gwn_l1_learn import _assemble, codes_at
from .gwn_l1_twin import network, twin_l1

FWHM = 0.10
ETAS = (0.2,) if SMOKE else (0.05, 0.2)           # set on G at run time (not at import)
T_CHECK = (2_000, 8_000) if SMOKE else (30_000, 100_000, 300_000)
VARIANTS = ([("2c", 0.1), ("2e", 1.0)] if SMOKE else
            [("2c", 0.1), ("2c", 0.01), ("2c", 0.001), ("2e", 1.0), ("2e", 0.1)])
CONDS = [(14, 0.0)] if SMOKE else [(14, 0.0), (20, 0.0), ("scat", 0.0), (14, G.NOISE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G.ETAS = ETAS
    torch.backends.cuda.matmul.allow_tf32 = False
    W = World(fwhm=FWHM)
    Tmax = max(T_CHECK)
    res = {}
    for ci, (side, noise) in enumerate(CONDS):
        name = f"side{side}_noise{noise:g}"
        IN, OUT, ax, (p0, p1) = G.make_steps(side, FWHM, noise, Tmax, W, 5000 + ci)
        r, finals = {}, {}
        for rule, ratio in VARIANTS:
            t0 = time.time()
            snaps = G.lock_in(IN, OUT, ax, rule, seed=5100 + ci, checks=set(T_CHECK), phi_ratio=ratio)
            for gi, eta in enumerate(G.ETAS):
                key = f"{rule}_ratio{ratio:g}_eta{eta:g}"
                for T in T_CHECK:
                    st = G.group_stats(snaps[T][0], gi, FWHM)
                    r[f"{key}_T{T}"] = st
                    mods = st["modules"]
                    print(f"{name} {key} T={T} ({time.time() - t0:.0f}s): pure {st['frac_pure_0.99']:.3f} "
                          f"cov {st[f'cov_conj_q{G.QMAX}_min']:.3f} "
                          f"weak-half {min(mods[str(l)]['cov_conj_weak_half'] for l in G.LAMBDAS):.2f} "
                          f"on const {max(mods[str(l)]['share_on_const'] for l in G.LAMBDAS):.2f}",
                          flush=True)
                finals[key] = (snaps[Tmax][0], gi, r[f"{key}_T{Tmax}"])
            del snaps
        del IN, OUT
        best = max(finals, key=lambda k: (finals[k][2]["frac_pure_0.99"] >= 0.95,
                                          finals[k][2][f"cov_conj_q{G.QMAX}_min"],
                                          finals[k][2]["frac_pure_0.99"]))
        Wf, gi, _ = finals[best]
        r["best"] = best
        if side == "scat":
            vis = np.argwhere(W.mask)
            cells = vis[np.random.default_rng(3).permutation(len(vis))[:5_000 if SMOKE else 50_000]]
            gsteps = W.steps
        else:
            cells = np.unique(np.concatenate([p0, p1[-1:]]), axis=0)
            idx = np.linspace(0, Tmax - 1, 512 if SMOKE else 8192).astype(int)
            gsteps = (p0[idx], p1[idx])
        codes = codes_at(cells, FWHM)
        per_all = G.route2_pairs(Wf, gi, dedup=False)
        l1, _ = _assemble(per_all, codes, FWHM)
        r["all_learned"] = network(l1, gsteps, FWHM, noise, W, f"{name} {best} all learned")
        r["all_twin"] = network(twin_l1(per_all, FWHM), gsteps, FWHM, noise, W, f"{name} {best} all twin")
        l1d, _ = _assemble(G.route2_pairs(Wf, gi, dedup=True), codes, FWHM)
        r["dedup_learned"] = network(l1d, gsteps, FWHM, noise, W, f"{name} {best} dedup learned")
        res[name] = r
        _dump(os.path.join(a.out, "lockin2.json"), res)
    print("DONE lockin2", flush=True)


if __name__ == "__main__":
    main()
