"""Sec 21 follow-up: do more environments, a bigger box and longer walks help the online L1?

Two questions from Sec 21:

* The slow-turn lock-in rule (2c, turn rate 0.001 of the weight rate) learns
  exact units, but patterns are held by unevenly many units and the all-unit
  network aliases at 0.27-0.34 (one unit per pattern 0.07-0.10). Is the
  unevenness a coverage effect, or a bias toward strong patterns that walking
  more places does not change? Measured directly here: units per pattern
  against the pattern's strength A(j)/A(0).
* Under cell noise 0.01 the lock-in rule failed at eta 0.05-0.2 within 300k
  steps, and the shift rule 1c aliased at about twice its twin. More steps
  allow a smaller rate.

Conditions, bump 0.10, 1M steps, without noise and at cell noise 0.01:
``multi`` = sessions of 10k steps cycling through 6 boxes of side 14, 18, 24,
30, 36 and 40 at random scaffold positions (each session a reflecting walk
from the box centre; no step crosses between boxes); ``box40`` = one 40-box.
Learners: 2c with turn ratio 0.001 at eta 0.2, 0.05 and 0.02 (6 l^2 units per
module each); 1c at rates 0.5, 0.1, 0.02 and 0.005. Networks as in
``gwn_l1_lockin2`` (all units, its exact twin, one unit per pattern) and
``gwn_l1_twin`` (1c learned and twin).

    python -m analysis.hopfield_probe.gwn_l1_multienv --out DIR
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from . import gwn_l1_online as G
from .gwn_check import DEV, LAMBDAS, SMOKE, World, _dump
from .gwn_l1_learn import OFFS, _assemble, codes_at, eig_pairs
from .gwn_l1_lms import RATES, lms_rates
from .gwn_l1_twin import network, twin_l1
from .gwn_local_gain import walk_in_env
from .rp_encoder import grid_codes_torch

FWHM = 0.10
NPOS = 1716
ETAS = (0.2, 0.05) if SMOKE else (0.2, 0.05, 0.02)           # set on G at run time (not at import)
PHI_RATIO = 0.001
T_CHECK = (4_000, 12_000) if SMOKE else (100_000, 300_000, 1_000_000)
BLOCK = 2_000 if SMOKE else 10_000
MULTI_SIDES = (14, 18, 24, 30, 36, 40)
CONDS = ([("multi", 0.0)] if SMOKE else
         [("multi", 0.0), ("multi", G.NOISE), ("box40", 0.0), ("box40", G.NOISE)])


def walks(kind, T, seed):
    """Per-session position arrays (each a walk inside one box) totalling T steps."""
    rng = np.random.default_rng(seed)
    if kind == "box40":
        envs = [(700, 900, 40)]
    else:
        envs = [(int(rng.integers(0, NPOS - s)), int(rng.integers(0, NPOS - s)), s) for s in MULTI_SIDES]
    out, t, k = [], 0, 0
    while t < T:
        x0, y0, s = envs[k % len(envs)]
        n = min(BLOCK, T - t)
        out.append(walk_in_env(x0, y0, s, n, seed * 1000 + k))
        t, k = t + n, k + 1
    return out, envs


def make_steps(kind, noise, T, seed):
    """As ``gwn_l1_online.make_steps``, over several sessions; one noisy code per
    moment within a session."""
    sessions, envs = walks(kind, T, seed)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    INs, OUTs, axs, p0s, p1s = [], [], [], [], []
    for pos in sessions:
        g = grid_codes_torch(torch.as_tensor(pos[:, 0], device=DEV), torch.as_tensor(pos[:, 1], device=DEV),
                             LAMBDAS, FWHM)
        if noise > 0:
            g = g + noise * torch.randn(g.shape, generator=gen).to(DEV)
        d = pos[1:] - pos[:-1]
        fwd = torch.tensor(d.sum(1) > 0, device=DEV)[:, None]
        G0, G1 = g[:-1], g[1:]
        IN, OUT = torch.where(fwd, G0, G1), torch.where(fwd, G1, G0)
        for X, acc in ((IN, INs), (OUT, OUTs)):
            Y = torch.zeros(len(X), len(LAMBDAS), G.LP, device=DEV)
            for m, l in enumerate(LAMBDAS):
                Y[:, m, :l * l] = X[:, OFFS[m]:OFFS[m + 1]]
            acc.append(Y)
        axs.append((d[:, 1] != 0).astype(np.int64))
        p0s.append(pos[:-1]); p1s.append(pos[1:])
    return (torch.cat(INs), torch.cat(OUTs), np.concatenate(axs),
            (np.concatenate(p0s), np.concatenate(p1s)), envs)


def count_vs_strength(Wsnap, gi):
    """Per module: units (and conjugates) per pattern among pure units, and the
    mean count on the strongest half of patterns over the weakest half."""
    res = {}
    for m, l in enumerate(LAMBDAS):
        L = l * l
        pur, j = G.unit_purity(Wsnap[m][:L][:, gi * G.KPER + np.arange(G.QMAX * L)], l)
        j = j[pur > 0.99]
        cnt = np.bincount(np.concatenate([j, G.conj_index(j, l)]), minlength=L).astype(float)
        S = G.strengths(l, FWHM)
        order = np.argsort(S)
        weak, strong = order[:L // 2], order[L - L // 2:]
        logS = np.log(S)
        cs, cw = float(cnt[strong].mean()), float(cnt[weak].mean())
        res[str(l)] = {"count_cv": float(cnt.std() / cnt.mean()) if cnt.sum() else None,
                       "mean_count_strong": cs, "mean_count_weak": cw,
                       "strong_over_weak": cs / cw if cw > 0 else None,
                       "corr_count_logstrength": float(np.corrcoef(cnt, logS)[0, 1]) if cnt.std() else None}
    return res


def gsteps_and_codes(p0, p1, T):
    cells = np.unique(np.concatenate([p0, p1]), axis=0)
    idx = np.linspace(0, T - 1, 512 if SMOKE else 8192).astype(int)
    return (p0[idx], p1[idx]), codes_at(cells, FWHM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G.ETAS = ETAS
    torch.backends.cuda.matmul.allow_tf32 = False
    W = World(fwhm=FWHM)
    Tmax = max(T_CHECK)
    res = {}
    for ci, (kind, noise) in enumerate(CONDS):
        name = f"{kind}_noise{noise:g}"
        IN, OUT, ax, (p0, p1), envs = make_steps(kind, noise, Tmax, 6000 + ci)
        r = {"envs": envs, "cells_visited": int(len(np.unique(np.concatenate([p0, p1]), axis=0)))}
        print(f"=== {name}: {len(envs)} boxes, {r['cells_visited']} cells", flush=True)
        gsteps, codes = gsteps_and_codes(p0, p1, Tmax)
        # route 2: slow-turn lock-in
        t0 = time.time()
        snaps = G.lock_in(IN, OUT, ax, "2c", seed=6100 + ci, checks=set(T_CHECK), phi_ratio=PHI_RATIO)
        finals = {}
        for gi, eta in enumerate(G.ETAS):
            for T in T_CHECK:
                st = G.group_stats(snaps[T][0], gi, FWHM)
                st["counts"] = count_vs_strength(snaps[T][0], gi)
                r[f"2c_eta{eta:g}_T{T}"] = st
                c = st["counts"]
                fmt = lambda x: None if x is None else round(x, 2)
                print(f"{name} 2c eta {eta:g} T={T} ({time.time() - t0:.0f}s): pure {st['frac_pure_0.99']:.3f} "
                      f"cov {st[f'cov_conj_q{G.QMAX}_min']:.3f} count CV "
                      f"{[fmt(c[str(l)]['count_cv']) for l in LAMBDAS]} units/pattern strong half vs weak half "
                      f"{[(fmt(c[str(l)]['mean_count_strong']), fmt(c[str(l)]['mean_count_weak'])) for l in LAMBDAS]} "
                      f"corr(count, log strength) {[fmt(c[str(l)]['corr_count_logstrength']) for l in LAMBDAS]}",
                      flush=True)
            finals[eta] = (snaps[Tmax][0], gi, r[f"2c_eta{eta:g}_T{Tmax}"])
        del snaps
        best = max(finals, key=lambda e: (finals[e][2]["frac_pure_0.99"] >= 0.95,
                                          finals[e][2][f"cov_conj_q{G.QMAX}_min"],
                                          finals[e][2]["frac_pure_0.99"]))
        Wf, gi, _ = finals[best]
        r["2c_best_eta"] = best
        per_all = G.route2_pairs(Wf, gi, dedup=False)
        l1, _ = _assemble(per_all, codes, FWHM)
        r["2c_all_learned"] = network(l1, gsteps, FWHM, noise, W, f"{name} 2c eta {best:g} all learned")
        r["2c_all_twin"] = network(twin_l1(per_all, FWHM), gsteps, FWHM, noise, W,
                                   f"{name} 2c eta {best:g} all twin")
        l1d, _ = _assemble(G.route2_pairs(Wf, gi, dedup=True), codes, FWHM)
        r["2c_dedup_learned"] = network(l1d, gsteps, FWHM, noise, W, f"{name} 2c eta {best:g} dedup learned")
        # route 1c
        lms = lms_rates(IN, OUT, ax, set(T_CHECK))
        del IN, OUT
        best1, key1 = None, None
        for T in T_CHECK:
            for ri, rate in enumerate(RATES):
                per, err = [], 0.0
                for m, l in enumerate(LAMBDAS):
                    L = l * l
                    Tx, Ty = (lms[T][ri, e, m, :L, :L].astype(np.float64) for e in (0, 1))
                    err = max(err, max(float(np.abs(X - G.exact_shift(l, e)).max())
                                       for e, X in ((0, Tx), (1, Ty))))
                    per.append(eig_pairs(Tx, Ty))
                st = G.pairs_stats(per, FWHM)
                st["max_abs_T_minus_P"] = err
                r[f"1c_rate{rate:g}_T{T}"] = st
                print(f"{name} 1c rate {rate:g} T={T}: pure {st['frac_pure_0.99']:.3f} "
                      f"cov {st['cov_conj_min']:.3f} |T-P| {err:.2e}", flush=True)
                if T == Tmax and (best1 is None or (round(st["frac_pure_0.99"], 3), -err) > key1):
                    best1, key1 = (rate, per), (round(st["frac_pure_0.99"], 3), -err)
        rate, per = best1
        r["1c_best_rate"] = rate
        l1c, _ = _assemble(per, codes, FWHM)
        r["1c_learned"] = network(l1c, gsteps, FWHM, noise, W, f"{name} 1c rate {rate:g} learned")
        r["1c_twin"] = network(twin_l1(per, FWHM), gsteps, FWHM, noise, W, f"{name} 1c rate {rate:g} twin")
        res[name] = r
        _dump(os.path.join(a.out, "multienv.json"), res)
    print("DONE multienv", flush=True)


if __name__ == "__main__":
    main()
