"""Sec 21 follow-up: the learning rate of route 1c, the online delta rule for the shift.

With rate 0.5 each step overwrites half of a synapse's estimate, so with cell
noise the estimate never averages: 1c reached purity 0.03-0.07 at noise 0.01
(``gwn_l1_online``), against 1.000 without noise. Rates 0.5, 0.1, 0.02 and
0.005 run side by side on one walk (bump 0.10; boxes 14 and 40 and the
scattered steps; noise 0 and 0.01; up to 300k steps). The best rate per
condition (most pure pairs, then smallest |T - P|) goes through the network,
learned and as its exact twin (``gwn_l1_twin``).

    python -m analysis.hopfield_probe.gwn_l1_lms --out DIR
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .gwn_check import DEV, LAMBDAS, SMOKE, World, _dump
from .gwn_l1_learn import _assemble, codes_at, eig_pairs
from .gwn_l1_online import LP, NOISE, exact_shift, make_steps, pairs_stats
from .gwn_l1_twin import network, twin_l1

RATES = (0.5, 0.1, 0.02, 0.005)
T_CHECK = (2_000, 8_000) if SMOKE else (10_000, 30_000, 100_000, 300_000)
CONDS = [(s, n) for n in (0.0, NOISE) for s in ((14,) if SMOKE else (14, 40, "scat"))]
FWHM = 0.10


@torch.no_grad()
def lms_rates(IN, OUT, ax, checks):
    """1c at every rate at once; {T: (R, 2, M, LP, LP) numpy}."""
    T, M, Lp = IN.shape
    R = len(RATES)
    rate = torch.tensor(RATES, device=DEV).repeat_interleave(M)[:, None, None]   # (R M, 1, 1)
    Ts = torch.zeros(2, R * M, Lp, Lp, device=DEV)
    inv_n = 1.0 / (IN ** 2).sum(-1)
    snaps = {}
    for t in range(T):
        a = int(ax[t])
        g0 = IN[t].repeat(R, 1)
        err = OUT[t].repeat(R, 1) - torch.bmm(Ts[a], g0[:, :, None])[:, :, 0]
        Ts[a] += rate * (err * inv_n[t].repeat(R)[:, None])[:, :, None] * g0[:, None, :]
        if t + 1 in checks:
            snaps[t + 1] = Ts.reshape(2, R, M, Lp, Lp).transpose(0, 1).cpu().numpy().copy()
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    W = World(fwhm=FWHM)
    Tmax = max(T_CHECK)
    res = {}
    for ci, (side, noise) in enumerate(CONDS):
        name = f"side{side}_noise{noise:g}"
        IN, OUT, ax, (p0, p1) = make_steps(side, FWHM, noise, Tmax, W, 2000 + ci)
        snaps = lms_rates(IN, OUT, ax, set(T_CHECK))
        del IN, OUT
        r = {}
        best, best_key = None, None
        for T in T_CHECK:
            for ri, rate in enumerate(RATES):
                per, err = [], 0.0
                for m, l in enumerate(LAMBDAS):
                    L = l * l
                    Tx, Ty = (snaps[T][ri, e, m, :L, :L].astype(np.float64) for e in (0, 1))
                    err = max(err, max(float(np.abs(X - exact_shift(l, e)).max())
                                       for e, X in ((0, Tx), (1, Ty))))
                    per.append(eig_pairs(Tx, Ty))
                st = pairs_stats(per, FWHM)
                st["max_abs_T_minus_P"] = err
                r[f"rate{rate:g}_T{T}"] = st
                print(f"{name} rate {rate:g} T={T}: pure {st['frac_pure_0.99']:.3f} "
                      f"cov {st['cov_conj_min']:.3f} |T-P| {err:.2e}", flush=True)
                if T == Tmax:
                    key = (round(st["frac_pure_0.99"], 3), -err)
                    if best is None or key > best_key:
                        best, best_key = (rate, per), key
        rate, per = best
        if side == "scat":
            vis = np.argwhere(W.mask)
            cells = vis[np.random.default_rng(3).permutation(len(vis))[:5_000 if SMOKE else 50_000]]
            gsteps = W.steps
        else:
            cells = np.unique(np.concatenate([p0, p1[-1:]]), axis=0)
            idx = np.linspace(0, Tmax - 1, 512 if SMOKE else 8192).astype(int)
            gsteps = (p0[idx], p1[idx])
        l1, _ = _assemble(per, codes_at(cells, FWHM), FWHM)
        r["best_rate"] = rate
        r["learned"] = network(l1, gsteps, FWHM, noise, W, f"{name} rate {rate:g} learned")
        r["twin"] = network(twin_l1(per, FWHM), gsteps, FWHM, noise, W, f"{name} rate {rate:g} twin")
        res[name] = r
        _dump(os.path.join(a.out, "lms_rate.json"), res)
    print("DONE lms", flush=True)


if __name__ == "__main__":
    main()
