"""Sec 21 follow-up: the shift rule under cell noise with a slow rate and the bias correction.

Under cell noise 0.01 the shift rule 1c has two errors. Variance: a fixed rate
remembers only the last ~1/rate visits of a cell. Bias: noise adds sigma^2 I to
the input covariance, which shrinks each pattern by c_j / (c_j + sigma^2),
unevenly in a box. At rate 0.02 the variance dominates (the correction cut
|T - P| from 0.04 to 0.011 but the network did not improve), and at rate 0.005
over 1M steps |T - P| stayed at 0.045, the bias floor (``gwn_l1_multienv``).
So: slow rates (0.005, 0.002) with and without the bias-compensated update

    T <- T + rate [(g1 - T g0) g0^T + sigma^2 T] / |g0|^2,

beside 1b with and without C - N sigma^2 I. Bump 0.10, cell noise 0.01, 1M
steps; the 6-box session walk of ``gwn_l1_multienv`` and the 14-box. Each
learner through the network, learned and as its exact twin.

    python -m analysis.hopfield_probe.gwn_l1_noise_slow --out DIR
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from . import gwn_l1_noisefix as NF
from . import gwn_l1_online as G
from .gwn_check import SMOKE, World, _dump
from .gwn_l1_learn import _assemble, codes_at
from .gwn_l1_multienv import make_steps as multi_steps
from .gwn_l1_twin import network, twin_l1

FWHM = 0.10
NF.RATES = (0.005, 0.002)
T_CHECK = (4_000, 12_000) if SMOKE else (300_000, 1_000_000)
CONDS = ["multi"] if SMOKE else ["multi", "box14"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    W = World(fwhm=FWHM)
    sig2 = G.NOISE ** 2
    Tmax = max(T_CHECK)
    res = {}
    for ci, kind in enumerate(CONDS):
        name = f"{kind}_noise{G.NOISE:g}"
        if kind == "multi":
            IN, OUT, ax, (p0, p1), _ = multi_steps("multi", G.NOISE, Tmax, 8000 + ci)
        else:
            IN, OUT, ax, (p0, p1) = G.make_steps(14, FWHM, G.NOISE, Tmax, W, 8000 + ci)
        snaps = NF.lms(IN, OUT, ax, set(T_CHECK), sig2)
        r, learners = {}, {}
        for T in T_CHECK:
            ops = {f"1c_rate{rt:g}" + ("_corrected" if c else ""): X for (rt, c), X in snaps[T].items()}
            ops.update({"1b" + ("_corrected" if c else ""): X
                        for c, X in NF.ridge_ops(IN, OUT, ax, T, sig2).items()})
            for k, X in ops.items():
                per, err = NF.pairs_and_err(X)
                st = G.pairs_stats(per, FWHM)
                st["max_abs_T_minus_P"] = err
                r[f"{k}_T{T}"] = st
                print(f"{name} {k} T={T}: pure {st['frac_pure_0.99']:.3f} cov {st['cov_conj_min']:.3f} "
                      f"|T-P| {err:.2e}", flush=True)
                if T == Tmax:
                    learners[k] = per
        del IN, OUT, snaps
        cells = np.unique(np.concatenate([p0, p1]), axis=0)
        idx = np.linspace(0, Tmax - 1, 512 if SMOKE else 8192).astype(int)
        gsteps, codes = (p0[idx], p1[idx]), codes_at(cells, FWHM)
        for k, per in learners.items():
            l1, _ = _assemble(per, codes, FWHM)
            r[f"{k}_learned"] = network(l1, gsteps, FWHM, G.NOISE, W, f"{name} {k} learned")
            r[f"{k}_twin"] = network(twin_l1(per, FWHM), gsteps, FWHM, G.NOISE, W, f"{name} {k} twin")
        res[name] = r
        _dump(os.path.join(a.out, "noise_slow.json"), res)
    print("DONE noise_slow", flush=True)


if __name__ == "__main__":
    main()
