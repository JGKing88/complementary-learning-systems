"""Sec 21 follow-up: removing the cell-noise bias from the shift estimate.

Cell noise is independent across moments, so it leaves A = sum g1 g0^T alone
but adds N sigma^2 I to C = sum g0 g0^T. The estimate A C^-1 then shrinks each
pattern by c_j / (c_j + sigma^2); a box visits phases unevenly, so the
shrinkage is not uniform and tilts the eigenvectors. With sigma^2 known (an
animal can measure it at rest, as the gains' at-rest correction does):

* 1b:   A (C + ridge)^-1                      (as gwn_l1_online)
* 1b-corrected: A (C - N sigma^2 I + ridge)^-1
* 1c:   T <- T + rate (g1 - T g0) g0^T / |g0|^2
* 1c-corrected (bias-compensated delta rule):
        T <- T + rate [(g1 - T g0) g0^T + sigma^2 T] / |g0|^2
        whose mean fixed point is the noise-free shift.

Bump 0.10, cell noise 0.01, boxes 14 and 40 and the scattered steps, up to
300k steps; each learner at the last step count through the network, learned
and as its exact twin (``gwn_l1_twin``).

    python -m analysis.hopfield_probe.gwn_l1_noisefix --out DIR
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .gwn_check import DEV, LAMBDAS, SMOKE, World, _dump
from .gwn_l1_learn import _assemble, codes_at, eig_pairs
from .gwn_l1_online import NOISE, RIDGE, exact_shift, make_steps, pairs_stats
from .gwn_l1_twin import network, twin_l1

FWHM = 0.10
RATES = (0.1, 0.02)
T_CHECK = (2_000, 8_000) if SMOKE else (30_000, 100_000, 300_000)
SIDES = (14,) if SMOKE else (14, 40, "scat")


@torch.no_grad()
def lms(IN, OUT, ax, checks, sig2):
    """1c at RATES, plain and bias-compensated; {T: {(rate, corrected): (2, M, LP, LP)}}."""
    T, M, Lp = IN.shape
    variants = [(r, c) for r in RATES for c in (False, True)]
    V = len(variants)
    rate = torch.tensor([r for r, _ in variants], device=DEV).repeat_interleave(M)[:, None, None]
    comp = torch.tensor([sig2 if c else 0.0 for _, c in variants], device=DEV).repeat_interleave(M)
    Ts = torch.zeros(2, V * M, Lp, Lp, device=DEV)
    inv_n = 1.0 / (IN ** 2).sum(-1)
    snaps = {}
    for t in range(T):
        a = int(ax[t])
        g0 = IN[t].repeat(V, 1)
        inv = inv_n[t].repeat(V)
        err = OUT[t].repeat(V, 1) - torch.bmm(Ts[a], g0[:, :, None])[:, :, 0]
        Ts[a] += rate * ((err * inv[:, None])[:, :, None] * g0[:, None, :]
                         + (comp * inv)[:, None, None] * Ts[a])
        if t + 1 in checks:
            X = Ts.reshape(2, V, M, Lp, Lp).cpu().numpy()
            snaps[t + 1] = {v: X[:, i].copy() for i, v in enumerate(variants)}
    return snaps


def ridge_ops(IN, OUT, ax, T, sig2):
    """1b and 1b-corrected at step count T; {corrected: (2, M, LP, LP)}."""
    M, Lp = IN.shape[1:]
    out = {False: np.zeros((2, M, Lp, Lp)), True: np.zeros((2, M, Lp, Lp))}
    for m, l in enumerate(LAMBDAS):
        L = l * l
        X0 = IN[:T, m, :L].double().cpu().numpy(); X1 = OUT[:T, m, :L].double().cpu().numpy()
        for e in (0, 1):
            s = ax[:T] == e
            A, C = X1[s].T @ X0[s], X0[s].T @ X0[s]
            for corr in (False, True):
                Cc = C - (s.sum() * sig2 * np.eye(L) if corr else 0)
                Cc = Cc + RIDGE * np.trace(C) / L * np.eye(L)
                out[corr][e, m, :L, :L] = np.linalg.solve(Cc, A.T).T
    return out


def pairs_and_err(ops):
    per, err = [], 0.0
    for m, l in enumerate(LAMBDAS):
        L = l * l
        Tx, Ty = (ops[e, m, :L, :L].astype(np.float64) for e in (0, 1))
        err = max(err, max(float(np.abs(X - exact_shift(l, e)).max()) for e, X in ((0, Tx), (1, Ty))))
        per.append(eig_pairs(Tx, Ty))
    return per, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    W = World(fwhm=FWHM)
    sig2 = NOISE ** 2
    Tmax = max(T_CHECK)
    res = {}
    for ci, side in enumerate(SIDES):
        name = f"side{side}_noise{NOISE:g}"
        IN, OUT, ax, (p0, p1) = make_steps(side, FWHM, NOISE, Tmax, W, 3000 + ci)
        snaps = lms(IN, OUT, ax, set(T_CHECK), sig2)
        learners = {}
        r = {}
        for T in T_CHECK:
            ops = {f"1c_rate{rt:g}" + ("_corrected" if c else ""): X for (rt, c), X in snaps[T].items()}
            ops.update({"1b" + ("_corrected" if c else ""): X
                        for c, X in ridge_ops(IN, OUT, ax, T, sig2).items()})
            for k, X in ops.items():
                per, err = pairs_and_err(X)
                st = pairs_stats(per, FWHM)
                st["max_abs_T_minus_P"] = err
                r[f"{k}_T{T}"] = st
                print(f"{name} {k} T={T}: pure {st['frac_pure_0.99']:.3f} cov {st['cov_conj_min']:.3f} "
                      f"|T-P| {err:.2e}", flush=True)
                if T == Tmax:
                    learners[k] = per
        del IN, OUT
        if side == "scat":
            vis = np.argwhere(W.mask)
            cells = vis[np.random.default_rng(3).permutation(len(vis))[:5_000 if SMOKE else 50_000]]
            gsteps = W.steps
        else:
            cells = np.unique(np.concatenate([p0, p1[-1:]]), axis=0)
            idx = np.linspace(0, Tmax - 1, 512 if SMOKE else 8192).astype(int)
            gsteps = (p0[idx], p1[idx])
        codes = codes_at(cells, FWHM)
        for k, per in learners.items():
            l1, _ = _assemble(per, codes, FWHM)
            r[f"{k}_learned"] = network(l1, gsteps, FWHM, NOISE, W, f"{name} {k} learned")
            r[f"{k}_twin"] = network(twin_l1(per, FWHM), gsteps, FWHM, NOISE, W, f"{name} {k} twin")
        res[name] = r
        _dump(os.path.join(a.out, "noisefix.json"), res)
    print("DONE noisefix", flush=True)


if __name__ == "__main__":
    main()
