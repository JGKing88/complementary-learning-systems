"""Sec 21 follow-up: route 1c with a learning rate that falls with experience.

With a fixed rate the delta rule for the shift remembers only the last ~1/rate
visits of each cell, so under cell noise it stays noisy (rate 0.02: network
alias 0.13-0.16 against 0.08-0.10 for its exact twin; ``gwn_l1_noisefix``).
Here each synapse T_ab has its own rate 1 / (1 + n_b), where n_b sums the
presynaptic cell b's normalised activity g0_b^2 / |g0|^2 over the steps along
the synapse's axis (a local count). For one-hot codes this makes column b the
running mean of the codes that followed cell b: least squares, online.

    T_e <- T_e + [(g1 - T_e g0) g0^T + c sigma^2 T_e] diag(rate) / |g0|^2,
    n_e <- n_e + g0^2 / |g0|^2,  rate = 1 / (1 + n_e)

with c = 0 (plain) or 1 (bias-compensated, as in ``gwn_l1_noisefix``). Bump
0.10, boxes 14 and 40 and the scattered steps at cell noise 0.01, and box 14
without noise; each learner at 300k steps through the network, learned and as
its exact twin, beside 1b at the same step count.

    python -m analysis.hopfield_probe.gwn_l1_decay --out DIR
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .gwn_check import DEV, SMOKE, World, _dump
from .gwn_l1_learn import _assemble, codes_at
from .gwn_l1_noisefix import FWHM, pairs_and_err, ridge_ops
from .gwn_l1_online import NOISE, make_steps, pairs_stats
from .gwn_l1_twin import network, twin_l1

T_CHECK = (2_000, 8_000) if SMOKE else (30_000, 100_000, 300_000)
CONDS = [(14, NOISE)] if SMOKE else [(14, NOISE), (40, NOISE), ("scat", NOISE), (14, 0.0)]


@torch.no_grad()
def lms_decay(IN, OUT, ax, checks, sig2):
    """{T: {corrected: (2, M, LP, LP)}} for the plain and bias-compensated rule."""
    T, M, Lp = IN.shape
    comp = torch.tensor([0.0, sig2], device=DEV).repeat_interleave(M)        # (2 M,)
    Ts = torch.zeros(2, 2 * M, Lp, Lp, device=DEV)
    n = torch.zeros(2, 2 * M, Lp, device=DEV)
    inv_n = 1.0 / (IN ** 2).sum(-1)
    snaps = {}
    for t in range(T):
        a = int(ax[t])
        g0 = IN[t].repeat(2, 1)
        inv = inv_n[t].repeat(2)
        n[a] += g0 ** 2 * inv[:, None]
        rate = 1.0 / (1.0 + n[a])                                              # per presynaptic cell
        err = OUT[t].repeat(2, 1) - torch.bmm(Ts[a], g0[:, :, None])[:, :, 0]
        upd = (err * inv[:, None])[:, :, None] * g0[:, None, :] + (comp * inv)[:, None, None] * Ts[a]
        Ts[a] += upd * rate[:, None, :]
        if t + 1 in checks:
            X = Ts.reshape(2, 2, M, Lp, Lp).cpu().numpy()
            snaps[t + 1] = {False: X[:, 0].copy(), True: X[:, 1].copy()}
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
        sig2 = noise ** 2
        IN, OUT, ax, (p0, p1) = make_steps(side, FWHM, noise, Tmax, W, 4000 + ci)
        snaps = lms_decay(IN, OUT, ax, set(T_CHECK), sig2)
        r, learners = {}, {}
        for T in T_CHECK:
            ops = {"1c_decay" + ("_corrected" if c else ""): X for c, X in snaps[T].items()}
            ops["1b"] = ridge_ops(IN, OUT, ax, T, sig2)[False]
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
            r[f"{k}_learned"] = network(l1, gsteps, FWHM, noise, W, f"{name} {k} learned")
            r[f"{k}_twin"] = network(twin_l1(per, FWHM), gsteps, FWHM, noise, W, f"{name} {k} twin")
        res[name] = r
        _dump(os.path.join(a.out, "decay.json"), res)
    print("DONE decay", flush=True)


if __name__ == "__main__":
    main()
