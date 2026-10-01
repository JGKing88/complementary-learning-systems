#!/usr/bin/env python3
"""The exact optimum of kernel-MSE + coding rate over amplitudes, frequencies fixed.

Objective-first check (docs/EXPERIMENTS_IDEAL_ENCODER.md Sec 12). The code is
``z(p) = [sqrt(a_i) cos(omega_i.p), sqrt(a_i) sin(omega_i.p)]`` with ``a`` on the
simplex (unit-norm codes). The loss is the Stage A kernel MSE on within-patch
pairs plus the recipe's coding-rate term (``losses.coding_rate_loss``):

    L(a) = mean_pairs (sum_i a_i cos(omega_i.Delta) - exp(-|Delta|^2/2r^2))^2
           - lam * (1/(2D)) logdet(I + (D/eps^2) Sigma(a)),
    Sigma(a) = E_p[z z^T] = A^1/2 M A^1/2,  M = E_p[phi phi^T],  A = diag(a, a)

over training positions p (the att0.5 recipe's patches). The MSE is a convex
quadratic in a; by Sylvester logdet(I + c A^1/2 M A^1/2) = logdet(I + c M^1/2 A M^1/2),
concave in a. So L is convex on the simplex and exponentiated-gradient descent
finds the global optimum.

Two frequency menus:
  * ``draw``: the ideal r=16 draw (512 frequencies, D = 1024) -- the network's
    actual budget;
  * ``flat``: every half-plane integer frequency with |n| <= --flat_radius, NOT
    pre-shaped to the Gaussian -- which spectrum does the loss ask for when the
    menu does not already contain the answer? (D = 2 x menu size.)

For each lam: the MSE, the rate, how many waves carry weight (participation
ratio 1/sum a^2), the kernel's half-height and C(1) on the axes, its largest
value beyond the 71-cell patch window, and the far-field sd over random scaffold
pairs. Weights are saved for the probe.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from analysis.hopfield_probe.ideal_net import IdealNet
from encoder_training.data import sample_nonoverlapping_patches

NPOS = 1716


def menus(r, flat_radius):
    draw = IdealNet(r=r).freqs.numpy().astype(np.float64)
    v = []
    R = int(flat_radius)
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            if (x > 0 or (x == 0 and y > 0)) and x * x + y * y <= R * R:
                v.append((x, y))
    v.append((0, 0))
    return {"draw": draw, "flat": np.array(v, dtype=np.float64)}


def stats_for(n, dev, r, n_pairs, n_pos, seed=0):
    """G, b, tt (pair MSE), M (position second moment) for menu n."""
    rng = np.random.RandomState(seed)
    nt = torch.from_numpy(n).to(dev)
    F = n.shape[0]
    G = torch.zeros(F, F, dtype=torch.float64, device=dev)
    b = torch.zeros(F, dtype=torch.float64, device=dev)
    tt, cnt = 0.0, 0
    for _ in range(max(1, n_pairs // 100_000)):
        m = 100_000
        d = rng.randint(0, 50, size=(m, 2)) - rng.randint(0, 50, size=(m, 2))
        d = d[(d != 0).any(1)].astype(np.float64)
        dt = torch.from_numpy(d).to(dev)
        X = torch.cos(2 * math.pi * dt @ nt.T / NPOS)
        t = torch.exp(-(dt ** 2).sum(1) / (2 * r * r))
        G += X.T @ X; b += X.T @ t; tt += float(t @ t); cnt += len(d)
    # training positions: the recipe's patches (seed 0), subsampled
    np.random.seed(seed)
    y0s, x0s, sizes = sample_nonoverlapping_patches(
        NPOS, NPOS, [50] * 118, None, placement="random")
    pts = []
    for y0, x0, s in zip(y0s, x0s, sizes):
        yy, xx = np.meshgrid(np.arange(y0, y0 + s), np.arange(x0, x0 + s),
                             indexing="ij")
        pts.append(np.stack([yy.ravel(), xx.ravel()], 1))
    pts = np.concatenate(pts)
    pts = pts[rng.choice(len(pts), size=min(n_pos, len(pts)), replace=False)]
    M = torch.zeros(2 * F, 2 * F, dtype=torch.float64, device=dev)
    for i in range(0, len(pts), 20_000):
        th = 2 * math.pi * torch.from_numpy(pts[i:i + 20_000].astype(np.float64)
                                            ).to(dev) @ nt.T / NPOS
        Phi = torch.cat([torch.cos(th), torch.sin(th)], 1)
        M += Phi.T @ Phi
    M /= len(pts)
    return G / cnt, b / cnt, tt / cnt, M


def optimise(G, b, tt, M, lam, eps, iters, dev):
    F = G.shape[0]
    D = 2 * F
    w, V = torch.linalg.eigh(M)
    Mh = (V * w.clamp_min(0).sqrt()) @ V.T                      # M^1/2
    c = D / (eps * eps)
    eye = torch.eye(D, dtype=torch.float64, device=dev)

    def parts(a):
        mse = a @ G @ a - 2 * a @ b + tt
        if lam == 0:
            return mse, torch.zeros((), dtype=a.dtype, device=dev)
        A2 = torch.cat([a, a])
        S = eye + c * (Mh * A2) @ Mh                            # I + c M^1/2 A M^1/2
        rate = torch.linalg.slogdet(S)[1] / (2 * D)
        return mse, rate

    a = torch.full((F,), 1.0 / F, dtype=torch.float64, device=dev)
    eta = 1.0
    prev = None
    for it in range(iters):
        a.requires_grad_(True)
        mse, rate = parts(a)
        L = mse - lam * rate
        g, = torch.autograd.grad(L, a)
        with torch.no_grad():
            if prev is not None and float(L) > prev:          # backtrack
                eta *= 0.5
            prev = float(L)
            g = g / (g.abs().max().clamp_min(1e-30))
            a = a.detach() * torch.exp(-eta * g)
            a = a / a.sum()
    a = a.detach()
    mse, rate = parts(a)
    return a, float(mse), float(rate)


def describe(a, n, r, dev, seed=1):
    a_np = a.cpu().numpy()
    nt = torch.from_numpy(n).to(dev)
    d = torch.arange(0, 301, dtype=torch.float64, device=dev)
    prof = []
    for ax in (0, 1):
        D = torch.stack([d, torch.zeros_like(d)], 1) if ax == 0 else \
            torch.stack([torch.zeros_like(d), d], 1)
        prof.append((torch.cos(2 * math.pi * D @ nt.T / NPOS) @ a).cpu().numpy())
    k = np.mean(prof, 0)
    rng = np.random.RandomState(seed)
    p = torch.from_numpy((rng.randint(0, NPOS, (40_000, 2)) -
                          rng.randint(0, NPOS, (40_000, 2))).astype(np.float64)).to(dev)
    far = (torch.cos(2 * math.pi * p @ nt.T / NPOS) @ a).cpu().numpy()
    below = np.where(k < 0.5)[0]
    absn = np.sqrt((n ** 2).sum(1))
    return {"participation": float(1.0 / (a_np ** 2).sum()),
            "n_weighted": int((a_np > 1e-3 * a_np.max()).sum()),
            "c1": float(k[1]), "r_half": int(below[0]) if len(below) else None,
            "max_abs_beyond_71": float(np.abs(k[72:]).max()),
            "far_sd": float(far.std()),
            "weighted_mean_abs_n": float((a_np * absn).sum()),
            "r_eff": float(math.sqrt(2.0 / (a_np * ((2 * math.pi * absn / NPOS) ** 2)
                                             ).sum())) if (a_np * absn).sum() > 0 else None,
            "kernel_axis_mean": k.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r", type=float, default=16.0)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5])
    ap.add_argument("--flat_radius", type=int, default=40)
    ap.add_argument("--menus", nargs="+", default=["draw", "flat"])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--n_pairs", type=int, default=1_000_000)
    ap.add_argument("--n_pos", type=int, default=60_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    M_ = menus(a.r, a.flat_radius)
    results = {}
    for name in a.menus:
        n = M_[name]
        print(f"\n=== menu {name}: {len(n)} frequencies, D = {2 * len(n)} ===", flush=True)
        G, b, tt, M = stats_for(n, dev, a.r, a.n_pairs, a.n_pos)
        for lam in a.lams:
            w, mse, rate = optimise(G, b, tt, M, lam, a.eps, a.iters, dev)
            d = describe(w, n, a.r, dev)
            d.update(lam=lam, mse=mse, rate=rate)
            results[f"{name}|{lam:g}"] = d
            np.save(os.path.join(a.out, f"weights_{name}_lam{lam:g}.npy"), w.cpu().numpy())
            print(f"lam {lam:<7g} mse {mse:.6f} rate {rate:.4f}  waves(PR) "
                  f"{d['participation']:7.1f}  weighted {d['n_weighted']:5d}  "
                  f"C(1) {d['c1']:.3f}  r_half {d['r_half']}  "
                  f"max>71 {d['max_abs_beyond_71']:.3f}  far_sd {d['far_sd']:.4f}  "
                  f"r_eff {d['r_eff']:.1f}", flush=True)
        # reference: equal weights on this menu
        eq = torch.full((len(n),), 1.0 / len(n), dtype=torch.float64, device=dev)
        d = describe(eq, n, a.r, dev)
        mse = float(eq @ G @ eq - 2 * eq @ b + tt)
        print(f"equal    mse {mse:.6f}  waves(PR) {d['participation']:.1f}  C(1) "
              f"{d['c1']:.3f}  r_half {d['r_half']}  max>71 "
              f"{d['max_abs_beyond_71']:.3f}  far_sd {d['far_sd']:.4f}", flush=True)
        results[f"{name}|equal"] = dict(d, mse=mse)
    with open(os.path.join(a.out, "rate_optimum.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
