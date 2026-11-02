"""Step 2 of Sec 19: the slowness readout. Does slow feature analysis on the
fixed random layer find a usable code without being shown the ideal?

Same environments and random layer as ``rp_readout_check``. The statistics are
what an animal living in those environments would collect, each environment
weighted by its area:

* ``B``: covariance of the hidden layer over every visited cell, pooled
  across environments (the decorrelation term).
* ``S``: mean of (h(p + e) - h(p))(h(p + e) - h(p))^T over unit steps e in x
  and y that stay inside one environment (the slowness term).

SFA solves ``S v = lam B v``: whiten ``B`` (keeping eigenvalues above 1e-9 of
the largest), then diagonalise the whitened ``S``. Output ``i`` has unit
variance over the visited cells and slowness ``lam_i``. For a wave of
wavevector ``w`` that slowness is ``|w|^2 / 2``.

Two readouts from the same features, each then randomly projected to 1024
dims (when wider) and normalised:

* ``topD``: the ``D`` slowest features, equal weight.
* ``gaussR``: every feature, weighted ``exp(-lam_i R^2 / 2)``. For pure waves
  that is the Gaussian kernel of width R exactly, so R = 16 targets the ideal.

Also reported: how many features are slower than ``1/r^2`` against the number
a complete set of waves would give, ``lam 1716^2 / pi`` (the count the ideal
needs), for r = 64, 32, 16, 8.

    python -m analysis.hopfield_probe.rp_sfa_check --fwhm 0.10 --H 8192 --variants hard50 --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .rp_encoder import RPEncoder, grid_codes_torch, make_random_layer, save_rp_encoder
from .rp_readout_check import (LAMBDAS, NPOS, VARIANTS, Ideal, grid_aliases, kernel_metrics,
                               make_envs, position_sets)

class _RFFLayer(torch.nn.Module):
    """Positive control: pure waves (the ideal code at a narrow ``r``, H/2
    frequencies) in place of the random layer. SFA + ``gauss16`` should then
    give a near-Gaussian kernel of width sqrt(16^2 + r^2) and no aliases."""

    def __init__(self, r: float, H: int, device):
        super().__init__()
        from .ideal_encoder import IdealEncoder
        self.enc = IdealEncoder(r=r, n_freq=H // 2, seed=1).to(device)
        self.H = 2 * (H // 2)

    def forward(self, codes):
        return self.enc(codes.double()).float() * math.sqrt(self.H / 2)


TOPD = (256, 1024, 4096)
GAUSS_R = (8.0, 16.0, 24.0)
COUNT_R = (64, 32, 16, 8)


def env_stats(layer, rects, fwhm, device):
    """Area-weighted mean, covariance and within-env step covariance of h."""
    H = layer.H
    s1 = torch.zeros(H, dtype=torch.float64, device=device)
    B = torch.zeros(H, H, dtype=torch.float64, device=device)
    S = torch.zeros(H, H, dtype=torch.float64, device=device)
    n, m = 0, 0
    rows = max(1, int(2.5e8 // (H * 100)))          # env rows per chunk
    for x0, y0, s in rects:
        for r0 in range(0, s, rows):
            r1 = min(s, r0 + rows + 1)               # one overlap row for x steps
            gx, gy = torch.meshgrid(torch.arange(x0 + r0, x0 + r1, device=device),
                                    torch.arange(y0, y0 + s, device=device), indexing="ij")
            h = layer(grid_codes_torch(gx.reshape(-1), gy.reshape(-1), LAMBDAS, fwhm))
            h = h.double().reshape(r1 - r0, s, H)
            own = h[: min(rows, s - r0)].reshape(-1, H)          # rows counted once
            s1 += own.sum(0)
            B.addmm_(own.T, own)
            n += len(own)
            dy = (own.reshape(-1, s, H)[:, 1:] - own.reshape(-1, s, H)[:, :-1]).reshape(-1, H)
            S.addmm_(dy.T, dy)
            m += len(dy)
            if r1 - r0 > 1:
                # steps (j, j+1) for j in [r0, r1 - 2]: the next chunk starts at
                # r0 + rows = r1 - 1, so each x step is counted exactly once
                dx = (h[1:] - h[:-1]).reshape(-1, H)
                S.addmm_(dx.T, dx)
                m += len(dx)
            del h, own, dy
    hbar = s1 / n
    B = B / n - torch.outer(hbar, hbar)
    S = S / m
    return hbar, B, S, n, m


def sfa(B, S, rel_floor=1e-9):
    """Slowness ``lam`` (ascending) and projection ``P`` (H, k): y = (h - hbar) @ P."""
    eb, Ub = torch.linalg.eigh(B)
    keep = eb > rel_floor * eb.max()
    Wb = Ub[:, keep] / eb[keep].sqrt()
    del Ub
    M = Wb.T @ S @ Wb
    lam, V = torch.linalg.eigh((M + M.T) / 2)
    return lam, Wb @ V, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwhm", type=float, required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--variants", default="hard50")
    ap.add_argument("--coverages", default="0.01,0.05,0.2")
    ap.add_argument("--hidden_seed", type=int, default=0)
    ap.add_argument("--env_seed", type=int, default=0)
    ap.add_argument("--save", default="", help="comma list of readouts to save, e.g. gauss16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    ideal = Ideal(16.0, dev)
    aliases, _ = grid_aliases(a.fwhm)
    save = set(filter(None, a.save.split(",")))
    for cov in [float(c) for c in a.coverages.split(",")]:
        mask, rects = make_envs(cov, a.env_seed)
        vis, unv, far, far_frac = position_sets(mask, a.env_seed)
        sets = {"visited": vis[:20_000], "unvisited": unv}
        if len(far) >= 1000:
            sets["far"] = far
        ideal_m = {k: kernel_metrics(ideal, p, aliases, dev) for k, p in sets.items()}
        for name in a.variants.split(","):
            tag = f"fwhm{a.fwhm:g}_H{a.H}_{name}_cov{cov:g}"
            path = os.path.join(a.out, f"{tag}.json")
            if os.path.exists(path):
                print("skip", tag, flush=True)
                continue
            t0 = time.time()
            if name.startswith("rff"):
                layer = _RFFLayer(float(name[3:]), a.H, dev)
            else:
                nonlin, gain, active = VARIANTS[name]
                layer = make_random_layer(a.H, seed=a.hidden_seed, nonlin=nonlin, gain=gain,
                                          active=active, fwhm_ratio=a.fwhm, device=dev)
            with torch.no_grad():
                hbar, B, S, n, m = env_stats(layer, rects, a.fwhm, dev)
                lam, P, k = sfa(B, S)
                del B, S
                lam_np = lam.cpu().numpy()
                counts = {str(r): {"sfa": int((lam_np < 1 / r**2).sum()),
                                   "waves": float(NPOS**2 / (math.pi * r**2))} for r in COUNT_R}
                res = {"tag": tag, "fwhm": a.fwhm, "H": a.H, "variant": name, "coverage": cov,
                       "coverage_actual": float(mask.mean()), "n_envs": len(rects),
                       "n_cells": n, "n_steps": m, "rank_B": k, "far_frac": far_frac,
                       "lam_quantiles": {str(q): float(np.quantile(lam_np, q))
                                         for q in (0.0, 0.001, 0.01, 0.05, 0.25, 0.5)},
                       "lam_first": lam_np[:16].tolist(), "slow_counts": counts,
                       "ideal": ideal_m, "readouts": {}}
                g = torch.Generator(device="cpu").manual_seed(1234)
                readouts = {f"top{D}": torch.ones(D, dtype=torch.float64, device=dev)
                            for D in TOPD if D <= P.shape[1]}
                for R in GAUSS_R:
                    # shifted by the slowest lam so the weights never underflow;
                    # an overall scale does not change the (normalised) code
                    readouts[f"gauss{R:g}"] = torch.exp(-(lam - lam[0]).clamp_min(0) * R**2 / 2)
                for rname, w in readouts.items():
                    Pw = P[:, :len(w)] * w
                    if Pw.shape[1] > 1024:
                        proj = torch.randn(Pw.shape[1], 1024, generator=g,
                                           dtype=torch.float64).to(dev) / math.sqrt(1024)
                        Pw = Pw @ proj
                    enc = RPEncoder(layer, hbar.float(), Pw.float(), fwhm_ratio=a.fwhm,
                                    meta={"coverage": cov, "n_envs": len(rects), "variant": name,
                                          "readout": f"sfa_{rname}"})

                    def unit(x, y, enc=enc):
                        return enc(grid_codes_torch(x, y, LAMBDAS, a.fwhm))

                    res["readouts"][rname] = {
                        "eff_n": float(w.sum() ** 2 / (w ** 2).sum()),
                        **{k2: kernel_metrics(unit, p, aliases, dev) for k2, p in sets.items()}}
                    if rname in save and not name.startswith("rff"):
                        save_rp_encoder(enc.cpu(), os.path.join(a.out, f"{tag}_{rname}.pt"))
                        enc.to(dev)
                    u = res["readouts"][rname]["unvisited"]
                    print(f"sfa {tag} {rname}: unv C1 {u['C1']:.3f} r_half {u['r_half']} "
                          f"far_sd {u['far_sd']:.3f} alias_max {u['alias_max']:.3f}", flush=True)
            res["seconds"] = time.time() - t0
            print(f"sfa {tag}: rank {k} slow counts "
                  + " ".join(f"r{r}:{c['sfa']}/{c['waves']:.0f}" for r, c in counts.items())
                  + f" | {res['seconds']:.0f}s", flush=True)
            with open(path, "w") as f:
                json.dump(res, f, indent=1)
            del layer, P, lam
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
