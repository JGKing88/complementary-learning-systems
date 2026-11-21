"""Sec 20.1: is the slowness gain local and incremental?

One walk inside ONE small square environment (side 20, and 10), reflecting
at the walls, on the exact 400k grid wave network (bump 0.25, r = 16 target).
For each unit, d_t = (C_t+1 - C_t)^2 + (S_t+1 - S_t)^2 at every step.

* locality: the spread of d_t over steps of the same axis, per unit (0 for a
  pure wave: d depends only on the step's direction, not on position).
* trace rule: gain = exp(-r^2 vbar / 2), vbar = mean d_t over the first T steps,
  T = 10, 100, 1000, 10000.
* depression rule: every step w <- w (1 - kappa d_t), kappa = 0.005, w_0 = 1.
  After T steps w = prod(1 - kappa d_t) ~ exp(-kappa T vbar), the Gaussian gain
  with r = sqrt(2 kappa T). Read out at T with sqrt(2 kappa T) = 8, 12, 16, 24.
* the many-environment gains (gwn_check: 8192 steps over 5% coverage) for
  comparison.

Kernel metrics (``World.metrics``) are on cells away from the 5% environment
set; the codes are functions of position only, so where they are measured does
not matter.

    python -m analysis.hopfield_probe.gwn_local_gain --out DIR
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

from .gwn_check import (DEV, LAMBDAS, NU, R, SMOKE, World, _dump, _line, build, l1_out,
                        slowness)
from .gwn_encoder import GWNEncoder, exact_l1, prune, random_triples, unit_waves

KAPPA = 0.005
T_TRACE = (10, 100, 1000, 10000)
R_DEP = (8, 12, 16, 24)


def walk_in_env(x0, y0, s, T, seed):
    """``(T + 1, 2)`` positions of a 4-direction random walk inside one square."""
    rng = np.random.default_rng(seed)
    pos = np.empty((T + 1, 2), dtype=np.int64)
    x, y = x0 + s // 2, y0 + s // 2
    pos[0] = x, y
    moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    for t, k in enumerate(rng.integers(0, 4, T)):
        dx, dy = moves[k]
        if not (x0 <= x + dx < x0 + s):
            dx = -dx
        if not (y0 <= y + dy < y0 + s):
            dy = -dy
        x, y = x + dx, y + dy
        pos[t + 1] = x, y
    return pos


def encoder_from_gains(l1, triples, w, fwhm, meta):
    keep = prune(w)
    enc = GWNEncoder(l1, triples[keep.cpu()], w[keep].cpu(), fwhm_ratio=fwhm, meta=meta)
    w2 = w.double() ** 2
    info = {"units": int(len(triples)), "kept": int(len(keep)),
            "strong": int((w > math.exp(-1)).sum()),
            "n_eff": float(w2[keep].sum() ** 2 / (w2[keep] ** 2).sum())}
    return enc.to(DEV).eval(), info


def run_env(l1, tr, side, W, fwhm, seed, chunk_u=16384, chunk_t=4096):
    T_dep = [int(round(r * r / (2 * KAPPA))) for r in R_DEP]
    T = max(max(T_TRACE), max(T_dep))
    if SMOKE:
        T_dep = [200]; T = 400
    x0, y0 = 700, 900
    pos = walk_in_env(x0, y0, side, T, seed)
    step = pos[1:] - pos[:-1]
    xstep = torch.tensor(step[:, 0] != 0, device=DEV)
    c, s = (t.float() for t in l1_out(l1, pos, fwhm))
    U = len(tr)
    checks_a = [t for t in T_TRACE if t <= T]
    sum_d = {t: torch.zeros(U, dtype=torch.float64, device=DEV) for t in checks_a}
    sum_log = {t: torch.zeros(U, dtype=torch.float64, device=DEV) for t in T_dep}
    spread = torch.zeros(U, dtype=torch.float64, device=DEV)
    for a in range(0, U, chunk_u):
        trc = tr[a:a + chunk_u].to(DEV)
        dmin = [torch.full((len(trc),), float("inf"), device=DEV) for _ in range(2)]
        dmax = [torch.full((len(trc),), -float("inf"), device=DEV) for _ in range(2)]
        cum_d = torch.zeros(len(trc), dtype=torch.float64, device=DEV)
        cum_l = torch.zeros(len(trc), dtype=torch.float64, device=DEV)
        for t0 in range(0, T, chunk_t):
            t1 = min(T, t0 + chunk_t)
            C, S = unit_waves(c[t0:t1 + 1], s[t0:t1 + 1], trc)
            d = (C[1:] - C[:-1]) ** 2 + (S[1:] - S[:-1]) ** 2        # (steps, units)
            xs = xstep[t0:t1]
            for ax, m in ((0, xs), (1, ~xs)):
                if m.any():
                    dmin[ax] = torch.minimum(dmin[ax], d[m].min(0).values)
                    dmax[ax] = torch.maximum(dmax[ax], d[m].max(0).values)
            cd = torch.cumsum(d.double(), 0) + cum_d
            cl = torch.cumsum(torch.log1p(-KAPPA * d.double()), 0) + cum_l
            for t in checks_a:
                if t0 < t <= t1:
                    sum_d[t][a:a + chunk_u] = cd[t - t0 - 1]
            for t in T_dep:
                if t0 < t <= t1:
                    sum_log[t][a:a + chunk_u] = cl[t - t0 - 1]
            cum_d, cum_l = cd[-1], cl[-1]
        spread[a:a + chunk_u] = torch.maximum(dmax[0] - dmin[0], dmax[1] - dmin[1]).double()
    res = {"side": side, "steps": T, "x_steps": int(xstep.sum()), "y_steps": int((~xstep).sum()),
           "locality_max_spread_of_d": float(spread.max()),
           "distinct_cells_visited": int(len({tuple(p) for p in pos.tolist()}))}
    print(f"local env side {side}: {T} steps ({res['x_steps']} x, {res['y_steps']} y), "
          f"max spread of per-step change over positions {res['locality_max_spread_of_d']:.2e}",
          flush=True)
    for t in checks_a:
        vbar = sum_d[t] / t
        vbar = torch.where(vbar < 1e-12, torch.full_like(vbar, float("inf")), vbar)  # the n = 0 unit
        enc, info = encoder_from_gains(l1, tr, torch.exp(-R * R * vbar / 2), fwhm,
                                       {"rule": "trace", "T": t, "side": side})
        m = W.metrics(enc)
        res[f"trace_T{t}"] = {**info, **m}
        _line(f"local side {side} trace T={t}", info, m)
    for t, r in zip(T_dep, R_DEP):
        # a unit that never changed (the n = 0 constant) is dropped, as in the trace rule
        w = torch.where(sum_log[t] == 0, torch.zeros_like(sum_log[t]), torch.exp(sum_log[t]))
        enc, info = encoder_from_gains(l1, tr, w, fwhm,
                                       {"rule": "depression", "T": t, "kappa": KAPPA, "side": side})
        m = W.metrics(enc)
        res[f"depression_T{t}"] = {**info, "r_pred": math.sqrt(2 * KAPPA * t),
                                   "half_width_pred": 1.1774 * math.sqrt(2 * KAPPA * t), **m}
        _line(f"local side {side} depression T={t} (sqrt(2 kappa T) = {math.sqrt(2 * KAPPA * t):.1f}, "
              f"Gaussian half-width {1.1774 * math.sqrt(2 * KAPPA * t):.1f})", info, m)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    fwhm = 0.25
    W = World(fwhm=fwhm)
    l1 = exact_l1(fwhm).to(DEV)
    tr = random_triples(l1, NU, seed=0)
    v, pw = slowness(l1, tr, W.steps, fwhm)
    enc, info = build(l1, tr, v, pw, fwhm, meta={"rule": "many-env baseline"})
    m = W.metrics(enc)
    res = {"many_env": {**info, **m}}
    _line("many-env baseline (8192 steps, 5% coverage)", info, m)
    for side in (20, 10):
        res[f"env{side}"] = run_env(l1, tr, side, W, fwhm, seed=side)
    _dump(os.path.join(a.out, "local_gain.json"), res)


if __name__ == "__main__":
    main()
