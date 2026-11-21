"""Field width, kernel shoulder, code speed and direction error over a grid of two-scale codes.

    python -m analysis.field_kernel.sweep --n_freq 512 --out DIR [--kind gauss|grid]

For each (fast share, fast scale) at ``n_freq`` waves (D = 2 n_freq units, 10% active, slow scale
r = 23.2): loop statistics (W_A, Pearson kernel at 8, 16 and 32 cells), the code speed
``v = E |z(x+1) - z(x)|^2`` of the unit-length code, and ``direction.run`` at readout steps 1, 2
and 4 with K = 5 stored goals. One JSON per n_freq and kind.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.coding_comparison import measures as CM
from analysis.field_kernel import direction as DR
from analysis.field_kernel.measure import NPOS, build, loops
from analysis.ideal_family import measure as FM

GAUSS = [(0.0, 2.0)] + [(a, rs) for rs in (1.0, 2.0, 3.0) for a in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)]
GRID = [(a, 0.0) for a in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)]


def spec_of(kind: str, a: float, rs: float, n_freq: int, r: float, active: float) -> str:
    s = f"ideal:r={r},n_freq={n_freq},seed=0,relu_mix=1,active={active}"
    if a > 0:
        s += f",fast={a},r_fast={rs}" if kind == "gauss" else f",grid_fast={a}"
    return s


def loop_stats(spec: str) -> dict:
    enc = build(spec)
    W, prof, speed = [], [], []
    for xs, ys in loops(0, 2):
        Z = enc.closed_form(xs, ys)                      # (positions, units), unit rows
        speed.append(((np.roll(Z, -1, axis=0) - Z) ** 2).sum(1).mean())
        R = Z.T
        s, _ = CM.lag_profile_1d(R, "pearson", circular=True, max_lag=64)
        prof.append(s)
        W.append(FM.loop_fields(R).length.astype(np.float64))
    W = np.concatenate(W)
    p = np.mean(prof, axis=0)
    return {"W_A": float(W.mean()), "W_median": float(np.median(W)),
            "fields_per_loop": float(W.size / (4 * enc.out_dim)),
            "K8": float(p[8]), "K16": float(p[16]), "K32": float(p[32]),
            "slope12": float((p[10] - p[14]) / 4), "speed": float(np.mean(speed))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_freq", type=int, required=True)
    ap.add_argument("--kind", default="gauss", choices=["gauss", "grid"])
    ap.add_argument("--r", type=float, default=23.2)
    ap.add_argument("--active", type=float, default=0.1)
    ap.add_argument("--trials", type=int, default=1500)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for fa, rs in (GAUSS if a.kind == "gauss" else GRID):
        spec = spec_of(a.kind, fa, rs, a.n_freq, a.r, a.active)
        row = {"spec": spec, "kind": a.kind, "alpha": fa, "r_fast": rs, "n_freq": a.n_freq,
               "D": 2 * a.n_freq, **loop_stats(spec)}
        for s in a.steps:
            o = DR.run(spec, step=s, trials=a.trials, walk=(s <= 4))
            row[f"err{s}"], row[f"acc{s}"], row[f"reach{s}"] = o["err_deg"], o["acc45"], o.get("reach")
        rows.append(row)
        print(json.dumps(row), flush=True)
        with open(f"{a.out}/{a.kind}_M{a.n_freq}_r{a.r}_p{a.active}.json", "w") as fh:
            json.dump(rows, fh)


if __name__ == "__main__":
    main()
