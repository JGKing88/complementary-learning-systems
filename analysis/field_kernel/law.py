"""Does one variable predict the direction error across fast share, fast scale and unit count?

    python -m analysis.field_kernel.law [SWEEP_DIR]

Candidate: the readout's noise-to-signal ratio ``x = sqrt(v / D) / slope``, with ``v`` the code
speed ``E |z(x+1) - z(x)|^2``, ``D`` the unit count and ``slope`` the Pearson kernel's fall per cell
at 12 cells. Fits ``log err1 = a + b log x`` on rows with error under 40 degrees (above that the
angle saturates toward 90) and reports the same fit against field width.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np

RES = "/orcd/pool/003/jackking/cls_runs/results/field_kernel/sweep"


def rows(d: str = RES, kind: str | None = None) -> list[dict]:
    out = []
    for f in sorted(glob.glob(f"{d}/*.json")):
        out += [r for r in json.load(open(f)) if kind is None or r["kind"] == kind]
    return out


def fit(x, y):
    A = np.stack([np.ones_like(x), np.log(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    pred = A @ coef
    r2 = 1 - ((np.log(y) - pred) ** 2).sum() / ((np.log(y) - np.log(y).mean()) ** 2).sum()
    return float(np.exp(coef[0])), float(coef[1]), float(r2)


def main() -> None:
    d = sys.argv[1] if len(sys.argv) > 1 else RES
    for kind in ("gauss", "grid"):
        R = rows(d, kind)
        if not R:
            continue
        D = np.array([r["D"] for r in R], float)
        v = np.array([r["speed"] for r in R])
        s = np.array([r["slope12"] for r in R])
        W = np.array([r["W_A"] for r in R])
        print(f"\n{kind}: {len(R)} codes, D in {sorted(set(D.astype(int)))}")
        for step in (1, 2, 4):
            e = np.array([r[f"err{step}"] for r in R])
            ok = e < 40
            for name, x in (("sqrt(v/D)/slope", np.sqrt(v / D) / s), ("1/(W sqrt(D))", 1 / (W * np.sqrt(D))),
                            ("sqrt(v/D)", np.sqrt(v / D))):
                c, b, r2 = fit(x[ok], e[ok])
                print(f"  step {step}: err = {c:9.1f} * [{name}]^{b:.2f}   R2 {r2:.3f}  (n {ok.sum()})")


if __name__ == "__main__":
    main()
