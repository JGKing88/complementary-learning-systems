"""Screen w57 (5% coverage) at matched res90, four seeds per arm.

Same protocol as `screen_w56_check.py`: probing every arm at one fixed gain
would put each at a different and mostly wrong chart length, so gain is swept
per arm and reported at the setting that lands res90 nearest 7 -- which is where
reach peaks (Sec 10.9). At matched res90 the remaining difference between arms
is the alias rate, which is what the two-stage rule says to minimise.

`w52_attract_fwhm/*_att0.5` is the 10% incumbent and the reference row: the
question this wave asks is how much of its 0.987 survives halving the coverage.

Prediction recorded before the results (Sec 10.11): the attract optimum should
move DOWN from 0.5, because attract and the coding-rate term trade against one
d_eff budget and lower coverage gives the spread term fewer distinct arena
positions to work from. So half_a0.25 should beat half_a0.5 should beat half_a1.
If 0.5 stays optimal, the two act independently.
"""
from __future__ import annotations

import glob
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
NPOS = 1716
GAINS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
TARGET = 7.0

GROUPS = [
    ("att0.5 @10% (ref)", f"{S}/w52_attract_fwhm/*_att0.5_seed=*"),
    ("w57 half_a0.25", f"{S}/w57_cov5/*_half_a0.25_seed=*"),
    ("w57 half_a0.5", f"{S}/w57_cov5/*_half_a0.5_seed=*"),
    ("w57 half_a1", f"{S}/w57_cov5/*_half_a1_seed=*"),
    ("w57 sm35_a0.5", f"{S}/w57_cov5/*_sm35_a0.5_seed=*"),
    ("w57 sm70_a0.5", f"{S}/w57_cov5/*_sm70_a0.5_seed=*"),
    ("w57 half_rate1", f"{S}/w57_cov5/*_half_rate1_seed=*"),
]

rng = np.random.RandomState(7)
n = 6000
px, py = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
qx, qy = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
far = np.hypot(px - qx, py - qy) > 200
fx, fy, gx, gy = px[far], py[far], qx[far], qy[far]
nref = 200
rx, ry = rng.randint(60, NPOS - 60, nref), rng.randint(60, NPOS - 60, nref)
offs = np.arange(1, 31)


def at_gain(field, enc, g):
    """(alias rate, res90, effective dimension) at inference gain g."""
    enc.gain = float(g)
    A = field.encode(fx, fy).astype(np.float64)
    B = field.encode(gx, gy).astype(np.float64)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    alias = float(((A * B).sum(1) > 0.25).mean())

    R = field.encode(rx, ry).astype(np.float64)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    prof = np.empty((nref, len(offs)))
    for i, o in enumerate(offs):
        Q = field.encode(np.clip(rx + o, 0, NPOS - 1), ry).astype(np.float64)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        prof[:, i] = (R * Q).sum(1)
    res = float(np.median(
        [offs[b[0]] if (b := np.flatnonzero(r < 0.9)).size else offs[-1] + 1
         for r in prof]))

    # Participation ratio of the code covariance -- Sec 10.11's variable.
    C = np.cov(A - A.mean(0), rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    deff = float(ev.sum() ** 2 / (ev ** 2).sum())
    return alias, res, deff


rows = []
for lab, pat in GROUPS:
    per_gain = {g: ([], [], []) for g in GAINS}
    for d in sorted(glob.glob(pat)):
        ck = os.path.join(d, "encoder_final.pt")
        if not os.path.exists(ck):
            continue
        enc, cfg, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
        field = Field(enc, list(cfg.lambdas), fwhm, own, NPOS)
        for g in GAINS:
            a, r, dd = at_gain(field, enc, g)
            per_gain[g][0].append(a)
            per_gain[g][1].append(r)
            per_gain[g][2].append(dd)
    if not per_gain[GAINS[0]][0]:
        print(f"  {lab}: none found", file=sys.stderr)
        continue
    best = min(GAINS, key=lambda g: abs(np.median(per_gain[g][1]) - TARGET))
    rows.append((float(np.median(per_gain[best][0])), lab, best,
                 float(np.median(per_gain[best][1])),
                 float(np.median(per_gain[best][2])),
                 len(per_gain[best][0]),
                 [float(x) for x in per_gain[best][0]]))

print(f"Gain chosen per arm to land res90 nearest {TARGET:g}. "
      f"Four encoder seeds each.\n")
print(f"{'arm':20s}{'gain':>6s}{'res90':>7s}{'alias':>9s}{'d_eff':>8s}"
      f"{'n':>3s}   per-seed alias")
print("-" * 88)
for alias, lab, g, res, deff, k, vals in sorted(rows):
    cells = " ".join(f"{v:.4f}" for v in vals)
    print(f"{lab:20s}{g:6d}{res:7.1f}{alias:9.4f}{deff:8.1f}{k:3d}   {cells}")

print("\nAt matched res90 the remaining difference is the alias rate. d_eff is")
print("the participation ratio of the code covariance, Sec 10.11's variable.")
