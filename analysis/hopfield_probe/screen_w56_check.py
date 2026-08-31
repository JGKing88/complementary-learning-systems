"""Screen w56, and report the gain each arm needs to land at res90 ~7.

Sec 10.9-10.10 leave a two-stage rule: get the alias rate under ~0.007, then
trim (attract, inference gain) to land res90 at 6.5-7, which is where reach
peaks. Probing every arm at one fixed gain would put each at a different and
mostly wrong chart length, so this sweeps gain per arm and reports the setting
nearest the target -- which also makes the arms comparable at matched res90, so
the remaining difference is the alias rate rather than the gain that was picked.

The incumbents are measured here on the same draw rather than quoted.
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
    ("att0.5 @own", f"{S}/w52_attract_fwhm/*_att0.5_seed=*"),
    ("att0.25", f"{S}/w55_nav_objective/*_att0.25_seed=*"),
    ("sm30", f"{S}/w55_nav_objective/*_sm30_seed=*"),
    ("L6 att2", f"{S}/w49_g100_knee/*eps1_rate0.5_seed=*"),
    ("w56 a0.5_sm30", f"{S}/w56_nav_combos/*_a0.5_sm30_seed=*"),
    ("w56 a0.75_sm30", f"{S}/w56_nav_combos/*_a0.75_sm30_seed=*"),
    ("w56 a1_sm30", f"{S}/w56_nav_combos/*_a1_sm30_seed=*"),
    ("w56 a0.75", f"{S}/w56_nav_combos/*_a0.75_seed=*"),
    ("w56 sm20", f"{S}/w56_nav_combos/*_sm20_seed=*"),
    ("w56 a0.5_rate1", f"{S}/w56_nav_combos/*_a0.5_rate1_seed=*"),
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
    return alias, res


rows = []
for lab, pat in GROUPS:
    per_gain = {g: ([], []) for g in GAINS}
    for d in sorted(glob.glob(pat)):
        ck = os.path.join(d, "encoder_final.pt")
        if not os.path.exists(ck):
            continue
        enc, cfg, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
        field = Field(enc, list(cfg.lambdas), fwhm, own, NPOS)
        for g in GAINS:
            a, r = at_gain(field, enc, g)
            per_gain[g][0].append(a)
            per_gain[g][1].append(r)
    if not per_gain[GAINS[0]][0]:
        print(f"  {lab}: none found", file=sys.stderr)
        continue
    best = min(GAINS, key=lambda g: abs(np.median(per_gain[g][1]) - TARGET))
    rows.append((float(np.median(per_gain[best][0])), lab, best,
                 float(np.median(per_gain[best][1])),
                 len(per_gain[best][0])))

print(f"Gain chosen per arm to land res90 nearest {TARGET:g}. "
      f"Four encoder seeds each.\n")
print(f"{'arm':18s}{'gain':>6s}{'res90':>7s}{'alias':>9s}{'n':>4s}")
print("-" * 46)
for alias, lab, g, res, k in sorted(rows):
    print(f"{lab:18s}{g:6d}{res:7.1f}{alias:9.4f}{k:4d}")
print("\nAt matched res90 the remaining difference is the alias rate, which is")
print("what the two-stage rule says to minimise. Lower is better.")
