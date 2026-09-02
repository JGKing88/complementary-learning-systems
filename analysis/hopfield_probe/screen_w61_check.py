"""Screen w61 (0.75%) and w60 (1.25%) against the rest of the ladder.

Two questions, both with predictions on record before this ran:

  * the attract optimum at 1.25% -- predicted **2.0**, continuing 0.5 / 0.5 /
    1.0 at 10 / 5 / 2.5%. 1.0 means the trend saturated; 4.0 means it is
    steeper than linear in log-coverage.
  * whether reach breaks from flat here. Coverage has been buying capacity, not
    reach, down to 2.5%; at 1.25% capacity should run out.

Gain is swept per arm to land res90 nearest 7, so arms are compared at matched
chart length and the difference left is the alias rate. `res90 max` is reported
because at low coverage an arm may not reach 7 at any gain, and a nearest-fit
would hide that.
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
GAINS = [3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
TARGET = 7.0

GROUPS = [
    ("10%  att0.5", f"{S}/w52_attract_fwhm/*_att0.5_seed=*"),
    ("5%   half_a0.5", f"{S}/w57_cov5/*_half_a0.5_seed=*"),
    ("2.5% q_a1", f"{S}/w58_cov2.5/*_q_a1_seed=*"),
    ("2.5% q_a2", f"{S}/w59_cov2.5_att_hi/*_q_a2_seed=*"),
    ("1.25% x_a1", f"{S}/w60_cov1.25/*_x_a1_seed=*"),
    ("1.25% x_a2", f"{S}/w60_cov1.25/*_x_a2_seed=*"),
    ("1.25% x_a4", f"{S}/w60_cov1.25/*_x_a4_seed=*"),
    ("1.25% sm35x_a2", f"{S}/w60_cov1.25/*_sm35x_a2_seed=*"),
    ("0.75% y27_a2", f"{S}/w61_cov0.75/*_y27_a2_seed=*"),
    ("0.75% y27_a4", f"{S}/w61_cov0.75/*_y27_a4_seed=*"),
    ("0.75% y35_a2", f"{S}/w61_cov0.75/*_y35_a2_seed=*"),
    ("0.75% y50_a2", f"{S}/w61_cov0.75/*_y50_a2_seed=*"),
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
    Zc = R - R.mean(0)
    ev = np.clip(np.linalg.eigvalsh(np.cov(Zc, rowvar=False)), 0, None)
    d_eff = float(ev.sum() ** 2 / (ev ** 2).sum())
    prof = np.empty((nref, len(offs)))
    for i, o in enumerate(offs):
        Q = field.encode(np.clip(rx + o, 0, NPOS - 1), ry).astype(np.float64)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        prof[:, i] = (R * Q).sum(1)
    res = float(np.median(
        [offs[b[0]] if (b := np.flatnonzero(r < 0.9)).size else offs[-1] + 1
         for r in prof]))
    return alias, res, d_eff


rows = []
for lab, pat in GROUPS:
    per = {g: ([], [], []) for g in GAINS}
    for d in sorted(glob.glob(pat)):
        ck = os.path.join(d, "encoder_final.pt")
        if not os.path.exists(ck):
            continue
        enc, cfg, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
        field = Field(enc, list(cfg.lambdas), fwhm, own, NPOS)
        for g in GAINS:
            a, r, de = at_gain(field, enc, g)
            per[g][0].append(a)
            per[g][1].append(r)
            per[g][2].append(de)
    if not per[GAINS[0]][0]:
        print(f"  {lab}: none found", file=sys.stderr)
        continue
    res_by_gain = {g: np.median(per[g][1]) for g in GAINS}
    best = min(GAINS, key=lambda g: abs(res_by_gain[g] - TARGET))
    rows.append((float(np.median(per[best][0])), lab, best,
                 float(res_by_gain[best]), max(res_by_gain.values()),
                 float(np.median(per[best][2])), len(per[best][0])))

print(f"Gain per arm chosen to land res90 nearest {TARGET:g}. Four seeds each.\n")
print(f"{'arm':18s}{'gain':>6s}{'res90':>7s}{'res90max':>10s}"
      f"{'alias':>9s}{'d_eff':>8s}{'n':>4s}")
print("-" * 62)
for alias, lab, g, res, rmax, de, k in sorted(rows):
    flag = "  <- cannot reach 7" if rmax < TARGET - 0.5 else ""
    print(f"{lab:18s}{g:6d}{res:7.1f}{rmax:10.1f}{alias:9.4f}{de:8.1f}"
          f"{k:4d}{flag}")
print("\nPrediction on record: the 1.25% optimum is x_a2. Lower alias is better.")
