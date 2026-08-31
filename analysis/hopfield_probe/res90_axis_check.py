"""res90 and alias rate for the arms at their own best gains.

The claim is that there is ONE axis -- chart length -- and that reach peaks near
res90 7 regardless of which combination of training knob and inference gain gets
you there. This checks the two arms that reached ~0.99 by different routes.
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

ARMS = [
    ("att2 (L6) @g100", f"{S}/w49_g100_knee/*eps1_rate0.5_seed=*", 100),
    ("att2 (L6) @g300", f"{S}/w49_g100_knee/*eps1_rate0.5_seed=*", 300),
    ("att1 @g100", f"{S}/w52_attract_fwhm/*_att1_seed=*", 100),
    ("att1 @g300", f"{S}/w52_attract_fwhm/*_att1_seed=*", 300),
    ("att0.5 @g100", f"{S}/w52_attract_fwhm/*_att0.5_seed=*", 100),
    ("att0.5 @g300", f"{S}/w52_attract_fwhm/*_att0.5_seed=*", 300),
    ("att0.25 @g30", f"{S}/w55_nav_objective/*_att0.25_seed=*", 30),
    ("att0.25 @g100", f"{S}/w55_nav_objective/*_att0.25_seed=*", 100),
    ("sm30 @g100", f"{S}/w55_nav_objective/*_sm30_seed=*", 100),
]

# measured continuous reach, probe seed 0, four encoder seeds, beta = gain
REACH = {
    "att2 (L6) @g100": 0.931, "att2 (L6) @g300": 0.971,
    "att1 @g100": 0.972, "att1 @g300": 0.981,
    "att0.5 @g100": 0.987, "att0.5 @g300": 0.977,
    "att0.25 @g30": 0.990, "att0.25 @g100": 0.974,
}

rng = np.random.RandomState(7)
n = 6000
px, py = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
qx, qy = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
far = np.hypot(px - qx, py - qy) > 200
fx, fy, gx, gy = px[far], py[far], qx[far], qy[far]
nref = 250
rx, ry = rng.randint(60, NPOS - 60, nref), rng.randint(60, NPOS - 60, nref)
offs = np.arange(1, 31)

print(f"{'arm':20s}{'gain':>6s}{'alias':>9s}{'res90':>7s}{'cont':>8s}")
print("-" * 50)
rows = []
for lab, pat, g in ARMS:
    al, rs = [], []
    for d in sorted(glob.glob(pat)):
        ck = os.path.join(d, "encoder_final.pt")
        if not os.path.exists(ck):
            continue
        enc, cfg, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
        enc.gain = float(g)
        field = Field(enc, list(cfg.lambdas), fwhm, g, NPOS)
        A = field.encode(fx, fy).astype(np.float64)
        B = field.encode(gx, gy).astype(np.float64)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        al.append(float(((A * B).sum(1) > 0.25).mean()))
        R = field.encode(rx, ry).astype(np.float64)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        prof = np.empty((nref, len(offs)))
        for i, o in enumerate(offs):
            Q = field.encode(np.clip(rx + o, 0, NPOS - 1), ry)
            Q = Q.astype(np.float64)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            prof[:, i] = (R * Q).sum(1)
        rs.append(float(np.median(
            [offs[b[0]] if (b := np.flatnonzero(r < 0.9)).size else offs[-1] + 1
             for r in prof])))
    if not al:
        continue
    rows.append((np.median(rs), lab, g, np.median(al), REACH.get(lab)))

for res, lab, g, alias, reach in sorted(rows, reverse=True):
    rc = f"{reach:8.3f}" if reach is not None else f"{'-':>8s}"
    print(f"{lab:20s}{g:6d}{alias:9.4f}{res:7.1f}{rc}")

print("\nSorted by res90. If one axis explains it, reach should rise to a peak")
print("and fall, independent of which knob produced the chart length.")
