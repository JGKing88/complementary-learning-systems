"""Screen w55 -- the first wave selected on the nav objective -- four seeds each.

Every arm is Level 6 with one thing moved, so `w49 eps1_rate0.5` is the control
and it is measured here on the same draw rather than quoted. `att0.5` (w52) is
the incumbent best and is included for the same reason.

One seed per arm has now evaporated five times in this campaign, so all four are
reported and the median is what ranks them.
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
GAINS = [100, 300]

GROUPS = [
    ("att0.5 (best)", f"{S}/w52_attract_fwhm/*_att0.5_seed=*"),
    ("att1", f"{S}/w52_attract_fwhm/*_att1_seed=*"),
    ("L6 att2 (control)", f"{S}/w49_g100_knee/*eps1_rate0.5_seed=*"),
    ("w55 att0.25", f"{S}/w55_nav_objective/*_att0.25_seed=*"),
    ("w55 rep0.1", f"{S}/w55_nav_objective/*_rep0.1_seed=*"),
    ("w55 sm30", f"{S}/w55_nav_objective/*_sm30_seed=*"),
    ("w55 g300_eps1", f"{S}/w55_nav_objective/*_g300_eps1_seed=*"),
    ("w55 g300_eps2", f"{S}/w55_nav_objective/*_g300_eps2_seed=*"),
    ("w55 g300_rate.25", f"{S}/w55_nav_objective/*_g300_rate.25_seed=*"),
]

rng = np.random.RandomState(7)
n = 6000
px, py = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
qx, qy = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
far = np.hypot(px - qx, py - qy) > 200
fx, fy, gx, gy = px[far], py[far], qx[far], qy[far]
nref = 250
rx, ry = rng.randint(60, NPOS - 60, nref), rng.randint(60, NPOS - 60, nref)
offs = np.arange(1, 31)


def measure(ck, g):
    enc, cfg, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
    enc.gain = float(g)
    field = Field(enc, list(cfg.lambdas), fwhm, g, NPOS)
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
    res = [offs[b[0]] if (b := np.flatnonzero(r < 0.9)).size else offs[-1] + 1
           for r in prof]
    return alias, float(np.median(res))


for g in GAINS:
    print(f"\n=== gain {g}, beta = gain ===")
    print(f"{'arm':20s}{'n':>3s}   "
          + "  ".join(f"{'s' + str(s):>7s}" for s in (42, 43, 44, 45))
          + f"{'median':>9s}{'res90':>7s}")
    print("-" * 72)
    rows = []
    for lab, pat in GROUPS:
        vals, res = {}, []
        for d in sorted(glob.glob(pat)):
            ck = os.path.join(d, "encoder_final.pt")
            if not os.path.exists(ck):
                continue
            a, r = measure(ck, g)
            vals[int(d.rsplit("=", 1)[1])] = a
            res.append(r)
        if not vals:
            print(f"{lab:20s}  none found")
            continue
        rows.append((np.median(list(vals.values())), lab, vals, res))
    for med, lab, vals, res in sorted(rows):
        cells = "  ".join(f"{vals[s]:7.4f}" if s in vals else f"{'-':>7s}"
                          for s in (42, 43, 44, 45))
        print(f"{lab:20s}{len(vals):3d}   {cells}{med:9.4f}"
              f"{np.median(res):7.1f}")

print("\nLower alias is better. res90 is a floor near 4-5, not a maximand.")
