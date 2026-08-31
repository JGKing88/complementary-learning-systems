"""Does fwhm 0.5's advantage over Level 6 survive four seeds?

The pool screen runs one seed per arm and this campaign has had two-seed results
reverse twice, so a 15% edge at seed 42 is a shortlist entry, not a finding.

`w52_attract_fwhm/fwhm0.5` is Level 6 with `fwhm_ratio` 0.25 -> 0.5 and nothing
else, and Sec 6.9 of the radius doc found the two TIE on `r_min` (9.0 against
9.5, inside spread). So if the alias-rate edge is real it is a knob that is free
on the old metric and better on the new one.

`w51_steps` is here for the same reason: 3x training steps is a measured null
for `r_min` (Sec 6.11b) and looked 14% better on the alias rate.
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
    ("L6 (fwhm0.25)", f"{S}/w49_g100_knee/*eps1_rate0.5_seed=*"),
    ("fwhm0.5", f"{S}/w52_attract_fwhm/*fwhm0.5_seed=*"),
    ("fwhm0.15", f"{S}/w52_attract_fwhm/*fwhm0.15_seed=*"),
    ("att1", f"{S}/w52_attract_fwhm/*att1_seed=*"),
    ("steps3x", f"{S}/w51_steps/*steps3x_seed=*"),
    ("steps2x", f"{S}/w51_steps/*steps2x_seed=*"),
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
    print(f"{'arm':16s}{'n':>3s}   " + "  ".join(f"{'s' + str(s):>7s}"
                                                 for s in (42, 43, 44, 45))
          + f"{'median':>9s}{'res90':>7s}")
    print("-" * 68)
    for lab, pat in GROUPS:
        vals, res = {}, []
        for d in sorted(glob.glob(pat)):
            ck = os.path.join(d, "encoder_final.pt")
            if not os.path.exists(ck):
                continue
            seed = int(d.rsplit("=", 1)[1])
            a, r = measure(ck, g)
            vals[seed] = a
            res.append(r)
        if not vals:
            print(f"{lab:16s}  none found")
            continue
        cells = "  ".join(f"{vals[s]:7.4f}" if s in vals else f"{'-':>7s}"
                          for s in (42, 43, 44, 45))
        print(f"{lab:16s}{len(vals):3d}   {cells}"
              f"{np.median(list(vals.values())):9.4f}"
              f"{np.median(res):7.1f}")

print("\nLower alias is better; res90 is the floor, not a thing to maximise.")
