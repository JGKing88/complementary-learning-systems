"""A cheap screen for continuous reach, run over the candidate encoder pool.

Dead goals -- environments whose continuous reach collapses to near zero -- are
goals with a co-stored competitor above cos ~0.25, and they account for all of
the reach shortfall. The rate of far-field pairs above that line is therefore a
direct predictor, and unlike the probe it costs seconds per encoder.

res90 is the guard on the other side: the local Gram-Schmidt basis is built from
one-cell neighbours, so it needs the code to stay locally informative, and gain
buys the alias rate by spending exactly that.

Reported at each encoder's own gain and at the gains above it, since raising
inference gain is the one lever that needs no retraining of the encoder.
"""
import glob
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

RUNS = "/orcd/pool/003/jackking/cls_runs"
S = RUNS + "/sweeps"
NPOS = 1716
GAINS = [5, 100, 300]

CANDS = [
    ("v35", RUNS + "/encoders/run_20260422_185816/encoder_best.pt"),
    ("L5 sm50_b4096", f"{S}/w39_batch_pairs/008_sm50_b4096_seed=42"),
    ("L6 eps1_rate0.5", f"{S}/w49_g100_knee/008_eps1_rate0.5_seed=42"),
    ("L7 att16", f"{S}/w53_attract_knee/004_att16_seed=42"),
    ("w54 att32", f"{S}/w54_attract_far/000_att32_seed=42"),
    ("w21 arena_spread x0.5", f"{S}/w21_arena_spread/000_frac0.5_seed=42"),
    ("w21 arena_spread x2", f"{S}/w21_arena_spread/002_frac2_seed=42"),
]
# Every remaining w54 arm, one seed each -- rep0.25 is the arm Sec 4.5 named.
for d in sorted(glob.glob(f"{S}/w54_attract_far/*/")):
    base = os.path.basename(d.rstrip("/"))
    if base.endswith("42") and "att32" not in base:
        CANDS.append(("w54 " + base.split("_", 1)[1].rsplit("_seed", 1)[0], d))
for d in sorted(glob.glob(f"{S}/w53_attract_knee/*/")):
    base = os.path.basename(d.rstrip("/"))
    if base.endswith("42") and "att16" not in base:
        CANDS.append(("w53 " + base.split("_", 1)[1].rsplit("_seed", 1)[0], d))

rng = np.random.RandomState(7)
N = 6000
px, py = rng.randint(0, NPOS, N), rng.randint(0, NPOS, N)
qx, qy = rng.randint(0, NPOS, N), rng.randint(0, NPOS, N)
far = np.hypot(px - qx, py - qy) > 200
fx, fy, gx2, gy2 = px[far], py[far], qx[far], qy[far]
nref = 250
rx, ry = rng.randint(60, NPOS - 60, nref), rng.randint(60, NPOS - 60, nref)
offs = np.arange(1, 31)

print(f"far pairs {far.sum()}, refs {nref}\n")
print(f"{'encoder':22s}{'own g':>7s}{'r_min':>7s}  "
      + "".join(f"{'g=' + str(g):>21s}" for g in GAINS))
print(f"{'':22s}{'':7s}{'':7s}  "
      + "".join(f"{'far>.25':>10s}{'res90':>7s}{'':4s}" for g in GAINS))
print("-" * 100)

for lab, p in CANDS:
    ck = p if p.endswith(".pt") else os.path.join(p, "encoder_final.pt")
    if not os.path.exists(ck):
        continue
    enc, cfg_e, own_gain, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
    field = Field(enc, list(cfg_e.lambdas), fwhm, own_gain, NPOS)
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    ur = raw.get("unique_radius")
    rmin = ur.get("r_min") if isinstance(ur, dict) else None

    cells = []
    for g in GAINS:
        enc.gain = float(g)
        A = field.encode(fx, fy).astype(np.float64)
        B = field.encode(gx2, gy2).astype(np.float64)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        cos = (A * B).sum(1)

        R = field.encode(rx, ry).astype(np.float64)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        prof = np.empty((nref, len(offs)))
        for i, o in enumerate(offs):
            Q = field.encode(np.clip(rx + o, 0, NPOS - 1), ry).astype(
                np.float64)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            prof[:, i] = (R * Q).sum(1)
        res = []
        for row in prof:
            below = np.flatnonzero(row < 0.9)
            res.append(offs[below[0]] if below.size else offs[-1] + 1)
        cells.append(f"{(cos > 0.25).mean():10.4f}{np.median(res):7.1f}    ")

    rs = f"{rmin:7.1f}" if rmin is not None else f"{'-':>7s}"
    print(f"{lab[:22]:22s}{own_gain:7.4g}{rs}  " + "".join(cells))

print("\nfar>.25 = fraction of distant pairs above the dead-goal overlap line.")
print("res90 = median cells before cosine to a reference falls below 0.9.")
print("r_min is the 20-reference value stored in the checkpoint, which the")
print("radius campaign says has been overturned four times -- it ranks arms.")
