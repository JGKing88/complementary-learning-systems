"""Why does low attract_lambda lower the far-field alias rate?

The proposed mechanism: the attract term and the coding-rate (spread) term pull
against each other. Attract asks nearby positions to be SIMILAR, which makes the
code vary slowly and so occupy fewer effective dimensions; the rate term asks the
code to spread over the sphere, which is what suppresses far-field collisions
(turning it off entirely gives an alias rate of 0.2059 against 0.004-0.06).
Lowering attract should therefore shift the balance and let the code spread.

Testable prediction: effective dimension should RISE as attract_lambda falls,
and the far-field cosine distribution should tighten around zero.

If effective dimension does not move, the mechanism is wrong and low attract is
doing something else.
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
SEED = 42

ARMS = [
    ("att16 (L7)", f"{S}/w53_attract_knee/*_att16_seed={SEED}"),
    ("att4", f"{S}/w52_attract_fwhm/*_att4_seed={SEED}"),
    ("att2 (L6)", f"{S}/w49_g100_knee/*eps1_rate0.5_seed={SEED}"),
    ("att1", f"{S}/w52_attract_fwhm/*_att1_seed={SEED}"),
    ("att0.5", f"{S}/w52_attract_fwhm/*_att0.5_seed={SEED}"),
    ("att0.25", f"{S}/w55_nav_objective/*_att0.25_seed={SEED}"),
    ("rate0 (no spread)", f"{S}/w48_g100_nospread/*_rate0_seed={SEED}"),
]

rng = np.random.RandomState(11)
N = 4000
sx, sy = rng.randint(0, NPOS, N), rng.randint(0, NPOS, N)
qx, qy = rng.randint(0, NPOS, N), rng.randint(0, NPOS, N)
far = np.hypot(sx - qx, sy - qy) > 200

print(f"{'arm':20s}{'eff dim':>9s}{'PR/D':>7s}{'far cos: mean':>15s}"
      f"{'sd':>8s}{'p99':>7s}{'max':>7s}{'>0.25':>8s}")
print("-" * 81)
for lab, pat in ARMS:
    hits = sorted(glob.glob(pat))
    if not hits:
        print(f"{lab:20s}  not found")
        continue
    ck = os.path.join(hits[0], "encoder_final.pt")
    enc, cfg, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
    field = Field(enc, list(cfg.lambdas), fwhm, own, NPOS)

    Z = field.encode(sx, sy).astype(np.float64)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    # Participation ratio of the code covariance: (sum ev)^2 / sum ev^2.
    # How many directions the code actually occupies, out of D.
    C = np.cov(Z - Z.mean(0), rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())

    W = field.encode(qx, qy).astype(np.float64)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    cos = (Z[far] * W[far]).sum(1)

    print(f"{lab:20s}{pr:9.1f}{pr / Z.shape[1]:7.3f}{cos.mean():15.4f}"
          f"{cos.std():8.4f}{np.percentile(cos, 99):7.3f}{cos.max():7.3f}"
          f"{(cos > 0.25).mean():8.4f}")

print("\nIf the mechanism holds, effective dimension rises and far-field cosine")
print("sd falls as attract_lambda falls. rate0 is the control: the spread term")
print("removed entirely.")
