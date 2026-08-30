"""Inference gain against the two things that decide continuous reach.

Dead goals are goals with a co-stored competitor above cos ~0.25-0.29, and the
competitors sit hundreds of cells away, so the quantity to minimise is the rate
of *far-field* pairs above that line. Raising the encoder's inference gain cuts
v35's count from 26 to 10 in 3861 pairs, which is why this sweeps it.

Gain cannot be free, though: Sec 3 of the probe results paid 9 degrees of mean
angular error going from gain 100 to 300 on v35. The local column is the cost
side -- res90 is the distance at which the cosine to a reference falls to 0.9,
i.e. how far the code stays locally informative, and it is the factor Sec 4.4b's
radius law multiplies.

No training here. This is what an existing checkpoint does when read at a
different gain, which is a config change for the probe and a retrain for a
policy.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

RUNS = "/orcd/pool/003/jackking/cls_runs"
S = RUNS + "/sweeps"
NPOS = 1716
GAINS = [1, 3.7, 10, 30, 100, 300, 1000, 3000]

ENC = [
    ("att16-s42", f"{S}/w53_attract_knee/004_att16_seed=42/encoder_final.pt"),
    ("v35", RUNS + "/encoders/run_20260422_185816/encoder_best.pt"),
]

rng = np.random.RandomState(7)
N = 6000
px, py = rng.randint(0, NPOS, N), rng.randint(0, NPOS, N)
qx, qy = rng.randint(0, NPOS, N), rng.randint(0, NPOS, N)
far = np.hypot(px - qx, py - qy) > 200
fx, fy, gx2, gy2 = px[far], py[far], qx[far], qy[far]

# Local probe: 400 references, cosine along +x out to 40 cells.
nref = 400
rx, ry = rng.randint(60, NPOS - 60, nref), rng.randint(60, NPOS - 60, nref)
offs = np.arange(1, 41)

for lab, ck in ENC:
    enc, cfg_e, own_gain, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
    field = Field(enc, list(cfg_e.lambdas), fwhm, own_gain, NPOS)
    print(f"\n=== {lab}   (checkpoint gain {own_gain:g}, "
          f"n_far={far.sum()}, n_ref={nref}) ===")
    print(f"{'gain':>7s}{'far>0.25':>10s}{'far>0.15':>10s}{'p99':>8s}"
          f"{'max':>8s}{'res90':>8s}{'res50':>8s}")
    print("-" * 59)
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

        def cross(level):
            out = []
            for row in prof:
                below = np.flatnonzero(row < level)
                out.append(offs[below[0]] if below.size else offs[-1] + 1)
            return float(np.median(out))

        print(f"{g:7g}{(cos > 0.25).mean():10.4f}{(cos > 0.15).mean():10.4f}"
              f"{np.percentile(cos, 99):8.3f}{cos.max():8.3f}"
              f"{cross(0.9):8.1f}{cross(0.5):8.1f}")

print("\nfar>0.25 is the fraction of distant pairs above the dead-goal line.")
print("res90 / res50 are median cells until cosine drops below 0.9 / 0.5.")
