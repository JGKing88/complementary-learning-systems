"""Do the high-overlap pairs sit where a single grid module realigns?

The probe's killing pairs are suggestive but there are only 20 of them, and the
worlds place their environments in a narrow vertical strip, so the displacement
distribution is not isotropic. This tests the same thing directly on the
encoder: draw many random position pairs, and ask whether cosine is predicted by
how many of the six module-axes (3 modules x 2 axes) realign exactly.

No probe, no Hopfield -- this is a property of the encoder and lambda alone.
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
N = 4000            # positions; pairs are drawn from a random permutation

ENC = [
    ("att16-s42", f"{S}/w53_attract_knee/004_att16_seed=42/encoder_final.pt",
     None),
    ("att32-s42", f"{S}/w54_attract_far/000_att32_seed=42/encoder_final.pt",
     None),
    ("v35", RUNS + "/encoders/run_20260422_185816/encoder_best.pt", None),
    ("v35-g100", RUNS + "/encoders/run_20260422_185816/encoder_best.pt",
     100.0),
]

rng = np.random.RandomState(7)
px = rng.randint(0, NPOS, N)
py = rng.randint(0, NPOS, N)
qx = rng.randint(0, NPOS, N)
qy = rng.randint(0, NPOS, N)
far = np.hypot(px - qx, py - qy) > 200          # far field only
px, py, qx, qy = px[far], py[far], qx[far], qy[far]

for lab, ck, gover in ENC:
    enc, cfg_e, gain, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
    lam = list(cfg_e.lambdas)
    g = gover if gover is not None else gain
    if gover is not None:
        enc.gain = float(gover)
    field = Field(enc, lam, fwhm, g, NPOS)

    A = field.encode(px, py).astype(np.float64)
    B = field.encode(qx, qy).astype(np.float64)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    cos = (A * B).sum(1)

    dx, dy = qx - px, qy - py
    folded = []
    for L in lam:
        for d in (dx, dy):
            a = np.abs(d) % L
            folded.append(np.minimum(a, L - a))
    folded = np.stack(folded, 1)
    n_zero = (folded == 0).sum(1)

    print(f"\n=== {lab}  (lambda {lam}, gain {g:g}, n={len(cos)} far pairs) ===")
    print(f"{'module-axes exactly aligned':30s}{'n':>7s}{'mean cos':>10s}"
          f"{'p99':>8s}{'max':>8s}{'frac>0.25':>11s}")
    print("-" * 74)
    for z in range(0, 5):
        sel = n_zero == z
        if sel.sum() < 5:
            continue
        c = cos[sel]
        print(f"{z:<30d}{sel.sum():7d}{c.mean():10.4f}"
              f"{np.percentile(c, 99):8.3f}{c.max():8.3f}"
              f"{(c > 0.25).mean():11.4f}")
    hi = cos > 0.25
    if hi.sum():
        print(f"  pairs above 0.25: {hi.sum()}, of which "
              f"{(n_zero[hi] > 0).mean():.0%} have at least one exact "
              f"alignment (base rate {(n_zero > 0).mean():.0%})")
    else:
        print("  no far pair exceeds 0.25")
