"""Does the cheap screen predict the reach the probe actually measured?

far>0.25 is proposed as a stand-in for a full probe run. Every arm in the
archive has both a measured continuous reach and a computable overlap rate, so
the claim is checkable on the arms that exist -- 9 encoder x setting pairs,
spanning gain 3.7 to 300 and beta 100 to 1e6.

Not a fit: the threshold came from the dead-goal analysis, not from this.
"""
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.stats import spearmanr

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

ROOT = "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260827"
RUNS = "/orcd/pool/003/jackking/cls_runs"
S = RUNS + "/sweeps"
V35 = RUNS + "/encoders/run_20260422_185816/encoder_best.pt"
NPOS = 1716
A16 = f"{S}/w53_attract_knee/004_att16_seed=42/encoder_final.pt"

ARMS = [
    ("production", "v35", V35, None),
    ("production", "L7-s42", A16, None),
    ("production", "L7-s43",
     f"{S}/w53_attract_knee/005_att16_seed=43/encoder_final.pt", None),
    ("att16_vs_att32", "att32-s42",
     f"{S}/w54_attract_far/000_att32_seed=42/encoder_final.pt", None),
    ("att16_vs_att32", "att32-s43",
     f"{S}/w54_attract_far/001_att32_seed=43/encoder_final.pt", None),
    ("gain300_beta1e6", "v35", V35, 300.0),
    ("gain300_beta1e6", "L7-s42", A16, 300.0),
    ("gain300_beta1e6", "L7-s43",
     f"{S}/w53_attract_knee/005_att16_seed=43/encoder_final.pt", 300.0),
    ("v35_gain100_beta1e6", "v35-g100-sat", V35, 100.0),
]

rng = np.random.RandomState(7)
N = 6000
px, py = rng.randint(0, NPOS, N), rng.randint(0, NPOS, N)
qx, qy = rng.randint(0, NPOS, N), rng.randint(0, NPOS, N)
far = np.hypot(px - qx, py - qy) > 200
fx, fy, gx2, gy2 = px[far], py[far], qx[far], qy[far]


def reach(arm, lab, K="5"):
    for f in glob.glob(f"{ROOT}/{arm}/*.json"):
        r = json.load(open(f))
        got = r.get("header", {}).get("label") or os.path.basename(f)[:-5]
        if got == lab:
            sc = r["test_d"]["k"][K]["1"]["continuous"]["scalars"]
            return sc["reach_rate"]["mean"]
    return None


print(f"{'arm/encoder':32s}{'gain':>6s}{'far>.25':>9s}"
      f"{'cont reach K=5':>16s}")
print("-" * 63)
xs, ys = [], []
for arm, lab, ck, gover in ARMS:
    y = reach(arm, lab)
    if y is None:
        continue
    enc, cfg_e, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
    g = gover if gover is not None else own
    enc.gain = float(g)
    field = Field(enc, list(cfg_e.lambdas), fwhm, g, NPOS)
    A = field.encode(fx, fy).astype(np.float64)
    B = field.encode(gx2, gy2).astype(np.float64)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    x = float((((A * B).sum(1)) > 0.25).mean())
    xs.append(x)
    ys.append(y)
    print(f"{(arm + '/' + lab)[:32]:32s}{g:6.0f}{x:9.4f}{y:16.3f}")

rho, p = spearmanr(xs, ys)
print(f"\nSpearman(far>0.25, continuous reach) = {rho:+.3f}, p = {p:.4f}, "
      f"n = {len(xs)}")
