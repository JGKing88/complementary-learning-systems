"""Where is the overlap threshold, and does saturation move it?

`max_cos` classifies dead goals at AUC 0.83-1.00, so there is a threshold. Two
things follow from where it sits:

  * it is the number a training objective would have to hold every co-stored
    pair under, and it is a *far-field* quantity -- the competitors sit ~370
    cells away, far outside any coding radius;
  * if the saturated arm tolerates a higher one, that is the capacity condition
    of the two-limit analysis paying off, and it says the fix is available at
    inference rather than only in training.

v35 appears twice: once at its production gain of 3.7 and once at gain 100 with
beta 1e6, which is the same encoder read through a different chart -- so the
embeddings, and hence max_cos, differ between the two rows.
"""
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

ROOT = "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260827"
RUNS = "/orcd/pool/003/jackking/cls_runs"
S = RUNS + "/sweeps"
V35 = RUNS + "/encoders/run_20260422_185816/encoder_best.pt"
NPOS = 1716
DEAD = 0.5

ARMS = [
    ("production", "L7-s42", f"{S}/w53_attract_knee/004_att16_seed=42/"
     "encoder_final.pt", None),
    ("production", "L7-s43", f"{S}/w53_attract_knee/005_att16_seed=43/"
     "encoder_final.pt", None),
    ("att16_vs_att32", "att32-s42", f"{S}/w54_attract_far/000_att32_seed=42/"
     "encoder_final.pt", None),
    ("att16_vs_att32", "att32-s43", f"{S}/w54_attract_far/001_att32_seed=43/"
     "encoder_final.pt", None),
    ("production", "v35", V35, None),
    ("v35_gain100_beta1e6", "v35-g100-sat", V35, 100.0),
    ("gain300_beta1e6", "L7-s42", f"{S}/w53_attract_knee/004_att16_seed=42/"
     "encoder_final.pt", 300.0),
]


def load(arm, lab):
    for f in glob.glob(f"{ROOT}/{arm}/*.json"):
        r = json.load(open(f))
        got = r.get("header", {}).get("label") or os.path.basename(f)[:-5]
        if got == lab:
            return r
    return None


for K in (5, 10):
    print(f"\n=== K={K} ===")
    print(f"{'arm/encoder':30s}{'gain':>6s}{'n':>4s}{'dead':>5s}"
          f"{'max_cos live p90':>18s}{'dead p10':>10s}"
          f"{'split':>7s}{'err':>5s}")
    print("-" * 85)
    for arm, lab, ckpt, gain_over in ARMS:
        r = load(arm, lab)
        if not r:
            continue
        kd = r["test_d"]["k"].get(str(K))
        if not kd:
            continue
        enc, cfg_e, gain, fwhm, _ = load_probe_encoder(
            ckpt, fwhm_fallback=0.25)
        g = gain_over if gain_over is not None else gain
        if gain_over is not None:
            enc.gain = float(gain_over)
        field = Field(enc, list(cfg_e.lambdas), fwhm, g, NPOS)

        vals = kd["1"]["continuous"]["scalars"]["reach_rate"]["values"]
        m = min(K, r["config"].get("n_score_envs", 5))
        mx, dead = [], []
        for w_i, world in enumerate(r["worlds"]):
            specs = world["specs"][:K]
            gx = np.array([s["goal"][0] + s["offset"][0] for s in specs])
            gy = np.array([s["goal"][1] + s["offset"][1] for s in specs])
            Z = field.encode(gx, gy).astype(np.float64)
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            G = Z @ Z.T
            for e in range(m):
                idx = w_i * m + e
                if idx >= len(vals):
                    continue
                others = [j for j in range(K) if j != e]
                mx.append(G[e, others].max())
                dead.append(vals[idx] < DEAD)
        mx, dead = np.array(mx), np.array(dead, bool)
        if len(mx) == 0:
            continue
        # Best single threshold, and how many it misclassifies.
        best, berr = None, len(mx) + 1
        for t in np.unique(mx):
            err = int((mx >= t)[~dead].sum() + (mx < t)[dead].sum())
            if err < berr:
                best, berr = float(t), err
        lp90 = np.percentile(mx[~dead], 90) if (~dead).any() else float("nan")
        dp10 = np.percentile(mx[dead], 10) if dead.any() else float("nan")
        print(f"{(arm + '/' + lab)[:30]:30s}{g:6.0f}{len(mx):4d}"
              f"{int(dead.sum()):5d}{lp90:18.3f}{dp10:10.3f}"
              f"{best:7.3f}{berr:5d}")

print("\n'split' is the max_cos threshold that misclassifies fewest goals;")
print("'err' is how many of n it gets wrong.")
