"""Does the overlap with co-stored goals predict which goal dies?

At K=1 no encoder loses a single environment, so every reach failure in the
probe is interference between the K stored goals. `multi_env_goals` stores the
first K goals of a world, each at its own arena offset, so the competitors for
env e are the other K-1 goals -- and their overlap with z_goal is computable
directly from the checkpoint.

If `max_cos` separates dead from live, the quantity to train against is the
worst co-stored overlap, which is the alias ceiling rather than `r_min`.
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
S = "/orcd/pool/003/jackking/cls_runs/sweeps"
NPOS = 1716
DEAD = 0.5

ARMS = [
    ("production", "L7-s42",
     f"{S}/w53_attract_knee/004_att16_seed=42/encoder_final.pt", 100.0),
    ("production", "L7-s43",
     f"{S}/w53_attract_knee/005_att16_seed=43/encoder_final.pt", 100.0),
    ("att16_vs_att32", "att32-s42",
     f"{S}/w54_attract_far/000_att32_seed=42/encoder_final.pt", 100.0),
    ("att16_vs_att32", "att32-s43",
     f"{S}/w54_attract_far/001_att32_seed=43/encoder_final.pt", 100.0),
]


def load(arm, lab):
    for f in glob.glob(f"{ROOT}/{arm}/*.json"):
        r = json.load(open(f))
        got = r.get("header", {}).get("label") or os.path.basename(f)[:-5]
        if got == lab:
            return r
    return None


K = 5
print("K=5. max_cos = worst cosine between the test goal's encoding and the")
print("four co-stored goals. min_sep = smallest arena distance to one.\n")
print(f"{'encoder':12s}{'n':>4s}{'max_cos dead':>14s}{'live':>8s}"
      f"{'min_sep dead':>14s}{'live':>8s}{'AUC':>7s}")
print("-" * 67)

for arm, lab, ckpt, _g in ARMS:
    r = load(arm, lab)
    enc, cfg_e, gain, fwhm, _ = load_probe_encoder(ckpt)
    field = Field(enc, list(cfg_e.lambdas), fwhm, gain, NPOS)

    vals = r["test_d"]["k"][str(K)]["1"]["continuous"]["scalars"][
        "reach_rate"]["values"]
    m = min(K, r["config"].get("n_score_envs", 5))

    mx, sep, dead = [], [], []
    for w_i, world in enumerate(r["worlds"]):
        specs = world["specs"][:K]
        gx = np.array([s["goal"][0] + s["offset"][0] for s in specs])
        gy = np.array([s["goal"][1] + s["offset"][1] for s in specs])
        Z = field.encode(gx, gy).astype(np.float64)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        G = Z @ Z.T
        D = np.hypot(gx[:, None] - gx[None, :], gy[:, None] - gy[None, :])
        for e in range(m):
            idx = w_i * m + e
            if idx >= len(vals):
                continue
            others = [j for j in range(K) if j != e]
            mx.append(G[e, others].max())
            sep.append(D[e, others].min())
            dead.append(vals[idx] < DEAD)

    mx, sep, dead = np.array(mx), np.array(sep), np.array(dead, bool)
    if dead.sum() == 0 or (~dead).sum() == 0:
        print(f"{lab:12s}{len(mx):4d}  (no split)")
        continue
    # Rank AUC of max_cos as a dead-goal classifier.
    order = np.argsort(mx)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(mx) + 1)
    n1, n0 = dead.sum(), (~dead).sum()
    auc = (ranks[dead].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    print(f"{lab:12s}{len(mx):4d}{mx[dead].mean():14.4f}{mx[~dead].mean():8.4f}"
          f"{sep[dead].mean():14.0f}{sep[~dead].mean():8.0f}{auc:7.2f}")

print("\nAUC 0.5 = max_cos says nothing about which goal dies; 1.0 = it")
print("orders them perfectly.")
