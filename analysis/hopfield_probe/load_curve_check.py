"""Does a better encoder flatten the load curve, or only shift it?

The K columns quoted in Sec 10.2 are not comparable to each other: `scored_envs`
measures `min(K, n_score_envs)` environments, so K=1 scores one env per world
and K>=5 scores five. The population grows with K below 5, which is the same
confound `n_score_envs` was added to kill on the retrieval axis.

Fixing it costs nothing -- `Scalars.values` is in scored-env order, so a fixed
env subset can be selected at every K. All 8 worlds are usable here because
this needs the reach values only, not the goal specs.

  env 0 only          -> 8 samples, available at every K
  envs 0-2            -> 24 samples, available at K >= 3
"""
import glob
import json
import os

import numpy as np

ROOT = "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260827"
DEAD = 0.5
KS = (1, 3, 5, 10, 20)

ARMS = [
    ("v35_gain100_beta1e6", "v35-g100-sat", "v35 g100+b1e6"),
    ("gain300_beta1e6", "v35", "v35 g300+b1e6"),
    ("production", "v35", "v35 production"),
    ("gain300_beta1e6", "L7-s42", "att16-s42 g300+b1e6"),
    ("production", "L7-s42", "att16-s42 production"),
    ("production", "L7-s43", "att16-s43 production"),
    ("att16_vs_att32", "att32-s42", "att32-s42 production"),
    ("beta1e6_own_gain", "L7-s42", "att16-s42 b1e6 alone"),
]


def load(arm, lab):
    for f in glob.glob(f"{ROOT}/{arm}/*.json"):
        r = json.load(open(f))
        got = r.get("header", {}).get("label") or os.path.basename(f)[:-5]
        if got == lab:
            return r
    return None


def curve(r, keep_envs):
    """(dead fraction, n) per K over a fixed env subset."""
    out = {}
    for K in KS:
        kd = r["test_d"]["k"].get(str(K))
        if not kd:
            out[K] = (None, 0)
            continue
        v = kd["1"]["continuous"]["scalars"]["reach_rate"]["values"]
        m = min(K, r["config"].get("n_score_envs", 5))
        sel = [x for i, x in enumerate(v) if (i % m) in keep_envs and i % m < m]
        if not sel:
            out[K] = (None, 0)
            continue
        out[K] = (float(np.mean([x < DEAD for x in sel])), len(sel))
    return out


for title, keep in (("env 0 only", {0}), ("envs 0-2", {0, 1, 2})):
    print(f"\n=== Dead-goal fraction against load, {title} ===")
    print(f"{'arm':26s}" + "".join(f"{'K=' + str(k):>9s}" for k in KS)
          + f"{'  n':>5s}")
    print("-" * 78)
    for arm, lab, show in ARMS:
        r = load(arm, lab)
        if not r:
            continue
        c = curve(r, keep)
        ns = {n for _f, n in c.values() if n}
        cells = "".join(f"{f:9.2f}" if f is not None else f"{'-':>9s}"
                        for f, _n in c.values())
        print(f"{show:26s}{cells}{max(ns) if ns else 0:5d}")

print("\n\nIf the encoder only shifted the curve, every arm would rise with K")
print("at the same rate from a different intercept. If it changes the slope,")
print("the per-competitor kill probability is what moved.")

print("\n=== Implied per-competitor kill probability ===")
print("Under P(dead) = 1 - (1-p)^(K-1), inverted at each K, envs 0-2.\n")
print(f"{'arm':26s}" + "".join(f"{'K=' + str(k):>9s}" for k in KS[1:]))
print("-" * 62)
for arm, lab, show in ARMS:
    r = load(arm, lab)
    if not r:
        continue
    c = curve(r, {0, 1, 2})
    cells = []
    for K in KS[1:]:
        f, n = c[K]
        if f is None or K == 1:
            cells.append(f"{'-':>9s}")
        elif f <= 0:
            cells.append(f"{'<0.01':>9s}")
        elif f >= 1:
            cells.append(f"{'-':>9s}")
        else:
            p = 1 - (1 - f) ** (1 / (K - 1))
            cells.append(f"{p:9.3f}")
    print(f"{show:26s}" + "".join(cells))
