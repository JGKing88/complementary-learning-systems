"""Dead goals: are they the same envs across encoders, and do they scale with K?

Joins per-environment continuous reach back to the goal that produced it.
`scored_envs` takes the first `min(K, n_score_envs)` envs of each world, so the
value at index i belongs to world i // m, env i % m with m = min(K, 5).

Three things separate the candidate explanations:
  * edge distance of dead vs live goals -- geometry;
  * dead fraction against K -- load / cross-talk;
  * overlap of the dead set between encoders -- whether "dead" is a property of
    the environment or of the encoder.
"""
import glob
import json
import os

import numpy as np

ROOT = "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260827"
SIZE = 20
DEAD = 0.5

WANT = [("production", "L7-s42"), ("production", "L7-s43"),
        ("production", "v35"), ("att16_vs_att32", "att32-s42"),
        ("att16_vs_att32", "att32-s43"),
        ("gain300_beta1e6", "L7-s42"),
        ("v35_gain100_beta1e6", "v35-g100-sat")]


def load(arm, lab):
    for f in glob.glob(f"{ROOT}/{arm}/*.json"):
        r = json.load(open(f))
        got = r.get("header", {}).get("label") or os.path.basename(f)[:-5]
        if got == lab:
            return r
    return None


def rows(r, K):
    """[(world, env, goal, reach)] for one K, or [] if absent."""
    kd = r["test_d"]["k"].get(str(K))
    if not kd:
        return []
    v = kd["1"]["continuous"]["scalars"]["reach_rate"]["values"]
    m = min(K, r["config"].get("n_score_envs", 5))
    out = []
    for i, x in enumerate(v):
        w, e = i // m, i % m
        if w >= len(r["worlds"]):
            continue
        g = r["worlds"][w]["specs"][e]["goal"]
        out.append((w, e, tuple(g), float(x)))
    return out


print("Dead-goal fraction (continuous reach < 0.5) against K\n")
print(f"{'encoder':26s}" + "".join(f"{'K=' + str(k):>9s}"
                                   for k in (1, 3, 5, 10, 20)))
print("-" * 71)
cache = {}
for arm, lab in WANT:
    r = load(arm, lab)
    cache[(arm, lab)] = r
    cells = []
    for K in (1, 3, 5, 10, 20):
        rr = rows(r, K)
        cells.append(f"{np.mean([x[3] < DEAD for x in rr]):9.2f}"
                     if rr else f"{'-':>9s}")
    print(f"{(arm + '/' + lab)[:26]:26s}" + "".join(cells))

print("\n\nGoal geometry, K=5. Edge distance = min distance to any arena wall.\n")
print(f"{'encoder':26s}{'n dead':>7s}{'edge dead':>11s}{'edge live':>11s}"
      f"{'corner dead':>12s}{'corner live':>12s}")
print("-" * 79)
for arm, lab in WANT:
    rr = rows(cache[(arm, lab)], 5)
    if not rr:
        continue
    ed = np.array([min(g[0], g[1], SIZE - 1 - g[0], SIZE - 1 - g[1])
                   for _w, _e, g, _x in rr], float)
    dead = np.array([x < DEAD for *_r, x in rr])
    if dead.sum() == 0:
        print(f"{(arm + '/' + lab)[:26]:26s}{0:7d}{'-':>11s}"
              f"{ed.mean():11.2f}{'-':>12s}{(ed <= 2).mean():12.2f}")
        continue
    print(f"{(arm + '/' + lab)[:26]:26s}{int(dead.sum()):7d}"
          f"{ed[dead].mean():11.2f}{ed[~dead].mean():11.2f}"
          f"{(ed[dead] <= 2).mean():12.2f}{(ed[~dead] <= 2).mean():12.2f}")

print("\n\nIs 'dead' a property of the environment or of the encoder? (K=5)\n")
sets = {}
for arm, lab in WANT:
    rr = rows(cache[(arm, lab)], 5)
    if rr:
        sets[lab if arm != "gain300_beta1e6" else lab + "-g300"] = {
            (w, e) for w, e, _g, x in rr if x < DEAD}
names = list(sets)
print(f"{'':22s}" + "".join(f"{n[:10]:>11s}" for n in names))
for a in names:
    cells = []
    for b in names:
        inter = len(sets[a] & sets[b])
        union = len(sets[a] | sets[b])
        cells.append(f"{inter}/{union}" if a != b else f"({len(sets[a])})")
    print(f"{a[:22]:22s}" + "".join(f"{c:>11s}" for c in cells))
print("\nCells are |A n B| / |A u B| of the dead-env sets; the diagonal is |A|.")
