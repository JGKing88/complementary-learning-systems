"""Is env size a lever on the basin, or is 21.6 an encoder property?

`r_exact_95` is a radius in cells, so it cannot exceed the largest goal-to-corner
distance the eval environment offers -- about 27 in a 20x20 arena and 55 in a
40x40 one. Across 198 probed encoders the maximum ever seen was 21.62 with 18
within 0.1 of it, which is a pin rather than a coincidence, so the top of the
coverage ladder may be compressed by the measurement.

Only the basin transfers between env sizes. A 40x40 arena is a harder navigation
problem with longer paths, so reach and the angular errors are not comparable
across the two runs and are printed only to make that visible rather than to be
compared.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

RUNS = {
    "env 20": "/home/jackking/.claude/jobs/d05f5770/tmp/probe_five",
    "env 40": "/home/jackking/.claude/jobs/d05f5770/tmp/probe_env40",
}
K, S = "5", "1"
ORDER = ["10% · att0.5", "5% · half_a0.5", "2.5% · q_a1",
         "1.25% · sm35x_a2", "0.75% · y50_a2"]


def scal(node, key):
    v = node.get(key)
    return v.get("mean") if isinstance(v, dict) else v


def load(root):
    out = {}
    for f in sorted(glob.glob(root + "/*.json")):
        if "manifest" in f:
            continue
        r = json.load(open(f))
        lab = r["header"].get("label") or os.path.basename(f)[:-5]
        ta = r["test_a"]["k"][K]["per_step"][S]
        td = r["test_d"]["k"][K][S]
        size = r["config"]["env_size"]
        out[lab] = dict(
            basin=scal(ta["scalars"], "r_exact_95"),
            exact=scal(ta["scalars"], "exact_frac"),
            reach=scal(td["continuous"]["scalars"], "reach_rate"),
            cap=float(np.hypot(size - 1, size - 1)),
        )
    return out


a, b = load(RUNS["env 20"]), load(RUNS["env 40"])

print("Basin radius (r_exact_95), K=5, s=1 — the only column comparable across")
print("env sizes.\n")
print(f"{'encoder':20s}{'env20':>8s}{'env40':>8s}{'change':>9s}"
      f"{'  |  '}{'cap20':>6s}{'cap40':>7s}")
print("-" * 66)
for lab in ORDER:
    if lab not in a or lab not in b:
        continue
    print(f"{lab:20s}{a[lab]['basin']:8.2f}{b[lab]['basin']:8.2f}"
          f"{b[lab]['basin'] - a[lab]['basin']:+9.2f}"
          f"  |  {a[lab]['cap']:6.1f}{b[lab]['cap']:7.1f}")

av = np.array([a[l]["basin"] for l in ORDER if l in a])
bv = np.array([b[l]["basin"] for l in ORDER if l in b])
print(f"\n  spread across the ladder: env20 {av.max() - av.min():.2f} cells, "
      f"env40 {bv.max() - bv.min():.2f}")
print(f"  top two rungs apart by:   env20 {av[0] - av[1]:+.2f}, "
      f"env40 {bv[0] - bv[1]:+.2f}")
print(f"  max basin:                env20 {av.max():.2f} of a {a[ORDER[0]]['cap']:.0f} cap, "
      f"env40 {bv.max():.2f} of {b[ORDER[0]]['cap']:.0f}")

print("\n\nNot comparable across env sizes, shown so the difference is visible:")
print(f"{'encoder':20s}{'reach20':>9s}{'reach40':>9s}{'exact20':>9s}"
      f"{'exact40':>9s}")
print("-" * 56)
for lab in ORDER:
    if lab not in a or lab not in b:
        continue
    print(f"{lab:20s}{a[lab]['reach']:9.3f}{b[lab]['reach']:9.3f}"
          f"{a[lab]['exact']:9.3f}{b[lab]['exact']:9.3f}")
