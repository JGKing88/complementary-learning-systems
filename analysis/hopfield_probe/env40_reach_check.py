"""Why does reach fall at env 40? "Harder navigation problem" was hand-waving.

Sec 10.1 established that reach-by-start-distance is FLAT at env 20 -- failures
are per-environment, not per-position -- so longer paths on their own should not
cost anything. Three candidate explanations, all checkable from the two runs:

  1. RANGE. The readout is `W_basis @ (recalled - current)`. At env 20 no two
     positions are more than 27 cells apart; at env 40 it is 55, and res90 is
     ~7, so the far pair's cosine has long since hit the far-field floor. If
     this is it, reach-by-distance should FALL with distance at env 40 while
     staying flat at env 20.

  2. PER-ENVIRONMENT. Dead goals (reach < 0.5 in an environment) simply became
     more common. That would show as a higher dead fraction with the
     by-distance curve still flat.

  3. STEP BUDGET. `flow_max_steps_factor * size` gives 80 steps at env 20 and
     160 at env 40, while the paths are twice as long -- so the budget scales,
     but arrival also needs the walker to get within 0.5 of a point in a 4x
     larger space. Visible as failures that end far from the goal but were
     still moving.

Only one of these is about the arena being "harder". The first is a real limit
on the encoder's readout range, and would mean env 20 was flattering it.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

RUNS = {20: "/home/jackking/.claude/jobs/d05f5770/tmp/probe_five",
        40: "/home/jackking/.claude/jobs/d05f5770/tmp/probe_env40"}
K, S = "5", "1"
WANT = "10% · att0.5"


def get(root):
    for f in sorted(glob.glob(root + "/*.json")):
        if "manifest" in f:
            continue
        r = json.load(open(f))
        if (r["header"].get("label") or "") == WANT:
            return r
    return None


print(f"{WANT}, K=5, s=1\n")
print("1. Continuous reach by start distance from the goal")
print(f"{'':10s}" + "".join(f"{d:>7d}" for d in range(0, 56, 4)))
print("-" * 76)
for size in (20, 40):
    r = get(RUNS[size])
    c = r["test_d"]["k"][K][S]["continuous"]["reach_by_dist"]
    lo = np.array(c["edges"][:-1], float)
    m = np.array([v if v is not None else np.nan for v in c["mean"]], float)
    n = np.array(c["n"], float)
    row = []
    for d in range(0, 56, 4):
        sel = (lo >= d) & (lo < d + 4) & (n > 30)
        row.append(f"{np.nanmean(m[sel]):7.2f}" if sel.any()
                   else f"{'':>7s}")
    print(f"env {size:<6d}" + "".join(row))

print("\n2. Per-environment failure, same env subset at both sizes")
print(f"{'':10s}{'mean reach':>12s}{'envs >0.95':>12s}{'envs <0.5':>11s}"
      f"{'exact':>8s}{'cells/env':>11s}")
print("-" * 66)
for size in (20, 40):
    r = get(RUNS[size])
    sc = r["test_d"]["k"][K][S]["continuous"]["scalars"]["reach_rate"]
    v = np.array(sc["values"], float)
    ex = r["test_a"]["k"][K]["per_step"][S]["scalars"]["exact_frac"]
    ex = ex["mean"] if isinstance(ex, dict) else ex
    print(f"env {size:<6d}{v.mean():12.3f}{(v > 0.95).mean():12.2f}"
          f"{(v < 0.5).mean():11.2f}{ex:8.3f}{size * size:11d}")

print("\n3. Where the non-arrivals stop (mean steps of those that DID arrive)")
for size in (20, 40):
    r = get(RUNS[size])
    sc = r["test_d"]["k"][K][S]["continuous"]["scalars"]
    ms = sc.get("mean_steps", {})
    ms = ms.get("mean") if isinstance(ms, dict) else ms
    budget = r["config"]["flow_max_steps_factor"] * size
    print(f"  env {size}: mean steps {ms:.1f} of a {budget} budget "
          f"({ms / budget:.0%})")
