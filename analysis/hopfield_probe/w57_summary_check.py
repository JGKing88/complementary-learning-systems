"""Both 5% arms, four seeds, three scaffold draws, against the 10% incumbent.

Sec 10.10 measured a fixed arm swinging up to 0.03 across scaffold draws, so a
single-draw gap under ~0.02 is not a result. The two arms are 0.0004 apart on
the alias screen and 0.009 apart on draw 0, which is exactly the regime where
one draw decides nothing.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

NEW = "/home/jackking/.claude/jobs/d05f5770/tmp/probe_w57"
ARCH = "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260827"
SEEDS = (42, 43, 44, 45)


def mo(d, key):
    v = d.get(key)
    return v.get("mean") if isinstance(v, dict) else v


def load(d, label):
    for f in glob.glob(d + "/*.json"):
        if "manifest" in f:
            continue
        r = json.load(open(f))
        if (r.get("header", {}).get("label")
                or os.path.basename(f)[:-5]) == label:
            return r
    return None


def reach(d, label):
    r = load(d, label)
    if r is None:
        return None
    return mo(r["test_d"]["k"]["5"]["1"]["continuous"]["scalars"], "reach_rate")


ARMS = [
    ("sm35_a0.5 (120x35, g50)", f"{NEW}/sm35_a0.5_ps%d", "sm35_a0.5-s%d"),
    ("half_a0.5 (59x50, g75)", f"{NEW}/half_a0.5_ps%d", "half_a0.5-s%d"),
]
REF = [("att0.5 @10% (g100)",
        [f"{ARCH}/attlow_g100", f"{ARCH}/att0.5_ps1", f"{ARCH}/att0.5_ps2"],
        "att0.5-s%d")]

print("Continuous reach, K=5, s=1, beta = gain. Four encoder seeds per cell.\n")
print(f"{'arm':26s}{'draw':>6s}" + "".join(f"{'s' + str(s):>8s}"
                                           for s in SEEDS)
      + f"{'median':>9s}")
print("-" * 73)
summary = {}
for lab, pat, lpat in ARMS:
    meds = []
    for ps in (0, 1, 2):
        vals = [reach(pat % ps, lpat % s) for s in SEEDS]
        ok = [v for v in vals if v is not None]
        if not ok:
            continue
        med = float(np.median(ok))
        meds.append(med)
        cells = "".join(f"{v:8.3f}" if v is not None else f"{'-':>8s}"
                        for v in vals)
        print(f"{lab if ps == 0 else '':26s}{ps:6d}{cells}{med:9.3f}")
    summary[lab] = meds
    print(f"{'':26s}{'mean':>6s}{'':32s}{np.mean(meds):9.3f}")
    print()

for lab, dirs, lpat in REF:
    meds = []
    for ps, d in enumerate(dirs):
        vals = [reach(d, lpat % s) for s in SEEDS]
        ok = [v for v in vals if v is not None]
        if not ok:
            continue
        med = float(np.median(ok))
        meds.append(med)
        cells = "".join(f"{v:8.3f}" if v is not None else f"{'-':>8s}"
                        for v in vals)
        print(f"{lab if ps == 0 else '':26s}{ps:6d}{cells}{med:9.3f}")
    summary[lab] = meds
    print(f"{'':26s}{'mean':>6s}{'':32s}{np.mean(meds):9.3f}")

print("\n\nFull eval of the 5% winner, probe seed 0:")
best = max((k for k in summary if "@10%" not in k),
           key=lambda k: np.mean(summary[k]))
print(f"  (winner by mean of three draws: {best})\n")
tag = "sm35_a0.5" if best.startswith("sm35") else "half_a0.5"
print(f"{'':14s}{'|err|':>8s}{'acc45':>8s}{'exact':>8s}{'basin':>8s}"
      f"{'disc':>8s}{'cont':>8s}{'s15':>8s}")
print("-" * 66)
for s in SEEDS:
    r = load(f"{NEW}/{tag}_ps0", f"{tag}-s{s}")
    if r is None:
        continue
    bc = r["test_bc"]["k"]["5"]["per_step"]
    ta = r["test_a"]["k"]["5"]["per_step"]["1"]
    td = r["test_d"]["k"]["5"]["1"]
    print(f"seed {s:<9d}"
          f"{mo(bc['1']['grid']['scalars'], 'abs_err_mean'):8.2f}"
          f"{mo(bc['1']['grid']['scalars'], 'acc45'):8.3f}"
          f"{mo(ta['scalars'], 'exact_frac'):8.3f}"
          f"{mo(ta['scalars'], 'r_exact_95'):8.2f}"
          f"{mo(td['discrete']['scalars'], 'reach_rate'):8.3f}"
          f"{mo(td['continuous']['scalars'], 'reach_rate'):8.3f}"
          f"{mo(bc['15']['grid']['scalars'], 'acc45'):8.3f}")

print("\n  load curve (dead-goal fraction, constant env subset, draw 0):")
for K in ("1", "3", "5", "10", "20"):
    fr = []
    for s in SEEDS:
        r = load(f"{NEW}/{tag}_ps0", f"{tag}-s{s}")
        kd = r["test_d"]["k"].get(K) if r else None
        if not kd:
            continue
        v = kd["1"]["continuous"]["scalars"]["reach_rate"]["values"]
        m = min(int(K), r["config"].get("n_score_envs", 5))
        sel = [x for i, x in enumerate(v) if (i % m) in (0, 1, 2)]
        fr.append(np.mean([x < 0.5 for x in sel]))
    print(f"    K={K:<3s} {np.median(fr):.2f}")
