"""Headline metrics for a probe output directory, beside the archived arms.

Same columns as Sec 1-4.5 of the results doc, except that `reach` here is the
**continuous** rate, which is the stated objective -- the doc's tables quote the
discrete one. Both are shown so the two can be cross-read.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260827"
K, S = "5", "1"


def mo(d, key):
    v = d.get(key)
    return v.get("mean") if isinstance(v, dict) else v


def row(r):
    td = r["test_d"]["k"][K][S]
    bc = r["test_bc"]["k"][K]["per_step"][S]
    ta = r["test_a"]["k"][K]["per_step"][S]
    dead = None
    kv = {}
    for kk in ("1", "3", "5", "10", "20"):
        kd = r["test_d"]["k"].get(kk)
        if not kd:
            continue
        vals = kd["1"]["continuous"]["scalars"]["reach_rate"]["values"]
        m = min(int(kk), r["config"].get("n_score_envs", 5))
        sel = [x for i, x in enumerate(vals) if (i % m) in (0, 1, 2)]
        kv[kk] = float(np.mean([x < 0.5 for x in sel])) if sel else None
    return dict(
        err=mo(bc["grid"]["scalars"], "abs_err_mean"),
        acc=mo(bc["grid"]["scalars"], "acc45"),
        exact=mo(ta["scalars"], "exact_frac"),
        basin=mo(ta["scalars"], "r_exact_all"),
        disc=mo(td["discrete"]["scalars"], "reach_rate"),
        cont=mo(td["continuous"]["scalars"], "reach_rate"),
        s15=mo(r["test_bc"]["k"][K]["per_step"]["15"]["grid"]["scalars"],
               "acc45"),
        dead=kv,
    )


def show(title, files):
    print(f"\n=== {title} ===")
    print(f"{'':16s}{'|err|':>8s}{'acc45':>8s}{'exact':>8s}{'basin':>8s}"
          f"{'disc':>8s}{'CONT':>8s}{'s15':>8s}   dead by K 1/3/5/10/20")
    print("-" * 108)
    conts = []
    for f in sorted(files):
        r = json.load(open(f))
        lab = r.get("header", {}).get("label") or os.path.basename(f)[:-5]
        d = row(r)
        conts.append(d["cont"])
        dk = " ".join(f"{d['dead'].get(k, float('nan')):.2f}"
                      for k in ("1", "3", "5", "10", "20"))
        print(f"{lab:16s}{d['err']:8.2f}{d['acc']:8.3f}{d['exact']:8.3f}"
              f"{d['basin']:8.2f}{d['disc']:8.3f}{d['cont']:8.3f}"
              f"{d['s15']:8.3f}   {dk}")
    if len(conts) > 1:
        print(f"{'median cont':16s}{'':40s}{np.median(conts):8.3f}")


if len(sys.argv) > 1:
    show(sys.argv[1], [f for f in glob.glob(sys.argv[1] + "/*.json") if "manifest" not in f])

print("\n\n--- archived reference arms, same settings ---")
for arm, keep in (("production", ("v35", "L7-s42", "L7-s43")),
                  ("gain300_beta1e6", ("v35", "L7-s42", "L7-s43")),
                  ("v35_gain100_beta1e6", ("v35-g100-sat",))):
    fs = []
    for f in glob.glob(f"{ROOT}/{arm}/*.json"):
        r = json.load(open(f))
        lab = r.get("header", {}).get("label") or os.path.basename(f)[:-5]
        if lab in keep:
            fs.append(f)
    show(arm, fs)
