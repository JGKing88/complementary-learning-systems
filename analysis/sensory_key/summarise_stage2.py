"""Pool the stage-2 evaluations of GRID_MLP_NAV_PLAN per model.

    python -m analysis.sensory_key.summarise_stage2

Reads the ``eval_all`` JSONs (nav det / stoch, discovery, exploration at 0 and
10 distractors) and the continual-protocol logs written by ``run_cl_ood.sh``
(its printed retention table), for idea 1 (``gmlp``) and Agent-HaSH q + |q| +
projection storage (``qmagproj``), seeds 42 and 43, checkpoints u3800 and
u3950. Each cell is the mean of the four seed x checkpoint values, with their
min-max in brackets.
"""
from __future__ import annotations

import glob
import json
import re

import numpy as np

ROOT = "/orcd/pool/003/jackking/cls_runs"
MODELS = {"gmlp": "idea 1", "qmagproj": "Agent-HaSH q+|q| proj"}
EVAL_KEYS = [
    ("nav_det", "success_rate"), ("nav_det", "mean_steps"),
    ("nav_stoch", "success_rate"), ("nav_stoch", "mean_steps"),
    ("exploration", "swept_coverage"), ("exploration", "goal_find_rate"),
    ("discovery", "reach_success_rate"), ("discovery", "store_success_rate"),
]


def eval_all_rows():
    rows = {}
    for f in sorted(glob.glob(f"{ROOT}/sensory_key/eval_all/*.json")):
        name = f.rsplit("/", 1)[1][:-5]               # e.g. gmlp_s42_u3800
        model = name.split("_s")[0]
        rows.setdefault(model, []).append(json.load(open(f)))
    return rows


def cl_rows():
    """Parse each run_cl_ood log's retention table."""
    rows = {}
    for f in sorted(glob.glob(f"{ROOT}/logs/cl_ood_*.out")):
        text = open(f).read()
        m = re.search(r"--- retention: (\w+)_s\d+ u(\d+) recorded", text)
        if not m or m.group(1) not in MODELS or "_stoch" not in text:
            continue
        num = r"\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)"
        p = re.search(r"primary \(own block, can store\)" + num, text)
        r = re.search(r"revisit \(store locked, live envs\)" + num, text)
        w = re.search(r"worst per-env retention delta[^:]*:\s*([+-][0-9.]+)", text)
        rows.setdefault(m.group(1), []).append(dict(
            first_success=float(p.group(2)), first_steps=float(p.group(3)),
            revisit_success=float(r.group(2)), revisit_steps=float(r.group(3)),
            revisits=int(r.group(1)), worst_delta=float(w.group(1))))
    return rows


def cell(vals):
    vals = np.asarray(vals, float)
    return f"{np.nanmean(vals):.3f} [{np.nanmin(vals):.3f}, {np.nanmax(vals):.3f}] (n={len(vals)})"


def main():
    ev, cl = eval_all_rows(), cl_rows()
    out = {}
    for model, label in MODELS.items():
        print(f"\n=== {label} ({model}) ===")
        for d in ("0", "10"):
            for sec, key in EVAL_KEYS:
                vals = [r[sec][d][key] for r in ev.get(model, []) if d in r.get(sec, {})]
                if vals:
                    print(f"  d={d:>2} {sec + '.' + key:34} {cell(vals)}")
                    out.setdefault(model, {})[f"d{d}.{sec}.{key}"] = vals
        for key in ("first_success", "first_steps", "revisit_success", "revisit_steps",
                    "worst_delta"):
            vals = [r[key] for r in cl.get(model, [])]
            if vals:
                print(f"  continual {key:28} {cell(vals)}")
                out.setdefault(model, {})[f"cl.{key}"] = vals
        n_rev = sum(r["revisits"] for r in cl.get(model, []))
        print(f"  continual revisits in total: {n_rev}")
    json.dump(out, open(f"{ROOT}/sensory_key/stage2_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
