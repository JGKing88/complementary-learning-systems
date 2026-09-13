"""Collect the B2 lifetime runs (plan sec 4B) into one table.

    python -m analysis.b2_results --prefix b2

Reads every `<CLS_RUNS>/agent_ckpts/goal_lifetimes_<prefix>*/final_tables.json`
and prints, per run and per eval set: readout 1 (h = 0, train x train),
readout 2 by episode at e0 / e1 / e2 / e5 / e19 with the by-episode slope,
and episode 0's own by-step row at s0..s10 -- the "how many steps to
measure the frame" readout. A `--scripted` run has no readout 1.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from cls_paths import checkpoints_dir

EPS = (0, 1, 2, 5, 19)
STEPS = (0, 1, 2, 3, 5, 10, 20)


def rows_for(path: str) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    argv = d["argv"]
    run = os.path.basename(os.path.dirname(path))
    out = []
    for name, agg in d["lifetime"].items():
        be = np.array(agg["by_episode"], dtype=float)
        ok = np.isfinite(be)
        slope = float(np.polyfit(np.arange(len(be))[ok], be[ok], 1)[0]) if ok.sum() > 2 else float("nan")
        r1 = None
        if "final" in d and name in d["final"]:
            r1 = d["final"][name]["trainxtrain"]["model"]["metric"]
        out.append({
            "run": run, "arm": "scripted" if d.get("scripted") else argv["arm"],
            "L": argv["num_layers"], "H": argv["hidden_size"], "seed": argv["seed"],
            "mix": argv.get("lattice_mix_standard_frac", 0.0), "set": name, "r1_tt": r1,
            "ep": [be[i] if i < len(be) else float("nan") for i in EPS], "slope": slope,
            "s": [agg["ep0_by_step"][i] if i < len(agg["ep0_by_step"]) else float("nan") for i in STEPS],
            "lattice": d.get("lattice", {}),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="b2")
    args = p.parse_args()
    root = os.path.join(checkpoints_dir(), f"goal_lifetimes_{args.prefix}*", "final_tables.json")
    paths = sorted(glob.glob(root))
    if not paths:
        raise SystemExit(f"no runs under {root}")
    rows = [r for pth in paths for r in rows_for(pth)]
    hdr = (f"{'run':34s} {'arm':9s} {'set':12s} {'R1tt':>6s} | "
           + " ".join(f"{'e'+str(e):>5s}" for e in EPS) + f" {'slope':>6s} | "
           + " ".join(f"{'s'+str(s):>5s}" for s in STEPS))
    print(hdr)
    for r in rows:
        r1 = "   -  " if r["r1_tt"] is None else f"{r['r1_tt']:6.1f}"
        print(f"{r['run'][:34]:34s} {r['arm']:9s} {r['set']:12s} {r1} | "
              + " ".join(f"{v:5.1f}" for v in r["ep"]) + f" {r['slope']:6.2f} | "
              + " ".join(f"{v:5.1f}" for v in r["s"]))
    for pth in paths:
        with open(pth) as f:
            d = json.load(f)
        lat = d.get("lattice")
        if lat:
            print(f"{os.path.basename(os.path.dirname(pth))[:34]:34s} lattice: {lat['n_lifetimes']} lifetimes, "
                  f"{lat['n_in_holdout']} in the held-out band, mix {lat['mix_standard_frac']}")


if __name__ == "__main__":
    main()
