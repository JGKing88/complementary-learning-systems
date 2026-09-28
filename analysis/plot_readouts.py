"""Encoder readouts over training vs the decode, on the same walks (plan sec 6.4).

    python -m analysis.plot_readouts --json a.json,b.json --runs RUN1=label1,RUN2=label2 \
        --readouts frame_recall,grad8 --decode p1_matched_sz50_s0 --range 49 --out fig.png

Every (run, readout) pair is a curve over the run's checkpoints (x = env-steps from
the checkpoint), beside the decode's held-out curve. One run with the four default
readouts reproduces the earlier single-run figure.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cls_paths import checkpoints_dir

NAMES = {"frame_goal": "frame, W (z_g - z_p)", "frame_recall": "frame, W (recall - z_p) [agent]",
         "grad4": "energy look-ahead, 4-nbr gradient", "grad8": "energy look-ahead, 8-nbr gradient",
         "argmax4": "energy look-ahead, 4-nbr argmax", "argmax8": "energy look-ahead, 8-nbr argmax"}
STYLES = {"frame_goal": "--", "frame_recall": "--", "grad4": "-.", "grad8": "-.", "argmax4": ":", "argmax8": ":"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="comma-separated encoder_readouts outputs")
    ap.add_argument("--runs", required=True, help="comma-separated RUN or RUN=label")
    ap.add_argument("--readouts", default="frame_goal,frame_recall,grad8,argmax8")
    ap.add_argument("--decode", required=True)
    ap.add_argument("--range", default="49")
    ap.add_argument("--floor", action="store_true", help="draw the perfect 8-neighbour argmax floor")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {}
    for j in a.json.split(","):
        R.update(json.load(open(j)))
    d = json.load(open(os.path.join(str(checkpoints_dir()), f"goal_pairs_{a.decode}", "final_tables.json")))
    held = [k for k in d["history"][0]["tables"] if k.startswith("heldout")][0]
    dx = np.array([h["env_steps"] for h in d["history"]], float)
    dy = np.array([h["tables"][held]["trainxtrain"]["model"]["metric"] for h in d["history"]], float)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(dx, dy, "-", color="k", lw=2, label="decode (MLP): displacement from odometry")
    c = 0
    fl = []
    for spec in a.runs.split(","):
        run, lab = (spec.split("=", 1) + [None])[:2]
        rows = sorted([(v["env_steps"], v["readouts"][a.range]) for k, v in R.items()
                       if k.split("/")[-2].endswith(run) and v.get("env_steps")],
                      key=lambda r: r[0])
        rows = [r for r in rows if r[0]]
        if not rows:
            print(f"no checkpoints for {run}"); continue
        x = np.array([r[0] for r in rows], float)
        fl += [r[1]["floor8"] for r in rows]
        for k in a.readouts.split(","):
            ax.plot(x, [r[1][k] for r in rows], STYLES[k], color=f"C{c}", lw=1.8, marker="o", ms=3,
                    label=f"encoder {lab or run}: {NAMES[k]}")
            c += 1
    if a.floor and fl:
        ax.axhline(np.mean(fl), color="0.5", ls=":", lw=1)
        ax.text(dx.min(), np.mean(fl) + 1.2, f"perfect 8-neighbour argmax ({np.mean(fl):.1f})", fontsize=8, color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("env-steps walked (same walks for every curve)")
    ax.set_ylabel(f"held-out direction error (deg), pairs within {a.range} cells")
    ax.set_title(a.title or "Same walks, same size & schedule: encoder readouts vs decode")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
