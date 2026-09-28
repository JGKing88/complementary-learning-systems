"""Encoder readouts over training vs the decode, on the same walks (plan sec 6.4).

    python -m analysis.plot_readouts --json encoder_readouts_sz50.json --run p1e_windowbal_sz50_ev49_s0 \
        --decode p1_matched_sz50_s0 --range 49 --out fig.png
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

READOUTS = [("frame_goal", "encoder: frame, W (z_g - z_p)", "--"),
            ("frame_recall", "encoder: frame, W (recall - z_p)  [agent's signal]", "--"),
            ("grad8", "encoder: energy look-ahead, 8-nbr gradient", "-."),
            ("argmax8", "encoder: energy look-ahead, 8-nbr argmax", ":")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--range", default="49")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = json.load(open(a.json))
    rows = sorted([(v["env_steps"], v["readouts"][a.range]) for k, v in R.items()
                   if a.run in k and v.get("env_steps")], key=lambda r: r[0])
    x = np.array([r[0] for r in rows], float)
    d = json.load(open(os.path.join(str(checkpoints_dir()), f"goal_pairs_{a.decode}", "final_tables.json")))
    held = [k for k in d["history"][0]["tables"] if k.startswith("heldout")][0]
    dx = np.array([h["env_steps"] for h in d["history"]], float)
    dy = np.array([h["tables"][held]["trainxtrain"]["model"]["metric"] for h in d["history"]], float)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(dx, dy, "-", color="C0", lw=2, label="decode (MLP): displacement from odometry")
    for i, (k, lab, ls) in enumerate(READOUTS):
        ax.plot(x, [r[1][k] for r in rows], ls, color=f"C{i + 1}", lw=1.8, marker="o", ms=3, label=lab)
    fl = np.mean([r[1]["floor8"] for r in rows])
    ax.axhline(fl, color="0.5", ls=":", lw=1)
    ax.text(x.min(), fl + 1.2, f"perfect 8-neighbour argmax ({fl:.1f})", fontsize=8, color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("env-steps walked (same walks for every curve)")
    ax.set_ylabel(f"held-out direction error (deg), pairs within {a.range} cells")
    ax.set_title(a.title or "Same walks, same size & schedule: encoder readouts vs decode")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
