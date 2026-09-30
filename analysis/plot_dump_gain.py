"""Online curves vs dump-trained points on one env-step axis, 50x50 (plan sec 6.4, 09-29).

    python -m analysis.plot_dump_gain --out fig.png

Lines: the matched decode and the online encoder (gain annealed 1 -> 100, or
held at 30 / 100 from the first update), held-out error over pairs within 49.
Points: the decode (this repo's trainer) and the encoder (its own trainer)
each trained to convergence on the same fixed walk dump, at the dump's
env-steps. The encoder points are from the 09-17 log table (within 49).
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

ENC_DUMP_49 = {524288: 24.5, 2097152: 20.2, 8388608: 20.1}      # log 2026-09-17, own trainer, within 49


def decode_curve(run):
    d = json.load(open(os.path.join(str(checkpoints_dir()), f"goal_pairs_{run}", "final_tables.json")))
    held = [k for k in d["history"][0]["tables"] if k.startswith("heldout")][0]
    x = [h["env_steps"] for h in d["history"]]
    y = [h["tables"][held]["trainxtrain"]["model"]["metric"] for h in d["history"]]
    return np.array(x, float), np.array(y, float), d["final"][held]["trainxtrain"]["model"]["metric"]


def encoder_curve(run):
    d = json.load(open(os.path.join(str(checkpoints_dir()), f"goal_pairs_{run}", "final.json")))
    return (np.array([h["env_steps"] for h in d["history"]], float),
            np.array([h["heldout"] for h in d["history"]], float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    x, y, _ = decode_curve("p1_matched_sz50_s0")
    ax.plot(x, y, "-", color="k", lw=2, label="decode, online (matched size & schedule)")
    for run, lab, c in [("p1e_windowbal_sz50_ev49_s0", "encoder, online, gain annealed 1 -> 100", "C0"),
                        ("p1e_gain30_sz50_ev49_s0", "encoder, online, gain fixed at 30", "C1"),
                        ("p1e_gain100_sz50_ev49_s0", "encoder, online, gain fixed at 100", "C3")]:
        x, y = encoder_curve(run)
        ax.plot(x, y, "-", color=c, lw=1.6, label=lab)
    dx, dy = [], []
    for tag in ("0.5M", "2M", "8M"):
        d = json.load(open(os.path.join(str(checkpoints_dir()), f"goal_pairs_p1d_dump_sz50_{tag}_s0",
                                        "final_tables.json")))
        held = [k for k in d["final"] if k.startswith("heldout")][0]
        dx.append(d["env_steps"]); dy.append(d["final"][held]["trainxtrain"]["model"]["metric"])
    ax.plot(dx, dy, "s", color="k", ms=9, label="decode trained to convergence on the walk dump")
    ax.plot(list(ENC_DUMP_49), list(ENC_DUMP_49.values()), "D", color="C0", ms=9,
            label="encoder (its own trainer, 1000 epochs) on the same dump")
    for xx, yy in zip(dx, dy):
        ax.annotate(f"{yy:.2f}", (xx, yy), textcoords="offset points", xytext=(6, 6), fontsize=8)
    for xx, yy in ENC_DUMP_49.items():
        ax.annotate(f"{yy:.1f}", (xx, yy), textcoords="offset points", xytext=(6, 6), fontsize=8, color="C0")
    ax.set_xscale("log")
    ax.set_ylim(0, 95)
    ax.set_xlabel("env-steps walked (same seed-0 walks for every curve and point)")
    ax.set_ylabel("held-out direction error (deg), pairs within 49 cells")
    ax.set_title("50x50: online curves vs training to convergence on a fixed walk dump")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
