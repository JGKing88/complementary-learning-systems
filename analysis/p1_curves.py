"""Phase 1 learning curves: decode vs encoder on the same walks (plan sec 6.4).

    python -m analysis.p1_curves --size 20 --out docs/figures/p1_size20.png

Held-out direction error against env-steps for every `train_decode_walk` run
(`final_tables.json`) and `train_encoder_walk` run (`final.json`) whose tag
matches the size, with the pre-trained encoders' readout on the same arenas
and A1's supervised number as reference lines.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cls_paths import checkpoints_dir


def decode_curve(path: str):
    d = json.load(open(path))
    hs = d["history"]
    held = [k for k in hs[0]["tables"] if k.startswith("heldout")][0]
    x = np.array([h["env_steps"] for h in hs], float)
    y = np.array([h["tables"][held]["trainxtrain"]["model"]["metric"] for h in hs], float)
    return x, y, d["argv"]


def encoder_curve(path: str):
    d = json.load(open(path))
    hs = d["history"]
    x = np.array([h["env_steps"] for h in hs], float)
    y = np.array([h["heldout"] for h in hs], float)
    return x, y, d["argv"]


def label_decode(a: dict) -> str:
    bits = [a.get("mode", "grid"), f"{a['n_envs']} envs"]
    bits.append("balanced |Δ|" if a.get("balance_range") else "raw walk pairs")
    if a.get("buffer") == "visited":
        bits.append(f"visited-set buffer ({a['walkers']} walker/arena)")
    elif a.get("buffer_updates", 20) == 1:
        bits.append(f"fresh {a['steps_per_update']}-step rollout, no memory ({a['walkers']} walker/arena)")
    else:
        bits.append("window buffer")
    if a.get("range_warmup_updates", 0):
        bits.append("range warm-up")
    if a.get("target", "direction") != "direction":
        bits.append(a["target"])
    return "decode: " + ", ".join(bits)


def label_encoder(a: dict) -> str:
    if a.get("buffer") == "visited":
        return f"encoder, visited-set buffer ({a['walkers']} walker/arena); within {a['max_abs']}"
    if a.get("batch_mode") == "arena1":
        rows = "balanced rows" if a.get("balance_rows") else "time-uniform rows"
        if a.get("buffer_updates", 20) == 1:
            return f"encoder, fresh {a['steps_per_update']}-step rollout, no memory, {rows}; within {a['max_abs']}"
        return f"encoder, window buffer ({a['walkers']} walkers/arena, 1 per batch), {rows}; within {a['max_abs']}"
    src = "i.i.d. positions (its own sampling)" if a.get("positions", "walk") == "iid" else "walk moments"
    lab = f"encoder: {src}, {a['labels']} labels, r={a['radius']:.0f}"
    if a.get("n_updates", 4000) != 4000:
        lab += f", {a['n_updates'] // 1000}k updates"
    if float(a.get("gain_end", 100)) != 100:
        lab += f", gain\u2192{a['gain_end']:.0f}"
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--refs", default="pre-trained encoders (att0.5 6.00 / ur029 5.96) + harness readout:6.0,A1 decode (teacher-labelled i.i.d. pairs):0.23",
                    help="name:value reference lines, degrees")
    ap.add_argument("--title", default="")
    ap.add_argument("--clean", action="store_true",
                    help="only the valid comparison: the balanced decode (and the raw-walk decode), the "
                         "encoder package's own-trainer points, and the reference lines")
    ap.add_argument("--points", default="",
                    help="extra markers 'label:x:y|label:x:y' (e.g. the encoder package's own trainer "
                         "on walk dumps at a few step budgets)")
    args = ap.parse_args()
    root = str(checkpoints_dir())
    sz = "" if args.size == 20 else f"_sz{args.size}"
    dec = sorted(glob.glob(os.path.join(root, f"goal_pairs_p1_*{sz}_s*", "final_tables.json")))
    enc = sorted(glob.glob(os.path.join(root, f"goal_pairs_p1e_*sz{args.size}*", "final.json")))
    if args.size == 20:
        dec = [p for p in dec if "_sz50" not in p]
    if args.clean:
        dec = [p for p in dec if "h8" not in p and "regular" not in p and "grid32" not in p]
        enc = [p for p in enc if ("online" in p or "window" in p or "fresh" in p) and "windowbal" not in p]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    styles, labelled = {}, set()
    for p in dec:
        x, y, a = decode_curve(p)
        lab = label_decode(a)
        first = lab not in labelled
        color = styles.setdefault(lab, f"C{len(styles)}")
        ax.plot(x, y, "-", color=color, lw=1.8 if first else 1.2, alpha=1.0 if first else 0.6,
                label=lab if first else None)
        labelled.add(lab)
    for p in enc:
        x, y, a = encoder_curve(p)
        lab = label_encoder(a)
        first = lab not in labelled
        color = styles.setdefault(lab, f"C{len(styles)}")
        ax.plot(x, y, "--", color=color, lw=1.8 if first else 1.2, alpha=1.0 if first else 0.6,
                label=lab if first else None)
        labelled.add(lab)
        i = int(np.argmin(y))
        ax.plot(x[i], y[i], "o", color=color, ms=5, mfc="white")     # its best point (its own protocol selects by eval)
    if args.points:
        groups = {}
        for spec in args.points.split("|"):
            lab, x, y = spec.rsplit(":", 2)
            groups.setdefault(lab, []).append((float(x), float(y)))
        for lab, pts in groups.items():
            pts.sort()
            color = styles.setdefault(lab, f"C{len(styles)}")
            ax.plot([q[0] for q in pts], [q[1] for q in pts], "s--", color=color, ms=7, mfc="white", mew=1.8, label=lab)
    for spec in args.refs.split(","):
        name, val = spec.rsplit(":", 1)
        ax.axhline(float(val), color="0.4", ls=":", lw=1)
        ax.text(2.0e4, float(val) * 1.08, name, fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("env-steps walked (same walks for every curve)")
    ax.set_ylabel("held-out direction error (deg), pairs within 19 cells" if args.size == 20
                  else f"held-out direction error (deg), pairs within {args.size - 1} cells")
    ax.set_title(args.title or f"Phase 1: decode vs encoder from random walks, {args.size}x{args.size} arenas, 64 envs")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    out = args.out or f"p1_size{args.size}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}: {len(dec)} decode runs, {len(enc)} encoder runs")


if __name__ == "__main__":
    main()
