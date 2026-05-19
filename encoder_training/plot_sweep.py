"""Plot comprehensive-eval val accuracies across a sweep.

Usage:
    python -m encoder_training.plot_sweep <sweep_dir>

Reads every <sweep_dir>/*/meta.json and result.json pair. Each run writes:
  - meta.json   {index, run_name, grid: {param: value, ...}}
  - result.json {val: {accuracy, ...}, train: {...}}

Emits:
  - <sweep_dir>/plots/val_acc_bar.png
  - <sweep_dir>/plots/results.csv
  - If exactly 1 swept key: val_acc_vs_param.png
  - If exactly 2 swept keys: val_acc_heatmap.png
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


def _fmt(v) -> str:
    if isinstance(v, list):
        return "-".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _load(sweep_dir: str):
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(sweep_dir, "*", "meta.json"))):
        run_dir = os.path.dirname(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        row = {**meta, "val_acc": None, "val_steps": None,
               "val_speed": None, "train_acc": None}
        rj = os.path.join(run_dir, "result.json")
        if os.path.exists(rj) and os.path.getsize(rj) > 0:
            try:
                with open(rj) as f:
                    r = json.load(f)
                v = r.get("val") or {}
                row["val_acc"] = v.get("accuracy")
                row["val_steps"] = v.get("mean_steps")
                row["val_speed"] = v.get("mean_speed")
                t = r.get("train") or {}
                row["train_acc"] = t.get("accuracy")
            except Exception as e:
                print(f"[warn] failed to parse {rj}: {e}")
        rows.append(row)
    rows.sort(key=lambda r: r.get("index", 0))
    keys = []
    for r in rows:
        for k in r.get("grid", {}):
            if k not in keys:
                keys.append(k)
    return keys, rows


def plot_bar(rows, out_path, title):
    idx = [r["index"] for r in rows]
    accs = [r["val_acc"] if r["val_acc"] is not None else np.nan for r in rows]
    labels = [" ".join(f"{k}={_fmt(v)}" for k, v in r["grid"].items())
              or r["run_name"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rows) + 2), 5))
    bars = ax.bar(idx, accs, color=[
        "tab:blue" if not np.isnan(a) else "lightgray" for a in accs])
    if not np.all(np.isnan(accs)):
        best = int(np.nanargmax(accs))
        bars[best].set_color("tab:orange")
        ax.annotate(f"best: {accs[best]:.3f}",
                    xy=(idx[best], accs[best]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("comprehensive val nav accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_1d(rows, key, out_path, title):
    pairs = [(r["grid"][key], r["val_acc"]) for r in rows
             if key in r["grid"] and r["val_acc"] is not None]
    if not pairs:
        return
    try:
        pairs.sort(key=lambda p: float(p[0]))
        xs = [float(p[0]) for p in pairs]; numeric = True
    except (TypeError, ValueError):
        xs = list(range(len(pairs))); numeric = False
    ys = [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    if not numeric:
        ax.set_xticks(xs)
        ax.set_xticklabels([_fmt(p[0]) for p in pairs], rotation=30, ha="right")
    elif min(xs) > 0 and max(xs) / min(xs) >= 10:
        ax.set_xscale("log")
    ax.set_xlabel(key); ax.set_ylabel("val nav accuracy")
    ax.set_ylim(0, 1.0); ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Wrote {out_path}")


def plot_2d(rows, keys, out_path, title):
    k0, k1 = keys
    vals0, vals1 = [], []
    for r in rows:
        g = r["grid"]
        if k0 in g and g[k0] not in vals0: vals0.append(g[k0])
        if k1 in g and g[k1] not in vals1: vals1.append(g[k1])
    for lst in (vals0, vals1):
        try: lst.sort(key=lambda x: float(x))
        except (TypeError, ValueError): pass
    grid = np.full((len(vals0), len(vals1)), np.nan)
    for r in rows:
        if r["val_acc"] is None: continue
        g = r["grid"]
        if k0 in g and k1 in g:
            grid[vals0.index(g[k0]), vals1.index(g[k1])] = r["val_acc"]
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(vals1) + 2),
                                    max(4, 0.6 * len(vals0) + 2)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis",
                   vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(vals1)))
    ax.set_xticklabels([_fmt(v) for v in vals1], rotation=30, ha="right")
    ax.set_yticks(range(len(vals0)))
    ax.set_yticklabels([_fmt(v) for v in vals0])
    ax.set_xlabel(k1); ax.set_ylabel(k0)
    for i in range(len(vals0)):
        for j in range(len(vals1)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="white" if grid[i, j] < 0.5 else "black",
                        fontsize=9)
    fig.colorbar(im, ax=ax, label="val nav accuracy")
    ax.set_title(title); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot sweep results")
    p.add_argument("sweep_dir")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.sweep_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    keys, rows = _load(args.sweep_dir)
    if not rows:
        print(f"No runs found under {args.sweep_dir}/*/meta.json")
        return
    done = sum(1 for r in rows if r["val_acc"] is not None)
    name = os.path.basename(args.sweep_dir.rstrip("/"))
    print(f"Sweep: {name} — {done}/{len(rows)} runs have results")
    for r in rows:
        acc = f"{r['val_acc']:.3f}" if r['val_acc'] is not None else "   -  "
        print(f"  [{r['index']:3d}] {r['run_name']:60s}  val_acc={acc}")

    plot_bar(rows, os.path.join(out_dir, "val_acc_bar.png"),
             title=f"sweep: {name}")
    if len(keys) == 1:
        plot_1d(rows, keys[0],
                os.path.join(out_dir, "val_acc_vs_param.png"),
                title=f"{name} — val acc vs {keys[0]}")
    elif len(keys) == 2:
        plot_2d(rows, keys,
                os.path.join(out_dir, "val_acc_heatmap.png"),
                title=f"{name} — val acc")

    csv_path = os.path.join(out_dir, "results.csv")
    cols = ["index", "run_name", "val_acc", "val_steps", "val_speed",
            "train_acc"] + keys
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            parts = [str(r["index"]), r["run_name"],
                     f"{r['val_acc']:.4f}" if r['val_acc'] is not None else "",
                     f"{r['val_steps']:.2f}" if r['val_steps'] is not None else "",
                     f"{r['val_speed']:.3f}" if r['val_speed'] is not None else "",
                     f"{r['train_acc']:.4f}" if r['train_acc'] is not None else ""]
            for k in keys:
                v = r["grid"].get(k, "")
                parts.append(_fmt(v) if v != "" else "")
            f.write(",".join(parts) + "\n")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
