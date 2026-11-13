"""Tables for GRID_MLP_NAV_PLAN §7b from idea 4's final evaluations.

    python -m analysis.idea4.summarise [--root /orcd/pool/003/jackking/cls_runs/idea4]

Reads each run's latest eval_step*.json (written at the end of
``train_memory`` or by ``--eval_ckpt``). The first table is the go/no-go at
N = 6, 30, 100 on idea 2's seed, copy and sequence reads, comparable cell for
cell with §10.9. The second is the capacity curve over every tested N,
averaged over the test seeds, copy reads with sequence reads in brackets.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np


def latest(run_dir):
    files = glob.glob(os.path.join(run_dir, "eval_step*.json"))
    if not files:
        return None, None
    f = max(files, key=lambda s: int(re.search(r"step(\d+)", s).group(1)))
    return int(re.search(r"step(\d+)", f).group(1)), json.load(open(f))


def width(name):
    m = re.match(r"h(\d+)", name)
    return (int(m.group(1)) if m else 0, name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/orcd/pool/003/jackking/cls_runs/idea4")
    a = p.parse_args()
    runs = sorted((d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d))),
                  key=width)
    loaded = [(r, *latest(os.path.join(a.root, r))) for r in runs]
    loaded = [x for x in loaded if x[1] is not None]

    print("\nGo/no-go, idea 2's test seed (copy / sequence, oldest and newest quarter of the copy read)\n")
    print("| run | step | N = 6 | N = 30 | N = 100 | c AUC, N = 6 / 100 |")
    print("|---|---|---|---|---|---|")
    for run, step, rows in loaded:
        by = {r["N"]: r for r in rows if r["seed"] == 10_000 + r["N"]}
        if not all(n in by for n in (6, 30, 100)):
            continue
        cells = [f"{by[n]['acc_copy']:.3f} / {by[n]['acc_seq']:.3f} "
                 f"({by[n]['copy_oldest_quarter']:.2f} / {by[n]['copy_newest_quarter']:.2f})"
                 for n in (6, 30, 100)]
        print(f"| {run} | {step} | " + " | ".join(cells)
              + f" | {by[6]['c_auc']:.2f} / {by[100]['c_auc']:.2f} |")

    Ns = sorted({r["N"] for _, _, rows in loaded for r in rows})
    print("\nCapacity, mean over test seeds (copy, sequence in brackets)\n")
    print("| run | " + " | ".join(f"N = {n}" for n in Ns) + " |")
    print("|---|" + "---|" * len(Ns))
    for run, step, rows in loaded:
        cells = []
        for n in Ns:
            rs = [r for r in rows if r["N"] == n]
            if not rs:
                cells.append("")
                continue
            cells.append(f"{np.mean([r['acc_copy'] for r in rs]):.2f} "
                         f"({np.mean([r['acc_seq'] for r in rs]):.2f})")
        print(f"| {run} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
