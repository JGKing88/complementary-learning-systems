"""Tables for plan §10.9 from the idea-2 checkpoint evaluations.

    python -m analysis.idea2.summarise [--root /orcd/pool/003/jackking/cls_runs/idea2]

Reads every <run>/eval_step*.json written by ``meta_train --eval_ckpt`` and
prints, per protocol, the meta-learned memory at N = 6, 30, 100 (with the
oldest and newest quarter and the c AUC) next to the best baseline rate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

RUNS = [
    ("meta_s0", "whole MLP, revisits in training, T 2-6"),
    ("once_s0", "whole MLP, T 2-10"),
    ("once_v200_s0", "whole MLP, 200-step visits"),
    ("once_k3_s0", "whole MLP, k = 3"),
    ("rh_s0", "R + H, ReLU keys"),
    ("rhlin_s0", "R + H, signed keys"),
    ("rhlin_v200_s0", "R + H, signed keys, 200-step visits"),
    ("rhlin_long_s0", "R + H, signed keys, then T 20-100"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/orcd/pool/003/jackking/cls_runs/idea2")
    a = p.parse_args()
    for mode in ("once", "revisit"):
        print(f"\n{mode}\n")
        print("| run | step | N = 6 | N = 30 | N = 100 | c AUC, N = 6 / 100 | best baseline, N = 6 / 30 / 100 |")
        print("|---|---|---|---|---|---|---|")
        for run, label in RUNS:
            files = sorted(glob.glob(os.path.join(a.root, run, "eval_step*.json")))
            if not files:
                continue
            rows = [r for r in json.load(open(files[-1])) if r["mode"] == mode]
            meta = {r["N"]: r for r in rows if r["model"] == "meta"}
            base = {}
            for r in rows:
                if r["model"] != "meta":
                    base[r["N"]] = max(base.get(r["N"], 0.0), r["acc"])
            cells = [f"{meta[n]['acc']:.3f} ({meta[n]['acc_oldest_quarter']:.2f} / "
                     f"{meta[n]['acc_newest_quarter']:.2f})" for n in (6, 30, 100)]
            print(f"| {label} | {meta[6]['ckpt_step']} | " + " | ".join(cells)
                  + f" | {meta[6]['c_auc']:.2f} / {meta[100]['c_auc']:.2f} | "
                  + " / ".join(f"{base[n]:.3f}" for n in (6, 30, 100)) + " |")


if __name__ == "__main__":
    main()
