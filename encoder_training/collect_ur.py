#!/usr/bin/env python3
"""Collect the ``unique_radius`` summary that ``train`` stored in each checkpoint.

``sweep_unique_radius`` re-scores checkpoints on a GPU; this only reads the
summary the training run already wrote, so it runs on a login node in seconds
and is the right tool for reading back a finished sweep.

Usage::

    python -m encoder_training.collect_ur <sweep_dir> [<sweep_dir> ...]
    python -m encoder_training.collect_ur --csv out.csv <sweep_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

COLS = ["sweep", "run", "ckpt", "r_min", "r_median", "r_max",
        "alias_ceiling_max", "alias_ceiling_mean", "r_at_cos0.5_median",
        "r_at_cos0.9_median", "mono_med_median", "epoch"]


def rows_for(run_dir: Path, sweep: str) -> list[dict]:
    out = []
    for stem in ("encoder_best", "encoder_final"):
        path = run_dir / f"{stem}.pt"
        if not path.exists():
            continue
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:                      # noqa: BLE001
            out.append({"sweep": sweep, "run": run_dir.name, "ckpt": stem,
                        "r_min": float("nan"), "error": str(exc)[:80]})
            continue
        ur = ck.get("unique_radius") or {}
        tc = ck.get("train_config") or {}
        row = {"sweep": sweep, "run": run_dir.name, "ckpt": stem,
               "epoch": ck.get("epoch", ""),
               "sizes": ",".join(str(s) for s in sorted(set(ck.get("sizes", [])))),
               "n_env": len(ck.get("sizes", [])),
               "coverage": round(sum(s * s for s in ck.get("sizes", [])) / 1716 ** 2, 4)
               if ck.get("sizes") else "",
               **{k: ur.get(k) for k in
                  ("r_min", "r_median", "r_max", "alias_ceiling_max",
                   "alias_ceiling_mean", "r_at_cos0.5_median",
                   "r_at_cos0.9_median", "mono_med_median")}}
        for k in ("repel_weight", "attract_lambda", "uniformity_lambda",
                  "exclude_cross_env_pairs"):
            row[k] = (tc.get("loss") or {}).get(k)
        for k in ("per_env_radius_frac", "local_radius", "single_env_batch"):
            row[k] = (tc.get("patches") or {}).get(k)
        for k in ("epochs", "batch_size", "lr", "seed", "gain_end", "fwhm_ratio"):
            row[k] = tc.get(k)
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="+", type=Path)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--best-only", action="store_true",
                   help="drop encoder_final rows")
    args = p.parse_args()

    rows: list[dict] = []
    for sw in args.sweeps:
        for run in sorted(sw.iterdir()):
            if run.is_dir() and run.name != "slurm":
                rows += rows_for(run, sw.name)
    df = pd.DataFrame(rows)
    if args.best_only and "ckpt" in df:
        df = df[df["ckpt"] == "encoder_best"]
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"wrote {args.csv} ({len(df)} rows)")
    show = [c for c in COLS if c in df] + \
           [c for c in ("sizes", "n_env", "coverage") if c in df]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
