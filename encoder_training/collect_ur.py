#!/usr/bin/env python3
"""Collect the ``unique_radius`` summary that ``train`` stored in each checkpoint.

``sweep_unique_radius`` re-scores checkpoints on a GPU; this only reads the
summary the training run already wrote, so it runs on a login node in seconds
and is the right tool for reading back a finished sweep.

Two cautions the output is arranged around:

* ``encoder_best`` is *selected* on ``r_min``, so its recorded value is a max
  over the ~10 evaluations of the run and is optimistically biased (§2.7).
  ``encoder_final`` is not selected on anything and is the honest column; both
  are printed, and a wave should be ranked on ``final`` unless the point is
  explicitly "the best checkpoint reachable".
* One seed is not a result. ``--by`` groups on the sweep's own grid keys and
  reports the median with the seed spread beside it, because the config effect
  and the seed effect are the same size in this metric (§2.6).

Usage::

    python -m encoder_training.collect_ur <sweep_dir> [<sweep_dir> ...]
    python -m encoder_training.collect_ur --by arm --ckpt final <sweep_dir>
    python -m encoder_training.collect_ur --csv out.csv <sweep_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

COLS = ["sweep", "run", "ckpt", "r_min", "r_median", "r_max",
        "alias_ceiling_max", "alias_ceiling_mean", "r_at_cos0.5_median",
        "r_at_cos0.9_median", "mono_med_median", "epoch"]


def grid_of(run_dir: Path) -> dict:
    """The grid point this run stands for, from the driver's own meta.json.

    Read rather than parsed out of the directory name: the name is built for a
    human and a value like a 93-entry patch list is not in it.
    """
    meta = run_dir / "meta.json"
    if not meta.exists():
        return {}
    try:
        cfg = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return {}
    # sweep_ecp writes the whole resolved config; the varying keys are found by
    # differencing across runs, which main() does once it has them all. Newer
    # runs also record the grid *labels*, which beat any reconstruction.
    grid = dict(cfg.get("config", cfg.get("grid", {})))
    grid.update(cfg.get("labels", {}))
    return grid


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


def short_value(key: str, value) -> str:
    """A group label a person can read.

    A patch mix is a 93-entry comma string; used verbatim it makes the grouped
    table wider than the terminal and hides the very comparison it is for. The
    driver's own names are the right labels, so look the value up there first
    and fall back to a shape summary for mixes defined elsewhere.
    """
    if key != "npos_list" or not isinstance(value, str) or len(value) < 24:
        return str(value)
    try:
        from encoder_training.sweep_ecp import SIZE_MIXES
        for name, spec in SIZE_MIXES.items():
            if spec == value:
                return name
    except ImportError:
        pass
    sizes = [int(s) for s in value.split(",")]
    uniq = sorted(set(sizes), reverse=True)
    return f"{len(sizes)}x{'/'.join(str(u) for u in uniq)}"


def varying_keys(grids: list[dict]) -> list[str]:
    """Which config keys actually move across the wave — the real grid axes."""
    if not grids:
        return []
    keys = set().union(*(g.keys() for g in grids))
    out = []
    for k in sorted(keys):
        vals = {json.dumps(g.get(k), sort_keys=True, default=str) for g in grids}
        if len(vals) > 1:
            out.append(k)
    return out


def group_report(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Median and seed spread per grid cell, for the columns worth ranking on.

    Median rather than mean because the metric is a worst case that bottoms out
    at 0, and ``spread`` is printed beside it because §2.6 measured the seed
    effect at roughly the size of the config effect — a cell median with no
    spread beside it is not readable.
    """
    agg = df.groupby(by, dropna=False).agg(
        n=("r_min", "size"),
        r_min_med=("r_min", "median"),
        r_min_max=("r_min", "max"),
        r_min_spread=("r_min", lambda s: float(np.nanmax(s) - np.nanmin(s))),
        r_median_med=("r_median", "median"),
        alias_max=("alias_ceiling_max", "median"),
        alias_mean=("alias_ceiling_mean", "median"),
        decay50=("r_at_cos0.5_median", "median"),
    )
    return agg.sort_values("r_min_med", ascending=False).round(3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="+", type=Path)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--ckpt", default="both", choices=["best", "final", "both"],
                   help="'final' is the unselected, unbiased one -- see §2.7")
    p.add_argument("--best-only", action="store_true",
                   help="deprecated alias for --ckpt best")
    p.add_argument("--by", nargs="*", default=None,
                   help="group on these config keys; empty = infer the axes "
                        "that vary across the wave")
    args = p.parse_args()

    rows: list[dict] = []
    for sw in args.sweeps:
        for run in sorted(sw.iterdir()):
            if run.is_dir() and run.name != "slurm":
                grid = grid_of(run)
                for row in rows_for(run, sw.name):
                    rows.append({**row, "_grid": grid})
    if not rows:
        print("no runs found")
        return
    df = pd.DataFrame(rows)
    grids = [r["_grid"] for r in rows]
    axes = [k for k in varying_keys(grids) if k not in ("run_name", "index")]
    for k in axes:
        df[k] = [short_value(k, g.get(k)) for g in grids]
    df = df.drop(columns=["_grid"])

    kind = "best" if args.best_only else args.ckpt
    if kind != "both":
        df = df[df["ckpt"] == f"encoder_{kind}"]
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"wrote {args.csv} ({len(df)} rows)")

    show = [c for c in COLS if c in df] + \
           [c for c in ("sizes", "n_env", "coverage") if c in df]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df[show].to_string(index=False))

        by = args.by if args.by else [a for a in axes if a != "seed"]
        if by and len(df):
            for ck, sub in df.groupby("ckpt"):
                print(f"\n--- by {by}  ({ck}) ---")
                print(group_report(sub, by).to_string())


if __name__ == "__main__":
    main()
