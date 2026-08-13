#!/usr/bin/env python3
"""Score every encoder checkpoint by its unique coding radius.

This is only a driver: it finds checkpoints, loads them, hands each to
``eval_unique_radius.evaluate_unique_radius``, and writes CSVs. The metric
(``unique_radius``) and the evaluation mechanism (``eval_unique_radius``) know
nothing about files, so the same evaluator can be called mid-training later.

Two outputs, written incrementally so a killed job keeps everything it finished:

``unique_radius_refs.csv``
    One row per (checkpoint, reference) -- 20 rows per encoder. Keeps the raw
    per-location radii, so any other aggregate can be recomputed without
    re-running the sweep.
``unique_radius_sweep.csv``
    One row per checkpoint. ``r_min`` is the headline: the worst of the sampled
    locations at trim=16.

Checkpoints under ``encoders/`` are heterogeneous -- several generations of
save format, and some with lambdas that leave no room for the border margin --
so a failure is recorded as a row with ``status`` set and the sweep continues.

Usage::

    python -m encoder_training.sweep_unique_radius --limit 5      # smoke test
    python -m encoder_training.sweep_unique_radius --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import cls_paths
from encoder_training.eval_unique_radius import (
    DEFAULT_BORDER, DEFAULT_N_REFS, evaluate_unique_radius, npos_for,
)
from encoder_training.unique_radius import (
    DEFAULT_MARGIN_RADII, DEFAULT_PROFILE_LEVELS, DEFAULT_TRIMS, HEADLINE_TRIM,
)

# The 302 top-level ``*.pt`` files are mostly pre-refactor saves whose
# state_dicts no longer match the current model classes, so they are not in the
# default set -- pass --pattern '*.pt' to include them and read the error
# column. The per-run directories carry the current format.
DEFAULT_PATTERNS = ("*/encoder_best.pt", "*/encoder_final.pt")

META_FIELDS = ["ckpt", "name", "status", "error", "lambdas", "Npos",
               "encoder_type", "out_dim", "gain", "fwhm_ratio", "epoch",
               "val_nav_acc", "eval_seconds"]


def ckpt_fwhm_ratio(ckpt: dict, fallback: float) -> float:
    """The smoothing the encoder was trained with, as ``evaluate_nav`` reads it.

    Feeding an encoder codes smoothed differently from training puts it off
    distribution, and the resulting similarity map says nothing about the code.
    """
    train_cfg = ckpt.get("train_config") or {}
    return float(train_cfg.get("fwhm_ratio", fallback))


def discover(root: Path, patterns) -> list[Path]:
    seen: dict[Path, None] = {}
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                seen.setdefault(p.resolve(), None)
    return list(seen)


def rel_to(path: Path, root: Path) -> str:
    """Key used for both the ``ckpt`` column and --resume, so they agree."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def display_name(path: Path, root: Path) -> str:
    """``encoder_100`` at the top level, ``run_20260424_005357/encoder_best`` below.

    Using the parent directory alone would collapse all 302 top-level
    ``encoder_<n>.pt`` files onto the name of the encoders directory itself.
    """
    return path.stem if path.parent == root else f"{path.parent.name}/{path.stem}"


def ckpt_meta(path: Path, root: Path, ckpt: dict) -> dict:
    mc = ckpt.get("model_config", {})
    lam = list(mc.get("lambdas", []))
    return {
        "ckpt": rel_to(path, root),
        "name": display_name(path, root),
        "status": "ok",
        "error": "",
        "lambdas": "-".join(str(int(x)) for x in lam),
        "Npos": npos_for(lam) if lam else "",
        "encoder_type": mc.get("encoder_type", ""),
        "out_dim": mc.get("out_dim", ""),
        "gain": float(ckpt.get("gain", mc.get("gain", float("nan")))),
        "fwhm_ratio": "",
        "epoch": ckpt.get("epoch", ""),
        "val_nav_acc": ckpt.get("val_nav_acc", ""),
        "eval_seconds": "",
    }


def field_names(args) -> tuple[list[str], list[str]]:
    """Fixed column order, derived from the sweep config rather than the data.

    Deriving it from the first successful record would make the header depend
    on which checkpoint happened to load first, and a later row with different
    keys would then be silently dropped by DictWriter.
    """
    trims, margins = args.trims, args.margin_radii
    levels = args.profile_levels

    ref_cols = ["ref_index", "ref_x", "ref_y", "border_dist", "max_r", "n_cells",
                "headline_trim", "r_headline", "saturated_headline",
                "alias_ceiling", "exclusion_radius", "cos_floor",
                # anisotropy-tolerant radii -- the disc columns above collapse
                # on any non-circular map, so rank on these
                "r_alias", "saturated_alias", "far_ceiling", "alias_exclusion",
                "n_rays", "r_monotone_min", "r_monotone_p25",
                "r_monotone_median", "r_monotone_max"]
    for t in trims:
        ref_cols += [f"r_trim{t}", f"saturated_trim{t}"]
    ref_cols += [f"margin_r{R}" for R in margins]
    ref_cols += [f"r_at_cos{lvl}" for lvl in levels]

    # r_min is the per-direction radius; disc_* are the isotropy-sensitive
    # columns, retained for reference but not to be ranked on.
    sum_cols = ["n_refs", "headline", "headline_trim", "r_min", "r_p25",
                "r_median", "r_mean", "r_max", "r_std", "n_saturated",
                "disc_min", "disc_median"]
    for t in trims:
        sum_cols += [f"r_min_trim{t}", f"r_median_trim{t}"]
    sum_cols += ["alias_min", "alias_median", "alias_max",
                 "mono_min", "mono_median", "mono_max",
                 "mono_med_min", "mono_med_median", "mono_med_max",
                 "far_ceiling_max", "far_ceiling_mean", "n_saturated_alias",
                 "n_rays"]
    sum_cols += ["alias_ceiling_max", "alias_ceiling_mean"]
    for R in margins:
        sum_cols += [f"margin_r{R}_min", f"margin_r{R}_mean"]
    sum_cols += [f"r_at_cos{lvl}_median" for lvl in levels]
    sum_cols += ["cos_floor_mean"]

    return META_FIELDS + ref_cols, META_FIELDS + sum_cols


class Sink:
    """Append-as-you-go CSV, so a timeout does not discard finished work."""

    def __init__(self, path: Path, fields: list[str], resume: bool):
        self.path, self.fields = path, fields
        exists = path.exists() and resume
        self.fh = path.open("a" if exists else "w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=fields,
                                     extrasaction="ignore")
        if not exists:
            self.writer.writeheader()
            self.fh.flush()

    def write(self, row: dict):
        self.writer.writerow(row)
        self.fh.flush()

    def close(self):
        self.fh.close()


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as fh:
        return {r["ckpt"] for r in csv.DictReader(fh) if r.get("ckpt")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoders-dir", type=Path, default=None,
                   help="default: cls_paths.encoders_dir()")
    p.add_argument("--pattern", action="append", dest="patterns", default=None,
                   help=f"glob under --encoders-dir, repeatable "
                        f"(default: {' '.join(DEFAULT_PATTERNS)})")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: <sweeps_dir>/unique_radius_<timestamp>")
    p.add_argument("--n-refs", type=int, default=DEFAULT_N_REFS)
    p.add_argument("--border", type=int, default=DEFAULT_BORDER)
    p.add_argument("--seed", type=int, default=0,
                   help="reference positions; shared by every encoder")
    p.add_argument("--trims", type=int, nargs="+", default=list(DEFAULT_TRIMS))
    p.add_argument("--headline-trim", type=int, default=HEADLINE_TRIM)
    p.add_argument("--margin-radii", type=int, nargs="+",
                   default=list(DEFAULT_MARGIN_RADII))
    p.add_argument("--profile-levels", type=float, nargs="+",
                   default=list(DEFAULT_PROFILE_LEVELS))
    p.add_argument("--fwhm-ratio", type=float, default=0.25,
                   help="grid-code smoothing, used only when a checkpoint's "
                        "train_config does not record its own")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--limit", type=int, default=None, help="first N checkpoints")
    p.add_argument("--resume", action="store_true",
                   help="append to existing CSVs and skip finished checkpoints")
    p.add_argument("--dry-run", action="store_true", help="list and exit")
    args = p.parse_args()

    root = (args.encoders_dir or cls_paths.encoders_dir()).resolve()
    patterns = args.patterns or list(DEFAULT_PATTERNS)
    out_dir = args.out_dir or (cls_paths.sweeps_dir()
                               / f"unique_radius_{time.strftime('%Y%m%d_%H%M%S')}")

    # The report always contains the headline trim, so the CSV header must too.
    args.trims = sorted(set(args.trims) | {args.headline_trim})

    ckpts = discover(root, patterns)
    sweep_csv = out_dir / "unique_radius_sweep.csv"
    done = already_done(sweep_csv) if args.resume else set()
    if done:
        ckpts = [c for c in ckpts if rel_to(c, root) not in done]
    if args.limit:
        ckpts = ckpts[:args.limit]

    print(f"encoders dir : {root}")
    print(f"patterns     : {patterns}")
    print(f"checkpoints  : {len(ckpts)}" + (f" ({len(done)} already done)" if done else ""))
    print(f"out dir      : {out_dir}")
    print(f"refs         : {args.n_refs} at border {args.border}, seed {args.seed}")
    print(f"trims        : {args.trims} (headline {args.headline_trim})")
    print(f"device       : {args.device}", flush=True)
    if args.dry_run:
        for c in ckpts[:40]:
            print("   ", c)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**{k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()},
         "encoders_dir": str(root), "patterns": patterns}, indent=2, default=str))

    ref_fields, sum_fields = field_names(args)
    refs_sink = Sink(out_dir / "unique_radius_refs.csv", ref_fields, args.resume)
    sweep_sink = Sink(sweep_csv, sum_fields, args.resume)

    # Imported here: torch.load of a checkpoint pulls in the model classes, and
    # a failure at import time should not look like a failure of the sweep.
    from encoder_training.train import load_encoder

    n_ok = n_err = 0
    for i, path in enumerate(ckpts, 1):
        t0 = time.time()
        meta = {"ckpt": rel_to(path, root), "name": display_name(path, root),
                "status": "error", "error": "", "eval_seconds": ""}
        try:
            encoder, ckpt = load_encoder(str(path), device=args.device)
            meta = ckpt_meta(path, root, ckpt)
            lam = list(ckpt["model_config"].get("lambdas", []))
            if not lam:
                raise ValueError("checkpoint has no lambdas")
            if npos_for(lam) <= 2 * args.border + 1:
                raise ValueError(
                    f"Npos={npos_for(lam)} too small for border={args.border}")

            fwhm = ckpt_fwhm_ratio(ckpt, args.fwhm_ratio)
            meta["fwhm_ratio"] = fwhm
            records, summary = evaluate_unique_radius(
                encoder, lambdas=lam, gain=float(ckpt["gain"]),
                n_refs=args.n_refs, border=args.border, seed=args.seed,
                trims=args.trims, headline_trim=args.headline_trim,
                margin_radii=args.margin_radii,
                profile_levels=args.profile_levels,
                device=args.device, batch_size=args.batch_size,
                fwhm_ratio=fwhm,
            )
            meta["eval_seconds"] = round(time.time() - t0, 1)
            for rec in records:
                refs_sink.write({**meta, **rec})
            sweep_sink.write({**meta, **summary})
            n_ok += 1
            print(f"[{i}/{len(ckpts)}] {meta['name']:<32} "
                  f"r_min={summary['r_min']:>6.1f}  "
                  f"median={summary['r_median']:>6.1f}  "
                  f"alias={summary['alias_ceiling_max']:.3f}  "
                  f"({meta['eval_seconds']}s)", flush=True)
        except Exception as exc:                       # noqa: BLE001 - keep going
            meta["status"] = "error"
            meta["error"] = f"{type(exc).__name__}: {exc}"[:300]
            meta["eval_seconds"] = round(time.time() - t0, 1)
            sweep_sink.write(meta)
            n_err += 1
            print(f"[{i}/{len(ckpts)}] {path.name:<32} ERROR {meta['error']}",
                  flush=True)
            if n_err <= 3:
                traceback.print_exc()
        finally:
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    refs_sink.close()
    sweep_sink.close()
    print(f"\n{n_ok} ok, {n_err} failed -> {sweep_csv}")


if __name__ == "__main__":
    main()
