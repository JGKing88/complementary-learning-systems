#!/usr/bin/env python3
"""``r_min`` is the crossing of the decay curve and the alias ceiling, quantified.

§4.2 said the radius is where the local decay falls to the far-field ceiling.
Treating the radial profile as Gaussian turns that into a formula. If

    decay50 = sigma * sqrt(2 ln 2)          (the reported r_at_cos0.5_median)

then the profile is ``exp(-d^2 / 2 sigma^2)`` and it reaches the alias ceiling
``C`` at

    r_pred = sigma * sqrt(2 ln(1/C)) = decay50 * sqrt(ln(1/C) / ln 2)

Checked against every checkpoint in the sweeps directory that recorded both
columns — 119 encoders spanning the mixed-batch regime, the single-env regime,
the uniformity and geometry rescue attempts, and the graded-target runs — the
prediction correlates +0.86 with the measured ``r_min`` at a median absolute
error of one cell.

Why it is worth having rather than just measuring: it says the two levers
multiply, so a wave can be *designed* instead of searched. It also explains §3
in one line. Every substitute tried there moved one factor at the other's
expense — uniformity took the ceiling from 0.988 to 0.806, a factor of 1.9 on
``sqrt(ln(1/C))``, while collapsing decay50 from 18 to 1, a factor of 18 the
other way. The graded target is the first knob that moves the decay and leaves
the ceiling alone.

Usage::

    python -m encoder_training.radius_law                      # all sweeps
    python -m encoder_training.radius_law <sweep_dir> ...
    python -m encoder_training.radius_law --target 21          # what is needed
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

import cls_paths

FWHM_TO_SIGMA = 1.0 / math.sqrt(2.0 * math.log(2.0))     # decay50 -> sigma


def predict(decay50: float, alias_ceiling: float) -> float:
    """Radius where a Gaussian of the given half-width meets the ceiling."""
    if not (0.0 < alias_ceiling < 1.0) or not decay50 or decay50 != decay50:
        return float("nan")
    sigma = decay50 * FWHM_TO_SIGMA
    return sigma * math.sqrt(2.0 * math.log(1.0 / alias_ceiling))


def required_decay50(target_r: float, alias_ceiling: float) -> float:
    """The decay width a given ceiling needs in order to reach ``target_r``."""
    return target_r / (FWHM_TO_SIGMA * math.sqrt(2.0 * math.log(1.0 / alias_ceiling)))


def collect(sweeps: list[Path]) -> pd.DataFrame:
    rows = []
    for d in sweeps:
        if not d.is_dir():
            continue
        for run in sorted(d.iterdir()):
            for stem in ("encoder_best", "encoder_final"):
                ck = run / f"{stem}.pt"
                if not ck.exists():
                    continue
                try:
                    c = torch.load(ck, map_location="cpu", weights_only=False)
                except Exception:                       # noqa: BLE001
                    continue
                ur = c.get("unique_radius") or {}
                d50 = ur.get("r_at_cos0.5_median")
                C = ur.get("alias_ceiling_max")
                if not ur or not d50 or C is None:
                    continue
                r_pred = predict(float(d50), float(C))
                if r_pred != r_pred:
                    continue
                rows.append({
                    "sweep": d.name, "run": run.name[:40], "ckpt": stem[8:],
                    "r_min": ur["r_min"], "decay50": float(d50),
                    "alias": round(float(C), 4), "r_pred": round(r_pred, 1),
                    "err": round(r_pred - ur["r_min"], 1),
                })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="*", type=Path)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--target", type=float, default=None,
                   help="print the decay width needed to reach this radius at "
                        "a range of alias ceilings")
    args = p.parse_args()

    if args.target:
        print(f"to reach r_min = {args.target:g}:\n")
        print("  alias ceiling   needed decay50   (Gaussian sigma)")
        for C in (0.99, 0.98, 0.96, 0.955, 0.94, 0.92, 0.90, 0.86, 0.82):
            need = required_decay50(args.target, C)
            print(f"       {C:.3f}          {need:6.1f}            {need * FWHM_TO_SIGMA:6.1f}")
        print()

    sweeps = args.sweeps or sorted(
        d for d in cls_paths.sweeps_dir().iterdir() if d.is_dir())
    df = collect(sweeps)
    if df.empty:
        print("no checkpoints with both columns recorded")
        return
    with pd.option_context("display.width", 200):
        print(df.sort_values("r_min", ascending=False).head(args.top)
              .to_string(index=False))
    print(f"\nn={len(df)}  corr(r_pred, r_min) = {df.r_pred.corr(df.r_min):+.3f}"
          f"  median |err| = {df.err.abs().median():.2f}"
          f"  within 3 cells: {(df.err.abs() <= 3).mean():.0%}")


if __name__ == "__main__":
    main()
