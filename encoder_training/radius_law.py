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

It predicts ``r_median`` more faithfully than ``r_min``. Both inputs are
whole-encoder summaries — a median decay width and a max alias ceiling — so what
comes out describes the typical reference, while ``r_min`` is the worst of
twenty. They agree while the references agree with each other, and part company
when they do not: at a near radius of 40 the decay is the widest in the
campaign and ``r_median`` is 15, but one reference sits at 5 and that is what
``r_min`` reports. **A large gap between ``r_pred`` and ``r_min`` is therefore a
reading in its own right** — it says the arena has stopped being uniform, not
that the formula has failed.

DOMAIN (§5.6i). The paragraph above is right about the mechanism and was too
relaxed about the consequence. Every checkpoint that validated this formula sat
at ``per_env_radius_frac <= 0.2``; past that the failure is not a wide error
bar but a systematic, *optimistic* one, because a wide attract radius improves
the typical direction while ruining the worst. At frac 0.25 the code has the
best decay50 (53) and the best median monotone length (58-68) of its wave and
an ``r_min`` of 2; at frac 0.4 it predicts 10.6 against a measured 0.5, the
largest residual in the campaign. The cheap check is ``mono_med`` (median over
directions) against ``r_median`` (median over references of the *worst*
direction): while they track, the code is roughly isotropic and the formula
applies; when they diverge, it does not.

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
LN10_9 = math.log(1.0 / 0.9)
GAUSSIAN_SHAPE = math.sqrt(LN10_9 / math.log(2.0))       # r_at_cos0.9 / decay50


def predict(decay50: float, alias_ceiling: float,
            r_at_cos90: float | None = None) -> float:
    """Radius where the decay profile meets the alias ceiling.

    Anchored on ``r_at_cos0.9`` when it is available, and on ``decay50``
    otherwise. The Gaussian assumption is only being used to extrapolate from a
    measured level of the profile out to the ceiling, and the ceiling sits
    around 0.95 — so anchoring at 0.9 asks the assumption to hold over a short
    stretch, while anchoring at 0.5 asks it to hold over the whole profile.

    That costs nothing when the profile is Gaussian, which it usually is:
    across 240 checkpoints the median ``r_at_cos0.9 / decay50`` is 0.389
    against the Gaussian 0.390. It matters for the ~8% that are not — the
    over-repelled arms have decay50 near 20 and ``r_at_cos0.9`` of 1, a sharp
    spike on a long tail. There the 0.5 anchor lands within 3 cells 53% of the
    time and the 0.9 anchor 89%.
    """
    if not (0.0 < alias_ceiling < 1.0):
        return float("nan")
    lnC = math.log(1.0 / alias_ceiling)
    if r_at_cos90 is not None and r_at_cos90 == r_at_cos90 and r_at_cos90 > 0:
        return r_at_cos90 * math.sqrt(lnC / LN10_9)
    if not decay50 or decay50 != decay50:
        return float("nan")
    return decay50 * FWHM_TO_SIGMA * math.sqrt(2.0 * lnC)


def required_decay50(target_r: float, alias_ceiling: float) -> float:
    """The decay width a given ceiling needs in order to reach ``target_r``."""
    return target_r / (FWHM_TO_SIGMA * math.sqrt(2.0 * math.log(1.0 / alias_ceiling)))


def gaussianity(decay50: float, r_at_cos90: float) -> float:
    """``r_at_cos0.9 / decay50``, which is 0.390 for a Gaussian profile.

    The law extrapolates a shape, so this says whether there is a shape to
    extrapolate. Across 240 checkpoints the median is 0.389; the outliers are
    the over-repelled arms, whose profile is a one-cell spike on a long tail.
    """
    return r_at_cos90 / decay50 if decay50 else float("nan")


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
                d90 = ur.get("r_at_cos0.9_median")
                C = ur.get("alias_ceiling_max")
                if not ur or not d50 or C is None:
                    continue
                r_pred = predict(float(d50), float(C), d90)
                if r_pred != r_pred:
                    continue
                rows.append({
                    "sweep": d.name, "run": run.name[:40], "ckpt": stem[8:],
                    "r_min": ur["r_min"], "decay50": float(d50),
                    "res90": d90,
                    # 0.390 for a Gaussian; far from it means the profile is a
                    # spike on a tail and the extrapolation is on thin ice.
                    "shape": round(d90 / float(d50), 3) if d90 else float("nan"),
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
