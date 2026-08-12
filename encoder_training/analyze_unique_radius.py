#!/usr/bin/env python3
"""Read a unique-radius sweep: spread, ranking, and whether r_min discriminates.

``r_min`` is a worst case over 20 locations *and* over directions, so on a
field of similar encoders it compresses -- most of them land in 0-3 and only an
outlier separates. That is the metric behaving correctly, not failing, but it
means the ranking has to come from the columns with spread (``r_median``,
``alias_min``, ``alias_ceiling_max``). This prints enough to tell which.

The correlation against ``val_nav_acc`` at the end is a cross-check, not a
target: nav eval runs inside ``env_size``-cell patches while the radius is
measured over the whole Npos^2 arena, so the two describe different scales and
a weak correlation is expected. Note also that ``encoders/`` contains
duplicate checkpoints saved under several run directories, which shrinks the
effective n well below the row count.

Usage::

    python -m encoder_training.analyze_unique_radius <sweep_dir_or_csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

arg = Path(sys.argv[1])
csv = arg / "unique_radius_sweep.csv" if arg.is_dir() else arg
df = pd.read_csv(csv)

ok = df[df["status"] == "ok"].copy()
print(f"{len(df)} rows: {len(ok)} ok, {len(df) - len(ok)} error")
if len(df) > len(ok):
    kinds = df.loc[df.status != "ok", "error"].str.split(":").str[0]
    print(f"  failures: {dict(kinds.value_counts())}")
if not len(ok):
    sys.exit(0)

ok["kind"] = np.where(ok["name"].str.contains("encoder_best"), "best",
             np.where(ok["name"].str.startswith("binary"), "binary", "final"))

print(f"\n--- r_min (per-direction worst case) ---")
print(ok["r_min"].describe().round(2).to_string())
print("\ndistribution:")
vc = ok["r_min"].value_counts().sort_index()
for v, n in vc.items():
    print(f"  {v:>6.1f}  {'#' * min(60, n):<60} {n}")

frac_low = float((ok["r_min"] <= 3).mean())
print(f"\n{frac_low:.0%} of encoders have r_min <= 3")
if frac_low > 0.8:
    print("  -> r_min is compressed; rank on the columns below instead")

print("\n--- spread of the secondary statistics ---")
cols = [c for c in ("r_median", "mono_median", "alias_min", "alias_median",
                    "alias_ceiling_max", "far_ceiling_mean",
                    "r_at_cos0.9_median", "r_at_cos0.5_median",
                    "cos_floor_mean", "disc_min") if c in ok]
print(ok[cols].describe().loc[["min", "50%", "max", "std"]].round(3).to_string())

print("\n--- top 15 by r_min, tie-broken by alias ceiling ---")
top = ok.sort_values(["r_min", "alias_ceiling_max"],
                     ascending=[False, True]).head(15)
show = [c for c in ("name", "kind", "r_min", "r_median", "alias_min",
                    "alias_ceiling_max", "out_dim", "gain", "val_nav_acc")
        if c in top]
print(top[show].to_string(index=False))

print("\n--- by checkpoint kind ---")
print(ok.groupby("kind")[["r_min", "r_median", "alias_ceiling_max"]]
      .agg(["count", "median", "max"]).round(2).to_string())

if "val_nav_acc" in ok:
    v = pd.to_numeric(ok["val_nav_acc"], errors="coerce")
    m = v.notna() & ok["r_min"].notna()
    if m.sum() > 10:
        print(f"\n--- does the radius predict nav accuracy? (n={int(m.sum())}) ---")
        for c in ("r_min", "r_median", "alias_ceiling_max"):
            if c in ok:
                print(f"  corr({c:<18}, val_nav_acc) = "
                      f"{np.corrcoef(ok.loc[m, c], v[m])[0, 1]:+.3f}")
