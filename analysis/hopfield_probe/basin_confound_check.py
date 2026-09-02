"""Is the basin-vs-coverage trend really about coverage, or about patch size?

The ladder's winners do not all share a patch size -- 118x50, 59x50, 30x50,
30x35, 9x50 -- so a decline in `r_exact_95` down the ladder could in principle
be tracking the patch geometry rather than the coverage.

Three checks:

  1. Restrict to the arms trained at 50-cell patches. If the trend survives on
     that subset alone, patch size is not what is driving it.
  2. Compare arms at MATCHED coverage that differ only in patch size -- 1.25%
     has 30x35 against 15x50, and 0.75% has 18x35, 9x50 and 30x27. That is the
     controlled version of the question.
  3. Report the largest basin seen anywhere, since the probe's env is 20x20 and
     a ceiling at the top of the ladder would compress the trend rather than
     explain it.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

ROOT = "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260827"
K, S = "5", "1"


def scal(node, key):
    v = node.get(key)
    return v.get("mean") if isinstance(v, dict) else v


def rows_from(pattern, want=None):
    out = []
    for f in sorted(glob.glob(f"{ROOT}/{pattern}/*.json")):
        if "manifest" in f:
            continue
        r = json.load(open(f))
        lab = r.get("header", {}).get("label") or os.path.basename(f)[:-5]
        if want and want not in lab:
            continue
        try:
            ta = r["test_a"]["k"][K]["per_step"][S]
        except (KeyError, TypeError):
            continue
        h = r["header"]
        out.append(dict(
            label=lab,
            cov=h.get("coverage"),
            n_patch=h.get("n_patches"),
            size=(h.get("patch_sizes") or [None])[0],
            basin=scal(ta["scalars"], "r_exact_95"),
            exact=scal(ta["scalars"], "exact_frac"),
        ))
    return out


print("=== 1. Ladder winners, with geometry ===")
print(f"{'label':22s}{'cov':>8s}{'patches':>9s}{'size':>6s}{'basin':>8s}"
      f"{'exact':>8s}")
print("-" * 61)
lad = rows_from("probe_five")
for r in sorted(lad, key=lambda x: -(x["cov"] or 0)):
    print(f"{r['label'][:22]:22s}{r['cov'] * 100:7.2f}%{r['n_patch']:9d}"
          f"{r['size']:6d}{r['basin']:8.2f}{r['exact']:8.3f}")

fifty = [r for r in lad if r["size"] == 50]
print(f"\n  restricted to 50-cell patches ({len(fifty)} of {len(lad)} rungs):")
for r in sorted(fifty, key=lambda x: -x["cov"]):
    print(f"    {r['cov'] * 100:6.2f}%  basin {r['basin']:6.2f}")

print("\n\n=== 2. Matched coverage, different patch size ===")
# These runs predate the coverage field in `encoder_header`, so the geometry is
# named here from the wave definitions rather than read back off the JSON.
GEOM = {"sm35x_a2": (30, 35), "x_a2": (15, 50),
        "y35_a2": (18, 35), "y50_a2": (9, 50)}
for tag, pats, arms in (
    ("1.25%", ("w60_ps0", "w60_ps1", "w60_ps2"), ("sm35x_a2", "x_a2")),
    ("0.75%", ("w61_ps0", "w61_ps1", "w61_ps2"), ("y35_a2", "y50_a2")),
):
    print(f"\n  {tag}:")
    for a in arms:
        rs = [r for pat in pats for r in rows_from(pat, a)]
        if not rs:
            continue
        n, sz = GEOM[a]
        b = [r["basin"] for r in rs]
        print(f"    {a:12s} {n:3d} x {sz:2d}   basin {np.median(b):6.2f}"
              f"   (n={len(b)} = 4 seeds x 3 draws)")

print("\n\n=== 3. Ceiling check ===")
allr = []
for d in sorted(os.listdir(ROOT)):
    if os.path.isdir(os.path.join(ROOT, d)):
        allr += rows_from(d)
b = np.array([r["basin"] for r in allr if r["basin"] is not None])
print(f"  {len(b)} probed encoders across the archive")
print(f"  max basin seen anywhere: {b.max():.2f}")
print(f"  how many within 0.1 of it: {(b > b.max() - 0.1).sum()}")
print("\n  A hard ceiling would show as many encoders pinned at one value,")
print("  which would compress the top of the ladder rather than explain it.")
