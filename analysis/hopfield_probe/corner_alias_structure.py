"""Where does an unseen position's worst alias sit, and why there?

Reads ``corner_check`` JSONs. For every reference outside the training corner
it takes the displacement ``a = alias_at - ref`` to the reference's worst alias
and asks two things:

  lattice   how many of the three modules (periods 11, 12, 13) the displacement
            leaves unchanged -- ``a mod lambda == (0, 0)``. A displacement that
            is a multiple of two periods keeps two module phases and changes
            one, so an alias there means the code is not reading the third
            module. The grid code's own revivals live exactly there.
  edge      whether the alias lies inside the corner, and if so how far it is
            from the corner's boundary -- a collapse onto the nearest edge is a
            different mechanism from a lattice one.

Prints a per-band histogram of the lattice count and of the alias location.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from analysis.hopfield_probe.corner_check import BANDS, DEFAULT_OUT

LAMBDAS = (11, 12, 13)


def modules_kept(a: np.ndarray) -> int:
    return sum(int(a[0] % l == 0 and a[1] % l == 0) for l in LAMBDAS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=DEFAULT_OUT)
    ap.add_argument("--only", default="corner500")
    args = ap.parse_args()

    refs = []
    for f in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        js = json.load(open(f))
        if js["label"].startswith(args.only):
            side = js["side"]
            refs.extend(js["refs"])
    if not refs:
        print("nothing matched")
        return

    print(f"{args.only}: {len(refs)} references, corner side {side}")
    print(f"\n  {'band':<13s}{'n':>3s}{'alias':>7s}{'in corner':>10s}"
          f"{'edge dist':>10s}{'|a|':>7s}{'modules kept 0/1/2/3':>22s}"
          f"{'residues (mod 11,12,13) of a, first ref':>40s}")
    for name, _lo, _hi in BANDS:
        b = [r for r in refs if r["band"] == name]
        if not b:
            continue
        a = np.array([np.array(r["alias_at"]) - np.array(r["ref"]) for r in b])
        kept = np.array([modules_kept(v) for v in a])
        inc = np.array([r["alias_in_corner"] for r in b], dtype=bool)
        # Distance of an in-corner alias from the corner's outer boundary.
        edge = np.array([side - 1 - max(r["alias_at"]) for r in b], dtype=float)
        hist = " ".join(f"{np.sum(kept == k):>2d}" for k in range(4))
        res = " ".join(f"({a[0][0] % l},{a[0][1] % l})" for l in LAMBDAS)
        print(f"  {name:<13s}{len(b):>3d}"
              f"{np.median([r['alias'] for r in b]):>7.3f}"
              f"{inc.mean():>10.2f}"
              f"{np.median(edge[inc]) if inc.any() else float('nan'):>10.1f}"
              f"{np.median(np.hypot(a[:, 0], a[:, 1])):>7.0f}"
              f"{hist:>22s}{res:>40s}")

    out = [r for r in refs if r["corner_dist"] > 0]
    print(f"\n  outside references: {len(out)}; alias inside the corner for "
          f"{np.mean([r['alias_in_corner'] for r in out]):.2f} of them")
    print("  per-reference (band, ref, alias_at, cos, modules kept):")
    for r in out[:40]:
        a = np.array(r["alias_at"]) - np.array(r["ref"])
        print(f"    {r['band']:<13s}{str(r['ref']):>13s}{str(r['alias_at']):>13s}"
              f"{r['alias']:>7.3f}{modules_kept(a):>4d}")


if __name__ == "__main__":
    main()
