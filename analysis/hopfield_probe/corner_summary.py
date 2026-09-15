"""Tables for the corner experiment: ``corner_check`` and the region probe.

Groups the seeds of each arm (corner500, scatter100, scatter118, untrained)
and prints, for the scan, one row per distance band, and for the probe, one row
per world region -- the headline columns of ``arm_summary.py`` at K=5, s=1.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from analysis.hopfield_probe.arm_summary import row as probe_row
from analysis.hopfield_probe.corner_check import BANDS

from cls_paths import results_dir

ROOT = os.path.join(str(results_dir()), "hopfield_probe/20260914")
# The ``_a0.5`` arms are w63, the same two layouts at the ladder's attract
# level; att0.5 is the ladder's own 10% encoder (w52), run in the corner region
# once to separate recipe from region density; "whole" is the unconfined
# probe, the ladder's own setting. Longer names first so prefix matching is
# unambiguous.
ARMS = ("corner500_a0.5", "scatter100_a0.5", "corner500", "scatter100",
        "scatter118", "att0.5", "untrained")
REGIONS = ("corner", "centre", "opposite", "whole")


def arm_of(label: str) -> str:
    for a in ARMS:
        if label.startswith(a):
            return a
    return label


def scan_tables(d: str) -> None:
    by_arm: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        js = json.load(open(f))
        by_arm.setdefault(arm_of(js["label"]), []).extend(js["refs"])
    if not by_arm:
        print(f"(no scan results in {d})")
        return
    print(f"SCAN -- full-arena cosine map per reference, {d}")
    print("  medians over references (both seeds pooled); alias = max cos "
          "beyond 50 cells; @corner = fraction of those aliases inside the "
          "training corner; <0.1 = fraction of the arena below cos 0.1\n")
    hdr = (f"  {'arm':<16s}{'band':<13s}{'n':>3s}{'C(1)':>7s}{'r_0.9':>7s}"
           f"{'r_mono':>8s}{'r_u16':>7s}{'alias':>7s}{'max':>6s}"
           f"{'@corner':>9s}{'<0.1':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for arm in ARMS:
        rows = by_arm.get(arm)
        if not rows:
            continue
        for name, _lo, _hi in BANDS:
            b = [r for r in rows if r["band"] == name]
            if not b:
                continue

            def g(k):
                return np.array([r[k] for r in b], dtype=float)

            print(f"  {arm:<16s}{name:<13s}{len(b):>3d}{g('c1').mean():>7.3f}"
                  f"{np.median(g('r_0.9')):>7.1f}"
                  f"{np.median(g('r_mono')):>8.1f}"
                  f"{np.median(g('r_u16')):>7.1f}"
                  f"{np.median(g('alias')):>7.3f}{g('alias').max():>6.3f}"
                  f"{g('alias_in_corner').mean():>9.2f}"
                  f"{g('frac_below_0.1').mean():>7.3f}")
        print()


def probe_tables(d: str) -> None:
    cells: dict[tuple[str, str], list[dict]] = {}
    for f in sorted(glob.glob(os.path.join(d, "*", "*.json"))):
        if f.endswith("manifest.json"):
            continue
        js = json.load(open(f))
        label = js["header"].get("label", "")
        parts = [p.strip() for p in label.split("·")]
        region = parts[-1]
        cells.setdefault((arm_of(parts[0]), region), []).append(probe_row(js))
    if not cells:
        print(f"(no probe results in {d})")
        return
    print(f"PROBE -- worlds confined to one 500-cell square, K=5, s=1, {d}")
    print("  means over seeds; basin = r_exact_all (cells); reach = "
          "continuous flow reach rate; exact = fraction of goal cues "
          "retrieving their own cell\n")
    hdr = (f"  {'arm':<16s}{'region':<10s}{'n':>3s}{'acc45':>8s}{'|err|':>7s}"
           f"{'exact':>8s}{'basin':>8s}{'reach':>8s}{'disc':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for arm in ARMS:
        for region in REGIONS:
            rs = cells.get((arm, region))
            if not rs:
                continue

            def m(k):
                return float(np.mean([r[k] for r in rs]))

            print(f"  {arm:<16s}{region:<10s}{len(rs):>3d}{m('acc'):>8.3f}"
                  f"{m('err'):>7.1f}{m('exact'):>8.3f}{m('basin'):>8.1f}"
                  f"{m('cont'):>8.3f}{m('disc'):>8.3f}")
        if any(a == arm for a, _ in cells):
            print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    scan_tables(os.path.join(args.root, "corner_check"))
    print()
    probe_tables(os.path.join(args.root, "corner_probe"))


if __name__ == "__main__":
    main()
