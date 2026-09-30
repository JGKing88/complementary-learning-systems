"""One comparison table for the ideal-encoder runs (``run_ideal.sh``).

    python -m analysis.hopfield_probe.ideal_summary [OUT]

OUT defaults to ``$CLS_RESULTS/hopfield_probe/ideal_encoder``. Reads
``OUT/probe/t*/*.json`` (``run.py``, K=5 s=1 columns via ``arm_summary.row``)
and ``OUT/scan/*.json`` (``ideal_scan_check.py``). Missing pieces are skipped,
so it can be run while jobs are still landing.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

from analysis.hopfield_probe.arm_summary import row
from cls_paths import results_dir

REGIONS = ("whole", "corner", "centre", "opposite")


def _f(v, fmt="8.3f"):
    return f"{v:{fmt}}" if isinstance(v, (int, float)) and v == v else \
        f"{'-':>{int(fmt.split('.')[0])}s}"


def probe_table(out: str) -> None:
    rows = {}
    for f in glob.glob(f"{out}/probe/t*/*.json"):
        if f.endswith("manifest.json"):
            continue
        r = json.load(open(f))
        lab = r["header"]["label"]
        enc, _, reg = lab.partition(" | ")
        try:
            rows[(enc, reg or "whole")] = row(r)
        except KeyError as e:
            print(f"  skip {f}: missing {e}")
    print("\n=== probe, K=5 s=1 (run.py; reach 'cont' is the objective) ===")
    print(f"{'encoder':<22s}{'region':<10s}{'|err|':>8s}{'acc45':>8s}"
          f"{'exact':>8s}{'basin':>8s}{'disc':>8s}{'cont':>8s}{'s15':>8s}"
          f"   dead@K 1/3/5/10/20")
    encs = sorted({e for e, _ in rows}, key=lambda s: (not s.startswith("ideal"), s))
    for e in encs:
        for reg in REGIONS:
            d = rows.get((e, reg))
            if d is None:
                continue
            dk = " ".join(_f(d["dead"].get(k), "4.2f")
                          for k in ("1", "3", "5", "10", "20"))
            print(f"{e:<22s}{reg:<10s}{_f(d['err'], '8.2f')}{_f(d['acc'])}"
                  f"{_f(d['exact'])}{_f(d['basin'], '8.2f')}{_f(d['disc'])}"
                  f"{_f(d['cont'])}{_f(d['s15'])}   {dk}")


def scan_table(out: str) -> None:
    files = sorted(glob.glob(f"{out}/scan/*.json"))
    if not files:
        return
    print("\n=== far field + full-scaffold scan (ideal_scan_check.py) ===")
    print(f"{'encoder':<22s}{'D':>6s}{'d_eff':>8s}{'far sd':>8s}"
          f"{'1/√deff':>8s}{'1/√D':>7s}{'>0.25':>8s}{'C(1)':>7s}{'r_0.9':>7s}"
          f"{'r_0.5':>7s}{'r_mono':>7s}{'r_u16':>7s}{'alias':>7s}{'max':>7s}"
          f"{'pred':>7s}{'band C(1) / alias spread':>28s}")
    for f in files:
        r = json.load(open(f))
        ff, refs = r["far_field"], r["refs"]
        D = ff["D"]

        def g(k, rs=refs):
            return np.array([x[k] for x in rs], dtype=float)

        # Extreme value of ~2.9M far cells: sd * sqrt(2 ln N), per reference.
        n_cells = 1716 ** 2
        pred = float(np.median(g("far_map_sd"))) * math.sqrt(2 * math.log(n_cells))
        bands = sorted({x["band"] for x in refs})
        c1s = [g("c1", [x for x in refs if x["band"] == b]).mean() for b in bands]
        als = [np.median(g("alias", [x for x in refs if x["band"] == b]))
               for b in bands]
        spread = (f"{min(c1s):.3f}-{max(c1s):.3f} / "
                  f"{min(als):.3f}-{max(als):.3f}")
        print(f"{r['label']:<22s}{D:>6d}{ff['d_eff']:8.1f}{ff['far_sd']:8.4f}"
              f"{ff['inv_sqrt_d_eff']:8.4f}{1 / math.sqrt(D):7.4f}"
              f"{ff['far_gt_0.25']:8.4f}{g('c1').mean():7.3f}"
              f"{np.median(g('r_0.9')):7.1f}{np.median(g('r_0.5')):7.1f}"
              f"{np.median(g('r_mono')):7.1f}{np.median(g('r_u16')):7.1f}"
              f"{np.median(g('alias')):7.3f}{g('alias').max():7.3f}"
              f"{pred:7.3f}{spread:>28s}")
    print("  pred = median whole-map far sd x sqrt(2 ln 1716^2) (~5.45 sd); "
          "compare the 7.6 sd (~0.24 at D=1024) predicted in the brief")


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else \
        str(results_dir() / "hopfield_probe/ideal_encoder")
    print(f"results: {out}")
    probe_table(out)
    scan_table(out)


if __name__ == "__main__":
    main()
