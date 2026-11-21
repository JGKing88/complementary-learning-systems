"""Tables for the independent-bump codes beside the rectified ideal, measure for measure.

    python -m analysis.rect_ideal.bump_summary [ROOT] [RECT_ROOT]

ROOT defaults to ``$CLS_RESULTS/rect_ideal_grid_bump`` (``probe/t*/*.json`` from
``run_bump_probe.sh``, ``kernel/*.json`` from ``bump.py check``); RECT_ROOT to
``$CLS_RESULTS/rect_ideal_grid``, whose ``grid_results.json`` holds the rectified ideal. Prints
every probe measure (``summary.MEASURES``, extracted by ``summary.probe_row``) at K = 5 and K = 20,
the bump codes beside the signed ideal and the rectified ideal at 2%, 10% and 20% of 1024 units,
plain and with inhibition; then the measured kernel of each bump code beside the formula. Writes
``ROOT/bump_results.json``. Missing runs are skipped.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from analysis.rect_ideal.bump import RESULTS, codes, label
from analysis.rect_ideal.summary import MEASURES, probe_row
from cls_paths import results_dir

KS = ("5", "20")
DIGITS = {"err": 1, "basin": 1, "basin95": 1, "cont_steps": 1}
REF_F = (0.02, 0.10, 0.20)


def load_bump(root: str) -> list[dict]:
    out = []
    for fn in sorted(glob.glob(f"{root}/probe/t*/*.json")):
        if fn.endswith("manifest.json"):
            continue
        r = json.load(open(fn))
        lab = r["header"]["label"]
        try:
            out.append({"label": lab, "spec": r["header"].get("path"), "inh": lab.endswith(" inh"),
                        "bump": r["header"].get("bump"), "K": probe_row(r)})
        except KeyError as e:
            print(f"  skip {fn}: missing {e}")
    return out


def load_reference(rect_root: str) -> list[dict]:
    res = json.load(open(os.path.join(rect_root, "grid_results.json")))
    keep = []
    for p in res["probe"]:
        if p["D"] != 1024:
            continue
        if p["kind"] == "ideal" or (p["kind"] == "rect" and p["f"] in REF_F):
            keep.append(p)
    order = {("ideal", None, False): 0}
    for j, f in enumerate(REF_F):
        order[("rect", f, False)] = 1 + j
        order[("rect", f, True)] = 1 + len(REF_F) + j
    return sorted(keep, key=lambda p: order[(p["kind"], p["f"], p["inh"])])


def _fmt(v, p):
    return f"{v:8.{p}f}" if isinstance(v, (int, float)) and v == v else f"{'-':>8s}"


def _row(name: str, K: dict) -> str:
    return f"{name:<32s}" + "".join(_fmt(K.get(m), DIGITS.get(m, 3 if m != 'dead' else 2))
                                    for m in MEASURES)


def print_tables(bump: list[dict], ref: list[dict]) -> None:
    by_label = {b["label"]: b for b in bump}
    head = f"{'':<32s}" + "".join(f"{'steps' if m == 'cont_steps' else m:>8s}" for m in MEASURES)
    for K in KS:
        for inh in (False, True):
            print(f"\n== K = {K}, one step, {'with inhibition' if inh else 'plain'} ==")
            print(head)
            for p in ref:
                if p["inh"] != inh and p["kind"] != "ideal":
                    continue
                name = "ideal D1024 (signed)" if p["kind"] == "ideal" else p["label"]
                print(_row(name, p["K"].get(K, {})))
            for c in codes():
                b = by_label.get(label(c, inh))
                print(_row(label(c, inh), b["K"].get(K, {}) if b else {}))


def load_kernels(root: str) -> list[dict]:
    out = []
    for fn in sorted(glob.glob(f"{root}/kernel/*.json")):
        r = json.load(open(fn))
        out.append({k: r[k] for k in ("label", "spec", "code", "bump", "check", "expected",
                                      "seconds")}
                   | {"kernel": {k: v for k, v in r["kernel"].items()
                                 if not k.startswith("curve")},
                      "kernel_curves": {k: v for k, v in r["kernel"].items()
                                        if k.startswith("curve")}})
    return out


def print_kernels(kern: list[dict]) -> None:
    if not kern:
        return
    print("\n== kernel of each bump code: measured (formula) ==")
    print(f"{'':<28s}{'fields/unit':>12s}{'half-width inh':>20s}{'raw floor':>20s}"
          f"{'inh floor':>10s}{'units>0.5':>10s}{'max diff':>10s}")
    by = {k["label"]: k for k in kern}
    for c in codes():
        k = by.get(label(c))
        if not k:
            continue
        e, m = k["expected"], k["kernel"]
        print(f"{k['label']:<28s}{k['bump']['fields_per_unit_mean']:12.1f}"
              f"{m['half_width_inh']:12.2f} ({e['half_width_inh']:5.2f})"
              f"{m['floor_raw']:12.4f} ({e['floor_raw']:.4f})"
              f"{m['floor_inh']:10.4f}{k['check']['units_above_half_per_place']:10.1f}"
              f"{k['check']['max_abs_diff_code']:10.1e}")


def main(argv: list[str]) -> int:
    root = argv[0] if argv else str(results_dir() / RESULTS)
    rect_root = argv[1] if len(argv) > 1 else str(results_dir() / "rect_ideal_grid")
    bump, ref, kern = load_bump(root), load_reference(rect_root), load_kernels(root)
    print(f"{len(bump)} bump probe runs in {root}; {len(ref)} reference runs; "
          f"{len(kern)} kernel checks")
    print_tables(bump, ref)
    print_kernels(kern)
    with open(os.path.join(root, "bump_results.json"), "w") as fh:
        json.dump({"probe": bump, "reference": ref, "kernel": kern, "Ks": list(KS)}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
