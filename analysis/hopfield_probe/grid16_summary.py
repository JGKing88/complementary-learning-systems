"""Markdown tables for the grid16 run (``grid16_check.py``).

    python -m analysis.hopfield_probe.grid16_summary OUT [--k 5]

Reads ``OUT/json/*.json`` and prints, per layout, the memory table and the
navigation table, plus the per-alpha sweep for each condition.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

SAT_NAME = {"unsat": "unsat", "sat": "sat (enc+recall)", "rsat": "recall-only sat"}
ORDER = [(s, e, st) for s in ("unsat", "sat", "rsat")
         for e in ("ideal", "att0.5") for st in ("hebb", "proj")]


def load(out):
    rows = {}
    for p in glob.glob(os.path.join(out, "json", "*.json")):
        with open(p) as f:
            r = json.load(f)
        rows[(r["encoder"], r["sat"], r["storage"], r["layout"], r["k"])] = r
    return rows


def _m(xs, key, sub=None):
    v = [x[sub][key] if sub else x[key] for x in xs]
    v = [np.nan if a is None else a for a in v]
    return float(np.nanmean(v)) if len(v) else float("nan")


def memory_row(r):
    best = next(s for s in r["sweep"] if s["alpha"] == r["best_alpha"])
    any_int = [s["alpha"] for s in r["sweep"] if s["interpolates"]]
    snaps = [s["alpha"] for s in r["sweep"] if s["snap"]]
    fp = r["fixed_point"]["alpha1"]
    fpb = r["fixed_point"]["best"]
    d = r["disc"]
    c = "converged"
    fixed = fp["30"] >= 0.999
    near = _m(d, "near_exact", c)
    self_ex = np.mean([x[c]["self_exact"] for x in d])
    land = _m(d, "self_land_dist", c)
    return {
        "best_alpha": r["best_alpha"],
        "interp": f"{'y' if best['interpolates'] else 'n'} ({best['mincos_in_mean']:.3f})",
        "interp_alphas": ",".join(f"{a:g}" for a in any_int) or "none",
        "snap_alphas": ",".join(f"{a:g}" for a in snaps) or "none",
        "fixed": f"{'y' if fixed else 'n'} " + "/".join(
            f"{fp[s]:.3f}" for s in ("1", "5", "15", "30")),
        "fixed_best": f"{fpb[str(r['T_MAX'])]:.3f}",
        "correct": f"{'y' if self_ex == 1 and near >= 0.99 else 'n'}: self {self_ex:.2f}, "
                   f"near {near:.2f}, off {land:.2f}, cos {_m(d, 'self_cos_goal', c):.3f}",
        "basin": f"{_m(d, 'r_exact_all', c):.1f} / {_m(d, 'r_exact_95', c):.1f}",
        "basin1": f"{_m(d, 'r_exact_all', 'one_step'):.1f} / {_m(d, 'r_exact_95', 'one_step'):.1f}",
        "conv": f"{best['frac_converged']:.2f}",
    }


def nav_rows(r):
    out = []
    for rd in ("a", "b"):
        for t in ("one_step", "converged"):
            key = f"{rd}|{t}"
            e = r["nav_env"][key]
            box = r["nav_box"]
            out.append({
                "readout": {"a": "(a) q", "b": "(b) grad"}[rd],
                "timing": {"one_step": "1 step", "converged": f"walk α={r['best_alpha']:g}"}[t],
                "acc45": e["acc45"], "err": e["abs_err"], "reach": e["reach"],
                "basin": _m(box, "r_reach_all", key) if box else float("nan"),
                "basin95": _m(box, "r_reach_95", key) if box else float("nan"),
                "n_box": len(box),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    rows = load(args.out)
    for k in (5, 20):
        for lay in ("whole", "region"):
            print(f"\n### Memory, {lay}, K = {k}\n")
            print("| encoder | sat | storage | best α | interpolates (min cos) | α that interpolate | α that snap "
                  "| fixed point α=1 (cos s1/5/15/30) | cos self @T, best α | correct fixed point | basin ii 100/95 (walk) | basin ii (1 step) |")
            print("|" + "---|" * 12)
            for s, e, st in ORDER:
                r = rows.get((e, s, st, lay, k))
                if r is None:
                    print(f"| {e} | {SAT_NAME[s]} | {st} | missing |" + " |" * 8)
                    continue
                m = memory_row(r)
                print(f"| {e} | {SAT_NAME[s]} | {st} | {m['best_alpha']:g} | {m['interp']} | {m['interp_alphas']} "
                      f"| {m['snap_alphas']} | {m['fixed']} | {m['fixed_best']} | {m['correct']} | {m['basin']} | {m['basin1']} |")
            print(f"\n### Navigation, {lay}, K = {k}\n")
            print("| encoder | sat | storage | readout | recall | acc45 | \\|err\\| ° | basin iii 100/95 | reach |")
            print("|" + "---|" * 9)
            for s, e, st in ORDER:
                r = rows.get((e, s, st, lay, k))
                if r is None:
                    continue
                for n in nav_rows(r):
                    print(f"| {e} | {SAT_NAME[s]} | {st} | {n['readout']} | {n['timing']} | {n['acc45']:.3f} "
                          f"| {n['err']:.1f} | {n['basin']:.1f} / {n['basin95']:.1f} | {n['reach']:.3f} |")
    print("\n### alpha sweeps\n")
    for key in sorted(rows):
        r = rows[key]
        print(f"\n{r['label']}")
        for s in r["sweep"]:
            tr = s["traj"]
            seq = " ".join("-" if tr[t]["dist"] is None else f"{tr[t]['dist']:.2f}"
                           for t in ("1", "2", "3", "5", "8", "12", "20", "30", "60", "100", "200", "600") if t in tr)
            print(f"  α={s['alpha']:<6g} interp={int(s['interpolates'])} snap={int(s['snap'])} mono {s['frac_monotone_in']:.2f} "
                  f"mincos_in {s['mincos_in_mean']:.3f} (p10 {s['mincos_in_p10']:.3f}) jump {s['jump_median']:.2f}"
                  f"@{s['jump_step_median']} dmin {s['dmin_mean']:.2f} first_hit {s['first_hit_median']} final_exact "
                  f"{s['final_exact']:.3f} final_d {s['final_dist_mean']} conv {s['frac_converged']:.2f} | {seq}")


if __name__ == "__main__":
    main()
