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
    # Snap = stall then a single jump covering >= 80% of the path, at step >= 2.
    # The cos dip is printed with it rather than required (grid16_check's
    # `snap` flag also required a dip < 0.95; the binary ideal code dips only
    # to ~0.96).
    snaps = [f"{s['alpha']:g} ({s['mincos_in_mean']:.3f})" for s in r["sweep"]
             if s["alpha"] < 1 and s["jump_median"] >= 0.8
             and (s["jump_step_median"] or 0) >= 2 and s["dmin_mean"] <= 1.0]
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
        "snap_alphas": ", ".join(snaps) or "none",
        "fixed": f"{'y' if fixed else 'n'} " + "/".join(
            f"{fp[s]:.3f}" for s in ("1", "5", "15", "30")),
        "fixed_best": f"{fpb[str(r['T_MAX'])]:.3f}",
        "correct": f"{'y' if self_ex == 1 and near >= 0.99 else 'n'}: self {self_ex:.2f}, "
                   f"near {near:.2f}, off {land:.2f}, cos {_m(d, 'self_cos_goal', c):.3f}",
        "basin": _basin(d, c, r["BASIN_R"]),
        "basin1": _basin(d, "one_step", r["BASIN_R"]),
        "conv": f"{best['frac_converged']:.2f}",
    }


def _r(d, tag, key, cap):
    """Mean radius over goals; '≥' when any goal hit the cap (a lower bound)."""
    hit = any(x[tag][key] >= cap for x in d)
    return f"{'≥' if hit else ''}{_m(d, key, tag):.1f}"


def _basin(d, tag, cap, k1="r_exact_all", k2="r_exact_95"):
    """mean 100% / 95% radius, each marked '≥' if any goal hit the cap."""
    return f"{_r(d, tag, k1, cap)} / {_r(d, tag, k2, cap)}"


def best_tag(r):
    how = r.get("best_alpha_how")
    a = f"{r['best_alpha']:g}"
    return a if how in (None, "interpolating") else f"{a} (none interp.)"


def fp_cells(sf):
    """('goal is a fixed point?', 'where self-recall settles') for one
    self_fixed_point result -- terse.

    Self-recall always stops somewhere under Hebbian storage (power iteration
    onto a cue-independent state), so "the state stopped" alone says nothing
    about the goal; the first cell is y only if every goal stops on itself.
    """
    if sf["correct"] >= 1.0 and sf["exists"] >= 1.0:
        return "y", "goal"
    where = []
    if sf["exists"] < 1.0:
        where.append(f"{1 - sf['exists']:.0%} still moving")
    if sf["off_mean_wrong"] is not None:
        where.append(f"{sf['off_mean_wrong']:.1f} cells off")
    if sf["other_env"] > 0:
        where.append(f"{sf['other_env']:.0%} other env")
    where.append(f"cos {sf['cos_goal_wrong']:.3f}")
    return "n", ", ".join(where)


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
                "basin": _basin([{"x": b[key]} for b in box], "x", r["BOX_R"],
                                "r_reach_all", "r_reach_95") if box else "n/a",
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
                          f"| {n['err']:.1f} | {n['basin']} | {n['reach']:.3f} |")
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


# --- compact tables for the EXPERIMENTS doc (``--doc``) -----------------------

def doc_memory(rows, lay, k, compact=False):
    """One row per condition, rsat side rows included.

    'Goal is fixed pt' / 'settles at' come from ``self_fixed_point``
    (self-recall from the stored goal) at alpha = 1 and at the chosen alpha; 'near-goal' is the
    second line of evidence -- starts within 2 cells, converged walk, fraction
    ending exactly on the goal; basin ii is after the walk.
    """
    head = ["encoder · sat · storage", "best α", "interpolates (min cos)"]
    if not compact:
        head.append("snaps at α (dip cos)")
    head += ["goal is fixed pt α=1", "settles at α=1", "goal is fixed pt best α",
             "settles at best α",
             "goals correct α=1 / best", "near-goal exact", "basin ii 100/95"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s, e, st in ORDER:
        r = rows[(e, s, st, lay, k)]
        m = memory_row(r)
        d, c = r["disc"], "converged"
        f1, c1 = fp_cells(r["self_fixed"]["alpha1"])
        fb, cb = fp_cells(r["self_fixed"]["best"])
        cells = [f"{e} · {s} · {st}", best_tag(r), m["interp"]]
        if not compact:
            cells.append(m["snap_alphas"])
        cells += [f1, c1, fb, cb,
                  f"{r['self_fixed']['alpha1']['correct']:.2f} / "
                  f"{r['self_fixed']['best']['correct']:.2f}",
                  f"{_m(d, 'near_exact', c):.2f}",
                  _basin(d, c, r["BASIN_R"])]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def doc_nav(rows, lay, k):
    """One row per condition; each cell acc45 / |err|° / reach / basin iii
    (100% first-failure radius on the goal-centred box; '≥' = at the cap)."""
    head = ["encoder · sat · storage", "best α", "(a) q, 1 step", "(a) q, after walk",
            "(b) grad, 1 step", "(b) grad, after walk"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s, e, st in ORDER:
        r = rows[(e, s, st, lay, k)]
        box = r["nav_box"]
        cells = [f"{e} · {s} · {st}", best_tag(r)]
        for key in ("a|one_step", "a|converged", "b|one_step", "b|converged"):
            v = r["nav_env"][key]
            bas = (_r([{"x": b[key]} for b in box], "x", "r_reach_all", r["BOX_R"])
                   if box else "n/a")
            cells.append(f"{v['acc45']:.3f} / {v['abs_err']:.1f} / {v['reach']:.2f} / {bas}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def doc_main(out):
    rows = load(out)
    for lay in ("whole", "region"):
        print(f"\n#### K = 5, {lay}: memory\n\n" + doc_memory(rows, lay, 5))
        print(f"\n#### K = 5, {lay}: navigation\n\n" + doc_nav(rows, lay, 5))
    for lay in ("whole", "region"):
        print(f"\n#### K = 20, {lay}: memory\n\n" + doc_memory(rows, lay, 20, compact=True))
        print(f"\n#### K = 20, {lay}: navigation\n\n" + doc_nav(rows, lay, 20))


if __name__ == "__main__":
    import sys
    if "--doc" in sys.argv:
        doc_main([a for a in sys.argv[1:] if a != "--doc"][0])
    else:
        main()
