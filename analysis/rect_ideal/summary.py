"""Tables and one JSON for the rectified-ideal grid.

    python -m analysis.rect_ideal.summary [ROOT]

ROOT defaults to ``$CLS_RESULTS/rect_ideal_grid``. Reads ``ROOT/probe/t*/*.json`` (probe runs)
and ``ROOT/fields/*.json`` (place fields), prints one f x D table per measure, and writes
``ROOT/grid_results.json``, which the tuning page reads. Missing runs are skipped, so it can run
while jobs are still landing.

Every probe measure is recorded at every K (1, 3, 5, 10, 20) after one recall step:

    err     mean |angle(q, bearing to goal)|, degrees, over the arena's non-goal cells
    acc45   share of those cells with error under 45 degrees (acc90: under 90)
    exact   share of the arena's cells whose recall decodes to the goal cell itself
    basin   largest radius r with every scaffold cell within r of the goal decoding to it
            (first failure, cues out to 64 cells); basin95: the same at 95% of cues
    disc    reach rate of the q walk on the lattice (one cardinal move per step)
    cont    reach rate of the continuous q walk (unit steps, <= 80, arrival within 0.5)
    dead    share of scored arenas (first 3 per world) with cont reach below 0.5
    s15     acc45 after 15 recall steps (K = 5 only in the tables)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

from analysis.rect_ideal.codes import DS, FS, f_tag
from cls_paths import results_dir

KS = ("1", "3", "5", "10", "20")
S = "1"
MEASURES = ("err", "acc45", "acc90", "exact", "basin", "basin95", "disc", "cont",
            "cont_steps", "dead", "s15")


def _m(d, key):
    v = d.get(key)
    return v.get("mean") if isinstance(v, dict) else v


def probe_row(r: dict) -> dict:
    out = {}
    for k in KS:
        if k not in r["test_bc"]["k"]:
            continue
        bc = r["test_bc"]["k"][k]["per_step"]
        ta = r["test_a"]["k"][k]["per_step"][S]["scalars"]
        td = r["test_d"]["k"][k][S]
        g = bc[S]["grid"]["scalars"]
        vals = td["continuous"]["scalars"]["reach_rate"]["values"]
        m = min(int(k), r["config"].get("n_score_envs", 5))
        sel = [x for i, x in enumerate(vals) if (i % m) in (0, 1, 2)]
        out[k] = {
            "err": _m(g, "abs_err_mean"), "acc45": _m(g, "acc45"), "acc90": _m(g, "acc90"),
            "exact": _m(ta, "exact_frac"), "basin": _m(ta, "r_exact_all"),
            "basin95": _m(ta, "r_exact_95"),
            "disc": _m(td["discrete"]["scalars"], "reach_rate"),
            "cont": _m(td["continuous"]["scalars"], "reach_rate"),
            "cont_steps": _m(td["continuous"]["scalars"], "mean_steps"),
            "dead": float(np.mean([x < 0.5 for x in sel])) if sel else None,
            "s15": _m(bc["15"]["grid"]["scalars"], "acc45") if "15" in bc else None,
        }
    return out


LABEL = re.compile(r"^(?:ideal D(?P<Di>\d+)|rect f(?P<f>\d+) D(?P<D>\d+)(?P<inh> inh)?)$")


def parse_label(lab: str) -> dict:
    m = LABEL.match(lab)
    if not m:
        raise ValueError(lab)
    if m["Di"]:
        return {"kind": "ideal", "f": None, "D": int(m["Di"]), "inh": False}
    return {"kind": "rect", "f": int(m["f"]) / 100, "D": int(m["D"]), "inh": bool(m["inh"])}


def load(root: str) -> dict:
    probe, fields = [], []
    for fn in sorted(glob.glob(f"{root}/probe/t*/*.json")):
        if fn.endswith("manifest.json"):
            continue
        r = json.load(open(fn))
        lab = r["header"]["label"]
        try:
            probe.append({"label": lab, **parse_label(lab), "spec": r["header"].get("path"),
                          "K": probe_row(r)})
        except KeyError as e:
            print(f"  skip {fn}: missing {e}")
    for fn in sorted(glob.glob(f"{root}/fields/*.json")):
        r = json.load(open(fn))
        k = r["kernel"]
        fields.append({
            "label": r["label"], **parse_label(r["label"]), "spec": r["spec"], "r": r["r"],
            "thresh": r["thresh"], "check": r["check"], "peak": r["peak"],
            "kernel": {key: v for key, v in k.items() if not key.startswith("curve")},
            "kernel_curves": {key: v for key, v in k.items() if key.startswith("curve")},
            "support": r["support"], "peak20": r["peak20"], "seconds": r["seconds"],
        })
    return {"probe": probe, "fields": fields, "Fs": list(FS), "Ds": list(DS),
            "Ks": [int(k) for k in KS]}


def _fmt(v, w=7, p=2):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) and v == v else f"{'-':>{w}s}"


def print_tables(res: dict, K: str = "5") -> None:
    idx = {(p["kind"], p["f"], p["D"], p["inh"]): p for p in res["probe"]}
    head = "".join(f"{'D' + str(D):>8s}" for D in DS)
    for meas, p in (("err", 1), ("acc45", 3), ("exact", 3), ("basin", 1), ("basin95", 1),
                    ("disc", 3), ("cont", 3), ("dead", 2), ("s15", 3)):
        print(f"\n== {meas} at K={K}, one step ==     {head}")
        row = idx
        line = "".join(_fmt(row.get(("ideal", None, D, False), {}).get("K", {}).get(K, {})
                            .get(meas), 8, p) for D in DS)
        print(f"{'ideal (signed)':<30s}{line}")
        for inh in (False, True):
            for f in FS:
                line = "".join(_fmt(row.get(("rect", f, D, inh), {}).get("K", {}).get(K, {})
                                    .get(meas), 8, p) for D in DS)
                print(f"{f_tag(f) + (' + inhibition' if inh else ''):<30s}{line}")
    if any(p["K"].get("20") for p in res["probe"]):
        print("\n== dead goals at K = 20 ==")
        for inh in (False, True):
            for f in FS:
                line = "".join(_fmt(row.get(("rect", f, D, inh), {}).get("K", {}).get("20", {})
                                    .get("dead"), 8, 2) for D in DS)
                print(f"{f_tag(f) + (' + inhibition' if inh else ''):<30s}{line}")

    fidx = {(q["f"], q["D"]): q for q in res["fields"]}
    for title, get, p in (
            ("support fields per unit, mean", lambda q: q["support"]["fields_per_unit"]["mean"], 0),
            ("support field diameter in grid periods, median",
             lambda q: q["support"]["diam_over_lambda"]["median"], 2),
            ("support field diameter in grid periods, mean", lambda q: q["support"]["diam_over_lambda"]["mean"], 2),
            ("support field diameter in grid periods, p95", lambda q: q["support"]["diam_over_lambda"]["p95"], 2),
            ("share of support fields under 9 cells", lambda q: q["support"]["share_fields_under9"], 2),
            ("largest field's share of a unit's active area",
             lambda q: q["support"]["largest_field_share"]["mean"], 2),
            ("20%-of-peak fields per unit, mean", lambda q: q["peak20"]["fields_per_unit"]["mean"], 0),
            ("20%-of-peak diameter in grid periods, median", lambda q: q["peak20"]["diam_over_lambda"]["median"], 2),
            ("kernel half-width after inhibition (cells)", lambda q: q["kernel"]["half_width_inh"], 1),
            ("signed ideal r=16 half-width at this D", lambda q: q["kernel"]["half_width_ideal_r16"], 1)):
        if not res["fields"]:
            break
        print(f"\n== {title} =={'':6s}{head}")
        for f in FS:
            line = "".join(_fmt(get(fidx[(f, D)]) if (f, D) in fidx else None, 8, p) for D in DS)
            print(f"{f_tag(f):<30s}{line}")


def main(argv: list[str]) -> int:
    root = argv[0] if argv else str(results_dir() / "rect_ideal_grid")
    res = load(root)
    print(f"{len(res['probe'])} probe runs, {len(res['fields'])} field runs in {root}")
    print_tables(res)
    with open(os.path.join(root, "grid_results.json"), "w") as fh:
        json.dump(res, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
