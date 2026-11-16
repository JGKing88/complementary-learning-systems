"""Score each population's measurements against the literature targets.

    python -m analysis.coding_comparison.scorecard [--tag v1]

Reads ``<results>/coding_comparison/<tag>/<pop>/*.json`` and
``literature_targets.json`` and writes ``scorecard.json`` and
``scorecard.md`` beside the population directories.

A cell of the scorecard is one (measure, region) pair. Its verdict comes from
the region's primary target for that measure:

``C``   consistent: the model value lies in the target's band
``I``   inconsistent
``U``   untestable: the review gives no number for this region and measure
``U*``  untestable: the model has no value (for example, no unit has a field)

Bands follow ``literature_targets.json``'s policy: a reported range exactly,
a point within +-30% (or +-0.1 for correlations), a bound as stated.

Each row says which field definition it uses. Rows matched to Rich 2014 or
to other absolute-rate rules use definition B at the primary mean rate
(0.75 Hz). Rows matched to Kjelstrup 2008's 20%-of-peak rule use A; rows
matched to Eliav 2021 (20% of the field peak, field peaks above 1 Hz) use Ap. A trailing ``~`` marks a verdict that changes somewhere in the 0.5 to 1 Hz
sweep, and a trailing ``r`` one that changes under a random rotation of the
units (signed codes only).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from cls_paths import results_dir

HERE = Path(__file__).resolve().parent
TARGETS = HERE / "literature_targets.json"
PRIMARY_RATE = "0.75"
RATES = ("0.5", "0.75", "1")
POP_ORDER = ["grid", "ideal", "ideal_rot", "ideal_sparse", "trained", "untrained", "hippocampus"]
REGION_SHORT = {
    "dorsal CA1": "dCA1", "dorsal CA3": "dCA3", "intermediate CA3": "iCA3",
    "ventral CA3": "vCA3", "dentate granule": "DG gc", "dentate mossy": "DG mc",
    "CA2": "CA2", "subiculum": "SUB", "bat CA1": "bat CA1", "bat CA3": "bat CA3",
}


def load_targets() -> dict:
    with open(TARGETS) as f:
        return json.load(f)


def rich_zero_fraction(L_W: float, T: dict | None = None) -> float:
    """Rich 2014's fitted P(no field) on a track ``L_W`` mean field widths long."""
    r = (T or load_targets())["rich2014"]
    return (r["beta_m"] / (r["beta_m"] + r["field_m"] * L_W)) ** r["r"]


# ---------------------------------------------------------------------------
# Extracting model values
# ---------------------------------------------------------------------------

def _get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _f1(R, v, L, *path):
    return _get(R.get("fields_1d"), "variants", v, "lengths", L, *path)


def _f2row(R, v, k):
    rows = _get(R.get("fields_2d"), "variants", v, "rows") or []
    for row in rows:
        if row["k"] == k:
            return row
    return None


def _nb_r(R, v):
    nb = _f1(R, v, "73.8", "fit", "negbin")
    if nb is None:
        return None
    return math.inf if nb.get("poisson_limit") else nb.get("r")


def _zero_curve(R, v, T):
    lens = _get(R.get("fields_1d"), "variants", v, "lengths") or {}
    devs = [abs(x["fit"]["frac_zero"] - rich_zero_fraction(x["length_W"], T))
            for x in lens.values() if x["length_W"] <= 74]
    return max(devs) if len(devs) >= 3 else None


def _kernel(R, construction, summary, field):
    return _get(R.get("kernel_1d"), "constructions", construction, "summaries", summary, field)


def _zero(R, construction, summary):
    s = _get(R.get("kernel_1d"), "constructions", construction, "summaries", summary)
    if s is None:
        return None
    return s["zero_crossing_W"] if s["zero_crossing_W"] is not None else math.inf


def _cos(R, b, field):
    """Raw-code cosine kernel (the same in every basis) over the basis's own W_A."""
    s = _get(R.get("kernel_1d"), "constructions", "raw|cosine", "summaries",
             f"{b.split('|')[-1]}|A")
    W = _get(R.get("meta"), "calibration", f"{b}|A", "W")
    if s is None or not W:
        return None
    x = s[field]
    return math.inf if x is None else x / W


def _rows2(R, v):
    return [r for r in (_get(R.get("fields_2d"), "variants", v, "rows") or [])
            if r.get("fields_per_active") is not None]


def _area_r(R, v):
    """Pearson r of fields per active unit against area (Tanni and Harland count place cells)."""
    rows = _rows2(R, v)
    if len(rows) < 3:
        return None
    a = np.array([r["area_cells"] for r in rows], dtype=float)
    f = np.array([r["fields_per_active"] for r in rows], dtype=float)
    return float(np.corrcoef(a, f)[0, 1]) if f.std() > 0 else None


def _area_ratio(R, v):
    rows = {r["k"]: r["fields_per_active"] for r in _rows2(R, v)}
    for k in sorted(rows):
        if 8 * k in rows and rows[k] > 0:
            return rows[8 * k] / rows[k]
    return None


def _at_Af(R, v, target):
    """Fields per active unit where the arena is ``target`` mean fields (Fenton 2008), log-interpolated."""
    pts = sorted((r["arena_over_Af"], r["fields_per_active"]) for r in _rows2(R, v)
                 if r.get("arena_over_Af"))
    if len(pts) < 2 or not (pts[0][0] <= target <= pts[-1][0]):
        return None
    x = np.log([p[0] for p in pts])
    return float(np.interp(math.log(target), x, [p[1] for p in pts]))


def _remap(R, b, *path):
    return _get(R.get("remapping"), "bases", b, *path)


def _sparsity(R, b, L):
    return _get(R.get("sparsity"), "tracks", "by_basis", b, L)


def _dim(R, b):
    return _get(R.get("phase2"), "corr_dim", b, "dimension")


# Each row: key, label, the definition its targets are matched to ("A", "B",
# or None for definition-free rows), and how to read the model's value given
# (results, basis, variant key). ``targets`` names the target row it scores
# against when that differs from its own key.
ROWS = [
    ("fields_per_unit_74W", "Fields per unit, track of 74 field widths", "B",
     lambda R, b, v, T: _f1(R, v, "73.8", "fit", "mean")),
    ("nb_r_74W", "Negative binomial shape r, 74 widths", "B", lambda R, b, v, T: _nb_r(R, v)),
    ("var_over_mean_74W", "Variance / mean of field counts, 74 widths", "B",
     lambda R, b, v, T: _f1(R, v, "73.8", "fit", "var_over_mean")),
    ("nb_preferred_74W", "Negative binomial beats Poisson (LR p < 0.01), 74 widths", "B",
     lambda R, b, v, T: (None if _f1(R, v, "73.8", "fit", "lr_p") is None
                         else bool(_f1(R, v, "73.8", "fit", "lr_p") < 0.01))),
    ("zero_curve", "Zero-field fraction against length: max distance from Rich's curve", "B",
     lambda R, b, v, T: _zero_curve(R, v, T)),
    ("active_frac_74W", "Active fraction (one or more fields), 74 widths", "B",
     lambda R, b, v, T: _f1(R, v, "73.8", "active_frac")),
    ("gap_cv_74W", "Gap CV between a unit's fields (units with 6+), 74 widths", "B",
     lambda R, b, v, T: _f1(R, v, "73.8", "layout", "gap_cv_median")),
    ("ks_reject_74W", "Share of units whose field positions reject uniform, 74 widths", "B",
     lambda R, b, v, T: _f1(R, v, "73.8", "layout", "ks_frac_reject_05")),
    ("size_cv_74W", "Field size CV, 74 widths", "B",
     lambda R, b, v, T: _f1(R, v, "73.8", "field_size", "cv")),
    ("per_location_74W", "Per-location active fraction, 74 widths", "B",
     lambda R, b, v, T: _f1(R, v, "73.8", "coverage_per_location")),
    ("size_cv_33W", "Field size CV, 34 widths", "Ap",
     lambda R, b, v, T: _f1(R, v, "33.8", "field_size", "cv")),
    ("size_ratio_33W", "Largest/smallest field within a unit, 34 widths", "Ap",
     lambda R, b, v, T: _f1(R, v, "33.8", "size_ratio", "mean")),
    ("gap_cv_33W", "Gap CV between a unit's fields, 34 widths", "Ap",
     lambda R, b, v, T: _f1(R, v, "33.8", "layout", "gap_cv_median")),
    ("fields_per_active_33W", "Fields per active unit, 34 widths", "Ap",
     lambda R, b, v, T: _f1(R, v, "33.8", "fields_per_active")),
    ("single_frac_33W", "Single-field share of active units, 34 widths", "Ap",
     lambda R, b, v, T: _f1(R, v, "33.8", "single_field_frac_active")),
    ("five_plus_frac_33W", "Share of active units with 5+ fields, 34 widths", "Ap",
     lambda R, b, v, T: _f1(R, v, "33.8", "five_plus_frac_active")),
    ("active_frac_33W", "Active fraction, 34 widths", "Ap",
     lambda R, b, v, T: _f1(R, v, "33.8", "active_frac")),
    ("area_linear_r", "Fields per active unit against arena area: Pearson r", "B",
     lambda R, b, v, T: _area_r(R, v)),
    ("area_ratio_8x", "Fields per active unit, ratio over an 8-fold area step (not scored)", "B",
     lambda R, b, v, T: _area_ratio(R, v)),
    ("fields_at_3p7Af", "Fields per active unit in an arena of 3.7 mean fields", "A",
     lambda R, b, v, T: _at_Af(R, v, 3.7)),
    ("fields_at_13p7Af", "Fields per active unit in an arena of 13.7 mean fields", "A",
     lambda R, b, v, T: _at_Af(R, v, 13.7)),
    ("fields_per_active_small", "Fields per active unit, small arena (8 A0)", "B",
     lambda R, b, v, T: (_f2row(R, v, 8) or {}).get("fields_per_active")),
    ("active_frac_small", "Active fraction, small arena (8 A0)", "B",
     lambda R, b, v, T: (_f2row(R, v, 8) or {}).get("active_frac")),
    ("per_location_small", "Per-location active fraction, small arena (8 A0)", "B",
     lambda R, b, v, T: (_f2row(R, v, 8) or {}).get("coverage_per_location")),
    ("sparsity_small", "Treves-Rolls sparsity, track of 4.6 widths", None,
     lambda R, b, v, T: _sparsity(R, b, "4.6")),
    ("kernel_fullwidth_W", "PV kernel full width at r = 0.5 / W_A (Pearson, rates)", "A",
     lambda R, b, v, T: _kernel(R, f"{b}|pearson", f"{b.split('|')[-1]}|A", "full_width_05_W")),
    ("kernel_zero_W", "PV kernel zero crossing / W_A (Pearson, rates)", "A",
     lambda R, b, v, T: _zero(R, f"{b}|pearson", f"{b.split('|')[-1]}|A")),
    ("kernel_floor", "PV kernel floor, 5 to 15 W_A (Pearson, rates)", "A",
     lambda R, b, v, T: _kernel(R, f"{b}|pearson", f"{b.split('|')[-1]}|A", "floor")),
    ("kernel_fullwidth_W_cos", "Cosine kernel full width at 0.5 / W_A (raw code)", "A",
     lambda R, b, v, T: _cos(R, b, "full_width_05"), "kernel_fullwidth_W"),
    ("kernel_zero_W_cos", "Cosine kernel zero crossing / W_A (raw code)", "A",
     lambda R, b, v, T: _cos(R, b, "zero_crossing"), "kernel_zero_W"),
    ("remap_pv", "PV correlation between arenas at matched places (Pearson, rates)", None,
     lambda R, b, v, T: _remap(R, b, "pv_matched", "pearson", "median")),
    ("remap_overlap_ratio", "Active-set overlap / chance between arenas", "B",
     lambda R, b, v, T: _remap(R, b, "overlap", v.split("|")[-1], "both_ratio", "median")),
    ("remap_rate_corr", "Mean-rate correlation across arenas", None,
     lambda R, b, v, T: _remap(R, b, "mean_rate_corr", "pearson", "median")),
    ("remap_lograte_corr", "Log mean-rate correlation across arenas", None,
     lambda R, b, v, T: _remap(R, b, "mean_rate_corr", "log_pearson", "median")),
    ("remap_map_above_null", "Units whose rate maps beat a shuffle (95th pct) across arenas", None,
     lambda R, b, v, T: _remap(R, b, "ratemap", "frac_above_null_p95", "median")),
    ("corr_dim", "Correlation dimension of PVs in an arena (phase 2)", None,
     lambda R, b, v, T: _dim(R, b)),
]


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def band(t: dict, policy: dict) -> tuple[float | None, float | None]:
    k = t["kind"]
    if k == "range":
        return t["low"], t["high"]
    if k == "point":
        v = t["value"]
        if t.get("tol") == "abs":
            return v - policy["abs_tol"], v + policy["abs_tol"]
        d = abs(v) * policy["rel_tol"]
        return v - d, v + d
    if k == "upper":
        return None, t["value"]
    if k == "lower":
        return t["value"], None
    if k == "curve":
        return 0.0, t["band_abs"]
    return None, None


def judge(t: dict, value, policy: dict) -> str:
    if t["kind"] == "info":
        return "info"
    if value is None:
        return "U*"
    if t["kind"] == "bool":
        return "C" if bool(value) == bool(t["value"]) else "I"
    lo, hi = band(t, policy)
    v = float(value)
    if math.isnan(v):
        return "U*"
    ok = (lo is None or v >= lo) and (hi is None or v <= hi)
    return "C" if ok else "I"


def band_text(t: dict, policy: dict) -> str:
    if t["kind"] == "bool":
        return "yes" if t["value"] else "no"
    if t["kind"] == "info":
        return f"{t['value']} (not scored)"
    lo, hi = band(t, policy)
    fmt = (lambda x: f"{x:.3g}")
    if lo is None:
        return f"<= {fmt(hi)}"
    if hi is None:
        return f">= {fmt(lo)}"
    return f"{fmt(lo)} to {fmt(hi)}"


def fmt_value(x) -> str:
    if x is None:
        return "none"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.3g}"


# ---------------------------------------------------------------------------
# Scoring a population
# ---------------------------------------------------------------------------

def load_results(pop_dir: Path) -> dict:
    out = {}
    for f in pop_dir.glob("*.json"):
        with open(f) as fh:
            out[f.stem] = json.load(fh)
    return out


def bases_of(R: dict) -> list[str]:
    return ["rectify", "onoff"] if R["meta"]["signed"] else ["identity"]


def variant_key(basis: str, defn: str | None, rate: str = PRIMARY_RATE) -> str:
    if defn is None:
        return f"{basis}|A"
    return f"{basis}|{defn}" if defn in ("A", "Ap") else f"{basis}|B{rate}"


# Rows whose value is a well-defined 0 when the definition finds no field on
# any of the calibration loops (8 full scaffold rows and columns): no unit is
# active anywhere, so none is active on a shorter track or in an arena either.
ZERO_ROWS = {"fields_per_unit_74W", "active_frac_74W", "per_location_74W", "active_frac_33W",
             "active_frac_small", "per_location_small"}


def silent(R: dict, v: str) -> bool:
    c = _get(R.get("meta"), "calibration", v)
    if not c or c.get("W") is not None or not c.get("fdef"):
        return False
    hist = c.get("history")
    return (c.get("n_fields") == 0) or bool(hist and hist[-1].get("n_fields") == 0)


def value_of(fn, key, R, b, v, T):
    x = fn(R, b, v, T)
    if x is None and key in ZERO_ROWS and silent(R, v):
        return 0.0
    return x


def score_population(R: dict, T: dict) -> dict:
    policy = T["policy"]
    rot_seeds = R["meta"]["params"]["rot_seeds"] if R["meta"]["signed"] else []
    by_row: dict = {}
    for t in T["targets"]:
        by_row.setdefault(t["row"], []).append(t)
    out = {}
    for basis in bases_of(R):
        rows = []
        for row in ROWS:
            key, label, defn, fn = row[:4]
            target_row = row[4] if len(row) > 4 else key
            v = variant_key(basis, defn)
            value = value_of(fn, key, R, basis, v, T)
            # Robustness: the same row at the other mean rates and in rotated bases.
            sweep = {r: value_of(fn, key, R, basis, variant_key(basis, defn, r), T)
                     for r in RATES} if defn == "B" else {}
            rots = {}
            for s in rot_seeds:
                rb = f"rot{s}|{basis}"
                rv = variant_key(rb, defn, PRIMARY_RATE if defn != "B" else "0.75")
                rots[s] = value_of(fn, key, R, rb, rv, T)
            cells = {}
            for t in by_row.get(target_row, []):
                verdict = judge(t, value, policy)
                flags = ""
                if sweep and t["kind"] != "info":
                    vs = {judge(t, x, policy) for x in sweep.values()}
                    if len(vs - {"U*"}) > 1 or (verdict != "U*" and "U*" in vs and len(vs) > 1):
                        flags += "~"
                if rots and t["kind"] != "info":
                    vr = {judge(t, x, policy) for x in rots.values() if x is not None}
                    if vr and vr != {verdict}:
                        flags += "r"
                cells.setdefault(t["region"], []).append({
                    "id": t["id"], "primary": t["primary"], "verdict": verdict, "flags": flags,
                    "band": band_text(t, policy), "tag": t["tag"], "kind": t["kind"]})
            rows.append({"row": key, "label": label, "definition": defn,
                         "variant": v if defn else basis, "value": value,
                         "sweep": sweep, "rotations": rots, "cells": cells})
        out[basis] = rows
    return out


def summarize(card: dict, regions: list[str]) -> dict:
    """Primary-target verdict counts per region."""
    out = {}
    for basis, rows in card.items():
        cnt = {r: {"C": 0, "I": 0, "U*": 0} for r in regions}
        for row in rows:
            for region, cells in row["cells"].items():
                for c in cells:
                    if c["primary"] and c["verdict"] in cnt[region]:
                        cnt[region][c["verdict"]] += 1
        out[basis] = cnt
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

LEGEND = ("Cells: C consistent, I inconsistent, U no literature number, U* no model value; "
          "~ the verdict changes within the 0.5 to 1 Hz sweep, r it changes under a random rotation. "
          "Brackets give the band and the source tag: ft full text, abs abstract, fig figure-read, "
          "der derived, inf inference, rev the review's summary table, asm assumed formula, "
          "sec secondary.")

TAG_CODES = [("full text", "ft"), ("abstract", "abs"), ("figure-read", "fig"),
             ("derived", "der"), ("infer", "inf"), ("review summary", "rev"),
             ("assume", "asm"), ("secondary", "sec")]


def tag_code(tag: str) -> str:
    """Short form of a verification tag: ft, abs, fig, der, inf, rev, asm, sec."""
    t = tag.lower()
    return "/".join(c for k, c in TAG_CODES if k in t) or "?"


def cell_text(cells: list[dict] | None, detail: bool = True) -> str:
    if not cells:
        return "U"
    prim = [c for c in cells if c["primary"]] or cells
    c = prim[0]
    if c["verdict"] == "info":
        return f"[{c['band']}]" if detail else ""
    head = c["verdict"] + c["flags"]
    return f"{head} [{c['band']}; {tag_code(c['tag'])}]" if detail else head


def population_table(rows: list[dict], regions: list[str], detail: bool = True) -> str:
    head = "| Measure | Def. | Model | " + " | ".join(REGION_SHORT[r] for r in regions) + " |"
    sep = "|---|---|---|" + "---|" * len(regions)
    lines = [head, sep]
    for row in rows:
        if not row["cells"]:
            continue
        lines.append(f"| {row['label']} | {row['definition'] or 'none'} | {fmt_value(row['value'])} | "
                     + " | ".join(cell_text(row["cells"].get(r), detail) for r in regions) + " |")
    return "\n".join(lines)


def targets_table(T: dict) -> str:
    policy = T["policy"]
    labels = {r[0]: r[1] for r in ROWS}
    lines = ["| Measure | Region | Band | Source tag | Primary |", "|---|---|---|---|---|"]
    for t in T["targets"]:
        lines.append(f"| {labels.get(t['row'], t['row'])} | {t['region']} | "
                     f"{band_text(t, policy)} | {t['tag']} | {'yes' if t['primary'] else 'no'} |")
    return "\n".join(lines)


def summary_table(cards: dict, regions: list[str]) -> str:
    head = "| Population | Convention | " + " | ".join(REGION_SHORT[r] for r in regions) + " |"
    lines = [head, "|---|---|" + "---|" * len(regions)]
    for pop, card in cards.items():
        for basis, cnt in summarize(card, regions).items():
            cells = []
            for r in regions:
                c = cnt[r]
                n = c["C"] + c["I"]
                cells.append(f"{c['C']}/{n}" if n else "U")
            lines.append(f"| {pop} | {basis} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", default="v1")
    p.add_argument("--root", type=Path, default=None)
    a = p.parse_args()
    root = a.root or (results_dir() / "coding_comparison" / a.tag)
    T = load_targets()
    regions = T["regions"]
    cards = {}
    for pop in POP_ORDER + sorted(d.name for d in root.iterdir()
                                  if d.is_dir() and d.name not in POP_ORDER and d.name != "logs"):
        d = root / pop
        if not (d / "meta.json").exists():
            continue
        cards[pop] = score_population(load_results(d), T)
    with open(root / "scorecard.json", "w") as f:
        json.dump(cards, f, indent=1, default=lambda o: None if isinstance(o, float) else str(o))
    md = ["# Coding comparison scorecard", "",
          "Primary-target counts: consistent / scored, per region.", "",
          summary_table(cards, regions), "", LEGEND, ""]
    for pop, card in cards.items():
        for basis, rows in card.items():
            md += [f"## {pop}, {basis}", "", population_table(rows, regions), ""]
    md += ["## Targets", "", targets_table(T), ""]
    (root / "scorecard.md").write_text("\n".join(md))
    print(f"wrote {root / 'scorecard.md'}")


if __name__ == "__main__":
    main()
