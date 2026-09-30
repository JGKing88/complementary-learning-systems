"""One self-contained page for every ideal-encoder result.

    python -m analysis.hopfield_probe.ideal_report OUT [--lede lede.html]

``OUT`` is the ideal_encoder results root (``run_ideal.sh`` / ``run_ideal_dyn.sh``).
The page is the standard probe report (``report.build``: Tests A-D, whole
arena) for every encoder, with one extra section in front for what that report
has no page for: the per-region probe table, the similarity scan, and the
recall dynamics (fixed points, the alpha walk, the chord) from
``ideal_dynamics_check.py``. Recomputes nothing; everything is read from JSON.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from analysis.hopfield_probe.arm_summary import row
from analysis.hopfield_probe.report import build as B
from analysis.hopfield_probe.report.figures import line_chart, table
from analysis.hopfield_probe.report.page import card, esc, run_header, single_page
from analysis.hopfield_probe.report.theme import CATEGORICAL, ordinal_colors

COL = [c[0] for c in CATEGORICAL]
REGIONS = ("whole", "corner", "centre", "opposite")
PROBE_ENCS = ["ideal r=2", "ideal r=4", "ideal r=8", "ideal r=16",
              "att0.5 s42", "att0.5 s43"]
# report label for each whole-arena result, in page order (first = primary)
REPORT = [
    ("probe", 12, "ideal r=16"),
    ("probe", 24, "ideal r=32"),
    ("probe", 28, "ideal r=48"),
    ("probe", 8, "ideal r=8"),
    ("sat", 0, "ideal r=16, saturated recall (β=1e6)"),
    ("probe", 16, "att0.5 · s42"),
    ("probe", 20, "att0.5 · s43"),
    ("sat", 4, "att0.5, saturated recall (β=1e6) · s42"),
]
DYN = ["ideal r=16 · β=100", "ideal r=16 · β=1e6", "att0.5 s42 · β=100",
       "att0.5 s42 · gain=β=1e6 (arm B)", "ideal r=32 · β=100",
       "ideal r=48 · β=100"]


def _one(path_glob):
    fs = [f for f in glob.glob(path_glob) if not f.endswith("manifest.json")]
    return json.load(open(fs[0])) if fs else None


def _fmt(v, f=".3f"):
    return "–" if v is None or (isinstance(v, float) and v != v) else format(v, f)


# ---------------------------------------------------------------------------

def probe_section(out: str) -> str:
    rows = []
    for tree, lab_sfx in (("probe", ""), ("sat", " (β=1e6)")):
        for f in sorted(glob.glob(f"{out}/{tree}/t*/*.json")):
            if f.endswith("manifest.json"):
                continue
            r = json.load(open(f))
            enc, _, reg = r["header"]["label"].partition(" | ")
            enc = enc.replace(" β=1e6", "")
            try:
                d = row(r)
            except KeyError:
                continue
            rows.append((enc + lab_sfx, reg or "whole", d))
    order = {e: i for i, e in enumerate(
        ["ideal r=16", "ideal r=32", "ideal r=48", "ideal r=16 (β=1e6)",
         "ideal r=8", "ideal r=4",
         "ideal r=2", "att0.5 s42", "att0.5 s42 (β=1e6)", "att0.5 s43"])}
    rows.sort(key=lambda x: (order.get(x[0], 99), REGIONS.index(x[1])
                             if x[1] in REGIONS else 9))
    body = [[e, reg, _fmt(d["err"], ".2f"), _fmt(d["acc"]), _fmt(d["exact"]),
             _fmt(d["basin"], ".1f"), _fmt(d["disc"]), _fmt(d["cont"]),
             _fmt(d["s15"]),
             " / ".join(_fmt(d["dead"].get(k), ".2f")
                        for k in ("1", "3", "5", "10", "20"))]
            for e, reg, d in rows]
    return card("Probe, K = 5, one recall step, every region", table(
        ["encoder", "region", "|err| °", "acc45", "exact", "basin",
         "reach disc", "reach cont", "acc45 @15", "dead @ K 1/3/5/10/20"],
        body, summary="table"),
        note="multi_env_goals memory, 8 worlds × 20 envs, env 20×20, Hebbian "
             "storage. β=1e6 rows use saturated recall (the ideal code itself "
             "has no tanh).")


def scan_section(out: str) -> str:
    body = []
    for f in sorted(glob.glob(f"{out}/scan/*.json")):
        r = json.load(open(f))
        ff, refs = r["far_field"], r["refs"]
        g = lambda k: np.array([x[k] for x in refs], dtype=float)
        body.append([r["label"], f"{ff['d_eff']:.0f}", f"{ff['far_sd']:.4f}",
                     f"{g('c1').mean():.3f}", f"{np.median(g('r_0.9')):.0f}",
                     f"{np.median(g('r_0.5')):.0f}",
                     f"{np.median(g('r_mono')):.0f}",
                     f"{np.median(g('alias')):.3f}", f"{g('alias').max():.3f}"])
    return card("Similarity structure over the whole scaffold", table(
        ["encoder", "d_eff", "far sd", "C(1)", "r_0.9", "r_0.5", "r_mono",
         "alias ceiling (median)", "alias (max)"], body, summary="table"),
        note="1/√1024 = 0.031. Alias ceiling = max cosine beyond 50 cells over "
             "the whole scaffold, per reference position. att0.5 s42's scan "
             "timed out; s43 is the reference.")


def fixed_section(dyn: dict) -> str:
    x = list(range(1, 31))
    charts = []
    for K in (5, 20):
        series = []
        for i, lab in enumerate(DYN):
            d = dyn.get(lab)
            if not d:
                continue
            for rule, dash in (("hebb", None), ("proj", "5 4")):
                fp = d["fixed_points"][f"{rule}|K={K}"]
                s = {"label": f"{lab} · {rule}", "color": COL[i],
                     "values": [fp[str(t)]["cos_self_mean"] for t in x]}
                if dash:
                    s["dash"] = dash
                series.append(s)
        charts.append(line_chart(x, series, xlabel="recall step (α = 1)",
                                 ylabel="cos(state, its own stored pattern)",
                                 title=f"K = {K}", ylim=(0.0, 1.02)))
    return card("Are the stored goals fixed points?",
                '<div class="grid2">' + "".join(charts) + "</div>",
                note="Recall started from each stored pattern itself. Solid = "
                     "Hebbian storage, dashed = projection storage (the "
                     "projector onto span(Z), diagonal kept). A fixed point "
                     "stays at its starting value.")


def walk_section(dyn: dict) -> str:
    x = list(range(1, 31))
    panels, rows = [], []
    for lab in DYN:
        d = dyn.get(lab)
        if not d:
            continue
        alphas = d["alphas"]
        cols = ordinal_colors(len(alphas))
        for rule in ("hebb", "proj"):
            series = []
            for a, c in zip(alphas, cols):
                w = d["walk"].get(f"{rule}|{a:g}")
                if not w:
                    continue
                series.append({"label": f"α = {a:g}", "color": c,
                               "values": [w["steps"][str(t)]["dist"] for t in x]})
                seq = [w["steps"][str(t)] for t in (1, 2, 3, 5, 8, 12, 20, 30)]
                rows.append([lab, rule, f"{a:g}", f"{w['start']:.2f}"]
                            + [_fmt(s["dist"], ".2f") for s in seq]
                            + [_fmt(min(w["steps"][str(t)]["cos"] for t in range(1, 13))),
                               _fmt(w["steps"]["30"]["cos"])])
            panels.append(line_chart(
                x, series, xlabel="recall step", ylabel="decoded distance to goal",
                title=f"{lab} · {rule}", ylim=(0.0, 12.0)))
    tbl = table(["encoder", "storage", "α", "start", "s1", "s2", "s3", "s5",
                 "s8", "s12", "s20", "s30", "min cos, steps 1–12", "cos at s30"], rows,
                summary="every α: decoded distance by step, the lowest cos to "
                        "the nearest cell's code during the walk (steps 1–12), "
                        "and the cos after 30 steps")
    return card("The α walk: does recall pass through intermediate positions?",
                '<div class="grid2">' + "".join(panels) + "</div>" + tbl,
                note="x ← normalize((1−α)x + α·tanh(βWx)), K = 5, cues from "
                     "every cell of the env. Each step is decoded to its nearest "
                     "cell. A walk falls smoothly with cos near 1; a snap stalls, "
                     "then jumps, with a cos dip in between.")


def chord_section(dyn: dict) -> str:
    panels, rows = [], []
    use = [l for l in DYN if "β=1e6" not in l or "arm B" in l]
    seps = sorted({s for l in use if dyn.get(l) for s in dyn[l]["chord"]},
                  key=float)
    for sep in seps:
        series = []
        for i, lab in enumerate(DYN):
            if lab not in use or not dyn.get(lab):
                continue
            c = dyn[lab]["chord"].get(sep)
            if not c or not c["rows"]:
                continue
            ts = [r["t"] for r in c["rows"]]
            series.append({"label": lab.split(" · β")[0], "color": COL[i],
                           "values": [r["dist"] for r in c["rows"]]})
            rows.append([lab.split(" · β")[0], sep, str(c["n_start"])]
                        + [_fmt(r["dist"], ".1f") for r in c["rows"]]
                        + [_fmt(min(r["cos"] for r in c["rows"]))])
        if series:
            panels.append(line_chart(ts, series, xlabel="t along the chord",
                                     ylabel="decoded distance to goal",
                                     title=f"start {sep} cells out"))
    ts_h = [f"t={t:g}" for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6,
                                  0.7, 0.8, 0.9, 1.0)]
    tbl = table(["encoder", "sep", "n"] + ts_h + ["min cos"], rows,
                summary="decoded distance at each t, and the lowest cos to the "
                        "nearest cell's code")
    return card("The chord from here to the goal",
                '<div class="grid2">' + "".join(panels) + "</div>" + tbl,
                note="x(t) = (1−t)·z(here) + t·z(goal), decoded at each t. "
                     "Independent of β (no recall), so the saturated ideal row "
                     "is omitted. For a Gaussian-kernel code the prediction is a "
                     "smooth walk while the start is within ~2r and a snap "
                     "beyond.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--lede", default=None, help="HTML snippet for the top")
    ap.add_argument("--page", default=None)
    a = ap.parse_args()
    out = a.out.rstrip("/")

    # 1. the standard report over the whole-arena results, relabelled
    inp = Path(out) / "report_input"
    inp.mkdir(exist_ok=True)
    man = {"encoders": []}
    for tree, t, lab in REPORT:
        r = _one(f"{out}/{tree}/t{t}/*.json")
        if r is None:
            print(f"missing {tree}/t{t} ({lab})")
            continue
        r["header"]["label"] = lab
        fn = "".join(c if c.isalnum() else "_" for c in lab) + ".json"
        (inp / fn).write_text(json.dumps(r))
        man["encoders"].append({"file": fn, "label": lab})
    (inp / "manifest.json").write_text(json.dumps(man))
    rep = B.build(inp, Path(out) / "report")
    results = [json.loads((inp / e["file"]).read_text()) for e in man["encoders"]]

    # 2. the ideal section
    dyn = {}
    for f in glob.glob(f"{out}/dynamics/*.json"):
        d = json.load(open(f))
        dyn[d["header"]["label"]] = d
    lede = Path(a.lede).read_text() if a.lede else ""
    ideal = (lede + probe_section(out) + scan_section(out) + fixed_section(dyn)
             + walk_section(dyn) + chord_section(dyn))
    style = ("<style>.grid2{display:grid;grid-template-columns:"
             "repeat(auto-fit,minmax(320px,1fr));gap:12px}"
             ".grid2 svg{max-width:100%;height:auto}</style>")

    # 3. stack it in front of the report's own sections
    sections = [("ideal", "Ideal encoder", style + ideal)]
    for name, lbl in B._TAB_NAMES:
        p = rep / name
        if p.exists():
            sections.append((name.replace(".html", ""), lbl,
                             B._strip_encoder_filter(B._body_of(p.read_text()))))
    prefix = B.encoder_filter(results)
    sections[1] = (sections[1][0], sections[1][1], prefix + sections[1][2])
    hdr = "".join(
        f'<span class="enc-hdr" data-encoder="{esc(n)}">'
        f'{run_header(B.representative(m)["header"], B._kv_seeds(m) if len(m) > 1 else "")}'
        f'</span>' for n, m in B.groups(results))
    page = Path(a.page) if a.page else Path(out) / "ideal_report.html"
    page.write_text("<title>Ideal Encoder Probe</title>\n" + single_page(
        "Ideal encoder · Hopfield probe", hdr, sections, out, fragment=True))
    print(f"wrote {page} ({page.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
