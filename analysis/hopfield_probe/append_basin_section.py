"""Append the saturation comparison to the coverage-ladder page.

`report.build` renders one run per page, and this compares two: the ladder at
`beta = gain` against the same encoders with the Hopfield saturated at
`beta = 1e6`. Only the recall loop gain differs; each encoder keeps its own
encoder gain, so this isolates Sec 7's condition (b) -- whether the stored
pattern is a fixed point -- from condition (a), which the encoder gain sets.

Charts are built with the report's own `line_chart` so they match the rest of
the page rather than being hand-rolled SVG.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.report.figures import line_chart
from analysis.hopfield_probe.report.page import card
from analysis.hopfield_probe.report.theme import CATEGORICAL

SRC = pathlib.Path("/home/jackking/.claude/jobs/d05f5770/tmp/probe_spliced/"
                   "report/report.fragment.html")
LIN = "/home/jackking/.claude/jobs/d05f5770/tmp/probe_spliced"
SAT = "/home/jackking/.claude/jobs/d05f5770/tmp/probe_spliced_b1e06"
DST = pathlib.Path("/home/jackking/.claude/jobs/d05f5770/tmp/"
                   "coverage_ladder.html")
K, S = "5", "1"
CAT1, CAT2 = CATEGORICAL[0][0], CATEGORICAL[1][0]


def load(d):
    """[(coverage, basin_median, self_fail)] for every seed in a run dir."""
    out = []
    for f in sorted(glob.glob(d + "/*.json")):
        if "manifest" in f:
            continue
        r = json.load(open(f))
        sc = r["test_a"]["k"][K]["per_step"][S]["scalars"]
        v = np.asarray(sc["r_exact_all"]["values"], float)
        out.append((r["header"]["coverage"], float(np.median(v)),
                    float(np.mean(v < 0))))
    return sorted(out, key=lambda t: -t[0])


def series(rows, idx, label, color):
    """A per-seed scatter plus a line through the per-coverage medians."""
    xs = [c * 100 for c, *_ in rows]
    vals = [t[idx] for t in rows]
    by = {}
    for c, *rest in rows:
        by.setdefault(round(c, 6), []).append(rest[idx - 1])
    med = []
    for c, *_ in rows:
        v = sorted(by[round(c, 6)])
        n = len(v)
        med.append(v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2)
    return xs, [{"label": label, "color": color, "width": 1.5,
                 "values": med},
                {"label": f"{label} · per seed", "color": color,
                 "points": True, "values": vals}]


def main() -> None:
    lin, sat = load(LIN), load(SAT)
    if not sat:
        raise SystemExit(f"no saturated results in {SAT}")

    xs, s_lin = series(lin, 1, "β = gain", CAT1)
    _xs, s_sat = series(sat, 1, "β = 1e6", CAT2)
    basin_fig = line_chart(xs, s_lin + s_sat,
                           xlabel="training coverage (%)",
                           ylabel="r_exact_all (cells)")

    xs2, f_lin = series(lin, 2, "β = gain", CAT1)
    _x2, f_sat = series(sat, 2, "β = 1e6", CAT2)
    fail_fig = line_chart(xs2, f_lin + f_sat,
                          xlabel="training coverage (%)",
                          ylabel="self-retrieval failure", ylim=(0, 1))

    def med_of(rows, idx):
        by = {}
        for c, *rest in rows:
            by.setdefault(round(c, 6), []).append(rest[idx - 1])
        return {c: float(np.median(v)) for c, v in by.items()}

    ml, ms = med_of(lin, 1), med_of(sat, 1)
    trs = "".join(
        f"<tr><td>{c * 100:.2f}%</td><td>{ml[c]:.1f}</td>"
        f"<td>{ms.get(c, float('nan')):.1f}</td>"
        f"<td>{ms.get(c, float('nan')) - ml[c]:+.1f}</td></tr>"
        for c in sorted(ml, reverse=True))

    section = f"""<section id="saturated" class="page">
<h1>Saturating the recall</h1>
<p class="lede">Same encoders, same encoder gains, one thing changed: the
Hopfield's loop gain <code>&beta;</code> goes from each encoder's own gain to
<b>1e6</b>. That is the knob deciding whether a stored pattern is a fixed point
at all &mdash; unsaturated, recall lands on the goal in one step and then drifts
off it; saturated, it lands and stays. The basin is where that should show.</p>

<div class="grid2">
{card("Basin vs. coverage, saturated and not", basin_fig,
      note=f"K={K}, s={S}. Median over (world, env) per seed; lines join the "
           f"per-coverage medians.")}
{card("Self-retrieval failure", fail_fig,
      note="Fraction of (world, env) pairs where the goal cue does not retrieve "
           "the goal, so no radius holds.")}
</div>

<div class="card"><table class="cmp">
<thead><tr><th>coverage</th><th>β = gain</th><th>β = 1e6</th><th>change</th>
</tr></thead><tbody>{trs}</tbody></table></div>
</section>"""

    html = SRC.read_text()
    i = html.rindex("</div>", 0, html.index("<script>"))
    html = html[:i] + section + html[i:]
    html = html.replace('<a href="#controls">Controls</a>',
                        '<a href="#controls">Controls</a>'
                        '<a href="#saturated">Saturated</a>', 1)
    style = """<style>
#saturated table.cmp { width: 100%; border-collapse: collapse; font-size: 13px;
  font-variant-numeric: tabular-nums; }
#saturated table.cmp th { text-align: right; font-size: 11px; letter-spacing:
  .05em; text-transform: uppercase; color: var(--muted); font-weight: 600;
  padding: 0 10px 8px 0; border-bottom: 1px solid var(--border); }
#saturated table.cmp th:first-child, #saturated table.cmp td:first-child {
  text-align: left; }
#saturated table.cmp td { padding: 7px 10px 7px 0; border-bottom: 1px solid
  var(--grid); text-align: right; }
#saturated .lede { max-width: 72ch; }
</style>"""
    html = html.replace("</style>", "</style>" + style, 1)
    DST.write_text("<title>Coverage Ladder</title>\n" + html)
    print(f"{DST.stat().st_size / 1e6:.2f} MB   section:",
          'id="saturated"' in DST.read_text())


if __name__ == "__main__":
    main()
