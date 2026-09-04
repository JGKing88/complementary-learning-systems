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


def load(d, regime):
    """[(coverage, basin_median, self_fail)] for every seed in a run dir.

    ``regime`` filters, or ``None`` takes the directory whole. The ladder
    directory needs the filter: it now also holds the full saturated arms of
    the 10% encoder, which run at the same beta as the right-hand series here
    and would otherwise be read into the left one. The saturated directory
    holds nothing else, so it is taken whole -- and must be, since it was
    written before `recall_regime` was recorded.
    """
    out = []
    for f in sorted(glob.glob(d + "/*.json")):
        if "manifest" in f:
            continue
        r = json.load(open(f))
        if regime is not None and r["header"].get("recall_regime") != regime:
            continue
        sc = r["test_a"]["k"][K]["per_step"][S]["scalars"]
        v = np.asarray(sc["r_exact_all"]["values"], float)
        out.append((r["header"]["coverage"], float(np.median(v)),
                    float(np.mean(v < 0))))
    return sorted(out, key=lambda t: -t[0])


ARMS = [("10%", "β = gain = 100"), ("10% β=1e6", "β = 1e6, gain 100"),
        ("10% gain=1e6, β=1e6", "β = 1e6, gain 1e6")]


def arms_rows():
    """The three saturation arms of the 10% encoder, pooled over seeds.

    Read out of the same directory the tabs are built from, so the table and
    the tabs cannot disagree.
    """
    by: dict[str, list] = {}
    for f in sorted(glob.glob(LIN + "/*.json")):
        if "manifest" in f:
            continue
        r = json.load(open(f))
        g = r["header"]["label"].split(" · ")[0].strip()
        by.setdefault(g, []).append(r)

    out = []
    for group, name in ARMS:
        rs = by.get(group)
        if not rs:
            continue
        a = [r["test_a"]["k"][K]["per_step"][S] for r in rs]
        bc = [r["test_bc"]["k"][K]["per_step"][S] for r in rs]
        d = [r["test_d"]["k"][K][S] for r in rs]
        v = np.concatenate([np.asarray(x["scalars"]["r_exact_all"]["values"],
                                       float) for x in a])
        per_seed = [float(np.median(np.asarray(
            x["scalars"]["r_exact_all"]["values"], float))) for x in a]
        out.append((
            name,
            np.mean([x["fixed_point"]["cos_self_mean"]["mean"] for x in a]),
            float(np.median(per_seed)),
            f"{int((v < 0).sum())}/{v.size}",
            np.mean([x["scalars"]["exact_frac"]["mean"] for x in a]),
            np.mean([x["grid"]["scalars"]["acc45"]["mean"] for x in bc]),
            np.mean([x["grid"]["scalars"]["abs_err_mean"]["mean"] for x in bc]),
            float(np.median([x["continuous"]["scalars"]["reach_rate"]["mean"]
                             for x in d])),
        ))
    return out


def arms_html():
    head = ("arm", "cos_self", "basin", "self-fail", "exact_hit", "acc45",
            "|err|", "cont reach")
    trs = "".join(
        f"<tr><td>{n}</td><td>{cs:.4f}</td><td>{b:.1f}</td><td>{sf}</td>"
        f"<td>{ex:.3f}</td><td>{ac:.3f}</td><td>{er:.1f}&deg;</td>"
        f"<td>{re_:.3f}</td></tr>"
        for n, cs, b, sf, ex, ac, er, re_ in arms_rows())
    ths = "".join(f"<th>{h}</th>" for h in head)
    return (f'<table class="cmp"><thead><tr>{ths}</tr></thead>'
            f'<tbody>{trs}</tbody></table>')


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
    lin, sat = load(LIN, "linear"), load(SAT, None)
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

    arms_table = arms_html()
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
<b>1e6</b>. This was taken to be the knob deciding whether a stored pattern is a
fixed point at all. <b>It is not.</b> Recalling a stored pattern from itself at
<code>&beta;</code>&nbsp;=&nbsp;1e6 returns <code>cos_self</code> = 0.957, not 1
&mdash; the fixed point is a hypercube corner <em>near</em> the memory, not the
memory. So the basin falling here is not the price of an attractor; it is the
price of a half-made one.</p>

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

<h2>The other half</h2>
<p class="lede">A pattern is a fixed point when it sits on a hypercube corner,
and that is the <em>encoder's</em> gain, not <code>&beta;</code>. At encoder
gain 1e6 the output is <code>tanh(1e6&thinsp;z)</code> = <code>sign(z)</code>,
so the pattern <b>is</b> a corner. The 10% encoder, four training seeds per arm,
both arms on this page as their own tabs.</p>
<div class="card">{arms_table}</div>
<p class="note">With both saturated the fixed point is exact, the goal cue never
fails in 64 (world, env) pairs, <code>exact_hit</code> is the highest in the
campaign, and <b>the basin does not shrink</b>. What goes instead is the
direction field: <code>q</code> is a finite difference of neighbouring cell
embeddings, and <code>sign(z)</code> has no usable local derivative &mdash; two
adjacent cells differ by a handful of flipped bits in no particular direction.
<code>acc90</code> holds at 0.71, so a coarse signal survives; nothing at 45&deg;
does. The two conditions span a square, and production sits at the only corner
where memory and direction both work.</p>
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
