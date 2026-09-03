"""Append the basin analysis to the coverage-ladder page.

`report.build` renders one encoder per selector entry and one run per page, so
it cannot show either of the two things this section needs: the basin measured
across TRAINING SEEDS (the page probes one seed per rung), or the history of how
the measurement was wrong. Both are added here as a final section with its own
nav tab.
"""
import pathlib

SRC = pathlib.Path("/home/jackking/.claude/jobs/d05f5770/tmp/probe_basin2/"
                   "report/report.fragment.html")
DST = pathlib.Path("/home/jackking/.claude/jobs/d05f5770/tmp/"
                   "coverage_ladder.html")

# coverage label, per-seed basin medians (42,43,44,45), the seed the page
# probes, and the per-seed self-retrieval failure rates.
ROWS = [
    ("10%",   "att0.5",    [25.5, 30.5, 28.0, 26.0], 43,
     [0.00, 0.00, 0.00, 0.00]),
    ("5%",    "half_a0.5", [23.0, 23.0, 23.0, 19.5], 43,
     [0.06, 0.00, 0.00, 0.00]),
    ("2.5%",  "q_a1",      [23.0, 19.0, 19.5, 12.0], 45,
     [0.06, 0.06, 0.06, 0.12]),
    ("1.25%", "sm35x_a2",  [10.0, 13.0, 18.5, 8.0], 44,
     [0.25, 0.19, 0.06, 0.38]),
    ("0.75%", "y50_a2",    [12.5, 6.0, 14.5, 16.0], 44,
     [0.19, 0.12, 0.12, 0.00]),
]
SEEDS = (42, 43, 44, 45)


def med(v):
    s = sorted(v)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def bar(v, width=150, cap=40.0):
    w = max(2.0, width * v / cap)
    return (f'<svg viewBox="0 0 {width + 40} 16" style="width:{width + 40}px;'
            f'height:16px;vertical-align:middle">'
            f'<rect x="0" y="3" width="{width}" height="10" rx="2" '
            f'fill="var(--grid)"/>'
            f'<rect x="0" y="3" width="{w:.1f}" height="10" rx="2" '
            f'fill="var(--cat1)"/>'
            f'<text x="{width + 4}" y="12" font-size="11" '
            f'font-family="ui-monospace,monospace" fill="var(--ink2)">'
            f'{v:.1f}</text></svg>')


trs = []
for cov, arm, vals, used, fails in ROWS:
    cells = "".join(
        f"<td class='{'used' if s == used else ''}'>{v:.1f}</td>"
        for s, v in zip(SEEDS, vals))
    trs.append(
        f"<tr><td>{cov}</td><td class='mono'>{arm}</td>{cells}"
        f"<td>{bar(med(vals))}</td>"
        f"<td style='text-align:right'>{med(fails):.2f}</td></tr>")

section = f"""<section id="basin" class="page">
<h1>The basin, and how the measurement was wrong twice</h1>

<p class="lede">The basin is <code>r_exact_all</code>: the largest radius within
which <b>every</b> cue retrieves the goal cell exactly, stopping at the first
failing radius so the guarantee nests all the way in. Cues are every cell in a
disc around the goal <b>in scaffold coordinates</b>, and the retrieval bank is
those same cells plus the other stored goals. No evaluation environment is
involved anywhere in it.</p>

<h2>Across training seeds</h2>
<p class="lede">The selector above shows <b>one training seed per rung</b>, and
for the basin that is not enough &mdash; it varies more across seeds than the
whole ladder varies across coverage. The seed each page entry uses is
highlighted; it was chosen on <em>reach</em>, which has a much tighter seed
spread and a different preference.</p>

<div class="card"><table class="cmp">
<thead><tr><th>coverage</th><th>arm</th>
<th>s42</th><th>s43</th><th>s44</th><th>s45</th>
<th>median of four</th><th>self-fail</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>
<p class="note">Highlighted cell = the seed the selector above probes.
<b>self-fail</b> is the fraction of (world, env) pairs where the goal cue does
not retrieve the goal at all, so no radius holds and the basin is undefined.</p>
</div>

<h2>What the seed choice was hiding</h2>
<p class="lede">Two rungs were badly served by their reach-winning seed, in
opposite directions. <b>2.5%</b> drew its worst seed (12.0 against an arm median
of 19.2) and <b>1.25%</b> drew its best (18.5 against 11.5) &mdash; which
between them inverted the ladder and made 2.5% look worse than 1.25%. On arm
medians the inversion is gone. <b>10%</b> was flattered too: 30.5 on the page
against 27.0 across seeds.</p>

<p class="lede">Corrected, the basin falls with coverage down to 1.25% &mdash;
<b>27.0 / 23.0 / 19.2 / 11.5</b> &mdash; and the bottom two rungs are not
separable, 11.5 against 13.5 with per-seed ranges of 8&ndash;18.5 and
6&ndash;16. Self-retrieval failure runs the other way, from 0.00 at the top to
0.22 at 1.25%: at low coverage the goal increasingly fails to retrieve
<em>itself</em>, which is a different failure from a shrinking basin and is
reported separately rather than folded in.</p>

<h2>Two corrections</h2>
<p class="lede"><b>The basin used to be measured over the evaluation
environment.</b> Cues were the env's own cells, so no cue sat more than ~27
cells from the goal at <code>env_size</code> 20 and the statistic could not
report a larger number however large the true basin was. Every basin figure
from before this change is censored, and the top of the ladder &mdash; where
the real basin is 27 and 23 &mdash; was reporting the arena.</p>

<p class="lede"><b>And the first fix had a bug.</b> The retrieval bank was built
as the disc cells plus <em>all</em> stored goals, which put this env's goal in
twice &mdash; once as the disc centre, once as its stored copy. Identical
position, so identical vector to within float noise, and the argmax returned
whichever copy happened to win. When the duplicate won, a correct retrieval
scored as a miss. It corrupted 6 of 16 values per encoder, not just the two that
showed up as outright failures, because the duplicate could also win at radius 2
or 3 and truncate the basin there. The bank now takes only the <b>other</b> K-1
stored goals.</p>
</section>"""

html = SRC.read_text()
i = html.rindex("</div>", 0, html.index("<script>"))
html = html[:i] + section + html[i:]
html = html.replace('<a href="#controls">Controls</a>',
                    '<a href="#controls">Controls</a>'
                    '<a href="#basin">Basin</a>', 1)

style = """<style>
#basin table.cmp { width: 100%; border-collapse: collapse; font-size: 13px;
  font-variant-numeric: tabular-nums; }
#basin table.cmp th { text-align: right; font-size: 11px; letter-spacing:
  .05em; text-transform: uppercase; color: var(--muted); font-weight: 600;
  padding: 0 10px 8px 0; border-bottom: 1px solid var(--border); }
#basin table.cmp th:first-child, #basin table.cmp th:nth-child(2),
#basin table.cmp td:first-child, #basin table.cmp td.mono {
  text-align: left; }
#basin table.cmp td { padding: 7px 10px 7px 0; border-bottom: 1px solid
  var(--grid); text-align: right; vertical-align: middle; }
#basin table.cmp td.mono { font-family: ui-monospace, monospace;
  font-size: 12px; color: var(--ink2); }
#basin table.cmp td.used { font-weight: 700; color: var(--cat1);
  background: color-mix(in srgb, var(--cat1) 10%, transparent); }
#basin .note { font-size: 12.5px; color: var(--muted); margin: 12px 0 0;
  max-width: 72ch; line-height: 1.5; }
#basin .lede { max-width: 72ch; }
</style>"""
html = html.replace("</style>", "</style>" + style, 1)

DST.write_text("<title>Coverage Ladder</title>\n" + html)
print(f"{DST.stat().st_size / 1e6:.2f} MB")
print("section present:", 'id="basin"' in DST.read_text())
