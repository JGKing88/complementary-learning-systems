"""Append the env-size analysis to the coverage-ladder page.

`report.build` renders one run per page, and this comparison spans two runs
(env 20 and env 40), so it cannot come from the builder -- it is added here as a
final section with its own nav tab.
"""
import pathlib
import re

SRC = pathlib.Path("/home/jackking/.claude/jobs/d05f5770/tmp/probe_five/"
                   "report/report.fragment.html")
DST = pathlib.Path("/home/jackking/.claude/jobs/d05f5770/tmp/"
                   "coverage_ladder.html")

ROWS = [
    # label, coverage, env20 basin, env40 basin
    ("10% · att0.5", "10.02%", 21.12, 30.73),
    ("5% · half_a0.5", "5.01%", 20.73, 27.68),
    ("2.5% · q_a1", "2.55%", 17.62, 17.95),
    ("1.25% · sm35x_a2", "1.25%", 16.62, 17.55),
    ("0.75% · y50_a2", "0.76%", 11.12, 12.95),
]

def bar(v, cap, width=190):
    """A basin bar drawn against its environment's geometric cap."""
    w = max(2.0, width * v / 55.2)
    capw = width * cap / 55.2
    clipped = v > cap - 6.0
    col = "var(--cat2)" if clipped else "var(--cat1)"
    return (f'<svg viewBox="0 0 {width + 46} 18" style="width:{width + 46}px;'
            f'height:18px;vertical-align:middle">'
            f'<rect x="0" y="4" width="{capw:.1f}" height="10" rx="2" '
            f'fill="var(--grid)"/>'
            f'<rect x="0" y="4" width="{w:.1f}" height="10" rx="2" '
            f'fill="{col}"/>'
            f'<text x="{width + 4}" y="13" font-size="11" '
            f'font-family="ui-monospace,monospace" fill="var(--ink2)">'
            f'{v:.2f}</text></svg>')

trs = []
for lab, cov, b20, b40 in ROWS:
    d = b40 - b20
    flag = ' style="font-weight:600"' if d > 5 else ""
    trs.append(
        f"<tr{flag}><td>{lab}</td><td>{cov}</td>"
        f"<td>{bar(b20, 26.9)}</td><td>{bar(b40, 55.2)}</td>"
        f"<td style='text-align:right;font-family:ui-monospace,monospace'>"
        f"{d:+.2f}</td></tr>")

section = f"""<section id="envsize" class="page">
<h1>Environment size and the basin ceiling</h1>
<p class="lede">The basin radius <code>r_exact_95</code> is measured in cells,
so it cannot exceed the largest goal-to-corner distance the evaluation
environment offers &mdash; about 27 cells in a 20&times;20 arena. Across 198
probed encoders the largest basin ever recorded was <b>21.62</b>, with 18 of
them within 0.1 of it. That is a pin, not a coincidence, so the question is
whether the top of the coverage ladder was measuring encoders or measuring the
arena.</p>

<p class="lede">Re-running the same five encoders at <code>env_size 40</code>,
which lifts the cap to ~55 cells, answers it. <b>The top two rungs were
clipped and the bottom three were not.</b></p>

<div class="card"><table class="cmp">
<thead><tr><th>encoder</th><th>coverage</th>
<th>basin, env 20 &nbsp;<span style="font-weight:400;color:var(--muted)">
(cap 26.9)</span></th>
<th>basin, env 40 &nbsp;<span style="font-weight:400;color:var(--muted)">
(cap 55.2)</span></th><th>change</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>
<p class="note">Bars are drawn to a common scale; the grey track is each
environment's geometric cap. Orange marks a basin close enough to its cap to be
suspect.</p></div>

<h2>What this changes</h2>
<p class="lede">The 10% and 5% encoders gain <b>9.6</b> and <b>7.0</b> cells
when given room; the 2.5%, 1.25% and 0.75% encoders gain 0.3&ndash;1.8, which is
noise. So the two ends were being measured differently: the bottom of the ladder
reports its real basin, and the top of it was reporting the arena.</p>

<p class="lede">Two consequences for the coverage panel on the overview.
<b>The spread is larger than it appears</b> &mdash; 17.8 cells across the ladder
rather than 10.0. And <b>the top two rungs are not equal</b>: they differ by
0.40 cells at env 20 and by <b>3.05</b> at env 40, so their apparent tie was
both of them sitting against the ceiling.</p>

<p class="lede">The corrected reading of basin against coverage is
<b>30.7 / 27.7 / 18.0 / 17.6 / 13.0</b>: a steep fall from 5% to 2.5%, a
plateau across 2.5% and 1.25%, then a further drop at 0.75%. That is a
different shape from the one the env-20 panel shows, and it is the one to
trust.</p>

<p class="note"><b>Only the basin transfers between the two runs.</b> A
40&times;40 arena is a harder navigation problem with longer paths, so reach
falls for every encoder (0.987 &rarr; 0.856 at 10% coverage) and the angular
errors move too. Those differences measure the arena, not the encoder, and are
not comparable across env sizes. The basin is a radius in cells and is the one
quantity where the comparison is clean.</p>
</section>"""

html = SRC.read_text()

# Insert before the wrap's closing div, which is the last one before <script>.
i = html.rindex("</div>", 0, html.index("<script>"))
html = html[:i] + section + html[i:]

# And a nav tab, so the section is reachable rather than only scrollable-to.
html = html.replace('<a href="#controls">Controls</a>',
                    '<a href="#controls">Controls</a>'
                    '<a href="#envsize">Env size</a>', 1)

# The comparison spans encoders, so it must survive the encoder filter -- the
# filter hides anything whose data-encoder does not match, and this section
# carries none, so it is left alone. Styling for the table only.
style = """<style>
#envsize table.cmp { width: 100%; border-collapse: collapse; font-size: 13px; }
#envsize table.cmp th { text-align: left; font-size: 11px; letter-spacing:
  .05em; text-transform: uppercase; color: var(--muted); font-weight: 600;
  padding: 0 12px 8px 0; border-bottom: 1px solid var(--border); }
#envsize table.cmp td { padding: 7px 12px 7px 0; border-bottom: 1px solid
  var(--grid); vertical-align: middle; }
#envsize .note { font-size: 12.5px; color: var(--muted); margin: 12px 0 0;
  max-width: 70ch; line-height: 1.5; }
#envsize .lede { max-width: 72ch; }
</style>"""
html = html.replace("</style>", "</style>" + style, 1)

DST.write_text("<title>Coverage Ladder</title>\n" + html)
print(f"{DST.stat().st_size / 1e6:.2f} MB")
print("envsize section present:", "id=\"envsize\"" in DST.read_text())
