"""Render matched explore trajectories, and the two explore failure modes.

Consumes `explore_traj.py`'s JSON. The design system is the project's own --
`exploit_report.CSS` tokens, the IBM Plex trio, the blue-biased neutrals -- so
this page sits beside the exploit pages rather than beside nothing.

The two SERIES colours are the house categorical --c1/--c2, except that the
house DARK steps fail the dataviz lightness band for chart marks (L 0.759 and
0.683 against a 0.48-0.67 band). They are re-snapped to the nearest passing
steps, which moves them by dE 10.7 -- close enough to still read as this
palette. Both modes were run through the validator:

  light  #b8791f / #6d5aa8   protan dE 24.3   normal dE 25.2   contrast OK
  dark   #c6852c / #9885da   deutan dE 23.7   normal dE 23.5   contrast OK

  python -m analysis.nav_tri.explore_page --json d0.json d10.json --out p.html
"""

import argparse
import json

import numpy as np

from analysis.nav_tri.exploit_report import CSS, FONTS, esc
from analysis.nav_tri.explore_traj import billiard_path, loop_stats

# House --c1/--c2, dark steps snapped into the validator's band.
SERIES = {
    "p20_e": ("var(--s1)", "kappa uncapped"),
    "p20_e_kcap": ("var(--s2)", "kappa <= 12.2"),
}

EXTRA_CSS = """
:root{ --s1:#b8791f; --s2:#6d5aa8; }
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){ --s1:#c6852c; --s2:#9885da; }
}
:root[data-theme="dark"]{ --s1:#c6852c; --s2:#9885da; }

.legend{display:flex;gap:18px;flex-wrap:wrap;align-items:center;
  margin:14px 0 2px;font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:11.5px;color:var(--muted)}
.legend b{display:inline-flex;align-items:center;gap:7px;font-weight:500;
  color:var(--ink)}
.swatch{width:11px;height:11px;border-radius:2px;flex:none}
.grid{display:grid;gap:14px;margin-top:14px}
.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.g4{grid-template-columns:repeat(4,minmax(0,1fr))}
@media (max-width:760px){ .g3,.g4{grid-template-columns:repeat(2,minmax(0,1fr))} }
@media (max-width:460px){ .g2,.g3,.g4{grid-template-columns:minmax(0,1fr)} }
.cell{background:var(--panel);border:1px solid var(--line);border-radius:9px;
  padding:11px 11px 9px;box-shadow:var(--shadow);min-width:0}
.cell svg{display:block;width:100%;height:auto}
.cap{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
  color:var(--muted);margin-top:8px;line-height:1.5;
  font-variant-numeric:tabular-nums}
.cap .k{color:var(--ink);font-weight:500}
.tag{display:inline-block;font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:9.5px;letter-spacing:.09em;text-transform:uppercase;
  padding:2px 6px;border-radius:3px;border:1px solid var(--line);
  background:var(--soft);color:var(--muted)}
.tag.bad{color:var(--c5);border-color:color-mix(in srgb,var(--c5) 40%,var(--line))}
.hero{background:var(--panel);border:1px solid var(--line);border-radius:11px;
  padding:18px;box-shadow:var(--shadow);display:grid;
  grid-template-columns:260px minmax(0,1fr);gap:22px;align-items:center;
  margin-top:16px}
@media (max-width:640px){ .hero{grid-template-columns:minmax(0,1fr)} }
.hero svg{display:block;width:100%;height:auto}
.stat{display:flex;flex-wrap:wrap;gap:10px 26px;margin-top:12px}
.stat div{min-width:96px}
.stat .n{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:19px;
  font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums;
  display:block;line-height:1.25}
.stat .l{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;
  color:var(--muted);letter-spacing:.07em;text-transform:uppercase}
.stat .crit{color:var(--c5)}
table{border-collapse:collapse;width:100%;margin-top:14px;font-size:12.5px}
th,td{text-align:right;padding:7px 9px;border-bottom:1px solid var(--line);
  font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
th{font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);font-weight:500}
td.hi{color:var(--ink);font-weight:600}
.scroll{overflow-x:auto}
.note{border-left:2px solid var(--line);padding:2px 0 2px 14px;
  margin-top:16px;color:var(--muted);font-family:"IBM Plex Serif",Georgia,serif;
  font-size:13.5px;line-height:1.66;max-width:66ch}
.note b{color:var(--ink);font-family:"IBM Plex Sans",sans-serif;
  font-size:12.5px;font-weight:600}
"""


def footprint(tr, size, color, *, w=210, show_cells=True):
    """One trajectory: visited-cell footprint under the path.

    The footprint is drawn because COVERAGE is the metric here and a path alone
    does not show retracing -- two paths of identical length can differ 2x in
    cells touched, and only the shaded set makes that visible.
    """
    pad = 8
    g = w - 2 * pad
    sc = g / float(size)

    def sx(x):
        return pad + (x + 0.5) * sc

    def sy(y):
        return pad + (size - 0.5 - y) * sc

    body = [f'<rect x="{pad}" y="{pad}" width="{g}" height="{g}" rx="3" '
            f'fill="var(--soft)" stroke="var(--line)" stroke-width="1"/>']
    if show_cells:
        for cx, cy in tr["cells"]:
            body.append(
                f'<rect x="{pad + cx * sc:.2f}" y="{pad + (size - 1 - cy) * sc:.2f}" '
                f'width="{sc:.2f}" height="{sc:.2f}" fill="{color}" '
                f'fill-opacity="0.17"/>')
    pts = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in tr["path"])
    body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="1.25" stroke-opacity="0.9" '
                f'stroke-linejoin="round" stroke-linecap="round"/>')
    p0, p1 = tr["path"][0], tr["path"][-1]
    body.append(f'<circle cx="{sx(p0[0]):.1f}" cy="{sy(p0[1]):.1f}" r="3.4" '
                f'fill="var(--panel)" stroke="{color}" stroke-width="1.6"/>')
    body.append(f'<circle cx="{sx(p1[0]):.1f}" cy="{sy(p1[1]):.1f}" r="3.2" '
                f'fill="{color}"/>')
    tip = (f'coverage {tr["coverage"]:.3f} · edge {tr["edge_frac"]:.2f} · '
           f'turn {tr["signed_turn_mean"]:+.3f} rad/step · '
           f'speed {tr["speed"]:.2f}')
    body.insert(0, f'<title>{esc(tip)}</title>')
    return (f'<svg viewBox="0 0 {w} {w}" role="img" '
            f'aria-label="{esc(tip)}">{"".join(body)}</svg>')


def cap(tr, extra=()):
    bits = [f'<span class="k">cov {tr["coverage"]:.3f}</span>',
            f'edge {tr["edge_frac"]:.2f}',
            f'turn {tr["signed_turn_mean"]:+.3f}']
    bits += list(extra)
    return '<div class="cap">' + " · ".join(bits) + '</div>'


def strip_chart(vals_by_label, *, w=880, h=150, xlab="", vline=None):
    """One axis, two identities: signed mean turn per trajectory.

    A strip rather than a histogram because the question is whether the two
    sets OVERLAP, and binning is a choice that can hide or manufacture a gap.
    """
    pad_l, pad_r, pad_t, pad_b = 46, 16, 16, 34
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    allv = [v for vs in vals_by_label.values() for v in vs]
    lo, hi = min(allv), max(allv)
    span = max(hi - lo, 1e-6)
    lo, hi = lo - 0.08 * span, hi + 0.08 * span

    def sx(v):
        return pad_l + (v - lo) / (hi - lo) * iw

    body = []
    for i in range(6):
        v = lo + (hi - lo) * i / 5.0
        x = sx(v)
        body.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                    f'y2="{pad_t + ih}" stroke="var(--line)" stroke-width="1"/>')
        body.append(f'<text x="{x:.1f}" y="{pad_t + ih + 17}" '
                    f'text-anchor="middle" font-size="10" '
                    f'font-family="IBM Plex Mono, monospace" '
                    f'fill="var(--muted)">{v:+.2f}</text>')
    if vline is not None:
        x = sx(vline)
        body.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                    f'y2="{pad_t + ih}" stroke="var(--muted)" '
                    f'stroke-width="1" stroke-dasharray="3 3"/>')

    rows = list(vals_by_label.items())
    band = ih / max(len(rows), 1)
    for r, (lab, vs) in enumerate(rows):
        col = SERIES[lab][0]
        yc = pad_t + band * (r + 0.5)
        for v in vs:
            body.append(f'<circle cx="{sx(v):.1f}" cy="{yc:.1f}" r="3.1" '
                        f'fill="{col}" fill-opacity="0.4"/>')
        med = sorted(vs)[len(vs) // 2]
        body.append(f'<line x1="{sx(med):.1f}" y1="{yc - 15:.1f}" '
                    f'x2="{sx(med):.1f}" y2="{yc + 15:.1f}" stroke="{col}" '
                    f'stroke-width="2.5"/>')
        # direct label, in ink not the series colour
        body.append(f'<text x="{pad_l - 8}" y="{yc + 3.5:.1f}" '
                    f'text-anchor="end" font-size="10.5" '
                    f'font-family="IBM Plex Mono, monospace" '
                    f'fill="var(--ink)">{esc(lab)}</text>')
        body.append(f'<text x="{sx(med):.1f}" y="{yc - 20:.1f}" '
                    f'text-anchor="middle" font-size="10" '
                    f'font-family="IBM Plex Mono, monospace" '
                    f'fill="var(--muted)">med {med:+.3f}</text>')
    body.append(f'<text x="{pad_l + iw / 2:.1f}" y="{h - 4}" '
                f'text-anchor="middle" font-size="10" '
                f'font-family="IBM Plex Mono, monospace" '
                f'fill="var(--muted)">{esc(xlab)}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{esc(xlab)}">{"".join(body)}</svg>')


def _pick(trials, lab, key, n, reverse=False):
    s = sorted(trials, key=lambda t: t["by_ckpt"][lab][key], reverse=reverse)
    return s[:n]


def build(d0, d10):
    size = d0["size"]
    labels = d0["labels"]
    T0 = d0["trials"]

    def col(lab):
        return SERIES[lab][0]

    H = []
    H.append("<title>Explore Trajectories</title>")
    H.append(FONTS)
    H.append(f"<style>{CSS}{EXTRA_CSS}</style>")
    H.append('<div class="wrap">')

    H.append("<h1>What exploring looks like, and how it fails</h1>")
    H.append('<p class="sub">P20 &middot; w52 encoder &middot; '
             '288 matched episodes &middot; 200 steps &middot; '
             'held-out envs</p>')
    H.append(
        '<p class="lede">Two policies, identical in everything but the cap on '
        'their directional concentration &kappa;. Every pair below was rolled '
        'from the <i>same start</i> with the <i>same memory contents</i>, so a '
        'difference between two arenas is the policy and nothing else. Shaded '
        'cells are the ones actually visited &mdash; the metric is coverage, '
        'and a path alone hides retracing.</p>')

    H.append('<div class="legend">')
    for lab in labels:
        H.append(f'<b><span class="swatch" style="background:{col(lab)}"></span>'
                 f'{esc(lab)}</b><span>{esc(SERIES[lab][1])}</span>')
    H.append('<span>&#9711; start &nbsp; &#9679; end</span>')
    H.append("</div>")

    # ---------------------------------------------------------------- fail 1
    H.append('<h2>Failure mode 1 &mdash; the wall pin '
             '<span class="meta">1 episode in 144</span></h2>')
    pin = None
    for t in T0:
        r = t["by_ckpt"]["p20_e"]
        if r["clip_frac"] > 0.5 and r["speed"] < 0.5:
            pin = t
            break
    if pin:
        r = pin["by_ckpt"]["p20_e"]
        H.append(
            '<p class="lede">The catastrophic mode, and it is rare. The agent '
            'reaches a wall and stays there: the boundary clip absorbs almost '
            'every step it asks for, so its <i>realized</i> speed collapses to '
            'a tenth of normal while it keeps commanding full stride. It ends '
            'the episode having seen twenty-one cells out of four hundred. '
            'This is the perimeter basin, and the diagnostic that names it is '
            'not coverage but the gap between commanded and realized '
            'motion.</p>')
        H.append('<div class="hero">')
        H.append(f'<div>{footprint(r, size, col("p20_e"), w=240)}</div>')
        H.append('<div>')
        H.append('<span class="tag bad">&#9888; wall pin</span>')
        H.append('<div class="stat">')
        for n, l, crit in ((f'{r["coverage"]:.3f}', "coverage", True),
                           (f'{r["speed"]:.2f}', "realized speed", True),
                           (f'{r["clip_frac"]:.2f}', "clipped steps", True),
                           (f'{r["edge_frac"]:.2f}', "on perimeter", True),
                           (f'{r["straightness"]:.3f}', "straightness", False)):
            H.append(f'<div><span class="n{" crit" if crit else ""}">{n}</span>'
                     f'<span class="l">{l}</span></div>')
        H.append("</div>")
        H.append('<p class="cap" style="margin-top:14px">Median episode for '
                 'this policy: coverage <span class="k">0.390</span>, speed '
                 '<span class="k">0.96</span>, clipped '
                 '<span class="k">0.03</span>, perimeter '
                 '<span class="k">0.13</span>.</p>')
        H.append("</div></div>")

    # ------------------------------------------------------- loops, BOTH arms
    LS = {lab: [loop_stats(t["by_ckpt"][lab]["path"]) for t in T0]
          for lab in labels}
    rng = np.random.RandomState(0)
    nulls = {}
    for lab in labels:
        sp = float(np.mean([t["by_ckpt"][lab]["speed"] for t in T0]))
        paths = [billiard_path(size, len(T0[0]["by_ckpt"][lab]["path"]), sp, rng)
                 for _ in range(len(T0))]
        nulls[lab] = ([loop_stats(p) for p in paths], sp)

    def m(rows, k):
        return float(np.mean([r[k] for r in rows]))

    H.append('<h2>Both policies loop <span class="meta">and the uncapped one '
             'loops more often</span></h2>')
    H.append(
        '<p class="lede">Looping is <i>not</i> what separates these two. Every '
        'episode of both policies comes back over ground it has already '
        'covered, and the uncapped arm does so <i>more</i> often than the '
        'capped one &mdash; and more often than a perfect billiard at its own '
        'speed. What differs is how long it <i>stays</i>. A billiard null is '
        'shown for each because in a 20&times;20 box a 200-step path '
        're-crosses itself by geometry alone; only the excess is looping.</p>')
    H.append('<div class="scroll"><table><thead><tr>'
             '<th>&nbsp;</th><th>re-crossings / episode</th>'
             '<th>vs null</th><th>share of steps on old ground</th>'
             '<th>vs null</th><th>handedness flips</th></tr></thead><tbody>')
    for lab in labels:
        nr, sp = nulls[lab]
        H.append(
            f'<tr><td>{esc(lab)}</td>'
            f'<td class="hi">{m(LS[lab], "return_events"):.1f}</td>'
            f'<td>{m(LS[lab], "return_events") - m(nr, "return_events"):+.1f}</td>'
            f'<td class="hi">{m(LS[lab], "near_past_frac"):.3f}</td>'
            f'<td>{m(LS[lab], "near_past_frac") - m(nr, "near_past_frac"):+.3f}</td>'
            f'<td>{m(LS[lab], "sign_flips"):.1f}</td></tr>')
        H.append(f'<tr><td style="color:var(--muted)">&nbsp;&nbsp;billiard null '
                 f'@ {sp:.2f}</td>'
                 f'<td style="color:var(--muted)">{m(nr, "return_events"):.1f}</td>'
                 f'<td>&nbsp;</td>'
                 f'<td style="color:var(--muted)">{m(nr, "near_past_frac"):.3f}</td>'
                 f'<td>&nbsp;</td>'
                 f'<td style="color:var(--muted)">{m(nr, "sign_flips"):.1f}</td></tr>')
    H.append("</tbody></table></div>")
    H.append(
        '<div class="note"><b>Read the two columns against each other.</b><br>'
        'The uncapped arm re-crosses <i>more</i> (12.5 events against 8.9) but '
        'spends <i>less</i> of the episode on old ground (0.310 against 0.445). '
        'It cuts across its own path and keeps going. The capped arm crosses '
        'less often and then dwells &mdash; its excess over the null is '
        '+0.140, more than double the uncapped arm&rsquo;s +0.062. '
        '<b>The cost is not looping. It is not leaving.</b></div>')

    H.append('<h3>uncapped &mdash; the most re-crossing episodes</h3>')
    idx = sorted(range(len(T0)),
                 key=lambda i: LS["p20_e"][i]["return_events"], reverse=True)[:4]
    H.append('<div class="grid g4">')
    for i in idx:
        r = T0[i]["by_ckpt"]["p20_e"]
        H.append('<div class="cell">'
                 + footprint(r, size, col("p20_e"))
                 + cap(r, [f're-cross {LS["p20_e"][i]["return_events"]}'])
                 + "</div>")
    H.append("</div>")

    # ---------------------------------------------------------------- fail 2
    H.append('<h2>Failure mode 2 &mdash; one-handed drift '
             '<span class="meta">systematic, every capped episode</span></h2>')
    H.append(
        '<p class="lede">Given that both arms loop, the thing that is specific '
        'to the capped policy is <i>handedness</i>. It turns the same '
        'direction on every step of every episode: mean signed turn +0.120 '
        'rad/step against the uncapped policy&rsquo;s +0.001, and the two '
        'distributions <i>do not overlap</i> &mdash; the slowest-turning '
        'capped episode (+0.094) still turns faster than the fastest-turning '
        'uncapped one (+0.036). A constant-rate turn traces an annulus, which '
        'is why this arm both avoids the perimeter and dwells in the middle. '
        'Straightness cannot see it: a cosine is unsigned, so a steady circle '
        'and an unbiased walk score the same.</p>')
    H.append('<div class="cell" style="margin-top:14px">')
    H.append(strip_chart(
        {lab: [t["by_ckpt"][lab]["signed_turn_mean"] for t in T0]
         for lab in labels},
        xlab="mean signed turn (rad/step) — one dot per episode, 144 each",
        vline=0.0))
    H.append('<div class="cap">Dashed line is zero, i.e. no turning bias. '
             'Thick tick is the median. The uncapped arm straddles zero and '
             'flips handedness ~8 times an episode; the capped arm never '
             'flips.</div>')
    H.append("</div>")

    circ = _pick(T0, "p20_e_kcap", "signed_turn_mean", 4, reverse=True)
    H.append('<div class="grid g4">')
    for t in circ:
        r = t["by_ckpt"]["p20_e_kcap"]
        H.append('<div class="cell">'
                 + footprint(r, size, col("p20_e_kcap"))
                 + cap(r) + "</div>")
    H.append("</div>")

    # ------------------------------------------------------------ the pairs
    H.append('<h2>Matched pairs '
             '<span class="meta">same start, same memory, different policy</span></h2>')
    H.append('<p class="lede">The 12% coverage gap, episode by episode. The '
             'uncapped arm reaches the walls and turns hard along them; the '
             'capped arm curls inward and revisits.</p>')
    ok = [t for t in T0 if t["by_ckpt"]["p20_e"]["coverage"] > 0.2]
    step = max(1, len(ok) // 4)
    for t in sorted(ok, key=lambda t: t["by_ckpt"]["p20_e"]["coverage"],
                    reverse=True)[::step][:4]:
        H.append('<div class="grid g2">')
        for lab in labels:
            r = t["by_ckpt"][lab]
            H.append('<div class="cell">'
                     f'<span class="tag">{esc(lab)}</span>'
                     + footprint(r, size, col(lab))
                     + cap(r, [f'revisit {r["revisit_frac"]:.2f}']) + "</div>")
        H.append("</div>")

    # ----------------------------------------------------------- the spread
    H.append('<h2>The spread <span class="meta">best, median and worst of 144'
             '</span></h2>')
    for lab in labels:
        srt = sorted(T0, key=lambda t: t["by_ckpt"][lab]["coverage"])
        # the wall pin is shown above; the interesting worst is the worst
        # ORDINARY episode, not the one pathology already accounted for
        ordinary = [t for t in srt
                    if not (t["by_ckpt"][lab]["clip_frac"] > 0.5
                            and t["by_ckpt"][lab]["speed"] < 0.5)]
        picks = [(ordinary[-1], "best"), (ordinary[len(ordinary) // 2], "median"),
                 (ordinary[0], "worst")]
        H.append(f'<h3>{esc(lab)}</h3>')
        H.append('<div class="grid g3">')
        for t, tag in picks:
            r = t["by_ckpt"][lab]
            H.append('<div class="cell">'
                     f'<span class="tag">{tag}</span>'
                     + footprint(r, size, col(lab)) + cap(r) + "</div>")
        H.append("</div>")

    # -------------------------------------------------------------- numbers
    H.append('<h2>Every axis, both arms <span class="meta">144 episodes each, '
             '0 distractors</span></h2>')
    KEYS = [("coverage", "coverage"), ("edge_frac", "on perimeter"),
            ("straightness", "straightness"),
            ("signed_turn_mean", "signed turn"),
            ("abs_turn_mean", "|turn|"), ("clip_frac", "clipped"),
            ("speed", "realized speed"), ("chase_q", "chase q")]
    H.append('<div class="scroll"><table><thead><tr><th>statistic</th>')
    for lab in labels:
        H.append(f"<th>{esc(lab)}</th>")
    H.append("<th>d10 uncapped</th><th>d10 capped</th></tr></thead><tbody>")
    for k, nice in KEYS:
        H.append(f"<tr><td>{nice}</td>")
        for lab in labels:
            v = sum(t["by_ckpt"][lab][k] for t in T0) / len(T0)
            hi = ' class="hi"' if k in ("coverage", "signed_turn_mean",
                                        "edge_frac") else ""
            H.append(f"<td{hi}>{v:+.3f}</td>" if k == "signed_turn_mean"
                     else f"<td{hi}>{v:.3f}</td>")
        for lab in labels:
            v = (sum(t["by_ckpt"][lab][k] for t in d10["trials"])
                 / len(d10["trials"]))
            H.append(f"<td>{v:+.3f}</td>" if k == "signed_turn_mean"
                     else f"<td>{v:.3f}</td>")
        H.append("</tr>")
    H.append("</tbody></table></div>")

    H.append(
        '<div class="note"><b>Two things this page does not show.</b><br>'
        '<code>revisit_frac</code> is not an independent diagnostic &mdash; it '
        'is coverage restated. At a fixed 200 steps, coverage = '
        '(1&nbsp;&minus;&nbsp;revisit)/2 exactly, and the measured correlation '
        'is &minus;1.00 in both arms. It is in the captions because it is '
        'legible there, not because it adds evidence. And <code>chase_q</code> '
        'stays near zero at ten distractors (+0.022 uncapped, &minus;0.016 '
        'capped), so neither arm is chasing a phantom recall &mdash; the '
        'failure modes here are motor, not memory.</div>')

    H.append('<p class="sub" style="margin-top:34px">'
             'p20_e = navigate_u700.pt (job 21695407) &middot; '
             'p20_e_kcap = navigate_u700.pt (job 21695408) &middot; '
             'EXPERIMENTS_NAV_P2 &sect;18</p>')
    H.append("</div>")
    return "\n".join(H)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True, nargs=2,
                   help="the d0 and d10 files from explore_traj")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    d0 = json.load(open(a.json[0]))
    d10 = json.load(open(a.json[1]))
    with open(a.out, "w") as fh:
        fh.write(build(d0, d10))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
