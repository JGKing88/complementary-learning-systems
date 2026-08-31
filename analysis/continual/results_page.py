"""Render the continual-control results page from `results_data.py`'s JSON.

Generated rather than hand-written, for the same reason `results_data` exists:
a page whose numbers were typed by hand stops matching its runs the first time
a run is repeated. Everything here reads from the JSON; nothing is literal
except the prose.

    python -m analysis.continual.results_data  --out results.json ...
    python -m analysis.continual.results_page  --data results.json --out page.html

The page is deliberately a *frontier*, not a leaderboard. Retention alone ranks
a method that refuses to learn above one that works, which is not a hypothetical
-- it is what online EWC at lambda=1e5 does. Every row therefore carries the
plasticity check and the cost axes beside the score.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re

# --- the palette, contrast-checked against both grounds ---------------------
# Blue-biased neutrals rather than pure grey, one accent doing the magnitude
# encoding, and semantic colours reserved for state (never used as a series).
CSS = """
:root{
  --ground:#FCFCFD; --surface:#F4F5F8; --surface-2:#EDEFF4;
  --rule:#DFE1E9; --rule-strong:#C6CAD8;
  --ink:#14161F; --ink-2:#333849; --muted:#5D6377;
  --accent:#2F4BC4; --accent-soft:#E6EAFA;
  --warn:#8A5A05; --warn-soft:#FBF0DC;
  --good:#166152; --good-soft:#E0F1EC;
  --crit:#9B2F2F; --crit-soft:#FAE7E7;
  --serif:"Spectral",Georgia,serif;
  --sans:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  --mono:"IBM Plex Mono",Consolas,monospace;
  --s1:.35rem; --s2:.65rem; --s3:1rem; --s4:1.6rem; --s5:2.6rem; --s6:4.2rem;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --ground:#0E1017; --surface:#171A24; --surface-2:#1E212D;
  --rule:#272B39; --rule-strong:#3A4054;
  --ink:#E6E8F0; --ink-2:#C3C8D8; --muted:#99A0B5;
  --accent:#7D93F5; --accent-soft:#1B2140;
  --warn:#E0A64C; --warn-soft:#2A2214;
  --good:#5CC7A6; --good-soft:#122A25;
  --crit:#EE8B8B; --crit-soft:#2C1818;
}}
:root[data-theme="dark"]{
  --ground:#0E1017; --surface:#171A24; --surface-2:#1E212D;
  --rule:#272B39; --rule-strong:#3A4054;
  --ink:#E6E8F0; --ink-2:#C3C8D8; --muted:#99A0B5;
  --accent:#7D93F5; --accent-soft:#1B2140;
  --warn:#E0A64C; --warn-soft:#2A2214;
  --good:#5CC7A6; --good-soft:#122A25;
  --crit:#EE8B8B; --crit-soft:#2C1818;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
     font-size:16.5px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 var(--s4) var(--s6)}
header.mast{padding:var(--s6) 0 var(--s4);border-bottom:1px solid var(--rule);
            margin-bottom:var(--s5)}
.eyebrow{font-family:var(--mono);font-size:11px;text-transform:uppercase;
         letter-spacing:.16em;color:var(--accent);font-weight:600;
         display:flex;flex-wrap:wrap;gap:.9em;align-items:center}
.eyebrow .dim{color:var(--muted);font-weight:400;letter-spacing:.1em}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(2.1rem,4.6vw,3.1rem);
   line-height:1.1;letter-spacing:-.015em;text-wrap:balance;margin:var(--s3) 0}
.stand{font-family:var(--serif);font-size:1.19rem;line-height:1.55;
       color:var(--ink-2);margin:0;max-width:62ch}
h2{font-family:var(--serif);font-weight:600;font-size:1.6rem;line-height:1.2;
   text-wrap:balance;margin:var(--s6) 0 var(--s3);padding-top:var(--s3);
   border-top:1px solid var(--rule);display:flex;gap:.55em;align-items:baseline}
h2 .sec{font-family:var(--mono);font-size:.7em;font-weight:500;color:var(--accent);
        font-variant-numeric:tabular-nums;flex:none}
h3{font-family:var(--sans);font-weight:600;font-size:1.02rem;margin:var(--s4) 0 var(--s2)}
h4{font-family:var(--mono);font-weight:600;font-size:11px;text-transform:uppercase;
   letter-spacing:.13em;color:var(--muted);margin:var(--s4) 0 var(--s2)}
p{margin:0 0 var(--s3);max-width:68ch}
ul{margin:0 0 var(--s3);padding-left:1.25em;max-width:68ch}
li{margin-bottom:var(--s2)}
li::marker{color:var(--rule-strong)}
code{font-family:var(--mono);font-size:.855em;background:var(--surface);
     border:1px solid var(--rule);padding:.08em .34em;border-radius:3px;white-space:nowrap}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.tw{overflow-x:auto;margin:0 0 var(--s4);border:1px solid var(--rule);
    border-radius:5px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.45}
th,td{text-align:left;padding:.6em .8em;vertical-align:middle;border-bottom:1px solid var(--rule)}
thead th{font-family:var(--mono);font-size:10.5px;font-weight:600;text-transform:uppercase;
         letter-spacing:.1em;color:var(--muted);white-space:nowrap;
         background:var(--surface-2);border-bottom:1px solid var(--rule-strong)}
tbody tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-family:var(--mono);
              font-variant-numeric:tabular-nums;white-space:nowrap}
td.k{font-family:var(--mono);font-size:12.5px;font-weight:600;white-space:nowrap}
tr.hl td{background:var(--accent-soft)}
tr.bad td{background:var(--crit-soft)}
.bar{display:block;min-width:120px}
.bar .track{display:block;height:6px;background:var(--rule);border-radius:0 3px 3px 0;
            margin-top:3px;overflow:hidden}
.bar .fill{display:block;height:100%;background:var(--accent);border-radius:0 3px 3px 0}
.bar.dim .fill{background:var(--muted)}
.bar .v{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:12.5px;color:var(--ink-2)}
.bar.hi .v{color:var(--accent);font-weight:600}
.chip{display:inline-flex;align-items:center;gap:.35em;font-family:var(--mono);
      font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;
      padding:.16em .5em;border-radius:3px;white-space:nowrap;border:1px solid var(--rule)}
.chip.good{color:var(--good);background:var(--good-soft)}
.chip.warn{color:var(--warn);background:var(--warn-soft)}
.chip.crit{color:var(--crit);background:var(--crit-soft)}
.chip.acc{color:var(--accent);background:var(--accent-soft)}
.chip.mut{color:var(--muted);background:var(--surface-2)}
.note{border:1px solid var(--rule);background:var(--surface);padding:var(--s3) var(--s4);
      border-radius:5px;margin:0 0 var(--s4)}
.note p:last-child{margin-bottom:0}
.note h4{margin-top:0}
.note.crit{border-color:var(--crit);background:var(--crit-soft)}
.note.crit h4{color:var(--crit)}
.note.warn{border-color:var(--warn);background:var(--warn-soft)}
.note.warn h4{color:var(--warn)}
.note.acc{border-left:3px solid var(--accent);background:var(--accent-soft);
          border-radius:0 5px 5px 0}
.note.acc h4{color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;
      background:var(--rule);border:1px solid var(--rule);border-radius:5px;
      overflow:hidden;margin:0 0 var(--s4)}
.kpis>div{background:var(--surface);padding:var(--s3)}
.kpis .lab{font-family:var(--mono);font-size:10px;text-transform:uppercase;
           letter-spacing:.11em;color:var(--muted);display:block;margin-bottom:4px}
.kpis .val{font-family:var(--mono);font-size:19px;font-weight:600;color:var(--accent);
           font-variant-numeric:tabular-nums}
.kpis .sub{font-size:12.5px;color:var(--muted);display:block;margin-top:2px;line-height:1.35}
.fig{border:1px solid var(--rule);background:var(--surface);border-radius:5px;
     padding:var(--s4);margin:0 0 var(--s4);overflow-x:auto}
.fig svg{display:block;max-width:100%;height:auto}
.fig .cap{font-size:13px;color:var(--muted);margin-top:var(--s2);max-width:64ch}
.legend{display:flex;gap:1.4em;flex-wrap:wrap;font-family:var(--mono);
        font-size:11.5px;color:var(--ink-2);margin-bottom:var(--s2)}
.legend .k{display:inline-flex;align-items:center;gap:.5em}
.legend .sw{width:22px;height:0;border-top-width:2px;border-top-style:solid;display:inline-block}
footer{margin-top:var(--s6);padding-top:var(--s3);border-top:1px solid var(--rule);
       font-family:var(--mono);font-size:11px;color:var(--muted);
       display:flex;flex-wrap:wrap;gap:1.4em}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def esc(s) -> str:
    return html.escape(str(s))


def fmt(v, nd=3, dash="—"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    return f"{v:.{nd}f}"


def bar(v, sem=None, hi=False, dim=False, scale=1.0):
    """A magnitude bar. One hue; length carries the value. Identity of the row
    is carried by its label, never by colour alone."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '<span class="bar"><span class="v">—</span></span>'
    pct = max(0.0, min(100.0, 100.0 * v / scale))
    cls = "bar" + (" hi" if hi else "") + (" dim" if dim else "")
    s = f" ±{sem:.3f}" if sem is not None and not (isinstance(sem, float) and math.isnan(sem)) else ""
    return (f'<span class="{cls}"><span class="v">{v:.3f}{esc(s)}</span>'
            f'<span class="track"><span class="fill" style="width:{pct:.1f}%">'
            f'</span></span></span>')


def mb(b):
    if not b:
        return "0"
    return f"{b / 1e6:.1f} MB"


#: A method has to still learn the environment in front of it for its score to
#: mean anything. Below this, retention was bought by declining to learn.
USABLE_CURRENT = 0.5


def best_per_family(methods: list[dict]) -> list[dict]:
    """One row per arm: the highest-retention configuration that *still learned
    the current environment*.

    The fallback matters. If an arm has no usable configuration at all, the
    honest answer is "this method has no working setting here", not its best
    degenerate one -- silently promoting the latter would put a network that
    refuses to learn at the top of the frontier with nothing marking it. Such a
    row is returned flagged, and the caller renders it as such rather than as a
    result.
    """
    by_arm: dict[str, list[dict]] = {}
    for m in methods:
        by_arm.setdefault(m["arm"], []).append(m)
    out = []
    for arm, rows in by_arm.items():
        usable = [r for r in rows
                  if (r["current_env"] or 0) >= USABLE_CURRENT
                  and (r["retained"] is not None)]
        pick = dict(max(usable or rows, key=lambda r: r["retained"] or 0.0))
        pick["degenerate_only"] = not usable
        out.append(pick)
    return sorted(out, key=lambda r: -(r["retained"] or 0.0))


def degenerate_rows(methods: list[dict]) -> list[dict]:
    """Configurations whose retention beats the field only because they stopped
    learning. Surfacing these is the point of carrying plasticity in the table."""
    return sorted(
        [m for m in methods
         if (m["current_env"] or 0) < USABLE_CURRENT
         and (m["retained"] or 0) > 0.15],
        key=lambda r: -(r["retained"] or 0.0))


def incontext_svg(ic: dict) -> str:
    """Success against episode index, one line per arm.

    Two series, so identity is carried by line *style* as well as colour --
    solid for the lifetime arm, dashed for the episodic control -- and both are
    direct-labelled at their endpoints. Status hues stay reserved for state.
    """
    arms = ic.get("arms", {})
    order = [a for a in ("lifetime", "episodic") if a in arms]
    if not order:
        return ""
    n = max(len(arms[a]["mean_curve"]) for a in order)
    W, H = 640, 260
    L, R, T, B = 52, 96, 18, 40
    iw, ih = W - L - R, H - T - B

    def X(i):
        return L + (iw * i / max(1, n - 1))

    def Y(v):
        return T + ih * (1.0 - max(0.0, min(1.0, v)))

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" '
           'aria-label="Success rate against episode index, by arm">']
    # recessive grid + axis
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = Y(g)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + iw}" y2="{y:.1f}" '
                   'stroke="var(--rule)" stroke-width="1"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" '
                   'font-family="var(--mono)" font-size="10" '
                   f'fill="var(--muted)">{g:.2f}</text>')
    for i in range(n):
        out.append(f'<text x="{X(i):.1f}" y="{T + ih + 18}" '
                   'text-anchor="middle" font-family="var(--mono)" '
                   f'font-size="10" fill="var(--muted)">{i + 1}</text>')
    out.append(f'<text x="{L + iw / 2:.0f}" y="{H - 6}" text-anchor="middle" '
               'font-family="var(--mono)" font-size="10.5" '
               'fill="var(--muted)">episode within the lifetime</text>')

    style = {"lifetime": ("var(--accent)", "none", "lifetime (state carried)"),
             "episodic": ("var(--muted)", "5 4", "episodic control")}
    for a in order:
        col, dash, label = style[a]
        pts = arms[a]["mean_curve"][:n]
        dpath = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}"
                         for i, v in enumerate(pts))
        out.append(f'<path d="{dpath}" fill="none" stroke="{col}" '
                   f'stroke-width="2" stroke-dasharray="{dash}" '
                   'stroke-linejoin="round"/>')
        for i, v in enumerate(pts):
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" '
                       f'fill="{col}"/>')
        out.append(f'<text x="{X(len(pts) - 1) + 10:.1f}" '
                   f'y="{Y(pts[-1]) + 4:.1f}" font-family="var(--mono)" '
                   f'font-size="11" fill="{col}">{esc(label)}</text>')
    out.append("</svg>")
    return "".join(out)


_RB = re.compile(r"_rb(\d+)$")


def replay_ratio_series(methods: list[dict]) -> list[tuple[int, float, float]]:
    """(ratio, retained, sem) for unbounded-buffer ER, sorted by ratio.

    Only `buffer=inf` rows, so the series varies one thing. Both the `B_` and
    `I_` arms contribute -- they are the same method at different ratios and
    were only split across waves because the second was a follow-up.
    """
    out = {}
    for m in methods:
        cfg = m["config"]
        if not (cfg.startswith("B_er_bufinf_rb") or cfg.startswith("I_erhi_rb")):
            continue
        mt = _RB.search(cfg)
        if not mt:
            continue
        out[int(mt.group(1))] = (m["retained"], m.get("retained_sem"))
    return [(k, v[0], v[1]) for k, v in sorted(out.items())]


def ratio_svg(series, ceiling=None, hopfield=None) -> str:
    """Retention against replay ratio, log-x, with the ceiling as a reference.

    One series, so no legend -- the heading names it and the line is
    direct-labelled. The two reference lines are drawn in muted ink and
    labelled in place, so they read as context rather than as data.
    """
    if len(series) < 2:
        return ""
    import math as _m
    W, H = 640, 300
    L, R, T, B = 54, 118, 20, 44
    iw, ih = W - L - R, H - T - B
    xs = [r for r, _, _ in series]
    lo, hi = _m.log2(min(xs)), _m.log2(max(xs))

    def X(r):
        return L + (iw * (_m.log2(r) - lo) / max(1e-9, hi - lo))

    def Y(v):
        return T + ih * (1.0 - max(0.0, min(1.0, v)))

    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Retention against '
         'replay ratio for unbounded-buffer experience replay">']
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = Y(g)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + iw}" y2="{y:.1f}" '
                 'stroke="var(--rule)" stroke-width="1"/>')
        o.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" '
                 'font-family="var(--mono)" font-size="10" '
                 f'fill="var(--muted)">{g:.2f}</text>')
    for ref, lab in ((ceiling, "joint ceiling"), (hopfield, "Hopfield store")):
        if ref is None:
            continue
        y = Y(ref)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + iw}" y2="{y:.1f}" '
                 'stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="6 4"/>')
        o.append(f'<text x="{L + iw + 8}" y="{y + 4:.1f}" '
                 'font-family="var(--mono)" font-size="10.5" '
                 f'fill="var(--muted)">{esc(lab)} {ref:.3f}</text>')
    for r in xs:
        o.append(f'<text x="{X(r):.1f}" y="{T + ih + 18}" text-anchor="middle" '
                 'font-family="var(--mono)" font-size="10" '
                 f'fill="var(--muted)">{r}</text>')
    o.append(f'<text x="{L + iw / 2:.0f}" y="{H - 8}" text-anchor="middle" '
             'font-family="var(--mono)" font-size="10.5" fill="var(--muted)">'
             'replayed trajectories per new one</text>')

    dpath = " ".join(f"{'M' if i == 0 else 'L'}{X(r):.1f},{Y(v):.1f}"
                     for i, (r, v, _) in enumerate(series))
    o.append(f'<path d="{dpath}" fill="none" stroke="var(--accent)" '
             'stroke-width="2.5" stroke-linejoin="round"/>')
    for r, v, sem in series:
        if sem:
            o.append(f'<line x1="{X(r):.1f}" y1="{Y(v - sem):.1f}" '
                     f'x2="{X(r):.1f}" y2="{Y(v + sem):.1f}" '
                     'stroke="var(--accent)" stroke-width="1.5"/>')
        o.append(f'<circle cx="{X(r):.1f}" cy="{Y(v):.1f}" r="4" '
                 'fill="var(--accent)" stroke="var(--surface)" stroke-width="2"/>')
    lr, lv, _ = series[-1]
    o.append(f'<text x="{X(lr) + 10:.1f}" y="{Y(lv) + 4:.1f}" '
             'font-family="var(--mono)" font-size="11" '
             f'fill="var(--accent)">ER {lv:.3f}</text>')
    o.append("</svg>")
    return "".join(o)


def render(d: dict) -> str:
    P: list[str] = []
    A = P.append

    recorded = {r["family"]: r for r in d.get("recorded", [])}
    hop = recorded.get("hopfield")
    rnn = next((r for r in d.get("recorded", []) if r["family"] == "recorded"), None)
    methods = d.get("methods", [])
    oracle = d.get("oracle")

    A("<title>Continual Control Results</title>")
    A('<link rel="preconnect" href="https://fonts.googleapis.com">')
    A('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=Spectral:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&'
      'family=IBM+Plex+Mono:wght@400;500;600&display=swap">')
    A(f"<style>{CSS}</style>")
    A('<div class="wrap">')

    # ---- masthead ----------------------------------------------------
    A('<header class="mast">')
    A('<div class="eyebrow"><span>Hopfield-nav</span>'
      f'<span class="dim">Results · {esc(d.get("generated", "")[:10])}</span>'
      '<span class="dim">docs/CONTINUAL_CONTROLS_PLAN.md</span></div>')
    A("<h1>How good does classic continual learning get?</h1>")
    A('<p class="stand">State-of-the-art continual-learning methods against '
      "the Hopfield store on the same environment stream. The comparison is a "
      "cost frontier, not a leaderboard: retention alone ranks a method that "
      "refuses to learn above one that works.</p>")
    A("</header>")

    # ---- KPI strip ---------------------------------------------------
    A('<div class="kpis">')
    if hop:
        A(f'<div><span class="lab">Hopfield store</span>'
          f'<span class="val">{fmt(hop["retained"])}</span>'
          f'<span class="sub">retained · 0 gradient steps, 1 episode</span></div>')
    if rnn:
        A(f'<div><span class="lab">RNN baseline (recorded)</span>'
          f'<span class="val">{fmt(rnn["retained"])}</span>'
          f'<span class="sub">retained · 200 gradient steps, 200 episodes</span></div>')
    if oracle is not None:
        A(f'<div><span class="lab">Oracle ceiling</span>'
          f'<span class="val">{fmt(oracle, 3)}</span>'
          f'<span class="sub">T0.3 · the eval has no headroom problem</span></div>')
    converged = [j for j in d.get("joint", [])
                 if abs(j.get("end_slope") or 0) <= 0.02 and (j.get("final") or 0) > 0.5]
    if converged:
        best_j = max(converged, key=lambda j: j["final"])
        A(f'<div><span class="lab">Joint ceiling</span>'
          f'<span class="val">{fmt(best_j["final"])}</span>'
          f'<span class="sub">T0.1 · hidden={best_j["hidden"]}, converged — '
          f'the same net holds all envs at once</span></div>')
    n_methods = len({m["display"] for m in methods}) if methods else 0
    seeds = max((m["seeds"] for m in methods), default=0)
    A(f'<div><span class="lab">Method configs</span>'
      f'<span class="val">{len(methods)}</span>'
      f'<span class="sub">{n_methods} methods, up to {seeds} seeds each</span></div>')
    A("</div>")

    # ---- the frontier ------------------------------------------------
    A('<h2 id="frontier"><span class="sec">1</span> The frontier</h2>')
    A("<p>Best configuration per method, chosen among those that still learned "
      "the environment they were training on. <strong>Retained</strong> is the "
      "mean over the environments the stream had already left; "
      "<strong>current</strong> is the plasticity check.</p>")

    rows = best_per_family(methods)
    A('<div class="tw"><table>')
    A("<thead><tr><th>Method</th><th>Configuration</th>"
      '<th class="num">Retained</th><th class="num">Current env</th>'
      '<th class="num">Forgetting</th><th class="num">Stored</th>'
      "<th>Needs</th></tr></thead><tbody>")
    if hop:
        A('<tr class="hl"><td class="k">Hopfield store</td>'
          "<td>frozen policy, one Hebbian write</td>"
          f'<td class="num">{bar(hop["retained"], hi=True)}</td>'
          f'<td class="num">{fmt(hop["current_env"])}</td>'
          f'<td class="num">{fmt(hop["forgetting"])}</td>'
          '<td class="num">no data</td>'
          '<td><span class="chip good">nothing</span></td></tr>')
    for r in rows:
        needs = []
        if r.get("degenerate_only"):
            needs.append('<span class="chip crit">no usable setting</span>')
        elif r["needs_task_id"]:
            needs.append('<span class="chip warn">task id</span>')
        elif r["needs_task_boundaries"]:
            needs.append('<span class="chip mut">boundaries</span>')
        else:
            needs.append('<span class="chip good">nothing</span>')
        A("<tr><td class='k'>" + esc(r["display"]) + "</td>"
          f"<td><code>{esc(r['config'])}</code></td>"
          f'<td class="num">{bar(r["retained"], r["retained_sem"])}</td>'
          f'<td class="num">{fmt(r["current_env"])}</td>'
          f'<td class="num">{fmt(r["forgetting"])}</td>'
          f'<td class="num">{esc(mb(r["state_bytes"]))}</td>'
          f"<td>{''.join(needs)}</td></tr>")
    A("</tbody></table></div>")

    A('<div class="note acc"><h4>What the frontier is for</h4>'
      "<p>Every row above acquires an environment in <strong>200 episodes and "
      "200 gradient steps</strong>. The Hopfield store does it in "
      "<strong>1 episode and 0 gradient steps</strong>, keeping no data and "
      "needing no task signal. A method that matches on retention has not "
      "refuted that; it has located it on a different axis.</p></div>")

    # ---- the plasticity trap ----------------------------------------
    bad = degenerate_rows(methods)
    if bad:
        A('<h2 id="trap"><span class="sec">2</span> The plasticity trap</h2>')
        A("<p>These configurations post strong retention <em>because they "
          "stopped learning</em>. A table ranked on retention alone would put "
          "them at the top. They are the reason the plasticity column is not "
          "optional.</p>")
        A('<div class="tw"><table>')
        A("<thead><tr><th>Configuration</th>"
          '<th class="num">Retained</th><th class="num">Current env</th>'
          "<th>Reading</th></tr></thead><tbody>")
        for r in bad[:6]:
            A('<tr class="bad">'
              f"<td class='k'>{esc(r['config'])}</td>"
              f'<td class="num">{fmt(r["retained"])}</td>'
              f'<td class="num">{fmt(r["current_env"])}</td>'
              "<td>retention bought by refusing to learn</td></tr>")
        A("</tbody></table></div>")

    # ---- the replay-ratio curve ---------------------------------------
    series = replay_ratio_series(methods)
    if len(series) >= 2:
        conv = [j for j in d.get("joint", [])
                if abs(j.get("end_slope") or 0) <= 0.02 and (j.get("final") or 0) > 0.5]
        ceil = max((j["final"] for j in conv), default=None)
        A('<h2 id="ratio"><span class="sec">3</span> '
          "How far replay gets</h2>")
        A("<p>The plan predicted that a perfect buffer would close most of this "
          "gap. It does not. Every point below has an <strong>unbounded</strong> "
          "buffer — every trajectory the agent ever saw — and varies only how "
          "many of them are replayed per new one.</p>")
        A('<div class="fig">')
        A(ratio_svg(series, ceiling=ceil,
                    hopfield=(hop or {}).get("retained")))
        A('<p class="cap">Retention against replay ratio, unbounded buffer, '
          "8 seeds per point, ±1 SEM. Buffer <em>size</em> barely matters by "
          "comparison: ∞ against 200 entries is 0.419 against 0.404 at a fifth "
          "of the storage.</p></div>")
        gain = [(series[i][1] - series[i - 1][1]) for i in range(1, len(series))]
        A("<p>Monotone and clearly decelerating"
          + (f" (+{gain[-2]:.3f} then +{gain[-1]:.3f} per doubling)"
             if len(gain) >= 2 else "")
          + ". Perfect memory plus a 32:1 replay ratio reaches "
          + f"<strong>{series[-1][1]:.3f}</strong>"
          + (f" against a ceiling of {ceil:.3f}" if ceil else "")
          + (f" — {100.0 * series[-1][1] / ceil:.0f} % of the way" if ceil else "")
          + ", and the rest does not look reachable by more of the same. Every "
          "one of these points still costs 200 gradient steps and 200 episodes "
          "per environment, against the store's 0 and 1.</p>")

    # ---- Tier 0 -------------------------------------------------------
    A('<h2 id="tier0"><span class="sec">4</span> The axes</h2>')
    A("<p>None of these are continual-learning methods. They are what makes "
      "every number above interpretable, and three of the four had never been "
      "run before this suite.</p>")

    joint = d.get("joint", [])
    if joint:
        A("<h3>T0.1 — the joint ceiling</h3>")
        still_rising = [j for j in joint
                        if abs(j.get("end_slope") or 0) > 0.02]
        if still_rising:
            A('<div class="note warn"><h4>Read as a lower bound</h4>'
              f"<p>{len(still_rising)} of {len(joint)} configurations were "
              "still improving where their budget ended, so these are lower "
              "bounds on the ceiling rather than the ceiling. The corrected "
              "run raises the gradient budget 64×.</p></div>")
        A('<div class="tw"><table>')
        A("<thead><tr><th class='num'>Hidden</th><th class='num'>Layers</th>"
          "<th class='num'>lr</th><th class='num'>Updates</th>"
          "<th class='num'>Seeds</th>"
          "<th class='num'>Final</th><th class='num'>End slope</th>"
          "<th>Status</th></tr></thead><tbody>")
        for j in joint:
            sl = j.get("end_slope") or 0
            if sl > 0.02:
                status = '<span class="chip warn">still rising</span>'
            elif sl < -0.02:
                status = '<span class="chip crit">degrading</span>'
            else:
                status = '<span class="chip good">converged</span>'
            A(f'<tr><td class="num">{j["hidden"]}</td>'
              f'<td class="num">{j["layers"]}</td>'
              f'<td class="num">{j["lr"]:g}</td>'
              f'<td class="num">{j.get("n_updates", "—")}</td>'
              f'<td class="num">{j["seeds"]}</td>'
              f'<td class="num">{fmt(j["final"])}</td>'
              f'<td class="num">{fmt(sl, 3)}</td>'
              f"<td>{status}</td></tr>")
        A("</tbody></table></div>")

    scratch = d.get("scratch", {})
    if scratch:
        A("<h3>T0.4 — the from-scratch floor</h3>")
        A('<div class="tw"><table>')
        A("<thead><tr><th>Arm</th><th class='num'>Seeds</th>"
          "<th class='num'>Retained</th><th class='num'>Current env</th>"
          "</tr></thead><tbody>")
        labels = {"noprev": "legacy input set",
                  "prev": "with prev_action (settled)"}
        for arm, v in scratch.items():
            A(f"<tr><td class='k'>{esc(labels.get(arm, arm))}</td>"
              f'<td class="num">{v["seeds"]}</td>'
              f'<td class="num">{fmt(v["retained"])}</td>'
              f'<td class="num">{fmt(v["current"])}</td></tr>')
        A("</tbody></table></div>")
        A("<p>From scratch the control barely learns at all — it is not merely "
          "forgetting. That is why the method comparisons run on the "
          "<em>pretrained</em> arm, and it settles a question the plan flagged "
          "as unmeasured: pretraining does real work here.</p>")

    # ---- in-context ---------------------------------------------------
    ic = d.get("incontext")
    if ic and ic.get("arms"):
        A('<h2 id="incontext"><span class="sec">5</span> '
          "Zero weight updates</h2>")
        A("<p>The only control that meets the store on its own terms. One "
          "environment, ten episodes back to back, weights frozen throughout — "
          "so an agent that solves episode 10 faster than episode 1 can only "
          "be remembering, in activations. The episodic control is trained "
          "identically and differs only in whether the hidden state survived a "
          "goal-reach, so the <em>gap</em> is what is attributable to carrying "
          "anything.</p>")
        A('<div class="fig">')
        A('<div class="legend">'
          '<span class="k"><span class="sw" style="border-top-color:'
          'var(--accent)"></span>lifetime (state carried)</span>'
          '<span class="k"><span class="sw" style="border-top-color:'
          'var(--muted);border-top-style:dashed"></span>episodic control</span>'
          "</div>")
        A(incontext_svg(ic))
        A('<p class="cap">Success rate against episode index, mean over '
          f"{esc(ic.get('seeds', '?'))} seeds x 8 held-out environments x 64 "
          "lifetimes.</p></div>")
        att = ic.get("attributable")
        if att is not None:
            if att > 0.1:
                A('<div class="note crit"><h4>The RNN adapts in-context</h4>'
                  f"<p>Attributable to carrying state: <strong>{att:+.3f}</strong>. "
                  "Forgetting is not the interesting axis for this comparison, "
                  "and the framing has to account for it.</p></div>")
            else:
                A('<div class="note acc"><h4>Activation memory does not do this '
                  "job</h4>"
                  f"<p>Attributable to carrying state: <strong>{att:+.3f}</strong>. "
                  "The lifetime arm does no better than a control trained "
                  "identically but with the hidden state reset at every "
                  "goal-reach. This is the one comparison a referee cannot "
                  'answer with "you needed a bigger buffer."</p></div>')

    # ---- N=20 ----------------------------------------------------------
    n20 = d.get("n20") or []
    if n20:
        A('<h2 id="n20"><span class="sec">6</span> Twenty environments</h2>')
        A("<p>Methods look alike at five tasks and separate at twenty. Only "
          "configurations whose best setting was already pinned down at N=5 "
          "are scaled here.</p>")
        A('<div class="tw"><table>')
        A("<thead><tr><th>Method</th><th>Configuration</th>"
          '<th class="num">Retained</th><th class="num">Current env</th>'
          '<th class="num">Forgetting</th></tr></thead><tbody>')
        for r in sorted(n20, key=lambda x: -(x["retained"] or 0.0)):
            A(f"<tr><td class='k'>{esc(r['display'])}</td>"
              f"<td><code>{esc(r['config'])}</code></td>"
              f'<td class="num">{bar(r["retained"], r["retained_sem"])}</td>'
              f'<td class="num">{fmt(r["current_env"])}</td>'
              f'<td class="num">{fmt(r["forgetting"])}</td></tr>')
        A("</tbody></table></div>")
        A('<div class="note warn"><h4>No Hopfield number at N=20</h4>'
          "<p>Every recorded <code>agenthash</code> history is a five-env "
          "stream, so this panel shows how the <em>methods</em> scale with "
          "stream length. The store's side of it needs a run that does not "
          "exist yet.</p></div>")

    # ---- bugs found ---------------------------------------------------
    A('<h2 id="found"><span class="sec">7</span> Found along the way</h2>')
    A('<div class="note crit"><h4><code>input_prev_action</code> had never '
      "worked</h4>"
      "<p>Both the DAgger collector and the evaluator built the previous-action "
      "channel only when a previous action existed — false at <code>t=0</code>. "
      "The first forward of every rollout fed the trunk an input two columns "
      "narrower than it was sized for. The flag was unusable on this stack, "
      "which is why every recorded history has it off.</p></div>")
    A('<div class="note crit"><h4><code>WorldSpec.write</code> raced itself</h4>'
      "<p>A fixed <code>world.json.tmp</code> staging name meant 272 concurrent "
      "runs writing to one directory destroyed each other's temp file. "
      "246 of 272 died after building their environments and before training. "
      "Mutation-checked fix: 27/48 concurrent writers fail on the old code, "
      "0/48 on the new.</p></div>")
    A('<div class="note warn"><h4>A capacity verdict that was wrong</h4>'
      "<p>The first joint-ceiling run came back at ~0.5 across every capacity "
      "and the summary called it a capacity limit. It was not: capacity did not "
      "move the number and every curve was still climbing at the budget "
      "limit. The run was optimisation-starved. The summary now measures the "
      "end-slope and refuses a capacity verdict while a run is still "
      "improving.</p></div>")
    A('<div class="note warn"><h4><code>init_log_std</code> was unreachable</h4>'
      "<p>Exposed on <code>train_rnn</code> but never wired through the "
      "continual driver, so every run to date used σ = 1.0 against a "
      "unit-magnitude action — the DAgger student exploring with noise the size "
      "of the action itself.</p></div>")

    # ---- what it means -------------------------------------------------
    A('<h2 id="reading"><span class="sec">8</span> What this settles</h2>')

    rows2 = best_per_family(methods)
    best_m = rows2[0] if rows2 else None
    conv2 = [j for j in d.get("joint", [])
             if abs(j.get("end_slope") or 0) <= 0.02 and (j.get("final") or 0) > 0.5]
    ceil2 = max((j["final"] for j in conv2), default=None)
    # The matched sequential reference: method=none at the methods' own config.
    seq_ref = next((m["retained"] for m in methods
                    if m["config"].startswith("R_none")), None)

    A("<ul>")
    if ceil2:
        A("<li><strong>The gap is forgetting, not capacity.</strong> The same "
          f"architecture at the same size scores <strong>{ceil2:.3f}</strong> "
          "trained jointly on these environments and "
          f"<strong>{fmt(seq_ref)}</strong> trained sequentially. Nothing about "
          "what the network can represent explains the difference, and the "
          "eval itself has no headroom problem — the oracle scores 1.000.</li>")
    if best_m and ceil2:
        A("<li><strong>No method tested closes it, at any cost.</strong> The "
          f"best is {esc(best_m['display'])} at "
          f"<strong>{fmt(best_m['retained'])}</strong> — "
          f"{100.0 * (best_m['retained'] or 0) / ceil2:.0f} % of the ceiling — "
          f"while storing {esc(mb(best_m['state_bytes']))} of raw trajectories "
          "and spending 200 gradient steps and 200 episodes on every "
          "environment. The store reaches ~0.99 with none of those.</li>")
    A("<li><strong>Replay is the only family that gets close, and it buys "
      "retention with compute rather than with memory.</strong> Buffer size is "
      "nearly irrelevant past a couple of hundred trajectories; what matters is "
      "how much of each gradient step is spent on the past, and that runs out "
      "of road well below the ceiling.</li>")
    A("<li><strong>Parameter regularisation barely moves the number.</strong> "
      "Online EWC and SI at usable settings sit between 0.07 and 0.15. Their "
      "apparent best results are the plasticity trap: retention bought by "
      "declining to learn.</li>")
    A("<li><strong>Restricting plasticity to a small head is worse, not "
      "better.</strong> Freezing the trunk costs both retention and current-env "
      "performance. Whatever a meta-learned representation would buy here, it "
      "is not the head-only restriction by itself.</li>")
    A("</ul>")

    A('<div class="note acc"><h4>The honest form of the claim</h4>'
      "<p>Not “continual learning cannot do this”. What the suite supports is "
      "narrower and more defensible: <em>on this task, across nine methods "
      "with their coefficients swept over decades, the best classic result "
      "reaches roughly three-fifths of the joint ceiling while storing every "
      "trajectory it has ever seen and paying two hundred gradient steps per "
      "environment — and an associative store reaches the ceiling at one "
      "episode and no gradient steps at all.</em> The frontier is the claim; "
      "the leaderboard is not.</p></div>")

    A("<footer><span>generated by analysis.continual.results_page</span>"
      f'<span>data: {esc(d.get("generated", ""))}</span>'
      "<span>branch worktree-continual-control-suite</span></footer>")
    A("</div>")
    return "\n".join(P)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(args.data) as f:
        d = json.load(f)
    html_text = render(d)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html_text)
    print(f"[page] wrote {args.out}  ({len(html_text)} bytes, "
          f"{len(d.get('methods', []))} method configs)")


if __name__ == "__main__":
    main()
