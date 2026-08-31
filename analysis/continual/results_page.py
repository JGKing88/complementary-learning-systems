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


def best_per_family(methods: list[dict]) -> list[dict]:
    """One row per arm, the configuration with the highest retention -- but only
    among those that still learned the current env. A method that scores well by
    freezing the network is reported separately, not silently ranked first."""
    by_arm: dict[str, list[dict]] = {}
    for m in methods:
        by_arm.setdefault(m["arm"], []).append(m)
    out = []
    for arm, rows in by_arm.items():
        usable = [r for r in rows
                  if (r["current_env"] or 0) >= 0.5 and (r["retained"] is not None)]
        pool = usable or rows
        out.append(max(pool, key=lambda r: r["retained"] or 0.0))
    return sorted(out, key=lambda r: -(r["retained"] or 0.0))


def degenerate_rows(methods: list[dict]) -> list[dict]:
    """Configurations whose retention beats the field only because they stopped
    learning. Surfacing these is the point of carrying plasticity in the table."""
    return sorted(
        [m for m in methods
         if (m["current_env"] or 0) < 0.5 and (m["retained"] or 0) > 0.15],
        key=lambda r: -(r["retained"] or 0.0))


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
    A('<p class="stand">Nine continual-learning methods against the Hopfield '
      "store on the same five-environment stream. The comparison is a "
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
    A(f'<div><span class="lab">Method configs</span>'
      f'<span class="val">{len(methods)}</span>'
      f'<span class="sub">across 9 methods, 8 seeds each</span></div>')
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
        if r["needs_task_id"]:
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

    # ---- Tier 0 -------------------------------------------------------
    A('<h2 id="tier0"><span class="sec">3</span> The axes</h2>')
    A("<p>None of these are continual-learning methods. They are what makes "
      "every number above interpretable, and three of the four had never been "
      "run before this suite.</p>")

    joint = d.get("joint", [])
    if joint:
        A("<h3>T0.1 — the joint ceiling</h3>")
        still_rising = [j for j in joint if (j.get("end_slope") or 0) > 0.02]
        if still_rising:
            A('<div class="note warn"><h4>Read as a lower bound</h4>'
              f"<p>{len(still_rising)} of {len(joint)} configurations were "
              "still improving where their budget ended, so these are lower "
              "bounds on the ceiling rather than the ceiling. The corrected "
              "run raises the gradient budget 64×.</p></div>")
        A('<div class="tw"><table>')
        A("<thead><tr><th class='num'>Hidden</th><th class='num'>Layers</th>"
          "<th class='num'>lr</th><th class='num'>Seeds</th>"
          "<th class='num'>Final</th><th class='num'>End slope</th>"
          "<th>Status</th></tr></thead><tbody>")
        for j in joint:
            rising = (j.get("end_slope") or 0) > 0.02
            A(f'<tr><td class="num">{j["hidden"]}</td>'
              f'<td class="num">{j["layers"]}</td>'
              f'<td class="num">{j["lr"]:g}</td>'
              f'<td class="num">{j["seeds"]}</td>'
              f'<td class="num">{fmt(j["final"])}</td>'
              f'<td class="num">{fmt(j.get("end_slope"), 3)}</td>'
              + ('<td><span class="chip warn">still rising</span></td>'
                 if rising else '<td><span class="chip good">converged</span></td>')
              + "</tr>")
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

    # ---- bugs found ---------------------------------------------------
    A('<h2 id="found"><span class="sec">4</span> Found along the way</h2>')
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
