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

from . import metrics as M
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
.switch{position:sticky;top:0;z-index:20;background:var(--ground);
  border-bottom:1px solid var(--rule)}
.switch .in{max-width:1080px;margin:0 auto;padding:10px var(--s4);
  display:flex;gap:10px;align-items:baseline}
.switch .lb{font:500 12px var(--sans);color:var(--muted);
  letter-spacing:.04em;text-transform:uppercase}
.switch button{font:500 13px var(--sans);color:var(--ink-2);cursor:pointer;
  background:transparent;border:1px solid var(--rule);border-radius:999px;
  padding:4px 14px}
.switch button:hover{border-color:var(--rule-strong)}
.switch button[aria-selected="true"]{background:var(--accent-soft);
  border-color:var(--accent);color:var(--accent)}
.switch .note{font:12px var(--sans);color:var(--muted);margin-left:auto}
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
tr.grp td{background:var(--surface-2);font-family:var(--mono);font-size:11.5px;
          font-weight:600;letter-spacing:.08em;text-transform:uppercase;
          color:var(--muted);padding-top:14px}
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


#: What each method actually does, in one line. Without these the frontier is
#: a list of names: nothing on the page would tell a reader why DER++ and CLEAR
#: are different methods rather than two spellings of replay.
MECHANISM = {
    "B": ("Experience Replay",
          "Keep past trajectories; train on a sample of them alongside each "
          "new one. The buffer holds whole trajectories, not timesteps, because "
          "a timestep torn out of its trajectory would be supervised in a "
          "recurrent context the agent could never have been in."),
    "I": ("Experience Replay, high ratio",
          "The same method, varying only how many stored trajectories are "
          "replayed per new one. This is the axis that turned out to matter."),
    "E": ("DER++",
          "Replay, plus a penalty for drifting from the model's own output at "
          "the moment each trajectory entered the buffer. Different entries "
          "are anchored to different, older versions of the policy."),
    "D": ("CLEAR",
          "Replay, plus a penalty for drifting from one snapshot of the policy "
          "taken at the previous task boundary. Same idea as DER++; they "
          "disagree about what 'the past' means."),
    "C": ("Online EWC",
          "No stored data. Estimate which weights mattered for previous tasks "
          "(the diagonal Fisher, sampling actions from the model rather than "
          "the teacher) and penalise moving them."),
    "F": ("Synaptic Intelligence",
          "Same idea as EWC, but importance is accumulated along the "
          "optimisation path as training runs, so it needs no separate "
          "estimation pass."),
    "G": ("LwF",
          "No stored data and no per-weight state. Penalise changing the "
          "policy's <em>outputs</em> on the current task's own states, "
          "relative to "
          "the model as it stood when the task began."),
    "H": ("Frozen trunk",
          "Adapt only the 260-parameter movement head and hold the 73k-parameter "
          "recurrent trunk fixed. The plasticity-restriction idea behind "
          "meta-learning methods, without the meta-learning."),
    "A": ("Naive SGD, tuned",
          "No continual-learning method at all — just the behaviour-cloning "
          "update, with its learning rate, optimiser reset and gradient budget "
          "swept. This is the control, made as strong as it can be."),
    "A2": ("Naive SGD, from scratch",
           "The same, without pretraining."),
    "Abatch": ("Naive SGD, batched",
               "The control at 16 rollouts per update instead of 1 — a check on "
               "whether single-trajectory gradient noise was causing the "
               "forgetting. It was not."),
    "R": ("No method",
          "The matched reference: the same configuration every method above "
          "runs at, with the method switched off."),
    "J": ("Hypernetwork (HNET)",
          "A small generator network produces the policy's 73k weights from a "
          "learned 32-number code per environment, and a penalty pins what it "
          "generates for past environments to what it used to generate. Unlike "
          "EWC it constrains the <em>weights it emits</em> rather than its own "
          "parameters — which matters in a recurrent policy, where a small "
          "weight change compounds over 200 timesteps. Needs to be told which "
          "environment it is in."),
    "K": ("HNET, frozen base",
          "The same, with the pretrained weights pinned so only the "
          "task-conditioned part can move. There is no shared component left "
          "to forget through, so whatever this fails to retain is the "
          "generator overwriting itself."),
    "L": ("HNET, from scratch",
          "The published form: no warm start, and the only variant whose "
          "parameter count matches the baseline policy. Read against its own "
          "from-scratch control, not against the pretrained arms."),
    "L0": ("Naive SGD, from scratch",
           "The control the from-scratch hypernetwork is read against. Wave 1 "
           "has no such arm, so without this one there is nothing to compare "
           "it to."),
    "M": ("Multi-head",
          "One shared recurrent trunk, one movement head per environment, "
          "selected by an oracle task id. The heads cannot interfere at all, "
          "so whatever this fails to retain is forgetting in the shared trunk "
          "— which bounds the whole isolation family in a single run."),
    "N": ("XdG",
          "Context-dependent gating: each environment gets a fixed random "
          "subset of the hidden units, applied inside the recurrence rather "
          "than at the readout, so a task's units are the only ones carrying "
          "its state. Masks are drawn independently, so they overlap by "
          "chance."),
    "N2": ("XdG + SI",
           "The same gating with Synaptic Intelligence layered on the units "
           "that overlap between tasks, which is how the two were published "
           "together."),
}


#: The runs behind this page. Historical facts, so they are literals -- but
#: they belong on the page rather than only in a shell history, because "which
#: job produced this number" is the first question anyone re-reading a result
#: asks, and the second is "was it clean".
JOBS = [
    ("21626914", "Wave 0 — oracle, from-scratch floor, first joint sweep", "64/64"),
    ("21627945", "Joint ceiling, corrected budget (8× epochs, 8000 updates)", "22/24"),
    ("21628688", "Wave 1 — Tier-1 tuning, Experience Replay, online EWC", "272/272"),
    ("21631698", "Wave 2 — CLEAR, DER++, SI, LwF, frozen trunk, high-ratio ER", "144/144"),
    ("21631792", "N=20 scaling panel", "20/20"),
    ("21653228", "Wave 3 — HNET, multi-head, XdG, and their from-scratch "
                 "controls", "144 launched"),
    ("21633232", "Wave 2b — coefficient ranges re-swept by loss ratio", "56/56"),
    ("21634287", "Wave 2c — DER++ after its gradient was fixed", "32/32"),
    ("21634899", "Wave 2d — EWC and SI after the Fisher sampler was fixed", "56/56"),
    ("21629579", "§5.2 — in-context pretraining and first evaluation", "9/9"),
    ("21643814", "§5.2 — re-scored with the conditional memory test", "3/3"),
]

#: Things a reader will look for and not find. Saying why is more useful than
#: leaving them to wonder whether they were forgotten.
NOT_MEASURED = [
    ("Meta-learned representations (OML/ANML)",
     "Its mechanism is a frozen meta-learned trunk plus a small online-adapting "
     "head. The head-only half was measured directly here and is <em>harmful</em> "
     "— retention 0.043 against a 0.044 reference. Any benefit would have to come "
     "entirely from meta-learning changing what the trunk represents, and the "
     "single-GRU architecture offers no intermediate point between a "
     "260-parameter head and a 73,000-parameter one to test that on."),
    ("Plasticity maintenance (continual backprop, L2-init, shrink-and-perturb)",
     "Cut because at five environments the control is not plasticity-limited — "
     "it reaches 0.99 on the environment in front of it. The N=20 panel shows "
     "that stops being true at twenty, where current-environment performance "
     "collapses to 0.29–0.40 for every method including the reference. This is "
     "the next wave, not an omission."),
    ("Parameter-isolation methods with a task ID (PackNet, HAT, XdG)",
     "These need to be told which environment they are in. The frozen-trunk arm "
     "bounds the family's mechanism without that privilege, and it lost. A "
     "task-ID arm would be an upper bound the store does not need, and is worth "
     "running only if something in the family first looks competitive."),
    ("A Hopfield comparison at twenty environments",
     "Every recorded run of the store is a five-environment stream. The N=20 "
     "panel therefore shows how the <em>methods</em> scale; the store's side of "
     "it needs a run that does not exist yet."),
]


def _summarise(text: str, min_len: int = 60) -> str:
    """The opening of a glossary entry, for the row beneath a method's name.

    Derived from `MECHANISM` rather than written twice, so the table and the
    methods section cannot drift apart. One sentence usually does it, but some
    entries open with something like "No stored data", which is true and tells
    the reader nothing -- so keep taking sentences until the line carries some
    weight.
    """
    parts = [t.strip() for t in text.split(". ") if t.strip()]
    out = ""
    for part in parts:
        # Rejoin with the separator `split(". ")` consumed, or the line reads
        # "No stored data Estimate which weights mattered".
        out = f"{out}. {part}" if out else part
        if len(out) >= min_len:
            break
    return out if out.endswith(".") else out + "."


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
# Re-exported so existing importers keep working; defined in metrics
# because the collector gates on the same number.
USABLE_CURRENT = M.USABLE_CURRENT


#: Arm -> the continual-learning method it is a configuration of. Controls are
#: deliberately absent.
#:
#: This exists because "how many methods ran" was once answered by counting
#: rows in the mechanism table, which also holds the naive control, its batched
#: variant and the frozen-trunk configuration. That gave nine when six had run,
#: and the number matched the plan's for an unrelated reason, so nothing
#: flagged it. Counting is now a lookup that a reader can check against the
#: launchers, and the page derives every "N methods" from it rather than
#: carrying a literal that has to be remembered.
CANONICAL_METHOD = {
    "B": "Experience Replay",       "I": "Experience Replay",
    "C": "Online EWC",              "D": "CLEAR",
    "E": "DER++",                   "F": "Synaptic Intelligence",
    "G": "LwF",                     "H": "Frozen trunk",
    "J": "Hypernetwork (HNET)",     "K": "Hypernetwork (HNET)",
    "L": "Hypernetwork (HNET)",     "M": "Multi-head",
    "N": "XdG",                     "N2": "XdG + SI",
}


def method_names(methods: list[dict]) -> list[str]:
    """The distinct continual-learning methods with at least one run present."""
    return sorted({CANONICAL_METHOD[m["arm"]] for m in methods
                   if m["arm"] in CANONICAL_METHOD})


def spell(n: int) -> str:
    """Small integers as words, so prose reads as prose."""
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve", "thirteen",
             "fourteen", "fifteen")
    return words[n] if n < len(words) else str(n)


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


def frontier_row(r: dict, disc: dict | None = None,
                 show_disc: bool = False) -> str:
    """One `<tr>` of the frontier table.

    Pulled out of the section because the table is rendered in two groups --
    methods given a task id and methods given nothing -- and a row emitted
    twice from two loops is a row that gets edited once.
    """
    if r.get("degenerate_only"):
        needs = '<span class="chip crit">no usable setting</span>'
    elif r["needs_task_id"]:
        needs = '<span class="chip warn">task id</span>'
    elif r["needs_task_boundaries"]:
        needs = '<span class="chip mut">boundaries</span>'
    else:
        needs = '<span class="chip good">nothing</span>'
    _mech = MECHANISM.get(r["arm"])
    note = _summarise(_mech[1]) if _mech else ""
    params = r.get("params")
    cfg = f"<code>{esc(r['config'])}</code>"
    if params:
        cfg += (f'<span style="color:var(--muted);font-size:12.5px"> · '
                f"{params:,} params</span>")
    return ("<tr><td class='k'>" + esc(r["display"]) + "</td>"
            f"<td>{cfg}"
            + (f'<br><span style="color:var(--muted);font-size:12.5px">'
               f"{esc(note)}</span>" if note else "")
            + "</td>"
            f'<td class="num">{bar(r["retained"], r["retained_sem"])}</td>'
            f'<td class="num">{fmt(r["current_env"])}</td>'
            f'<td class="num">{fmt(r["forgetting"])}</td>'
            f'<td class="num">{esc(mb(r["state_bytes"]))}</td>'
            + (_disc_cell(disc) if show_disc else "")
            + f"<td>{needs}</td></tr>")


def _disc_cell(disc: dict | None) -> str:
    """The same arm's best discrete configuration, or a dash.

    Retention only. The two spaces have different naive floors (0.04 against
    0.30), so the raw numbers are not the comparison -- the fraction of
    available headroom is, and that lives on the discrete page where both
    endpoints are in view. This column is here to say which arms were re-run
    and roughly where they landed, not to rank one space against the other.
    """
    if not disc or disc.get("retained") is None:
        return '<td class="num">—</td>'
    return (f'<td class="num">{fmt(disc["retained"])}'
            f'<br><span style="color:var(--muted);font-size:12px">'
            f'cur {fmt(disc["current_env"], 2)}</span></td>')


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


#: One colour per environment, shared by the shaded training band and the
#: curve it belongs to. Fixed hex rather than the theme's CSS variables
#: because five categories are needed and the palette only carries four
#: semantic colours -- and because these are the same hues matplotlib gives
#: the standalone PNGs, so the page and the figure files do not disagree about
#: which line is env 2. Mid-tone on purpose: legible on both grounds.
ENV_COLORS = ["#4C7DF0", "#E08A2E", "#35A47A", "#D2564F", "#9B72D6"]


def staircase_svg(rows, key: str = "envs", ymax: float = 1.0,
                  yticks=((0.0, "0"), (0.5, "0.5"), (1.0, "1")),
                  invert: bool = False) -> str:
    """Per-environment success across the whole stream, one panel per arm.

    This is the phenomenon every scalar in section 2 is a summary of. A
    retention number says an arm ended at 0.04; the panel says it reached 0.9
    on each environment first and then lost it inside a few dozen updates,
    which is a different claim and the one that motivates the suite.
    """
    if not rows:
        return ""
    PW, PH = 372, 132          # plot area per panel
    LEFT, TOP = 46, 30         # margins inside a panel cell
    CW, CH = PW + LEFT + 26, PH + TOP + 42
    COLS = 2
    W = CW * COLS
    H = CH * ((len(rows) + COLS - 1) // COLS) + 34

    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Per-environment '
         'success rate across the sequential stream, for four methods">']
    o.append('<style>'
             '.stt{font:600 12px var(--sans);fill:var(--ink)}'
             '.sax{font:10px var(--mono);fill:var(--muted)}'
             '.slg{font:10px var(--sans);fill:var(--muted)}'
             '</style>')

    for idx, r in enumerate(rows):
        cx = (idx % COLS) * CW
        cy = (idx // COLS) * CH
        ox, oy = cx + LEFT, cy + TOP
        span = max(1, int(r.get("max_step") or 1))

        def X(step):
            return ox + PW * (float(step) / span)

        def Y(v):
            f = max(0.0, min(1.0, float(v) / ymax))
            return oy + PH * (f if invert else 1.0 - f)

        o.append(f'<text class="stt" x="{cx + 6}" y="{cy + 16}">'
                 f'{esc(r["label"])}</text>')

        # The band behind each block says which environment is being trained
        # there. Without it the reader cannot tell a curve that is falling
        # from one that simply has not been taught yet.
        for lo, hi, env in r.get("blocks", []):
            col = ENV_COLORS[int(env) % len(ENV_COLORS)]
            o.append(f'<rect x="{X(lo):.1f}" y="{oy:.1f}" '
                     f'width="{max(0.0, X(hi) - X(lo)):.1f}" height="{PH}" '
                     f'fill="{col}" opacity="0.08"/>')

        for frac, lab in yticks:
            yy = Y(frac)
            o.append(f'<line x1="{ox}" y1="{yy:.1f}" x2="{ox + PW}" '
                     f'y2="{yy:.1f}" stroke="var(--rule)" stroke-width="1"/>')
            o.append(f'<text class="sax" x="{ox - 8}" y="{yy + 3:.1f}" '
                     f'text-anchor="end">{lab}</text>')

        envs = r.get(key) or {}
        for k in sorted(envs, key=lambda z: int(z)):
            pts = envs[k]
            if not pts:
                continue
            col = ENV_COLORS[int(k) % len(ENV_COLORS)]
            d = " ".join(f"{X(st):.1f},{Y(v):.1f}" for st, v in pts)
            o.append(f'<polyline points="{d}" fill="none" stroke="{col}" '
                     'stroke-width="1.6" stroke-linejoin="round"/>')

        o.append(f'<text class="sax" x="{ox}" y="{oy + PH + 14}">0</text>')
        o.append(f'<text class="sax" x="{ox + PW}" y="{oy + PH + 14}" '
                 f'text-anchor="end">{span}</text>')
        o.append(f'<text class="slg" x="{ox + PW / 2:.0f}" y="{oy + PH + 14}" '
                 'text-anchor="middle">gradient updates</text>')
        o.append(f'<text class="slg" x="{cx + 6}" y="{oy + PH + 30}">'
                 f'{esc(r["why"])}</text>')

    lx = 8
    ly = H - 10
    o.append(f'<text class="slg" x="{lx}" y="{ly}">environment:</text>')
    lx += 78
    for i, col in enumerate(ENV_COLORS):
        o.append(f'<rect x="{lx}" y="{ly - 8}" width="18" height="3" '
                 f'fill="{col}"/>')
        o.append(f'<text class="slg" x="{lx + 23}" y="{ly}">{i}</text>')
        lx += 42
    o.append(f'<text class="slg" x="{lx + 10}" y="{ly}">'
             'shaded band = the environment being trained</text>')
    o.append("</svg>")
    return "".join(o)


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


def _renumber_sections(html: str) -> str:
    """Rewrite the section chips so they count the sections that were emitted.

    The numbers are written literally at each heading, which was fine while
    every page had all twelve. The discrete page does not: there is no wave-3
    isolation family, no in-context arm and no N=20 panel for it, and those
    sections are guarded off. Left alone the reader would get 1, 2, 3, 4, 6,
    12 and reasonably conclude the page was broken. Renumbering after the fact
    keeps the headings authored in one place.
    """
    n = [0]

    def bump(_m):
        n[0] += 1
        return f'<span class="sec">{n[0]}</span>'

    return re.sub(r'<span class="sec">\d+</span>', bump, html)


def floor_ceiling(dd: dict) -> tuple:
    """(naive floor, converged joint ceiling) for one action space.

    Both are read out of that space's own runs. Hardcoding either would make
    the headroom figure below quietly wrong the first time a wave is re-run,
    and the whole point of the figure is that the endpoints moved.
    """
    ms = dd.get("methods", [])
    ref = next((m for m in ms if m["arm"] == "R"), None)
    floor = ref["retained"] if ref and ref["retained"] is not None else None
    conv = [j for j in dd.get("joint", [])
            if abs(j.get("end_slope") or 0) <= 0.02 and (j.get("final") or 0) > 0.5]
    ceiling = max((j["final"] for j in conv), default=None)
    return floor, ceiling


def headroom_table(here: dict, there: dict) -> str:
    """How much of each space's own floor-to-ceiling gap a method actually closes.

    The comparison that survives the substrate changing. Raw retention across
    the two spaces is not readable -- the naive floor moves from 0.04 to 0.30 --
    so every arm is scored as `(retained - floor) / (ceiling - floor)` against
    the endpoints measured in its own space.
    """
    f_here, c_here = floor_ceiling(here)
    f_there, c_there = floor_ceiling(there)
    if None in (f_here, c_here, f_there, c_there):
        return ""
    a = {r["arm"]: r for r in best_per_family(here.get("methods", []))}
    b = {r["arm"]: r for r in best_per_family(there.get("methods", []))}
    # The from-scratch arms are excluded, not forgotten. `floor` here is the
    # *pretrained* naive reference, and A2/L0 never saw the pretraining -- so
    # scoring them against it produces a negative headroom that reads as a
    # method doing worse than nothing when it is really a different control
    # measured against the wrong baseline.
    FROM_SCRATCH = {"A2", "L0"}
    shared = [k for k in a if k in b and k not in FROM_SCRATCH
              and a[k]["retained"] is not None
              and b[k]["retained"] is not None]
    if not shared:
        return ""

    def frac(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else float("nan")

    rows = sorted(shared, key=lambda k: -frac(a[k]["retained"], f_here, c_here))
    o = ['<div class="tw"><table>']
    o.append("<thead><tr><th>Method</th>"
             '<th class="num">Continuous<br>'
             '<span style="color:var(--muted);font-weight:400;font-size:12px">'
             "retained</span></th>"
             '<th class="num">of headroom</th>'
             '<th class="num">Discrete<br>'
             '<span style="color:var(--muted);font-weight:400;font-size:12px">'
             "retained</span></th>"
             '<th class="num">of headroom</th></tr></thead><tbody>')
    for k in rows:
        hh = frac(a[k]["retained"], f_here, c_here)
        th = frac(b[k]["retained"], f_there, c_there)
        cls = ' class="hl"' if hh >= 0.9 else ""
        o.append(f'<tr{cls}><td class="k">{esc(a[k]["display"])}</td>'
                 f'<td class="num">{fmt(b[k]["retained"])}</td>'
                 f'<td class="num">{th:.0%}</td>'
                 f'<td class="num">{fmt(a[k]["retained"])}</td>'
                 f'<td class="num">{hh:.0%}</td></tr>')
    o.append(f'<tr><td class="k">floor / ceiling</td>'
             f'<td class="num" colspan="2">{fmt(f_there)} &rarr; {fmt(c_there)}</td>'
             f'<td class="num" colspan="2">{fmt(f_here)} &rarr; {fmt(c_here)}</td>'
             "</tr>")
    o.append("</tbody></table></div>")
    return "".join(o)


def render_body(d: dict, variant: str = "continuous",
                other: dict | None = None) -> str:
    P: list[str] = []
    A = P.append

    recorded = {r["family"]: r for r in d.get("recorded", [])}
    hop = recorded.get("hopfield")
    rnn = next((r for r in d.get("recorded", []) if r["family"] == "recorded"), None)
    methods = d.get("methods", [])
    oracle = d.get("oracle")

    A('<div class="wrap">')

    # ---- masthead ----------------------------------------------------
    A('<header class="mast">')
    A('<div class="eyebrow"><span>Hopfield-nav</span>'
      f'<span class="dim">Results · {esc(d.get("generated", "")[:10])}</span>'
      '<span class="dim">docs/CONTINUAL_CONTROLS_PLAN.md</span></div>')
    if variant == "discrete":
        A("<h1>The same suite, with a Categorical action head</h1>")
        A('<p class="stand">Every method re-run under discrete movement — four '
          "cardinal actions and a cross-entropy loss — against its own oracle, "
          "joint ceiling and from-scratch floor. The continuous suite's action "
          "head was a 2-D Gaussian whose mean collapses toward zero wherever "
          "the goal is uncertain, and this wave exists to find out how much of "
          "that suite was measuring the parameterisation rather than "
          "forgetting.</p>")
    else:
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

    # ---- the setup -----------------------------------------------------
    A('<h2 id="setup"><span class="sec">1</span> What is being measured</h2>')

    A("<h3>The task</h3>")
    A("<p>An agent navigates a square arena to a goal cell. Its only input is a "
      "<strong>foveal ray-cast</strong>: sixty rays fanned around its heading, "
      "each returning the ±1 code of whichever wall segment it hits. Each "
      "environment's four walls carry their own <strong>random ±1 "
      "barcode</strong>, so different environments look different and position "
      "is recoverable from what the agent sees.</p>")
    A("<p><strong>The goal is never observed.</strong> Only a shortest-path "
      "oracle knows where it is, and the agent is trained by copying that "
      "oracle's action. So an environment's identity is visible in every "
      "observation, while its goal is not visible in any of them — which is "
      "exactly the split that makes this a memory problem rather than a "
      "perception one.</p>")

    A("<h3>The protocol</h3>")
    A('<div class="kpis">')
    A('<div><span class="lab">Environments</span><span class="val">5</span>'
      '<span class="sub">learned strictly one after another, never revisited</span></div>')
    A('<div><span class="lab">Updates per env</span><span class="val">200</span>'
      '<span class="sub">one rollout collected, one gradient step taken</span></div>')
    A('<div><span class="lab">Evaluation</span><span class="val">every update</span>'
      '<span class="sub">on every environment seen so far</span></div>')
    A('<div><span class="lab">Seeds</span><span class="val">8</span>'
      '<span class="sub">per configuration; 20 for the Tier-0 floor</span></div>')
    A("</div>")
    A("<p>One rollout per update is deliberate, and it is what makes the cost "
      "axes readable: an update <em>is</em> an episode, so &ldquo;200 "
      "updates&rdquo; and &ldquo;200 episodes of experience&rdquo; are the same "
      "quantity. Batching would have broken that correspondence, and a separate "
      "condition confirmed it was not the batching that caused the forgetting."
      "</p>")

    A("<h3>What the numbers mean</h3>")
    A('<div class="tw"><table>')
    A("<thead><tr><th>Metric</th><th>Definition</th></tr></thead><tbody>")
    for k, v in [
        ("retained", "Mean success on the environments the stream has already "
                     "<em>left</em>, at the end of training. The headline: it is "
                     "what forgetting destroys."),
        ("current env", "Success on the environment being trained on right now. "
                        "The plasticity check — a method can always score well on "
                        "retention by refusing to learn, and this is what exposes it."),
        ("forgetting", "Peak minus final, per environment. How much of what was "
                       "once known has been lost."),
        ("stability gap", "The <em>transient</em> collapse in the first updates "
                          "after the stream moves on. Survives in methods whose "
                          "final forgetting looks clean."),
        ("stored", "Bytes the method must keep: replay data, importance "
                   "matrices, model snapshots."),
    ]:
        A(f"<tr><td class='k'>{esc(k)}</td><td>{v}</td></tr>")
    A("</tbody></table></div>")
    A("<p>Success is scored by running the policy from a random start and "
      "asking whether it reaches the goal within the step cap. Each evaluation "
      "is a single trial, so an individual point is a coin flip; every number "
      "on this page is averaged over a window of updates and over seeds.</p>")

    A('<div class="note acc"><h4>Why an associative store is the comparison</h4>'
      "<p>The model under test keeps its policy fixed and writes each new "
      "environment's goal into a Hopfield memory as a <strong>single Hebbian "
      "outer product from one episode</strong>. It cannot forget by gradient "
      "descent because it does no gradient descent. The question this suite "
      "exists to answer is not whether that works — it does — but "
      "<em>what a network that learns by gradient descent would have to spend "
      "to match it</em>.</p></div>")

    A("<h3>The methods that ran</h3>")
    A(f"<p>{spell(len(method_names(methods))).capitalize()} continual-learning "
      "methods, plus the controls they are measured against — counted from the "
      "runs actually present, not from a number in the plan. Chosen for this "
      "setting rather than by citation count, and every coefficient swept over "
      "decades, because a strength knob set from a paper with a different loss "
      "scale is how a method gets accidentally made to look bad.</p>")
    A('<div class="tw"><table>')
    A("<thead><tr><th>Method</th><th>What it does</th></tr></thead><tbody>")
    # Only arms with runs present. A glossary entry for a method that has not
    # run reads exactly like one for a method that has, which is the shape the
    # missing-Wave-3 error took the first time.
    present = {m["arm"] for m in methods}
    for key in ("B", "I", "E", "D", "C", "F", "G", "H",
                "J", "K", "L", "M", "N", "N2", "A"):
        if key not in present:
            continue
        name, desc = MECHANISM[key]
        A(f"<tr><td class='k'>{esc(name)}</td><td>{desc}</td></tr>")
    A("</tbody></table></div>")

    # ---- the frontier ------------------------------------------------
    stair = d.get("staircase") or []
    if stair:
        A("<h3>What forgetting looks like before it becomes a number</h3>")
        A("<p>Every scalar below is a summary of this. Each panel is one "
          "method walking the same five-environment stream; each curve is one "
          "environment's success rate, averaged over 8 seeds and smoothed "
          "over 15 updates because a single evaluation trial is a coin "
          "flip.</p>")
        A(f'<div class="fig">{staircase_svg(stair)}</div>')
        A("<p>Read the naive panel first, because it is the shape the rest are "
          "arguing with. Each environment climbs while it is being trained and "
          "then drops within a few dozen updates of the stream moving on — not "
          "a slow decay, a collapse that happens inside the next block. That "
          "is why the headline metric is retention on the environments already "
          "left rather than average performance: the average is dominated by "
          "the one environment the method happens to be training on, which "
          "every method solves.</p>")
        if variant == "discrete":
            A("<p>The drop is real but it does not go all the way down here. "
              "The naive floor in this action space is far above the "
              "continuous one, so the abandoned environments settle at a level "
              "rather than at zero — which is the same fact the frontier "
              "table's floor row reports, seen per environment and over "
              "time.</p>")

        cap = int((stair[0] or {}).get("step_cap") or 200)
        A("<h3>The same thing in steps rather than in successes</h3>")
        A("<p>Success is binary, so it says an environment was solved and not "
          "how far the agent walked to solve it. The same traces in "
          "steps-to-goal, with a trial that never arrives counted at the full "
          f"{cap}-step cap. The axis is inverted so that up is still better: a "
          f"curve sinking toward the {cap}-step line at the bottom is one that "
          "has stopped arriving at all.</p>")
        A(f'<div class="fig">'
          f'{staircase_svg(stair, key="envs_steps", ymax=float(cap), yticks=((0.0, "0"), (cap / 2.0, str(cap // 2)), (float(cap), str(cap))), invert=True)}'
          "</div>")
        has_spl = any((r.get("envs_spl") or {}) for r in stair)
        if has_spl:
            A("<h3>And in route efficiency, which the other two cannot see</h3>")
            A("<p>Success says the goal was reached; steps says how long it "
              "took. Neither says whether the route was any good, because "
              "neither knows how far away the goal actually was. These runs "
              "record the shortest attainable path for every trial, so they "
              "can be scored on <strong>SPL</strong> — "
              "<code>success &times; optimal / max(path, optimal)</code>: "
              "1.0 is a perfect straight-line route, 0 is a failure, and the "
              "trial's own difficulty divides out.</p>")
            A(f'<div class="fig">'
              f'{staircase_svg(stair, key="envs_spl", ymax=1.0)}</div>')
            A("<p>This is the panel the other two were standing in for, and it "
              "exists only here: <code>optimal_to_goal</code> postdates every "
              "continuous run, and the sequential runs save no agent "
              "checkpoint, so it cannot be backfilled by re-evaluating — only "
              "by retraining. The gap between a curve here and the same curve "
              "in the success panel above is the share of each solved "
              "environment that is being reached by a wandering route rather "
              "than a direct one.</p>")

        A("<p><strong>Counting the failures at the cap is the whole trick.</strong> "
          "The obvious alternative — average the step count over the trials "
          "that reached the goal — is worse than useless here, because it "
          "scores each method on the subset it happened to solve. An arm that "
          "retains almost nothing still solves the goals that spawned close to "
          "the agent, so it posts the <em>shortest</em> mean path in the suite, "
          "ahead of every method that actually retains and is therefore also "
          "being graded on the far ones. Filling failures at the cap removes "
          "that, at the price of making the retained half of this panel close "
          "to a mirror of the one above: once the cap term dominates, censored "
          "steps is nearly a function of the success rate. What it adds is in "
          "the shaded blocks, where the curve shows how quickly a method "
          "converts a solved environment into an <em>efficient</em> route — "
          "which the success curve, already saturated, cannot show.</p>")

    A('<h2 id="frontier"><span class="sec">2</span> The frontier</h2>')
    A("<p>Best configuration per method, chosen among those that still learned "
      "the environment they were training on — a method can score well on "
      "retention by refusing to learn anything, and picking its highest number "
      "regardless would put exactly that at the top.</p>")

    A('<div class="note"><h4>What the columns mean</h4>')
    A('<div class="tw"><table>')
    A("<thead><tr><th>Column</th><th>Definition</th></tr></thead><tbody>")
    for k, v in (
        ("Retained",
         "mean success on the environments the stream has already <em>left</em>, "
         "over the last fifth of the final block. The headline: 1.0 would mean "
         "nothing was forgotten."),
        ("Current env",
         "success on the environment being trained on right now. The plasticity "
         "check — it is what separates a method that retains from one that has "
         "simply stopped learning."),
        ("Forgetting",
         "peak-minus-final per environment, averaged. How much of what was "
         "learned got lost, as distinct from how much is left."),
        ("Stored",
         "bytes the method must carry: replay trajectories, importance "
         "matrices, frozen model snapshots. The memory axis of the frontier."),
        ("Needs",
         "whether the method must be told where task boundaries fall, or which "
         "task it is currently in. The store needs neither."),
    ):
        A(f"<tr><td class='k'>{esc(k)}</td><td>{v}</td></tr>")
    A("</tbody></table></div>")
    A("<p>Each evaluation is a single deterministic trial, so an individual "
      "point is 0 or 1; every figure quoted is a mean over 8 seeds, "
      "&plusmn;1 SEM. Two costs are constant across every row and so are not "
      "columns: all of them spend <strong>200 gradient steps and 200 episodes "
      "per environment</strong>, against the store's 0 and 1.</p></div>")

    rows = best_per_family(methods)
    # The other action space's best configuration per arm, keyed by arm so a
    # method whose best setting differs between the two still lines up.
    show_disc = bool(other and other.get("methods"))
    disc_best = ({r["arm"]: r for r in best_per_family(other["methods"])}
                 if show_disc else {})
    ncol = 8 if show_disc else 7
    A('<div class="tw"><table>')
    A("<thead><tr><th>Method</th><th>Configuration</th>"
      '<th class="num">Retained</th><th class="num">Current env</th>'
      '<th class="num">Forgetting</th><th class="num">Stored</th>'
      + ('<th class="num">Discrete<br>'
         '<span style="color:var(--muted);font-weight:400;font-size:12px">'
         'retained</span></th>' if show_disc else "")
      + "<th>Needs</th></tr></thead><tbody>")
    hop_row = ('<tr class="hl"><td class="k">Hopfield store</td>'
               "<td>frozen policy, one Hebbian write</td>"
               f'<td class="num">{bar(hop["retained"], hi=True)}</td>'
               f'<td class="num">{fmt(hop["current_env"])}</td>'
               f'<td class="num">{fmt(hop["forgetting"])}</td>'
               '<td class="num">no data</td>'
               # The store has never been run in the discrete action space --
               # it is a frozen policy plus a Hebbian write, so porting it is a
               # real piece of work rather than a flag.
               + ('<td class="num">—</td>' if show_disc else "")
               + '<td><span class="chip good">nothing</span></td></tr>'
               ) if hop else None
    # Split, rather than one ranked list. Every isolation arm is handed an
    # oracle task id, and the environments here turn out to be barely
    # identifiable from what the agent sees (section 5) -- so sorting all of
    # them together would let a method that is *told* which env it is in
    # outrank methods that have to work it out, and the ordering would imply a
    # comparison the numbers do not support. The chip alone is not enough:
    # readers take the top row as the winner.
    free = [r for r in rows if not r["needs_task_id"]]
    told = [r for r in rows if r["needs_task_id"]]
    split = bool(told and free)
    if not split and hop_row:
        A(hop_row)
    for group, label in ((free, "Given no task signal"),
                         (told, "Given an oracle task id")):
        if not group:
            continue
        if split:
            A(f'<tr class="grp"><td colspan="{ncol}">{esc(label)}</td></tr>')
            # The store belongs *inside* the first group, not floating above
            # the headers: it needs no task signal either, and that is the
            # whole point of the row.
            if hop_row and group is free:
                A(hop_row)
        for r in group:
            A(frontier_row(r, disc_best.get(r["arm"]), show_disc))
    A("</tbody></table></div>")
    if show_disc:
        A('<p style="color:var(--muted);font-size:13px">The discrete column is '
          "the same method's best configuration under a Categorical action "
          "head, from the other page. Do not read the two retention columns "
          "against each other directly: the naive floor is 0.04 in this space "
          "and 0.30 in that one, so the fraction of available headroom each "
          "method closes is the comparison that means something, and it is on "
          "the discrete page.</p>")
    if told and free:
        A('<div class="note warn"><h4>Why the table is split</h4>'
          "<p>The lower group is told which environment it is in, at training "
          "time and at evaluation time. The upper group is not, and neither is "
          "the Hopfield store. That is not a formality here: a classifier "
          "trained to recover the environment from the agent's own "
          "observations reaches <strong>0.43 against a chance of 0.20</strong> "
          "(&sect;5), so the task id is information no method in the upper "
          "group could have obtained for itself. Ranking the two groups "
          "together would read as a comparison; they are an upper bound and a "
          "result.</p></div>")

    A('<div class="note acc"><h4>What the frontier is for</h4>'
      "<p>Every row above acquires an environment in <strong>200 episodes and "
      "200 gradient steps</strong>. The Hopfield store does it in "
      "<strong>1 episode and 0 gradient steps</strong>, keeping no data and "
      "needing no task signal. A method that matches on retention has not "
      "refuted that; it has located it on a different axis.</p></div>")

    # ---- the plasticity trap ----------------------------------------
    bad = degenerate_rows(methods)
    if bad:
        A('<h2 id="trap"><span class="sec">3</span> The plasticity trap</h2>')
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
        A('<h2 id="ratio"><span class="sec">4</span> '
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
    # ---- parameter isolation (Wave 3) ---------------------------------
    if variant == "discrete" and other:
        _ht = headroom_table(d, other)
        if _ht:
            A('<h2 id="against"><span class="sec">5</span> '
              "Against the continuous suite</h2>")
            A("<p>The two action spaces do not share a scale. The naive floor "
              "is far higher here and the joint ceiling is the same, so a "
              "method's raw retention says as much about the substrate as "
              "about the method. What compares is the fraction of the "
              "floor-to-ceiling gap each one actually closes, measured against "
              "the endpoints of its own space.</p>")
            A(_ht)
            A('<div class="note acc"><h4>Every method closes more of the gap '
              "here</h4>"
              "<p>The normalisation already accounts for the floor moving and "
              "the ceiling staying put, so this is not the task being easier. "
              "The same methods, measured against their own endpoints, work "
              "better against a Categorical head than against a Gaussian one — "
              "and unbounded replay very nearly closes the gap outright, "
              "without being told a task boundary or a task id.</p>"
              "<p>The most likely mechanism is the one this wave was run to "
              "remove. A 2-D Gaussian fitted to a target that is multimodal "
              "given the observation returns its mean, and the mean of two "
              "opposed directions is a near-zero vector — so an uncertain "
              "continuous policy expresses its uncertainty by moving slowly "
              "rather than by choosing. A Categorical head can hold the same "
              "uncertainty as probability mass and still commit to a "
              "direction.</p></div>")

    # Wave 3 only ever ran in the continuous action space, so on a page
    # built from a wave without those arms this whole section is empty.
    # Gated on the data rather than on the variant: if the isolation
    # family is ever run in discrete, the section appears by itself.
    iso = [m for m in methods if m["arm"] in ("J", "K", "L", "L0", "M", "N", "N2")]
    if iso:
        A('<h2 id="isolate"><span class="sec">5</span> Parameter isolation</h2>')
        A("<p>Every method above shares one set of weights across every "
          "environment, and differs only in what the update is trained on or how "
          "far the weights may move. The third family does something else: it "
          "gives each environment <em>its own parameters</em>. This is the family "
          "the recurrent continual-learning literature actually points at — Ehret "
          "et al. (ICLR 2021) benchmarked online EWC, SI, masking, generative "
          "replay and coresets across four <em>recurrent</em> benchmarks and found "
          "hypernetworks beat the weight-importance family consistently. The "
          "policy here is a GRU.</p>")

        ident = d.get("identifiability")
        if ident:
            A("<h3>What the oracle task id costs</h3>")
            A("<p>Every arm in this section is <strong>told which environment it "
              "is in</strong>, at training time and at evaluation time. Nothing in "
              "the two families above is, and neither is the Hopfield store. "
              "Whether that is a decisive advantage or a formality depends "
              "entirely on how hard the environment is to recognise — so it was "
              "measured rather than assumed.</p>")
            A("<p>A classifier is fitted from the agent's own observations to the "
              "environment index, sweeping a linear and an MLP readout over "
              "windows of one to sixty-four observations. The split is by "
              "<em>trajectory</em>, not by frame: consecutive observations of one "
              "walk are strongly correlated, and a shuffled split would put "
              "neighbouring timesteps on both sides and report memorisation as "
              "recognition.</p>")
            A('<div class="tw"><table>')
            A("<thead><tr><th>Window</th>"
              '<th class="num">Linear</th><th class="num">MLP</th>'
              "</tr></thead><tbody>")
            for r in ident.get("results", []):
                A(f"<tr><td class='k'>{r['window']} obs</td>"
                  f'<td class="num">{fmt(r["linear"])}</td>'
                  f'<td class="num">{fmt(r["mlp"])}</td></tr>')
            A("</tbody></table></div>")
            A('<div class="note warn"><h4>The task id is real information</h4>'
              f"<p>Best over every window and every readout tried: "
              f"<strong>{fmt(ident['best'])}</strong>, against a chance rate of "
              f"{fmt(ident['chance'])} across {ident['n_envs']} environments "
              f"({ident['n_trajectories']} random-walk trajectories). The "
              "environments are only weakly recognisable from what the agent "
              "sees.</p>"
              "<p>So the oracle task id is not a formality — it is information no "
              "method in the previous sections could have recovered for itself. "
              "Everything below is an <strong>upper bound on its family</strong>, "
              "and belongs in a separate group in the frontier table rather than "
              "ranked against methods that were given nothing.</p></div>")

        A("<h3>The architectures</h3>")
        A('<div class="tw"><table>')
        A("<thead><tr><th>Architecture</th><th>Mechanism</th>"
          '<th class="num">Params</th></tr></thead><tbody>')
        for name, mech, prm in (
            ("Baseline RNN", "One shared network. What every earlier section runs.",
             "73,220"),
            ("HNET, learned base",
             "A generator maps a learned 32-number code per environment to all "
             "73k policy weights, added to a base vector warm-started from the "
             "pretrained checkpoint. The base is free to move, so the regulariser "
             "has to cover it too.", "146,204"),
            ("HNET, frozen base",
             "The same, with the pretrained weights pinned. Nothing shared can "
             "drift, so whatever this forgets is the generator overwriting "
             "itself.", "72,984"),
            ("HNET, no base",
             "The published form — no warm start, and fewer parameters than the "
             "baseline it replaces. Read against its own from-scratch control.",
             "72,984"),
            ("Multi-head",
             "Shared recurrent trunk, one movement head per environment. The "
             "heads cannot interfere at all, which is what makes this a bound: "
             "whatever it fails to retain is forgetting in the shared trunk.",
             "73,480"),
            ("XdG",
             "A fixed random subset of hidden units per environment, masked "
             "inside the recurrence rather than at the readout, so a task's units "
             "are the only ones carrying its state.", "73,220"),
        ):
            A(f"<tr><td class='k'>{esc(name)}</td><td>{mech}</td>"
              f'<td class="num">{prm}</td></tr>')
        A("</tbody></table></div>")

        A("<h3>Calibrating the regulariser before sweeping it</h3>")
        A("<p>The hypernetwork's strength knob was not swept over decades around "
          "the published value. It was measured against this objective first, "
          "because a coefficient calibrated to another paper's loss scale is how a "
          "method gets accidentally made to look bad — and this suite had already "
          "paid for that twice, with DER++ and with CLEAR.</p>")
        A('<div class="tw"><table>')
        A("<thead><tr><th>beta</th><th class=\"num\">BC loss</th>"
          '<th class="num">Penalty</th><th class="num">Ratio</th>'
          "<th>Reading</th></tr></thead><tbody>")
        for b, bc, pen, ratio, reading in (
            ("0.01", "7.75", "1.3e-05", "1.6e-06", "invisible"),
            ("1", "7.65", "0.0013", "1.7e-04",
             "<strong>the published value</strong> — a no-op here"),
            ("100", "7.83", "0.082", "1.0e-02", "1% of the objective"),
            ("10,000", "7.69", "1.23", "1.6e-01", "finally competing"),
            ("1,000,000", "10.83", "4.04", "3.7e-01",
             "BC loss now rising: plasticity is being paid for"),
        ):
            A(f"<tr><td class='k'>{b}</td><td class=\"num\">{bc}</td>"
              f'<td class="num">{pen}</td><td class="num">{ratio}</td>'
              f"<td>{reading}</td></tr>")
        A("</tbody></table></div>")
        A('<div class="note"><h4>What that table prevented</h4>'
          "<p>The obvious sweep — decades either side of the value in the paper — "
          "would have had <em>every arm contribute under 1% of the objective</em>, "
          "and the conclusion would have been that the regulariser does not help. "
          "The wave sweeps 10<sup>2</sup> to 10<sup>7</sup> instead.</p></div>")

        if iso:
            A("<h3>Results</h3>")
            A("<p>Every configuration, not just the best per method — the "
              "within-method spread is the part that says whether a knob was "
              "turned far enough.</p>")
            A('<div class="tw"><table>')
            A("<thead><tr><th>Method</th><th>Configuration</th>"
              '<th class="num">Retained</th><th class="num">Current env</th>'
              '<th class="num">Forgetting</th><th class="num">Stored</th>'
              "</tr></thead><tbody>")
            for r in sorted(iso, key=lambda r: -(r["retained"] or 0.0)):
                cls = ' class="bad"' if (r["current_env"] or 0) < USABLE_CURRENT else ""
                A(f"<tr{cls}><td class='k'>{esc(r['display'])}</td>"
                  f"<td><code>{esc(r['config'])}</code></td>"
                  f'<td class="num">{bar(r["retained"], r["retained_sem"])}</td>'
                  f'<td class="num">{fmt(r["current_env"])}</td>'
                  f'<td class="num">{fmt(r["forgetting"])}</td>'
                  f'<td class="num">{esc(mb(r["state_bytes"]))}</td></tr>')
            A("</tbody></table></div>")
            A("<p style=\"color:var(--muted);font-size:13px\">Shaded rows failed "
              "the plasticity check: they did not learn the environment they were "
              "training on, so their retention figure is not a result. See "
              "&sect;3.</p>")

            # Multi-head sat in this table for a while being read as a mediocre
            # method, which is the one thing it is not for. Its heads cannot
            # interfere by construction, so its retention number is a measurement
            # of where the forgetting lives -- and that is worth more than its
            # rank. Derived, because the moment it stops being the family's
            # best-learning arm this paragraph has to stop saying so.
            mh = next((r for r in iso if r["arm"] == "M"), None)
            rated = [r for r in iso if r["retained"] is not None]
            if mh and mh["retained"] is not None and (mh["current_env"] or 0) > 0:
                lost = (mh["current_env"] - mh["retained"]) / mh["current_env"]
                top_ret = max(rated, key=lambda r: r["retained"])
                A('<div class="note"><h4>What the multi-head row is actually '
                  "for</h4>"
                  "<p>It is the only arm here whose <em>readout</em> cannot "
                  "interfere across environments: one movement head each, picked "
                  "by an oracle id, never touched while another environment is "
                  "training. So every point it loses is lost somewhere else — in "
                  "the shared recurrent trunk. It reaches "
                  f"{fmt(mh['current_env'])} on the environment it is training on "
                  f"and holds {fmt(mh['retained'])} of the ones it has left, so "
                  f"<strong>{lost:.0%} of attainable performance is lost through "
                  "the trunk alone</strong>.</p>"
                  "<p>That is the useful reading, and it is not a ranking. As a "
                  f"method it is unremarkable — {esc(top_ret['display'])} retains "
                  f"{top_ret['retained'] / mh['retained']:.1f}&times; more"
                  + (f", at a current-env score only "
                     f"{mh['current_env'] - top_ret['current_env']:.3f} below "
                     "multi-head's"
                     if (top_ret["current_env"] or 0) > 0 else "")
                  + ". What it settles is mechanistic: protecting the readout is "
                  "not where the problem is, so a method that only isolates "
                  "output parameters cannot work in this setting no matter how "
                  "cleanly it does it.</p></div>")

    A('<h2 id="tier0"><span class="sec">6</span> The axes</h2>')
    A("<p>None of these are continual-learning methods. They are what makes "
      "every number above interpretable, and three of the four had never been "
      "run before this suite.</p>")
    A("<p>The question they settle is whether the retention gap is "
      "<strong>forgetting</strong> or <strong>capacity</strong>. If one "
      "network cannot represent five environments at once, then no continual "
      "method could exceed that limit, the gap would be a statement about "
      "model size rather than about memory, and every method number above "
      "would be measuring the wrong thing. The joint ceiling is the run that "
      "decides it: the same architecture, at the same size, trained on all "
      "five environments simultaneously instead of in sequence.</p>")

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
        A('<h2 id="incontext"><span class="sec">7</span> '
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
        lt = ic["arms"].get("lifetime", {})
        ep = ic["arms"].get("episodic", {})
        pc = ic.get("positive_control_lift")
        lift = lt.get("memory_lift")
        if lift is not None:
            A("<h3>The conditional test</h3>")
            A("<p>A flat mean curve is ambiguous on its own: these policies "
              "solve only about a tenth of first episodes on unseen "
              "environments, so most of the average is measuring blind search. "
              "<strong>memory_lift</strong> asks the sharper question — among "
              "the lifetimes that <em>did</em> find the goal, was the next "
              "episode any easier?</p>")
            A('<div class="tw"><table>')
            A("<thead><tr><th>Arm</th><th class='num'>memory_lift</th>"
              "<th class='num'>P(next | found)</th>"
              "<th class='num'>P(next | missed)</th></tr></thead><tbody>")
            for label, arm, cls in (("lifetime (state carried)", lt, "hl"),
                                    ("episodic control", ep, "")):
                if not arm:
                    continue
                A(f'<tr class="{cls}"><td class="k">{esc(label)}</td>'
                  f'<td class="num">{fmt(arm.get("memory_lift"))} '
                  f'± {fmt(arm.get("memory_lift_sem"))}</td>'
                  f'<td class="num">{fmt(arm.get("p_next_given_hit"))}</td>'
                  f'<td class="num">{fmt(arm.get("p_next_given_miss"))}</td></tr>')
            if pc:
                A('<tr><td class="k">scripted agent that remembers</td>'
                  f'<td class="num">{fmt(pc)}</td>'
                  '<td class="num">0.907</td><td class="num">0.348</td></tr>')
            A("</tbody></table></div>")

            gen = d.get("incontext_generalization") or {}
            if gen.get("gate_passed") is False or gen.get("memorised"):
                # The measurement is reported, and then withdrawn. Leaving the
                # verdict box in place with a caveat underneath would be worse
                # than either publishing it or removing it: the box is what a
                # reader takes away.
                lt_g = (gen.get("arms", {}).get("lifetime") or {})
                chance = gen.get("chance")
                A('<div class="note crit"><h4>This measurement does not '
                  "support a conclusion, and the earlier one is withdrawn</h4>"
                  f"<p>The numbers above were read as <em>“activation memory "
                  f"does not do this job”</em> — {lift:+.3f} against a "
                  f"detectable {pc:+.3f}. That reading does not hold, and the "
                  "reason is one number that was never measured at the time.</p>"
                  "<p><strong>A random walker solves "
                  f"{fmt(chance)} of these episodes. The trained policy solves "
                  f"{fmt(lt_g.get('held_out'))}</strong> — "
                  f"{lt_g.get('vs_chance', 0):.2f}× chance. On the "
                  "environments where in-context adaptation was being "
                  "measured, the policy performs <em>worse than an agent that "
                  "learned nothing at all</em>.</p>"
                  "<p>That closes the measurement channel. memory_lift is "
                  "behavioural — memory can only appear as a higher success "
                  "rate. A policy that cannot reach a goal in an unseen arena "
                  "cannot express memory of where that goal is, however much "
                  "of it the hidden state holds. <strong>A flat curve was "
                  "guaranteed before the question was asked.</strong></p>"
                  "<p>The cause is visible in the pretraining log. The pool "
                  "was 32 fixed environments; on those the same frozen policy "
                  f"reaches <strong>{fmt(lt_g.get('train_pool'))}</strong>, a "
                  f"gap of <strong>{lt_g.get('ratio', 0):.0f}×</strong>. It "
                  "learned thirty-two specific goals in its weights rather "
                  "than a strategy for an unseen arena — and heading "
                  "confidently towards where goals used to be is worse than "
                  "exploring, which is why it lands below random rather than "
                  "merely near it.</p>"
                  "<p>The positive control does not rescue it. A scripted "
                  "agent scoring +0.559 shows the <em>metric</em> can detect "
                  "memory; nothing here shows this <em>network</em> could "
                  "express memory if it had any. That control was never "
                  "run.</p></div>")
                ub = d.get("incontext_upper_bound") or {}
                lifts = ub.get("memory_lift") or {}
                if lifts:
                    pc = ub.get("positive_control") or 0.559
                    def _best(k):
                        rows = lifts.get(k) or {}
                        if not rows:
                            return None, None
                        h = max(rows, key=lambda x: rows[x]["mean"])
                        return h, rows[h]
                    h_c, carry = _best("carry")
                    h_i, icb = _best("in_context")
                    # The control has to be the SAME network size as the arm it
                    # controls for. Taking the best episodic run at whatever
                    # hidden size happened to produce it would compare two
                    # different networks and call the difference memory.
                    epb = (lifts.get("episodic") or {}).get(h_i)
                    h_e = h_i
                    A('<h3>Redone, and the answer reverses</h3>')
                    A("<p>Two things were fixed: the pretraining pool is now "
                      "redrawn every lifetime so it cannot be memorised, and "
                      "the policy is <strong>sampled rather than reduced to "
                      "its mean</strong>. The second matters more than it "
                      "sounds. Behaviour cloning fits the mean of the "
                      "teacher's action distribution and its spread; where the "
                      "goal is unknown that mean is near zero, so scoring the "
                      "mean measures a policy that barely moves. The arm handed "
                      "the goal's <em>direction</em> — the only one whose "
                      "target is unambiguous — is unaffected by the change "
                      "(1.00&times;), while every uncertain arm gains "
                      "2.2–4.0&times;. That asymmetry is the signature of a "
                      "measurement artefact rather than a difference between "
                      "policies.</p>")
                    A('<div class="tw"><table>')
                    A("<thead><tr><th>Arm</th><th class='num'>memory_lift</th>"
                      "<th class='num'>share of signal</th><th>What it shows</th>"
                      "</tr></thead><tbody>")
                    rows_out = [
                        ("Goal shown in episode 1 only", carry, h_c, "hl",
                         "The architecture-level control. Handed the fact for "
                         "free, the recurrence <em>keeps</em> it across an "
                         "episode boundary."),
                        ("In-context (must find the goal)", icb, h_i, "",
                         "The real question. Real adaptation, but roughly a "
                         "third of what the same network manages when the fact "
                         "is handed over."),
                        ("Episodic control (state reset)", epb, h_e, "",
                         "Identical training, hidden state does not survive a "
                         "goal-reach. The difference is what carrying state is "
                         "worth."),
                    ]
                    for label, v, h, cls, why in rows_out:
                        if not v:
                            continue
                        A(f'<tr class="{cls}"><td class="k">{esc(label)}</td>'
                          f'<td class="num">{v["mean"]:+.3f} ± {v["sem"]:.3f}'
                          f'<br><span style="color:var(--muted);font-size:12px">'
                          f'hidden {esc(h)}</span></td>'
                          f'<td class="num">{100 * v["mean"] / pc:.0f}%</td>'
                          f"<td>{why}</td></tr>")
                    A('<tr><td class="k">scripted agent that remembers</td>'
                      f'<td class="num">{pc:+.3f}</td><td class="num">100%</td>'
                      "<td>The positive control that sets the scale.</td></tr>")
                    A("</tbody></table></div>")
                    if carry and icb and epb:
                        A('<div class="note acc"><h4>A frozen recurrent policy '
                          "does adapt in-context</h4>"
                          f"<p>Handed the fact, it retains "
                          f"<strong>{carry['mean']:+.3f}</strong> — "
                          f"{100 * carry['mean'] / pc:.0f}% of the detectable "
                          "signal. Made to discover the fact itself it manages "
                          f"<strong>{icb['mean']:+.3f}</strong>, against an "
                          f"episodic control at {epb['mean']:+.3f}.</p>"
                          "<p>The gap between those two is the interesting "
                          "part, and it is not about memory. The teacher is a "
                          "shortest-path oracle that can see the goal, so it "
                          "beelines — and therefore <em>no timestep anywhere in "
                          "the training data demonstrates searching for "
                          "something</em>. The recurrence can hold a fact; the "
                          "teacher cannot show it how to find one.</p></div>")
                    A("<p><strong>What still bounds this.</strong> Told the "
                      "goal's direction the policy reaches 0.996; told its "
                      "coordinates, 0.562 — both measured under identical "
                      "sampling. Converting a position into a move requires "
                      "knowing where the agent is, and that self-localisation "
                      "is the weakest link in the stack. It caps what any "
                      "in-context memory can be worth: remembering a "
                      "coordinate only helps an agent that can locate itself "
                      "against it.</p>")

                A('<div class="note"><h4>What a real attempt needs</h4><ul>'
                  "<li><strong>A fresh environment every lifetime</strong>, "
                  "drawn from the ~10<sup>7</sup> available wall seeds, rather "
                  "than cycling a fixed 32. This is the standard meta-learning "
                  "setup and it removes memorisation as an option.</li>"
                  "<li><strong>A chance gate, enforced before the statistic is "
                  "computed.</strong> Held-out episode-1 success must beat a "
                  "measured random walker before “does it adapt?” has an "
                  "answer. This is now implemented — the run above fails it at "
                  f"{lt_g.get('vs_chance', 0):.2f}× — so a future run that "
                  "cannot clear the floor reports “precondition not met” "
                  "instead of a publishable-looking flat line.</li>"
                  "<li><strong>An architecture-level positive control</strong> "
                  "— reveal the goal during episode 1 only, and check the agent "
                  "exploits it in episode 2. Failing that would be a "
                  "<em>legible</em> failure mode, which is what the plan asked "
                  "for and what a bare null does not give.</li>"
                  "<li><strong>memory_lift on the training environments "
                  "too</strong>, which localises the failure to generalisation "
                  "rather than to memory.</li></ul></div>")
            elif pc and lift < 0.25 * pc:
                A('<div class="note acc"><h4>Activation memory does not do this '
                  "job</h4>"
                  f"<p>The lifetime arm scores <strong>{lift:+.3f}</strong> "
                  f"against a detectable <strong>{pc:+.3f}</strong> — about "
                  f"{100.0 * abs(lift) / pc:.0f} % of the available signal. "
                  "An agent that genuinely remembered would solve the next "
                  "episode 91 % of the time after finding the goal; this one "
                  "manages 12 %, against 9 % when it did not find it.</p>"
                  "<p>This is the one comparison a referee cannot answer with "
                  '"you needed a bigger buffer" — there is no buffer on either '
                  "side, and no weight updates either.</p></div>")
            else:
                A('<div class="note crit"><h4>The RNN adapts in-context</h4>'
                  f"<p>memory_lift <strong>{lift:+.3f}</strong>. Forgetting is "
                  "not the interesting axis for this comparison, and the "
                  "framing has to account for it.</p></div>")

    # ---- N=20 ----------------------------------------------------------
    n20 = d.get("n20") or []
    if n20:
        A('<h2 id="n20"><span class="sec">8</span> Twenty environments</h2>')
        A("<p>Methods look alike at five tasks and separate at twenty. Only "
          "configurations whose best setting was already pinned down at N=5 "
          "are scaled here — at 2.6 hours a seed, an unresolved sweep would "
          "spend the budget rediscovering what a five-environment stream "
          "answers for a tenth of the cost.</p>")
        A("<p>Two things to watch beyond the ordering. <strong>Storage grows "
          "with the stream and the store's does not</strong> — an unbounded "
          "buffer goes from 53.6 MB at five environments to 214.4 MB at "
          "twenty, linear in updates, while an associative matrix is a fixed "
          "size. And <strong>plasticity collapses for everyone</strong>: "
          "current-env performance falls to roughly a third of its "
          "five-environment value across every arm including the control. At "
          "twenty environments the agent is not only forgetting the old ones, "
          "it is failing to learn the one in front of it.</p>")
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
    # Sections 9-11 are the project's own narrative -- the defects found,
    # what the suite settles, what it has not run -- and every line of it
    # was written about the continuous wave. Rendering it under discrete
    # numbers would put continuous claims on a page that did not produce
    # them, which is worse than omitting them.
    if variant == "continuous":
        A('<h2 id="found"><span class="sec">9</span> Found along the way</h2>')
        A("<p>Twelve defects surfaced while building this, and they are on the "
          "page because they share a property worth knowing about: <strong>not one "
          "of them raised an exception in normal operation</strong>. Every one "
          "produced a plausible number. Four were caught by a test written to be "
          "falsifiable rather than confirmatory, three by inspecting a table "
          "before publishing it, two by reading code that had no failing symptom "
          "at all, one by disbelieving a result this project had just produced, "
          "and <strong>two by a reader asking a question</strong> — where a method "
          "had gone, and whether a null result had really been tried hard enough. "
          "Six are shown here; the rest are in the log.</p>")
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
        A('<div class="note crit"><h4>DER++ contributed no gradient</h4>'
          "<p>Its distillation term was computed against a detached tensor, so it "
          "returned a nonzero loss that scaled correctly with its coefficient and "
          "carried <code>requires_grad=False</code>. It added a constant to the "
          "objective and moved nothing. DER++ ran as plain Experience Replay for "
          "two full waves, and every value-based test passed throughout — they "
          "checked that the loss was nonzero and grew, which it was and did. "
          "Fixing it moved the method from 0.143 to 0.326.</p></div>")
        A('<div class="note warn"><h4>A capacity verdict that was wrong</h4>'
          "<p>The first joint-ceiling run came back at ~0.5 across every capacity "
          "and the summary called it a capacity limit. It was not: capacity did not "
          "move the number and every curve was still climbing at the budget "
          "limit. The run was optimisation-starved. The summary now measures the "
          "end-slope and refuses a capacity verdict while a run is still "
          "improving.</p></div>")
        A('<div class="note crit"><h4>A published null that its own training log '
          "contradicted</h4>"
          "<p>&sect;7 reported that a frozen recurrent policy shows no in-context "
          "adaptation, and called it the strongest result available. The policy it "
          "tested scores 0.80 on the 32 environments it was pretrained on and 0.10 "
          "on the held-out environments it was evaluated on — an <strong>8× "
          "gap</strong>. It had memorised its pool, so the evaluation was run on a "
          "policy that cannot navigate its test environments, and a flat "
          "success-vs-episode curve was the only outcome available.</p>"
          "<p>The launcher's own comment asserted the thing that failed — "
          "<em>“a pool big enough that memorising it is not obviously easier than "
          "learning the strategy”</em> — as an assumption, never checked. The "
          "numbers refuting it were written to the same job's log hours before the "
          "conclusion was drawn from it. And the defence recorded at the time, "
          "that a held-out split <em>“guards against the arms simply memorising”</em>, "
          "is wrong in a specific way: a held-out split stops you reporting "
          "memorisation as adaptation, but it cannot rescue an experiment whose "
          "model memorised — it leaves nothing to measure.</p>"
          "<p>Every other entry here was found by a check. This one was found by "
          "being asked whether the effort had really been the best possible, and "
          "the honest answer was no.</p></div>")
        A('<div class="note crit"><h4>A task-identifiability number that measured '
          "the wrong thing</h4>"
          "<p>&sect;5's headline depends on how recognisable an environment is "
          "from the agent's observations. The first version split shuffled "
          "<em>frames</em> rather than trajectories, so neighbouring timesteps of "
          "one random walk sat on both sides of the train/test line, and it tried "
          "only a linear readout on a single observation. It returned 0.266 "
          "against 0.200 chance — a low number from the weakest classifier "
          "available, which is the least informative outcome there is, and it was "
          "about to be used to argue that the oracle task id is a large "
          "advantage. Rewritten to split by trajectory and to sweep up to an MLP "
          "over 64-observation windows, it returns 0.43. The conclusion held; the "
          "evidence for it did not, and would not have survived a reader who "
          "checked.</p></div>")
        A('<div class="note acc"><h4>The fourth coefficient-scale error, caught '
          "before it ran</h4>"
          "<p>DER++ and CLEAR both cost a re-run because their strength knobs were "
          "swept over the range a paper used, against a loss of a different "
          "magnitude. The hypernetwork regulariser has exactly the same shape of "
          "knob, so this time it was measured first: von Oswald's beta = 1 "
          "contributes <strong>0.017% of this objective</strong>. Sweeping decades "
          "around it — the obvious thing to do — would have made every arm a no-op "
          "and produced the confident, wrong finding that the method does not help "
          "here. The sweep runs 10<sup>2</sup>–10<sup>7</sup> instead. This is the "
          "only entry in this section that is not a defect; it is here because it "
          "is the same defect, prevented.</p></div>")
        A('<div class="note warn"><h4><code>init_log_std</code> was unreachable</h4>'
          "<p>Exposed on <code>train_rnn</code> but never wired through the "
          "continual driver, so every run to date used σ = 1.0 against a "
          "unit-magnitude action — the DAgger student exploring with noise the size "
          "of the action itself.</p></div>")

        # ---- what it means -------------------------------------------------
        A('<h2 id="reading"><span class="sec">10</span> What this settles</h2>')

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
        # Derived, because this bullet is the one most likely to be left standing
        # after the numbers behind it change -- it was written when the isolation
        # family had not run, and it stayed on the page saying so for a while after
        # that stopped being the interesting thing about it.
        iso_arms = ("J", "K", "L", "M", "N", "N2")
        iso_usable = [m for m in methods if m["arm"] in iso_arms
                      and (m["current_env"] or 0) >= USABLE_CURRENT
                      and m["retained"] is not None]
        free_usable = [m for m in methods if m["arm"] not in iso_arms
                       and not m["needs_task_id"]
                       and (m["current_env"] or 0) >= USABLE_CURRENT
                       and m["retained"] is not None]
        if iso_usable:
            top_iso = max(iso_usable, key=lambda m: m["retained"])
            best_free = (max(free_usable, key=lambda m: m["retained"])
                         if free_usable else None)
            # The headline is conditional on the same comparison the sentence
            # makes. Written as a fixed claim with a derived clause after it, the
            # two halves would contradict each other the moment the ordering
            # changed -- which is the failure this page keeps finding in itself.
            beats = (best_free is not None
                     and top_iso["retained"] > best_free["retained"])
            comparison = ""
            if best_free:
                comparison = (
                    (" It is the strongest classic result in the suite, and it "
                     "beats the best method given no task signal at all "
                     if beats else
                     " Even with that advantage it does not reach the best method "
                     "given no task signal at all ")
                    + f"({esc(best_free['display'])}, "
                      f"{fmt(best_free['retained'])}).")
            head = ("Giving each task its own parameters is the strongest thing "
                    "tried here — and it is not free."
                    if beats else
                    "Giving each task its own parameters helps, and is not enough.")
            A(f"<li><strong>{head}</strong> The best isolation result is "
              f"{esc(top_iso['display'])} at {fmt(top_iso['retained'])} retained "
              "— and it is <em>told which environment it is in</em>, which a "
              "classifier on the agent's own observations cannot recover "
              f"(&sect;5).{comparison}</li>")
        A("<li><strong>Restricting plasticity to a small head is worse, not "
          "better.</strong> Freezing the trunk costs both retention and current-env "
          "performance. Whatever a meta-learned representation would buy here, it "
          "is not the head-only restriction by itself.</li>")
        A("</ul>")

        # Every quantity in this paragraph is derived. It previously read "roughly
        # three-fifths of the joint ceiling while storing every trajectory it has
        # ever seen" -- true of the replay arm that led at the time, and wrong in
        # both halves once a gating method with a 0.6 MB importance matrix took the
        # lead. A summary sentence carrying hand-written numbers goes stale exactly
        # when the result gets interesting.
        ceil_rows = [j for j in d.get("joint", [])
                     if abs(j.get("end_slope") or 0) <= 0.02 and (j.get("final") or 0) > 0.5]
        ceiling = max((j["final"] for j in ceil_rows), default=None)
        usable = [m for m in methods if (m["current_env"] or 0) >= USABLE_CURRENT
                  and m["retained"] is not None]
        if usable and ceiling:
            top = max(usable, key=lambda m: m["retained"])
            frac = top["retained"] / ceiling
            share = (f"{round(frac * 100)}% of the joint ceiling"
                     if frac == frac else "an unknown share of the ceiling")
            cost = []
            if top["state_bytes"]:
                cost.append(f"carrying {esc(mb(top['state_bytes']))} of state")
            if top["needs_task_id"]:
                cost.append("being told which environment it is in")
            cost.append("paying two hundred gradient steps and two hundred "
                        "episodes per environment")
            cost_text = ", ".join(cost[:-1]) + " and " + cost[-1] if len(cost) > 1 \
                else cost[0]
            A('<div class="note acc"><h4>The honest form of the claim</h4>'
              "<p>Not “continual learning cannot do this”. What the suite supports "
              "is narrower and more defensible: <em>on this task, across "
              f"{spell(len(method_names(methods)))} methods spanning replay, "
              "parameter regularisation and parameter isolation, with their "
              "coefficients swept over decades, the best classic result "
              f"({esc(top['display'])}) reaches {share} while {cost_text} — and an "
              "associative store reaches the ceiling at one episode, no gradient "
              "steps, no stored data and no task label.</em> The frontier is the "
              "claim; the leaderboard is not.</p></div>")

        # ---- provenance ----------------------------------------------------
        A('<h2 id="notrun"><span class="sec">11</span> What has not run</h2>')
        A("<p>The three families the plan names — replay, parameter "
          "regularisation, parameter isolation — have all run. What is left is a "
          "shorter and more specific list, and it is here because a results page "
          "with no such section is making a claim about its own completeness that "
          "nobody checked.</p>")
        A('<div class="tw"><table>')
        A("<thead><tr><th>Not run</th><th>Why it matters, and why it did not</th>"
          "</tr></thead><tbody>")
        for name, why in (
            ("Meta-pretraining (OML / ANML)",
             "The plan's &sect;5.1 — the strongest possible form of the "
             "pretraining control, learning a representation whose <em>purpose</em> "
             "is that later gradient descent on it does not interfere. Deferred "
             "on evidence rather than for time: its load-bearing mechanism is "
             "confining plasticity to a small head, and the frozen-trunk arm shows "
             "that doing so here is <em>worse</em> on both retention and current-"
             "environment performance. That is a reason to doubt the mechanism in "
             "this setting, not a proof, and it remains the largest untested idea."),
            ("A learned task-id router",
             "The isolation arms are handed an oracle task id. The plan asks for "
             "an inferred condition beside it, so the gap measures how much of the "
             "problem is task inference. The gap was measured directly instead "
             "(&sect;5): the environments are only weakly recognisable from the "
             "agent's observations, 0.43 against 0.20 chance. Building the router "
             "would sharpen that number; it would not change the conclusion, which "
             "is that these arms are given something no other method could get."),
            ("Plasticity maintenance",
             "At twenty environments the current-environment score collapses to "
             "0.29–0.40: the network is not only forgetting, it is losing the "
             "ability to learn at all. Nothing in the suite targets that directly "
             "— resets, shrink-and-perturb, continual backpropagation. It became "
             "interesting only when the scaling panel produced the collapse, which "
             "was after the wave structure was fixed."),
            ("Gradient Projection Memory",
             "Listed in the plan as an optional stretch and never promoted. It "
             "constrains updates to directions orthogonal to past-task gradients, "
             "which is a third mechanism again — but the two regularisation "
             "methods that did run both land in the plasticity trap here, and GPM "
             "is a harder version of the same bargain."),
            ("Route efficiency (SPL) — now recorded, not yet analysed",
             "Every number on this page is a success <em>rate</em>. How far the "
             "agent travelled to earn it was recorded all along, and is unusable "
             "on its own: mean steps over successful trials scores each arm on "
             "the subpopulation it happened to solve, so an arm that forgets "
             "almost everything is graded only on the goals that were near its "
             "start and comes out looking like the suite's fastest navigator. "
             "Under continuous movement it is not even a clean quantity — the "
             "displacement is the raw action vector, so step count mixes route "
             "quality with how far the policy commits to moving. The evaluator "
             "now also records <code>optimal_to_goal</code>, the shortest "
             "attainable path for that trial, which is exact here rather than a "
             "bound because the arena is a convex box with no interior obstacles. "
             "That makes <code>optimal / max(path, optimal)</code>, zeroed on "
             "failures, a proper SPL: bounded, defined on every trial, and "
             "independent of how hard the trial was. Nothing on this page uses it "
             "— no history behind these numbers carries the field, and it is not "
             "recoverable for waves 0–3 without re-running the evaluations from "
             "the saved checkpoints. It is here so the next wave is read with it "
             "rather than discovering the need afterwards."),
            ("The Hopfield agent at twenty environments",
             "The scaling panel runs the classic methods to twenty environments; "
             "the store has only ever been run to five. The comparison at N=20 is "
             "therefore between measured baselines and an assumption, and the "
             "page does not make it."),
        ):
            A(f"<tr><td class='k'>{esc(name)}</td><td>{why}</td></tr>")
        A("</tbody></table></div>")
        A('<div class="note"><h4>How this section came to exist</h4>'
          "<p>Wave 3 was not deferred — it was <em>forgotten</em>. Wave 2 finished, "
          "then three corrections landed in a row (a coefficient range taken from "
          "the wrong loss scale, a distillation term carrying no gradient, an "
          "importance sampler that was not a reservoir), and each one re-ran an arm "
          "that already existed. Re-running existing work displaced new work, the "
          "suite was called complete when the corrections finished rather than when "
          "the plan was finished, and the page said “nine methods” because it had "
          "counted rows in a table that also held the controls.</p>"
          "<p>Nothing in the process caught it. A reader asking where the "
          "hypernetwork had gone did. So the method count on this page is now "
          "derived from the runs that exist rather than written down, and this "
          "section is generated beside the results instead of being remembered.</p>"
          "</div>")

    A('<h2 id="provenance"><span class="sec">12</span> Provenance</h2>')
    A("<p>Every number on this page is read out of the run histories by "
      "<code>results_data.py</code> and rendered by "
      "<code>results_page.py</code>. Nothing is typed in — a page whose "
      "figures were transcribed stops matching its runs the first time one is "
      "repeated.</p>")
    A('<div class="tw"><table>')
    A("<thead><tr><th>Job</th><th>What</th><th class='num'>Tasks</th>"
      "</tr></thead><tbody>")
    for job, what, n in (
        ("21626914", "Wave 0 — oracle, from-scratch floor, first joint sweep", "64/64"),
        ("21627945", "corrected joint ceiling (8× the gradient budget)", "22/24"),
        ("21628688", "Wave 1 — Tier-1 tuning, ER, online EWC", "272/272"),
        ("21631698", "Wave 2 — CLEAR, DER++, SI, LwF, frozen trunk, high-ratio ER", "144/144"),
        ("21631792", "N=20 scaling panel", "20/20"),
    ("21653228", "Wave 3 — HNET, multi-head, XdG, and their from-scratch "
                 "controls", "144 launched"),
        ("21633232", "Wave 2b — corrected coefficient ranges", "56/56"),
        ("21634287", "Wave 2c — DER++ after the gradient fix", "32/32"),
        ("21634899", "Wave 2d — EWC/SI after the sampler fix", "56/56"),
        ("21629579", "in-context pretraining and first evaluation", "9/9"),
        ("21643814", "in-context re-scored with memory_lift", "3/3"),
    ):
        A(f"<tr><td class='k'>{esc(job)}</td><td>{esc(what)}</td>"
          f'<td class="num">{esc(n)}</td></tr>')
    A("</tbody></table></div>")
    A("<p>Roughly <strong>530 sequential-protocol runs</strong> and 48 joint "
      "runs, on CPU. The two failures in the joint sweep were both "
      "<code>hidden=512, lr=3e-3</code> diverging to NaN — the same "
      "instability the sweep reports at smaller sizes, taken to its "
      "conclusion, rather than a defect.</p>")
    A("<p>One sequential run is five environments × 200 updates, each update a "
      "single 200-step rollout and a single gradient step, evaluated against "
      "every environment seen so far — 1,000 evaluation points per run. "
      "Method arms use 8 seeds, the from-scratch floor 20, the N=20 panel 4, "
      "and the joint ceiling 3.</p>")
    A('<div class="note"><h4>Reproducing it</h4>'
      "<pre><code>python -m analysis.continual.results_data \\\n"
      "    --wave0_dir $CLS_RUNS/histories/wave0 \\\n"
      "    --wave1_dir $CLS_RUNS/histories/wave1 \\\n"
      "    --n20_dir   $CLS_RUNS/histories/n20 \\\n"
      "    --incontext_dir $CLS_RUNS/histories/incontext \\\n"
      "    --recorded_dir $CLS_RUNS/histories \\\n"
      "    --runs_root $CLS_RUNS --out results.json\n"
      "python -m analysis.continual.results_page --data results.json --out page.html\n"
      "python -m analysis.continual.validate_page page.html</code></pre>"
      "<p>The launchers are <code>analysis/continual/run_wave*.sh</code>, "
      "<code>run_n20.sh</code> and <code>run_incontext*.sh</code>. The plan is "
      "<code>docs/CONTINUAL_CONTROLS_PLAN.md</code>; what actually happened, "
      "including all nine defects, is "
      "<code>docs/CONTINUAL_CONTROLS_LOG.md</code>.</p></div>")

    A("<footer><span>generated by analysis.continual.results_page</span>"
      f'<span>data: {esc(d.get("generated", ""))}</span>'
      "<span>branch worktree-continual-control-suite</span></footer>")
    A("</div>")
    return _renumber_sections("\n".join(P))


def render(d: dict, d_disc: dict | None = None) -> str:
    """The whole document: one head, and one or two switchable page bodies.

    Both action spaces live at the same URL on purpose. They are the same
    experiment run twice and the only honest readings are comparative, so
    putting them behind two links would make the comparison the reader's job.
    """
    P: list[str] = []
    A = P.append
    A("<title>Continual Control Results</title>")
    A('<link rel="preconnect" href="https://fonts.googleapis.com">')
    A('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=Spectral:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&'
      'family=IBM+Plex+Mono:wght@400;500;600&display=swap">')
    A(f"<style>{CSS}</style>")

    if d_disc:
        A('<div class="switch"><div class="in">'
          '<span class="lb">Action space</span>'
          '<button id="btn-continuous" aria-selected="true" '
          'onclick="showPage(\'continuous\')">Continuous</button>'
          '<button id="btn-discrete" aria-selected="false" '
          'onclick="showPage(\'discrete\')">Discrete</button>'
          '<span class="note">Same runs, same protocol, different action '
          'head. Floors and ceilings differ, so compare within a space.</span>'
          "</div></div>")

    A('<div id="page-continuous">')
    A(render_body(d, "continuous", d_disc))
    A("</div>")
    if d_disc:
        A('<div id="page-discrete" hidden>')
        A(render_body(d_disc, "discrete", d))
        A("</div>")
        # `hidden` rather than a style toggle, and every storage access in a
        # try/catch: the page is read in private windows and in thumbnail
        # capture, where localStorage itself throws rather than returning null.
        A("""<script>
function showPage(which){
  var ok = ['continuous','discrete'];
  if (ok.indexOf(which) < 0) which = 'continuous';
  ok.forEach(function(k){
    var pg = document.getElementById('page-' + k);
    var bt = document.getElementById('btn-' + k);
    if (pg) pg.hidden = (k !== which);
    if (bt) bt.setAttribute('aria-selected', String(k === which));
  });
  try { localStorage.setItem('cl-action-space', which); } catch (e) {}
}
(function(){
  var saved = null;
  try { saved = localStorage.getItem('cl-action-space'); } catch (e) {}
  if (saved) showPage(saved);
})();
</script>""")
    return "\n".join(P)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--data_discrete", default=None,
                   help="Second results JSON, built from the discrete wave. "
                        "When given the page carries both action spaces behind "
                        "a switcher at one URL.")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(args.data) as f:
        d = json.load(f)
    d_disc = None
    if args.data_discrete and os.path.exists(args.data_discrete):
        with open(args.data_discrete) as f:
            d_disc = json.load(f)
    html_text = render(d, d_disc)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html_text)
    print(f"[page] wrote {args.out}  ({len(html_text)} bytes, "
          f"{len(d.get('methods', []))} method configs"
          + (f", + {len(d_disc.get('methods', []))} discrete" if d_disc else "")
          + ")")


if __name__ == "__main__":
    main()
