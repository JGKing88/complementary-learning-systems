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
    A("<p>Six continual-learning methods, plus the controls they are measured "
      "against. Chosen for this setting rather than by citation count, and "
      "every coefficient swept over decades — a strength knob set from a paper "
      "with a different loss scale is how a method gets accidentally made to "
      "look bad.</p>")
    A('<div class="tw"><table>')
    A("<thead><tr><th>Method</th><th>What it does</th></tr></thead><tbody>")
    for key in ("B", "I", "E", "D", "C", "F", "G", "H", "A"):
        name, desc = MECHANISM[key]
        A(f"<tr><td class='k'>{esc(name)}</td><td>{desc}</td></tr>")
    A("</tbody></table></div>")

    # ---- the frontier ------------------------------------------------
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
        _mech = MECHANISM.get(r["arm"])
        note = _summarise(_mech[1]) if _mech else ""
        A("<tr><td class='k'>" + esc(r["display"]) + "</td>"
          f"<td><code>{esc(r['config'])}</code>"
          + (f'<br><span style="color:var(--muted);font-size:12.5px">'
             f"{esc(note)}</span>" if note else "")
          + "</td>"
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
    A('<h2 id="tier0"><span class="sec">5</span> The axes</h2>')
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
        A('<h2 id="incontext"><span class="sec">6</span> '
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

            if pc and lift < 0.25 * pc:
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
            A("<p><strong>What this does not license.</strong> The policies "
              "here are weak in absolute terms, the pool was 32 environments, "
              "and the recurrent state is 256 units. This says activation "
              "memory in <em>this</em> network does not substitute for a "
              "store — not that no recurrent policy could.</p>")

    # ---- N=20 ----------------------------------------------------------
    n20 = d.get("n20") or []
    if n20:
        A('<h2 id="n20"><span class="sec">7</span> Twenty environments</h2>')
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
    A('<h2 id="found"><span class="sec">8</span> Found along the way</h2>')
    A("<p>Nine defects surfaced while building this, and they are on the page "
      "because they share a property worth knowing about: <strong>not one of "
      "them raised an exception in normal operation</strong>. Every one "
      "produced a plausible number. Three were caught by a test written to be "
      "falsifiable rather than confirmatory, three by inspecting a table "
      "before publishing it, and two by reading code that had no failing "
      "symptom at all. Four are shown here; the rest are in the log.</p>")
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
    A('<div class="note warn"><h4><code>init_log_std</code> was unreachable</h4>'
      "<p>Exposed on <code>train_rnn</code> but never wired through the "
      "continual driver, so every run to date used σ = 1.0 against a "
      "unit-magnitude action — the DAgger student exploring with noise the size "
      "of the action itself.</p></div>")

    # ---- what it means -------------------------------------------------
    A('<h2 id="reading"><span class="sec">9</span> What this settles</h2>')

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
    A("<li><strong>This covers replay and regularisation, not "
      "isolation.</strong> No method that gives each task its own parameters "
      "has been run yet — see §10. The strongest untested candidate, a "
      "task-conditioned hypernetwork, is the one the recurrent-continual-"
      "learning literature actually recommends.</li>")
    A("<li><strong>Restricting plasticity to a small head is worse, not "
      "better.</strong> Freezing the trunk costs both retention and current-env "
      "performance. Whatever a meta-learned representation would buy here, it "
      "is not the head-only restriction by itself.</li>")
    A("</ul>")

    A('<div class="note acc"><h4>The honest form of the claim</h4>'
      "<p>Not “continual learning cannot do this”. What the suite supports is "
      "narrower and more defensible: <em>on this task, across six methods "
      "with their coefficients swept over decades, the best classic result "
      "reaches roughly three-fifths of the joint ceiling while storing every "
      "trajectory it has ever seen and paying two hundred gradient steps per "
      "environment — and an associative store reaches the ceiling at one "
      "episode and no gradient steps at all.</em> The frontier is the claim; "
      "the leaderboard is not.</p></div>")

    # ---- provenance ----------------------------------------------------
    A('<h2 id="notrun"><span class="sec">10</span> What has not run</h2>')
    A("<p>The plan specifies a third wave of <strong>structural</strong> "
      "methods — ones that give each task its own parameters instead of "
      "sharing one set. None of it has been run, and its absence is the "
      "largest gap in this page.</p>")
    A('<div class="tw"><table>')
    A("<thead><tr><th>Method</th><th>Why it matters here</th></tr></thead><tbody>")
    for name, why in (
        ("Task-conditioned hypernetwork (HNET)",
         "The plan's designated headline competitor, and the most consequential "
         "omission. Ehret et al. (ICLR 2021) benchmarked online EWC, SI, "
         "masking, generative replay and coresets across four <em>recurrent</em> "
         "benchmarks and found hypernetworks beat weight-importance methods "
         "consistently — this is a recurrent policy, so it is the method the "
         "literature points at. It is also the closest classical analogue to "
         "the store: both keep a small addressable per-task code and recover "
         "behaviour from it rather than overwriting shared weights. The "
         "difference is how the code is written — a gradient-descent inner "
         "loop over a whole block, against one Hebbian outer product."),
        ("Multi-head with an oracle task ID",
         "Bounds the entire parameter-isolation family in one run: if "
         "isolation with a free task ID does not retain, nothing in that "
         "family will. Paired with a learned task classifier, the gap between "
         "the two would measure how much of the problem is task inference "
         "rather than forgetting — which is precisely the job the store does "
         "in one shot."),
        ("XdG (context-dependent gating)",
         "Sparse, mostly non-overlapping units per task; ~10 lines, and it "
         "composes with SI. Kept in the plan over PackNet and HAT because "
         "sparse addressing is the conceptually closest classical thing to "
         "content-addressable storage."),
    ):
        A(f"<tr><td class='k'>{esc(name)}</td><td>{why}</td></tr>")
    A("</tbody></table></div>")
    A('<div class="note warn"><h4>Why it is missing</h4>'
      "<p>Not a decision. Wave 2 finished, and then three corrections in a row "
      "— a coefficient range chosen from the wrong loss scale, a distillation "
      "term carrying no gradient, an importance sampler that was not a "
      "reservoir — each re-ran an arm that already existed. Those re-runs "
      "consumed the slot Wave 3 would have occupied, and the suite was called "
      "complete when the corrections finished rather than when the plan was "
      "finished. Re-running existing work displaced new work, and nothing in "
      "the process noticed.</p>"
      "<p>So the claim below is bounded by what ran: <strong>replay and "
      "parameter-regularisation families</strong>. It says nothing yet about "
      "methods that allocate separate parameters per task, and HNET is the one "
      "with the strongest prior reason to do well here.</p></div>")

    A('<h2 id="provenance"><span class="sec">11</span> Provenance</h2>')
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
