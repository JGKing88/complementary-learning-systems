"""Example trajectories from a trained navigator, as a standalone page.

`exploit_report` exists to explain FAILURES and organises everything by motion
class. A model at success 1.000 has no failures, so this renders the other
case: what the successful paths actually look like, grouped by distractor count
and ordered by start distance -- which is the axis the episodes were selected
along, so the ordering carries information rather than decorating.

Styling, tokens and the arena chart are imported from `exploit_report` so this
page sits in the same visual system as every other diagnostic in the phase.

Usage:
    python -m analysis.nav_tri.traj_page traj.json -o traj.html
"""
from __future__ import annotations

import argparse
import html
import json

import numpy as np

from analysis.nav_tri.exploit_report import CSS, FONTS, traj_chart

# The failure reports colour paths by motion class (c1/c2/c3). Everything here
# is a success, so one colour is right, and the page's own accent is the one
# that says so. The goal ring is --c4 at 16% fill, faint enough that a 1.4px
# accent line beside it reads as a separate mark rather than the same hue twice.
PATH = "var(--accent)"

EXTRA = """
.wrap{max-width:1120px;margin:0 auto;padding:40px 24px 72px}
.head{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:8px}
.head h1{font-family:"IBM Plex Serif",Georgia,serif;font-size:2.1rem;
  line-height:1.15;margin:0 0 8px;text-wrap:balance;letter-spacing:-.01em}
.head .sub{font-family:"IBM Plex Serif",Georgia,serif;color:var(--muted);
  font-size:1.02rem;margin:0;max-width:64ch;line-height:1.55}
.strip{display:flex;flex-wrap:wrap;gap:0;margin:22px 0 30px;
  border:1px solid var(--line);border-radius:3px;overflow:hidden;
  background:var(--panel)}
.strip div{flex:1 1 150px;padding:13px 16px;border-right:1px solid var(--line)}
.strip div:last-child{border-right:0}
.strip dt{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.63rem;
  letter-spacing:.11em;text-transform:uppercase;color:var(--muted);margin:0 0 5px}
.strip dd{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:1.18rem;
  margin:0;color:var(--ink);font-variant-numeric:tabular-nums;font-weight:500}
.strip dd small{font-size:.72rem;color:var(--muted);font-weight:400}
.note{font-family:"IBM Plex Serif",Georgia,serif;font-size:.92rem;
  line-height:1.6;color:var(--muted);max-width:70ch;margin:0 0 34px;
  padding-left:14px;border-left:2px solid var(--line)}
.band{margin:0 0 40px}
.band h2{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.72rem;
  letter-spacing:.13em;text-transform:uppercase;color:var(--ink);
  margin:0 0 3px;font-weight:600}
.band .cap{font-family:"IBM Plex Serif",Georgia,serif;font-size:.9rem;
  color:var(--muted);margin:0 0 16px}
.grid{display:grid;gap:14px;
  grid-template-columns:repeat(auto-fill,minmax(190px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:3px;
  padding:10px 10px 8px;margin:0;display:flex;flex-direction:column;gap:6px}
.card svg{display:block;width:100%;height:auto}
.card .rows{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.7rem;
  font-variant-numeric:tabular-nums;color:var(--muted);
  display:grid;grid-template-columns:auto 1fr;gap:1px 10px}
.card .rows b{color:var(--ink);font-weight:500;text-align:right}
.card .tag{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.6rem;
  letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
  border-top:1px solid var(--line);padding-top:6px;margin-top:2px}
.legend{display:flex;flex-wrap:wrap;gap:20px;align-items:center;
  font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.74rem;
  color:var(--muted);margin:0 0 22px}
.legend span{display:flex;align-items:center;gap:7px}
.legend i{width:11px;height:11px;border-radius:50%;display:inline-block}
.foot{border-top:1px solid var(--line);padding-top:16px;margin-top:8px;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.68rem;
  color:var(--muted);line-height:1.8;word-break:break-word}
"""


def esc(s):
    return html.escape(str(s))


def card(r, radius, size):
    """One arena plus the numbers that describe the path in it."""
    traj = np.asarray(r["traj"], dtype=float)
    path = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=-1)))
    d0 = float(r["d_start"])
    # Ratio against THIS episode's own straight line, not the 10.5-cell pool
    # mean -- a per-episode plot deserves a per-episode denominator. Values
    # just under 1.0 are correct: capture is a ball of `radius`, so the agent
    # stops short of the goal centre.
    ratio = path / max(d0, 1e-9)
    st = r.get("straightness")
    return (
        '<figure class="card">'
        + traj_chart(r["traj"], (r["goal_x"], r["goal_y"]), radius, size,
                     color=PATH, w=190)
        + '<div class="rows">'
        f'<b>{d0:.1f}</b><span>cells to goal</span>'
        f'<b>{int(r["steps"])}</b><span>steps</span>'
        f'<b>{path:.1f}</b><span>cells walked</span>'
        f'<b>{ratio:.2f}&times;</b><span>of straight line</span>'
        '</div>'
        f'<div class="tag">env {esc(r["env_idx"])} &middot; trial '
        f'{esc(r["trial"])} &middot; straightness '
        + (f'{st:+.2f}' if st is not None and st == st else '&mdash;')
        + '</div></figure>')


def build(data, ckpt):
    g0 = {g["n_dist"]: g for g in data["groups"]}
    first = data["groups"][0]
    radius = first["stats"]["radius"]
    size = first["stats"]["size"]

    def band(n_dist, title, cap):
        g = g0.get(n_dist)
        if not g:
            return ""
        shown = sorted((r for r in g["rows"] if r.get("traj")),
                       key=lambda r: r["d_start"])
        if not shown:
            return ""
        return (f'<section class="band"><h2>{esc(title)}</h2>'
                f'<p class="cap">{cap}</p><div class="grid">'
                + "".join(card(r, radius, size) for r in shown)
                + '</div></section>')

    # Headline numbers come from every episode run, not the plotted subset.
    def rate(n):
        g = g0.get(n)
        if not g:
            return "&mdash;"
        rows = g["rows"]
        ok = sum(r["success"] for r in rows)
        return (f'{ok}/{len(rows)}'
                f'<small> &middot; {ok / len(rows):.3f}</small>')

    allrows = [r for g in data["groups"] for r in g["rows"] if r["success"]]
    med_steps = np.median([r["steps"] for r in allrows])
    med_spd = np.median([r["speed"] for r in allrows if r.get("speed")])

    out = [
        "<title>p19_kcap Beelines</title>",
        FONTS,
        f"<style>{CSS}{EXTRA}</style>",
        '<div class="wrap">',
        '<header class="head"><h1>What the delivered navigator '
        'actually does</h1>',
        '<p class="sub">Sixteen example paths from <code>p19_kcap</code> at '
        'update 800 &mdash; the arm that reached a beeline at u150. Each box '
        'is one 20&times;20 arena: hollow marker is where the agent started, '
        'filled marker is where it stopped, the ring is the capture ball it '
        'had to reach.</p></header>',
        '<dl class="strip">'
        f'<div><dt>Success, no distractors</dt><dd>{rate(0)}</dd></div>'
        f'<div><dt>Success, ten distractors</dt><dd>{rate(10)}</dd></div>'
        f'<div><dt>Median steps</dt><dd>{med_steps:.0f}</dd></div>'
        f'<div><dt>Median speed</dt><dd>{med_spd:.2f}'
        '<small> / 1.00 cap</small></dd></div></dl>',
        '<p class="note">These are illustrations, not evidence. They were '
        'picked by rule, not by eye: within each start-distance bucket the '
        'episodes closest to that bucket&rsquo;s <i>median</i> straightness '
        'were kept, so no path here is a flattering outlier. The claim they '
        'illustrate &mdash; success 1.000 with path length within a few '
        'percent of the straight line &mdash; is established by the 192 '
        'episodes behind the numbers above.</p>',
        '<div class="legend">'
        f'<span><i style="background:{PATH}"></i>path walked</span>'
        '<span><i style="background:var(--c4);opacity:.5"></i>capture ball, '
        'radius 1.0</span>'
        '<span>ordered by distance to goal, nearest first</span></div>',
        band(0, "No distractors",
             "Memory holds the goal alone."),
        band(10, "Ten distractors",
             "Memory holds the goal plus ten competing patterns; the readout "
             "has to pick the right one."),
        '<div class="foot">'
        f'checkpoint &nbsp;{esc(ckpt)}<br>'
        'encoder &nbsp;w52_attract_fwhm/001_att0.5_seed=43 &nbsp;&middot;&nbsp; '
        'encoder_gain 100 &nbsp;&middot;&nbsp; hopfield_beta 100 '
        '&nbsp;&middot;&nbsp; speed learned in [0.5, 1.0] '
        '&nbsp;&middot;&nbsp; log_kappa_max 2.5<br>'
        f'split &nbsp;{esc(first["split"])} &nbsp;&middot;&nbsp; '
        f'{esc(data.get("trials", "?"))} trials &times; 6 envs '
        f'&nbsp;&middot;&nbsp; max_steps {esc(data.get("max_steps", "?"))}'
        '</div>',
        '</div>',
    ]
    return "".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json")
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    data = json.load(open(a.json))
    ckpt = data["groups"][0].get("ckpt", "?")
    with open(a.out, "w") as f:
        f.write(build(data, ckpt))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
