"""A wall of trajectories: every env, every rollout, both models, matched.

Built because the aggregate statistics kept answering a subtly different
question than the one being asked -- `signed_turn_mean` hid the looping and
then the bidirectional circling, `straightness` hid the curl. This page makes
no claims. It draws the paths.

Layout is per ENV, with the two models on adjacent rows and matched starts, so
column j is the SAME start in the SAME arena under two policies. That is the
comparison the aggregates could not carry.

  python -m analysis.nav_tri.traj_grid --json grid.json --out page.html
"""

import argparse
import json

from analysis.nav_tri.exploit_report import CSS, FONTS, esc

# One colour per series, assigned by position. This was a two-entry dict keyed
# on the p20_e/p20_e_kcap pair, so any run comparing a different or longer set
# of arms silently KeyError'd -- a poor way to find out.
PALETTE = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)",
           "var(--s5)", "var(--s6)"]


def colour_for(labels):
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}

EXTRA = """
:root{ --s1:#b8791f; --s2:#6d5aa8; --s3:#1f7a6b;
        --s4:#a33b52; --s5:#3a6ea5; --s6:#6b7d24; }
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){ --s1:#c6852c; --s2:#9885da; --s3:#3fb5a0;
        --s4:#e0798c; --s5:#6fa5db; --s6:#a3ba4a; }
}
:root[data-theme="dark"]{ --s1:#c6852c; --s2:#9885da; --s3:#3fb5a0;
        --s4:#e0798c; --s5:#6fa5db; --s6:#a3ba4a; }
.envblk{margin-top:26px}
.envhd{font-family:"IBM Plex Mono",monospace;font-size:11px;
  letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
  border-bottom:1px solid var(--line);padding-bottom:6px;margin-bottom:10px}
.row{display:grid;grid-template-columns:78px repeat(10,minmax(0,1fr));
  gap:6px;align-items:center;margin-bottom:6px}
.rowlab{font-family:"IBM Plex Mono",monospace;font-size:10.5px;
  color:var(--ink);text-align:right;padding-right:4px}
.cellwrap{background:var(--panel);border:1px solid var(--line);
  border-radius:5px;padding:3px;min-width:0}
.cellwrap svg{display:block;width:100%;height:auto}
.n{font-family:"IBM Plex Mono",monospace;font-size:9px;color:var(--muted);
  text-align:center;margin-top:1px;font-variant-numeric:tabular-nums}
.legend{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin:12px 0;
  font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--muted)}
.legend b{display:inline-flex;align-items:center;gap:7px;font-weight:500;
  color:var(--ink)}
.sw{width:11px;height:11px;border-radius:2px;flex:none}
@media (max-width:900px){
  .row{grid-template-columns:60px repeat(5,minmax(0,1fr))}
}
"""


def mini(tr, size, color, w=100):
    """One path, small. Footprint faint so it does not fight the line."""
    pad = 3
    g = w - 2 * pad
    sc = g / float(size)

    def sx(x):
        return pad + (x + 0.5) * sc

    def sy(y):
        return pad + (size - 0.5 - y) * sc

    body = [f'<rect x="{pad}" y="{pad}" width="{g}" height="{g}" rx="2" '
            f'fill="var(--soft)" stroke="var(--line)" stroke-width="0.8"/>']
    for cx, cy in tr["cells"]:
        body.append(
            f'<rect x="{pad + cx * sc:.2f}" y="{pad + (size - 1 - cy) * sc:.2f}" '
            f'width="{sc:.2f}" height="{sc:.2f}" fill="{color}" '
            f'fill-opacity="0.13"/>')
    pts = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in tr["path"])
    body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="0.9" stroke-opacity="0.95" '
                f'stroke-linejoin="round"/>')
    p0 = tr["path"][0]
    body.append(f'<circle cx="{sx(p0[0]):.1f}" cy="{sy(p0[1]):.1f}" r="2.2" '
                f'fill="var(--panel)" stroke="{color}" stroke-width="1.1"/>')
    tip = (f'cov {tr["coverage"]:.3f} · edge {tr["edge_frac"]:.2f} · '
           f'turn {tr["signed_turn_mean"]:+.3f} rad/step')
    body.insert(0, f'<title>{esc(tip)}</title>')
    return (f'<svg viewBox="0 0 {w} {w}" role="img" '
            f'aria-label="{esc(tip)}">{"".join(body)}</svg>')


def build(d):
    size = d["size"]
    labels = d["labels"]
    SERIES = colour_for(labels)
    envs = sorted({t["env"] for t in d["trials"]})

    H = ["<title>Explore Trajectory Wall</title>", FONTS,
         f"<style>{CSS}{EXTRA}</style>", '<div class="wrap">']
    H.append(f"<h1>Every rollout, all {len(labels)} policies, "
             f"matched starts</h1>")
    H.append(f'<p class="sub">{len(envs)} envs &times; '
             f'{len(d["trials"]) // max(len(envs), 1)} rollouts &middot; '
             f'{d.get("max_steps", "?")} steps &middot; '
             f'{esc(str(d.get("split", "held-out")))} &middot; '
             f'{d.get("n_distractors", 0)} distractors</p>')
    H.append('<p class="lede">Column <i>j</i> is the same start in the same '
             'arena under every policy. Shaded cells are those visited; the '
             'hollow dot is the start. Numbers under each are coverage. '
             'No claims on this page &mdash; the aggregates kept answering a '
             'different question than the one asked, so this is here to be '
             'looked at.</p>')
    H.append('<div class="legend">')
    for lab in labels:
        H.append(f'<b><span class="sw" style="background:{SERIES[lab]}"></span>'
                 f'{esc(lab)}</b>')
    H.append('<span>&#9711; start</span></div>')

    for e in envs:
        rows = sorted([t for t in d["trials"] if t["env"] == e],
                      key=lambda t: t["trial"])
        H.append('<div class="envblk">')
        H.append(f'<div class="envhd">env {e}</div>')
        for lab in labels:
            H.append('<div class="row">')
            H.append(f'<div class="rowlab">{esc(lab)}</div>')
            for t in rows:
                r = t["by_ckpt"][lab]
                H.append('<div><div class="cellwrap">'
                         + mini(r, size, SERIES[lab])
                         + '</div>'
                         + f'<div class="n">{r["coverage"]:.2f}</div></div>')
            H.append("</div>")
        H.append("</div>")

    ck = d.get("ckpts", [])
    foot = " &middot; ".join(
        f"{esc(lab)} = {esc('/'.join(str(ck[i]).split('/')[-2:]))}"
        for i, lab in enumerate(labels) if i < len(ck))
    H.append(f'<p class="sub" style="margin-top:34px">{foot}</p>')
    H.append("</div>")
    return "\n".join(H)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    with open(a.out, "w") as fh:
        fh.write(build(json.load(open(a.json))))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
