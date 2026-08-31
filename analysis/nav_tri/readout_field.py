"""Where does the recalled direction field point, and where does it trap you?

EXPERIMENTS_NAV_P2 §13.5 found that 61% of exploit failures are *loops*: the
agent settles into a tight ball (radius 1.41 +/- 0.18) that is never centred on
the goal, and holds it for the whole horizon. §13.5.3 found that separately
trained policies land on the SAME phantom for the same goal -- centres agreeing
to 1.32 cells against 7.68 expected by chance -- which says the trap is in the
shared readout, not in any policy.

This module tests that directly, with **no policy and no rollouts**. `q(x)` is a
vector field over cells: for one memory (goal plus its distractors) it can be
evaluated everywhere at once. Then integrate it -- step along `unit(q)` at fixed
length, clip at the arena, exactly as the env would -- from every cell, and see
where the flow ends up.

The prediction is sharp. If the loops are sinks of `q`, then integrating the
field must reproduce them: the same locations, from the same basins, with no
policy involved. If the field flows cleanly to the goal from everywhere, then
the loops are the policy's doing after all and §13.5.3 is a coincidence of
shared training.

Usage:
    python -m analysis.nav_tri.readout_field \
        --ckpt $CLS_RUNS/agent_ckpts/<run>/navigate_u2000.pt \
        --n_distractors 10 --trials 4 --json field.json --html field.html
"""
from __future__ import annotations

import argparse
import html
import json

import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint, eval_env_set,
)
from hopfield_nav.evaluation.metrics import random_start
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield_nav.world import generate as gen
from analysis.nav_tri.signal_separability import _q_at


def field_over_cells(vh, hop, size, offset, device):
    """`q` at every cell of a size x size arena. Returns (size, size, 2)."""
    gx, gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)
    q, _ms, _rec = _q_at(vh, hop, cells, offset, device, multistep=())
    return np.asarray(q, dtype=np.float64).reshape(size, size, 2)


def integrate(field, size, goal, radius, *, step=1.0, max_steps=200,
              starts=None):
    """Follow `unit(q)` from every start. Returns (endpoints, reached, paths).

    Mirrors the env: fixed step length, sampled at the SNAPPED cell, clipped to
    the arena. Deliberately not a smarter integrator -- the question is where
    the agent's own update rule goes, not where an ideal one would.
    """
    if starts is None:
        gx, gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        starts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    pos = np.asarray(starts, dtype=np.float64).copy()
    g = np.asarray(goal, dtype=np.float64)
    reached = np.zeros(len(pos), dtype=bool)
    trail = [pos.copy()]
    for _ in range(max_steps):
        cell = np.clip(np.rint(pos).astype(int), 0, size - 1)
        v = field[cell[:, 0], cell[:, 1]]
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        d = np.divide(v, np.maximum(n, 1e-12)) * step
        d[n[:, 0] < 1e-12] = 0.0
        live = ~reached
        pos[live] = np.clip(pos[live] + d[live], 0.0, size - 1.0)
        reached |= np.linalg.norm(pos - g[None, :], axis=-1) <= radius
        trail.append(pos.copy())
    return pos, reached, np.asarray(trail)


def find_sinks(endpoints, reached, *, tol=2.0, min_basin=3):
    """Cluster the endpoints of flows that never reached the goal.

    Single-link clustering at `tol` cells. Crude on purpose: the claim is that
    the trapped flows pile up in a few tight places, and if that is true any
    reasonable clustering finds them.
    """
    pts = endpoints[~reached]
    if not len(pts):
        return []
    clusters = []
    for p in pts:
        for c in clusters:
            if np.linalg.norm(p - c["sum"] / c["n"]) <= tol:
                c["sum"] += p
                c["n"] += 1
                break
        else:
            clusters.append({"sum": p.copy(), "n": 1})
    out = [{"centre": (c["sum"] / c["n"]).tolist(), "basin": int(c["n"])}
           for c in clusters if c["n"] >= min_basin]
    return sorted(out, key=lambda c: -c["basin"])


def analyse(vh, env, offset, cfg, device, embed_dim, n_dist, trials, seed,
            step, max_steps, rng=None, draw_trials=None):
    """One env: reproduce exploit_diag's memory draws, map the field for each.

    `rng` is threaded ACROSS envs by the caller and `draw_trials` draws are
    consumed per env even when fewer are analysed. Both matter: exploit_diag
    consumes one RNG stream sequentially over (env, trial), so drawing a
    different NUMBER of trials in env 0 silently offsets every env after it and
    the memories stop matching. Only env 0 would still line up -- which is
    exactly the trap this signature exists to close.
    """
    size = env.size
    goal = env.goal_location
    radius = float(cfg.env.goal_radius)
    if rng is None:
        rng = np.random.RandomState(seed)
    draw_trials = trials if draw_trials is None else draw_trials
    out = []
    for b in range(draw_trials):
        pats = [goal_encoding(vh, offset, goal)]
        if n_dist > 0:
            pats.extend(sample_distractors(vh, offset, size, n_dist, rng))
        order = rng.permutation(len(pats))
        pats = [pats[j] for j in order]
        start = random_start(size, goal, rng)
        if b >= trials:
            continue          # draw consumed for alignment, not analysed

        hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
        for pat in pats:
            hop.input_memory(torch.from_numpy(pat).float())

        # Overlap statistics: how strongly does each stored pattern correlate
        # with the code at each cell? The claim the gain grid points at is that
        # a near-binary code keeps the DISTRACTOR overlaps uniformly small,
        # while a smooth code has a fat tail where one occasionally correlates
        # enough to bend the field. Measured here rather than asserted.
        _gx, _gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        _cells = np.stack([_gx.ravel(), _gy.ravel()], axis=1)
        _x = vh.get_encoded_state(_cells, offset).astype(np.float64)
        _x /= np.maximum(np.linalg.norm(_x, axis=-1, keepdims=True), 1e-12)
        _P = np.asarray(pats, dtype=np.float64)
        _P /= np.maximum(np.linalg.norm(_P, axis=-1, keepdims=True), 1e-12)
        _ov = np.abs(_x @ _P.T)                      # (cells, patterns)
        _gi = int(np.where(order == 0)[0][0]) if n_dist > 0 else 0
        _dist = np.delete(_ov, _gi, axis=1) if _ov.shape[1] > 1 else _ov[:, :0]
        overlap = {
            "goal_mean": float(_ov[:, _gi].mean()),
            "dist_mean": float(_dist.mean()) if _dist.size else 0.0,
            "dist_p99": float(np.percentile(_dist, 99)) if _dist.size else 0.0,
            "dist_max": float(_dist.max()) if _dist.size else 0.0,
        }

        fld = field_over_cells(vh, hop, size, offset, device)
        ends, reached, _ = integrate(fld, size, goal, radius, step=step,
                                     max_steps=max_steps)
        sinks = find_sinks(ends, reached)
        out.append({
            "trial": b, "goal": [int(goal[0]), int(goal[1])],
            "start": [int(start[0]), int(start[1])],
            "goal_basin": float(reached.mean()),
            "sinks": [{**s,
                       "dist_to_goal": float(np.linalg.norm(
                           np.asarray(s["centre"]) - np.asarray(goal, float)))}
                      for s in sinks],
            # Downsampled for the picture; the numbers above are on the full grid.
            "overlap": overlap,
            "field": fld[::2, ::2].round(3).tolist(),
            "size": size, "radius": radius,
        })
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _esc(s):
    return html.escape(str(s))


def render(data, title="Readout Field"):
    from analysis.nav_tri.exploit_report import CSS, FONTS
    out = [f"<title>{_esc(title)}</title>", FONTS, f"<style>{CSS}</style>",
           '<style>.fld{display:grid;'
           'grid-template-columns:repeat(auto-fit,minmax(230px,1fr));'
           'gap:14px;margin:14px 0}'
           '.fld figure{margin:0;background:var(--panel);'
           'border:1px solid var(--line);border-radius:8px;padding:9px;'
           'box-shadow:var(--shadow)}'
           '.fld figcaption{font-family:"IBM Plex Mono",monospace;font-size:10px;'
           'color:var(--muted);line-height:1.5;margin-top:5px;text-align:center}'
           '.fld figcaption b{color:var(--ink)}</style>',
           '<div class="wrap">', f'<h1>{_esc(title)}</h1>']
    tot = sum(len(e["cells"]) for e in data["envs"])
    out.append(f'<p class="sub">{tot} memories &middot; '
               f'{data["n_distractors"]} distractors &middot; no policy, no '
               f'rollouts</p>')
    out.append('<p class="lede">Each panel is the recalled direction field '
               '<code>q(x)</code> for one memory, evaluated at every cell. '
               'Arrows are the direction the readout would send an agent. The '
               'ring is the capture ball. Red dots are <b>sinks</b> &mdash; '
               'where the flow ends up when it does not reach the goal &mdash; '
               'found by integrating from all 400 cells with the env\'s own '
               'update rule: fixed step, snapped sampling, clipped at the '
               'arena.</p>')

    for e in data["envs"]:
        for c in e["cells"]:
            out.append('<div class="fld"><figure>')
            out.append(_field_svg(c))
            gb = c["goal_basin"]
            sink_txt = " &middot; ".join(
                f'({s["centre"][0]:.1f},{s["centre"][1]:.1f}) n={s["basin"]}'
                for s in c["sinks"][:3]) or "none"
            out.append(f'<figcaption>env {e["env_idx"]} trial {c["trial"]} '
                       f'&middot; goal ({c["goal"][0]},{c["goal"][1]})<br>'
                       f'<b>{100 * gb:.0f}%</b> of cells flow to the goal<br>'
                       f'sinks: {sink_txt}</figcaption>')
            out.append('</figure></div>')
    out.append('</div>')
    return "\n".join(out)


def _field_svg(c, w=230):
    size = c["size"]
    fld = np.asarray(c["field"], dtype=float)
    pad = 10
    g = w - 2 * pad
    sc = g / size
    step = size / fld.shape[0]

    def sx(x):
        return pad + (x + 0.5) * sc

    def sy(y):
        return pad + (size - 1 - y + 0.5) * sc

    body = [f'<rect class="arena" x="{pad}" y="{pad}" width="{g}" '
            f'height="{g}" rx="2"/>']
    for i in range(fld.shape[0]):
        for j in range(fld.shape[1]):
            v = fld[i, j]
            n = float(np.hypot(*v))
            if n < 1e-9:
                continue
            x, y = i * step, j * step
            ux, uy = v[0] / n, v[1] / n
            L = sc * step * 0.42
            x0, y0 = sx(x), sy(y)
            body.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" '
                        f'x2="{x0 + ux * L:.1f}" y2="{y0 - uy * L:.1f}" '
                        f'stroke="var(--muted)" stroke-width="0.9" '
                        f'stroke-opacity="0.75"/>')
    gx, gy = c["goal"]
    body.append(f'<circle cx="{sx(gx):.1f}" cy="{sy(gy):.1f}" '
                f'r="{max(c["radius"] * sc, 3):.1f}" fill="var(--c4)" '
                f'fill-opacity="0.18" stroke="var(--c4)" stroke-width="1.2"/>')
    for s in c["sinks"]:
        cx, cy = s["centre"]
        body.append(f'<circle cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" r="4" '
                    f'fill="var(--c5)" fill-opacity="0.85">'
                    f'<title>sink ({cx:.1f},{cy:.1f}) basin {s["basin"]}</title>'
                    f'</circle>')
    return (f'<svg viewBox="0 0 {w} {w}" width="100%" role="img">'
            + "".join(body) + '</svg>')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_distractors", type=int, default=10)
    p.add_argument("--trials", type=int, default=4,
                   help="Memories analysed per env.")
    p.add_argument("--draw_trials", type=int, default=None,
                   help="Draws CONSUMED per env, default = --trials. Must "
                        "equal the exploit_diag run's --trials (32) to compare "
                        "against it: the RNG is one stream over (env, trial), "
                        "so consuming a different number in env 0 offsets every "
                        "later env and only env 0 still lines up.")
    p.add_argument("--envs", type=int, default=None)
    p.add_argument("--split", default="recorded")
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the exploit_diag run being compared "
                        "against, or the memories differ and the sinks are "
                        "not the ones whose orbits were measured.")
    p.add_argument("--step", type=float, default=1.0,
                   help="Integration step length. The frozen-speed arms move "
                        "exactly 1.0 per step.")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", required=True)
    p.add_argument("--html", default=None)
    p.add_argument("--encoder_gain", type=float, default=None,
                   help="Override the checkpoint's encoder gain (code "
                        "sharpness). The field needs no policy, so this sweeps "
                        "a knob on ANY existing checkpoint without training "
                        "one -- which is how the encoder change and the recall "
                        "change get separated.")
    p.add_argument("--hopfield_beta", type=float, default=None,
                   help="Override the checkpoint's recall sharpness. At the "
                        "default the tanh argument is ~1e-4 and retrieval is a "
                        "linear blend; raising it saturates.")
    p.add_argument("--npos", type=int, default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    if args.envs is not None:
        cfg.num_val_envs = args.envs
    if args.npos is not None:
        print(f"  WARNING: --npos {args.npos} overrides the scaffold; "
              f"tool-validation only.")
        cfg.vectorhash.Npos = args.npos

    if args.encoder_gain is not None:
        cfg.encoder_gain = args.encoder_gain
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device),
        getattr(cfg, "encoder_gain", None))
    if args.hopfield_beta is not None:
        cfg.hopfield.beta = float(args.hopfield_beta)
    elif cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    print(f"encoder gain {gain:g}   hopfield beta {cfg.hopfield.beta:g}")
    embed_dim = enc_cfg.out_dim
    torch.manual_seed(0)
    np.random.seed(0)

    levels = gen.parse_levels(args.split)
    if levels is None:
        envs, vh, offsets = build_eval_world(
            cfg, encoder, str(device),
            ckpt_path=(None if args.npos is not None else args.ckpt))
    else:
        es = eval_env_set(cfg, encoder, str(device), ckpt_path=args.ckpt,
                          levels=levels, val_seed=args.val_seed,
                          n_envs=cfg.num_val_envs)
        envs, vh, offsets = es["envs"], es["field"], es["offsets"]

    print(f"goal_radius {cfg.env.goal_radius}  step {args.step}  "
          f"{len(envs)} envs x {args.trials} memories")

    draw = args.draw_trials if args.draw_trials is not None else args.trials
    if draw != args.trials:
        print(f"  consuming {draw} draws per env, analysing {args.trials}")
    out = {"ckpt": args.ckpt, "n_distractors": args.n_distractors,
           "step": args.step, "seed": args.seed, "trials": args.trials,
           "draw_trials": draw, "envs": []}
    # One stream across envs, exactly as exploit_diag._probe_group consumes it.
    rng = np.random.RandomState(args.seed)
    for i, env in enumerate(envs):
        cells = analyse(vh, env, offsets[i], cfg, device, embed_dim,
                        args.n_distractors, args.trials, args.seed,
                        args.step, args.max_steps, rng=rng, draw_trials=draw)
        out["envs"].append({"env_idx": i, "cells": cells})
        for c in cells:
            s = c["sinks"]
            print(f"  env {i} trial {c['trial']} goal {tuple(c['goal'])}: "
                  f"{100 * c['goal_basin']:.0f}% flow to goal, "
                  f"{len(s)} sink(s)"
                  + (f" -- biggest ({s[0]['centre'][0]:.1f},"
                     f"{s[0]['centre'][1]:.1f}) basin {s[0]['basin']} "
                     f"at {s[0]['dist_to_goal']:.1f} from goal" if s else ""))

    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.json}")
    if args.html:
        with open(args.html, "w") as fh:
            fh.write(render(out))
        print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
