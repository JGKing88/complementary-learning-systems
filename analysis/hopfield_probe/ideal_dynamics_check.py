"""Recall dynamics of the ideal encoder: fixed points, the alpha walk, the chord.

Three questions, each already asked of the trained encoders (THEORY Sec 7.1,
``alpha_walk_check.py``, ``chord_manifold_check.py``), now asked of the analytic
ideal code (``ideal_encoder.py``) with the SAME functions, so the rows compare:

  1. **Fixed points.** Recall from each stored pattern itself for up to 30 steps
     at alpha = 1, under ``hebb`` and ``proj`` storage. ``proj`` stores the
     orthogonal projector onto span(Z) without zeroing the diagonal, so
     ``P z = z``: in the linear recall regime every stored pattern should be an
     exact fixed point (cos to itself 1.000 at every step). Under ``hebb`` it
     should drift (matched-filter power iteration).
  2. **The alpha walk.** Cues from every cell of the env, alpha swept; decode
     every step to its nearest cell and record cos(state, that cell's code).
     A walk is a sequence of decoded distances that falls smoothly with cos
     staying near 1; a snap is a stall followed by a jump with a cos dip.
  3. **The chord.** ``x(t) = (1-t) z(here) + t z(goal)`` at several start
     distances. For a Gaussian-kernel code the blend's similarity profile over
     cells is a sum of two Gaussians, unimodal iff the separation is <= 2r, so
     the prediction is a smooth walk up to ~2r and a snap beyond it.

One JSON per encoder header. The headers are built here rather than read from a
probe manifest, because the ideal encoder has no checkpoint:

  ideal r=16 beta=100   (the probe's default: beta = gain, tanh linear)
  ideal r=16 beta=1e6   (recall tanh saturated; the code itself has no tanh)
  att0.5 s42 beta=100   (production, paired)
  att0.5 s42 gain=beta=1e6  (arm B, paired)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.hopfield_probe.alpha_walk_check import walk
from analysis.hopfield_probe.attractor import fixed_point_probe, retrieve
from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (ProbeConfig, build_cell_bank,
                                             build_memory, load_probe_encoder,
                                             local_cells, sample_worlds,
                                             scored_envs)

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
PROD = f"{S}/w52_attract_fwhm/000_att0.5_seed=42/encoder_final.pt"
IDEAL = "ideal:r=16,n_freq=512,seed=0,gain=100"
HEADERS = [
    {"label": "ideal r=16 · β=100", "path": IDEAL, "gain": 100.0,
     "beta": 100.0, "fwhm_ratio": 0.25},
    {"label": "ideal r=16 · β=1e6", "path": IDEAL, "gain": 100.0,
     "beta": 1e6, "fwhm_ratio": 0.25},
    {"label": "att0.5 s42 · β=100", "path": PROD, "gain": 100.0,
     "beta": 100.0, "fwhm_ratio": 0.25},
    {"label": "att0.5 s42 · gain=β=1e6 (arm B)", "path": PROD, "gain": 1e6,
     "beta": 1e6, "fwhm_ratio": 0.25},
]
ALPHAS = (1.0, 0.95, 0.9, 0.8, 0.5, 0.2, 0.05, 0.01, 0.003)
SEPS = (5.0, 10.0, 15.0, 20.0)
TS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0)
FP_STEPS = tuple(range(1, 31))


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-30)


def _field(header, cfg):
    enc, ecfg, gain, fwhm, _h = load_probe_encoder(
        header["path"], fwhm_fallback=header.get("fwhm_ratio", 0.25))
    gain = float(header.get("gain", gain))
    enc.gain = gain
    return Field(enc, list(ecfg.lambdas), fwhm, gain, cfg.Npos)


def fixed_points(header, k, rule):
    """Self-recall over 30 steps at alpha = 1, pooled over worlds."""
    cfg = ProbeConfig(n_worlds=4, n_envs_per_world=20, env_size=20, Npos=1716,
                      k_values=(k,), steps=FP_STEPS, seed=0, basin_radius=0,
                      alpha=1.0, storage_rule=rule,
                      beta_override=float(header["beta"]))
    field = _field(header, cfg)
    pooled = {s: [] for s in FP_STEPS}
    for w in sample_worlds(cfg):
        rng = np.random.RandomState(w.seed * 31 + k)
        mem = build_memory(field, w, k, cfg, rng)
        fp = fixed_point_probe(mem, cfg)
        for s in FP_STEPS:
            pooled[s].append(fp[str(s)])
    return {str(s): {key: float(np.nanmean([d[key] for d in pooled[s]]))
                     for key in pooled[s][0]} for s in FP_STEPS}


def chord(header, k, sep):
    """``chord_manifold_check.chord`` with the header's beta, as a dict."""
    cfg = ProbeConfig(n_worlds=1, n_envs_per_world=20, env_size=20, Npos=1716,
                      k_values=(k,), steps=(1,), seed=0, basin_radius=0,
                      beta_override=float(header["beta"]))
    field = _field(header, cfg)
    w = sample_worlds(cfg)[0]
    rng = np.random.RandomState(w.seed * 31 + k)
    build_memory(field, w, k, cfg, rng)
    bank = build_cell_bank(field, w, k, cfg, rng)
    env = scored_envs(cfg, k)[0]
    off, goal = w.specs[env].offset, np.array(w.specs[env].goal)
    cells = local_cells(cfg.env_size)
    d = np.sqrt(((cells - goal) ** 2).sum(1))
    here = cells[np.abs(d - sep) < 0.75]
    if here.shape[0] == 0:
        return {"n_start": 0, "rows": []}
    z_here = _unit(field.encode(here[:, 0] + off[0], here[:, 1] + off[1]))
    z_goal = _unit(field.encode(np.full(here.shape[0], goal[0] + off[0]),
                                np.full(here.shape[0], goal[1] + off[1])))
    rows = []
    for t in TS:
        x = _unit((1.0 - t) * z_here + t * z_goal)
        idx, val = retrieve(x, bank, cfg)
        r_env, r_x, r_y = bank.decode(idx)
        same = r_env == env
        dist = np.sqrt((r_x - goal[0]) ** 2 + (r_y - goal[1]) ** 2)
        rows.append({"t": t,
                     "dist": float(np.mean(dist[same])) if same.any() else None,
                     "cos": float(val.mean()), "in_env": float(same.mean())})
    return {"n_start": int(here.shape[0]), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, required=True,
                    help="which of HEADERS to run")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    h = HEADERS[args.index]
    os.makedirs(args.out, exist_ok=True)
    out = {"header": h, "k": args.k, "alphas": list(ALPHAS),
           "fixed_points": {}, "walk": {}, "chord": {}}

    for rule in ("hebb", "proj"):
        for k in (5, 20):
            out["fixed_points"][f"{rule}|K={k}"] = fixed_points(h, k, rule)
            s1, s30 = (out["fixed_points"][f"{rule}|K={k}"][s]
                       for s in ("1", "30"))
            print(f"[fixed] {h['label']:34s} {rule:4s} K={k:2d}  cos_self "
                  f"s1 {s1['cos_self_mean']:.4f}  s30 {s30['cos_self_mean']:.4f}"
                  f"  self-consistent s30 {s30['frac_self_consistent']:.2f}",
                  flush=True)

    walk_header = {"path": h["path"], "gain": h["gain"], "beta": h["beta"],
                   "fwhm_ratio": h["fwhm_ratio"]}
    for rule in ("hebb", "proj"):
        for a in ALPHAS:
            s0, res = walk(walk_header, a, args.k, args.max_steps, rule)
            out["walk"][f"{rule}|{a:g}"] = {
                "start": s0,
                "steps": {str(s): {"dist": float(v[0]), "cos": float(v[1]),
                                   "in_env": float(v[2])}
                          for s, v in res.items()}}
            seq = " ".join(f"{res[s][0]:.2f}" for s in (1, 2, 3, 5, 8, 12, 20, 30))
            print(f"[walk]  {h['label']:34s} {rule:4s} α={a:<6g} start "
                  f"{s0:.2f}: {seq}  min cos "
                  f"{min(v[1] for v in res.values()):.3f}", flush=True)

    for sep in SEPS:
        out["chord"][f"{sep:g}"] = chord(h, args.k, sep)
        rows = out["chord"][f"{sep:g}"]["rows"]
        if rows:
            print(f"[chord] {h['label']:34s} sep {sep:g}: " + " ".join(
                f"{r['dist'] if r['dist'] is not None else float('nan'):.1f}"
                for r in rows) + f"  min cos {min(r['cos'] for r in rows):.3f}",
                flush=True)

    slug = "".join(c if c.isalnum() else "_" for c in h["label"])
    path = os.path.join(args.out, f"{slug}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
