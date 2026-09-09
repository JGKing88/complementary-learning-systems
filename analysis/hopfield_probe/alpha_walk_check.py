"""Does lowering alpha make the recall walk *through* encoded states?

Sec 6 decodes each recall step to its nearest cell and finds a one-step jump at
alpha = 1, but its ``L7 alpha 0.9`` row reads 8.2, 6.01, 3.08, 1.66, 0.60 --
which looks like a walk. Two things were never checked.

  1. Whether it reproduces on the PRODUCTION encoder, not just on L7.
  2. Whether those intermediate states are actually ON the code manifold.
     "Decodes to a cell 3.08 from the goal" is not "is the encoded state of a
     position 3.08 from the goal": a vector far off the manifold still has a
     nearest cell. ``retrieve`` already returns the max cosine, so the missing
     column is free.

The mechanism to test, from THEORY_ENCODER_HOPFIELD.md Sec 7.1: with alpha < 1
the state is a blend of the cue and the memory span, so it sits on the CHORD
from z(here) toward z(goal). A chord lies close to the manifold exactly when the
manifold is flat over that range -- the ballistic property (J3). So the
prediction is that the continuous code walks with a HIGH cosine to its nearest
cell, and the binarised one (arm B) either jumps or walks off-manifold with a
LOW cosine, because a blend of two hypercube corners is near no corner.

Note the norms. At alpha = 1 the cue's coefficient is exactly zero, so the state
is the recall term alone whatever its size. Below 1 the two compete, and they are
comparable only where ``(1-a)`` matches ``a*||tanh(beta W x)||`` -- about 0.9 for
production, where the recall term's norm is ~0.1, and about 0.03 for a saturated
arm, where it is sqrt(D) = 32. So the interesting alpha range is arm-dependent
and the sweep spans both.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.hopfield_probe.attractor import retrieve
from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (ProbeConfig, build_cell_bank,
                                             build_memory, load_probe_encoder,
                                             local_cells, recall_trajectory,
                                             sample_worlds, scored_envs)

DEFAULT_DIR = os.path.join(
    os.environ.get("CLS_RESULTS", "/orcd/pool/003/jackking/cls_runs/results"),
    "hopfield_probe/20260827/probe_ladder7")
ALPHAS = (1.0, 0.95, 0.9, 0.8, 0.5, 0.2, 0.05)
SHOW = (1, 2, 3, 5, 8, 12, 20, 30)


def headers(d: str, want: list[str]):
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    return [(e["label"], e["header"]) for e in man["encoders"]
            if any(e["label"].startswith(w) for w in want)]


def walk(header: dict, alpha: float, k: int, max_steps: int):
    """Per-step (distance to goal, cos to nearest cell, fraction in env)."""
    enc, ecfg, gain, fwhm, _h = load_probe_encoder(
        header["path"], fwhm_fallback=header.get("fwhm_ratio", 0.25))
    gain = float(header.get("gain", gain))
    enc.gain = gain
    beta = header.get("beta")
    cfg = ProbeConfig(n_worlds=2, n_envs_per_world=20, env_size=20, Npos=1716,
                      k_values=(k,), steps=(1,), seed=0, basin_radius=0,
                      alpha=alpha,
                      beta_override=float(beta) if beta else None)
    field = Field(enc, list(ecfg.lambdas), fwhm, gain, cfg.Npos)

    cells = local_cells(cfg.env_size)
    snaps = tuple(range(1, max_steps + 1))
    dist = {s: [] for s in snaps}
    cos = {s: [] for s in snaps}
    inenv = {s: [] for s in snaps}
    start = []

    for w in sample_worlds(cfg):
        rng = np.random.RandomState(w.seed * 31 + k)
        mem = build_memory(field, w, k, cfg, rng)
        bank = build_cell_bank(field, w, k, cfg, rng)
        for env in scored_envs(cfg, k):
            off = w.specs[env].offset
            goal = np.array(w.specs[env].goal)
            cues = field.encode(cells[:, 0] + off[0], cells[:, 1] + off[1])
            traj = recall_trajectory(mem, cues, snaps, cfg)
            start.append(np.sqrt(((cells - goal) ** 2).sum(1)).mean())
            for s in snaps:
                idx, val = retrieve(traj[s], bank, cfg)
                r_env, r_x, r_y = bank.decode(idx)
                same = r_env == env
                here = np.stack([r_x, r_y], axis=1).astype(float)
                d = np.sqrt(((here - goal) ** 2).sum(1))
                dist[s].append(float(np.mean(d[same])) if same.any() else np.nan)
                cos[s].append(float(np.mean(val)))
                inenv[s].append(float(same.mean()))

    return (float(np.mean(start)),
            {s: (np.nanmean(dist[s]), np.mean(cos[s]), np.mean(inenv[s]))
             for s in snaps})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--labels", nargs="+", default=["10% · ", "10% gain=1e6"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=30)
    args = ap.parse_args()

    seen: set[str] = set()
    for label, header in headers(args.dir, args.labels):
        arm = label.split(" · ")[0]
        if arm in seen:
            continue
        seen.add(arm)
        print(f"\n{'=' * 78}\n=== {label} ===")
        for alpha in ALPHAS:
            s0, res = walk(header, alpha, args.k, args.max_steps)
            print(f"\n  alpha {alpha:<5g}  start {s0:.2f} cells from goal")
            print(f"    {'step':<6s}" + "".join(f"{s:>8d}" for s in SHOW))
            print(f"    {'cells':<6s}"
                  + "".join(f"{res[s][0]:>8.2f}" for s in SHOW))
            print(f"    {'cos':<6s}"
                  + "".join(f"{res[s][1]:>8.3f}" for s in SHOW)
                  + "   <- to its own nearest cell")
            print(f"    {'in env':<6s}"
                  + "".join(f"{res[s][2]:>8.2f}" for s in SHOW))


if __name__ == "__main__":
    main()
