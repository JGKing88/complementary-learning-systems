"""Is the ~0.99 plateau the encoder, or the arrival radius?

The best arms post discrete reach ~1.000 with continuous 0.98-0.99, so every
start reaches the goal CELL and about 1% never gets within 0.5 of the goal
POINT. Two very different causes:

  geometric -- the walker converges to a fixed point of the snap, or oscillates
    across a cell boundary, and ends up parked within ~0.5-1.5 cells of the goal
    forever. That is `ARRIVAL_RADIUS` against a float position, i.e. a probe
    parameter, and no encoder can fix it.

  real -- the walker ends far from the goal, so the field genuinely fails there.

`continuous_flow` records final `steps` = -1 for anything that never arrived,
but not where it stopped, so this re-runs the walk on the archived q fields and
records the final distance. If the failures pile up just outside 0.5, the metric
has a ceiling and further alias-rate work buys nothing measurable.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.flow import ARRIVAL_RADIUS, _unit_q
from analysis.hopfield_probe.harness import (ProbeConfig, build_memory,
                                             load_probe_encoder, local_cells,
                                             sample_worlds, scored_envs)
from analysis.hopfield_probe.qfield import cell_q_field

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
ARCH = "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260827"
NPOS = 1716
K, STEP = 5, 1

ARMS = [
    ("att0.25 @g30", f"{S}/w55_nav_objective/*_att0.25_seed=42", 30.0),
    ("att0.5 @g100", f"{S}/w52_attract_fwhm/*_att0.5_seed=42", 100.0),
]


def final_distances(field, world, cfg, k):
    """Final distance to the goal for every start that never arrived."""
    out = []
    rng = np.random.RandomState(world.seed * 13 + k)
    mem = build_memory(field, world, k, cfg, rng)
    size = cfg.env_size
    for e in scored_envs(cfg, k):
        qf, _c, _b = cell_q_field(field, world, e, mem, cfg)
        q = qf[STEP]
        goal = np.array(world.specs[e].goal, dtype=float)
        qh = _unit_q(q)
        p = local_cells(size).astype(float)
        arrived = np.zeros(p.shape[0], dtype=bool)
        for _t in range(cfg.flow_max_steps_factor * size):
            d = np.linalg.norm(p - goal, axis=1)
            arrived |= d <= ARRIVAL_RADIUS
            if arrived.all():
                break
            cx = np.clip(np.round(p[:, 0]), 0, size - 1).astype(np.int64)
            cy = np.clip(np.round(p[:, 1]), 0, size - 1).astype(np.int64)
            stp = qh[cx * size + cy] * cfg.continuous_scale
            stp[arrived] = 0.0
            p = np.clip(p + stp, -0.5, size - 0.5)
        d = np.linalg.norm(p - goal, axis=1)
        out.append(d[~arrived])
    return np.concatenate(out) if out else np.array([])


cfg = ProbeConfig(n_worlds=8, n_envs_per_world=20, env_size=20, Npos=NPOS,
                  k_values=(K,), steps=(STEP,), seed=0)
worlds = sample_worlds(cfg)

for lab, pat, gain in ARMS:
    ck = os.path.join(sorted(glob.glob(pat))[0], "encoder_final.pt")
    enc, ecfg, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
    enc.gain = gain
    field = Field(enc, list(ecfg.lambdas), fwhm, gain, NPOS)

    d = np.concatenate([final_distances(field, w, cfg, K) for w in worlds])
    total = len(worlds) * len(scored_envs(cfg, K)) * cfg.env_size ** 2
    print(f"\n=== {lab} ===")
    print(f"non-arrivals: {len(d)} of {total} starts "
          f"({len(d) / total:.2%})")
    if len(d) == 0:
        continue
    for lo, hi in ((0.5, 1.0), (1.0, 1.5), (1.5, 3.0), (3.0, 6.0),
                   (6.0, 1e9)):
        m = (d >= lo) & (d < hi)
        tag = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
        print(f"  final distance {tag:>8s}: {m.sum():5d}"
              f"  ({m.mean():6.1%} of failures)")
    print(f"  median final distance: {np.median(d):.2f} cells")

print("\nFailures parked just outside 0.5 are the arrival radius, not the")
print("encoder. Failures far away are real field failures.")
