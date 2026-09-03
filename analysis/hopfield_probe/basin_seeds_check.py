"""Is 2.5%'s low basin the encoder, or the one seed the ladder happens to use?

The coverage-ladder page probes one encoder seed per rung -- whichever won on
reach -- and the basin has a much wider per-(world, env) spread than reach does.
2.5% reads 12.0 against 1.25%'s 18.5, which is out of order for the ladder, so
the first thing to rule out is that `q_a1` seed 45 is simply a bad draw.

All four training seeds of both rungs, plus the runner-up arm at 2.5%
(`q_a2`, which the attract sweep put level with `q_a1`).
"""
from __future__ import annotations

import glob
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.attractor import basin_probe
from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (ProbeConfig, build_memory,
                                             load_probe_encoder, sample_worlds,
                                             scored_envs)

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
NPOS, K, STEP = 1716, 5, 1

ARMS = [
    ("2.5%  q_a1", f"{S}/w58_cov2.5/*_q_a1_seed=*", 100.0),
    ("2.5%  q_a2", f"{S}/w59_cov2.5_att_hi/*_q_a2_seed=*", 200.0),
    ("1.25% sm35x_a2", f"{S}/w60_cov1.25/*_sm35x_a2_seed=*", 100.0),
    ("5%    half_a0.5", f"{S}/w57_cov5/*_half_a0.5_seed=*", 75.0),
    ("10%   att0.5", f"{S}/w52_attract_fwhm/*_att0.5_seed=*", 100.0),
    ("0.75% y50_a2", f"{S}/w61_cov0.75/*_y50_a2_seed=*", 200.0),
]

cfg = ProbeConfig(n_worlds=8, n_envs_per_world=20, env_size=20, Npos=NPOS,
                  k_values=(K,), steps=(STEP,), seed=0)
worlds = sample_worlds(cfg)

print(f"Basin (median r_exact_all over 16 world/env pairs), K={K}, s={STEP}\n")
print(f"{'arm':18s}{'seed':>6s}{'median':>8s}{'p25':>7s}{'p75':>7s}"
      f"{'self-fail':>11s}")
print("-" * 57)
for lab, pat, gain in ARMS:
    meds = []
    for d in sorted(glob.glob(pat)):
        ck = os.path.join(d, "encoder_final.pt")
        if not os.path.exists(ck):
            continue
        seed = int(d.rsplit("=", 1)[1])
        enc, ecfg, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
        enc.gain = gain
        field = Field(enc, list(ecfg.lambdas), fwhm, gain, NPOS)
        vals = []
        for w in worlds:
            mem = build_memory(field, w, K, cfg,
                               np.random.RandomState(w.seed * 31 + K))
            for e in scored_envs(cfg, K)[:cfg.basin_envs]:
                r = basin_probe(field, w, e, mem, cfg, steps=(STEP,))
                if r:
                    vals.append(r[str(STEP)]["r_exact_all"])
        v = np.array(vals, float)
        meds.append(np.median(v))
        print(f"{lab:18s}{seed:6d}{np.median(v):8.1f}"
              f"{np.percentile(v, 25):7.1f}{np.percentile(v, 75):7.1f}"
              f"{(v < 0).mean():11.2f}")
    if meds:
        print(f"{'':18s}{'median':>6s}{np.median(meds):8.1f}")
    print()
