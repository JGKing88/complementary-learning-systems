"""The basin measured without reference to any environment.

`run_test_a` draws its cues from `local_cells(env_size)`, so `r_exact_95` cannot
exceed the arena diagonal -- ~27 cells at env 20, ~55 at env 40. That bound has
nothing to do with the encoder. A basin is a property of the encoder and the
stored memory, and the cue can sit anywhere in the scaffold.

`basin_probe` drops the environment entirely: EVERY cell in the disc of radius
`cfg.basin_radius` around the goal, in scaffold coordinates, is both a cue and a
bank row, plus the stored goals. Every cell and not a sample of them, because
retrieval is an argmax and a sparse bank lets the goal win by default whenever
the state's true nearest cell is missing from the menu.

`exact` keeps Test A's definition unchanged: the retrieved cell IS the goal cell.
This runs it on the coverage ladder and prints the env-censored numbers beside
it.
"""
from __future__ import annotations

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

LADDER = [
    ("10%   att0.5", f"{S}/w52_attract_fwhm/001_att0.5_seed=43", 100.0,
     21.12, 30.73),
    ("5%    half_a0.5", f"{S}/w57_cov5/001_half_a0.5_seed=43", 75.0,
     20.73, 27.68),
    ("2.5%  q_a1", f"{S}/w58_cov2.5/011_q_a1_seed=45", 100.0, 17.62, 17.95),
    ("1.25% sm35x_a2", f"{S}/w60_cov1.25/014_sm35x_a2_seed=44", 100.0,
     16.62, 17.55),
    ("0.75% y50_a2", f"{S}/w61_cov0.75/014_y50_a2_seed=44", 200.0,
     11.12, 12.95),
]

# Every cell in the disc is a cue AND a bank row, so cost is O(R^2). 64 is
# ample against the largest basin ever measured (30.7 at env 40).
cfg = ProbeConfig(n_worlds=8, n_envs_per_world=20, env_size=20, Npos=NPOS,
                  k_values=(K,), steps=(STEP,), seed=0, basin_radius=64)
worlds = sample_worlds(cfg)

print(f"Basin radius (r_exact_all), K={K}, s={STEP}. `free` uses every cell within "
      f"{cfg.basin_radius} cells of the\ngoal in scaffold coordinates, as both "
      f"cue and bank, with no environment involved.\n")
print(f"{'encoder':18s}{'env 20':>9s}{'env 40':>9s}{'free':>9s}"
      f"{'p25':>8s}{'p75':>8s}{'exact':>8s}")
print(f"{'':18s}{'(cap 27)':>9s}{'(cap 55)':>9s}{'(cap 200)':>9s}")
print("-" * 69)

for lab, d, gain, b20, b40 in LADDER:
    enc, ecfg, own, fwhm, _ = load_probe_encoder(d + "/encoder_final.pt",
                                                 fwhm_fallback=0.25)
    enc.gain = gain
    field = Field(enc, list(ecfg.lambdas), fwhm, gain, NPOS)

    vals, fracs = [], []
    for w in worlds:
        rng = np.random.RandomState(w.seed * 31 + K)
        mem = build_memory(field, w, K, cfg, rng)
        for e in scored_envs(cfg, K):
            r = basin_probe(field, w, e, mem, cfg, steps=(STEP,))
            if r:
                vals.append(r[str(STEP)]["r_exact_all"])
                fracs.append(r[str(STEP)]["exact_frac"])
    v = np.array(vals, float)
    print(f"{lab:18s}{b20:9.2f}{b40:9.2f}{v.mean():9.2f}"
          f"{np.percentile(v, 25):8.1f}{np.percentile(v, 75):8.1f}"
          f"{np.mean(fracs):8.3f}")

print(f"\n{len(vals)} (world, env) pairs per encoder. `exact` is the fraction of "
      f"all cells within\n{cfg.basin_radius} of the goal that retrieve it, so "
      f"it is dominated by the large-radius\nannuli and is not comparable to "
      f"the env-bounded exact_frac.")
