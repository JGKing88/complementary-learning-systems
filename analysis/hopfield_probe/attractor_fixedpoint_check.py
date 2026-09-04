"""Is there a fixed point at beta = 1e6, and is it the stored pattern?

Sec 10.16 said "it is an attractor now" off the trajectory probe, which decodes
the state to a CELL and found it stops moving. Sec 10.20 said "it was never an
attractor" off `cos_self` = 0.957. Those cannot both be the whole story.

Under saturation the update is x <- normalize(tanh(beta W x)), and at beta = 1e6
that is normalize(sign(Wx)) -- so the image of the map is always a hypercube
corner. A continuous stored pattern therefore CANNOT be a fixed point. The
question is whether some corner near it is, and whether the map lands there and
stays.

Measured, per stored pattern z, at K=5:

  cos(recall(z), z)          -- what `fixed_point_probe` reports
  cos(z, sign(z))            -- cos_bin, how near a corner the pattern is
  cos(recall(z), sign(z))    -- is the landing point exactly z's binarisation?
  cos(recall(x), x) for x = recall(z)   -- is the landing point a FIXED POINT?
  cos(recall^15(z), recall(z))          -- does it stay for 15 steps?
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (ProbeConfig, build_memory,
                                             load_probe_encoder,
                                             recall_trajectory, sample_worlds)

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
CK = f"{S}/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt"
K = 5


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


for beta, tag in ((None, "beta = gain = 100  (production)"),
                  (1e6, "beta = 1e6         (saturated)")):
    cfg = ProbeConfig(n_worlds=8, n_envs_per_world=20, env_size=20, Npos=1716,
                      k_values=(K,), steps=(1, 2, 15), seed=0,
                      beta_override=beta)
    worlds = sample_worlds(cfg)
    enc, ecfg, _g, fwhm, _h = load_probe_encoder(CK, fwhm_fallback=0.25)
    enc.gain = 100.0
    field = Field(enc, list(ecfg.lambdas), fwhm, 100.0, 1716)

    c_self, c_bin, c_land, c_fix, c_hold = [], [], [], [], []
    for w in worlds:
        mem = build_memory(field, w, K, cfg,
                           np.random.RandomState(w.seed * 31 + K))
        Z = unit(mem.Z)
        B = unit(np.sign(Z))

        traj = recall_trajectory(mem, Z, (1, 2, 15), cfg)
        X1, X15 = unit(traj[1]), unit(traj[15])
        # One more step FROM the landing point: a fixed point returns itself.
        X2 = unit(recall_trajectory(mem, traj[1], (1,), cfg)[1])

        c_self += list((X1 * Z).sum(1))
        c_bin += list((Z * B).sum(1))
        c_land += list((X1 * B).sum(1))
        c_fix += list((X2 * X1).sum(1))
        c_hold += list((X15 * X1).sum(1))

    def m(v):
        return f"{np.mean(v):.6f}  (min {np.min(v):.6f})"

    print(f"\n=== {tag} ===  att0.5-s43, K={K}, {len(c_self)} stored patterns")
    print(f"  cos(recall(z), z)        {m(c_self)}   <- fixed_point_probe")
    print(f"  cos(z, sign(z))          {m(c_bin)}   <- cos_bin")
    print(f"  cos(recall(z), sign(z))  {m(c_land)}   <- landed on z's corner?")
    print(f"  cos(recall(x), x)        {m(c_fix)}   <- x = recall(z) a FIXED "
          f"POINT?")
    print(f"  cos(recall^15(z), x)     {m(c_hold)}   <- does it stay?")
