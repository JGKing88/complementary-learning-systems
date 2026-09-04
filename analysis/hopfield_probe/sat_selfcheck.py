"""If the memory is a fixed point, how can the goal cue fail to retrieve it?

The objection is exact: a state equal to the stored pattern has cosine 1 to it
and cannot lose an argmax. So either the saturated state is NOT the stored
pattern, or the basin probe is wrong.

Candidate: the saturated update is `x <- sign(Wx)/sqrt(D)`, so the fixed point
is a hypercube CORNER near the pattern, not the pattern. Its cosine to the goal
is roughly `cos_bin` (0.955 for att0.5), and a neighbouring cell one step away
has cosine ~0.998 to the goal -- so the corner may sit almost equally far from
the goal and from its neighbours, making the argmax a near-tie.

Measured here, for the goal cue only:
  * cos(recalled, goal cell) and cos(recalled, best OTHER cell);
  * the margin between them, which is what the argmax turns on;
  * how far away the winner is when the goal loses.
"""
from __future__ import annotations

import glob
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (ProbeConfig, build_memory,
                                             load_probe_encoder,
                                             recall_trajectory, sample_worlds,
                                             scored_envs)

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
CK = f"{S}/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt"
NPOS, K, R = 1716, 5, 64


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


for beta, tag in ((None, "beta = gain (100)"), (1e6, "beta = 1e6")):
    cfg = ProbeConfig(n_worlds=8, n_envs_per_world=20, env_size=20, Npos=NPOS,
                      k_values=(K,), steps=(1,), seed=0, beta_override=beta)
    worlds = sample_worlds(cfg)
    enc, ecfg, own, fwhm, _ = load_probe_encoder(CK, fwhm_fallback=0.25)
    enc.gain = 100.0
    field = Field(enc, list(ecfg.lambdas), fwhm, 100.0, NPOS)

    a = np.arange(-R, R + 1)
    dx, dy = np.meshgrid(a, a, indexing="ij")
    d = np.hypot(dx, dy).ravel()
    ins = d <= R
    dxf, dyf, d = dx.ravel()[ins], dy.ravel()[ins], d[ins]

    cg, cw, marg, wd, cbin = [], [], [], [], []
    for w in worlds:
        mem = build_memory(field, w, K, cfg,
                           np.random.RandomState(w.seed * 31 + K))
        for e in scored_envs(cfg, K)[:2]:
            off, goal = w.specs[e].offset, w.specs[e].goal
            gx, gy = int(goal[0] + off[0]), int(goal[1] + off[1])
            cells = field.encode(gx + dxf, gy + dyf)
            gi = int(np.flatnonzero(d == 0.0)[0])
            B = unit(cells)

            x = unit(recall_trajectory(mem, cells[gi:gi + 1], (1,), cfg)[1])[0]
            cos = B @ x
            other = cos.copy()
            other[gi] = -np.inf
            j = int(other.argmax())
            cg.append(cos[gi])
            cw.append(cos[j])
            marg.append(cos[gi] - cos[j])
            wd.append(d[j])
            # How near a corner the goal pattern itself is.
            z = B[gi]
            b = np.sign(z) / np.sqrt(z.size)
            cbin.append(float(z @ b / np.linalg.norm(b)))

    cg, cw, marg, wd = map(np.asarray, (cg, cw, marg, wd))
    print(f"\n=== {tag} ===   att0.5-s43, K={K}, goal cue only, "
          f"{len(cg)} (world, env) pairs")
    print(f"  cos(recalled, goal cell)   mean {cg.mean():.4f}  "
          f"min {cg.min():.4f}")
    print(f"  cos(recalled, best other)  mean {cw.mean():.4f}  "
          f"max {cw.max():.4f}")
    print(f"  margin goal - other        mean {marg.mean():+.5f}  "
          f"min {marg.min():+.5f}")
    print(f"  goal loses in              {(marg < 0).sum()} of {len(marg)}"
          f"   winner at d = {sorted(set(wd[marg < 0].tolist()))[:5]}")
    print(f"  cos_bin of the goal pattern  {np.mean(cbin):.4f}")
