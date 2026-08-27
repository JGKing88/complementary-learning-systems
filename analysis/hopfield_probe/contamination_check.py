"""Raising beta makes recall return a CORNER, not the stored pattern. Does that
contaminate q?

At saturation, normalize(tanh(beta*Wx)) -> sign(Wx)/sqrt(D). The stored pattern
is only cos ~0.96 from its own binarisation, so the recalled vector differs from
what was stored by a residual of norm sqrt(1 - 0.96^2) ~ 0.28 -- against a
one-cell displacement of ~0.037. Seven times larger. That should wreck the
readout.

But q is not the displacement; it is the displacement *projected onto a 2-D
local basis*. Decompose it:

    q = W_b (recalled - z_c)
      = W_b (z_goal - z_c)          <- signal
      + W_b (recalled - z_goal)     <- contamination

and measure the two terms in the plane that actually reaches the policy.
"""
import sys
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (
    ProbeConfig, build_memory, load_probe_encoder, local_cells,
    recall_trajectory, sample_worlds,
)
from analysis.hopfield_probe.qfield import project_q

R = "/orcd/pool/003/jackking/cls_runs"
enc, cfg_m, g0, fwhm, _ = load_probe_encoder(
    f"{R}/sweeps/w53_attract_knee/004_att16_seed=42/encoder_final.pt")
ENC_GAIN = 300.0
enc.gain = ENC_GAIN

base = ProbeConfig(Npos=1716, env_size=20, n_worlds=3, n_envs_per_world=5,
                   k_values=(5,), steps=(1,), n_alias=10, n_score_envs=2,
                   n_cont_samples=1, n_cont_annulus=0)
base.validate()
worlds = sample_worlds(base)
field = Field(enc, list(cfg_m.lambdas), fwhm, ENC_GAIN, base.Npos)
cells = local_cells(base.env_size)

print(f"encoder gain {ENC_GAIN:.0f}, K=5, one recall step")
print(f"{'beta':>9s} {'cos(recall,':>12s} {'|signal|':>9s} {'|contam|':>9s} "
       f"{'contam/':>8s} {'angle q vs':>11s}")
print(f"{'':9s} {'z_goal)':>12s} {'in plane':>9s} {'in plane':>9s} "
      f"{'signal':>8s} {'oracle q':>11s}")
print("-" * 66)

import dataclasses
for beta in (300.0, 1e4, 1e5, 1e6, 1e7):
    cfg = dataclasses.replace(base, beta_override=beta)
    cos_r, sig, con, ang = [], [], [], []
    for w in worlds:
        rng = np.random.RandomState(w.seed)
        mem = build_memory(field, w, 5, cfg, rng)
        for e in range(2):
            off = w.specs[e].offset
            zc = field.encoded_state(cells, off)
            basis = field.local_basis(cells, off)
            zg = field.encoded_state(np.array([w.specs[e].goal]), off)[0]
            rec = recall_trajectory(mem, zc, (1,), cfg)[1]

            rn = rec / np.linalg.norm(rec, axis=1, keepdims=True)
            cos_r.append(float(np.mean(rn @ zg)))

            q_true = project_q(basis, zc, np.broadcast_to(zg, zc.shape))
            q_cont = project_q(basis, np.zeros_like(zc),
                               rn - np.broadcast_to(zg, zc.shape))
            q_real = project_q(basis, zc, rn)

            sig.append(float(np.mean(np.linalg.norm(q_true, axis=1))))
            con.append(float(np.mean(np.linalg.norm(q_cont, axis=1))))
            a = np.degrees(np.abs(np.arctan2(
                q_real[:, 1] * q_true[:, 0] - q_real[:, 0] * q_true[:, 1],
                q_real[:, 0] * q_true[:, 0] + q_real[:, 1] * q_true[:, 1])))
            ang.append(float(np.mean(a)))

    s, c = np.mean(sig), np.mean(con)
    print(f"{beta:9.0f} {np.mean(cos_r):12.4f} {s:9.4f} {c:9.4f} "
          f"{c / s:8.3f} {np.mean(ang):10.2f}°")

print("\n'angle q vs oracle q' is how far the real q rotates away from the q a")
print("perfect recall of the STORED pattern would give. That is the number the")
print("policy feels; |contam|/|signal| is why.")
