"""Decouple the recall gain from the encoder gain.

Production sets `cfg.hopfield.beta = encoder_gain`, so one number does two
unrelated jobs: it sets how saturated the *embedding* is (the corner condition)
and how saturated the *recall* is (the loop gain). They need very different
values.

Per coordinate, (W z)_i ~ 1/D^1.5, so saturating the recall needs
beta ~ D^1.5 ~ 3e4 at unit-norm storage -- while the corner arrives at encoder
gain ~300 and the q readout is already degrading by gain 3000. Tied together,
no single value can satisfy both. Swept apart, they might.
"""
import sys
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np
import torch

from analysis.hopfield_probe.encode import Field, grid_codes
from analysis.hopfield_probe.harness import (
    ProbeConfig, build_memory, load_probe_encoder, local_cells, sample_worlds,
    scored_envs,
)
from analysis.hopfield_probe.qfield import GridAcc, cell_q_field

R = "/orcd/pool/003/jackking/cls_runs"
enc, cfg_m, g0, fwhm, _ = load_probe_encoder(
    f"{R}/sweeps/w53_attract_knee/004_att16_seed=42/encoder_final.pt")

ENC_GAIN = 300.0        # corner met (cos_bin 0.960), best measured acc45
BETAS = [300.0, 1e3, 1e4, 3e4, 1e5, 1e6]

rng = np.random.RandomState(0)
gx, gy = rng.randint(0, 1716, 25), rng.randint(0, 1716, 25)
codes = torch.from_numpy(grid_codes(list(cfg_m.lambdas), gx, gy, fwhm))
enc.gain = ENC_GAIN
with torch.no_grad():
    Z = torch.nn.functional.normalize(
        torch.tanh(ENC_GAIN * enc.net(codes)), dim=-1).numpy().astype(np.float64)

W = (Z.T @ Z) / Z.shape[1]
np.fill_diagonal(W, 0.0)
per_coord = np.abs(W @ Z[0]).mean()

cfg = ProbeConfig(Npos=1716, env_size=20, n_worlds=3, n_envs_per_world=5,
                  k_values=(5,), steps=(1,), n_alias=2000, n_score_envs=3,
                  n_cont_samples=1, n_cont_annulus=0)
cfg.validate()
worlds = sample_worlds(cfg)
cells = local_cells(cfg.env_size)

print(f"encoder gain {ENC_GAIN:.0f} (cos_bin 0.960), 25 patterns, D=1024")
print(f"mean |(W z)_i| = {per_coord:.3e}  ->  saturating tanh needs "
      f"beta ~ {1 / per_coord:.3g}\n")
print(f"{'beta':>9s} {'beta*|Wz|_i':>12s} {'fixed pt':>9s} {'pairwise':>9s} "
      f"{'B |err|':>9s} {'acc45':>8s}")
print("-" * 62)

for beta in BETAS:
    fp = []
    for k in range(25):
        x = Z[k].copy()
        for _ in range(20):
            x = np.tanh(beta * (W @ x))
            n = np.linalg.norm(x)
            x = x / n if n > 1e-12 else x
        fp.append(float(np.dot(x, Z[k])))
    # Do different cues still land in different places?
    ends = []
    for k in range(25):
        x = Z[k].copy()
        for _ in range(20):
            x = np.tanh(beta * (W @ x))
            n = np.linalg.norm(x)
            x = x / n if n > 1e-12 else x
        ends.append(x)
    E = np.array(ends)
    pair = float(np.mean((E @ E.T)[np.triu_indices(25, 1)]))

    field = Field(enc, list(cfg_m.lambdas), fwhm, ENC_GAIN, cfg.Npos)
    sub = ProbeConfig(**{**cfg.__dict__, "beta_override": beta})
    acc = GridAcc(sub)
    for w in worlds:
        rw = np.random.RandomState(w.seed)
        mem = build_memory(field, w, 5, sub, rw)
        for e in scored_envs(sub, 5):
            qf, _c, _b = cell_q_field(field, w, e, mem, sub)
            acc.add_env(qf[1], cells, w.specs[e].goal)
    j = acc.scalars.to_json()

    print(f"{beta:9.0f} {beta * per_coord:12.3f} {np.median(fp):9.4f} "
          f"{pair:9.4f} {j['abs_err_mean']['mean']:8.1f}° "
          f"{j['acc45']['mean'] * 100:7.1f}%")

print("\nnav_p2 5.7 corner target: fixed pt 0.976 at M=25.")
print("A real attractor needs fixed pt high AND pairwise low -- both, since a")
print("single global attractor scores a perfect fixed point trivially.")
