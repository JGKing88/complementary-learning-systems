"""Push the encoder gain at inference: does the corner arrive, and does q survive?

Two questions in one sweep.

1. nav_p2 5.7's corner needs cos-to-binarisation ~0.954. The encoders sit below
   it because the tanh argument is `gain * pre-activation` and the learned
   pre-activations are ~0.02 RMS. Overriding gain at inference raises the
   argument without retraining.
2. 5.7 left open whether "the tangent projection still decodes direction from a
   saturated pattern -- the question that matters, because direction is what the
   agent uses." Test B's acc45 answers exactly that, so measure it at each gain.
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
PATH = f"{R}/sweeps/w53_attract_knee/004_att16_seed=42/encoder_final.pt"
GAINS = [100.0, 300.0, 1000.0, 3000.0, 10000.0]

enc, cfg_m, gain0, fwhm, _ = load_probe_encoder(PATH)
cfg = ProbeConfig(Npos=1716, env_size=20, n_worlds=3, n_envs_per_world=5,
                  k_values=(5,), steps=(1,), n_alias=2000, n_score_envs=3,
                  n_cont_samples=1, n_cont_annulus=0)
cfg.validate()
worlds = sample_worlds(cfg)
cells = local_cells(cfg.env_size)
rng0 = np.random.RandomState(0)


def corner(z):
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    b = np.sign(z)
    return float(np.mean(np.sum(z * b, 1) / (np.linalg.norm(z, 1 and 1, keepdims=False)
                                             * np.linalg.norm(b, axis=1))))


def cos_bin(z):
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    b = np.sign(z) / np.sqrt(z.shape[1])
    return float(np.mean(np.sum(z * b, 1)
                         / (np.linalg.norm(z, axis=1) * np.linalg.norm(b, axis=1))))


print(f"L7-s42, trained gain {gain0:.0f}, pre-activation RMS ~0.021")
print(f"{'infer gain':>11s} {'gain*RMS':>9s} {'cos_bin':>8s} "
      f"{'fixed pt':>9s} {'B |err|':>9s} {'acc45':>8s}")
print("-" * 60)

gx = rng0.randint(0, 1716, size=64)
gy = rng0.randint(0, 1716, size=64)
codes = torch.from_numpy(grid_codes(list(cfg_m.lambdas), gx, gy, fwhm))
with torch.no_grad():
    rms = float(np.sqrt((enc.net(codes).numpy() ** 2).mean()))

for g in GAINS:
    enc.gain = g                                  # inference-time override
    field = Field(enc, list(cfg_m.lambdas), fwhm, g, cfg.Npos)

    with torch.no_grad():
        z = torch.nn.functional.normalize(torch.tanh(g * enc.net(codes)),
                                          dim=-1).numpy().astype(np.float64)
    cb = cos_bin(z)

    # Fixed point under the production dynamics, unit-norm storage.
    Zs = z[:25]
    W = (Zs.T @ Zs) / Zs.shape[1]
    np.fill_diagonal(W, 0.0)
    fp = []
    for k in range(Zs.shape[0]):
        x = Zs[k].copy()
        for _ in range(20):
            x = np.tanh(g * (W @ x))
            n = np.linalg.norm(x)
            x = x / n if n > 1e-12 else x
        fp.append(float(np.dot(x, Zs[k])))

    acc = GridAcc(cfg)
    for w in worlds:
        rngw = np.random.RandomState(w.seed)
        mem = build_memory(field, w, 5, cfg, rngw)
        for e in scored_envs(cfg, 5):
            qf, _c, _b = cell_q_field(field, w, e, mem, cfg)
            acc.add_env(qf[1], cells, w.specs[e].goal)
    j = acc.scalars.to_json()

    print(f"{g:11.0f} {g * rms:9.2f} {cb:8.4f} {np.median(fp):9.4f} "
          f"{j['abs_err_mean']['mean']:8.1f}° {j['acc45']['mean'] * 100:7.1f}%")

print("\nnav_p2 5.7 corner: cos_bin 0.954, fixed pt 0.976 at M=25.")
print("acc45 chance is 25%.")
