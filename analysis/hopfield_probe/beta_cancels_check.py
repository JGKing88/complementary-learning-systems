"""In what sense is the recall tanh "inert"? Beta cancels -- until it does not.

Saying "the tanh is inert" is loose, and it invites the obvious objection that
beta = 1e6 plainly changes everything. The precise claim is stronger and
narrower.

The update (alpha = 1, the production value) is

    x <- normalize(tanh(beta * W x))

If `|beta * (Wx)_i| << 1` then `tanh(v) = v` and the step is

    x <- normalize(beta * W x) = normalize(W x)

and **beta divides out of the normalisation exactly**. Not approximately: every
beta below the knee produces the identical trajectory, so beta is not a weak
knob there, it is a no-op. What breaks the cancellation is tanh bending, which
needs `|beta * (Wx)_i| ~ 1`; with `(Wx)_i ~ D^-1.5` that is `beta ~ D^1.5`.

Measured here: recall from the stored patterns across a beta ladder spanning
seven decades, each output compared against the beta = 1 output. The prediction
is a step function -- identical to float precision up to the knee, then moving.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (ProbeConfig, build_memory,
                                             load_probe_encoder,
                                             recall_trajectory, sample_worlds,
                                             tanh_argument)

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
CK = f"{S}/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt"
NPOS, K = 1716, 5
BETAS = (1.0, 10.0, 100.0, 1e3, 1e4, 32768.0, 1e5, 1e6)


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-30)


def main() -> None:
    enc, ecfg, _g, fwhm, _h = load_probe_encoder(CK, fwhm_fallback=0.25)
    enc.gain = 100.0
    field = Field(enc, list(ecfg.lambdas), fwhm, 100.0, NPOS)

    ref1, ref15 = None, None
    print(f"D^1.5 = {1024 ** 1.5:.0f} is where beta*(Wx) reaches order 1\n")
    print(f"  {'beta':>9s}{'|beta*Wx| p50':>15s}{'cos to beta=1':>16s}"
          f"{'cos_self':>11s}{'cos(s15, s1)':>14s}")
    for beta in BETAS:
        cfg = ProbeConfig(n_worlds=4, n_envs_per_world=20, env_size=20,
                          Npos=NPOS, k_values=(K,), steps=(1, 15), seed=0,
                          beta_override=beta)
        outs1, outs15, selfs, args = [], [], [], []
        for w in sample_worlds(cfg):
            mem = build_memory(field, w, K, cfg,
                               np.random.RandomState(w.seed * 31 + K))
            Z = unit(mem.Z)
            traj = recall_trajectory(mem, Z, (1, 15), cfg)
            x1, x15 = unit(traj[1]), unit(traj[15])
            outs1.append(x1)
            outs15.append(x15)
            selfs.append((x1 * Z).sum(1))
            args.append(np.abs(tanh_argument(mem, Z, cfg)))
        x1 = np.concatenate(outs1)
        x15 = np.concatenate(outs15)
        if ref1 is None:
            ref1 = x1
        cos_ref = float(np.mean((x1 * ref1).sum(1)))
        print(f"  {beta:>9.0f}{np.percentile(np.concatenate(args), 50):>15.3e}"
              f"{cos_ref:>16.6f}"
              f"{float(np.mean(np.concatenate(selfs))):>11.4f}"
              f"{float(np.mean((x15 * x1).sum(1))):>14.4f}")


if __name__ == "__main__":
    main()
