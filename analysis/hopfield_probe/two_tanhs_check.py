"""There are two tanh nonlinearities in this system, at wildly different scales.

Sec 1.3 and the whole "Hopfield is a linear associative memory" result say the
tanh is numerically INERT. Sec 10.20 says binarisation destroys the magnitudes.
Both are true, because they are different tanhs:

  RECALL     x <- normalize((1-a)x + a*tanh(beta * W x))     argument beta*(Wx)
  ENCODER    z <- normalize(tanh(gain * u)),  u = mlp(code)  argument gain*u

`(Wx)_i` is order `1/D^1.5` because W is a sum of K rank-1 terms over unit
vectors with the diagonal zeroed, so at beta = 100 the recall argument is ~4e-3
and `tanh(4e-3) = 4e-3` to five decimals -- inert, and saturating it needs
beta ~ D^1.5 = 32768.

`u` is an MLP output with no such 1/D suppression, so at gain = 100 the encoder
argument is order 1-10 and the tanh is already most of the way to a hypercube
corner. Nothing about the recall result transfers to it.

Measured here: both arguments, on the same checkpoint, plus how much of the
encoder's output is still graded rather than pinned at +-1.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from analysis.hopfield_probe.encode import grid_codes
from analysis.hopfield_probe.harness import (ProbeConfig, build_memory,
                                             load_probe_encoder, sample_worlds,
                                             tanh_argument)
from analysis.hopfield_probe.encode import Field

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
CK = f"{S}/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt"
NPOS = 1716


def main() -> None:
    enc, ecfg, _g, fwhm, _h = load_probe_encoder(CK, fwhm_fallback=0.25)
    rng = np.random.RandomState(0)
    gx = rng.randint(64, NPOS - 64, size=400)
    gy = rng.randint(64, NPOS - 64, size=400)
    codes = torch.from_numpy(grid_codes(list(ecfg.lambdas), gx, gy, fwhm))

    with torch.no_grad():
        u = enc.net(codes).numpy()          # pre-nonlinearity activations
    a = np.abs(u).ravel()
    print(f"pre-nonlinearity |u|        mean {a.mean():.4f}   "
          f"p50 {np.percentile(a, 50):.4f}   p99 {np.percentile(a, 99):.4f}")

    print("\n=== the ENCODER's tanh:  z = tanh(gain * u) ===")
    print(f"  {'gain':>8s}{'|arg| p50':>12s}{'|arg| p99':>12s}"
          f"{'|arg|<1':>10s}{'|tanh| p50':>13s}{'cos to corner':>15s}")
    for g in (1.0, 10.0, 100.0, 1000.0, 1e6):
        arg = g * u
        t = np.tanh(arg)
        tn = t / np.linalg.norm(t, axis=1, keepdims=True)
        b = np.sign(t) / np.sqrt(t.shape[1])
        cos_bin = float(np.mean((tn * b).sum(1) / np.linalg.norm(b, axis=1)))
        aa = np.abs(arg).ravel()
        print(f"  {g:>8.0f}{np.percentile(aa, 50):>12.3f}"
              f"{np.percentile(aa, 99):>12.1f}"
              f"{np.mean(aa < 1):>9.1%}"
              f"{np.percentile(np.abs(t).ravel(), 50):>13.4f}"
              f"{cos_bin:>15.4f}")

    print("\n=== the HOPFIELD's tanh:  tanh(beta * W x) ===")
    field = Field(enc, list(ecfg.lambdas), fwhm, 100.0, NPOS)
    enc.gain = 100.0
    for beta in (100.0, 1e6):
        cfg = ProbeConfig(n_worlds=2, n_envs_per_world=20, env_size=20,
                          Npos=NPOS, k_values=(5,), steps=(1,), seed=0,
                          beta_override=beta)
        w = sample_worlds(cfg)[0]
        mem = build_memory(field, w, 5, cfg, np.random.RandomState(w.seed * 31 + 5))
        arg = np.abs(tanh_argument(mem, mem.Z, cfg))
        print(f"  {'beta':>8s} {beta:>10.0f}   |arg| p50 "
              f"{np.percentile(arg, 50):.3e}   p99 {np.percentile(arg, 99):.3e}"
              f"   |arg|<0.01 at {np.mean(arg < 0.01):.1%}")

    print(f"\n  saturation threshold for the recall loop is beta ~ D^1.5 = "
          f"{int(1024 ** 1.5)}")


if __name__ == "__main__":
    main()
