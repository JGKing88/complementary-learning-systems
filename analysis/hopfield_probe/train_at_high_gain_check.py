"""Would TRAINING an encoder at gain 1e6 recover the direction field?

Sec 10.20's arm B evaluates a gain-100 encoder at gain 1e6, so the obvious
objection is that the encoder was never fitted to a binary output and a trained
one would do better. Two separate questions, and they have different answers:

**(1) Can it train at all?**  `z = tanh(g*u)` has gradient `g*sech^2(g*u)`.
Measured below across five gains: the fraction of units with a numerically
nonzero gradient, and the size of the gradient on the ones that survive.

**(2) Could any binary code do better?**  The failure is that displacement
accumulates as `sqrt(k)` (diffusive) rather than `k` (ballistic), so
`cos(z(x+k) - z(x), d_fwd)` decays. There is a ceiling on this for binary codes
and it does not depend on training:

    a translation-invariant binary code flips some average `m` coordinates per
    unit step, so after k steps the Hamming distance is AT MOST `k*m` -- with
    equality exactly when no coordinate ever flips back. Then
    `||z(x+k) - z(x)|| = 2*sqrt(H(k)/D)` grows as `sqrt(k)`, and the alignment
    with a one-cell difference is `cos = H(1)/sqrt(H(1)*H(k)) = 1/sqrt(k)`.

    Ballistic growth would need `H(k) ~ k^2`, i.e. later steps flipping more
    coordinates than earlier ones, which translation invariance forbids.

So `1/sqrt(k)` is the best any binary code can do, trained or not. Measured
below: how close the binarised encoder already sits to it, via `H(k)/(k*H(1))`
-- 1.0 means no coordinate ever flips back, i.e. the code is already optimal.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from analysis.hopfield_probe.encode import Field, grid_codes
from analysis.hopfield_probe.harness import load_probe_encoder

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
CK = f"{S}/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt"
NPOS, N_SAMPLE = 1716, 1200
KS = (1, 2, 4, 8, 16, 32, 64)


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-30)


def gradient_health() -> None:
    """Does a gradient survive the output nonlinearity at each gain?"""
    enc, ecfg, _g, fwhm, _h = load_probe_encoder(CK, fwhm_fallback=0.25)
    rng = np.random.RandomState(0)
    gx = rng.randint(64, NPOS - 64, size=64)
    gy = rng.randint(64, NPOS - 64, size=64)
    codes = torch.from_numpy(grid_codes(list(ecfg.lambdas), gx, gy, fwhm))

    print("=== (1) can it train? gradient through tanh(g*u) ===")
    print(f"  {'gain':>8s}{'|grad| mean':>14s}{'|grad| max':>13s}"
          f"{'units with grad':>18s}")
    for g in (100.0, 300.0, 1e3, 1e4, 1e6):
        for p in enc.parameters():
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad = None
        enc.gain = g
        out = enc(codes)
        # Any scalar loss will do; the question is whether d loss / d params is
        # numerically nonzero, not what the loss is.
        out.pow(2).sum().backward()
        gr = torch.cat([p.grad.reshape(-1) for p in enc.parameters()
                        if p.grad is not None]).abs()
        print(f"  {g:>8.0f}{gr.mean():>14.3e}{gr.max():>13.3e}"
              f"{(gr > 0).float().mean().item():>17.1%}")


def binary_ceiling() -> None:
    """How close is the binarised code to the best a binary code can be?

    Also runs the SIGN statistics at gain 100, where they are not what the code
    uses. If the two gains walk the same sign trajectory, then binarising is not
    losing sign information at all -- it is discarding the magnitudes, and the
    magnitudes are the whole of the difference.
    """
    rng = np.random.RandomState(0)
    gx = rng.randint(64, NPOS - 64, size=N_SAMPLE)
    gy = rng.randint(64, NPOS - 64, size=N_SAMPLE)

    print("\n=== (2) is there room above it? the binary-code ceiling ===")
    print(f"  {'k cells north':<26s}" + "".join(f"{k:>8d}" for k in KS))
    for gain in (1e6, 100.0):
        enc, ecfg, _g, fwhm, _h = load_probe_encoder(CK, fwhm_fallback=0.25)
        enc.gain = gain
        field = Field(enc, list(ecfg.lambdas), fwhm, gain, NPOS)

        base = field.encode(gx, gy)
        z0 = np.sign(base)
        d1 = unit(unit(field.encode(gx, gy + 1)) - unit(base))
        h1m = float(np.mean(np.mean(
            z0 != np.sign(field.encode(gx, gy + 1)), axis=1)))

        hk, al, nk = [], [], []
        for k in KS:
            raw = field.encode(gx, gy + k)
            hk.append(float(np.mean(np.mean(z0 != np.sign(raw), axis=1))))
            d = unit(raw) - unit(base)
            nk.append(float(np.mean(np.linalg.norm(d, axis=1))))
            al.append(float(np.mean((unit(d) * d1).sum(1))))

        print(f"  -- gain {gain:g} --")
        print(f"  {'H(k) / (k * H(1))':<26s}"
              + "".join(f"{v / (k * h1m):>8.3f}" for k, v in zip(KS, hk)))
        print(f"  {'||dk|| / (k * ||d1||)':<26s}"
              + "".join(f"{v / (k * nk[0]):>8.3f}" for k, v in zip(KS, nk)))
        print(f"  {'measured cos(dk, d1)':<26s}"
              + "".join(f"{v:>8.3f}" for v in al))
        print(f"  {'1/sqrt(k)':<26s}"
              + "".join(f"{1 / np.sqrt(k):>8.3f}" for k in KS))
        print(f"  {'measured x sqrt(k)':<26s}"
              + "".join(f"{v * np.sqrt(k):>8.3f}" for k, v in zip(KS, al)))


if __name__ == "__main__":
    gradient_health()
    binary_ceiling()
