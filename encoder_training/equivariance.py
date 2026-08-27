#!/usr/bin/env python3
"""How close is the learned code to being a function of the *offset* alone?

The grid code is exactly translation-equivariant by construction. ``Phi(x)`` is
a one-hot encoding of ``(x mod 11, x mod 12, x mod 13)`` per axis, smoothed by a
convolution on the torus, so translating by ``a`` cyclically permutes each
module's index and smoothing commutes with it::

    Phi(x + a) = P_a Phi(x)          P_a a fixed permutation matrix

The group is Z11 x Z12 x Z13 per axis -- order 1716^2 -- and it acts
*transitively* on the arena. The whole arena is one orbit.

If the encoder were equivariant, ``f(P_a Phi) = rho(a) f(Phi)`` with rho
orthogonal, transitivity would give ``z(x) = rho(x) z_0`` and, since the group
is abelian,

    cos(z(x), z(y)) = cos(z_0, rho(y - x) z_0) = k(y - x)

-- similarity would depend on the offset and nothing else. Every reference
position would be identical, training on one patch would determine the kernel
everywhere, and ``r_min`` would equal ``r_median``.

It is not free, because a generic MLP has no equivariance bias and has to learn
it. THAT FAILURE IS WHAT THIS MEASURES. For each offset delta, sample many pairs
``(x, x + delta)`` from different absolute positions and report the spread of
``cos(z(x), z(x + delta))``. Zero spread means equivariant; the spread is the
equivariance residual.

Three predictions this was written to test, recorded before it was run:

1. Spread grows with ``|delta|``, sharply beyond the training patch diagonal --
   a patch of side s only ever exhibits offsets up to ``s * sqrt(2)``, so beyond
   that the kernel is unconstrained even under perfect equivariance.
2. Spread correlates with the ``r_median - r_min`` gap across encoders, because
   that gap *is* the failure of similarity to be a function of delta alone.
3. sm100 should be more equivariant at small delta than lo_mixtop despite a
   worse decay, which would explain how it wins while losing on both factors of
   §4.4b's law.

Usage::

    python -m encoder_training.equivariance <ckpt.pt> [more.pt ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from encoder_training.eval_unique_radius import grid_code_batch, npos_for

# Offsets to probe: a log-ish ladder, each as (dy, dx). Axis-aligned and
# diagonal both, because an anisotropic code fails them differently.
DEFAULT_OFFSETS = [2, 5, 10, 20, 35, 50, 71, 100, 141, 200, 283, 400]


def _encode(encoder, lambdas, gain, ys, xs, device, fwhm_ratio, batch=8192):
    out = []
    for i in range(0, len(ys), batch):
        phi = torch.as_tensor(
            grid_code_batch(lambdas, ys[i:i + batch], xs[i:i + batch],
                            fwhm_ratio),
            device=device)
        with torch.no_grad():
            z = encoder(phi, gain)
        out.append(torch.nn.functional.normalize(z, dim=-1))
    return torch.cat(out)


def equivariance_profile(encoder, ckpt, lambdas, gain, *, n_pairs: int,
                         device: str, fwhm_ratio: float, seed: int = 0,
                         offsets=DEFAULT_OFFSETS) -> None:
    npos = npos_for(lambdas)
    rng = np.random.default_rng(seed)

    y0s, x0s, sizes = ckpt.get("y0s"), ckpt.get("x0s"), ckpt.get("sizes")
    diag = max(sizes) * np.sqrt(2) if sizes else float("nan")
    inside = None
    if y0s:
        inside = np.zeros((npos, npos), dtype=bool)
        for y0, x0, s in zip(y0s, x0s, sizes):
            inside[y0:y0 + s, x0:x0 + s] = True

    print(f"\nequivariance: spread of cos at fixed offset "
          f"({n_pairs} pairs per offset)")
    if sizes:
        print(f"   largest training patch {max(sizes)} -> diagonal {diag:.0f} "
              f"cells; beyond that no within-env pair ever existed")
    print(f"   {'|d|':>5} {'dir':>5} {'mean cos':>9} {'sd':>7} {'p5':>7} "
          f"{'p95':>7} {'range':>7}  {'both in patch':>13}")

    for d in offsets:
        for name, (dy, dx) in (("axis", (0, d)),
                               ("diag", (int(round(d / np.sqrt(2))),) * 2)):
            ys = rng.integers(0, npos, size=n_pairs)
            xs = rng.integers(0, npos, size=n_pairs)
            ys2 = (ys + dy) % npos
            xs2 = (xs + dx) % npos
            za = _encode(encoder, lambdas, gain, ys, xs, device, fwhm_ratio)
            zb = _encode(encoder, lambdas, gain, ys2, xs2, device, fwhm_ratio)
            cos = (za * zb).sum(-1).cpu().numpy()
            frac = (float(np.mean(inside[ys, xs] & inside[ys2, xs2]))
                    if inside is not None else float("nan"))
            eff = int(round(np.hypot(dy, dx)))
            print(f"   {eff:>5} {name:>5} {cos.mean():>9.4f} {cos.std():>7.4f} "
                  f"{np.percentile(cos, 5):>7.4f} "
                  f"{np.percentile(cos, 95):>7.4f} "
                  f"{cos.max() - cos.min():>7.4f}  {frac:>12.1%}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", type=Path)
    p.add_argument("--n_pairs", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from encoder_training.train import load_encoder
    for path in args.ckpts:
        encoder, ckpt = load_encoder(str(path), device=args.device)
        lam = list(ckpt["model_config"]["lambdas"])
        fwhm = float((ckpt.get("train_config") or {}).get("fwhm_ratio", 0.25))
        print(f"\n{'=' * 78}\n{path}\n{'=' * 78}")
        equivariance_profile(encoder, ckpt, lam, float(ckpt["gain"]),
                             n_pairs=args.n_pairs, device=args.device,
                             fwhm_ratio=fwhm, seed=args.seed)


if __name__ == "__main__":
    main()
