#!/usr/bin/env python3
"""An exactly translation-equivariant encoder on the grid code.

§7 established that ``Phi(x + a) = P_a Phi(x)`` for a fixed permutation ``P_a``,
that the group is Z11 x Z12 x Z13 per axis, and that it acts *transitively* on
the arena. An equivariant encoder therefore satisfies ``z(x) = rho(x) z_0`` and
so ``cos(z(x), z(y)) = k(y - x)`` -- similarity a function of the offset alone,
every reference identical, and ``r_min = r_median`` exactly.

The group is abelian, so its irreducible representations are one-dimensional
**characters**::

    chi_{p}(x) = exp(2*pi*i * (p1*x/11 + p2*x/12 + p3*x/13))

and an equivariant code is a stack of such characters with learnable
amplitudes. Nothing else is available: an equivariant linear map must commute
with rho, and for distinct characters that forces it to be a per-character
scaling and phase.

Where the characters come from, and why this is a function of Phi rather than
of x: module m's block is a smoothed one-hot at ``x mod lambda_m``, so its 2D
DFT coefficient at frequency ``(p, q)`` has modulus fixed by the smoothing and
phase exactly ``-2*pi*(p*y + q*x)/lambda_m``. Multiplying one coefficient from
each module gives a character of the full group. The moduli are independent of
position, so the construction is exactly equivariant, not approximately.

Two uses:

* ``build_gaussian(...)`` sets the amplitudes analytically to a target kernel
  width and trains nothing. That is the cheap test of §8.1's premise -- if
  ``r_min != r_median`` for this code the whole equivariance account is wrong.
* the amplitudes are ``nn.Parameter``s, so the same module trains under the
  campaign's contrastive loss to ask the separate question of whether that loss
  *finds* a good kernel from patches alone.

Frequencies are indexed by the integer ``m`` with ``omega = m / prod(lambdas)``
cycles per cell, since ``p1/11 + p2/12 + p3/13 = (156 p1 + 143 p2 + 132 p3)/1716``
for lambdas (11, 12, 13). Low spatial frequencies come from *near-cancelling*
combinations -- (1, -1, 0) gives 13, (0, 1, -1) gives 11 -- which is the beat
structure the modular code encodes position with.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def character_table(lambdas, p_max: int = 2, m_max: int | None = None):
    """Integer frequency triples and the position-frequency each induces.

    Returns ``(triples, m)`` where ``triples`` is ``(K, n_modules)`` of the
    per-module integer frequencies and ``m`` is ``(K,)`` with the induced
    frequency in units of ``1/prod(lambdas)`` cycles per cell.
    """
    lam = np.asarray(lambdas, dtype=np.int64)
    total = int(np.prod(lam))
    cofactor = total // lam                      # 156, 143, 132 for 11,12,13

    rng = range(-p_max, p_max + 1)
    triples, ms = [], []
    for p in itertools.product(rng, repeat=len(lam)):
        m = int(np.dot(cofactor, p)) % total
        m_signed = m - total if m > total // 2 else m       # nearest to zero
        if m_signed == 0 and any(p):
            continue                              # aliases of DC, no position info
        if m_max is not None and abs(m_signed) > m_max:
            continue
        triples.append(p)
        ms.append(m_signed)
    if not triples:
        raise ValueError("no characters survived the cutoff")
    order = np.argsort(np.abs(ms))
    return np.array(triples)[order], np.array(ms)[order]


class EquivariantCharacterEncoder(nn.Module):
    """Stack of grid-code characters with learnable amplitudes.

    ``forward`` returns a unit-norm code whose pairwise cosine depends only on
    the offset between positions, by construction.
    """

    def __init__(self, lambdas, p_max: int = 2, m_max: int | None = 120,
                 amp_init: torch.Tensor | None = None):
        super().__init__()
        self.lambdas = list(lambdas)
        triples, ms = character_table(lambdas, p_max=p_max, m_max=m_max)
        # (K, n_modules) frequencies, used for BOTH axes independently: a 2D
        # character is a product of an x-character and a y-character, so the
        # code carries every (row, col) pair from the 1D table.
        self.register_buffer("triples", torch.as_tensor(triples, dtype=torch.long))
        self.register_buffer("m", torch.as_tensor(ms, dtype=torch.long))
        K = len(triples)
        self.n_char = K * K                       # (row freq, col freq) pairs
        self.out_dim = 2 * self.n_char
        self.log_amp = nn.Parameter(
            torch.zeros(self.n_char) if amp_init is None else amp_init.log())

        offs, ranges = 0, []
        for l in self.lambdas:
            ranges.append((offs, offs + l * l, l))
            offs += l * l
        self.module_ranges = ranges

    def characters(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, n_char)`` complex character values, moduli independent of x."""
        per_module = []
        for (start, end, l), col in zip(self.module_ranges,
                                        self.triples.T):
            blk = x[:, start:end].view(-1, l, l).to(torch.float32)
            spec = torch.fft.fft2(blk)                        # (B, l, l)
            # gather the requested (row, col) frequency pair per character
            idx = (col % l)
            per_module.append(spec[:, idx][:, :, idx])        # (B, K, K)
        prod = per_module[0]
        for c in per_module[1:]:
            prod = prod * c
        return prod.reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor, gain: float | None = None) -> torch.Tensor:
        c = self.characters(x)
        a = self.log_amp.exp().to(c.real.dtype)
        z = torch.cat([c.real * a, c.imag * a], dim=-1)
        return F.normalize(z, p=2, dim=-1)


def build_gaussian(lambdas, sigma: float, p_max: int = 2,
                   m_max: int | None = 120) -> EquivariantCharacterEncoder:
    """Amplitudes set analytically so the kernel is a Gaussian of width sigma.

    A stationary kernel's Fourier amplitudes are the square root of its power
    spectrum, and a Gaussian of width sigma in space has a Gaussian spectrum of
    width 1/(2*pi*sigma) in frequency. No training involved -- this exists to
    test whether equivariance alone gives r_min = r_median.
    """
    enc = EquivariantCharacterEncoder(lambdas, p_max=p_max, m_max=m_max)
    total = int(np.prod(lambdas))
    m = enc.m.to(torch.float64)
    # 2D frequency magnitude in cycles/cell for each (row, col) character pair
    fr = (m[:, None] / total) ** 2 + (m[None, :] / total) ** 2
    amp = torch.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * fr).sqrt()
    enc.log_amp.data = amp.reshape(-1).clamp_min(1e-12).log().float()
    return enc
