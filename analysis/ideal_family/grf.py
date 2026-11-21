"""Gaussian random fields with Gaussian covariance, drawn directly by spectral synthesis.

These are the large-M limit of one rotated ideal unit, sampled on the same
integer lattice: on a periodic line or box of side ``L`` the field is a sum
over integer frequencies ``k`` with power ``proportional to exp(-(2 pi k / L)^2 r^2 / 2)``,
which is the limit of the encoder's own frequency draw along a scaffold loop.
Every sample has unit variance in expectation, so levels are in SDs.
"""
from __future__ import annotations

import math

import numpy as np


def _spectrum_1d(L: int, r: float) -> np.ndarray:
    k = np.fft.rfftfreq(L, d=1.0) * L               # 0 .. L/2
    S = np.exp(-0.5 * (2 * math.pi * k / L) ** 2 * r * r)
    full = S.copy()
    full[1:] *= 2.0                                   # +k and -k
    if L % 2 == 0:
        full[-1] = S[-1]                              # Nyquist counted once
    return S / full.sum()


def grf_1d(n: int, L: int, r: float, rng: np.random.Generator) -> np.ndarray:
    """``(n, L)`` periodic unit-variance fields with correlation ``exp(-d^2 / 2 r^2)``."""
    S = _spectrum_1d(L, r)
    c = (rng.standard_normal((n, S.size)) + 1j * rng.standard_normal((n, S.size))) * np.sqrt(S / 2)
    c[:, 0] = rng.standard_normal(n) * math.sqrt(S[0])
    if L % 2 == 0:
        c[:, -1] = rng.standard_normal(n) * math.sqrt(S[-1])
    return np.fft.irfft(c * L, n=L, axis=1)


def grf_2d(n: int, B: int, r: float, rng: np.random.Generator) -> np.ndarray:
    """``(n, B, B)`` periodic unit-variance fields with correlation ``exp(-|d|^2 / 2 r^2)``.

    The real part of a full complex synthesis with ``E|c_k|^2 = 2 S_k`` has
    covariance ``sum_k S_k cos(k . d)``, so no Hermitian bookkeeping is needed.
    """
    ky = np.fft.fftfreq(B, d=1.0) * B
    S = np.exp(-0.5 * (2 * math.pi / B) ** 2 * r * r * (ky[:, None] ** 2 + ky[None, :] ** 2))
    S = S / S.sum()
    c = (rng.standard_normal((n, B, B)) + 1j * rng.standard_normal((n, B, B))) * np.sqrt(S)
    return np.fft.ifft2(c * B * B, axes=(1, 2)).real


def arena_box(side: int, r: float) -> int:
    """A periodic box big enough that an arena crop does not see the wrap (side + 8 r)."""
    return int(math.ceil((side + 8 * r) / 32.0) * 32)


__all__ = ["grf_1d", "grf_2d", "arena_box"]
