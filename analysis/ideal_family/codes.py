"""The ideal family's codes, from the closed form, without the scaffold or the grid code.

``Family(r, M, seed)`` holds one frequency draw and one rotation:

* frequencies: ``hopfield_probe.ideal_encoder.draw_frequencies(r, M, seed, 1716)``,
  the draw ``ideal:r=..,n_freq=M,seed=..`` makes (and the viewer's
  ``draw_frequencies_nmin(..., 0)``);
* rotation: the viewer's ``Q = H D3 H D2 H D1 / D^1.5`` with sign rows
  ``default_rng(seed).choice([-1, 1], (3, D))``, applied by a fast
  Walsh-Hadamard transform. Seed 0 is the viewer's ``ROT``.

Variants, all as ``(positions, units)`` rates:

``native``        ``relu(z)``: plane-wave units, rectified (the suite's primary convention)
``rotated``       ``relu(Q z)``: the same threshold at 0, so ``p = 0.5``
``sparse``        ``relu(Q z - t / sqrt(D)) / |.|``: the viewer's sparse code, unit length per place
``sparse_raw``    ``relu(Q z - t / sqrt(D))`` without the per-place normalisation

``t = Phi^-1(1 - p)``. With ``r = 23.2, M = 512, seed = 0, p = 0.1`` the
``sparse`` variant is the viewer's ``ideal_sparse``; with ``r = 16`` and
``seed = 0`` the ``rotated`` variant before rectification is ``ideal_rot``.
``hopfield_nav/tests/test_ideal_family.py`` checks both against
``coding_comparison.populations``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from analysis.hopfield_probe.ideal_encoder import draw_frequencies

from . import theory as T

NPOS = 1716
ROT_ROUNDS = 3


def fwht(x: np.ndarray) -> np.ndarray:
    """Unnormalised Walsh-Hadamard transform along the last axis, Sylvester (natural) order."""
    x = np.array(x, dtype=np.float64 if x.dtype == np.float64 else np.float32, copy=True)
    n = x.shape[-1]
    lead = x.shape[:-1]
    h = 1
    while h < n:
        y = x.reshape(*lead, n // (2 * h), 2, h)
        a = y[..., 0, :].copy()
        b = y[..., 1, :]
        y[..., 0, :] += b
        y[..., 1, :] = a - b
        h *= 2
    return x


def rotation_signs(D: int, seed: int, rounds: int = ROT_ROUNDS) -> np.ndarray:
    return np.random.default_rng(seed).choice([-1, 1], size=(rounds, D))


def apply_rotation(z: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """``z @ Q.T`` for the viewer's ``Q = H D_k ... H D_1 / D^(k/2)`` (rows are places)."""
    D = z.shape[-1]
    y = z
    for s in signs:
        y = fwht(y * s.astype(y.dtype))
    return y / np.asarray(D ** (len(signs) / 2), dtype=y.dtype)


@dataclass
class Family:
    r: float
    M: int
    seed: int = 0
    npos: int = NPOS
    dtype: type = np.float64
    freqs: np.ndarray = field(init=False, repr=False)
    signs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.freqs = draw_frequencies(self.r, self.M, self.seed, self.npos).astype(np.int64)
        D = 2 * self.M
        if D & (D - 1):
            raise ValueError(f"D = 2M = {D} must be a power of 2 for the Hadamard rotation")
        self.signs = rotation_signs(D, self.seed)

    @property
    def D(self) -> int:
        return 2 * self.M

    @property
    def sd(self) -> float:
        """SD of one unit over the scaffold: ``1/sqrt(D)``."""
        return 1.0 / math.sqrt(self.D)

    def native(self, xs, ys) -> np.ndarray:
        """``(P, D)``: ``[cos, sin](2 pi (n . p mod Npos) / Npos) / sqrt(M)``, as ``IdealEncoder.closed_form``."""
        p = np.stack([np.asarray(xs), np.asarray(ys)], 1).astype(np.int64)
        th = np.mod(p @ self.freqs.T, self.npos).astype(np.float64) * (2 * math.pi / self.npos)
        out = np.empty((p.shape[0], self.D), dtype=self.dtype)
        a = 1.0 / math.sqrt(self.M)
        out[:, :self.M] = np.cos(th) * a
        out[:, self.M:] = np.sin(th) * a
        return out

    def rotated(self, z: np.ndarray) -> np.ndarray:
        return apply_rotation(z, self.signs)

    def threshold(self, p: float) -> float:
        """Absolute threshold for active fraction ``p``: ``Phi^-1(1 - p) / sqrt(D)`` (the viewer's)."""
        return T.t_of_p(p) * self.sd

    def realized_kernel_1d(self, lags: np.ndarray, axis: int = 0) -> np.ndarray:
        """``(1/M) sum_i cos(2 pi n_i,axis d / Npos)``: the draw's exact raw kernel along an axis."""
        n = self.freqs[:, axis].astype(np.float64)
        return np.cos(2 * math.pi * np.outer(lags, n) / self.npos).mean(axis=1)

    def raw_kernel_torus(self) -> np.ndarray:
        """``K(d)`` at every displacement on the scaffold torus, by one inverse FFT."""
        N = self.npos
        H = np.zeros((N, N))
        np.add.at(H, (self.freqs[:, 0] % N, self.freqs[:, 1] % N), 1.0)
        return np.fft.ifft2(H).real * (N * N / self.M)

    def native_rate_kernel_torus(self, kmax: int = 16) -> np.ndarray:
        """Anchor-averaged cosine kernel of the rectified native code on the whole torus.

        ``sum_k (a_k^2 / 2) mean_i cos(k omega_i . d) + a_0^2``, over ``1/4``,
        with the harmonics ``k n_i`` placed in one inverse FFT (``theory.native_kernel``).
        """
        N = self.npos
        H = np.zeros((N, N))
        a = T._A_RELU_COS
        for k in range(1, kmax + 1):
            if a[k] == 0:
                continue
            np.add.at(H, ((k * self.freqs[:, 0]) % N, (k * self.freqs[:, 1]) % N), a[k] ** 2 / 2)
        K = np.fft.ifft2(H).real * (N * N / self.M)
        return (K + a[0] ** 2) / 0.25


VARIANT_PS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)


def variant_names(ps=VARIANT_PS) -> list[str]:
    out = ["native", "rotated"]
    out += [f"sparse_p{p:g}" for p in ps]
    out += [f"sparse_raw_p{p:g}" for p in ps if p != 0.5]
    return out


def parse_variant(name: str) -> tuple[str, float | None]:
    if name in ("native", "rotated"):
        return name, (0.5 if name == "rotated" else None)
    kind, p = name.rsplit("_p", 1)
    return kind, float(p)


def rates(name: str, z: np.ndarray | None, y: np.ndarray | None, sd: float) -> np.ndarray:
    """Rates ``(P, D)`` of variant ``name`` from the native code ``z`` and its rotation ``y``.

    ``sd`` is the unit SD (``1/sqrt(D)`` for the encoder, 1 for unit-variance GRFs).
    """
    kind, p = parse_variant(name)
    if kind == "native":
        return np.maximum(z, 0)
    if kind == "rotated":
        return np.maximum(y, 0)
    s = np.maximum(y - T.t_of_p(p) * sd, 0)
    if kind == "sparse":
        n = np.linalg.norm(s, axis=1, keepdims=True)
        s = s / np.where(n > 0, n, 1.0)
    return s


def threshold_sd(name: str) -> float | None:
    """The variant's threshold in unit SDs (``t``), or None for the native code."""
    kind, p = parse_variant(name)
    return None if kind == "native" else T.t_of_p(p)


__all__ = ["NPOS", "Family", "fwht", "apply_rotation", "rotation_signs", "VARIANT_PS",
           "variant_names", "parse_variant", "rates", "threshold_sd"]
