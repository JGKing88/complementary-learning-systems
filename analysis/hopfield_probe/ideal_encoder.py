"""An analytic "ideal" encoder: 2D random Fourier features, read off the grid code.

Target: a code whose similarity is a Gaussian kernel of scaffold displacement,

    <z(p), z(q)>  ~=  k(p - q) = exp(-|p - q|^2 / 2 r^2)      on the Npos^2 torus,

at D = 1024 and exactly unit norm. Random Fourier features give it: draw
``n_freq`` integer frequency vectors ``n = (n_x, n_y)`` with each component
``round(Normal(0, sigma))``, ``sigma = Npos / (2 pi r)`` (so ``omega = 2 pi n /
Npos`` has sd ``1/r``), reduced to ``(-Npos/2, Npos/2]``, and emit

    z(p) = n_freq^-1/2 [cos(omega_i . p), sin(omega_i . p)]_{i=1..n_freq}

Every row has norm exactly 1, and ``z(p).z(q) = mean_i cos(omega_i . (p - q))``,
whose expectation over the draw is ``k`` and whose far-field spread is
``1/sqrt(2 n_freq) = 1/sqrt(D)``.

It is computed FROM THE GRID CODE, not from (x, y), so it is a drop-in encoder
``(N, sum lambda^2) -> (N, 2 n_freq)`` for the probe harness:

1. per module ``m`` (period ``l``) and axis, decode the phase: marginalise the
   ``l x l`` block over the other axis, then take the circular mean over the
   ``l`` cells, ``phi = 2 pi (p mod l) / l``. The wrapped Gaussian bump
   (``encode.grid_codes``) is symmetric about the integer phase, so this is
   exact, not an estimate; the unsmoothed one-hot code decodes too.
2. By the Chinese remainder theorem, with ``c_m = Npos / l_m`` and harmonics
   ``j_m = n * c_m^{-1} mod l_m`` (for 11, 12, 13: ``j = (6n, -n, 7n)``),
   ``sum_m c_m j_m == n (mod Npos)``, so

       2 pi n p / Npos  ==  sum_m j_m phi_m        (mod 2 pi)

   per axis. The angle for frequency ``(n_x, n_y)`` is the sum over modules
   and both axes.
3. Emit ``cos`` and ``sin`` of it, over ``sqrt(n_freq)``.

``gain`` is carried only because ``run.py`` and ``controls.py`` read
``field.gain`` as the default Hopfield beta. It never touches the code (and
below beta ~ D^1.5 the recall divides beta out anyway -- THEORY Sec 1.3). Set
it to the reference encoder's so the recall regime is the same.

Layout convention: block row ``i * l + j`` with ``i`` the x phase, ``j`` the y
phase, as ``encode.grid_codes`` builds it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import torch

DEFAULT_LAMBDAS = (11, 12, 13)


def crt_harmonics(n: np.ndarray, lambdas) -> np.ndarray:
    """``(len(lambdas), *n.shape)`` int: ``j_m = n * (Npos/l_m)^{-1} mod l_m``."""
    lambdas = [int(l) for l in lambdas]
    npos = int(np.prod(lambdas))
    n = np.asarray(n, dtype=np.int64)
    out = []
    for l in lambdas:
        c = npos // l
        inv = pow(c % l, -1, l)                # needs pairwise-coprime lambdas
        out.append(np.mod(n * inv, l))
    return np.stack(out, axis=0)


def draw_frequencies(r: float, n_freq: int, seed: int, npos: int) -> np.ndarray:
    """``(n_freq, 2)`` int64 frequency vectors in ``(-npos/2, npos/2]``."""
    sigma = npos / (2.0 * math.pi * float(r))
    rng = np.random.RandomState(int(seed))
    n = np.rint(rng.normal(0.0, sigma, size=(int(n_freq), 2))).astype(np.int64)
    half = npos // 2
    # Reduce mod npos into (-half, half] (npos even here; odd works too).
    n = np.mod(n + half - 1, npos) - (half - 1)
    return n


@dataclass
class IdealModelConfig:
    """Duck-types the ``EncoderModelConfig`` fields the probe harness reads."""
    lambdas: list = field(default_factory=lambda: list(DEFAULT_LAMBDAS))
    out_dim: int = 1024
    encoder_type: str = "ideal_rff"
    output_nonlinearity: str = "none"
    hidden_dim: int = 0
    num_hidden_layers: int = 0
    gain: float = 100.0

    @property
    def in_dim(self) -> int:
        return sum(l * l for l in self.lambdas)


class IdealEncoder(torch.nn.Module):
    """Grid code ``(N, sum l^2)`` -> Gaussian-kernel RFF ``(N, 2 n_freq)``."""

    def __init__(self, r: float, n_freq: int = 512, seed: int = 0,
                 lambdas=DEFAULT_LAMBDAS, gain: float = 100.0,
                 weights: str | np.ndarray | None = None):
        super().__init__()
        # Optional per-frequency weights a_i >= 0 (e.g. a least-squares fit for
        # this draw): wave i is scaled by sqrt(a_i / sum a), so z.z' =
        # sum_i a_i cos(omega_i . Delta) / sum a. None = equal weights.
        self.weights_path = weights if isinstance(weights, str) else None
        if weights is None:
            a = np.ones(int(n_freq))
        else:
            a = np.load(weights) if isinstance(weights, str) else np.asarray(weights)
            a = np.clip(np.asarray(a, dtype=np.float64), 0.0, None)
            if a.shape != (int(n_freq),):
                raise ValueError(f"weights shape {a.shape} != ({n_freq},)")
        self.register_buffer("amp", torch.from_numpy(np.sqrt(a / a.sum())))
        self.r = float(r)
        self.n_freq = int(n_freq)
        self.seed = int(seed)
        self.lambdas = [int(l) for l in lambdas]
        self.npos = int(np.prod(self.lambdas))
        self.gain = float(gain)

        freqs = draw_frequencies(self.r, self.n_freq, self.seed, self.npos)
        jx = crt_harmonics(freqs[:, 0], self.lambdas)       # (M, n_freq)
        jy = crt_harmonics(freqs[:, 1], self.lambdas)
        # Rows ordered (m0 x, m0 y, m1 x, m1 y, ...) to match _phases.
        J = np.empty((2 * len(self.lambdas), self.n_freq), dtype=np.float64)
        J[0::2] = jx
        J[1::2] = jy
        self.register_buffer("freqs", torch.from_numpy(freqs))
        self.register_buffer("harmonics", torch.from_numpy(J))

        offs, off = [], 0
        for l in self.lambdas:
            offs.append(off)
            off += l * l
        self._offsets = offs
        self.in_dim = off

    @property
    def out_dim(self) -> int:
        return 2 * self.n_freq

    def model_config(self) -> IdealModelConfig:
        return IdealModelConfig(lambdas=list(self.lambdas), out_dim=self.out_dim,
                                gain=self.gain)

    def _phases(self, codes: torch.Tensor) -> torch.Tensor:
        """``(N, 2M)`` float64 phases in radians, ``(m0 x, m0 y, m1 x, ...)``."""
        codes = codes.to(torch.float64)
        cols = []
        for l, off in zip(self.lambdas, self._offsets):
            block = codes[:, off:off + l * l].reshape(-1, l, l)   # [N, i=x, j=y]
            ang = 2.0 * math.pi * torch.arange(
                l, dtype=torch.float64, device=codes.device) / l
            c, s = torch.cos(ang), torch.sin(ang)
            for marg in (block.sum(dim=2), block.sum(dim=1)):     # x, then y
                cols.append(torch.atan2(marg @ s, marg @ c))
        return torch.stack(cols, dim=1)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[-1] != self.in_dim:
            raise ValueError(f"expected (N, {self.in_dim}) grid codes, got "
                             f"{tuple(codes.shape)}")
        theta = self._phases(codes) @ self.harmonics.to(codes.device)
        out = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        out = out * torch.cat([self.amp, self.amp]).to(out)
        return out.to(codes.dtype if codes.is_floating_point()
                      else torch.float32)

    def closed_form(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """``cos/sin(2 pi n . p / Npos) / sqrt(n_freq)`` straight from (x, y)."""
        n = self.freqs.cpu().numpy()
        p = np.stack([np.asarray(xs), np.asarray(ys)], axis=1).astype(np.int64)
        # Integer phase mod Npos first: exact, no float blow-up at large n.p.
        k = np.mod(p @ n.T, self.npos).astype(np.float64)
        theta = 2.0 * math.pi * k / self.npos
        amp = self.amp.cpu().numpy()
        return np.concatenate([np.cos(theta), np.sin(theta)], axis=1) \
            * np.concatenate([amp, amp])


# --- probe-harness hook ------------------------------------------------------
#
# ``load_probe_encoder`` treats a ``--ckpt`` of the form
#     ideal:r=4[,n_freq=512][,seed=0][,gain=100]
# as this encoder, so every existing script that takes a checkpoint path runs
# it unchanged.

IDEAL_PREFIX = "ideal:"


def is_ideal_spec(path) -> bool:
    return isinstance(path, str) and path.startswith(IDEAL_PREFIX)


def parse_ideal_spec(spec: str) -> dict:
    body = spec[len(IDEAL_PREFIX):]
    kw: dict = {}
    for part in filter(None, re.split(r"[,;]", body)):
        k, _, v = part.partition("=")
        k = k.strip()
        if k == "weights":
            kw[k] = v.strip()
            continue
        if k not in ("r", "n_freq", "seed", "gain"):
            raise ValueError(f"unknown ideal-encoder key {k!r} in {spec!r}")
        kw[k] = int(v) if k in ("n_freq", "seed") else float(v)
    if "r" not in kw:
        raise ValueError(f"{spec!r} needs r=<kernel width>")
    return kw


def load_ideal_encoder(spec: str, *, device="cpu",
                       fwhm_override=None, fwhm_fallback=None):
    """Same return shape as ``harness.load_probe_encoder``.

    The phase decode is exact at any smoothing width, so ``fwhm_ratio`` only
    decides which grid code is fed in; it defaults to production's 0.25.
    """
    kw = parse_ideal_spec(spec)
    enc = IdealEncoder(**kw).to(device).eval()
    fwhm = (fwhm_override if fwhm_override is not None else
            fwhm_fallback if fwhm_fallback is not None else 0.25)
    cfg = enc.model_config()
    header = {
        "path": spec, "name": spec, "coverage": None, "n_patches": None,
        "patch_sizes": None, "gain": enc.gain, "fwhm_ratio": float(fwhm),
        "fwhm_was_overridden": fwhm_override is not None,
        "lambdas": list(enc.lambdas), "out_dim": enc.out_dim,
        "hidden_dim": 0, "num_hidden_layers": 0,
        "encoder_type": cfg.encoder_type,
        "output_nonlinearity": cfg.output_nonlinearity,
        "n_params": 0, "epoch": None, "val_nav_acc": None,
        "unique_radius": None,
        "ideal": {"r": enc.r, "n_freq": enc.n_freq, "seed": enc.seed,
                  "weights": enc.weights_path},
    }
    return enc, cfg, enc.gain, float(fwhm), header


__all__ = ["IdealEncoder", "IdealModelConfig", "crt_harmonics",
           "draw_frequencies", "is_ideal_spec", "load_ideal_encoder",
           "parse_ideal_spec"]
