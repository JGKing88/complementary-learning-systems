"""The ideal encoder as a hand-designed network with explicit weights.

``IdealEncoder`` computes the ideal code with array operations. This module
writes the same computation as layers, so it reads as an architecture:

    grid code (434)
      -> Linear 434 -> 12, fixed      readout: cos and sin of each module's
                                      phase, per axis (the j = 1 wave, the
                                      strongest one in the smoothed code)
      -> atan2 on each pair (12 -> 6) the six phases phi_{m, axis}
      -> Linear 6 -> n_freq, no bias  INTEGER weights: row i holds wave i's
                                      harmonics (j_{11,x}, j_{11,y}, ..., j_{13,y})
                                      from the Chinese remainder theorem
      -> cos, sin, / sqrt(n_freq)     the 2 n_freq-dim code, unit norm

Why the integers matter. ``atan2`` returns angles in (-pi, pi], so each
phase jumps by 2 pi once per module period (where it crosses pi, e.g. between
x mod 11 = 5 and 6). With
integer weights, each output's input then jumps by an integer multiple of
2 pi, which ``cos`` and ``sin`` cannot see, so the output is continuous and
periodic in every module. Non-integer weights would put a discontinuity at
every module wrap, every 11-13 cells.

``trainable_harmonics=True`` makes the 6 -> n_freq layer a free real-valued
parameter (initialised at the integers, or randomly with ``init="random"``),
for the question of whether training finds the integers on its own.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .ideal_encoder import DEFAULT_LAMBDAS, IdealModelConfig, crt_harmonics, \
    draw_frequencies


class Atan2Pairs(torch.nn.Module):
    """``(N, 2P)`` laid out ``(c0, s0, c1, s1, ...)`` -> ``(N, P)`` angles."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(x[:, 1::2], x[:, 0::2])


class CosSin(torch.nn.Module):
    """``theta (N, F)`` -> ``[cos theta, sin theta] / sqrt(F)``, unit norm."""

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=1) \
            / math.sqrt(theta.shape[1])


def phase_readout_weights(lambdas) -> np.ndarray:
    """``(4 M, sum l^2)``: rows ``(cos x, sin x, cos y, sin y)`` per module.

    Cell ``(i, k)`` of module ``l`` (block row ``i * l + k``; ``i`` the x
    phase, ``k`` the y phase) gets ``cos/sin(2 pi i / l)`` in the x rows and
    ``cos/sin(2 pi k / l)`` in the y rows: the j = 1 Fourier component of the
    module's activity, marginalised over the other axis.
    """
    lambdas = [int(l) for l in lambdas]
    W = np.zeros((4 * len(lambdas), sum(l * l for l in lambdas)))
    off = 0
    for m, l in enumerate(lambdas):
        ang = 2.0 * math.pi * np.arange(l) / l
        i, k = np.meshgrid(np.arange(l), np.arange(l), indexing="ij")
        i, k = i.ravel(), k.ravel()
        W[4 * m + 0, off:off + l * l] = np.cos(ang[i])
        W[4 * m + 1, off:off + l * l] = np.sin(ang[i])
        W[4 * m + 2, off:off + l * l] = np.cos(ang[k])
        W[4 * m + 3, off:off + l * l] = np.sin(ang[k])
        off += l * l
    return W


class IdealNet(torch.nn.Module):
    """Grid code ``(N, sum l^2)`` -> ideal code ``(N, 2 n_freq)``, as layers."""

    def __init__(self, r: float, n_freq: int = 512, seed: int = 0,
                 lambdas=DEFAULT_LAMBDAS, gain: float = 100.0, *,
                 trainable_harmonics: bool = False, init: str = "integer"):
        super().__init__()
        self.r, self.n_freq, self.seed = float(r), int(n_freq), int(seed)
        self.lambdas = [int(l) for l in lambdas]
        self.npos = int(np.prod(self.lambdas))
        self.gain = float(gain)

        W1 = phase_readout_weights(self.lambdas)
        self.readout = torch.nn.Linear(W1.shape[1], W1.shape[0], bias=False,
                                       dtype=torch.float64)
        with torch.no_grad():
            self.readout.weight.copy_(torch.from_numpy(W1))
        self.readout.weight.requires_grad_(False)
        self.phase = Atan2Pairs()

        freqs = draw_frequencies(self.r, self.n_freq, self.seed, self.npos)
        jx = crt_harmonics(freqs[:, 0], self.lambdas)          # (M, n_freq)
        jy = crt_harmonics(freqs[:, 1], self.lambdas)
        J = np.empty((self.n_freq, 2 * len(self.lambdas)))     # (F, 6)
        J[:, 0::2] = jx.T
        J[:, 1::2] = jy.T
        self.register_buffer("freqs", torch.from_numpy(freqs))
        self.register_buffer("integer_harmonics", torch.from_numpy(J))
        self.harmonics = torch.nn.Linear(J.shape[1], J.shape[0], bias=False,
                                         dtype=torch.float64)
        with torch.no_grad():
            if init == "integer":
                self.harmonics.weight.copy_(torch.from_numpy(J))
            elif init == "random":
                torch.nn.init.uniform_(self.harmonics.weight, -6.0, 6.0)
            elif init == "noisy":
                self.harmonics.weight.copy_(torch.from_numpy(J))
                self.harmonics.weight.add_(
                    torch.empty_like(self.harmonics.weight).uniform_(-0.3, 0.3))
            else:
                raise ValueError(f"init={init!r}")
        self.harmonics.weight.requires_grad_(bool(trainable_harmonics))
        self.init = init
        self.out = CosSin()
        self.in_dim = W1.shape[1]

    @property
    def out_dim(self) -> int:
        return 2 * self.n_freq

    def model_config(self) -> IdealModelConfig:
        return IdealModelConfig(lambdas=list(self.lambdas), out_dim=self.out_dim,
                                encoder_type="ideal_net", gain=self.gain)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        x = codes.to(torch.float64)
        theta = self.harmonics(self.phase(self.readout(x)))
        return self.out(theta).to(codes.dtype if codes.is_floating_point()
                                  else torch.float32)

    def integer_distance(self) -> float:
        """Max |w - round(w)| over the harmonic weights (0 = exact integers)."""
        w = self.harmonics.weight.detach()
        return float((w - torch.round(w)).abs().max())


# --- checkpoints and the probe-harness hook ----------------------------------

IDEALNET_PREFIX = "idealnet:"


def save_ideal_net(net: IdealNet, path: str, **extra) -> None:
    """Enough to rebuild the net exactly: its constructor args and layer 3."""
    torch.save({"kind": "ideal_net", "r": net.r, "n_freq": net.n_freq,
                "seed": net.seed, "lambdas": net.lambdas, "gain": net.gain,
                "init": getattr(net, "init", None),
                "harmonics": net.harmonics.weight.detach().cpu(), **extra}, path)


def load_ideal_net(path: str) -> tuple[IdealNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = IdealNet(r=ck["r"], n_freq=ck["n_freq"], seed=ck["seed"],
                   lambdas=ck["lambdas"], gain=ck.get("gain", 100.0))
    with torch.no_grad():
        net.harmonics.weight.copy_(ck["harmonics"].to(torch.float64))
    return net.eval(), ck


def is_idealnet_spec(path) -> bool:
    return isinstance(path, str) and path.startswith(IDEALNET_PREFIX)


def load_idealnet_encoder(spec: str, *, device="cpu", fwhm_override=None,
                          fwhm_fallback=None):
    """``harness.load_probe_encoder``'s return shape for ``idealnet:<ckpt>``."""
    path = spec[len(IDEALNET_PREFIX):]
    net, ck = load_ideal_net(path)
    net = net.to(device)
    fwhm = (fwhm_override if fwhm_override is not None else
            fwhm_fallback if fwhm_fallback is not None else 0.25)
    cfg = net.model_config()
    header = {
        "path": spec, "name": spec, "coverage": None, "n_patches": None,
        "patch_sizes": None, "gain": net.gain, "fwhm_ratio": float(fwhm),
        "fwhm_was_overridden": fwhm_override is not None,
        "lambdas": list(net.lambdas), "out_dim": net.out_dim,
        "hidden_dim": 0, "num_hidden_layers": 0,
        "encoder_type": cfg.encoder_type,
        "output_nonlinearity": cfg.output_nonlinearity,
        "n_params": int(net.harmonics.weight.numel()), "epoch": ck.get("epoch"),
        "val_nav_acc": None, "unique_radius": None,
        "ideal": {"r": net.r, "n_freq": net.n_freq, "seed": net.seed,
                  "init": ck.get("init"),
                  "integer_distance": net.integer_distance()},
    }
    return net, cfg, net.gain, float(fwhm), header
