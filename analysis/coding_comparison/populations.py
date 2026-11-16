"""One adapter from a spec string to a population of model units.

A population is a function from scaffold cells ``(x, y)`` to one activity
vector per cell. Every measure in this package reads populations only through
``Population.code``, so adding an encoder is one line: a new entry in
``POPULATIONS`` (or ``--pop name=spec`` on the command line), where ``spec`` is

``grid``, ``untrained``, ``hippocampus``, ``ideal_rot``, ``ideal_sparse``
    the tuning viewer's recipes (``analysis/tuning_viewer/export.py``), rebuilt
    here from the same functions and constants;
``ideal:r=16,...``
    an analytic ideal encoder (``hopfield_probe/ideal_encoder.py``);
any other string
    a checkpoint path, loaded by ``hopfield_probe.harness.load_probe_encoder``,
    which reads ``gain`` and ``fwhm_ratio`` from the checkpoint itself.

The scaffold is ``Npos x Npos`` (1716 for lambdas 11, 12, 13) and wraps, so
coordinates are reduced mod ``Npos``. Nothing here builds the full field: a
caller encodes only the cells it needs, and ``code`` works in chunks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable

import numpy as np
import torch

from analysis.hopfield_probe.encode import encode_positions, grid_codes
from analysis.hopfield_probe.harness import load_probe_encoder
from analysis.tuning_viewer.export import (
    ENCODER, HPC, IDEAL, ROT, SPARSE, draw_frequencies_nmin, draw_wpg,
    hadamard_rotation, hpc_rates)

# The tuning viewer's populations, in its order. ``trained`` is the cosyne
# att0.5 checkpoint; the grid code is smoothed at that checkpoint's fwhm.
POPULATIONS: dict[str, str] = {
    "grid": "grid",
    "ideal": IDEAL,
    "ideal_rot": "ideal_rot",
    "ideal_sparse": "ideal_sparse",
    "trained": ENCODER,
    "untrained": "untrained",
    "hippocampus": "hippocampus",
}

LABELS = {
    "grid": "Grid code",
    "ideal": "Ideal encoder",
    "ideal_rot": "Ideal, rotated",
    "ideal_sparse": "Sparse ReLU (10%)",
    "trained": "Trained encoder",
    "untrained": "Untrained encoder",
    "hippocampus": "Hippocampus (Vector-HaSH)",
}


@dataclass
class Population:
    """A named code: ``code(xs, ys)`` is ``(len(xs), n_units)`` float32."""

    name: str
    spec: str
    label: str
    npos: int
    n_units: int
    signed: bool
    encode: Callable[[np.ndarray, np.ndarray], np.ndarray] = field(repr=False)
    meta: dict = field(default_factory=dict)

    def code(self, xs, ys, *, chunk: int = 8192) -> np.ndarray:
        xs = np.mod(np.asarray(xs, dtype=np.int64).reshape(-1), self.npos)
        ys = np.mod(np.asarray(ys, dtype=np.int64).reshape(-1), self.npos)
        out = np.empty((xs.size, self.n_units), dtype=np.float32)
        for s in range(0, xs.size, chunk):
            out[s:s + chunk] = self.encode(xs[s:s + chunk], ys[s:s + chunk])
        return out


def _reference_encoder():
    """The trained checkpoint: lambdas, fwhm and gain come from it, never from defaults."""
    enc, cfg, gain, fwhm, header = load_probe_encoder(ENCODER)
    enc.eval()
    return enc, cfg, float(gain), float(fwhm), header


def _torch_encoder(enc, lambdas, fwhm) -> Callable:
    def f(xs, ys):
        return encode_positions(enc, lambdas, fwhm, xs, ys)
    return f


def make_population(name: str, spec: str | None = None) -> Population:
    """Build the population ``name`` from ``spec`` (default ``POPULATIONS[name]``)."""
    spec = POPULATIONS.get(name, name) if spec is None else spec
    label = LABELS.get(name, name)
    torch.set_grad_enabled(False)

    if spec == "grid":
        _, cfg, _, fwhm, _ = _reference_encoder()
        lambdas = [int(l) for l in cfg.lambdas]
        npos = int(np.prod(lambdas))
        return _finish(name, spec, label, npos,
                       lambda xs, ys: grid_codes(lambdas, xs, ys, fwhm),
                       {"lambdas": lambdas, "fwhm_ratio": fwhm})

    if spec == "hippocampus":
        _, cfg, _, _, _ = _reference_encoder()
        lambdas = [int(l) for l in cfg.lambdas]
        npos = int(np.prod(lambdas))
        ng = sum(l * l for l in lambdas)
        W = draw_wpg(HPC["Np"], ng, HPC["c"], HPC["seed"])
        return _finish(name, spec, label, npos,
                       lambda xs, ys: hpc_rates(W, lambdas, HPC["thresh"], xs, ys),
                       dict(HPC, lambdas=lambdas))

    if spec == "untrained":
        _, cfg, _, fwhm, _ = _reference_encoder()
        lambdas = [int(l) for l in cfg.lambdas]
        torch.manual_seed(0)
        from encoder_training.models import create_encoder
        enc = create_encoder(cfg, "cpu").eval()
        return _finish(name, spec, label, int(np.prod(lambdas)),
                       _torch_encoder(enc, lambdas, fwhm),
                       {"lambdas": lambdas, "fwhm_ratio": fwhm, "gain": float(enc.gain),
                        "torch_seed": 0})

    if spec in ("ideal_rot", "ideal_sparse"):
        ideal, _, _, _, _ = load_probe_encoder(IDEAL)
        npos = int(ideal.npos)
        freqs = ideal.freqs.cpu().numpy().astype(np.int64)
        amp = float(ideal.amp.cpu().numpy()[0])
        D = 2 * len(freqs)
        signs = np.random.default_rng(ROT["seed"]).choice([-1, 1], size=(ROT["rounds"], D))
        Q = hadamard_rotation(signs).astype(np.float64)
        if spec == "ideal_rot":
            def f(xs, ys):
                return (ideal.closed_form(xs, ys) @ Q.T).astype(np.float32)
            return _finish(name, spec, label, npos, f, dict(ROT, ideal=IDEAL))
        # export.py's sparse recipe, line for line: the r = 23.2 draw, the same
        # rotation, a threshold for 10% active, then unit length at each cell.
        sfreqs = draw_frequencies_nmin(SPARSE["r"], len(freqs), ideal.seed, npos, 0)
        thresh = NormalDist().inv_cdf(1 - SPARSE["active"]) / math.sqrt(D)

        def g(xs, ys):
            p = np.stack([xs, ys], 1).astype(np.int64)
            th = np.mod(p @ sfreqs.T, npos) * 2 * np.pi / npos
            zs = np.concatenate([np.cos(th), np.sin(th)], 1) * amp
            y = np.maximum(zs @ Q.T - thresh, 0)
            n = np.linalg.norm(y, axis=1, keepdims=True)
            return (y / np.where(n > 0, n, 1.0)).astype(np.float32)
        return _finish(name, spec, label, npos, g, dict(SPARSE, thresh=thresh))

    # An analytic ideal spec or a checkpoint path.
    enc, cfg, gain, fwhm, header = load_probe_encoder(spec)
    enc.eval()
    lambdas = [int(l) for l in cfg.lambdas]
    npos = int(np.prod(lambdas))
    f = _torch_encoder(enc, lambdas, fwhm)
    if hasattr(enc, "closed_form") and not getattr(enc, "binary", False):
        # Same numbers as the grid-code route (checked below), without the decode.
        rng = np.random.default_rng(0)
        xs, ys = rng.integers(0, npos, 512), rng.integers(0, npos, 512)
        diff = float(np.abs(enc.closed_form(xs, ys) - f(xs, ys)).max())
        if diff > 1e-5:
            raise RuntimeError(f"{spec}: closed form differs from the encoder by {diff:.2e}")
        f = (lambda xs, ys: enc.closed_form(xs, ys).astype(np.float32))
    return _finish(name, spec, label, npos, f,
                   {"lambdas": lambdas, "fwhm_ratio": fwhm, "gain": gain,
                    "header": header})


def _finish(name, spec, label, npos, f, meta) -> Population:
    """Probe the unit count and sign from 4096 random cells."""
    rng = np.random.default_rng(12345)
    xs, ys = rng.integers(0, npos, 4096), rng.integers(0, npos, 4096)
    z = np.asarray(f(xs, ys))
    return Population(name=name, spec=spec, label=label, npos=int(npos),
                      n_units=int(z.shape[1]), signed=bool((z < 0).any()),
                      encode=f, meta=meta)


__all__ = ["POPULATIONS", "LABELS", "Population", "make_population"]
