"""The grid wave network (docs/EXPERIMENTS_IDEAL_ENCODER.md Sec 20).

    L0 grid code g (434)
    L1 per-module wave pairs (c, s) = g @ Wc, g @ Ws      fixed or learned, within one module
    L2 wave unit k: one L1 pair per module (random), multiplied
         u = c11 c12 - s11 s12,  v = s11 c12 + c11 s12
         C = u c13 - v s13,      S = v c13 + u s13          (C = cos Theta, S = sin Theta)
    L3 gain w_k from the unit's own slowness; code = [w C, w S] / norm

With exact L1 (``exact_l1``) the pair for module m and pattern j = (jx, jy) is
``c = sum_ab cos(2 pi (jx a + jy b)/l) g_ab / A_m(j) = cos theta``, theta = 2 pi
(jx x + jy y)/l, and unit k is the plane wave n = 156 j11 + 143 j12 + 132 j13
(mod 1716) per axis. L1 runs in float64: the fine patterns are divided by
A_m(j) ~ 3e-4 at bump width 0.25.

Multiplication modes: ``exact``; ``coef`` (each of the 8 +-1 coefficients of a
unit scaled by a fixed 1 + delta); ``quarter`` (xy = (f(x+y) - f(x-y)) / 4 with
f = |.|^p, exact at p = 2).

The probe loads a saved network with the spec ``gwn:<path.pt>``; its
``fwhm_ratio`` is the bump width L1 was built for.
"""
from __future__ import annotations

import math

import numpy as np
import torch

DEFAULT_LAMBDAS = (11, 12, 13)
NPOS = 1716
GWN_PREFIX = "gwn:"


def _bump(l: int, fwhm: float) -> np.ndarray:
    d = np.minimum(np.arange(l), l - np.arange(l))
    if fwhm <= 0:
        return (d == 0).astype(np.float64)
    sig = l * fwhm / (2 * math.sqrt(2 * math.log(2)))
    return np.exp(-0.5 * (d / sig) ** 2)


def bump_strength(l: int, fwhm: float, j: np.ndarray) -> np.ndarray:
    """``bhat(j) = sum_d B(d) cos(2 pi j d / l)``, the bump's strength at repeat j."""
    B = _bump(l, fwhm)
    d = np.arange(l)
    return (B[None, :] * np.cos(2 * np.pi * np.asarray(j)[:, None] * d[None, :] / l)).sum(1)


class L1:
    """(c, s) readout pairs: ``Wc``, ``Ws`` (Ng, P) float64, ``module`` (P,), and
    ``pattern`` (P, 2) for exact pairs (-1 for learned ones)."""

    def __init__(self, Wc, Ws, module, pattern=None, lambdas=DEFAULT_LAMBDAS):
        self.Wc = torch.as_tensor(Wc, dtype=torch.float64)
        self.Ws = torch.as_tensor(Ws, dtype=torch.float64)
        self.module = torch.as_tensor(module, dtype=torch.long)
        P = self.Wc.shape[1]
        self.pattern = (torch.full((P, 2), -1, dtype=torch.long) if pattern is None
                        else torch.as_tensor(pattern, dtype=torch.long))
        self.lambdas = [int(l) for l in lambdas]

    def pairs_of(self, m: int) -> torch.Tensor:
        return torch.nonzero(self.module == m).flatten()

    def to(self, device):
        self.Wc, self.Ws = self.Wc.to(device), self.Ws.to(device)
        self.module, self.pattern = self.module.to(device), self.pattern.to(device)
        return self


def exact_l1(fwhm: float, lambdas=DEFAULT_LAMBDAS) -> L1:
    """Every pattern j in Z_l^2 of every module, normalised to unit amplitude."""
    ng = sum(l * l for l in lambdas)
    Wc, Ws, mod, pat = [], [], [], []
    off = 0
    for m, l in enumerate(lambdas):
        jx, jy = np.meshgrid(np.arange(l), np.arange(l), indexing="ij")
        jx, jy = jx.ravel(), jy.ravel()
        A = bump_strength(l, fwhm, jx) * bump_strength(l, fwhm, jy)
        a, b = np.meshgrid(np.arange(l), np.arange(l), indexing="ij")
        ph = 2 * np.pi * (jx[None, :] * a.ravel()[:, None] + jy[None, :] * b.ravel()[:, None]) / l
        wc = np.zeros((ng, l * l)); ws = np.zeros((ng, l * l))
        wc[off:off + l * l] = np.cos(ph) / A[None, :]
        ws[off:off + l * l] = np.sin(ph) / A[None, :]
        Wc.append(wc); Ws.append(ws)
        mod.append(np.full(l * l, m)); pat.append(np.stack([jx, jy], 1))
        off += l * l
    return L1(np.concatenate(Wc, 1), np.concatenate(Ws, 1), np.concatenate(mod),
              np.concatenate(pat), lambdas)


def random_triples(l1: L1, n_units: int, seed: int) -> torch.Tensor:
    """``(n_units, 3)`` global pair indices, one uniformly random pair per module."""
    g = torch.Generator().manual_seed(int(seed))
    cols = []
    for m in range(len(l1.lambdas)):
        idx = l1.pairs_of(m).cpu()
        cols.append(idx[torch.randint(0, len(idx), (int(n_units),), generator=g)])
    return torch.stack(cols, 1)


def all_triples(l1: L1) -> torch.Tensor:
    a, b, c = (l1.pairs_of(m).cpu() for m in range(3))
    A, B, C = torch.meshgrid(a, b, c, indexing="ij")
    return torch.stack([A.reshape(-1), B.reshape(-1), C.reshape(-1)], 1)


def triple_frequencies(l1: L1, triples: torch.Tensor) -> torch.Tensor:
    """``(U, 2)`` plane-wave frequency n in (-858, 858] for exact pairs."""
    pat = l1.pattern.cpu()[triples.cpu()]                      # (U, 3, 2)
    c = torch.tensor([NPOS // l for l in l1.lambdas])
    n = (pat * c[None, :, None]).sum(1) % NPOS
    return torch.where(n > NPOS // 2, n - NPOS, n)


def multiply(c, s, mult="exact", param=None, coef=None):
    """C, S from per-module (c, s) lists of 3 tensors, (..., U) each."""
    if mult == "quarter":
        p = float(param)
        mul = lambda x, y: ((x + y).abs() ** p - (x - y).abs() ** p) / 4
    else:
        mul = lambda x, y: x * y
    k = coef if coef is not None else [1.0] * 8
    c0, c1, c2 = c
    s0, s1, s2 = s
    u = k[0] * mul(c0, c1) - k[1] * mul(s0, s1)
    v = k[2] * mul(s0, c1) + k[3] * mul(c0, s1)
    C = k[4] * mul(u, c2) - k[5] * mul(v, s2)
    S = k[6] * mul(v, c2) + k[7] * mul(u, s2)
    return C, S


def unit_waves(L_c, L_s, triples, mult="exact", param=None, coef=None):
    """L1 outputs (N, P) -> C, S (N, U) for ``triples`` (U, 3)."""
    c = [L_c[:, triples[:, m]] for m in range(3)]
    s = [L_s[:, triples[:, m]] for m in range(3)]
    kk = None if coef is None else [coef[:, i][None, :] for i in range(8)]
    return multiply(c, s, mult, param, kk)


GAIN_RULES = {
    "gauss": lambda v, r: torch.exp(-r * r * v / 2),
    "cut": lambda v, r: (v < 2 / r**2).to(v.dtype),
    "rational": lambda v, r: 1 / (1 + r * r * v / 2),
    "linear": lambda v, r: torch.clamp(1 - r * r * v / 4, min=0),
}


def prune(gains: torch.Tensor, mass: float = 0.995, cap: int = 6000) -> torch.Tensor:
    """Indices of the fewest units holding ``mass`` of sum w^2 (at most ``cap``)."""
    w2 = gains.double() ** 2
    order = torch.argsort(w2, descending=True)
    cum = torch.cumsum(w2[order], 0) / w2.sum()
    k = int(torch.searchsorted(cum, torch.tensor(mass, dtype=cum.dtype, device=cum.device)).item()) + 1
    return order[:min(k, cap, len(order))]


class GWNEncoder(torch.nn.Module):
    """Grid code ``(N, 434)`` -> unit-norm code ``(N, 2U)`` over the kept units."""

    def __init__(self, l1: L1, triples, gains, *, fwhm_ratio: float, mult="exact",
                 param=None, coef=None, gain: float = 100.0, meta: dict | None = None):
        super().__init__()
        self.register_buffer("Wc", l1.Wc.double().cpu())
        self.register_buffer("Ws", l1.Ws.double().cpu())
        self.register_buffer("module", l1.module.cpu())
        self.register_buffer("pattern", l1.pattern.cpu())
        self.register_buffer("triples", torch.as_tensor(triples, dtype=torch.long).cpu())
        self.register_buffer("w", torch.as_tensor(gains, dtype=torch.float64).cpu())
        self.register_buffer("coef", torch.as_tensor(coef, dtype=torch.float64).cpu()
                             if coef is not None else torch.ones(0, 8, dtype=torch.float64))
        self.lambdas = list(l1.lambdas)
        self.fwhm_ratio, self.mult, self.param = float(fwhm_ratio), mult, param
        self.gain = float(gain)
        self.meta = dict(meta or {})

    @property
    def out_dim(self) -> int:
        return 2 * int(self.triples.shape[0])

    def waves(self, codes):
        g = codes.to(self.Wc.dtype)
        coef = self.coef if self.coef.shape[0] else None
        return unit_waves(g @ self.Wc, g @ self.Ws, self.triples, self.mult, self.param, coef)

    def forward(self, codes):
        C, S = self.waves(codes)
        y = torch.cat([C * self.w, S * self.w], 1)
        y = y / y.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return y.to(torch.float32)

    def model_config(self):
        from .ideal_encoder import IdealModelConfig
        return IdealModelConfig(lambdas=list(self.lambdas), out_dim=self.out_dim,
                                encoder_type="grid_wave_network",
                                output_nonlinearity="unit_norm",
                                hidden_dim=int(self.Wc.shape[1]), num_hidden_layers=2,
                                gain=self.gain)


def save_gwn(enc: GWNEncoder, path: str) -> None:
    torch.save({"kind": "gwn", "Wc": enc.Wc, "Ws": enc.Ws, "module": enc.module,
                "pattern": enc.pattern, "triples": enc.triples, "w": enc.w,
                "coef": enc.coef, "lambdas": enc.lambdas, "fwhm_ratio": enc.fwhm_ratio,
                "mult": enc.mult, "param": enc.param, "gain": enc.gain, "meta": enc.meta}, path)


def load_gwn(path: str, device="cpu") -> GWNEncoder:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    l1 = L1(ck["Wc"], ck["Ws"], ck["module"], ck["pattern"], ck["lambdas"])
    coef = ck["coef"] if ck["coef"].shape[0] else None
    return GWNEncoder(l1, ck["triples"], ck["w"], fwhm_ratio=ck["fwhm_ratio"],
                      mult=ck["mult"], param=ck["param"], coef=coef, gain=ck["gain"],
                      meta=ck["meta"]).to(device).eval()


def is_gwn_spec(path) -> bool:
    return isinstance(path, str) and path.startswith(GWN_PREFIX)


def load_gwn_encoder(spec: str, *, device="cpu", fwhm_override=None, fwhm_fallback=None):
    """``harness.load_probe_encoder``'s return shape for ``gwn:<path.pt>``."""
    enc = load_gwn(spec[len(GWN_PREFIX):], device=device)
    fwhm = float(fwhm_override) if fwhm_override is not None else enc.fwhm_ratio
    cfg = enc.model_config()
    header = {
        "path": spec, "name": spec, "coverage": enc.meta.get("coverage"),
        "n_patches": None, "patch_sizes": None, "gain": enc.gain, "fwhm_ratio": fwhm,
        "fwhm_was_overridden": fwhm_override is not None, "lambdas": list(enc.lambdas),
        "out_dim": enc.out_dim, "hidden_dim": int(enc.Wc.shape[1]), "num_hidden_layers": 2,
        "encoder_type": cfg.encoder_type, "output_nonlinearity": "unit_norm",
        "n_params": int(enc.w.numel()), "epoch": None, "val_nav_acc": None,
        "unique_radius": None, "gwn": {"units": int(enc.triples.shape[0]), "mult": enc.mult,
                                       "param": enc.param, **enc.meta},
    }
    return enc, cfg, enc.gain, fwhm, header


__all__ = ["L1", "exact_l1", "random_triples", "all_triples", "triple_frequencies",
           "unit_waves", "multiply", "GAIN_RULES", "prune", "GWNEncoder", "save_gwn",
           "load_gwn", "is_gwn_spec", "load_gwn_encoder", "bump_strength"]
