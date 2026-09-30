"""Read a trained IdealNet harmonic layer: integers, frequencies, kernel.

What the layer's weights say, in the order the Stage A questions ask them:

1. **Integrality.** Per row, the largest distance of its six weights from the
   nearest integer. A row within 0.05 is counted integral.
2. **Which arena frequency each integral row implements.** Per axis,
   ``n = 156 j_11 + 143 j_12 + 132 j_13 (mod 1716)``, reduced to (-858, 858].
   This is the whole identity of a row: different integers can implement the
   same frequency, and the ideal encoder's own integers are one random draw.
3. **The spectrum's width.** For a Gaussian kernel of width r, the frequency
   vectors have ``E|omega|^2 = 2 / r^2``, so ``r_eff = sqrt(2 / mean|omega|^2)``
   over the integral rows.
4. **The kernel the whole layer produces** (integral or not), measured on the
   scaffold: profiles along both axes from several references (their spread is
   the translation-invariance check), the far-field sd over random pairs, and
   the alias ceiling (max similarity beyond 50 cells over the whole scaffold)
   for two references.

Phases are computed from coordinates, wrapped into (-pi, pi] exactly as atan2
returns them, which on integer positions equals the network's own readout.
"""
from __future__ import annotations

import math

import numpy as np
import torch

LAMBDAS = (11, 12, 13)
NPOS = 1716
CRT_C = (156, 143, 132)        # NPOS / lambda


def integrality(W: np.ndarray) -> dict:
    dev = np.abs(W - np.rint(W)).max(axis=1)
    return {"dev_median": float(np.median(dev)), "dev_mean": float(dev.mean()),
            "frac_int_05": float(np.mean(dev < 0.05)),
            "frac_int_01": float(np.mean(dev < 0.01))}


def decode_frequencies(J: np.ndarray) -> np.ndarray:
    """``(F, 6)`` integer harmonics (m0x, m0y, m1x, m1y, m2x, m2y) -> ``(F, 2)`` n."""
    J = np.asarray(J, dtype=np.int64)
    n = np.stack([sum(c * J[:, 2 * m + ax] for m, c in enumerate(CRT_C))
                  for ax in (0, 1)], axis=1)
    half = NPOS // 2
    return np.mod(n + half - 1, NPOS) - (half - 1)


def r_eff(n: np.ndarray) -> float:
    w2 = ((2 * math.pi * n / NPOS) ** 2).sum(1)
    m = float(w2.mean()) if len(w2) else float("nan")
    return math.sqrt(2.0 / m) if m > 0 else float("inf")


def phases_of(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``(N, 6)`` phases for integer positions, atan2 convention."""
    cols = []
    for lam in LAMBDAS:
        for c in (u, v):
            a = 2 * math.pi * torch.remainder(c, lam).double() / lam
            cols.append(torch.where(a > math.pi, a - 2 * math.pi, a))
    return torch.stack(cols, 1)


def _code(W: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
    th = ph @ W.T
    return torch.cat([torch.cos(th), torch.sin(th)], 1) / math.sqrt(W.shape[0])


def diagnose(W: np.ndarray, *, r: float, device="cpu", n_refs: int = 6,
             max_delta: int = 300, n_far: int = 20000, n_alias_refs: int = 2,
             seed: int = 0) -> dict:
    out = integrality(W)
    dev = np.abs(W - np.rint(W)).max(axis=1)
    J = np.rint(W[dev < 0.05]).astype(np.int64)
    n = decode_frequencies(J)
    out.update(n_integral=int(len(J)),
               r_eff_integral=r_eff(n),
               frac_dc=float(np.mean((n == 0).all(1))) if len(n) else None,
               n_distinct=int(len({tuple(x) for x in np.abs(n)})),
               abs_n_quantiles=[float(q) for q in np.percentile(
                   np.sqrt((n.astype(float) ** 2).sum(1)), [10, 50, 90])]
               if len(n) else None,
               frequencies=n.tolist())

    Wt = torch.from_numpy(W).double().to(device)
    rng = np.random.RandomState(seed)
    refs = rng.randint(0, NPOS, size=(n_refs, 2))
    d = torch.arange(0, max_delta + 1, device=device)
    prof = {0: [], 1: []}
    for (x0, y0) in refs:
        zr = _code(Wt, phases_of(torch.tensor([x0], device=device),
                                 torch.tensor([y0], device=device)))
        for ax in (0, 1):
            u = (x0 + d) if ax == 0 else torch.full_like(d, x0)
            v = torch.full_like(d, y0) if ax == 0 else (y0 + d)
            prof[ax].append((_code(Wt, phases_of(u, v)) @ zr.T).squeeze(1)
                            .cpu().numpy())
    P = np.concatenate([np.stack(prof[0]), np.stack(prof[1])])   # (2 n_refs, D)
    mean = P.mean(0)
    target = np.exp(-np.arange(max_delta + 1) ** 2 / (2 * r * r))
    below = np.where(mean < 0.5)[0]
    out.update(kernel_mean=mean.tolist(),
               kernel_target=target.tolist(),
               # translation invariance: spread across references, per axis
               # (x and y profiles legitimately differ for a finite draw)
               kernel_ref_spread_max=float(max(
                   (np.stack(prof[ax]).max(0) - np.stack(prof[ax]).min(0)).max()
                   for ax in (0, 1))),
               c1=float(mean[1]),
               r_half=int(below[0]) if len(below) else None,
               kernel_rmse_0_71=float(np.sqrt(((mean[:72] - target[:72]) ** 2)
                                              .mean())),
               kernel_max_abs_beyond_71=float(np.abs(mean[72:]).max()))

    a = torch.from_numpy(rng.randint(0, NPOS, size=(n_far, 2))).to(device)
    b = torch.from_numpy(rng.randint(0, NPOS, size=(n_far, 2))).to(device)
    za, zb = _code(Wt, phases_of(a[:, 0], a[:, 1])), \
        _code(Wt, phases_of(b[:, 0], b[:, 1]))
    out["far_sd"] = float((za * zb).sum(1).std())

    alias = []
    grid = torch.arange(NPOS, device=device)
    for (x0, y0) in refs[:n_alias_refs]:
        zr = _code(Wt, phases_of(torch.tensor([x0], device=device),
                                 torch.tensor([y0], device=device))).squeeze(0)
        best = -1.0
        for xs in range(0, NPOS, 64):
            xx = grid[xs:xs + 64]
            U = xx.repeat_interleave(NPOS)
            V = grid.repeat(len(xx))
            s = _code(Wt, phases_of(U, V)) @ zr
            dx = torch.remainder(U - x0, NPOS)
            dx = torch.minimum(dx, NPOS - dx)
            dy = torch.remainder(V - y0, NPOS)
            dy = torch.minimum(dy, NPOS - dy)
            far = (dx * dx + dy * dy) > 50 * 50
            if far.any():
                best = max(best, float(s[far].max()))
        alias.append(best)
    out["alias_ceiling"] = alias
    return out
