"""The rectified-ideal grid: sparsity f by number of units D.

Each code is ``ideal_encoder.IdealEncoder`` with ``relu_mix=1, active=f``:

    y(p) = relu(Q z(p) - t) / |.|,   t = Phi^-1(1 - f) / sqrt(D)

where ``z`` is the ideal code with ``n_freq = D / 2`` integer frequencies (a cos and a sin unit
each) and ``Q`` the fixed D x D Hadamard rotation (``rot_seed = 0``). So D is the number of units
and the number of waves together, and the unrectified ideal of the same D is each column's
baseline.

Kernel width is matched, not fixed. Rectifying narrows the population kernel, more so at small f.
``matched_r(f)`` widens the ideal's kernel radius r until the rectified code's kernel, measured
after the one global inhibition constant is removed, falls to 0.5 at the same distance as the
ideal's at r = 16. It is exact in the large-D limit:

* the ideal's kernel along an axis is ``k_r(d) = sum_n P_r(n) cos(2 pi n d / Npos)``, with
  ``P_r`` the law of ``round(N(0, sigma^2))``, ``sigma = Npos / (2 pi r)``
  (``ideal_encoder.draw_frequencies``);
* at two places with ideal similarity rho, the D values of ``sqrt(D) Q z`` are pairs of standard
  normals with correlation rho, so the rectified code's cosine is ``h(rho) / h(1)`` with
  ``h(rho) = E[(u - tau)+ (v - tau)+]``, ``tau = Phi^-1(1 - f)``;
* removing the global mean c and renormalising gives ``(h(rho) - h(0)) / (h(1) - h(0))``.

The finite draw at each D moves the realised width a little; ``fields.py`` measures it.

    python -m analysis.rect_ideal.codes            # print the grid and the matched r
    python -m analysis.rect_ideal.codes probe I    # "spec|label" of probe task I
    python -m analysis.rect_ideal.codes field I    # "spec|label" of field task I
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

NPOS = 1716
LAMBDAS = (11, 12, 13)
LAMBDA_REF = 12.0                    # sizes are reported over this period (modules are 11-13)
R_IDEAL = 16.0
FS = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
DS = (256, 512, 1024, 2048, 4096)
BASE = "seed=0,gain=100"


# --- the ideal kernel --------------------------------------------------------

def _p_round(r: float, nmax: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    sigma = NPOS / (2.0 * math.pi * r)
    nmax = nmax or int(math.ceil(12 * sigma)) + 2
    n = np.arange(-nmax, nmax + 1)
    p = ndtr((n + 0.5) / sigma) - ndtr((n - 0.5) / sigma)
    return n, p / p.sum()


def ideal_kernel(r: float, d) -> np.ndarray:
    """Expected ideal similarity at axis offset d (cells), for kernel radius r."""
    n, p = _p_round(r)
    d = np.atleast_1d(np.asarray(d, dtype=float))
    return (p[None, :] * np.cos(2 * math.pi * np.outer(d, n) / NPOS)).sum(1)


def half_width(kernel, lo: float = 0.0, hi: float = 400.0) -> float:
    """First distance where ``kernel(d)`` falls to 0.5."""
    return float(brentq(lambda d: float(kernel(d)[0]) - 0.5, lo, hi, xtol=1e-6))


# --- the rectified kernel ----------------------------------------------------

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(200)
_GH_W = _GH_W / _GH_W.sum()


def _phi(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def h_relu(rho: float, tau: float) -> float:
    """``E[(u - tau)+ (v - tau)+]`` for standard normals with correlation rho."""
    u = _GH_X
    a = np.maximum(u - tau, 0.0)
    if rho >= 1.0:
        return float((_GH_W * a * a).sum())
    m = rho * u - tau
    s = math.sqrt(max(1.0 - rho * rho, 0.0))
    inner = m * ndtr(m / s) + s * _phi(m / s)
    return float((_GH_W * a * inner).sum())


def rect_kernel_centred(rho: float, f: float) -> float:
    tau = float(ndtri(1.0 - f))
    h0, h1 = h_relu(0.0, tau), h_relu(1.0, tau)
    return (h_relu(rho, tau) - h0) / (h1 - h0)


def matched_r(f: float) -> float:
    """r at which the inhibited rectified code's half-width equals the ideal's at r = 16."""
    d_half = half_width(lambda d: ideal_kernel(R_IDEAL, d))
    rho_star = brentq(lambda rho: rect_kernel_centred(rho, f) - 0.5, 0.0, 0.999999)
    return float(brentq(lambda r: float(ideal_kernel(r, d_half)[0]) - rho_star, 1.0, 200.0,
                        xtol=1e-6))


# --- the grid ----------------------------------------------------------------

def f_tag(f: float) -> str:
    return f"f{round(100 * f):02d}"


def rect_spec(f: float, D: int, inhib: bool = False) -> str:
    s = (f"ideal:r={matched_r(f):.4f},n_freq={D // 2},{BASE},relu_mix=1,active={f}")
    return s + (",center_global=1" if inhib else "")


def ideal_spec(D: int) -> str:
    return f"ideal:r={R_IDEAL:g},n_freq={D // 2},{BASE}"


def probe_tasks() -> list[tuple[str, str]]:
    """65 tasks: per D, the signed ideal, then every f plain and with inhibition."""
    out = []
    for D in DS:
        out.append((ideal_spec(D), f"ideal D{D}"))
        for f in FS:
            out.append((rect_spec(f, D), f"rect {f_tag(f)} D{D}"))
            out.append((rect_spec(f, D, inhib=True), f"rect {f_tag(f)} D{D} inh"))
    return out


def field_tasks() -> list[tuple[str, str]]:
    """30 tasks: every (f, D). Fields are the code's own, so inhibition plays no part."""
    return [(rect_spec(f, D), f"rect {f_tag(f)} D{D}") for D in DS for f in FS]


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] in ("probe", "field"):
        tasks = probe_tasks() if argv[0] == "probe" else field_tasks()
        spec, label = tasks[int(argv[1])]
        print(f"{spec}|{label}")
        return 0
    d16 = half_width(lambda d: ideal_kernel(R_IDEAL, d))
    print(f"ideal r=16 half-width {d16:.3f} cells")
    for f in FS:
        r = matched_r(f)
        rho = ideal_kernel(r, d16)[0]
        print(f"{f_tag(f)}  r={r:7.3f}  ideal sim at d_half {rho:.4f}  "
              f"rect centred {rect_kernel_centred(rho, f):.4f}")
    print(f"{len(probe_tasks())} probe tasks, {len(field_tasks())} field tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
