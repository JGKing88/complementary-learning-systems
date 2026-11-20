"""The rectified-ideal grid: sparsity f by number of units D, for one set of grid modules.

Each code is ``ideal_encoder.IdealEncoder`` with ``relu_mix=1, active=f``:

    y(p) = relu(Q z(p) - t) / |.|,   t = Phi^-1(1 - f) / sqrt(D)

where ``z`` is the ideal code with ``n_freq = D / 2`` integer frequencies (a cos and a sin unit
each) and ``Q`` the fixed D x D Hadamard rotation (``rot_seed = 0``). So D is the number of units
and the number of waves together, and the unrectified ideal of the same D is each column's
baseline.

Kernel width is matched, not fixed. Rectifying narrows the population kernel, more so at small f.
``matched_r(f)`` widens the ideal's kernel radius r until the rectified code's kernel, measured
after the one global inhibition constant is removed, falls to 0.5 at the same distance as the
ideal's at the variant's own r (16 cells for modules 11, 12, 13). It is exact in the large-D limit:

* the ideal's kernel along an axis is ``k_r(d) = sum_n P_r(n) cos(2 pi n d / Npos)``, with
  ``P_r`` the law of ``round(N(0, sigma^2))``, ``sigma = Npos / (2 pi r)``
  (``ideal_encoder.draw_frequencies``);
* at two places with ideal similarity rho, the D values of ``sqrt(D) Q z`` are pairs of standard
  normals with correlation rho, so the rectified code's cosine is ``h(rho) / h(1)`` with
  ``h(rho) = E[(u - tau)+ (v - tau)+]``, ``tau = Phi^-1(1 - f)``;
* removing the global mean c and renormalising gives ``(h(rho) - h(0)) / (h(1) - h(0))``.

The finite draw at each D moves the realised width a little; ``fields.py`` measures it.

**Variants.** The ideal code depends on the grid modules only through the scaffold size
``Npos = prod(lambdas)``: its units are waves over position. ``m111213`` is the original grid.
``m111523_r16`` puts the same kernel (r = 16 cells) on the 11, 15, 23 scaffold (3795 cells);
``m111523_rs`` scales r with the modules' geometric-mean period, ``Npos^(1/3)`` (15.59 against
11.97), to r = 20.84. Sizes are reported over ``lambda_ref``: 12 cells for the original modules and
the geometric mean, 15.59 cells, for 11, 15, 23.

    python -m analysis.rect_ideal.codes [VARIANT]               # print the grid and the matched r
    python -m analysis.rect_ideal.codes probe I [VARIANT]       # "spec|label" of probe task I
    python -m analysis.rect_ideal.codes field I [VARIANT]       # "spec|label" of field task I
    python -m analysis.rect_ideal.codes var KEY [VARIANT]       # one variant setting (npos, ...)
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

FS = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
DS = (256, 512, 1024, 2048, 4096)
BASE = "seed=0,gain=100"


def _gm_period(lambdas) -> float:
    return float(np.prod(lambdas)) ** (1.0 / len(lambdas))


VARIANTS = {
    "m111213": {"lambdas": (11, 12, 13), "r_ideal": 16.0, "lambda_ref": 12.0,
                "results": "rect_ideal_grid"},
    "m111523_r16": {"lambdas": (11, 15, 23), "r_ideal": 16.0,
                    "lambda_ref": round(_gm_period((11, 15, 23)), 4),
                    "results": "rect_ideal_grid_m111523_r16"},
    "m111523_rs": {"lambdas": (11, 15, 23),
                   "r_ideal": round(16.0 * _gm_period((11, 15, 23)) / _gm_period((11, 12, 13)), 4),
                   "lambda_ref": round(_gm_period((11, 15, 23)), 4),
                   "results": "rect_ideal_grid_m111523_rs"},
}
DEFAULT = "m111213"


def npos(variant: str = DEFAULT) -> int:
    return int(np.prod(VARIANTS[variant]["lambdas"]))


# Kept for callers written before the variants (the original grid).
NPOS = npos(DEFAULT)
LAMBDAS = VARIANTS[DEFAULT]["lambdas"]
LAMBDA_REF = VARIANTS[DEFAULT]["lambda_ref"]
R_IDEAL = VARIANTS[DEFAULT]["r_ideal"]


# --- the ideal kernel --------------------------------------------------------

def _p_round(r: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    sigma = N / (2.0 * math.pi * r)
    nmax = int(math.ceil(12 * sigma)) + 2
    n = np.arange(-nmax, nmax + 1)
    p = ndtr((n + 0.5) / sigma) - ndtr((n - 0.5) / sigma)
    return n, p / p.sum()


def ideal_kernel(r: float, d, N: int = NPOS) -> np.ndarray:
    """Expected ideal similarity at axis offset d (cells), for kernel radius r on an N scaffold."""
    n, p = _p_round(r, N)
    d = np.atleast_1d(np.asarray(d, dtype=float))
    return (p[None, :] * np.cos(2 * math.pi * np.outer(d, n) / N)).sum(1)


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


def matched_r(f: float, variant: str = DEFAULT) -> float:
    """r at which the inhibited rectified code's half-width equals the variant's ideal's."""
    N, r0 = npos(variant), VARIANTS[variant]["r_ideal"]
    d_half = half_width(lambda d: ideal_kernel(r0, d, N))
    rho_star = brentq(lambda rho: rect_kernel_centred(rho, f) - 0.5, 0.0, 0.999999)
    return float(brentq(lambda r: float(ideal_kernel(r, d_half, N)[0]) - rho_star, 1.0, 200.0,
                        xtol=1e-6))


# --- the grid ----------------------------------------------------------------

def f_tag(f: float) -> str:
    return f"f{round(100 * f):02d}"


def _mods(variant: str) -> str:
    """The spec's modules key; empty for the original modules, so those specs are unchanged."""
    if variant == DEFAULT:
        return ""
    return ",lambdas=" + "-".join(str(l) for l in VARIANTS[variant]["lambdas"])


def rect_spec(f: float, D: int, inhib: bool = False, variant: str = DEFAULT) -> str:
    s = (f"ideal:r={matched_r(f, variant):.4f},n_freq={D // 2},{BASE}{_mods(variant)},"
         f"relu_mix=1,active={f}")
    return s + (",center_global=1" if inhib else "")


def ideal_spec(D: int, variant: str = DEFAULT) -> str:
    return f"ideal:r={VARIANTS[variant]['r_ideal']:g},n_freq={D // 2},{BASE}{_mods(variant)}"


def probe_tasks(variant: str = DEFAULT) -> list[tuple[str, str]]:
    """65 tasks: per D, the signed ideal, then every f plain and with inhibition."""
    out = []
    for D in DS:
        out.append((ideal_spec(D, variant), f"ideal D{D}"))
        for f in FS:
            out.append((rect_spec(f, D, variant=variant), f"rect {f_tag(f)} D{D}"))
            out.append((rect_spec(f, D, inhib=True, variant=variant), f"rect {f_tag(f)} D{D} inh"))
    return out


def field_tasks(variant: str = DEFAULT) -> list[tuple[str, str]]:
    """30 tasks: every (f, D). Fields are the code's own, so inhibition plays no part."""
    return [(rect_spec(f, D, variant=variant), f"rect {f_tag(f)} D{D}") for D in DS for f in FS]


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("probe", "field"):
        variant = argv[2] if len(argv) > 2 else DEFAULT
        tasks = probe_tasks(variant) if argv[0] == "probe" else field_tasks(variant)
        spec, label = tasks[int(argv[1])]
        print(f"{spec}|{label}")
        return 0
    if argv and argv[0] == "var":
        variant = argv[2] if len(argv) > 2 else DEFAULT
        print(npos(variant) if argv[1] == "npos" else VARIANTS[variant][argv[1]])
        return 0
    variant = argv[0] if argv else DEFAULT
    N, r0 = npos(variant), VARIANTS[variant]["r_ideal"]
    d0 = half_width(lambda d: ideal_kernel(r0, d, N))
    print(f"{variant}: modules {VARIANTS[variant]['lambdas']}, Npos {N}, "
          f"lambda_ref {VARIANTS[variant]['lambda_ref']}, ideal r={r0:g} half-width {d0:.3f} cells")
    for f in FS:
        r = matched_r(f, variant)
        rho = ideal_kernel(r, d0, N)[0]
        print(f"{f_tag(f)}  r={r:7.3f}  ideal sim at d_half {rho:.4f}  "
              f"rect centred {rect_kernel_centred(rho, f):.4f}")
    print(f"{len(probe_tasks(variant))} probe tasks, {len(field_tasks(variant))} field tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
