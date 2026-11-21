"""Independent-bump place codes for the Hopfield probe, beside the rectified ideal.

Each code is ``hopfield_probe.bump_encoder.BumpEncoder``: D units, each with a Poisson number of
Gaussian fields (peak 1, uniform centres on the 1716 x 1716 scaffold torus, log-normal widths with a
fixed mean FWHM), rates scaled to unit length. ``dens`` is fields per unit per 100 x 100 cells.

Kernel. For Poisson fields at density ``rho = dens / 1e4`` the mean rate is ``mu = rho E[2 pi
sigma^2]`` and the covariance at distance d is ``C(d) = rho E[pi sigma^2 exp(-d^2 / (4 sigma^2))]``.
After the one global mean constant is removed the population kernel is ``C(d) / C(0)``; the raw
cosine has far-field floor ``2 mu / (1 + 2 mu)``. With one width the half-width is ``FWHM /
sqrt(2)``, so FWHM 27 cells gives 19.09 cells, the rectified ideal's (18.5 to 19.1 at 1024 units).
``matched_fwhm(spread)`` finds the mean FWHM that keeps that half-width when widths spread.

Codes (16 probe tasks, each plain and with ``inhib:``):

* 1024 units, one width (FWHM 27), dens 0.05, 0.2, 0.66, 0.85, 3. 0.2, 0.66 and 0.85 give 59, 194
  and 250 fields per unit over the scaffold, the rectified ideal's 58, 194 and 251 at 2%, 10% and
  20% active.
* 1024 units, spread 0.5, dens 0.66, mean FWHM matched.
* 256 and 4096 units, one width, dens 0.66.

    python -m analysis.rect_ideal.bump                     # the codes, expected widths and floors
    python -m analysis.rect_ideal.bump probe I             # "spec|label" of probe task I
    python -m analysis.rect_ideal.bump ntasks              # number of probe tasks
    python -m analysis.rect_ideal.bump check [OUT] [I ...]  # exactness + measured kernel per code
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
FWHM0 = 27.0
HALF0 = FWHM0 / math.sqrt(2.0)                  # 19.09 cells
DENS = (0.05, 0.2, 0.66, 0.85, 3.0)
BASE = "seed=0,gain=100"
RESULTS = "rect_ideal_grid_bump"

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(200)
_GH_W = _GH_W / _GH_W.sum()


# --- the expected kernel ---------------------------------------------------------------------

def _sigmas(fwhm: float, spread: float) -> np.ndarray:
    return fwhm / FWHM_PER_SIGMA * math.exp(-0.5 * spread ** 2) * np.exp(spread * _GH_X)


def expected(fwhm: float, spread: float, dens: float, d=None) -> dict:
    """Large-D kernel after removing the mean, raw floor, mean rate and half-width."""
    s = _sigmas(fwhm, spread)
    rho = dens / 1e4
    mu = rho * float((_GH_W * 2 * math.pi * s ** 2).sum())

    def C(x):
        x = np.atleast_1d(np.asarray(x, float))
        return rho * (_GH_W[None, :] * math.pi * s[None, :] ** 2
                      * np.exp(-x[:, None] ** 2 / (4 * s[None, :] ** 2))).sum(1)

    c0 = float(C(0.0)[0])
    hw = brentq(lambda x: float(C(x)[0]) / c0 - 0.5, 0.0, 2000.0, xtol=1e-8)
    out = {"mu": mu, "floor_raw": 2 * mu / (1 + 2 * mu), "half_width_inh": hw,
           "fields_per_unit": dens * 1716 ** 2 / 1e4}
    if d is not None:
        out["curve_inh"] = (C(d) / c0).tolist()
        out["curve_raw"] = ((C(d) + mu * mu) / (c0 + mu * mu)).tolist()
    return out


def matched_fwhm(spread: float, half: float = HALF0) -> float:
    """Mean FWHM at which the kernel after removing the mean falls to 0.5 at ``half`` cells."""
    return float(brentq(lambda w: expected(w, spread, 1.0)["half_width_inh"] - half, 1.0, 200.0,
                        xtol=1e-8))


# --- the codes -------------------------------------------------------------------------------

def codes() -> list[dict]:
    out = [dict(D=1024, fwhm=FWHM0, spread=0.0, dens=n) for n in DENS]
    out.append(dict(D=1024, fwhm=round(matched_fwhm(0.5), 4), spread=0.5, dens=0.66))
    out += [dict(D=D, fwhm=FWHM0, spread=0.0, dens=0.66) for D in (256, 4096)]
    return out


def spec(c: dict, inhib: bool = False) -> str:
    s = f"bump:D={c['D']},fwhm={c['fwhm']:g},spread={c['spread']:g},dens={c['dens']:g},{BASE}"
    return ("inhib:" + s) if inhib else s


def label(c: dict, inhib: bool = False) -> str:
    return (f"bump D{c['D']} w{c['fwhm']:.4g} s{c['spread']:g} n{c['dens']:g}"
            + (" inh" if inhib else ""))


def probe_tasks() -> list[tuple[str, str]]:
    return [(spec(c, inh), label(c, inh)) for c in codes() for inh in (False, True)]


# --- checks ----------------------------------------------------------------------------------

class _XY:
    """``fields.py``'s code interface: ``code(xs, ys)`` -> unit-length torch rows, ``.D``."""

    def __init__(self, enc):
        self.enc, self.D = enc, enc.D

    def __call__(self, xs, ys):
        return self.enc.code(xs, ys)


def check(c: dict, out_dir: str | None) -> dict:
    """Exactness against every field in float64, then the measured kernel beside the formula.

    The kernel uses ``fields.py``'s functions, as the rectified ideal's ``half_width_inh`` does:
    mean cosine between ``p`` and ``p + d e_axis`` over 1500 random places, both axes, d <= 64,
    with the inhibition constant (mean over 65,536 random cells, the ``inhib:`` wrapper's) removed
    and the code renormalised; floors over 8000 random pairs.
    """
    import torch
    from analysis.hopfield_probe.encode import grid_codes
    from analysis.hopfield_probe.harness import load_probe_encoder
    from analysis.rect_ideal import fields as F

    t0 = time.time()
    enc, cfg, _, fwhm_ratio, header = load_probe_encoder(spec(c), fwhm_fallback=0.25)
    t1 = time.time()
    g = np.random.default_rng(3)
    xs = np.concatenate([g.integers(0, 1716, 296), [0, 1715, 0, 1715]])
    ys = np.concatenate([g.integers(0, 1716, 296), [0, 0, 1715, 1715]])
    codes_ = torch.from_numpy(grid_codes(list(cfg.lambdas), xs, ys, fwhm_ratio))
    px, py = enc.positions(codes_)
    with torch.no_grad():
        z = enc(codes_).double().numpy()
    raw = enc.rates(xs, ys).double().numpy()
    ex = enc.exact_rates(xs, ys)
    zex = ex / np.linalg.norm(ex, axis=1, keepdims=True)
    chk = {"n": int(xs.size),
           "decode_exact": bool((px.numpy() == xs).all() and (py.numpy() == ys).all()),
           "max_abs_diff_raw": float(np.abs(raw - ex).max()),
           "max_abs_diff_code": float(np.abs(z - zex).max()),
           "bound_per_field": math.exp(-0.5 * enc.cut ** 2),
           "units_above_half_per_place": float((ex > 0.5).sum(1).mean()),
           "seconds_exact": None}
    t2 = time.time()
    chk["seconds_exact"] = t2 - t1
    if chk["max_abs_diff_code"] > 1e-4 or not chk["decode_exact"]:
        raise SystemExit(f"{label(c)}: tile index disagrees with the exact sum: {chk}")

    code = _XY(enc)
    cst = F.global_mean(code)
    Craw, Cinh = F.kernel(code), F.kernel(code, cst)
    fl = F.floor(code)
    kern = {"inhibition_constant": cst, "floor_raw": fl, "floor_inh": F.floor(code, cst),
            "half_width_raw": F.cross(Craw, 0.5),
            "half_width_raw_to_floor": F.cross(Craw, 0.5 * (1 + fl)),
            "half_width_inh": F.cross(Cinh, 0.5),
            "curve_raw": Craw.tolist(), "curve_inh": Cinh.tolist()}
    exp_ = expected(c["fwhm"], c["spread"], c["dens"], d=np.arange(65))
    res = {"label": label(c), "spec": spec(c), "code": c, "bump": header["bump"],
           "check": chk, "kernel": kern, "expected": exp_,
           "seconds": {"build": t1 - t0, "exact": t2 - t1, "kernel": time.time() - t2}}
    print(f"{label(c):<28s} decode {chk['decode_exact']}  diff raw {chk['max_abs_diff_raw']:.1e} "
          f"code {chk['max_abs_diff_code']:.1e}  half-width inh {kern['half_width_inh']:.2f} "
          f"(formula {exp_['half_width_inh']:.2f})  floor raw {fl:.4f} "
          f"(formula {exp_['floor_raw']:.4f})  inh floor {kern['floor_inh']:.4f}  "
          f"units>0.5 {chk['units_above_half_per_place']:.1f}  "
          f"[{time.time() - t0:.0f}s]", flush=True)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.join(out_dir, label(c).replace(" ", "_") + ".json")
        with open(fn, "w") as fh:
            json.dump(res, fh)
    return res


def main(argv: list[str]) -> int:
    if argv and argv[0] == "ntasks":
        print(len(probe_tasks()))
        return 0
    if argv and argv[0] == "probe":
        s, lab = probe_tasks()[int(argv[1])]
        print(f"{s}|{lab}")
        return 0
    if argv and argv[0] == "check":
        out = argv[1] if len(argv) > 1 and argv[1] != "-" else None
        which = [int(a) for a in argv[2:]] or range(len(codes()))
        for i in which:
            check(codes()[i], out)
        return 0
    print(f"one width: FWHM {FWHM0:g} -> half-width {HALF0:.3f} cells; "
          f"spread 0.5 matched mean FWHM {matched_fwhm(0.5):.4f}")
    for i, c in enumerate(codes()):
        e = expected(c["fwhm"], c["spread"], c["dens"])
        print(f"{i:2d} {label(c):<28s} half-width {e['half_width_inh']:.3f}  mu {e['mu']:.4f}  "
              f"raw floor {e['floor_raw']:.4f}  fields/unit {e['fields_per_unit']:.1f}")
    print(f"{len(probe_tasks())} probe tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
