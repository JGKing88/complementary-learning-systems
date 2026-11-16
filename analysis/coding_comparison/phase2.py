"""Phase 2: correlation dimension and position decoding, in square arenas.

Both run through ``run.py --measures phase2`` and write ``phase2.json``.

Correlation dimension (Grassberger-Procaccia): sample ``n_dim`` cells of an
arena, take the Euclidean distance between every pair of their population
vectors, and fit the slope of log C(eps) against log eps where the pair
fraction C lies between ``c_lo`` and ``c_hi``. Nakai 2024 report 3.6 for CA1
and 5.8 for subiculum in an open field; the review does not give their
scaling range, so the comparison is rough.

Decoding: rates at the Rich scale (population mean 0.75 Hz), Poisson spike
counts in a ``dt`` window, a maximum-likelihood decoder over every cell of the
arena, and the median error in mean field widths, for growing random subsets
of units. The review has no scale-free decoding target, so this is reported,
not scored.
"""
from __future__ import annotations

import math

import numpy as np

from . import geometry as geo
from . import measures as M

DEFAULTS = {"dim_arena_A0": [4, 16, 64], "n_dim": 2500, "c_lo": 1e-3, "c_hi": 1e-1,
            "dec_arena_A0": 16, "dec_side_cap": 160, "dec_n_test": 400, "dt": 0.5,
            "rate_floor_hz": 0.01, "dec_sizes": [5, 10, 20, 50, 100, 200, 500, 1000, 2000],
            "dec_repeats": 3}


def corr_dimension(X: np.ndarray, c_lo: float, c_hi: float) -> dict:
    """Grassberger-Procaccia slope for points ``X`` (n, features)."""
    sq = (X * X).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    iu = np.triu_indices(X.shape[0], 1)
    d = np.sqrt(D2[iu])
    d = np.sort(d[d > 0])
    if d.size < 100:
        return {"dimension": None}
    eps = np.geomspace(d[max(0, int(1e-4 * d.size))], d[-1], 60)
    C = np.searchsorted(d, eps) / d.size
    sel = (C >= c_lo) & (C <= c_hi)
    out = {"eps": eps, "C": C, "fit_range_C": [c_lo, c_hi], "n_points": int(X.shape[0])}
    if sel.sum() < 3:
        out["dimension"] = None
        return out
    out["dimension"] = float(np.polyfit(np.log(eps[sel]), np.log(C[sel]), 1)[0])
    return out


def decode_errors(F: np.ndarray, xy: np.ndarray, sizes, *, dt, n_test, repeats, rng) -> dict:
    """Median ML decoding error (cells) against the number of units.

    ``F`` is ``(cells, units)`` rates in Hz at the arena's cells, ``xy`` their
    coordinates. Each test draws Poisson counts at a random true cell.
    """
    P, U = F.shape
    out = {}
    for n in sizes:
        if n > U:
            continue
        errs = []
        for _ in range(repeats):
            units = rng.choice(U, n, replace=False)
            f = F[:, units]
            logf, sumf = np.log(f * dt), f.sum(axis=1) * dt
            true = rng.integers(0, P, n_test)
            k = rng.poisson(f[true] * dt)
            ll = k @ logf.T - sumf[None, :]
            est = ll.argmax(axis=1)
            errs.append(np.hypot(*(xy[est] - xy[true]).T))
        e = np.concatenate(errs)
        out[str(n)] = {"median": float(np.median(e)), "mean": float(e.mean())}
    return out


def run_phase2(pop, P, cal, W_ref, rng, t0) -> dict:
    from .run import log, rates_of, variants_for

    Q = dict(DEFAULTS, **P.get("phase2", {}))
    out = {"params": Q, "corr_dim": {}, "corr_dim_by_arena": {}, "decoding": {}}
    if W_ref is None:
        return out
    npos = pop.npos
    A0 = math.pi / 4 * W_ref ** 2
    convs = sorted({v.conv for v in variants_for(pop, P) if v.rot == 0})
    for k in Q["dim_arena_A0"]:
        side = int(round(math.sqrt(k * A0)))
        (x0, y0), = geo.spread_offsets(1, side, npos, 1, seed=int(rng.integers(1 << 30)))
        n = min(Q["n_dim"], side * side)
        idx = rng.choice(side * side, n, replace=False)
        xs, ys = geo.arena(x0, y0, side)
        z = pop.code(xs[idx], ys[idx]).astype(np.float64)
        res = {"raw": corr_dimension(z, Q["c_lo"], Q["c_hi"])}
        for c in convs:
            res[c] = corr_dimension(rates_of(z, c).T, Q["c_lo"], Q["c_hi"])
        out["corr_dim_by_arena"][str(k)] = {"side": side, **res}
        if k == 16:
            out["corr_dim"] = res
        log(f"phase2 corr dim, {k} A0: " + ", ".join(
            f"{b}={r.get('dimension') and round(r['dimension'], 2)}" for b, r in res.items()), t0)

    side = int(min(round(math.sqrt(Q["dec_arena_A0"] * A0)), Q["dec_side_cap"]))
    (x0, y0), = geo.spread_offsets(1, side, npos, 1, seed=int(rng.integers(1 << 30)))
    xs, ys = geo.arena(x0, y0, side)
    z = pop.code(xs, ys)
    xy = np.stack([xs - x0, ys - y0], axis=1).astype(np.float64)
    for c in convs:
        hz = None
        sc = cal.get(f"{c}|B0.75")
        if sc and sc.get("fdef"):
            hz = sc["fdef"]["rate_scale"]
        if hz is None:
            continue
        F = rates_of(z, c).T.astype(np.float64) * hz + Q["rate_floor_hz"]
        err = decode_errors(F, xy, Q["dec_sizes"], dt=Q["dt"], n_test=Q["dec_n_test"],
                            repeats=Q["dec_repeats"], rng=rng)
        W = cal[f"{c}|B0.75"]["W"] or W_ref
        out["decoding"][c] = {"side": side, "W_cells": W,
                              "median_error_W": {n: e["median"] / W for n, e in err.items()},
                              "errors_cells": err}
        log(f"phase2 decoding {c}: " + ", ".join(
            f"{n}:{e['median'] / W:.2f}W" for n, e in err.items()), t0)
    return out


__all__ = ["corr_dimension", "decode_errors", "run_phase2"]
