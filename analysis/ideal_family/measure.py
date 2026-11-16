"""Single-unit and population statistics for the ideal family, on top of the suite's measures.

Field detection, Treves-Rolls sparsity, count fits, lag profiles and kernel
summaries are ``analysis.coding_comparison.measures``, unchanged, so numbers
here are comparable to the coding-comparison scorecard. This module adds:

* fields on a closed loop: each unit's map is rolled to start at its minimum,
  so a field that crosses the loop's seam is one field, not two;
* per-unit coverage (fraction of positions with rate > 0);
* the area holding 95% of a unit's activity: the fewest bins whose rates sum
  to 95% of the unit's total, as a fraction of the environment;
* gap CV with the wrap-around gap on loops;
* summaries over all units and over units active in the environment.

Everything takes rate maps ``(units, positions)`` or ``(units, S, S)``.
"""
from __future__ import annotations

import math

import numpy as np

from analysis.coding_comparison import measures as CM

DEF_A = CM.FieldDef("A", "peak_fraction", frac=0.2, min_len=3, min_area=9)


def roll_to_min(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roll each row so it starts at its minimum. Returns the rolled maps and the shifts."""
    U, L = R.shape
    s = np.argmin(R, axis=1)
    idx = (s[:, None] + np.arange(L)[None, :]) % L
    return np.take_along_axis(R, idx, axis=1), s


def loop_fields(R: np.ndarray, fdef: CM.FieldDef = DEF_A) -> CM.Fields1D:
    """Definition-A fields on a closed loop (positions in rolled coordinates)."""
    Rr, _ = roll_to_min(np.asarray(R, dtype=np.float64))
    return CM.detect_fields_1d(Rr, fdef)


def gap_cvs(F: CM.Fields1D, *, circular: bool, min_fields: int = 6) -> np.ndarray:
    """Per-unit CV of gaps between consecutive field centres (units with >= ``min_fields``).

    On a loop the gap from the last centre back round to the first is included,
    so a unit with n fields has n gaps.
    """
    out = []
    L = F.length_bins
    order = np.lexsort((F.center, F.unit))
    u, c = F.unit[order], F.center[order]
    bounds = np.flatnonzero(np.diff(u)) + 1
    for cu in (np.split(c, bounds) if c.size else []):
        if cu.size < min_fields:
            continue
        g = np.diff(cu)
        if circular:
            g = np.append(g, L - cu[-1] + cu[0])
        if g.mean() > 0:
            out.append(g.std() / g.mean())
    return np.asarray(out)


def area95(R: np.ndarray, q: float = 0.95) -> np.ndarray:
    """Fraction of the environment in the fewest bins that hold ``q`` of each unit's activity.

    NaN for a silent unit. ``R`` is ``(units, positions...)``, non-negative.
    """
    X = np.asarray(R, dtype=np.float64).reshape(R.shape[0], -1)
    tot = X.sum(axis=1)
    S = -np.sort(-X, axis=1)
    c = np.cumsum(S, axis=1)
    n = (c < q * tot[:, None]).sum(axis=1) + 1
    return np.where(tot > 0, n / X.shape[1], np.nan)


def tr_summary(R: np.ndarray) -> dict:
    """Treves-Rolls sparsity three ways.

    ``active``: the mean of each unit's own ratio over units that fire
    somewhere (the suite's convention). ``silent_zero``: the same mean with
    silent units scored 0. ``pooled``: ``(mean rate)^2 / mean(rate^2)`` over
    all units and positions together, the population analogue of ``a(t)``.
    """
    X = np.asarray(R, dtype=np.float64).reshape(R.shape[0], -1)
    m1, m2 = X.mean(axis=1), (X * X).mean(axis=1)
    act = m2 > 0
    tr = np.where(act, m1 * m1 / np.where(act, m2, 1), 0.0)
    return {"active": float(tr[act].mean()) if act.any() else None,
            "silent_zero": float(tr.mean()),
            "pooled": float(m1.mean() ** 2 / m2.mean()) if m2.mean() > 0 else None,
            "frac_firing": float(act.mean())}


def _mean(x) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


def unit_stats(R: np.ndarray, F, counts: np.ndarray, sizes: np.ndarray, env_size: float,
               gaps: np.ndarray | None = None) -> dict:
    """Summaries of one environment for one code. ``env_size`` is cells or cells^2."""
    X = np.asarray(R, dtype=np.float64).reshape(R.shape[0], -1)
    act = counts > 0
    cov = (X > 0).mean(axis=1)
    fit = CM.fit_count_models(counts)
    out = {
        "n_units": int(X.shape[0]), "env_size": float(env_size),
        "active_frac": float(act.mean()),
        "fields_per_unit": float(counts.mean()),
        "fields_per_active": float(counts[act].mean()) if act.any() else None,
        "fields_per_unit_per_size": float(counts.mean() / env_size),
        "single_field_share_active": float((counts[act] == 1).mean()) if act.any() else None,
        "single_field_share_all": float((counts == 1).mean()),
        "var_over_mean": fit["var_over_mean"],
        "negbin_r": (fit["negbin"] or {}).get("r"),
        "mean_field_size": float(sizes.mean()) if sizes.size else None,
        "median_field_size": float(np.median(sizes)) if sizes.size else None,
        "field_size_cv": float(sizes.std() / sizes.mean()) if sizes.size else None,
        "n_fields": int(sizes.size),
        "coverage_all": float(cov.mean()),
        "coverage_active": float(cov[act].mean()) if act.any() else None,
        "coverage_firing": _mean(np.where(cov > 0, cov, np.nan)),
        "area95_firing": _mean(area95(X)),
        "area95_active": _mean(area95(X[act])) if act.any() else None,
        "tr": tr_summary(X),
    }
    if F is not None:
        out["field_coverage_all"] = float(sizes.sum() / (X.shape[0] * X.shape[1]))
    if gaps is not None:
        out["gap_cv_median"] = float(np.median(gaps)) if gaps.size else None
        out["gap_cv_mean"] = float(gaps.mean()) if gaps.size else None
        out["gap_cv_n_units"] = int(gaps.size)
    return out


def loop_stats(R: np.ndarray, fdef: CM.FieldDef = DEF_A) -> tuple[dict, CM.Fields1D]:
    F = loop_fields(R, fdef)
    counts = F.counts
    st = unit_stats(R, F, counts, F.length.astype(np.float64), R.shape[1],
                    gap_cvs(F, circular=True))
    return st, F


def segment_stats(R: np.ndarray, fdef: CM.FieldDef = DEF_A) -> dict:
    F = CM.detect_fields_1d(np.asarray(R, dtype=np.float64), fdef)
    return unit_stats(R, F, F.counts, F.length.astype(np.float64), R.shape[1],
                      gap_cvs(F, circular=False))


def arena_stats(R3: np.ndarray, fdef: CM.FieldDef = DEF_A) -> dict:
    """``R3`` is ``(units, S, S)``."""
    F = CM.detect_fields_2d(np.asarray(R3, dtype=np.float64), fdef)
    S = R3.shape[1]
    st = unit_stats(R3.reshape(R3.shape[0], -1), F, F.counts, F.area, S * S)
    return st


def pooled(parts: list[dict]) -> dict:
    """Average a list of per-environment summaries key by key (numbers only)."""
    out = {}
    keys = parts[0].keys() if parts else []
    for k in keys:
        vals = [p.get(k) for p in parts]
        if isinstance(vals[0], dict):
            out[k] = pooled([v for v in vals if v is not None])
        elif all(isinstance(v, (int, float)) or v is None for v in vals):
            out[k] = _mean([v for v in vals if v is not None])
    return out


def kernel_summary(d: np.ndarray, s: np.ndarray, width: float) -> dict:
    """``coding_comparison.measures.kernel_summary`` with the suite's far range (5 to 15 W)."""
    return CM.kernel_summary(d, s, width=width, far=(5.0, 15.0))


__all__ = ["DEF_A", "roll_to_min", "loop_fields", "gap_cvs", "area95", "tr_summary",
           "unit_stats", "loop_stats", "segment_stats", "arena_stats", "pooled",
           "kernel_summary"]
