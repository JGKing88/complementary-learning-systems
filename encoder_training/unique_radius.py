"""The unique coding radius of a similarity map.

The question
-----------
Stand at a reference position and compare every other position's encoding to
yours. How large a disc around yourself can you carve out such that *everything
inside it* scores higher than *everything outside it*? If that holds at radius
r, then similarity alone certifies proximity: observing a similarity of at
least ``inner_min`` means you are within r cells of the reference.

The condition, at radius r::

    min{ sim(p) : |p - ref| <= r }  >  (trim+1)-th largest{ sim(q) : |q - ref| > r }

evaluated outward and stopped at the **first** r that fails. First failure, not
last success: the condition is not monotone in r (it can fail at r=10 and hold
again at r=50 once the offending cell falls inside the disc) and it is
vacuously true at the outermost radius, where nothing lies beyond. A radius
only means anything if the guarantee nests all the way in.

What ``trim`` buys
------------------
``trim`` drops that many top-scoring cells outside the disc, so a single
anomalous cell cannot decide the answer. It is part of the claim rather than a
fudge factor -- at trim=16 the statement is "within r cells, with at most 16
exceptional positions in the whole arena". That is why it is a fixed *count*
and not a percentile: at Npos=1716 the map holds 2.94M cells, so a 99.9th
percentile would discard 2944 of them and quietly erase a genuinely aliased
region. A count states the exception budget outright.

Trim is neutral exactly where it should be. Against cones with an analytically
known radius (a decay ``1 - d/300`` plus a 3600-cell alias plateau at level A,
so r* = 300(1-A)), trim=0/4/16/64 all return the same answer at every scale --
trim cannot reach a plateau that large. It moves the result only when the
binding constraint is the immediate neighbourhood, which is the ties-and-noise
case it exists to absorb.

What sets the number
--------------------
Two different things break the condition, and the companion statistics here
tell you which:

* **the ring just outside** -- for small r the outside set includes the annulus
  at r+1, whose similarity is barely below the disc edge, so the condition is
  effectively demanding that the radial profile strictly decrease. Flat spots
  (quantisation, ties, the noise floor) fail here.
* **a distant alias** -- past the local decay, the outside is dominated by the
  highest similarity anywhere else in the arena. ``alias_ceiling`` reports it.

So the radius is the lesser of where the profile stops decreasing and where it
decays to the alias ceiling. Since the radius is a threshold crossing it has
cliff behaviour -- two encoders can differ a lot and both report 0 -- which is
what ``margin`` is for: the same quantity *before* the crossing is taken.

Conventions
-----------
Radii are in cells, binned to integers, and report the last **passing** bin, so
every value reads as "at least r" and is conservative by up to one cell. The
metric depends on the map only through Euclidean distance, so it is invariant
to whether the caller indexes ``[x, y]`` or ``[y, x]`` -- unlike the ray-based
version in ``viz``, there is no axis convention to get wrong.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

DEFAULT_TRIMS: tuple[int, ...] = (0, 4, 16, 64)
HEADLINE_TRIM: int = 16
DEFAULT_MARGIN_RADII: tuple[int, ...] = (5, 10, 25, 50)
DEFAULT_PROFILE_LEVELS: tuple[float, ...] = (0.9, 0.5, 0.1)


def _bin_by_radius(cos_map: np.ndarray, i0: float, i1: float):
    """Group cells into integer-radius shells around (i0, i1).

    Returns (s_sorted, starts, nbin) where ``s_sorted[starts[b]:starts[b+1]]``
    holds every similarity at ``floor(distance) == b``. The sort is a radix
    sort on int32, so this is linear in the number of cells and is the only
    part of the computation that scales with map area.
    """
    n0, n1 = cos_map.shape
    a0 = np.arange(n0, dtype=np.float64)[:, None] - float(i0)
    a1 = np.arange(n1, dtype=np.float64)[None, :] - float(i1)
    rb = np.sqrt(a0 * a0 + a1 * a1).astype(np.int32).ravel()  # floor via cast

    order = np.argsort(rb, kind="stable")
    nbin = int(rb.max()) + 1
    starts = np.searchsorted(rb[order], np.arange(nbin + 1))
    return cos_map.ravel()[order].astype(np.float64), starts, nbin


def _outer_ceilings(s_sorted, starts, nbin, trims):
    """``out[j, b]`` = the (trims[j]+1)-th largest similarity beyond shell b.

    One backward sweep maintaining the K largest values seen so far, K one more
    than the largest trim, which yields every trim level from a single pass.
    """
    K = int(max(trims)) + 1
    out = np.full((len(trims), nbin), -np.inf)
    top = np.empty(0)                       # sorted ascending, length <= K
    for b in range(nbin - 1, -1, -1):
        for j, t in enumerate(trims):
            # top is ascending, so the (t+1)-th largest is top[-(t+1)]
            out[j, b] = top[-(t + 1)] if len(top) >= t + 1 else -np.inf
        chunk = s_sorted[starts[b]:starts[b + 1]]
        if chunk.size:
            merged = np.concatenate([top, chunk]) if top.size else chunk
            if merged.size > K:
                merged = np.partition(merged, -K)[-K:]
            top = np.sort(merged)
    return out


def _shell_stats(s_sorted, starts, nbin):
    """Per-shell min / mean / count, and the running min over the disc."""
    bin_min = np.full(nbin, np.inf)
    bin_mean = np.full(nbin, np.nan)
    bin_cnt = np.zeros(nbin, dtype=np.int64)
    for b in range(nbin):
        chunk = s_sorted[starts[b]:starts[b + 1]]
        if chunk.size:
            bin_min[b] = chunk.min()
            bin_mean[b] = chunk.mean()
            bin_cnt[b] = chunk.size
    return np.minimum.accumulate(bin_min), bin_mean, bin_cnt


def _first_failure(inner_min, outer_hi, max_r):
    """Walk outward; return (radius, saturated) at the first failing shell."""
    nbin = len(inner_min)
    limit = min(int(np.floor(max_r)), nbin - 1)
    ok = inner_min[:limit + 1] > outer_hi[:limit + 1]
    bad = np.flatnonzero(~ok)
    if bad.size == 0:
        return float(limit), True           # never failed: value is a floor
    return float(bad[0] - 1) if bad[0] > 0 else 0.0, False


def unique_radius(
    cos_map: np.ndarray,
    i0: float,
    i1: float,
    *,
    trim: int = HEADLINE_TRIM,
    max_r: float | None = None,
) -> tuple[float, bool]:
    """Unique coding radius of ``cos_map`` about (i0, i1).

    Returns ``(radius, saturated)``. ``saturated`` means the walk reached
    ``max_r`` without failing, so the radius is a lower bound rather than a
    measurement. ``max_r`` defaults to the reference's distance to the nearest
    edge, beyond which the disc would be clipped by the map.
    """
    rep = unique_radius_report(cos_map, i0, i1, trims=(trim,), max_r=max_r,
                               headline_trim=trim, margin_radii=(),
                               profile_levels=())
    return rep[f"r_trim{trim}"], rep[f"saturated_trim{trim}"]


def unique_radius_report(
    cos_map: np.ndarray,
    i0: float,
    i1: float,
    *,
    trims: Sequence[int] = DEFAULT_TRIMS,
    headline_trim: int = HEADLINE_TRIM,
    margin_radii: Iterable[int] = DEFAULT_MARGIN_RADII,
    profile_levels: Iterable[float] = DEFAULT_PROFILE_LEVELS,
    exclusion_radius: int = 50,
    floor_radius: int = 200,
    max_r: float | None = None,
) -> dict:
    """Every unique-radius statistic for one reference, from a single sort.

    Beyond the radius at each trim level this reports the diagnostics that stay
    informative when the radius itself bottoms out at 0:

    ``alias_ceiling``
        Trimmed highest similarity anywhere beyond ``exclusion_radius``. This
        is the ceiling the decay curve has to clear, so it explains *why* a
        radius collapsed, and it is far steadier than the radius.
    ``margin_r{R}``
        ``inner_min(R) - outer_hi(R)`` at fixed radii: the signed quantity
        whose zero crossing is the radius. Still separates encoders when every
        radius reads 0.
    ``r_at_cos{L}``
        First radius where the shell-mean similarity falls below L -- a smooth
        local width, robust where the radius is brittle.
    ``cos_floor``
        Mean similarity beyond ``floor_radius``: the background level.
    """
    trims = tuple(int(t) for t in trims)
    if headline_trim not in trims:
        trims = tuple(sorted(set(trims) | {int(headline_trim)}))

    n0, n1 = cos_map.shape
    border = float(min(i0, i1, (n0 - 1) - i0, (n1 - 1) - i1))
    if max_r is None:
        max_r = border

    s_sorted, starts, nbin = _bin_by_radius(cos_map, i0, i1)
    outer_hi = _outer_ceilings(s_sorted, starts, nbin, trims)
    inner_min, bin_mean, bin_cnt = _shell_stats(s_sorted, starts, nbin)

    rep: dict = {
        "ref_i0": float(i0),
        "ref_i1": float(i1),
        "border_dist": border,
        "max_r": float(max_r),
        "n_cells": int(bin_cnt.sum()),
    }
    for j, t in enumerate(trims):
        r, sat = _first_failure(inner_min, outer_hi[j], max_r)
        rep[f"r_trim{t}"] = r
        rep[f"saturated_trim{t}"] = bool(sat)

    hj = trims.index(int(headline_trim))
    rep["headline_trim"] = int(headline_trim)
    rep["r_headline"] = rep[f"r_trim{headline_trim}"]
    rep["saturated_headline"] = rep[f"saturated_trim{headline_trim}"]

    ex = min(int(exclusion_radius), nbin - 1)
    rep["alias_ceiling"] = float(outer_hi[hj, ex])
    rep["exclusion_radius"] = int(ex)

    for R in margin_radii:
        b = int(R)
        rep[f"margin_r{R}"] = (
            float(inner_min[b] - outer_hi[hj, b]) if b < nbin else float("nan"))

    for lvl in profile_levels:
        below = np.flatnonzero(np.nan_to_num(bin_mean, nan=-np.inf) < lvl)
        rep[f"r_at_cos{lvl}"] = float(below[0]) if below.size else float("nan")

    fr = min(int(floor_radius), nbin - 1)
    tail = s_sorted[starts[fr]:]
    rep["cos_floor"] = float(tail.mean()) if tail.size else float("nan")
    return rep


__all__ = [
    "DEFAULT_TRIMS",
    "HEADLINE_TRIM",
    "DEFAULT_MARGIN_RADII",
    "DEFAULT_PROFILE_LEVELS",
    "unique_radius",
    "unique_radius_report",
]
