"""Streaming accumulators, so the suite does not have to keep what it measures.

The naive shape of these probes -- one record per (world, env, K, steps, cell)
-- is 3e9 numbers at the spec's defaults. Nothing needs that: every figure in
the report is either a binned curve, a pooled 2-D map, or a per-world scalar.

So values are folded in as they are produced:

``BinnedStat``
    mean/std/n exactly (running sums), and median/IQR from a **value
    histogram** per distance bin. A percentile cannot be accumulated the way a
    mean can; a histogram over a known range can, and at 1-degree resolution
    for an angle it is exact enough to plot. This is the only approximation in
    the package and it is stated rather than hidden.

``Map2D``
    running sum and count per cell, for the goal-relative and env-absolute
    heatmaps, which pool over every world and env.

Per-cell *maps* survive only for the handful of (world, env) pairs the report
draws as raw examples -- ``ProbeConfig.n_map_worlds`` / ``n_map_envs``.
Aggregates hide structure and one raw example is what catches a harness bug,
but keeping every example is what makes the output unusable.
"""
from __future__ import annotations

import numpy as np

# Value-histogram ranges, per quantity. Chosen so the resolution is finer than
# anything the report draws: 1 degree for angles, 0.01 for a cosine.
VALUE_RANGES = {
    "angle_deg": (0.0, 180.0, 180),
    "signed_angle_deg": (-180.0, 180.0, 360),
    "cos": (-1.0, 1.0, 200),
    "frac": (0.0, 1.0, 100),
    "qnorm": (0.0, 2.0, 200),
    "cells": (0.0, 64.0, 128),
    "excess_deg": (-180.0, 180.0, 360),
}


class BinnedStat:
    """A quantity binned against distance, with exact means and histogram
    percentiles."""

    def __init__(self, dist_edges: np.ndarray, value_kind: str = "angle_deg"):
        self.edges = np.asarray(dist_edges, dtype=float)
        self.nb = len(self.edges) - 1
        lo, hi, nv = VALUE_RANGES[value_kind]
        self.kind = value_kind
        self.v_lo, self.v_hi, self.nv = lo, hi, nv
        self.hist = np.zeros((self.nb, nv), dtype=np.int64)
        self.sum = np.zeros(self.nb)
        self.sumsq = np.zeros(self.nb)
        self.count = np.zeros(self.nb, dtype=np.int64)

    def add(self, dist: np.ndarray, values: np.ndarray) -> None:
        dist = np.asarray(dist, dtype=float).ravel()
        values = np.asarray(values, dtype=float).ravel()
        ok = np.isfinite(dist) & np.isfinite(values)
        if not ok.any():
            return
        d, v = dist[ok], values[ok]
        b = np.clip(np.digitize(d, self.edges) - 1, 0, self.nb - 1)

        np.add.at(self.sum, b, v)
        np.add.at(self.sumsq, b, v * v)
        np.add.at(self.count, b, 1)

        vb = np.clip(
            ((v - self.v_lo) / (self.v_hi - self.v_lo) * self.nv).astype(int),
            0, self.nv - 1)
        np.add.at(self.hist, (b, vb), 1)

    def _pct(self, p: float) -> list:
        out = []
        centers = (self.v_lo + (np.arange(self.nv) + 0.5)
                   * (self.v_hi - self.v_lo) / self.nv)
        for b in range(self.nb):
            n = self.hist[b].sum()
            if n == 0:
                out.append(None)
                continue
            cum = np.cumsum(self.hist[b])
            j = int(np.searchsorted(cum, p / 100.0 * n))
            out.append(float(centers[min(j, self.nv - 1)]))
        return out

    def to_json(self) -> dict:
        n = self.count
        mean = np.where(n > 0, self.sum / np.maximum(n, 1), np.nan)
        var = np.where(n > 1,
                       self.sumsq / np.maximum(n, 1) - mean ** 2, np.nan)
        return {
            "edges": self.edges.tolist(),
            "n": n.tolist(),
            "mean": [None if not np.isfinite(m) else float(m) for m in mean],
            "std": [None if not np.isfinite(v) else float(np.sqrt(max(v, 0.0)))
                    for v in var],
            "p10": self._pct(10), "p25": self._pct(25), "p50": self._pct(50),
            "p75": self._pct(75), "p90": self._pct(90),
        }


class Map2D:
    """Running mean per cell of a fixed-shape grid."""

    def __init__(self, shape: tuple[int, int]):
        self.sum = np.zeros(shape)
        self.count = np.zeros(shape, dtype=np.int64)

    def add(self, ix: np.ndarray, iy: np.ndarray, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float).ravel()
        ok = np.isfinite(values)
        if not ok.any():
            return
        np.add.at(self.sum, (ix[ok], iy[ok]), values[ok])
        np.add.at(self.count, (ix[ok], iy[ok]), 1)

    def to_json(self) -> dict:
        m = np.where(self.count > 0,
                     self.sum / np.maximum(self.count, 1), np.nan)
        return {
            "shape": list(m.shape),
            "mean": [[None if not np.isfinite(v) else float(v) for v in row]
                     for row in m],
            "n": self.count.tolist(),
        }


class CategoryByDist:
    """Counts of a small categorical outcome per distance bin."""

    def __init__(self, dist_edges: np.ndarray, n_cat: int):
        self.edges = np.asarray(dist_edges, dtype=float)
        self.nb = len(self.edges) - 1
        self.counts = np.zeros((self.nb, n_cat), dtype=np.int64)

    def add(self, dist: np.ndarray, cat: np.ndarray) -> None:
        dist = np.asarray(dist, dtype=float).ravel()
        cat = np.asarray(cat, dtype=int).ravel()
        ok = np.isfinite(dist)
        b = np.clip(np.digitize(dist[ok], self.edges) - 1, 0, self.nb - 1)
        np.add.at(self.counts, (b, cat[ok]), 1)

    def to_json(self) -> dict:
        tot = self.counts.sum(axis=1, keepdims=True)
        frac = np.where(tot > 0, self.counts / np.maximum(tot, 1), np.nan)
        return {
            "edges": self.edges.tolist(),
            "counts": self.counts.tolist(),
            "frac": [[None if not np.isfinite(v) else float(v) for v in row]
                     for row in frac],
        }


class Scalars:
    """A named bag of per-world scalars, so across-world spread survives.

    A bare mean is not reportable for these evaluations -- they swing hard
    enough that a directional claim needs the distribution -- so every headline
    number keeps one value per world here.
    """

    def __init__(self):
        self.data: dict[str, list[float]] = {}

    def add(self, name: str, value: float) -> None:
        self.data.setdefault(name, []).append(float(value))

    def to_json(self) -> dict:
        out = {}
        for k, v in self.data.items():
            a = np.asarray(v, dtype=float)
            a = a[np.isfinite(a)]
            out[k] = {
                "values": [float(x) for x in v if np.isfinite(x)],
                "mean": float(a.mean()) if a.size else None,
                "std": float(a.std()) if a.size else None,
                "p25": float(np.percentile(a, 25)) if a.size else None,
                "p50": float(np.percentile(a, 50)) if a.size else None,
                "p75": float(np.percentile(a, 75)) if a.size else None,
                "n": int(a.size),
            }
        return out


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """Signed angle difference into ``[-pi, pi)``.

    Half-open on the *left*: exactly opposite reads as ``-pi``, never ``+pi``.
    Only the measure-zero antipodal case can tell the two apart, and every
    aggregate here takes ``abs`` of this, but the convention is stated because
    a sector or sign-of-error plot would notice.
    """
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def continuous_dist_edges(max_d: float) -> np.ndarray:
    """Distance bins for Test C: much finer near zero, where the story is.

    Uniform-area sampling puts ``proportional to d`` samples in each bin, so
    the near-goal bins carrying the headline are also the sparsest -- which is
    why every curve reports its per-bin ``n`` and why the annulus refinement
    exists.
    """
    fine = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    coarse = list(np.arange(10.0, max_d + 2.0, 2.0))
    return np.array(sorted(set(fine + coarse)), dtype=float)


__all__ = [
    "BinnedStat", "CategoryByDist", "Map2D", "Scalars",
    "continuous_dist_edges", "wrap_to_pi",
]
