"""Place fields of one rectified-ideal code over the whole scaffold (no arenas).

    python -m analysis.rect_ideal.fields --spec SPEC --label LABEL --out DIR

Every unit is evaluated at all 1716 x 1716 scaffold cells. The scaffold is a torus (the code's
frequencies are integers), so fields that cross an edge are joined to their other half.

Two field definitions, both from the same pass:

* ``support``: a connected region (4-neighbour) where the unit is active, ``y > 0``. This is the
  code's own threshold, and the primary definition.
* ``peak20``: a connected region where the unit is at or above 20% of its own peak over the
  scaffold (Kjelstrup 2008's rule, with the whole scaffold as the environment).

Field size is the equivalent diameter ``2 sqrt(area / pi)`` in cells, divided by 12 cells, the
middle grid period (modules 11, 12, 13). Every field is kept, single cells included; summaries
also give the share under 9 cells and the statistics without them.

Outputs ``DIR/<label>.json`` (summaries, histograms, kernel widths, self-check) and
``DIR/<label>_fields.npz`` (unit index and area of every field, fields per unit, coverage).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from analysis.hopfield_probe.ideal_encoder import IdealEncoder, parse_ideal_spec
from analysis.rect_ideal.codes import LAMBDA_REF, LAMBDAS, NPOS, R_IDEAL

PEAK_FRAC = 0.2
MIN_AREA = 9
NB = (NPOS + 7) // 8
DEFS = ("support", "peak20")
LOG_EDGES = np.round(np.arange(-1.2, 2.31, 0.05), 3)       # log10(diameter / 12)


class FastCode:
    """``IdealEncoder.closed_form`` in float32 torch, from a cos/sin table of the integer phase."""

    def __init__(self, enc: IdealEncoder):
        self.n = enc.freqs.cpu().numpy().astype(np.int64)
        amp = enc.amp.cpu().numpy()
        self.amp = torch.from_numpy(np.concatenate([amp, amp]).astype(np.float32))
        ang = 2.0 * math.pi * np.arange(NPOS) / NPOS
        self.cos_t = torch.from_numpy(np.cos(ang).astype(np.float32))
        self.sin_t = torch.from_numpy(np.sin(ang).astype(np.float32))
        self.relu = bool(enc.relu_mix)
        if self.relu:
            self.rotT = torch.from_numpy(enc.rot.cpu().numpy().T.astype(np.float32)).contiguous()
            self.t = float(enc.thresh)
        self.D = enc.out_dim

    def waves(self, xs, ys) -> torch.Tensor:
        k = (np.outer(np.asarray(xs, np.int64), self.n[:, 0])
             + np.outer(np.asarray(ys, np.int64), self.n[:, 1])) % NPOS
        kt = torch.from_numpy(k)
        return torch.cat([self.cos_t[kt], self.sin_t[kt]], dim=1) * self.amp

    def __call__(self, xs, ys) -> torch.Tensor:
        z = self.waves(xs, ys)
        if not self.relu:
            return z
        y = torch.clamp(z @ self.rotT - self.t, min=0.0)
        return y / torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(1e-12)

    def rows(self, x0: int, x1: int) -> torch.Tensor:
        xs = np.repeat(np.arange(x0, x1), NPOS)
        ys = np.tile(np.arange(NPOS), x1 - x0)
        return self(xs, ys)


# --- kernel widths -----------------------------------------------------------

def global_mean(code: FastCode) -> float:
    """``center_global``'s constant: mean activity over 65,536 random cells (seed 12345)."""
    g = np.random.default_rng(12345)
    xs, ys = g.integers(0, NPOS, 65536), g.integers(0, NPOS, 65536)
    tot = 0.0
    for i in range(0, 65536, 8192):
        tot += float(code(xs[i:i + 8192], ys[i:i + 8192]).sum())
    return tot / (65536 * code.D)


def kernel(code: FastCode, c: float = 0.0, n: int = 1500, dmax: int = 64,
           seed: int = 7) -> np.ndarray:
    """Mean cosine between ``p`` and ``p + d e_axis``, both axes averaged; ``c`` subtracted first."""
    p = np.random.default_rng(seed).integers(0, NPOS, (n, 2))

    def enc(xs, ys):
        y = code(xs, ys) - c
        return y / torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(1e-12)

    y0 = enc(p[:, 0], p[:, 1])
    C = np.zeros(dmax + 1)
    for d in range(dmax + 1):
        C[d] = 0.5 * (float((y0 * enc((p[:, 0] + d) % NPOS, p[:, 1])).sum(1).mean())
                      + float((y0 * enc(p[:, 0], (p[:, 1] + d) % NPOS)).sum(1).mean()))
    return C


def floor(code: FastCode, c: float = 0.0, n: int = 8000, seed: int = 11) -> float:
    g = np.random.default_rng(seed)
    a, b = g.integers(0, NPOS, (n, 2)), g.integers(0, NPOS, (n, 2))
    ya, yb = code(a[:, 0], a[:, 1]) - c, code(b[:, 0], b[:, 1]) - c
    ya = ya / torch.linalg.norm(ya, dim=1, keepdim=True).clamp_min(1e-12)
    yb = yb / torch.linalg.norm(yb, dim=1, keepdim=True).clamp_min(1e-12)
    return float((ya * yb).sum(1).mean())


def cross(C: np.ndarray, level: float) -> float | None:
    below = np.flatnonzero(C < level)
    if below.size == 0 or below[0] == 0:
        return None
    i = int(below[0])
    return float(i - 1 + (C[i - 1] - level) / (C[i - 1] - C[i]))


# --- fields ------------------------------------------------------------------

_BITS: dict = {}


def label_unit(job: tuple[str, int]) -> np.ndarray:
    """Areas of one unit's fields, joined across the torus edges."""
    which, u = job
    m = np.unpackbits(_BITS[which][u], axis=1, count=NPOS).astype(bool)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.zeros(0, np.int64)
    a = np.concatenate([lab[0, :], lab[:, 0]])
    b = np.concatenate([lab[-1, :], lab[:, -1]])
    keep = (a > 0) & (b > 0)
    if keep.any():
        g = coo_matrix((np.ones(int(keep.sum())), (a[keep] - 1, b[keep] - 1)), shape=(n, n))
        nc, comp = connected_components(g, directed=False)
    else:
        nc, comp = n, np.arange(n)
    per_label = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    return np.bincount(comp, weights=per_label, minlength=nc).astype(np.int64)


def describe(x) -> dict | None:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std()),
            "min": float(x.min()), "p5": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]),
            "max": float(x.max())}


def summarise(areas_by_unit: list[np.ndarray]) -> tuple[dict, dict]:
    D = len(areas_by_unit)
    counts = np.array([len(a) for a in areas_by_unit])
    counts9 = np.array([int((a >= MIN_AREA).sum()) for a in areas_by_unit])
    unit = np.repeat(np.arange(D), counts).astype(np.int32)
    area = (np.concatenate(areas_by_unit) if counts.sum() else np.zeros(0)).astype(np.int64)
    active = np.array([a.sum() for a in areas_by_unit], dtype=float)
    largest = np.array([a.max() / a.sum() if len(a) else np.nan for a in areas_by_unit])
    diam = 2.0 * np.sqrt(area / math.pi)
    big = area >= MIN_AREA
    lg = np.log10(np.maximum(diam, 1e-9) / LAMBDA_REF)
    hist_n, _ = np.histogram(lg, bins=LOG_EDGES)
    hist_a, _ = np.histogram(lg, bins=LOG_EDGES, weights=area)
    cmax = int(counts.max()) if D else 0
    c_edges = np.unique(np.linspace(0, cmax + 1, 61).round()).astype(int)
    c_hist, _ = np.histogram(counts, bins=c_edges)
    s = {
        "n_fields": int(area.size),
        "fields_per_unit": describe(counts),
        "units_without_field": int((counts == 0).sum()),
        "fields_per_unit_min9": describe(counts9),
        "share_fields_under9": float((~big).mean()) if area.size else None,
        "area_cells": describe(area),
        "diam_cells": describe(diam),
        "diam_over_lambda": describe(diam / LAMBDA_REF),
        "diam_over_lambda_min9": describe(diam[big] / LAMBDA_REF),
        "diam_over_lambda_area_weighted_mean": (
            float((diam * area).sum() / area.sum() / LAMBDA_REF) if area.size else None),
        "coverage_per_unit": describe(active / NPOS ** 2),
        "largest_field_share": describe(largest[np.isfinite(largest)]),
        "hist_log10_diam_over_lambda": {"edges": LOG_EDGES.tolist(),
                                        "fields": hist_n.tolist(),
                                        "cells": hist_a.tolist()},
        "hist_fields_per_unit": {"edges": c_edges.tolist(), "units": c_hist.tolist()},
    }
    arrays = {"unit": unit, "area": area, "fields_per_unit": counts,
              "coverage": active / NPOS ** 2}
    return s, arrays


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=8, help="scaffold rows per chunk")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = ap.parse_args(argv)

    t0 = time.time()
    torch.set_num_threads(args.workers)
    kw = parse_ideal_spec(args.spec)
    if not kw.get("relu_mix") or kw.get("center") or kw.get("center_global"):
        raise SystemExit("fields need a relu_mix code without centring")
    enc = IdealEncoder(**kw).eval()
    code = FastCode(enc)
    D = code.D
    out = {"spec": args.spec, "label": args.label, "D": D, "n_freq": enc.n_freq,
           "r": enc.r, "active": enc.active, "thresh": code.t, "npos": NPOS,
           "lambdas": list(LAMBDAS), "lambda_ref": LAMBDA_REF, "min_area": MIN_AREA,
           "peak_frac": PEAK_FRAC, "connectivity": 4}

    # Self-check against the encoder's own float64 closed form.
    g = np.random.default_rng(3)
    xs, ys = g.integers(0, NPOS, 2000), g.integers(0, NPOS, 2000)
    ref = enc.closed_form(xs, ys)
    mine = code(xs, ys).numpy()
    out["check"] = {"max_abs_diff": float(np.abs(ref - mine).max()),
                    "support_mismatch": float(((ref > 0) != (mine > 0)).mean())}
    if out["check"]["max_abs_diff"] > 1e-4:
        raise SystemExit(f"fast code disagrees with closed_form: {out['check']}")

    # Kernel widths: raw cosine, after the global constant, and the signed ideal at this D.
    c = global_mean(code)
    Craw, Cinh = kernel(code), kernel(code, c)
    ideal = FastCode(IdealEncoder(r=R_IDEAL, n_freq=enc.n_freq, seed=enc.seed).eval())
    Cid = kernel(ideal)
    fl = floor(code)
    out["kernel"] = {
        "inhibition_constant": c, "floor_raw": fl, "floor_inh": floor(code, c),
        "half_width_raw": cross(Craw, 0.5),
        "half_width_raw_to_floor": cross(Craw, 0.5 * (1 + fl)),
        "half_width_inh": cross(Cinh, 0.5),
        "half_width_ideal_r16": cross(Cid, 0.5),
        "curve_raw": Craw.tolist(), "curve_inh": Cinh.tolist(), "curve_ideal_r16": Cid.tolist(),
    }
    t1 = time.time()

    # Pass 1: support bits and every unit's peak. Pass 2: the 20%-of-peak bits.
    bits = {w: np.zeros((D, NPOS, NB), dtype=np.uint8) for w in DEFS}
    peak = torch.zeros(D)

    def put(which, mask: torch.Tensor, x0, x1):
        m = mask.reshape(x1 - x0, NPOS, D).permute(2, 0, 1).contiguous().numpy()
        bits[which][:, x0:x1, :] = np.packbits(m, axis=2)

    for x0 in range(0, NPOS, args.rows):
        x1 = min(x0 + args.rows, NPOS)
        y = code.rows(x0, x1)
        peak = torch.maximum(peak, y.max(0).values)
        put("support", y > 0, x0, x1)
    thr = PEAK_FRAC * peak
    for x0 in range(0, NPOS, args.rows):
        x1 = min(x0 + args.rows, NPOS)
        y = code.rows(x0, x1)
        put("peak20", (y >= thr) & (y > 0), x0, x1)
    out["peak"] = describe(peak.numpy())
    t2 = time.time()

    _BITS.update(bits)
    arrays = {}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for w in DEFS:
            areas = pool.map(label_unit, [(w, u) for u in range(D)], chunksize=8)
            out[w], arr = summarise(areas)
            arrays.update({f"{w}_{k}": v for k, v in arr.items()})
    t3 = time.time()
    out["seconds"] = {"kernel": t1 - t0, "code": t2 - t1, "label": t3 - t2}

    os.makedirs(args.out, exist_ok=True)
    slug = args.label.replace(" ", "_")
    with open(os.path.join(args.out, f"{slug}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(os.path.join(args.out, f"{slug}_fields.npz"), **arrays)
    s = out["support"]
    print(f"{args.label}: D {D}, r {enc.r:.3f}, half-width inh {out['kernel']['half_width_inh']}"
          f" (ideal {out['kernel']['half_width_ideal_r16']}), fields/unit "
          f"{s['fields_per_unit']['mean']:.1f}, diam/12 median "
          f"{s['diam_over_lambda']['median']:.2f}, seconds {out['seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
