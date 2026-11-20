"""Vector-HaSH place cells at matched sparsity, for each grid-module set of the rectified-ideal grid.

    python -m analysis.rect_ideal.hpc fields --variant m111213_onehot   # place fields, all six f
    python -m analysis.rect_ideal.hpc export                            # page data, every variant

Variants are a module set (11, 12, 13 or 11, 15, 23) and an input: ``onehot`` is the layer as the
model builds it; ``smooth`` feeds the same weights the smoothed grid code (Gaussian bumps, FWHM 0.25
of each module's period, the encoder's input), folded into ``W_s = Wpg B`` so that
``Wpg g_smooth = W_s g_onehot`` at every position (``smoothed_wpg``, copied from the tuning viewer's
export) and everything below applies unchanged with ``W_s``.

The place layer is ``VectorHash.build_scaffold``'s: ``p = relu(Wpg g - theta)`` with ``g`` the
one-hot grid code (one active phase cell per module) and ``Wpg`` (Np x Ng) i.i.d. N(0, 1) with
about half the entries pruned (c = 0.5), drawn after ``np.random.seed(0)``; Np = 400, the
controller's setting. ``draw_wpg``, ``phase_sums`` and ``active_fraction`` are copied from
``analysis/tuning_viewer/export.py`` (branch worktree-tuning-viewer), where ``check_hpc_recipe``
matches the draw to ``build_scaffold`` exactly.

The threshold is set per sparsity f (``codes.FS``) so that a share f of all (cell, place) pairs is
active over the whole scaffold. That is exact without visiting the scaffold: the periods are
pairwise coprime and ``Npos`` is their product, so along each axis every combination of module
phases occurs once, and a cell's input over the scaffold is its weight sum over every combination
of one phase per module. One global threshold, as in Vector-HaSH, so cells differ in how much
they fire.

Fields use the same definitions and summaries as ``fields.py`` (support and 20% of peak, joined
across the torus edges, equivalent diameter over the variant's grid period: 12 cells for 11, 12,
13 and the geometric mean, 15.6 cells, for 11, 15, 23). The pass is a copy of ``fields.py``'s.
Results: ``$CLS_RESULTS/rect_ideal_grid/hpc/``.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from analysis.hopfield_probe.encode import grid_codes
from analysis.rect_ideal import fields as F
from analysis.rect_ideal.codes import FS, VARIANTS, f_tag
from cls_paths import results_dir

HPC_MODULES = {
    "m111213": {"lambdas": (11, 12, 13), "lambda_ref": VARIANTS["m111213"]["lambda_ref"],
                "label": "11, 12, 13"},
    "m111523": {"lambdas": (11, 15, 23), "lambda_ref": VARIANTS["m111523_r16"]["lambda_ref"],
                "label": "11, 15, 23"},
}
INPUTS = ("onehot", "smooth")
HPC_VARIANTS = {f"{m}_{i}": dict(v, modules=m, input=i) for i in INPUTS for m, v in HPC_MODULES.items()}
NP, C, SEED, FWHM = 400, 0.5, 0, 0.25
REF = {"x0": 1708, "y0": 1710, "S": 16}
N_CHECK = 16


def results_root():
    return results_dir() / "rect_ideal_grid" / "hpc"


# --- the place layer (copied from analysis/tuning_viewer/export.py) ----------

def draw_wpg(Np: int, Ng: int, c: float, seed: int) -> np.ndarray:
    """``VectorHash.build_scaffold``'s Wpg draw, after ``np.random.seed(seed)``."""
    np.random.seed(seed)
    W = np.random.randn(Np, Ng)
    prune = int((1 - c) * Np * Ng)
    mask = np.ones((Np, Ng))
    mask[np.random.randint(0, Np, prune), np.random.randint(0, Ng, prune)] = 0
    return mask * W


def phase_sums(W: np.ndarray, lambdas) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per cell: sorted sums over every phase of all modules but the last, and the last module's weights."""
    offs = np.cumsum([0] + [l * l for l in lambdas])
    out = []
    for row in W:
        rest = np.zeros(1)
        for m in range(len(lambdas) - 1):
            rest = (rest[:, None] + row[offs[m]:offs[m + 1]][None, :]).ravel()
        out.append((np.sort(rest), row[offs[-2]:offs[-1]]))
    return out


def active_fraction(sums, thresh: float) -> float:
    """Fraction of (cell, place) pairs with ``W g_onehot > thresh`` over the whole scaffold, exactly."""
    above = sum(int((a.size - np.searchsorted(a, thresh - b, side="right")).sum()) for a, b in sums)
    return above / (len(sums) * sums[0][0].size * sums[0][1].size)


def matched_thresh(sums, target: float) -> float:
    """The threshold at which ``active_fraction`` is ``target``, by bisection over the whole range."""
    lo = float(min(a[0] + b.min() for a, b in sums)) - 1e-9
    hi = float(max(a[-1] + b.max() for a, b in sums)) + 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if active_fraction(sums, mid) > target else (lo, mid)
    return 0.5 * (lo + hi)


def smoothed_wpg(Wpg: np.ndarray, lambdas, fwhm: float) -> np.ndarray:
    """``W_s`` with ``Wpg @ grid_codes(.., fwhm) == W_s @ grid_codes(.., 0)`` at every position.

    Module ``m``'s block of the smoothed code depends only on that module's phase, so it is
    ``B_m e_phase`` with ``B_m[r, c]`` the block's unit ``r`` at phase ``c``.
    """
    Ws, off = np.array(Wpg, dtype=np.float64), 0
    for l in lambdas:
        i, j = np.meshgrid(np.arange(l), np.arange(l), indexing="ij")
        B = grid_codes(lambdas, i.ravel(), j.ravel(), fwhm).astype(np.float64)[:, off:off + l * l].T
        Ws[:, off:off + l * l] = Wpg[:, off:off + l * l] @ B
        off += l * l
    return Ws


def layer(variant: str):
    """``(lambdas, W_eff, Wpg)``: the weights the one-hot arithmetic uses, and the drawn ``Wpg``."""
    v = HPC_VARIANTS[variant]
    lam = v["lambdas"]
    W = draw_wpg(NP, sum(l * l for l in lam), C, SEED)
    return lam, (smoothed_wpg(W, lam, FWHM) if v["input"] == "smooth" else W), W


def rates_np(W, lambdas, thresh, xs, ys) -> np.ndarray:
    """``relu(Wpg g - thresh)`` at scaffold cells in float64, g one-hot (the page's arithmetic)."""
    xs, ys = np.asarray(xs, np.int64), np.asarray(ys, np.int64)
    off, tot = 0, 0.0
    for l in lambdas:
        tot = tot + W[:, off + (xs % l) * l + (ys % l)].T
        off += l * l
    return np.maximum(tot - thresh, 0.0)


class HpcCode:
    """The place layer in float64 torch, with ``fields.FastCode``'s interface."""

    def __init__(self, W: np.ndarray, lambdas, thresh: float):
        self.WT = torch.from_numpy(np.ascontiguousarray(W.T))        # (Ng, Np)
        self.lambdas = [int(l) for l in lambdas]
        self.offs = np.cumsum([0] + [l * l for l in self.lambdas])[:-1]
        self.t = float(thresh)
        self.D = W.shape[0]

    def __call__(self, xs, ys) -> torch.Tensor:
        xs, ys = np.asarray(xs, np.int64), np.asarray(ys, np.int64)
        tot = None
        for l, o in zip(self.lambdas, self.offs):
            v = self.WT[torch.from_numpy(o + (xs % l) * l + (ys % l))]
            tot = v if tot is None else tot + v
        return torch.clamp(tot - self.t, min=0.0)

    def rows(self, x0: int, x1: int) -> torch.Tensor:
        return self(np.repeat(np.arange(x0, x1), F.NPOS), np.tile(np.arange(F.NPOS), x1 - x0))


def label_unit_hist(job):
    """One cell's fields as an area histogram: a place cell can have a million single-cell fields
    on the 11, 15, 23 scaffold, so the per-field areas ``fields.py`` keeps do not fit in memory."""
    areas = F.label_unit(job)
    vals, cnts = np.unique(areas, return_counts=True)
    return (vals, cnts, int(areas.size), int(areas.sum()), int(areas.max()) if areas.size else 0,
            int((areas >= F.MIN_AREA).sum()))


def describe_w(vals, cnts) -> dict | None:
    """``fields.describe`` of the values ``vals`` (ascending) repeated ``cnts`` times, exactly,
    with numpy's default (linear) percentiles."""
    vals, cnts = np.asarray(vals, float), np.asarray(cnts, np.int64)
    N = int(cnts.sum())
    if N == 0:
        return None
    mean = float((vals * cnts).sum() / N)
    cum = np.cumsum(cnts)

    def pct(q):
        pos = q / 100 * (N - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, N - 1)
        vlo, vhi = vals[np.searchsorted(cum, lo + 1)], vals[np.searchsorted(cum, hi + 1)]
        return float(vlo + (pos - lo) * (vhi - vlo))
    return {"n": N, "mean": mean, "sd": float(math.sqrt(((vals - mean) ** 2 * cnts).sum() / N)),
            "min": float(vals[0]), "p5": pct(5), "p25": pct(25), "median": pct(50),
            "p75": pct(75), "p95": pct(95), "max": float(vals[-1])}


def summarise_hist(per_unit) -> tuple[dict, dict]:
    """``fields.summarise``'s summary, from per-cell area histograms."""
    counts = np.array([u[2] for u in per_unit])
    counts9 = np.array([u[5] for u in per_unit])
    active = np.array([u[3] for u in per_unit], dtype=float)
    largest = np.array([u[4] / u[3] if u[2] else np.nan for u in per_unit])
    allv = np.concatenate([u[0] for u in per_unit]) if counts.sum() else np.zeros(0, np.int64)
    allc = np.concatenate([u[1] for u in per_unit]) if counts.sum() else np.zeros(0, np.int64)
    A, inv = np.unique(allv, return_inverse=True)
    Cn = np.bincount(inv, weights=allc, minlength=A.size).astype(np.int64)
    diam = 2.0 * np.sqrt(A / math.pi)
    big = A >= F.MIN_AREA
    lg = np.log10(np.maximum(diam, 1e-9) / F.LAMBDA_REF)
    hist_n, _ = np.histogram(lg, bins=F.LOG_EDGES, weights=Cn)
    hist_a, _ = np.histogram(lg, bins=F.LOG_EDGES, weights=Cn * A)
    cmax = int(counts.max()) if counts.size else 0
    c_edges = np.unique(np.linspace(0, cmax + 1, 61).round()).astype(int)
    c_hist, _ = np.histogram(counts, bins=c_edges)
    n_fields = int(Cn.sum())
    s = {
        "n_fields": n_fields,
        "fields_per_unit": F.describe(counts),
        "units_without_field": int((counts == 0).sum()),
        "fields_per_unit_min9": F.describe(counts9),
        "share_fields_under9": float(Cn[~big].sum() / n_fields) if n_fields else None,
        "area_cells": describe_w(A, Cn),
        "diam_cells": describe_w(diam, Cn),
        "diam_over_lambda": describe_w(diam / F.LAMBDA_REF, Cn),
        "diam_over_lambda_min9": describe_w(diam[big] / F.LAMBDA_REF, Cn[big]),
        "diam_over_lambda_area_weighted_mean": (
            float((diam * A * Cn).sum() / (A * Cn).sum() / F.LAMBDA_REF) if n_fields else None),
        "coverage_per_unit": F.describe(active / F.NPOS ** 2),
        "largest_field_share": F.describe(largest[np.isfinite(largest)]),
        "hist_log10_diam_over_lambda": {"edges": F.LOG_EDGES.tolist(),
                                        "fields": hist_n.round().astype(int).tolist(),
                                        "cells": hist_a.round().astype(int).tolist()},
        "hist_fields_per_unit": {"edges": c_edges.tolist(), "units": c_hist.tolist()},
    }
    arrays = {"fields_per_unit": counts, "coverage": active / F.NPOS ** 2,
              "area_values": A, "area_counts": Cn}
    return s, arrays


def field_pass(code, D: int, rows: int, workers: int, out: dict) -> dict:
    """``fields.py``'s two passes and labelling (support, 20% of peak), into ``out``; the summaries
    come from per-cell area histograms (``summarise_hist``)."""
    N, NB = F.NPOS, F.NB
    bits = {w: np.zeros((D, N, NB), dtype=np.uint8) for w in F.DEFS}
    peak = torch.zeros(D, dtype=torch.float64)

    def put(which, mask, x0, x1):
        m = mask.reshape(x1 - x0, N, D).permute(2, 0, 1).contiguous().numpy()
        bits[which][:, x0:x1, :] = np.packbits(m, axis=2)

    for x0 in range(0, N, rows):
        x1 = min(x0 + rows, N)
        y = code.rows(x0, x1)
        peak = torch.maximum(peak, y.max(0).values)
        put("support", y > 0, x0, x1)
    thr = F.PEAK_FRAC * peak
    for x0 in range(0, N, rows):
        x1 = min(x0 + rows, N)
        y = code.rows(x0, x1)
        put("peak20", (y >= thr) & (y > 0), x0, x1)
    out["peak"] = F.describe(peak.numpy())
    F._BITS.clear()
    F._BITS.update(bits)
    arrays = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for w in F.DEFS:
            per_unit = pool.map(label_unit_hist, [(w, u) for u in range(D)], chunksize=8)
            out[w], arr = summarise_hist(per_unit)
            arrays.update({f"{w}_{k}": v for k, v in arr.items()})
    F._BITS.clear()
    return arrays


def run_fields(variant: str, out_dir: str, rows: int, workers: int) -> None:
    torch.set_num_threads(workers)
    lam, W, _ = layer(variant)
    N = int(np.prod(lam))
    F.NPOS, F.NB, F.LAMBDAS = N, (N + 7) // 8, lam
    F.LAMBDA_REF = HPC_VARIANTS[variant]["lambda_ref"]
    sums = phase_sums(W, lam)
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for f in FS:
        t0 = time.time()
        th = matched_thresh(sums, f)
        code = HpcCode(W, lam, th)
        out = {"variant": variant, "input": HPC_VARIANTS[variant]["input"], "fwhm": FWHM,
               "label": f"vhpc {f_tag(f)} {variant}", "f": f, "thresh": th,
               "active_exact": active_fraction(sums, th), "Np": NP, "c": C, "seed": SEED,
               "lambdas": list(lam), "npos": N, "lambda_ref": F.LAMBDA_REF,
               "min_area": F.MIN_AREA, "peak_frac": F.PEAK_FRAC, "connectivity": 4}
        g = np.random.default_rng(3)
        xs, ys = g.integers(0, N, 2000), g.integers(0, N, 2000)
        out["check"] = {"max_abs_diff": float(np.abs(rates_np(W, lam, th, xs, ys) - code(xs, ys).numpy()).max())}
        c = F.global_mean(code)
        Craw, Cinh = F.kernel(code), F.kernel(code, c)
        fl = F.floor(code)
        out["kernel"] = {"inhibition_constant": c, "floor_raw": fl, "floor_inh": F.floor(code, c),
                         "half_width_raw": F.cross(Craw, 0.5),
                         "half_width_raw_to_floor": F.cross(Craw, 0.5 * (1 + fl)),
                         "half_width_inh": F.cross(Cinh, 0.5),
                         "curve_raw": Craw.tolist(), "curve_inh": Cinh.tolist()}
        t1 = time.time()
        arrays = field_pass(code, NP, rows, workers, out)
        out["seconds"] = {"kernel": t1 - t0, "fields": time.time() - t1}
        slug = f"vhpc_{variant}_{f_tag(f)}"
        with open(os.path.join(out_dir, f"{slug}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        np.savez_compressed(os.path.join(out_dir, f"{slug}_fields.npz"), **arrays)
        summary.append({k: v for k, v in out.items()})
        s = out["support"]
        print(f"{slug}: thresh {th:.4f}, active {out['active_exact']:.4f} (coverage "
              f"{s['coverage_per_unit']['mean']:.4f}), fields/unit {s['fields_per_unit']['mean']:.1f}, "
              f"diam median {s['diam_over_lambda']['median']:.3f} periods, kernel half-width "
              f"{out['kernel']['half_width_raw']}, {out['seconds']}", flush=True)
    with open(os.path.join(out_dir, f"hpc_results_{variant}.json"), "w") as fh:
        json.dump(summary, fh)


def export(out: str) -> None:
    """Page data for every variant: weights, thresholds, Python check values and the results."""
    os.makedirs(out, exist_ok=True)
    r_, c_ = np.divmod(np.arange(REF["S"] ** 2), REF["S"])
    index = []
    for variant, v in HPC_VARIANTS.items():
        lam, W, Wpg = layer(variant)
        N = int(np.prod(lam))
        xs, ys = (REF["x0"] + c_) % N, (REF["y0"] + r_) % N
        # Check values straight from the model's arithmetic (the smoothed code times Wpg for the
        # smoothed input), so the page's check also covers the fold into W_s.
        g = grid_codes(lam, xs, ys, FWHM if v["input"] == "smooth" else 0.0).astype(np.float64) @ Wpg.T
        sums = phase_sums(W, lam)
        units = np.unique(np.round(np.linspace(0, NP - 1, N_CHECK)).astype(int))
        codes, refs = [], []
        for f in FS:
            th = matched_thresh(sums, f)
            refs.append(np.maximum(g - th, 0.0)[:, units].astype(np.float32).ravel())
            codes.append({"key": f_tag(f), "f": f, "thresh": th, "active_exact": active_fraction(sums, th)})
        with open(os.path.join(out, f"hpc_wpg_{variant}.b64.txt"), "w") as fh:
            fh.write(base64.b64encode(np.ascontiguousarray(W, dtype="<f8").tobytes()).decode())
        with open(os.path.join(out, f"hpc_ref_{variant}.b64.txt"), "w") as fh:
            fh.write(base64.b64encode(np.concatenate(refs).astype("<f4").tobytes()).decode())
        man = {"variant": variant, "modules": v["modules"], "input": v["input"], "fwhm": FWHM,
               "label": v["label"], "lambdas": list(lam), "npos": N,
               "lambda_ref": v["lambda_ref"], "Np": NP, "Ng": int(W.shape[1]), "c": C, "seed": SEED,
               "Fs": list(FS), "codes": codes, "check_units": units.tolist(), "ref_arena": REF}
        with open(os.path.join(out, f"hpc_manifest_{variant}.json"), "w") as fh:
            json.dump(man, fh, indent=1)
        res = results_root() / f"hpc_results_{variant}.json"
        dest = os.path.join(out, f"hpc_results_{variant}.json")
        if res.exists():
            rows = json.load(open(res))
            for row in rows:                       # the page needs no kernel curves
                row["kernel"] = {k: x for k, x in row["kernel"].items() if not k.startswith("curve")}
            with open(dest, "w") as fh:
                json.dump(rows, fh)
        index.append({"key": variant, "modules": v["modules"], "input": v["input"],
                      "label": v["label"], "npos": N, "lambda_ref": v["lambda_ref"],
                      "has_results": os.path.exists(dest)})
        print(variant, [(c["key"], round(c["thresh"], 4), round(c["active_exact"], 5)) for c in codes])
    with open(os.path.join(out, "hpc_variants.json"), "w") as fh:
        json.dump(index, fh, indent=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fields")
    a.add_argument("--variant", required=True, choices=sorted(HPC_VARIANTS))
    a.add_argument("--out", default=None)
    a.add_argument("--rows", type=int, default=8)
    a.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    b = sub.add_parser("export")
    here = os.path.dirname(os.path.abspath(__file__))
    b.add_argument("--out", default=os.path.join(here, "page", "data"))
    args = ap.parse_args(argv)
    if args.cmd == "fields":
        run_fields(args.variant, args.out or str(results_root()), args.rows, args.workers)
    else:
        export(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
