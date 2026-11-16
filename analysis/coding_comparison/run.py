"""Run the place-cell literature's measurements on model populations.

    python -m analysis.coding_comparison.run --pops grid,ideal [--pop name=spec] [--tag v1] [--quick]

Writes one JSON per measure to ``<results>/coding_comparison/<tag>/<pop>/``
(``cls_paths.results_dir``):

``meta.json``        the population, every parameter, rate scales, and the
                     mean field width ``W`` of each analysis variant
``fields_1d.json``   field counts on zigzag tracks of several lengths
``fields_2d.json``   field counts in square arenas whose areas double
``kernel_1d.json``   PV similarity against separation along scaffold loops
``kernel_2d.json``   PV similarity against distance in 2D
``remapping.json``   arenas at different scaffold locations, compared
``sparsity.json``    Treves-Rolls sparsity of each unit's rate map

An analysis variant is a basis (the native one, or a random rotation of a
signed code), a rate convention (identity for non-negative codes; rectify or
on/off split for signed ones) and a field definition:

``A``      20% of the unit's peak in that environment, at least 3 cells
           (9 in 2D), no smoothing.
``B<r>``   Rich 2014: one global rate scale that makes the population mean
           rate ``r`` Hz over random scaffold cells, Gaussian smoothing with SD
           (10/65) W, a 2 Hz threshold and a minimum length (15/65) W, where W
           is the variant's own mean field width. Because W depends on the
           smoothing, W is found by fixed-point iteration from W_A.

Lengths are then expressed in units of the variant's own W, and areas in
units of its own mean field area.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cls_paths import results_dir

from . import geometry as geo
from . import measures as M
from .populations import POPULATIONS, Population, make_population

HERE = Path(__file__).resolve().parent

RICH = {"field_m": 0.65, "min_len_m": 0.15, "smooth_m": 0.10, "thresh_hz": 2.0}
# Rich 2014's track stages, 3, 10, 22 and 48 m, and twice the last, in units
# of their 0.65 m mean field.
STAGES_W = [3 / 0.65, 10 / 0.65, 22 / 0.65, 48 / 0.65, 96 / 0.65]

PARAMS = {
    "stages_W": STAGES_W,
    "def_A": {"frac": 0.2, "min_len": 3, "min_area": 9},
    "rich": RICH,
    "mean_rates": [0.5, 0.75, 1.0],
    "rot_seeds": [1, 2, 3],
    "def_Ap": {"min_peak_hz": 1.0, "rate": "0.75"},
    "rot_defs": ["A", "Ap", "B0.75"],
    "n_scale_cells": 16384,
    "n_loops": 8,
    "n_tracks": 16,
    "gap_W": 4.0,
    "area_steps": [2, 4, 8, 16, 32, 64, 128],
    "n_arena_locs": 2,
    "arena_side_cap": 1024,
    "kernel_far_W": [5.0, 15.0],
    "ring_anchors": 96, "ring_angles": 16, "ring_radii": 48, "ring_max_W": 20.0,
    "remap_n": 8, "remap_area_A0": 8, "remap_side_cap": 256,
    "b_iters": 12, "b_tol": 0.01,
    "seed": 0,
}
QUICK = {"n_loops": 2, "n_tracks": 4, "area_steps": [4, 8, 16], "n_arena_locs": 1,
         "rot_seeds": [1], "ring_anchors": 16, "ring_radii": 16, "remap_n": 3,
         "arena_side_cap": 192, "remap_side_cap": 64, "n_scale_cells": 4096}


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    rot: int
    conv: str
    defn: str

    @property
    def basis(self) -> tuple[int, str]:
        return (self.rot, self.conv)

    @property
    def key(self) -> str:
        return (f"rot{self.rot}|" if self.rot else "") + f"{self.conv}|{self.defn}"


def basis_key(rot: int, conv: str) -> str:
    return (f"rot{rot}|" if rot else "") + conv


def def_names(P) -> list[str]:
    return ["A", "Ap"] + [f"B{r:g}" for r in P["mean_rates"]]


def variants_for(pop: Population, P) -> list[Variant]:
    convs = ["rectify", "onoff"] if pop.signed else ["identity"]
    out = [Variant(0, c, d) for c in convs for d in def_names(P)]
    if pop.signed:
        out += [Variant(s, c, d) for s in P["rot_seeds"] for c in convs for d in P["rot_defs"]]
    return out


def to_basis(z: np.ndarray, Q: np.ndarray | None) -> np.ndarray:
    return z if Q is None else (z.astype(np.float64) @ Q.T).astype(np.float32)


def rates_of(z_PU: np.ndarray, conv: str) -> np.ndarray:
    """``(positions, units)`` activity to ``(cells, positions)`` rates."""
    return M.apply_convention(np.ascontiguousarray(z_PU.T), conv, axis=0)


def def_A(P) -> M.FieldDef:
    a = P["def_A"]
    return M.FieldDef("A", "peak_fraction", frac=a["frac"], min_len=a["min_len"],
                      min_area=a["min_area"])


def def_B(scale: float, W: float) -> M.FieldDef:
    ml = RICH["min_len_m"] / RICH["field_m"] * W
    return M.FieldDef("B", "absolute", thresh_hz=RICH["thresh_hz"], rate_scale=scale,
                      smooth_sd=RICH["smooth_m"] / RICH["field_m"] * W,
                      min_len=max(1.0, ml), min_area=max(1.0, ml * ml))


def fdef_from(d: dict) -> M.FieldDef:
    return M.FieldDef(**d)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def clean(o):
    """JSON-safe: numpy scalars to Python, non-finite floats to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write(out: Path, name: str, obj) -> None:
    with open(out / f"{name}.json", "w") as f:
        json.dump(clean(obj), f, indent=1)


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()
    try:
        return {"sha": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--", "analysis/coding_comparison"))}
    except OSError:
        return {"sha": None}


def place(n: int, side: int, npos: int, sep: int, seed: int) -> dict:
    """Arena corners: ``sep`` cells apart if possible, else touching, else spaced on the diagonal."""
    for s in (sep, 0):
        try:
            offs = geo.spread_offsets(n, side, npos, s, seed)
            return {"offsets": offs, "sep": s, "disjoint": True}
        except RuntimeError:
            continue
    offs = geo.spread_offsets(n, side, npos, 0, seed, strict=False)
    return {"offsets": offs, "sep": None, "disjoint": geo.spread_ok(offs, side, npos)}


def nanmean(x) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


# ---------------------------------------------------------------------------
# Calibration: rate scales and mean field widths
# ---------------------------------------------------------------------------

def calibrate(pop: Population, P, bases: dict, variants: list[Variant], rng, t0) -> dict:
    npos = pop.npos
    xs, ys = rng.integers(0, npos, P["n_scale_cells"]), rng.integers(0, npos, P["n_scale_cells"])
    z = pop.code(xs, ys)
    scales = {}
    for rot, conv in sorted({v.basis for v in variants}):
        m = float(rates_of(to_basis(z, bases[rot]), conv).mean())
        scales[basis_key(rot, conv)] = {"mean_activity": m,
                                        "Hz_per_unit": {f"{r:g}": (r / m if m > 0 else None)
                                                        for r in P["mean_rates"]}}
    rows = rng.integers(0, npos, P["n_loops"])
    loops = [pop.code(*geo.loop(npos, i % 2, int(rows[i]))) for i in range(P["n_loops"])]
    calib = {}
    for rot, conv in sorted({v.basis for v in variants}):
        R_loops = [rates_of(to_basis(zl, bases[rot]), conv) for zl in loops]
        fa = def_A(P)
        la = np.concatenate([M.detect_fields_1d(R, fa).length for R in R_loops])
        WA = float(la.mean()) if la.size else None
        calib[Variant(rot, conv, "A").key] = {
            "W": WA, "W_median": float(np.median(la)) if la.size else None,
            "n_fields": int(la.size), "fdef": fa.to_dict()}
        for v in variants:
            if v.basis != (rot, conv) or v.defn == "A":
                continue
            hz = scales[basis_key(rot, conv)]["Hz_per_unit"]
            if v.defn == "Ap":
                calib[v.key] = calibrate_Ap(R_loops, hz[P["def_Ap"]["rate"]], P)
                continue
            calib[v.key] = calibrate_B(R_loops, hz[v.defn[1:]], WA, P)
        log(f"calibrated {basis_key(rot, conv)}: W_A={WA}", t0)
    return {"scales": scales, "calib": calib, "loops": loops, "loop_rows": rows.tolist()}


def def_Ap(scale: float, P) -> M.FieldDef:
    """Eliav 2021: 20% of the field peak, field peaks above 1 Hz (here at the 0.75 Hz scale)."""
    return def_A(P).with_(name="Ap", rate_scale=scale, min_peak_hz=P["def_Ap"]["min_peak_hz"])


def calibrate_Ap(R_loops, scale, P) -> dict:
    if scale is None:
        return {"W": None, "fdef": None}
    fdef = def_Ap(scale, P)
    lens = np.concatenate([M.detect_fields_1d(R, fdef).length for R in R_loops])
    return {"W": float(lens.mean()) if lens.size else None,
            "W_median": float(np.median(lens)) if lens.size else None,
            "n_fields": int(lens.size), "fdef": fdef.to_dict()}


def calibrate_B(R_loops, scale, W0, P) -> dict:
    if scale is None or W0 is None:
        return {"W": None, "fdef": None, "history": [], "converged": False}
    W, hist, converged = float(W0), [], False
    for _ in range(P["b_iters"]):
        fdef = def_B(scale, W)
        lens = np.concatenate([M.detect_fields_1d(R, fdef).length for R in R_loops])
        if lens.size == 0:
            hist.append({"W_in": W, "W_out": None, "n_fields": 0})
            return {"W": None, "fdef": fdef.to_dict(), "history": hist, "converged": False}
        Wn = float(lens.mean())
        hist.append({"W_in": W, "W_out": Wn, "n_fields": int(lens.size)})
        if abs(Wn - W) <= P["b_tol"] * W:
            converged = True
            break
        W = Wn
    fdef = def_B(scale, W)
    lens = np.concatenate([M.detect_fields_1d(R, fdef).length for R in R_loops])
    return {"W": float(lens.mean()) if lens.size else None,
            "W_median": float(np.median(lens)) if lens.size else None,
            "n_fields": int(lens.size), "fdef": fdef.to_dict(), "history": hist,
            "converged": converged}


# ---------------------------------------------------------------------------
# 1D field counts on zigzag tracks
# ---------------------------------------------------------------------------

def fields_1d(pop, P, bases, variants, cal, W_ref, rng, t0) -> tuple[dict, dict]:
    npos = pop.npos
    gap = int(np.clip(round(P["gap_W"] * W_ref), 16, npos // 6))
    cap = geo.serpentine_capacity(npos, npos, gap)
    want = [round(max(P["stages_W"]) * cal[v.key]["W"]) for v in variants
            if cal[v.key]["W"] is not None]
    need = int(min(cap, max(want))) if want else 0
    acc: dict = {v.key: {} for v in variants}
    sp_acc: dict = {}
    track_meta = []
    for i in range(P["n_tracks"]):
        axis = i % 2
        x0, y0 = (int(c) for c in rng.integers(0, npos, 2))
        xs, ys = geo.serpentine(x0, y0, need, run=npos, gap=gap, axis=axis)
        track_meta.append({"x0": x0, "y0": y0, "axis": axis})
        z = pop.code(xs, ys)
        for rot in sorted({v.rot for v in variants}):
            zr = to_basis(z, bases[rot])
            for conv in sorted({v.conv for v in variants if v.rot == rot}):
                R = rates_of(zr, conv)
                WA = cal[Variant(rot, conv, "A").key]["W"]
                for LW in P["stages_W"]:
                    if WA is None:
                        continue
                    Lc = int(round(LW * WA))
                    if 1 <= Lc <= need:
                        s = M.sparsity(R[:, :Lc])
                        sp_acc.setdefault(basis_key(rot, conv), {}).setdefault(
                            f"{LW:.1f}", []).append(nanmean(s))
                for v in variants:
                    if v.basis != (rot, conv) or cal[v.key]["W"] is None:
                        continue
                    W = cal[v.key]["W"]
                    fdef = fdef_from(cal[v.key]["fdef"])
                    for LW in P["stages_W"]:
                        Lc = int(round(LW * W))
                        if Lc < 1 or Lc > need:
                            continue
                        F = M.detect_fields_1d(R[:, :Lc], fdef)
                        a = acc[v.key].setdefault(f"{LW:.1f}", {
                            "length_W": LW, "length_cells": Lc, "counts": [], "lengths": [],
                            "layouts": [], "ratios": [], "coverage": [], "unit_cov": []})
                        a["counts"].append(F.counts)
                        a["lengths"].append(F.length)
                        a["layouts"].append(M.unit_layouts(F))
                        a["ratios"].extend(M.unit_size_ratios(F.unit, F.length))
                        cov = F.coverage()
                        a["coverage"].append(float(cov.mean()))
        log(f"track {i + 1}/{P['n_tracks']} ({need} cells)", t0)

    out = {"gap_cells": gap, "capacity_cells": cap, "track_cells": need, "tracks": track_meta,
           "variants": {}}
    for v in variants:
        W = cal[v.key]["W"]
        res = {"W": W, "fdef": cal[v.key]["fdef"], "lengths": {}, "skipped_W": []}
        for LW in P["stages_W"]:
            a = acc[v.key].get(f"{LW:.1f}")
            if a is None:
                res["skipped_W"].append(LW)
                continue
            counts = np.concatenate(a["counts"])
            per_track = [float(np.mean(c == 0)) for c in a["counts"]]
            lens = np.concatenate(a["lengths"]).astype(np.float64)
            active = counts > 0
            n_act = int(active.sum())
            fs = M.field_size_stats(lens)
            if fs.get("n"):
                fs["mean_W"] = fs["mean"] / W
            res["lengths"][f"{LW:.1f}"] = {
                "length_W": LW, "length_cells": a["length_cells"], "n_tracks": len(a["counts"]),
                "n_samples": int(counts.size),
                "fit": M.fit_count_models(counts),
                "count_hist": np.bincount(counts).tolist(),
                "frac_zero_per_track": per_track,
                "mean_per_track": [float(np.mean(c)) for c in a["counts"]],
                "active_frac": float(active.mean()),
                "fields_per_active": float(counts[active].mean()) if n_act else None,
                "single_field_frac_active": float(np.mean(counts[active] == 1)) if n_act else None,
                "five_plus_frac_active": float(np.mean(counts[active] >= 5)) if n_act else None,
                "field_size": fs,
                "layout": M.summarize_layouts(a["layouts"]),
                "size_ratio": M.summarize_ratios(a["ratios"]),
                "coverage_per_location": float(np.mean(a["coverage"])),
            }
        out["variants"][v.key] = res
    sp = {"by_basis": {k: {LW: nanmean(vals) for LW, vals in d.items()} for k, d in sp_acc.items()},
          "note": "mean over active units of each unit's Treves-Rolls sparsity on the track, "
                  "lengths in units of W_A"}
    return out, sp


# ---------------------------------------------------------------------------
# 2D arenas
# ---------------------------------------------------------------------------

def _arena_fields(zr: np.ndarray, side: int, conv: str, vlist, cal, block: int = 128):
    """Detect fields in one arena for every variant of one basis and convention."""
    P_, U = zr.shape
    nc = 2 * U if conv == "onoff" else U
    per_v = {v.key: {"counts": np.zeros(nc, np.int64), "areas": [], "cover": 0.0,
                     "unit_cov": np.zeros(nc)} for v in vlist}
    spars = np.full(nc, np.nan)
    for b0 in range(0, U, block):
        zb = zr[:, b0:b0 + block]
        Rb = rates_of(zb, conv)                                 # (b', P)
        nb = zb.shape[1]
        idx = np.arange(b0, b0 + nb)
        if conv == "onoff":
            idx = np.concatenate([idx, U + idx])
        spars[idx] = M.sparsity(Rb)
        Rb = Rb.reshape(-1, side, side)
        for v in vlist:
            F = M.detect_fields_2d(Rb, fdef_from(cal[v.key]["fdef"]), block=block)
            d = per_v[v.key]
            d["counts"][idx] += F.counts
            d["areas"].append(F.area)
            d["cover"] += float(F.coverage_frac.mean()) * Rb.shape[0]
            d["unit_cov"][idx] = F.unit_coverage
    for d in per_v.values():
        d["cover"] /= nc
    return per_v, spars


def fields_2d(pop, P, bases, variants, cal, W_ref, rng, t0) -> tuple[dict, dict]:
    npos = pop.npos
    A0 = math.pi / 4 * W_ref ** 2
    sizes, skipped = [], []
    for k in P["area_steps"]:
        side = int(round(math.sqrt(k * A0)))
        (sizes if 3 <= side <= P["arena_side_cap"] else skipped).append((k, side))
    acc: dict = {}
    sp_acc: dict = {}
    placements: dict = {}
    for k, side in sizes:
        pl = place(P["n_arena_locs"], side, npos, int(2 * W_ref) + 1, int(rng.integers(1 << 30)))
        placements[k] = {"sep": pl["sep"], "disjoint": pl["disjoint"]}
        for (x0, y0) in pl["offsets"]:
            z = pop.code(*geo.arena(x0, y0, side))
            for rot in sorted({v.rot for v in variants}):
                zr = to_basis(z, bases[rot])
                for conv in sorted({v.conv for v in variants if v.rot == rot}):
                    vl = [v for v in variants if v.basis == (rot, conv) and cal[v.key]["W"] is not None]
                    per_v, spars = _arena_fields(zr, side, conv, vl, cal)
                    sp_acc.setdefault(basis_key(rot, conv), {}).setdefault(str(k), []).append(
                        nanmean(spars))
                    for v in vl:
                        d = per_v[v.key]
                        acc.setdefault(v.key, {}).setdefault(k, []).append(
                            {"side": side, "x0": x0, "y0": y0, "counts": d["counts"],
                             "areas": np.concatenate(d["areas"]) if d["areas"] else np.zeros(0),
                             "cover": d["cover"], "unit_cov": d["unit_cov"]})
                del zr
            log(f"arena {k} A0 (side {side}) at ({x0},{y0})", t0)

    out = {"A0_cells": A0, "A0_rule": "pi/4 * W_ref^2, W_ref the primary variant's W_A",
           "W_ref": W_ref, "sizes": [{"k": k, "side": s} for k, s in sizes],
           "skipped_sizes": [{"k": k, "side": s} for k, s in skipped],
           "placements": placements, "variants": {}}
    for v in variants:
        if v.key not in acc:
            continue
        rows = []
        for k, side in sizes:
            arr = acc[v.key].get(k, [])
            if not arr:
                continue
            counts = np.concatenate([a["counts"] for a in arr])
            areas = np.concatenate([a["areas"] for a in arr])
            active = counts > 0
            n_act = int(active.sum())
            Af = float(areas.mean()) if areas.size else None
            rows.append({
                "k": k, "side": side, "area_cells": side * side, "n_arenas": len(arr),
                "fields_per_unit": float(counts.mean()),
                "fields_per_unit_per_arena": [float(a["counts"].mean()) for a in arr],
                "active_frac": float(active.mean()),
                "fields_per_active": float(counts[active].mean()) if n_act else None,
                "single_field_frac_active": float(np.mean(counts[active] == 1)) if n_act else None,
                "mean_field_area": Af,
                "median_field_area": float(np.median(areas)) if areas.size else None,
                "arena_over_Af": (side * side / Af) if Af else None,
                "coverage_per_location": float(np.mean([a["cover"] for a in arr])),
                "unit_coverage_active": float(np.concatenate(
                    [a["unit_cov"][a["counts"] > 0] for a in arr]).mean()) if n_act else None,
                "count_fit": M.fit_count_models(counts),
            })
        out["variants"][v.key] = {"W": cal[v.key]["W"], "rows": rows,
                                  "scaling": area_scaling(rows)}
    sp = {"by_basis": {k: {kk: nanmean(vv) for kk, vv in d.items()} for k, d in sp_acc.items()},
          "note": "mean over active units of Treves-Rolls sparsity, keyed by arena area in A0"}
    return out, sp


def area_scaling(rows: list[dict]) -> dict:
    """Linearity of fields per unit against arena area, and Fenton-style interpolation."""
    if len(rows) < 3:
        return {"n_sizes": len(rows), "testable": False}
    a = np.array([r["area_cells"] for r in rows], dtype=np.float64)
    f = np.array([r["fields_per_unit"] for r in rows], dtype=np.float64)
    out = {"n_sizes": len(rows), "testable": True}
    if f.std() > 0:
        rr = float(np.corrcoef(a, f)[0, 1])
        slope, icpt = np.polyfit(a, f, 1)
        out.update(pearson_r=rr, r2=rr * rr, slope_per_cell=float(slope), intercept=float(icpt))
    else:
        out.update(pearson_r=None, r2=None, slope_per_cell=0.0, intercept=float(f.mean()))
    pos = f > 0
    if pos.sum() >= 3:
        out["loglog_exponent"] = float(np.polyfit(np.log(a[pos]), np.log(f[pos]), 1)[0])
    out["ratio_first_to_8x"] = None
    for i, r in enumerate(rows):
        j = [jj for jj, s in enumerate(rows) if s["k"] == 8 * r["k"]]
        if j and f[i] > 0:
            out["ratio_first_to_8x"] = {"from_k": r["k"], "ratio": float(f[j[0]] / f[i])}
            break
    # Fields per unit where the arena is 3.7 and 13.7 mean fields (Fenton 2008's
    # two rooms), interpolated in log(arena / A_f) over this series.
    x = np.array([r["arena_over_Af"] or np.nan for r in rows])
    ok = np.isfinite(x) & (x > 0)
    order = np.argsort(x[ok])
    lx, ff = np.log(x[ok][order]), f[ok][order]
    interp = {}
    for target in (3.7, 13.7):
        lt = math.log(target)
        interp[f"{target:g}"] = float(np.interp(lt, lx, ff)) if lx.size >= 2 and lx[0] <= lt <= lx[-1] else None
    out["fields_per_unit_at_arena_over_Af"] = interp
    return out


# ---------------------------------------------------------------------------
# Population similarity against distance
# ---------------------------------------------------------------------------

def kernel_1d(pop, P, bases, variants, cal, loops, t0) -> dict:
    curves: dict = {}
    for rot in sorted({v.rot for v in variants}):
        for zl in loops:
            zr = to_basis(zl, bases[rot])
            for m in M.SIMILARITIES:
                if m == "cosine" and rot:
                    continue        # unchanged by rotation; checked in the tests
                s, _ = M.lag_profile_1d(zr.T, m, circular=True)
                curves.setdefault(f"{basis_key(rot, 'raw')}|{m}", []).append(s)
            for conv in sorted({v.conv for v in variants if v.rot == rot}):
                R = rates_of(zr, conv)
                for m in M.SIMILARITIES:
                    s, _ = M.lag_profile_1d(R, m, circular=True)
                    curves.setdefault(f"{basis_key(rot, conv)}|{m}", []).append(s)
    out = {"far_W": P["kernel_far_W"], "constructions": {}}
    for key, ss in curves.items():
        s = np.nanmean(np.stack(ss), axis=0)
        d = np.arange(s.size, dtype=np.float64)
        parts = key.split("|")
        rot = int(parts[0][3:]) if parts[0].startswith("rot") else 0
        conv = parts[-2]
        summaries = {}
        convs = [conv] if conv != "raw" else sorted({v.conv for v in variants if v.rot == rot})
        for c in convs:
            for dn in def_names(P):
                vk = Variant(rot, c, dn).key
                if vk in cal and cal[vk]["W"]:
                    summaries[f"{c}|{dn}"] = M.kernel_summary(d, s, width=cal[vk]["W"],
                                                              far=tuple(P["kernel_far_W"]))
        out["constructions"][key] = {"s": s, "n_loops": len(ss), "summaries": summaries}
    log("kernel 1d done", t0)
    return out


def kernel_2d(pop, P, cal, W_ref, rng, t0, variants) -> dict:
    npos = pop.npos
    rmax = int(min(P["ring_max_W"] * W_ref, npos // 2 - 1))
    radii = np.unique(np.rint(np.geomspace(1, max(rmax, 2), P["ring_radii"])).astype(np.int64))
    dx, dy, dist = geo.rings(radii, P["ring_angles"])
    na = P["ring_anchors"]
    ax, ay = rng.integers(0, npos, na), rng.integers(0, npos, na)
    za = pop.code(ax, ay)
    zo = pop.code((ax[:, None] + dx[None, :]).reshape(-1), (ay[:, None] + dy[None, :]).reshape(-1))
    zrep = np.repeat(za, dx.size, axis=0)
    dd = np.tile(dist, na)
    edges = np.concatenate([[0.5], (radii[:-1] + radii[1:]) / 2.0, [radii[-1] + 0.5]])
    out = {"radii": radii, "n_anchors": na, "n_angles": P["ring_angles"], "constructions": {}}
    convs = sorted({v.conv for v in variants if v.rot == 0})
    sets = [("raw", zrep, zo)]
    for c in convs:
        sets.append((c, rates_of(zrep, c).T, rates_of(zo, c).T))
    for name, A, B in sets:
        for m in M.SIMILARITIES:
            sim = M.pair_similarity(A.T, B.T, m)
            curve = M.bin_curve(dd, sim, edges)
            d = np.concatenate([[0.0], np.asarray(curve["d"], dtype=np.float64)])
            s = np.concatenate([[1.0], np.asarray(curve["s"], dtype=np.float64)])
            summaries = {}
            for c in (convs if name == "raw" else [name]):
                for dn in def_names(P):
                    vk = Variant(0, c, dn).key
                    if vk in cal and cal[vk]["W"]:
                        summaries[f"{c}|{dn}"] = M.kernel_summary(
                            d, s, width=cal[vk]["W"], far=tuple(P["kernel_far_W"]))
            out["constructions"][f"{name}|{m}"] = {"d": d, "s": s, "n": [0] + curve["n"],
                                                   "summaries": summaries}
    log("kernel 2d done", t0)
    return out


# ---------------------------------------------------------------------------
# Remapping across scaffold locations
# ---------------------------------------------------------------------------

def remapping(pop, P, bases, variants, cal, W_ref, rng, t0) -> dict:
    npos = pop.npos
    A0 = math.pi / 4 * W_ref ** 2
    side = int(round(math.sqrt(P["remap_area_A0"] * A0)))
    capped = side > P["remap_side_cap"]
    side = int(max(3, min(side, P["remap_side_cap"])))
    pl = place(P["remap_n"], side, npos, int(2 * W_ref) + 1, int(rng.integers(1 << 30)))
    offs = pl["offsets"]
    codes = [pop.code(*geo.arena(x0, y0, side)) for x0, y0 in offs]
    pairs = [(i, j) for i in range(len(codes)) for j in range(i + 1, len(codes))]
    out = {"side": side, "side_capped": capped, "area_over_A0": side * side / A0,
           "offsets": offs, "sep_cells": pl["sep"], "disjoint": pl["disjoint"],
           "n_pairs": len(pairs), "bases": {}}

    def summarize(vals):
        x = np.asarray([v for v in vals if v is not None], dtype=np.float64)
        x = x[np.isfinite(x)]
        if not x.size:
            return None
        return {"median": float(np.median(x)), "mean": float(x.mean()),
                "min": float(x.min()), "max": float(x.max()), "n": int(x.size)}

    for rot in sorted({v.rot for v in variants}):
        zs = [to_basis(z, bases[rot]) for z in codes]
        if True:
            pv = {}
            for m in M.SIMILARITIES:
                if m == "cosine" and rot:
                    continue
                med = [float(np.nanmedian(M.pair_similarity(zs[i].T, zs[j].T, m))) for i, j in pairs]
                pv[m] = summarize(med)
            out["bases"][basis_key(rot, "raw")] = {"pv_matched": pv}
        for conv in sorted({v.conv for v in variants if v.rot == rot}):
            Rs = [rates_of(z, conv) for z in zs]
            res: dict = {"pv_matched": {}, "mean_rate_corr": {}, "ratemap": {}, "overlap": {}}
            for m in M.SIMILARITIES:
                res["pv_matched"][m] = summarize(
                    [float(np.nanmedian(M.pair_similarity(Rs[i], Rs[j], m))) for i, j in pairs])
            mr = [M.mean_rate_correlation(Rs[i], Rs[j]) for i, j in pairs]
            res["mean_rate_corr"] = {"pearson": summarize([x["pearson"] for x in mr]),
                                     "log_pearson": summarize([x["log_pearson"] for x in mr])}
            rm = [M.ratemap_correlations(Rs[i], Rs[j], seed=k) for k, (i, j) in enumerate(pairs)]
            res["ratemap"] = {"median_corr": summarize([x["median"] for x in rm]),
                              "frac_above_null_p95": summarize([x["frac_above_null_p95"] for x in rm])}
            for v in variants:
                if v.basis != (rot, conv) or cal[v.key]["W"] is None:
                    continue
                fd = fdef_from(cal[v.key]["fdef"])
                act = [M.detect_fields_2d(R.reshape(-1, side, side), fd).counts > 0 for R in Rs]
                ov = [M.active_set_overlap(act[i], act[j]) for i, j in pairs]
                res["overlap"][v.defn] = {
                    "active_frac": summarize([a.mean() for a in act]),
                    **{k: summarize([o[k] for o in ov]) for k in
                       ("both_frac", "both_expected", "both_ratio", "jaccard",
                        "jaccard_expected", "jaccard_ratio")}}
            out["bases"][basis_key(rot, conv)] = res
        log(f"remapping basis rot{rot} done", t0)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PHASE1 = ("kernels", "fields", "remapping")


def run_population(name: str, spec: str, P: dict, out_root: Path,
                   measures: tuple[str, ...] = PHASE1) -> None:
    t0 = time.time()
    pop = make_population(name, spec)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(P["seed"])
    variants = variants_for(pop, P)
    bases = {0: None}
    for s in sorted({v.rot for v in variants if v.rot}):
        bases[s] = M.random_orthogonal(pop.n_units, s)
    log(f"{name}: {pop.n_units} units, signed={pop.signed}, {len(variants)} variants", t0)

    cb = calibrate(pop, P, bases, variants, rng, t0)
    cal = cb["calib"]
    primary = Variant(0, "rectify" if pop.signed else "identity", "A")
    W_ref = cal[primary.key]["W"]
    meta = {"name": name, "spec": spec, "label": pop.label, "n_units": pop.n_units,
            "signed": pop.signed, "npos": pop.npos, "pop_meta": pop.meta, "params": P,
            "git": git_state(), "variants": [v.key for v in variants],
            "primary_variant": primary.key, "W_ref": W_ref, "measures": list(measures),
            "rate_scales": cb["scales"], "calibration": cal, "loop_rows": cb["loop_rows"]}
    if "kernels" in measures or not (out / "meta.json").exists():
        write(out, "meta", meta)
    if W_ref is None:
        log(f"{name}: no fields under the primary definition; only kernels are measured", t0)

    # Each block draws from its own generator, so running a subset of the
    # measures reproduces exactly what the full run would have measured.
    def sub(i):
        return np.random.default_rng([P["seed"], i])

    if "kernels" in measures:
        write(out, "kernel_1d", kernel_1d(pop, P, bases, variants, cal, cb["loops"], t0))
        write(out, "kernel_2d", kernel_2d(pop, P, cal, W_ref or 1.0, sub(1), t0, variants))
    # "fields" runs both; "fields1d" or "fields2d" reruns one and keeps the
    # other's half of sparsity.json.
    sp = {}
    if (out / "sparsity.json").exists():
        with open(out / "sparsity.json") as f:
            sp = json.load(f)
    if W_ref is not None and ({"fields", "fields1d"} & set(measures)):
        f1, sp["tracks"] = fields_1d(pop, P, bases, variants, cal, W_ref, sub(2), t0)
        write(out, "fields_1d", f1)
        write(out, "sparsity", sp)
    if W_ref is not None and ({"fields", "fields2d"} & set(measures)):
        f2, sp["arenas"] = fields_2d(pop, P, bases, variants, cal, W_ref, sub(3), t0)
        write(out, "fields_2d", f2)
        write(out, "sparsity", sp)
    if W_ref is not None and "remapping" in measures:
        write(out, "remapping", remapping(pop, P, bases, variants, cal, W_ref, sub(4), t0))
    if "phase2" in measures:
        from .phase2 import run_phase2
        write(out, "phase2", run_phase2(pop, P, cal, W_ref, sub(5), t0))
    if "kernels" in measures:
        meta["elapsed_s"] = time.time() - t0
        write(out, "meta", meta)
    log(f"{name}: done ({','.join(measures)})", t0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pops", default=",".join(POPULATIONS),
                   help="comma-separated names from POPULATIONS")
    p.add_argument("--pop", action="append", default=[],
                   help="extra population as name=spec (spec: checkpoint path or ideal:...)")
    p.add_argument("--tag", default="v1")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--quick", action="store_true", help="small settings for a smoke test")
    p.add_argument("--measures", default=",".join(PHASE1),
                   help="comma-separated subset of kernels,fields (or fields1d, fields2d),remapping,phase2")
    a = p.parse_args()
    P = dict(PARAMS, **(QUICK if a.quick else {}))
    root = a.out or (results_dir() / "coding_comparison" / a.tag)
    todo = [(n, POPULATIONS[n]) for n in filter(None, a.pops.split(","))]
    todo += [tuple(s.split("=", 1)) for s in a.pop]
    for name, spec in todo:
        run_population(name, spec, P, root, tuple(filter(None, a.measures.split(","))))


if __name__ == "__main__":
    main()
