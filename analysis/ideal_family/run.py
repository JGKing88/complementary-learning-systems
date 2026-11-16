"""The ideal-family simulation grid: one JSON per configuration.

    python -m analysis.ideal_family.run --list                     # print the grid
    python -m analysis.ideal_family.run --task I --n-tasks N        # run configs I, I+N, ...
    python -m analysis.ideal_family.run --only enc_r16_M512_s0      # one config

A configuration is either an encoder draw (``enc``: r, M, seed) or a direct
GRF reference (``grf``: r, seed; the M = infinity limit of the rotated
family, 1024 independent units). Each one measures every variant in
``codes.variant_names()`` on

* loops: 4 scaffold rows and 4 columns (1716 cells each), definition A per
  loop, fields joined across the seam;
* long: the 4 rows (and the 4 columns) as one 6864-cell environment with one
  peak per unit, so definition A's level is set over the whole track;
* short: the first 4.6 and 15.4 W_A cells of each loop, as straight tracks
  (the suite's Treves-Rolls track is the 4.6 W_A one);
* arenas (when the config asks for 2D): squares of side 4 r, 8 r and 16 r,
  two places each;
* kernels: cosine of the raw code, and cosine and Pearson of each variant's
  rates, along the loops;
* alias (encoder only): the raw kernel at every displacement on the scaffold
  torus (one inverse FFT), its largest value beyond 3 kernel widths, the
  rate kernels predicted from it, and a check of those at the worst
  displacement over 256 anchors.

Results go to ``<out>/configs/<name>.json``; a config whose file exists is
skipped, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from analysis.coding_comparison import measures as CM

from . import codes as C
from . import grf as G
from . import measure as IM
from . import theory as T

OUT = Path("/orcd/pool/003/jackking/cls_runs/results/ideal_family/v1")
HERE = Path(__file__).resolve().parent

R_MAIN = [4.0, 8.0, 16.0, 23.2, 32.0]
M_ALL = [32, 64, 128, 256, 512, 1024, 2048]
R_DORSAL = [2.0, 2.5, 3.0, 3.5, 5.0, 6.0]
SEEDS = [0, 1, 2]
ARENA_K = [4, 8, 16]          # arena sides in units of r
N_ARENA_LOCS = 2
SHORT_W = [4.6, 15.4]          # Rich's 3 and 10 m tracks in field widths
GRF_UNITS = 1024
ALIAS_ANCHORS = 256


def grid() -> list[dict]:
    out = []
    for r in R_MAIN:
        for M in M_ALL:
            for s in SEEDS:
                out.append({"kind": "enc", "r": r, "M": M, "seed": s,
                            "do2d": (M == 512) or (r == 16.0)})
    for r in R_DORSAL:
        for s in SEEDS:
            out.append({"kind": "enc", "r": r, "M": 512, "seed": s, "do2d": True})
    for r in sorted(R_MAIN + R_DORSAL):
        for s in SEEDS:
            out.append({"kind": "grf", "r": r, "seed": s, "do2d": True})
    for c in out:
        c["name"] = config_name(c)
    # Heaviest first, so array tasks finish together.
    return sorted(out, key=lambda c: -cost(c))


def config_name(c: dict) -> str:
    r = f"{c['r']:g}"
    return (f"enc_r{r}_M{c['M']}_s{c['seed']}" if c["kind"] == "enc"
            else f"grf_r{r}_s{c['seed']}")


def cost(c: dict) -> float:
    D = 2 * c.get("M", GRF_UNITS // 2)
    one = D * 8 * 1716 * 13
    two = (sum((k * c["r"]) ** 2 for k in ARENA_K) * N_ARENA_LOCS * D * 13) if c["do2d"] else 0
    return one + 3 * two


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def git_state() -> dict:
    gc = HERE.parent.parent / "GIT_COMMIT"
    if gc.exists():
        return json.loads(gc.read_text())
    def run(*a):
        return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()
    try:
        return {"sha": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--", "analysis/ideal_family"))}
    except OSError:
        return {"sha": None}


def log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def long_fields(R_rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Definition A over several loops treated as one environment (one peak per unit).

    Each loop's fields are found circularly at ``0.2 * peak`` with the peak taken
    over all loops. Returns per-unit counts, all field lengths, centres offset by
    loop index, and the units of each field.
    """
    peak = np.max([R.max(axis=1) for R in R_rows], axis=0)
    thr = 0.2 * peak
    scale = np.where(thr > 0, 1.0 / np.where(thr > 0, thr, 1.0), 0.0)
    fdef = CM.FieldDef("A_long", "absolute", thresh_hz=1.0, min_len=3)
    counts = np.zeros(R_rows[0].shape[0], np.int64)
    lens, cents, units = [], [], []
    for k, R in enumerate(R_rows):
        Rr, sh = IM.roll_to_min(R * scale[:, None])
        F = CM.detect_fields_1d(Rr, fdef)
        F = CM.Fields1D(F.n_units, F.length_bins, F.unit, F.start, F.stop, F.center)
        counts += F.counts
        lens.append(F.length)
        cents.append(F.center + k * R.shape[1])
        units.append(F.unit)
    return counts, np.concatenate(lens), np.concatenate(cents), np.concatenate(units)


def gap_cv_from(units, cents, total_len, min_fields=6) -> np.ndarray:
    F = CM.Fields1D(int(units.max()) + 1 if units.size else 0, int(total_len), units,
                    np.zeros_like(units), np.zeros_like(units), cents)
    return IM.gap_cvs(F, circular=True, min_fields=min_fields)


def kernel_curves(R_loops: list[np.ndarray], forms=("cosine", "pearson")) -> dict:
    """Mean circular lag profile over loops. ``R`` is ``(units, positions)``."""
    out = {}
    for f in forms:
        out[f] = np.nanmean(np.stack([CM.lag_profile_1d(R, f, circular=True)[0] for R in R_loops]), 0)
    return out


def kernel_summ(s: np.ndarray, W: float | None) -> dict:
    d = np.arange(s.size, dtype=np.float64)
    hw05 = CM.first_crossing(d, s, 0.5)
    hw02 = CM.first_crossing(d, s, 0.2)
    zc = CM.first_crossing(d, s, 0.0)
    out = {"fw05": None if hw05 is None else 2 * hw05,
           "fw02": None if hw02 is None else 2 * hw02, "zero_crossing": zc}
    if W:
        sel = (d >= 5 * W) & (d <= 15 * W)
        out["floor"] = float(np.nanmean(s[sel])) if sel.any() else None
        tail = d >= 5 * W
        out["tail_max"] = float(np.nanmax(s[tail])) if tail.any() else None
    return out


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------

class Source:
    """Rates of every variant at any set of places, for an encoder draw or GRF units."""

    def __init__(self, c: dict):
        self.c = c
        self.kind = c["kind"]
        self.r = float(c["r"])
        if self.kind == "enc":
            self.fam = C.Family(self.r, int(c["M"]), int(c["seed"]))
            self.sd = self.fam.sd
            self.variants = C.variant_names()
        else:
            self.fam = None
            self.sd = 1.0
            self.variants = [v for v in C.variant_names() if v != "native"]
            self.rng = np.random.default_rng([int(c["seed"]), 1000 + int(round(10 * self.r))])

    def loop_codes(self, n_rows=4, n_cols=4) -> list[tuple[np.ndarray | None, np.ndarray]]:
        """``[(z, y)]`` per loop, ``(positions, units)``; rows first, then columns."""
        out = []
        if self.kind == "enc":
            rng = np.random.default_rng([int(self.c["seed"]), 7])
            rows = rng.integers(0, C.NPOS, n_rows)
            cols = rng.integers(0, C.NPOS, n_cols)
            t = np.arange(C.NPOS)
            for y0 in rows:
                z = self.fam.native(t, np.full(C.NPOS, y0))
                out.append((z, self.fam.rotated(z)))
            for x0 in cols:
                z = self.fam.native(np.full(C.NPOS, x0), t)
                out.append((z, self.fam.rotated(z)))
        else:
            for _ in range(n_rows + n_cols):
                y = G.grf_1d(GRF_UNITS, C.NPOS, self.r, self.rng).T     # (positions, units)
                out.append((None, y))
        return out

    def arena_code(self, side: int, loc_rng) -> tuple[np.ndarray | None, np.ndarray, dict]:
        if self.kind == "enc":
            x0, y0 = (int(v) for v in loc_rng.integers(0, C.NPOS, 2))
            yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
            fam = C.Family(self.r, int(self.c["M"]), int(self.c["seed"]))
            fam.dtype = np.float32
            z = fam.native(xx.reshape(-1) + x0, yy.reshape(-1) + y0)
            return z, fam.rotated(z), {"x0": x0, "y0": y0}
        B = G.arena_box(side, self.r)
        ys = []
        for b0 in range(0, GRF_UNITS, 64):
            f = G.grf_2d(min(64, GRF_UNITS - b0), B, self.r, self.rng)[:, :side, :side]
            ys.append(f.reshape(f.shape[0], -1).astype(np.float32))
        return None, np.concatenate(ys, 0).T.copy(), {"box": B}


def per_unit_level(y_loops_units: list[np.ndarray], t: float, sd: float) -> dict:
    """Mean per-unit maximum (SDs) and definition A's level ``t + 0.2 (max - t)`` over active units."""
    mx = np.concatenate([y.max(axis=0) for y in y_loops_units]) / sd
    act = mx > t
    uA = t + 0.2 * (mx[act] - t)
    return {"max_mean": float(mx.mean()), "max_sd": float(mx.std()),
            "uA_mean": float(uA.mean()) if act.any() else None,
            "uA_sd": float(uA.std()) if act.any() else None}


def run_config(c: dict, out_dir: Path, t0: float) -> dict:
    src = Source(c)
    res = {"config": c, "variants": {}, "git": git_state(), "host": platform.node()}
    loops = src.loop_codes()
    log(f"{c['name']}: loops encoded", t0)
    L = C.NPOS

    if src.kind == "enc":
        fam = src.fam
        lags = np.arange(L // 2 + 1)
        kx = fam.realized_kernel_1d(lags, 0)
        ky = fam.realized_kernel_1d(lags, 1)
        res["realized"] = {
            "fw05_x": 2 * CM.first_crossing(lags.astype(float), kx, 0.5),
            "fw05_y": 2 * CM.first_crossing(lags.astype(float), ky, 0.5),
        }
        res["realized"]["r_eff"] = 0.5 * (res["realized"]["fw05_x"] + res["realized"]["fw05_y"]) \
            / T.gaussian_width(0.5, 1.0)
        # Per-unit spectral moments along x, straight from the rotation.
        Q = C.apply_rotation(np.eye(fam.D), fam.signs).T            # rows are units
        w = Q[:, :fam.M] ** 2 + Q[:, fam.M:] ** 2                    # (units, M)
        om = 2 * math.pi * fam.freqs / C.NPOS
        lam_x = (w * om[None, :, 0] ** 2).sum(1) * fam.r ** 2
        lam_y = (w * om[None, :, 1] ** 2).sum(1) * fam.r ** 2
        res["spectral"] = {"lam_x_mean": float(lam_x.mean()), "lam_x_cv2": float(lam_x.var() / lam_x.mean() ** 2),
                           "pred_cv2": T.lambda2_cv2(fam.M), "sum_w2_times_M1_over_2":
                           float((w ** 2).sum(1).mean() * (fam.M + 1) / 2),
                           "row_norm_dev": float(np.abs(w.sum(1) - 1).max())}
        unit_lam = {"x": lam_x, "y": lam_y}
        # raw cosine kernel along loops
        raw = kernel_curves([y.T for _, y in loops], forms=("cosine",))["cosine"]
        res["raw_kernel"] = {"s": raw[:min(raw.size, int(15 * src.r) + 1)], **kernel_summ(raw, None),
                             "realized_x": kx[:min(raw.size, int(15 * src.r) + 1)]}
    else:
        unit_lam = None
        raw = kernel_curves([y.T for _, y in loops], forms=("cosine",))["cosine"]
        res["raw_kernel"] = {"s": raw[:min(raw.size, int(15 * src.r) + 1)], **kernel_summ(raw, None)}

    for v in src.variants:
        tv = C.threshold_sd(v)
        R_loops = [C.rates(v, z, y, src.sd).T.copy() for z, y in loops]      # (units, positions)
        per_loop, sizes, loopsF = [], [], []
        for R in R_loops:
            st, F = IM.loop_stats(R)
            per_loop.append(st)
            sizes.append(F.length)
        sizes = np.concatenate(sizes).astype(np.float64)
        WA = float(sizes.mean()) if sizes.size else None
        vr = {"W_A": WA, "W_A_median": float(np.median(sizes)) if sizes.size else None,
              "n_fields": int(sizes.size), "loops": IM.pooled(per_loop),
              "loops_x": IM.pooled(per_loop[:4]), "loops_y": IM.pooled(per_loop[4:])}
        if tv is not None:
            vr["level"] = per_unit_level([y for _, y in loops], tv, src.sd)
        # long environment: rows, then columns
        longs = []
        for part in (slice(0, 4), slice(4, 8)):
            Rs = R_loops[part]
            counts, lens, cents, units = long_fields(Rs)
            Rcat = np.concatenate(Rs, axis=1)
            st = IM.unit_stats(Rcat, None, counts, lens.astype(np.float64), Rcat.shape[1],
                               gap_cv_from(units, cents, Rcat.shape[1]))
            st["W_long"] = float(lens.mean()) if lens.size else None
            if unit_lam is not None:
                lam = unit_lam["x" if part.start == 0 else "y"]
                ok = counts > 0
                st["count_vs_sqrt_lam_corr"] = (float(np.corrcoef(counts[ok], np.sqrt(lam[ok]))[0, 1])
                                                if ok.sum() > 2 and counts[ok].std() > 0 else None)
            longs.append(st)
        vr["long"] = IM.pooled(longs)
        vr["long_x"], vr["long_y"] = longs
        # short straight tracks from each loop's start
        if WA:
            vr["short"] = {}
            for LW in SHORT_W:
                Ls = int(round(LW * WA))
                if 3 <= Ls <= L:
                    vr["short"][f"{LW:g}"] = dict(IM.pooled([IM.segment_stats(R[:, :Ls]) for R in R_loops]),
                                                  length_cells=Ls)
        # kernels of the rates
        kc = kernel_curves(R_loops)
        vr["kernel"] = {f: {"s": s[:min(s.size, int(15 * src.r) + 1)], **kernel_summ(s, WA)}
                        for f, s in kc.items()}
        res["variants"][v] = vr
        del R_loops
        log(f"{c['name']}: {v} 1D done (W_A={WA})", t0)

    if src.kind == "enc":
        res["alias"] = alias_block(src, res, t0)
    if c["do2d"]:
        res["arenas"] = arena_block(src, t0)
    return res


def alias_block(src: Source, res: dict, t0: float) -> dict:
    fam = src.fam
    N = C.NPOS
    K = fam.raw_kernel_torus()
    d = np.minimum(np.arange(N), N - np.arange(N)).astype(np.float64)
    dist = np.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
    out = {}
    fw_raw = res["raw_kernel"]["fw05"] or T.gaussian_width(0.5, src.r)
    Rex = 3 * fw_raw
    far = dist > Rex
    idx = np.argmax(np.where(far, K, -np.inf))
    ix, iy = np.unravel_index(idx, K.shape)
    pred = T.alias_prediction(fam.M, src.r, excl_radius=Rex)
    out["raw"] = {"excl_radius": Rex, "max": float(K[ix, iy]), "at": [int(ix), int(iy)],
                  "far_sd": float(K[far].std()), "far_mean": float(K[far].mean()),
                  "pred_sd": T.far_kernel_sd(fam.M), "pred_max": pred["alias"],
                  "pred_u_star": pred["u_star"], "area_over_r2": pred["area_over_r2"],
                  "max_over_sd": float(K[ix, iy] / K[far].std())}
    # Rate kernels: predicted from the raw maximum beyond each variant's own 3 widths.
    rng = np.random.default_rng([int(src.c["seed"]), 11])
    ax, ay = rng.integers(0, N, ALIAS_ANCHORS), rng.integers(0, N, ALIAS_ANCHORS)
    z0 = fam.native(ax, ay)
    y0 = fam.rotated(z0)
    rand_d = [(int(a), int(b)) for a, b in rng.integers(N // 8, N - N // 8, (4, 2))]
    Kn = None
    for v in src.variants:
        vr = res["variants"][v]
        fw = vr["kernel"]["cosine"]["fw05"]
        if fw is None:
            continue
        Rv = 3 * fw
        farv = dist > Rv
        tv = C.threshold_sd(v)
        if v == "native":
            if Kn is None:
                Kn = fam.native_rate_kernel_torus()
            kk = np.argmax(np.where(farv, Kn, -np.inf))
            jx, jy = np.unravel_index(kk, K.shape)
            entry = {"excl_radius": Rv, "at": [int(jx), int(jy)], "raw_at": float(K[jx, jy]),
                     "pred_cos": float(Kn[jx, jy]), "pred_floor_cos": 4 / math.pi ** 2}
        else:
            kk = np.argmax(np.where(farv, K, -np.inf))
            jx, jy = np.unravel_index(kk, K.shape)
            kmax = float(K[jx, jy])
            entry = {"excl_radius": Rv, "at": [int(jx), int(jy)], "raw_at": kmax,
                     "pred_cos": float(T.kernel_cos(np.array([kmax]), tv)[0]),
                     "pred_pearson": float(T.kernel_pearson(np.array([kmax]), tv)[0]),
                     "pred_floor_cos": float(T.treves_rolls(tv))}
        # anchors at the worst displacement and at random far displacements
        def measured(dx, dy):
            z1 = fam.native(ax + dx, ay + dy)
            y1 = fam.rotated(z1)
            A = C.rates(v, z0, y0, fam.sd).T
            B = C.rates(v, z1, y1, fam.sd).T
            return {f: float(np.nanmean(CM.pair_similarity(A, B, f))) for f in ("cosine", "pearson")}
        entry["measured"] = measured(int(jx), int(jy))
        entry["measured_random_far"] = [measured(a, b) for a, b in rand_d]
        out[v] = entry
    log(f"{src.c['name']}: alias done (raw max {out['raw']['max']:.4f}, pred {out['raw']['pred_max']:.4f})", t0)
    return out


def arena_block(src: Source, t0: float) -> dict:
    loc_rng = np.random.default_rng([int(src.c["seed"]), 9])
    out = {}
    for k in ARENA_K:
        side = int(round(k * src.r))
        if side < 8:
            continue
        acc = {v: [] for v in src.variants}
        levels = {v: [] for v in src.variants}
        places = []
        for _ in range(N_ARENA_LOCS):
            z, y, where = src.arena_code(side, loc_rng)
            places.append(where)
            for v in src.variants:
                R = C.rates(v, z, y, src.sd).T.reshape(-1, side, side)
                acc[v].append(IM.arena_stats(R))
                tv = C.threshold_sd(v)
                if tv is not None:
                    levels[v].append(per_unit_level([y], tv, src.sd))
                del R
            del z, y
        out[f"{k}r"] = {"side": side, "places": places,
                        "variants": {v: dict(IM.pooled(acc[v]),
                                             **({"level": IM.pooled(levels[v])} if levels[v] else {}))
                                     for v in src.variants}}
        log(f"{src.c['name']}: arenas {k}r (side {side}) done", t0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    cfgs = grid()
    if a.list:
        for i, c in enumerate(cfgs):
            print(i, c["name"], f"do2d={c['do2d']}", f"cost={cost(c):.3g}")
        print(len(cfgs), "configs")
        return
    if a.only:
        want = set(a.only.split(","))
        todo = [c for c in cfgs if c["name"] in want]
        if len(todo) != len(want):
            raise SystemExit(f"unknown configs: {want - {c['name'] for c in todo}}")
    else:
        todo = cfgs[(a.task or 0)::a.n_tasks]
    out = a.out / "configs"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for c in todo:
        f = out / f"{c['name']}.json"
        if f.exists():
            log(f"skip {c['name']} (exists)", t0)
            continue
        res = run_config(c, out, t0)
        res["elapsed_s"] = time.time() - t0
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(clean(res)))
        os.replace(tmp, f)
        log(f"DONE {c['name']}", t0)


if __name__ == "__main__":
    main()
