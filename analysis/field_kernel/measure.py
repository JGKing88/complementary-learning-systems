"""Kernel and single-unit statistics of one ``ideal:`` code, with the prediction from its drive.

    python -m analysis.field_kernel.measure --spec "ideal:r=16,relu_mix=1,active=0.1" [--out f.json]

Everything is computed from ``IdealEncoder.closed_form`` on scaffold cells:

* 8 full scaffold loops (4 rows, 4 columns, 1716 cells): the Pearson and cosine similarity against
  separation, definition-A fields (run at or above 20% of the unit's peak on that loop, at least
  3 cells; ``analysis.coding_comparison.measures``), and the correlation of the units' drive
  before the threshold, ``rho(d)``.
* square arenas (default sides 20, 40 and 128 cells, 4 places each): 2D definition-A fields.

Prediction: the rate kernel is ``K(rho(d); t)`` (``analysis.ideal_family.theory``), with rho the
measured drive correlation, so the test is whether the drive's correlation alone fixes the kernel.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from analysis.coding_comparison import measures as CM
from analysis.hopfield_probe.ideal_encoder import IdealEncoder, parse_ideal_spec
from analysis.ideal_family import measure as FM
from analysis.ideal_family import theory as TH

NPOS = 1716
LAMBDA = 12.0


def build(spec: str) -> IdealEncoder:
    kw = parse_ideal_spec(spec)
    kw.pop("gain", None)
    return IdealEncoder(**kw)


def loops(seed: int = 0, n: int = 4):
    """``n`` rows and ``n`` columns of the scaffold, as (xs, ys) pairs."""
    rng = np.random.default_rng(seed + 99)
    c = rng.integers(0, NPOS, size=2 * n)
    a = np.arange(NPOS)
    return [(a, np.full(NPOS, v)) for v in c[:n]] + [(np.full(NPOS, v), a) for v in c[n:]]


def drive_of(enc: IdealEncoder, xs, ys) -> np.ndarray:
    """``(positions, units)`` drive before the threshold, in units of its SD (1/sqrt(D))."""
    return enc.drive(enc._waves_np(xs, ys), enc._grid_np(xs, ys)) * math.sqrt(enc.out_dim)


def crossing(s: np.ndarray, level: float) -> float | None:
    """First separation where ``s`` falls to ``level`` (linear interpolation), in cells."""
    below = np.flatnonzero(s < level)
    if below.size == 0 or below[0] == 0:
        return None
    i = below[0]
    return float(i - 1 + (s[i - 1] - level) / (s[i - 1] - s[i]))


def first_rise(s: np.ndarray, tol: float = 1e-3) -> int | None:
    """First separation after which the similarity rises by more than ``tol``."""
    up = np.flatnonzero(np.diff(s) > tol)
    return int(up[0]) if up.size else None


def kernel_stats(s: np.ndarray) -> dict:
    far = s[100:800]
    fw05 = crossing(s, 0.5)
    return {
        "fw05": None if fw05 is None else 2 * fw05,
        "fw02": None if crossing(s, 0.2) is None else 2 * crossing(s, 0.2),
        "at": {str(d): float(s[d]) for d in (2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64)},
        "first_rise": first_rise(s[:100]),
        "far_mean": float(far.mean()), "far_max": float(far.max()), "far_min": float(far.min()),
    }


def size_quantiles(x: np.ndarray) -> dict:
    if x.size == 0:
        return {}
    q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"mean": float(x.mean()), "q10": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "q90": float(q[4]),
            # share of all in-field cells that lie in fields no larger than twice the median
            "small_mass": float(x[x <= 2 * q[2]].sum() / x.sum())}


def measure(spec: str, *, seed: int = 0, sides=(20, 40, 128), n_places: int = 4) -> dict:
    enc = build(spec)
    t = enc.thresh * math.sqrt(enc.out_dim) if enc.relu_mix else None
    out = {"spec": spec, "t": t, "D": enc.out_dim}

    prof = {"pearson": [], "cos": [], "rho": []}
    stats, widths, counts = [], [], []
    for xs, ys in loops(seed):
        R = enc.closed_form(xs, ys).T                       # (units, positions)
        if not enc.relu_mix:
            R = np.maximum(R, 0.0)
        for m, key in (("pearson", "pearson"), ("cosine", "cos")):
            s, _ = CM.lag_profile_1d(R, m, circular=True)
            prof[key].append(s)
        if enc.relu_mix:
            Y = drive_of(enc, xs, ys).T
            Y = Y / np.sqrt((Y * Y).mean(axis=0, keepdims=True))     # per-place norm over units
            F = np.fft.rfft(Y, axis=1)
            prof["rho"].append(np.fft.irfft((F * F.conj()).sum(0), n=NPOS)[:NPOS // 2 + 1]
                               / (Y.shape[0] * NPOS) * Y.shape[0])
        st, Fl = FM.loop_stats(R)
        stats.append(st)
        widths.append(Fl.length.astype(np.float64))
        counts.append(Fl.counts)
    for k in prof:
        if prof[k]:
            prof[k] = np.mean(prof[k], axis=0)
    w = np.concatenate(widths)
    pooled = FM.pooled(stats)
    out["loop"] = {
        "W_A": float(w.mean()), "W_quant": size_quantiles(w),
        "fields_per_unit": float(np.mean(np.concatenate(counts))),
        "field_coverage": pooled["field_coverage_all"],
        "coverage": pooled["coverage_all"], "tr_pooled": pooled["tr"]["pooled"],
        "var_over_mean": pooled["var_over_mean"],
    }
    out["kernel"] = {k: kernel_stats(prof[k]) for k in ("pearson", "cos")}
    out["profile"] = {k: [float(v) for v in prof[k][:201]] for k in prof if len(prof[k])}
    if enc.relu_mix:
        rho = np.clip(prof["rho"] / prof["rho"][0], -1, 1)
        out["kernel"]["rho"] = kernel_stats(rho)
        pred = np.array([TH.kernel_pearson(float(x), t) for x in rho[:801]])
        out["kernel"]["pearson_pred"] = kernel_stats(pred)
        out["profile"]["pearson_pred"] = [float(v) for v in pred[:201]]
        # second spectral moment of the drive along a loop, as a length: 1/sqrt(lambda_2)
        out["drive_len"] = float(1.0 / math.sqrt(max(2.0 * (1.0 - rho[1]), 1e-12)))

    rng = np.random.default_rng(seed + 7)
    out["arena"] = {}
    for S in sides:
        parts, areas = [], []
        for _ in range(n_places):
            x0, y0 = rng.integers(0, NPOS - S, size=2)
            gx, gy = np.meshgrid(np.arange(S) + x0, np.arange(S) + y0, indexing="ij")
            R = enc.closed_form(gx.ravel(), gy.ravel())
            if not enc.relu_mix:
                R = np.maximum(R, 0.0)
            R3 = R.T.reshape(-1, S, S)
            F2 = CM.detect_fields_2d(R3, FM.DEF_A)
            c = F2.counts
            act = c > 0
            parts.append({"active_frac": float(act.mean()),
                          "fields_per_active": float(c[act].mean()) if act.any() else 0.0,
                          "single_field_share": float((c[act] == 1).mean()) if act.any() else 0.0,
                          "per_place_in_field": float(F2.coverage_frac.mean())})
            areas.append(F2.area)
        a = np.concatenate(areas)
        d = 2 * np.sqrt(a / math.pi)
        out["arena"][str(S)] = dict(FM.pooled(parts), diam=size_quantiles(d),
                                    diam_of_mean_area=float(2 * math.sqrt(a.mean() / math.pi))
                                    if a.size else None)
    return out


def line(o: dict) -> str:
    k, l = o["kernel"], o["loop"]
    kp = k["pearson"]
    pred = k.get("pearson_pred", {}).get("fw05")
    a40 = o["arena"].get("40", {})
    f = lambda v, p=1: "   -" if v is None else f"{v:.{p}f}"
    return (f"FW05 {f(kp['fw05'])} (pred {f(pred)})  K(16) {kp['at']['16']:.2f} K(32) {kp['at']['32']:.2f}"
            f"  rise {kp['first_rise']}  far max {kp['far_max']:.3f} |"
            f" W_A {l['W_A']:.1f} med {l['W_quant']['median']:.0f}  W_A/lam {l['W_A'] / LAMBDA:.2f}"
            f"  fields/loop {l['fields_per_unit']:.1f} |"
            f" 40-arena: active {f(a40.get('active_frac'), 2)} fields/active {f(a40.get('fields_per_active'), 2)}"
            f" diam med {f(a40.get('diam', {}).get('median'))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = []
    for spec in args.spec:
        o = measure(spec, seed=args.seed)
        res.append(o)
        print(f"{spec}\n    {line(o)}", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    main()
