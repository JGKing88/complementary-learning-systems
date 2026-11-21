"""Tables of predicted against measured values from the ideal-family grid.

    python -m analysis.ideal_family.analyse [--root DIR]

Reads ``<root>/configs/*.json`` (written by ``run.py``) and writes
``<root>/tables.md`` and ``<root>/summary.json``. Every predicted number comes
from ``theory.py``; every measured number is a mean over seeds (and SD where
shown) of what ``run.py`` recorded.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import theory as T
from .codes import VARIANT_PS
from .run import OUT

LAMBDA = 12.0                     # mean grid period, cells (modules 11, 12, 13)
L_LOOP = 1716
L_LONG = 4 * 1716
SPARSE = [f"sparse_raw_p{p:g}" for p in VARIANT_PS if p != 0.5]
SPARSE_N = [f"sparse_p{p:g}" for p in VARIANT_PS]
THRESH = ["rotated"] + SPARSE


def load(root: Path) -> list[dict]:
    rows = []
    for f in sorted((root / "configs").glob("*.json")):
        rows.append(json.loads(f.read_text()))
    return rows


def pick(rows, kind, r=None, M=None):
    out = [x for x in rows if x["config"]["kind"] == kind]
    if r is not None:
        out = [x for x in out if abs(x["config"]["r"] - r) < 1e-9]
    if M is not None:
        out = [x for x in out if x["config"].get("M") == M]
    return out


def get(d, path):
    """``d[a][b]...`` for ``path = "a.b..."``; keys may contain dots (``sparse_raw_p0.1``, ``4.6``),
    so each step takes the longest run of tokens that is a key."""
    toks = path.split(".")
    i = 0
    while i < len(toks):
        if not isinstance(d, dict):
            return None
        for j in range(len(toks), i, -1):
            k = ".".join(toks[i:j])
            if k in d:
                d, i = d[k], j
                break
        else:
            return None
    return d


def ms(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if v.size == 0:
        return None, None
    return float(v.mean()), float(v.std()) if v.size > 1 else 0.0


def p_of(v):
    if v == "native":
        return None
    if v == "rotated":
        return 0.5
    return float(v.rsplit("_p", 1)[1])


def t_of(v):
    p = p_of(v)
    return None if p is None else T.t_of_p(p)


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, tuple):
        m, s = x
        if m is None:
            return "-"
        return f"{m:.{nd}f}" if not s else f"{m:.{nd}f} ± {s:.{nd}f}"
    return f"{x:.{nd}f}"


def table(headers, body) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(out)


def r_eff(x):
    """The draw's realized kernel SD (cells), from its full width at 0.5; nominal r for GRFs."""
    return get(x, "realized.r_eff") or x["config"]["r"]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def raw_kernel(rows) -> tuple[str, dict]:
    body, data = [], {}
    for r in sorted({x["config"]["r"] for x in pick(rows, "enc")}):
        for M in sorted({x["config"]["M"] for x in pick(rows, "enc", r)}):
            xs = pick(rows, "enc", r, M)
            fw = ms([get(x, "raw_kernel.fw05") / r for x in xs])
            reff = ms([r_eff(x) / r for x in xs])
            ratio = ms([(get(x, "raw_kernel.fw02") or np.nan) / get(x, "raw_kernel.fw05") for x in xs])
            zc = ms([(get(x, "raw_kernel.zero_crossing") or np.nan) / r for x in xs])
            sd = ms([get(x, "alias.raw.far_sd") * math.sqrt(2 * M) for x in xs])
            amax = ms([get(x, "alias.raw.max") for x in xs])
            pred = xs[0]["alias"]["raw"]["pred_max"]
            data[f"r{r:g}_M{M}"] = {"fw05_over_r": fw, "r_eff_over_r": reff, "fw02_over_fw05": ratio,
                                    "zero_crossing_over_r": zc, "far_sd_times_sqrt2M": sd,
                                    "alias_max": amax, "alias_pred": pred}
            body.append([f"{r:g}", M, fmt(fw), fmt(reff), fmt(ratio), fmt(zc, 2), fmt(sd),
                         fmt(amax), fmt(pred)])
    head = ["r", "M", "FW05 / r (2.355)", "r_eff / r", "FW02 / FW05 (1.524)", "zero crossing / r",
            "far SD x sqrt(2M) (1)", "alias max", "alias predicted"]
    return table(head, body), data


def single_unit(rows, kind="grf", M=None, env="loops") -> tuple[str, dict]:
    """Scale-free single-unit stats against theory, every r, for the GRF or one M."""
    body, data = [], {}
    rs = sorted({x["config"]["r"] for x in pick(rows, kind, M=M)})
    for v in THRESH:
        t = t_of(v)
        for r in rs:
            xs = pick(rows, kind, r, M)
            if not xs:
                continue
            th = T.def_A_1d(t, L_LOOP, r)
            cov = ms([get(x, f"variants.{v}.loops.coverage_all") for x in xs])
            tr = ms([get(x, f"variants.{v}.loops.tr.pooled") for x in xs])
            wa = ms([get(x, f"variants.{v}.W_A") / r_eff(x) for x in xs])
            fpw = ms([get(x, f"variants.{v}.loops.field_coverage_all") for x in xs])
            uA = ms([get(x, f"variants.{v}.level.uA_mean") for x in xs])
            data[f"{v}|r{r:g}"] = {"coverage": cov, "tr_pooled": tr, "W_A_over_r": wa,
                                   "fields_per_field_width": fpw, "u_A": uA,
                                   "pred": {"coverage": float(T.Phibar(t)), "tr": float(T.treves_rolls(t)),
                                            "W_A_over_r": th["W_A"] / r,
                                            "fields_per_field_width": th["field_coverage"],
                                            "u_A": th["u_A_mean"]}}
            body.append([v, f"{r:g}", f"{fmt(cov)} ({T.Phibar(t):.3f})", f"{fmt(tr, 4)} ({T.treves_rolls(t):.4f})",
                         f"{fmt(wa, 2)} ({th['W_A'] / r:.2f})", f"{fmt(fpw)} ({th['field_coverage']:.3f})",
                         f"{fmt(uA, 2)} ({th['u_A_mean']:.2f})"])
    head = ["variant", "r", "coverage (p)", "Treves-Rolls, pooled (a(t))", "W_A / r (Rice)",
            "fields per field width (Rice)", "u_A (Rice)"]
    return table(head, body), data


def kernels(rows) -> tuple[str, dict]:
    body, data = [], {}
    for v in THRESH + ["native"]:
        t = t_of(v)
        for kind, M in (("grf", None), ("enc", 512)):
            if kind == "grf" and v == "native":
                continue
            for form, key in (("cos", "cosine"), ("pearson", "pearson")):
                xs = [x for x in pick(rows, kind, M=M) if x["config"]["r"] >= 8]
                if not xs:
                    continue
                fw = ms([get(x, f"variants.{v}.kernel.{key}.fw05") / r_eff(x) for x in xs])
                fw2 = ms([(get(x, f"variants.{v}.kernel.{key}.fw02") or np.nan) / r_eff(x) for x in xs])
                fl = ms([get(x, f"variants.{v}.kernel.{key}.floor") for x in xs])
                if v == "native":
                    from scipy import optimize
                    pr = 2 * optimize.brentq(lambda d: T.native_kernel(np.array([d]), 1.0, form)[0] - 0.5,
                                             1e-3, 5)
                    pr2 = None if form == "cos" else 2 * optimize.brentq(
                        lambda d: T.native_kernel(np.array([d]), 1.0, form)[0] - 0.2, 1e-3, 5)
                    pfl = 4 / math.pi ** 2 if form == "cos" else 0.0
                else:
                    pr = T.kernel_width(0.5, 1.0, t, form)
                    try:
                        pr2 = T.kernel_width(0.2, 1.0, t, form)
                    except ValueError:
                        pr2 = None
                    pfl = float(T.treves_rolls(t)) if form == "cos" else 0.0
                lab = "GRF" if kind == "grf" else f"M={M}"
                data[f"{v}|{lab}|{form}"] = {"fw05_over_r": fw, "fw02_over_r": fw2, "floor": fl,
                                             "pred_fw05": pr, "pred_fw02": pr2, "pred_floor": pfl}
                body.append([v, lab, form, f"{fmt(fw)} ({pr:.3f})",
                             f"{fmt(fw2)} ({'-' if pr2 is None else f'{pr2:.3f}'})",
                             f"{fmt(fl, 4)} ({pfl:.4f})"])
    head = ["variant", "code", "form", "FW05 / r (theory)", "FW02 / r (theory)", "floor (theory)"]
    return table(head, body), data


def width_matching() -> tuple[str, dict]:
    target = T.gaussian_width(0.5, 16.0)
    body, data = [], {}
    for p in VARIANT_PS:
        t = T.t_of_p(p)
        rc = T.r_for_width(target, 0.5, t, "cos")
        rp = T.r_for_width(target, 0.5, t, "pearson")
        data[f"{p:g}"] = {"r_cos": rc, "r_pearson": rp, "rho_star_cos": T.rho_at_level(0.5, t, "cos"),
                          "rho_star_pearson": T.rho_at_level(0.5, t, "pearson")}
        body.append([f"{p:g}", f"{t:.3f}", f"{data[f'{p:g}']['rho_star_cos']:.3f}", f"{rc:.2f}",
                     f"{data[f'{p:g}']['rho_star_pearson']:.3f}", f"{rp:.2f}"])
    head = ["p", "t", "rho* (cosine)", "r for 37.7-cell FW05, cosine", "rho* (Pearson)",
            "r for 37.7-cell FW05, Pearson"]
    return table(head, body), data


def counts(rows, kind="grf", M=None, r_list=None) -> tuple[str, dict]:
    body, data = [], {}
    for v in THRESH:
        t = t_of(v)
        for r in (r_list or sorted({x["config"]["r"] for x in pick(rows, kind, M=M)})):
            xs = pick(rows, kind, r, M)
            if not xs:
                continue
            for env, L, key in (("loop", L_LOOP, "loops"), ("long", L_LONG, "long")):
                vm = ms([get(x, f"variants.{v}.{key}.var_over_mean") for x in xs])
                pr = T.fano_def_A(t, L, r)
                gap = ms([get(x, f"variants.{v}.{key}.gap_cv_median") for x in xs])
                fpu = ms([get(x, f"variants.{v}.{key}.fields_per_active") for x in xs])
                use = pr["total"] if t < 1.0 else pr["total_peaks"]
                nf = T.def_A_1d(t, L, r)["fields_per_active_unit_peak"]
                data[f"{v}|r{r:g}|{env}"] = {"var_over_mean": vm, "pred": use, "pred_low": pr["total"],
                                             "pred_high": pr["total_peaks"], "F": pr["F"],
                                             "gap_cv": gap, "renewal_gap_cv": math.sqrt(pr["F"]),
                                             "fields_per_active": fpu, "pred_fields": nf}
                body.append([v, f"{r:g}", env, fmt(fpu, 1), f"{nf:.1f}",
                             fmt(vm), f"{use:.3f}", f"{pr['F']:.3f}", fmt(gap), f"{math.sqrt(pr['F']):.3f}"])
    head = ["variant", "r", "env", "fields per active unit", "(Rice + top field)", "var / mean", "(predicted)",
            "F(u_A)", "gap CV (median)", "sqrt F (renewal)"]
    return table(head, body), data


def finite_M(rows, r=16.0) -> tuple[str, dict]:
    body, data = [], {}
    for v in ["rotated", "sparse_raw_p0.1", "sparse_p0.1", "sparse_raw_p0.02"]:
        t = t_of(v)
        grf = pick(rows, "grf", r)
        for M in [None] + sorted({x["config"]["M"] for x in pick(rows, "enc", r)}):
            xs = grf if M is None else pick(rows, "enc", r, M)
            if not xs:
                continue
            tr = ms([get(x, f"variants.{v}.loops.tr.pooled") for x in xs])
            cov = ms([get(x, f"variants.{v}.loops.coverage_all") for x in xs])
            wa = ms([get(x, f"variants.{v}.W_A") / r_eff(x) for x in xs])
            vm_loop = ms([get(x, f"variants.{v}.loops.var_over_mean") for x in xs])
            vm_long = ms([get(x, f"variants.{v}.long_x.var_over_mean") for x in xs])
            gap = ms([get(x, f"variants.{v}.loops.gap_cv_median") for x in xs])
            corr = ms([get(x, f"variants.{v}.long_x.count_vs_sqrt_lam_corr") for x in xs])
            pr = T.fano_def_A(t, L_LONG, r)
            base = pr["total"] if t < 1.0 else pr["total_peaks"]
            if M is None:
                pred_long = base
            else:
                het = np.mean([get(x, f"variants.{v}.long_x.fields_per_active") * get(x, "spectral.lam_x_cv2") / 4
                               for x in xs])
                pred_long = base + float(het)
            lab = "GRF" if M is None else str(M)
            data[f"{v}|{lab}"] = {"tr": tr, "coverage": cov, "W_A_over_r_eff": wa, "vm_loop": vm_loop,
                                  "vm_long": vm_long, "pred_vm_long": pred_long, "gap_cv": gap,
                                  "count_lam_corr": corr}
            body.append([v, lab, fmt(cov), fmt(tr, 4), fmt(wa, 2), fmt(vm_loop), fmt(vm_long),
                         f"{pred_long:.3f}", fmt(corr, 2), fmt(gap)])
    head = ["variant", "M", "coverage", "TR pooled", "W_A / r_eff", "var/mean, loop",
            "var/mean, 6864 cells", "(predicted)", "corr(count, sqrt lambda_2)", "gap CV"]
    return table(head, body), data


def spectral(rows) -> tuple[str, dict]:
    body, data = [], {}
    for M in sorted({x["config"]["M"] for x in pick(rows, "enc")}):
        xs = pick(rows, "enc", M=M)
        cv = ms([get(x, "spectral.lam_x_cv2") * (M + 1) for x in xs])
        sw = ms([get(x, "spectral.sum_w2_times_M1_over_2") for x in xs])
        data[str(M)] = {"lam_cv2_times_M1": cv, "sum_w2_times_M1_over_2": sw}
        body.append([M, fmt(sw), fmt(cv, 2)])
    head = ["M", "E sum w^2 x (M+1)/2 (Dirichlet: 1)", "CV^2(lambda_2) x (M+1) (a_cv2, 2 in expectation)"]
    return table(head, body), data


def alias_rates(rows) -> tuple[str, dict]:
    body, data = [], {}
    for v in ["native", "rotated", "sparse_p0.5", "sparse_p0.1", "sparse_raw_p0.1", "sparse_p0.02"]:
        err_c, err_p, exc = [], [], []
        for x in pick(rows, "enc"):
            a = get(x, f"alias.{v}")
            if not a:
                continue
            err_c.append(a["measured"]["cosine"] - a["pred_cos"])
            if "pred_pearson" in a:
                err_p.append(a["measured"]["pearson"] - a["pred_pearson"])
            fl = np.mean([m["cosine"] for m in a["measured_random_far"]])
            exc.append((a["measured"]["cosine"] - fl) / max(a["raw_at"], 1e-9))
        if not err_c:
            continue
        data[v] = {"cos_err_mean": float(np.mean(err_c)), "cos_err_maxabs": float(np.max(np.abs(err_c))),
                   "pearson_err_mean": float(np.mean(err_p)) if err_p else None,
                   "pearson_err_maxabs": float(np.max(np.abs(err_p))) if err_p else None,
                   "excess_over_raw": float(np.median(exc)),
                   "pred_slope_cos": None if v == "native" else T.far_slope(t_of(v), "cos")}
        body.append([v, len(err_c), f"{np.mean(err_c):+.4f}", f"{np.max(np.abs(err_c)):.4f}",
                     "-" if not err_p else f"{np.mean(err_p):+.4f}",
                     "-" if not err_p else f"{np.max(np.abs(err_p)):.4f}",
                     f"{np.median(exc):.3f}",
                     "-" if v == "native" else f"{T.far_slope(t_of(v), 'cos'):.3f}"])
    head = ["variant", "configs", "cosine: measured - predicted, mean", "max abs",
            "Pearson: mean", "max abs", "(alias - floor) / raw alias, median", "dK/drho at 0 (theory)"]
    return table(head, body), data


def native(rows) -> tuple[str, dict]:
    """Native code on loops, against the closed forms and against each draw's own frequencies."""
    from analysis.hopfield_probe.ideal_encoder import draw_frequencies
    body, data = [], {}
    for r in sorted({x["config"]["r"] for x in pick(rows, "enc")}):
        for M in sorted({x["config"]["M"] for x in pick(rows, "enc", r)}):
            xs = pick(rows, "enc", r, M)
            wa = ms([get(x, "variants.native.W_A") / r for x in xs])
            ratio_w, ratio_tr = [], []
            for x in xs:
                n = draw_frequencies(r, M, x["config"]["seed"], L_LOOP)
                for ax_, key in ((0, "loops_x"), (1, "loops_y")):
                    pr = T.native_loop_from_draw(n[:, ax_])
                    ratio_w.append(get(x, f"variants.native.{key}.mean_field_size") / pr["W_A"])
                    ratio_tr.append(get(x, f"variants.native.{key}.tr.active") / pr["tr_active"])
            rw, rt = ms(ratio_w), ms(ratio_tr)
            tr = ms([get(x, "variants.native.loops.tr.active") for x in xs])
            cov = ms([get(x, "variants.native.loops.coverage_all") for x in xs])
            fcov = ms([get(x, "variants.native.loops.field_coverage_all") for x in xs])
            vm = ms([get(x, "variants.native.loops.var_over_mean") for x in xs])
            fcv = ms([get(x, "variants.native.loops.field_size_cv") for x in xs])
            data[f"r{r:g}_M{M}"] = {"W_A_over_r": wa, "W_over_draw_pred": rw, "tr_over_draw_pred": rt,
                                    "tr": tr, "coverage": cov, "field_coverage": fcov, "var_over_mean": vm,
                                    "field_size_cv": fcv, "pred_vm": T.native_fano_1d(L_LOOP, r)}
            body.append([f"{r:g}", M, fmt(wa, 2), fmt(rw, 3), fmt(tr), fmt(rt, 3), fmt(cov), fmt(fcov),
                         fmt(vm, 2), f"{T.native_fano_1d(L_LOOP, r):.2f}", fmt(fcv, 2)])
    head = ["r", "M", "W_A / r (3.433)", "W_A / draw's prediction", "TR active (0.405)",
            "TR / draw's prediction", "coverage (0.5)", "field coverage (0.436)", "var/mean loop",
            "(predicted)", "field size CV"]
    return table(head, body), data


def arenas(rows, kind="grf", M=None) -> tuple[str, dict]:
    body, data = [], {}
    for v in ["rotated", "sparse_raw_p0.3", "sparse_raw_p0.1", "sparse_raw_p0.02", "sparse_p0.1"]:
        for k in (4, 8, 16):
            pts = []
            for x in pick(rows, kind, M=M):
                a = get(x, f"arenas.{k}r")
                if not a:
                    continue
                st = a["variants"].get(v)
                if not st:
                    continue
                r = x["config"]["r"]
                side = a["side"]
                uA = get(st, "level.uA_mean")
                sq = T.square_fields(uA, side, r) if uA and uA > 0 else None
                pts.append({"r": r, "side": side, "fpu": st["fields_per_unit"],
                            "area_r2": (st["mean_field_size"] or np.nan) / r ** 2,
                            "single": st["single_field_share_active"], "vm": st["var_over_mean"],
                            "cov": st["coverage_all"], "uA": uA,
                            "ec": sq["count"] if sq else None,
                            "area_pred": sq["area"] / r ** 2 if sq else None,
                            "area_interior": float(T.mean_area_2d(uA, r)) / r ** 2 if uA and uA > 0 else None,
                            "a95": st["area95_active"], "tr": st["tr"]["active"]})
            if not pts:
                continue
            big = [q for q in pts if q["r"] >= 8]
            use = big or pts

            def m(key):
                return ms([q[key] for q in use])
            data[f"{v}|{k}r"] = {kk: m(kk) for kk in ("fpu", "ec", "area_r2", "area_pred", "area_interior", "single",
                                                      "vm", "cov", "uA", "a95", "tr")}
            data[f"{v}|{k}r"]["by_r"] = pts
            body.append([v, f"{k} r", fmt(m("uA"), 2), fmt(m("fpu"), 2), fmt(m("ec"), 2),
                         fmt(m("area_r2"), 2), fmt(m("area_pred"), 2), fmt(m("single")), fmt(m("vm")),
                         fmt(m("cov")), fmt(m("a95")), fmt(m("tr"))])
    head = ["variant", "side", "u_A", "fields per unit", "(EC, 9-cell cut)", "field area / r^2",
            "(EC in the box, 9-cell cut)",
            "single-field share (active)", "var/mean", "coverage", "95% area (active)", "TR (active)"]
    return table(head, body), data


def short_tracks(rows) -> tuple[str, dict]:
    body, data = [], {}
    for v in ["native", "rotated", "sparse_p0.1", "sparse_raw_p0.1", "sparse_raw_p0.02"]:
        for kind, M, r in (("grf", None, 16.0), ("enc", 512, 16.0), ("enc", 512, 23.2)):
            if kind == "grf" and v == "native":
                continue
            xs = pick(rows, kind, r, M)
            if not xs:
                continue
            for LW in ("4.6", "15.4"):
                a = ms([get(x, f"variants.{v}.short.{LW}.tr.active") for x in xs])
                sz = ms([get(x, f"variants.{v}.short.{LW}.tr.silent_zero") for x in xs])
                po = ms([get(x, f"variants.{v}.short.{LW}.tr.pooled") for x in xs])
                ff = ms([get(x, f"variants.{v}.short.{LW}.tr.frac_firing") for x in xs])
                act = ms([get(x, f"variants.{v}.short.{LW}.active_frac") for x in xs])
                single = ms([get(x, f"variants.{v}.short.{LW}.single_field_share_active") for x in xs])
                pred = T.NATIVE_TR if v == "native" else float(T.treves_rolls(t_of(v)))
                lab = "GRF" if kind == "grf" else f"M={M}"
                data[f"{v}|{lab}|r{r:g}|{LW}"] = {"active": a, "silent_zero": sz, "pooled": po,
                                                  "frac_firing": ff, "active_frac": act,
                                                  "single": single, "pred_all": pred}
                body.append([v, lab, f"{r:g}", LW, fmt(ff), fmt(a), fmt(sz), fmt(po), f"{pred:.3f}",
                             fmt(act), fmt(single)])
    head = ["variant", "code", "r", "track (W_A)", "share firing", "TR, firing units", "TR, silent = 0",
            "TR, pooled", "a(t)", "share with a field", "single-field share"]
    return table(head, body), data


def scale(rows) -> tuple[str, dict]:
    """W_A and kernel widths over the grid period lambda = 12 cells, M = 512, every r."""
    body, data, dorsal = [], {}, []
    for v in ["native", "rotated"] + SPARSE_N:
        rs = sorted({x["config"]["r"] for x in pick(rows, "enc", M=512)})
        pts = []
        for r in rs:
            xs = pick(rows, "enc", r, 512)
            wa = ms([get(x, f"variants.{v}.W_A") / LAMBDA for x in xs])
            kc = ms([get(x, f"variants.{v}.kernel.cosine.fw05") / LAMBDA for x in xs])
            kp = ms([get(x, f"variants.{v}.kernel.pearson.fw05") / LAMBDA for x in xs])
            al = ms([get(x, f"alias.{v}.measured.cosine") for x in xs])
            alf = ms([np.mean([m["cosine"] for m in get(x, f"alias.{v}.measured_random_far")])
                      for x in xs if get(x, f"alias.{v}")])
            alp = ms([get(x, f"alias.{v}.measured.pearson") for x in xs])
            pts.append({"r": r, "W_A_over_lambda": wa, "fw05_cos_over_lambda": kc,
                        "fw05_pearson_over_lambda": kp, "alias_cos": al, "alias_floor_cos": alf,
                        "alias_pearson": alp})
            body.append([v, f"{r:g}", fmt(wa, 2), fmt(kc, 2), fmt(kp, 2), fmt(al), fmt(alf), fmt(alp)])
        # r where W_A / lambda crosses 0.3, 0.4, 0.5 (log-linear interpolation), and what that r implies
        if not pts:
            continue
        xr = np.array([q["r"] for q in pts])
        yw = np.array([q["W_A_over_lambda"][0] or np.nan for q in pts])

        def at(key, rr):
            y = np.array([(q[key][0] if q[key][0] is not None else np.nan) for q in pts])
            ok = np.isfinite(y)
            return float(np.interp(math.log(rr), np.log(xr[ok]), y[ok])) if ok.sum() >= 2 else None
        hit = {}
        for target in (0.3, 0.4, 0.5):
            ok = np.isfinite(yw)
            if ok.sum() >= 2 and yw[ok].min() <= target <= yw[ok].max():
                rr = float(np.exp(np.interp(math.log(target), np.log(yw[ok]), np.log(xr[ok]))))
                hit[f"{target:g}"] = {"r": rr, "fw05_cos_over_lambda": at("fw05_cos_over_lambda", rr),
                                      "fw05_pearson_over_lambda": at("fw05_pearson_over_lambda", rr),
                                      "alias_cos": at("alias_cos", rr), "alias_floor_cos": at("alias_floor_cos", rr),
                                      "alias_pearson": at("alias_pearson", rr)}
                dorsal.append([v, f"{target:g}", f"{rr:.2f}", fmt(hit[f'{target:g}']['fw05_cos_over_lambda'], 2),
                               fmt(hit[f'{target:g}']['fw05_pearson_over_lambda'], 2),
                               fmt(hit[f'{target:g}']['alias_cos']), fmt(hit[f'{target:g}']['alias_floor_cos']),
                               fmt(hit[f'{target:g}']['alias_pearson'])])
            else:
                hit[f"{target:g}"] = None
        data[v] = {"points": pts, "r_for_W_over_lambda": hit}
    head = ["variant", "r", "W_A / lambda", "FW05 cosine / lambda", "FW05 Pearson / lambda",
            "alias, cosine", "far floor, cosine", "alias, Pearson"]
    dhead = ["variant", "target W_A / lambda", "r (cells)", "FW05 cosine / lambda", "FW05 Pearson / lambda",
             "alias, cosine", "far floor, cosine", "alias, Pearson"]
    return ("Settings that reach a target W_A / lambda (interpolated in log r):\n\n" + table(dhead, dorsal)
            + "\n\nEvery r:\n\n" + table(head, body)), data


def structural(rows) -> tuple[str, dict]:
    """Spread across r (M = 512 and GRF) of statistics that should depend on p alone."""
    stats_ = {"coverage": "loops.coverage_all", "TR pooled": "loops.tr.pooled",
              "fields per field width": "loops.field_coverage_all",
              "W_A / r_eff": None, "var/mean loop": "loops.var_over_mean", "gap CV": "loops.gap_cv_median"}
    body, data = [], {}
    for kind, M in (("grf", None), ("enc", 512)):
        for v in ["rotated", "sparse_raw_p0.1", "sparse_raw_p0.02"]:
            for name, path in stats_.items():
                by_r = {}
                for r in sorted({x["config"]["r"] for x in pick(rows, kind, M=M)}):
                    xs = pick(rows, kind, r, M)
                    if path is None:
                        vals = [get(x, f"variants.{v}.W_A") / r_eff(x) for x in xs]
                    else:
                        vals = [get(x, f"variants.{v}.{path}") for x in xs]
                    by_r[r] = ms(vals)[0]
                main = [by_r[r] for r in by_r if r >= 8 and by_r[r] is not None]
                allr = [by_r[r] for r in by_r if r >= 4 and by_r[r] is not None]
                small = [by_r[r] for r in by_r if r < 4 and by_r[r] is not None]
                lab = "GRF" if kind == "grf" else f"M={M}"
                cv_main = float(np.std(main) / np.mean(main)) if main and np.mean(main) else None
                data[f"{lab}|{v}|{name}"] = {"by_r": by_r, "cv_r_ge_8": cv_main}
                body.append([lab, v, name, fmt(float(np.min(main)) if main else None),
                             fmt(float(np.max(main)) if main else None),
                             fmt(cv_main) if cv_main is not None else "-",
                             fmt(by_r.get(4.0)), fmt(by_r.get(2.0)) if small else "-"])
    head = ["code", "variant", "statistic", "min over r = 8..32", "max over r = 8..32", "CV over r = 8..32",
            "r = 4", "r = 2"]
    return table(head, body), data


SECTIONS = [
    ("Raw kernel and alias against r and M", raw_kernel),
    ("Single units, GRF limit, every r (Rice predictions in brackets)", lambda R: single_unit(R, "grf")),
    ("Single units, encoder M = 512, every r", lambda R: single_unit(R, "enc", 512)),
    ("Rate kernels in units of r (r >= 8)", kernels),
    ("r that holds FW05 at the raw r = 16 kernel's 37.7 cells", lambda R: width_matching()),
    ("Field counts, GRF limit", lambda R: counts(R, "grf")),
    ("Field counts, encoder M = 512", lambda R: counts(R, "enc", 512)),
    ("Finite M at r = 16", finite_M),
    ("Per-unit spectral spread", spectral),
    ("Alias in the rate kernels: measured at the worst displacement against g(K)", alias_rates),
    ("Native plane-wave code", native),
    ("2D arenas, GRF limit (r >= 8)", lambda R: arenas(R, "grf")),
    ("2D arenas, encoder M = 512 (r >= 8)", lambda R: arenas(R, "enc", 512)),
    ("Treves-Rolls on short tracks", short_tracks),
    ("Scale: widths over the grid period (lambda = 12 cells), M = 512", scale),
    ("Structural claim: spread across r", structural),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=OUT)
    a = ap.parse_args()
    rows = load(a.root)
    print(f"{len(rows)} configs")
    md, summary = [f"# Ideal family: tables ({len(rows)} configs)\n"], {}
    for title, fn in SECTIONS:
        try:
            tab, data = fn(rows)
        except Exception as e:                       # one empty section must not stop the rest
            tab, data = f"(failed: {e!r})", {}
        md.append(f"## {title}\n\n{tab}\n")
        summary[title] = data
    (a.root / "tables.md").write_text("\n".join(md))
    (a.root / "summary.json").write_text(json.dumps(summary, default=float, indent=1))
    print(f"wrote {a.root / 'tables.md'}")


if __name__ == "__main__":
    main()
