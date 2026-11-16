"""Figures for the ideal-family write-up.

    python -m analysis.ideal_family.figures [--root RESULTS] [--out FIGURES]

PDFs use the 14 core fonts (Helvetica, Symbol) and no mathtext, so they open
cleanly in Illustrator. Lines are theory; markers are measurements (mean over
seeds, bars one SD).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402

from . import theory as T                                           # noqa: E402
from .analyse import LAMBDA, L_LONG, L_LOOP, get, load, ms, pick, r_eff  # noqa: E402
from .codes import VARIANT_PS                                       # noqa: E402
from .run import OUT                                                # noqa: E402

FIG = Path("/orcd/pool/003/jackking/cls_runs/figures/ideal_family/v1")

plt.rcParams.update({
    "pdf.use14corefonts": True, "pdf.fonttype": 42, "ps.useafm": True,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False, "text.usetex": False, "mathtext.default": "regular",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "legend.frameon": False,
})
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
# Ordinal blue ramp for p (0.5 lightest to 0.02 darkest), steps 250..700 of the reference palette.
P_COL = dict(zip(VARIANT_PS, ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]))
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]     # categorical slots 1-3
R_COL = dict(zip([4.0, 8.0, 16.0, 23.2, 32.0], ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]))


def raw_name(p):
    return "rotated" if p == 0.5 else f"sparse_raw_p{p:g}"


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def errpt(ax, x, mstd, **kw):
    m, s = mstd
    if m is None:
        return
    ax.errorbar([x], [m], yerr=[s or 0], fmt=kw.pop("fmt", "o"), ms=kw.pop("ms", 3.5),
                capsize=0, elinewidth=0.8, **kw)


def grid_(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
def fig_kernels(rows, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    d = np.linspace(0, 4, 200)
    rho = np.exp(-0.5 * d * d)
    ax = axs[0]
    ax.plot(d, rho, color=INK, lw=1.0, ls=":", label="raw code")
    for p in (0.5, 0.1, 0.02):
        t = T.t_of_p(p)
        ax.plot(d, T.kernel_cos(rho, t), color=P_COL[p], lw=1.2, label=f"cosine, p = {p:g}")
        ax.plot(d, T.kernel_pearson(rho, t), color=P_COL[p], lw=1.2, ls="--")
        xs = pick(rows, "grf", 16.0)
        if xs:
            for form, mk in (("cosine", "o"), ("pearson", "s")):
                s = np.nanmean([get(x, f"variants.{raw_name(p)}.kernel.{form}.s") for x in xs], axis=0)
                lag = np.arange(s.size) / 16.0
                sel = (lag <= 4) & (np.arange(s.size) % 6 == 0)
                ax.plot(lag[sel], s[sel], mk, ms=2.5, mfc="white", mec=P_COL[p], mew=0.8)
    ax.set_xlabel("separation / r")
    ax.set_ylabel("population similarity")
    ax.set_title("Thresholded kernel K(rho(d); t)\nlines theory (solid cosine, dashed Pearson),\nmarkers GRF r = 16")
    ax.legend(loc="upper right", fontsize=6)
    grid_(ax)

    ax = axs[1]
    ps = np.geomspace(0.01, 0.6, 60)
    for form, ls, lab in (("cos", "-", "cosine"), ("pearson", "--", "Pearson")):
        ax.plot(ps, [T.kernel_width(0.5, 1.0, T.t_of_p(p), form) for p in ps], color=INK2, ls=ls,
                label=f"{lab}, theory")
    ax.axhline(T.gaussian_width(0.5, 1.0), color=INK, ls=":", lw=1.0, label="raw code")
    for kind, M, c, mk in (("grf", None, CAT[2], "o"), ("enc", 512, CAT[1], "s")):
        xs = [x for x in pick(rows, kind, M=M) if x["config"]["r"] >= 8]
        for p in VARIANT_PS:
            for form, key in (("cos", "cosine"), ("pearson", "pearson")):
                v = ms([get(x, f"variants.{raw_name(p)}.kernel.{key}.fw05") / r_eff(x) for x in xs])
                errpt(ax, p * (1.04 if kind == "enc" else 0.96), v, color=c, fmt=mk,
                      mfc=c if key == "cosine" else "white", mec=c,
                      label=None)
    ax.plot([], [], "o", color=CAT[2], ms=3.5, label="GRF (r = 8 to 32)")
    ax.plot([], [], "s", color=CAT[1], ms=3.5, label="encoder, M = 512")
    ax.set_xscale("log")
    ax.set_xlabel("active fraction p")
    ax.set_ylabel("full width at 0.5 / r")
    ax.set_title("Thresholding narrows the kernel\n(filled cosine, open Pearson)")
    ax.legend(fontsize=6, loc="lower right")
    grid_(ax)

    ax = axs[2]
    target = T.gaussian_width(0.5, 16.0)
    ax.plot(ps, [T.r_for_width(target, 0.5, T.t_of_p(p), "cos") for p in ps], color=INK2, label="cosine")
    ax.plot(ps, [T.r_for_width(target, 0.5, T.t_of_p(p), "pearson") for p in ps], color=INK2, ls="--",
            label="Pearson")
    ax.plot([0.1], [23.2], "D", color=CAT[1], ms=4, label="viewer's sparse code (r = 23.2)")
    ax.axhline(16.0, color=INK, ls=":", lw=1.0, label="raw code, r = 16")
    ax.set_xscale("log")
    ax.set_xlabel("active fraction p")
    ax.set_ylabel("r (cells)")
    ax.set_title("r that keeps FW05 = 37.7 cells")
    ax.legend(fontsize=6)
    grid_(ax)
    fig.tight_layout()
    save(fig, out, "kernels")


def fig_single_units(rows, out):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2))
    ps = np.geomspace(0.01, 0.6, 60)
    ts = [T.t_of_p(p) for p in ps]
    rs = [8.0, 16.0, 32.0]
    # coverage
    ax = axs[0]
    ax.plot(ps, ps, color=INK2, label="p")
    for i, r in enumerate(rs):
        for p in VARIANT_PS:
            xs = pick(rows, "enc", r, 512)
            errpt(ax, p * (0.95 + 0.05 * i), ms([get(x, f"variants.{raw_name(p)}.loops.coverage_all") for x in xs]),
                  color=R_COL[r])
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("p"), ax.set_ylabel("coverage (rate > 0)")
    grid_(ax)
    # Treves-Rolls
    ax = axs[1]
    ax.plot(ps, T.treves_rolls(np.array(ts)), color=INK2, label="a(t), all units")
    for i, r in enumerate(rs):
        xs = pick(rows, "enc", r, 512)
        for p in VARIANT_PS:
            errpt(ax, p, ms([get(x, f"variants.{raw_name(p)}.loops.tr.pooled") for x in xs]), color=R_COL[r])
            errpt(ax, p, ms([get(x, f"variants.{raw_name(p)}.short.4.6.tr.active") for x in xs]),
                  color=R_COL[r], fmt="^", mfc="white", mec=R_COL[r])
    ax.plot([], [], "o", color=INK2, ms=3, label="loops, pooled")
    ax.plot([], [], "^", color=INK2, mfc="white", ms=3, label="4.6 W_A track, firing units")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("p"), ax.set_ylabel("Treves-Rolls sparsity")
    ax.legend(fontsize=5.5, loc="upper left")
    grid_(ax)
    # W_A / r
    ax = axs[2]
    for r in rs:
        ax.plot(ps, [T.def_A_1d(t, L_LOOP, r)["W_A"] / r for t in ts], color=R_COL[r], lw=1.0)
        xs = pick(rows, "enc", r, 512)
        for p in VARIANT_PS:
            errpt(ax, p, ms([get(x, f"variants.{raw_name(p)}.W_A") / r_eff(x) for x in xs]), color=R_COL[r])
    ax.set_xscale("log")
    ax.set_xlabel("p"), ax.set_ylabel("W_A / r (1716-cell loop)")
    grid_(ax)
    # fields per field width
    ax = axs[3]
    for r in rs:
        ax.plot(ps, [T.def_A_1d(t, L_LOOP, r)["field_coverage"] for t in ts], color=R_COL[r], lw=1.0,
                label=f"r = {r:g}")
        xs = pick(rows, "enc", r, 512)
        for p in VARIANT_PS:
            errpt(ax, p, ms([get(x, f"variants.{raw_name(p)}.loops.field_coverage_all") for x in xs]),
                  color=R_COL[r])
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("p"), ax.set_ylabel("fields per field width")
    ax.legend(fontsize=6)
    grid_(ax)
    fig.suptitle("Single units, encoder M = 512 (markers) against Rice and the GRF maximum (lines)",
                 fontsize=8)
    fig.tight_layout()
    save(fig, out, "single_units")


def fig_counts(rows, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    ts = np.linspace(-0.2, 2.2, 25)
    ps = [float(T.Phibar(t)) for t in ts]
    for ax, L, key, title in ((axs[0], L_LOOP, "loops", "1716-cell loop"),
                              (axs[1], L_LONG, "long", "6864-cell environment")):
        for r in (8.0, 32.0):
            pr = [T.fano_def_A(t, L, r) for t in ts]
            ax.plot(ps, [q["total"] for q in pr], color=R_COL[r], ls="--", lw=0.9)
            ax.plot(ps, [q["total_peaks"] for q in pr], color=R_COL[r], ls="-", lw=0.9)
            ax.plot(ps, [q["F"] for q in pr], color=R_COL[r], ls=":", lw=0.9)
            for kind, mk in (("grf", "o"), ("enc", "s")):
                xs = pick(rows, kind, r, None if kind == "grf" else 512)
                for p in VARIANT_PS:
                    errpt(ax, p, ms([get(x, f"variants.{raw_name(p)}.{key}.var_over_mean") for x in xs]),
                          color=R_COL[r], fmt=mk, mfc=R_COL[r] if kind == "grf" else "white", mec=R_COL[r])
        ax.axhline(1, color=INK, lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("p")
        ax.set_ylabel("variance / mean of field counts")
        ax.set_title(title)
        grid_(ax)
    axs[0].plot([], [], ":", color=INK2, label="F(u_A), fixed level")
    axs[0].plot([], [], "--", color=INK2, label="+ level term (low u_A)")
    axs[0].plot([], [], "-", color=INK2, label="peak form (high u_A)")
    axs[0].plot([], [], "o", color=INK2, ms=3, label="GRF")
    axs[0].plot([], [], "s", color=INK2, mfc="white", ms=3, label="encoder M = 512")
    axs[0].legend(fontsize=5.5, loc="upper left")
    ax = axs[2]
    u = np.linspace(-1, 4, 80)
    ax.plot(u, [T.fano_upcross(x, 10.0) for x in u], color=INK2, label="F(u), Rice second moment")
    ax.plot(u, [math.sqrt(T.fano_upcross(x, 10.0)) for x in u], color=INK2, ls="--", label="sqrt F (renewal gap CV)")
    for kind, mk in (("grf", "o"), ("enc", "s")):
        for r in (8.0, 16.0, 32.0):
            xs = pick(rows, kind, r, None if kind == "grf" else 512)
            for p in VARIANT_PS:
                uA = ms([get(x, f"variants.{raw_name(p)}.level.uA_mean") for x in xs])[0]
                if uA is None:
                    continue
                errpt(ax, uA, ms([get(x, f"variants.{raw_name(p)}.loops.gap_cv_median") for x in xs]),
                      color=R_COL[r], fmt=mk, mfc=R_COL[r] if kind == "grf" else "white", mec=R_COL[r])
    ax.axhline(1, color=INK, lw=0.6)
    ax.set_xlabel("level u_A (SDs)")
    ax.set_ylabel("Fano factor / gap CV")
    ax.set_title("Gap CV between a unit's fields")
    ax.legend(fontsize=5.5, loc="lower right")
    grid_(ax)
    fig.tight_layout()
    save(fig, out, "counts")


def fig_finite_M(rows, out):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2))
    Ms = np.array([32, 64, 128, 256, 512, 1024, 2048])
    ax = axs[0]
    for r in [4.0, 8.0, 16.0, 23.2, 32.0]:
        excl = 3 * T.gaussian_width(0.5, r)
        ax.plot(Ms, [T.alias_prediction(int(M), r, excl_radius=excl)["alias"] for M in Ms], color=R_COL[r],
                lw=0.9, label=f"r = {r:g}")
        for M in Ms:
            errpt(ax, M, ms([get(x, "alias.raw.max") for x in pick(rows, "enc", r, int(M))]), color=R_COL[r])
    ax.set_xscale("log", base=2), ax.set_yscale("log")
    ax.set_xlabel("M"), ax.set_ylabel("largest far kernel value")
    ax.set_title("Alias over the whole scaffold")
    ax.legend(fontsize=5.5)
    grid_(ax)
    ax = axs[1]
    for v, c in (("rotated", CAT[0]), ("sparse_raw_p0.1", CAT[1])):
        t = T.t_of_p(0.5 if v == "rotated" else 0.1)
        pr = T.fano_def_A(t, L_LONG, 16.0)
        base = pr["total"] if t < 1 else pr["total_peaks"]
        pred = []
        for M in Ms:
            xs = pick(rows, "enc", 16.0, int(M))
            errpt(ax, M, ms([get(x, f"variants.{v}.long_x.var_over_mean") for x in xs]), color=c)
            if xs:
                pred.append(base + np.mean([get(x, f"variants.{v}.long_x.fields_per_active") *
                                            get(x, "spectral.lam_x_cv2") / 4 for x in xs]))
            else:
                pred.append(np.nan)
        ax.plot(Ms, pred, color=c, lw=0.9, label=f"{'rectified' if v == 'rotated' else 'p = 0.1'}")
        g = pick(rows, "grf", 16.0)
        errpt(ax, 4096, ms([get(x, f"variants.{v}.long_x.var_over_mean") for x in g]), color=c, fmt="D")
    ax.axhline(1, color=INK, lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 128, 512, 2048, 4096])
    ax.set_xticklabels(["32", "128", "512", "2048", "GRF"])
    ax.set_xlabel("M"), ax.set_ylabel("var / mean, 6864 cells, r = 16")
    ax.set_title("Units of one draw differ")
    ax.legend(fontsize=6)
    grid_(ax)
    ax = axs[2]
    for v, c in (("rotated", CAT[0]), ("sparse_raw_p0.1", CAT[1]), ("sparse_raw_p0.02", CAT[2])):
        t = T.t_of_p({"rotated": 0.5, "sparse_raw_p0.1": 0.1, "sparse_raw_p0.02": 0.02}[v])
        ax.axhline(T.def_A_1d(t, L_LOOP, 16.0)["W_A"] / 16.0, color=c, lw=0.8, ls="--")
        for M in Ms:
            errpt(ax, M, ms([get(x, f"variants.{v}.W_A") / r_eff(x) for x in pick(rows, "enc", 16.0, int(M))]),
                  color=c)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M"), ax.set_ylabel("W_A / r_eff, r = 16")
    ax.set_title("Field width")
    grid_(ax)
    ax = axs[3]
    for v, c in (("rotated", CAT[0]), ("sparse_raw_p0.1", CAT[1]), ("sparse_raw_p0.02", CAT[2])):
        g = pick(rows, "grf", 16.0)
        gm = ms([get(x, f"variants.{v}.loops.gap_cv_median") for x in g])[0]
        if gm:
            ax.axhline(gm, color=c, lw=0.8, ls="--")
        for M in Ms:
            errpt(ax, M, ms([get(x, f"variants.{v}.loops.gap_cv_median") for x in pick(rows, "enc", 16.0, int(M))]),
                  color=c)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M"), ax.set_ylabel("gap CV (median unit), r = 16")
    ax.set_title("Quasi-periodic units at small M\n(dashed: GRF)")
    grid_(ax)
    fig.tight_layout()
    save(fig, out, "finite_M")


def fig_arenas(rows, out):
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.5))
    for kind, mk in (("grf", "o"), ("enc", "s")):
        for x in pick(rows, kind, M=None if kind == "grf" else 512):
            r = x["config"]["r"]
            for kk, a in (x.get("arenas") or {}).items():
                for p in VARIANT_PS:
                    st = a["variants"].get(raw_name(p))
                    uA = get(st, "level.uA_mean")
                    if not st or uA is None:
                        continue
                    if uA <= 0:
                        continue
                    sq = T.square_fields(uA, a["side"], r)
                    col = P_COL[p]
                    axs[0].plot(sq["count"], st["fields_per_unit"], mk, ms=2.5, color=col,
                                mfc=col if kind == "grf" else "white", mec=col)
                    if st["mean_field_size"]:
                        axs[1].plot(sq["area"] / r ** 2, st["mean_field_size"] / r ** 2, mk,
                                    ms=2.5, color=col, mfc=col if kind == "grf" else "white", mec=col)
    for ax in axs:
        lim = [0.05, 400]
        ax.plot(lim, lim, color=INK, lw=0.6)
        ax.set_xscale("log"), ax.set_yscale("log")
        grid_(ax)
    axs[0].set_xlabel("EC count at u_A, 9-cell cut")
    axs[0].set_ylabel("fields per unit (measured)")
    axs[0].set_title("Field counts in square arenas")
    axs[1].set_xlabel("EC area in the box / r^2, 9-cell cut")
    axs[1].set_ylabel("mean field area / r^2 (measured)")
    axs[1].set_title("Field areas")
    for p in VARIANT_PS:
        axs[1].plot([], [], "o", color=P_COL[p], ms=3, label=f"p = {p:g}")
    axs[1].legend(fontsize=5.5, loc="upper left")
    fig.tight_layout()
    save(fig, out, "arenas_2d")


def fig_native(rows, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    Ms = [32, 64, 128, 256, 512, 1024, 2048]
    ax = axs[0]
    ax.axhline(T.native_mean_width_1d(1.0), color=INK2, lw=0.9, label="2 arccos(0.2) sqrt(pi/2) = 3.43")
    for r in [4.0, 8.0, 16.0, 23.2, 32.0]:
        for M in Ms:
            errpt(ax, M, ms([get(x, "variants.native.W_A") / r for x in pick(rows, "enc", r, M)]), color=R_COL[r])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M"), ax.set_ylabel("native W_A / r")
    ax.legend(fontsize=6)
    grid_(ax)
    ax = axs[1]
    m = np.array(Ms, dtype=float)
    ax.fill_between(m, -1.07 / np.sqrt(m), 1.07 / np.sqrt(m), color=GRID, label="+- 1.07 / sqrt(M)")
    for r in [4.0, 8.0, 16.0, 23.2, 32.0]:
        for M in Ms:
            for x in pick(rows, "enc", r, M):
                a = get(x, "variants.native.loops_x.mean_field_size")
                b = get(x, "variants.native.loops_y.mean_field_size")
                if a and b:
                    ax.plot(M, math.log(a / b), "o", ms=2.5, color=R_COL[r])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M"), ax.set_ylabel("ln(W_A along x / along y)")
    ax.set_title("Direction dependence of one draw")
    ax.legend(fontsize=6)
    grid_(ax)
    ax = axs[2]
    rr = np.linspace(3, 34, 50)
    ax.plot(rr, [T.native_fano_1d(L_LOOP, x) for x in rr], color=INK2, lw=0.9,
            label="(L / 2 pi r)(1 - 2/pi) / sqrt(2/pi)")
    for r in [4.0, 8.0, 16.0, 23.2, 32.0]:
        errpt(ax, r, ms([get(x, "variants.native.loops.var_over_mean") for x in pick(rows, "enc", r, 512)]),
              color=R_COL[r])
    ax.set_xlabel("r"), ax.set_ylabel("native var / mean, 1716-cell loop")
    ax.set_yscale("log")
    ax.legend(fontsize=6)
    grid_(ax)
    fig.tight_layout()
    save(fig, out, "native")


def fig_scale(rows, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    rs = sorted({x["config"]["r"] for x in pick(rows, "enc", M=512)})
    for ax in axs[:2]:
        ax.axhspan(0.3, 0.5, color=GRID, lw=0)
    for p in VARIANT_PS:
        v = f"sparse_p{p:g}"
        c = P_COL[p]
        wa = [ms([get(x, f"variants.{v}.W_A") / LAMBDA for x in pick(rows, "enc", r, 512)]) for r in rs]
        axs[0].errorbar(rs, [w[0] for w in wa], yerr=[w[1] for w in wa], color=c, marker="o", ms=3,
                        lw=1.0, capsize=0, label=f"p = {p:g}")
        kp = [ms([get(x, f"variants.{v}.kernel.pearson.fw05") / LAMBDA for x in pick(rows, "enc", r, 512)])
              for r in rs]
        axs[1].errorbar(rs, [w[0] for w in kp], yerr=[w[1] for w in kp], color=c, marker="o", ms=3, lw=1.0,
                        capsize=0)
        al = [ms([(get(x, f"alias.{v}.measured.pearson") or np.nan) for x in pick(rows, "enc", r, 512)])
              for r in rs]
        axs[2].errorbar(rs, [w[0] for w in al], yerr=[w[1] for w in al], color=c, marker="o", ms=3, lw=1.0,
                        capsize=0)
    wa = [ms([get(x, "variants.native.W_A") / LAMBDA for x in pick(rows, "enc", r, 512)]) for r in rs]
    axs[0].plot(rs, [w[0] for w in wa], color=CAT[1], marker="s", ms=3, lw=1.0, label="native")
    raw = [ms([get(x, "raw_kernel.fw05") / LAMBDA for x in pick(rows, "enc", r, 512)]) for r in rs]
    axs[1].plot(rs, [w[0] for w in raw], color=INK, ls=":", marker="s", ms=3, lw=1.0, label="raw code")
    for ax in axs:
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlabel("r (cells)")
        grid_(ax)
    axs[0].set_ylabel("W_A / lambda (lambda = 12 cells)")
    axs[0].set_title("Field width over the grid period\n(band: dorsal CA1, 0.3 to 0.5)")
    axs[0].legend(fontsize=5.5, loc="upper left")
    axs[1].set_ylabel("kernel FW05 (Pearson) / lambda")
    axs[1].set_title("Population kernel width")
    axs[1].legend(fontsize=6)
    axs[2].set_ylabel("Pearson similarity at the worst far displacement")
    axs[2].set_title("Largest alias, M = 512")
    fig.tight_layout()
    save(fig, out, "scale")


def fig_structural(rows, out):
    stats_ = [("coverage", "loops.coverage_all"), ("Treves-Rolls", "loops.tr.pooled"),
              ("W_A / r_eff", None), ("fields per field width", "loops.field_coverage_all"),
              ("var / mean, loop", "loops.var_over_mean"), ("gap CV", "loops.gap_cv_median")]
    fig, axs = plt.subplots(1, len(stats_), figsize=(7.2, 2.0))
    for ax, (name, path) in zip(axs, stats_):
        for p in (0.5, 0.1, 0.02):
            v = raw_name(p)
            for kind, mk in (("grf", "o"), ("enc", "s")):
                rs = sorted({x["config"]["r"] for x in pick(rows, kind, M=None if kind == "grf" else 512)})
                ys = []
                for r in rs:
                    xs = pick(rows, kind, r, None if kind == "grf" else 512)
                    vals = [get(x, f"variants.{v}.W_A") / r_eff(x) for x in xs] if path is None else \
                        [get(x, f"variants.{v}.{path}") for x in xs]
                    ys.append(ms(vals)[0])
                ax.plot(rs, [np.nan if y is None else y for y in ys], marker=mk, ms=2.5, lw=0.8,
                        color=P_COL[p], mfc=P_COL[p] if kind == "grf" else "white", mec=P_COL[p])
        ax.set_xscale("log")
        if name in ("coverage", "Treves-Rolls", "fields per field width"):
            ax.set_yscale("log")
        ax.set_xlabel("r")
        ax.set_title(name)
        grid_(ax)
    for p in (0.5, 0.1, 0.02):
        axs[0].plot([], [], color=P_COL[p], label=f"p = {p:g}")
    axs[0].legend(fontsize=5.5)
    fig.suptitle("Scale-free statistics against r (circles GRF, squares encoder M = 512); "
                 "fixed 1716-cell loops", fontsize=8)
    fig.tight_layout()
    save(fig, out, "structural")


FIGS = [fig_kernels, fig_single_units, fig_counts, fig_finite_M, fig_arenas, fig_native, fig_scale,
        fig_structural]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=FIG)
    a = ap.parse_args()
    rows = load(a.root)
    for f in FIGS:
        try:
            f(rows, a.out)
            print("wrote", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
