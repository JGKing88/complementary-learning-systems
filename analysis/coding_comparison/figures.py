"""Figures for docs/EXPERIMENTS_CODING_COMPARISON.md.

    python -m analysis.coding_comparison.figures [--tag v1]

Reads ``<results>/coding_comparison/<tag>/`` and writes PNGs to
``<figures>/coding_comparison/<tag>/``. Every figure is small multiples, one
panel per population, so a population is named by its panel title rather
than by a colour.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from cls_paths import figures_dir, results_dir  # noqa: E402

from .scorecard import POP_ORDER, load_results, load_targets, rich_zero_fraction  # noqa: E402

# Reference palette (dataviz skill), categorical slots in fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
LABELS = {"grid": "Grid code", "ideal": "Ideal", "ideal_rot": "Ideal, rotated",
          "ideal_sparse": "Sparse ReLU", "trained": "Trained", "untrained": "Untrained",
          "hippocampus": "Hippocampus"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "figure.dpi": 150,
    "savefig.bbox": "tight", "axes.titlecolor": INK,
})


def primary_basis(R) -> str:
    return "rectify" if R["meta"]["signed"] else "identity"


def grid_axes(n, ncols=4, w=2.1, h=1.8, sharey=True):
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), sharey=sharey,
                            squeeze=False)
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    return fig, axs.flat


def fig_kernel(res, T, out):
    pops = list(res)
    fig, axs = grid_axes(len(pops))
    pts = np.array(T["kjelstrup2008_ventral_curve"]["points_W"])
    for ax, pop in zip(axs, pops):
        R = res[pop]
        b = primary_basis(R)
        W = R["meta"]["calibration"][f"{b}|A"]["W"]
        k = R["kernel_1d"]["constructions"]
        d = np.arange(len(k["raw|cosine"]["s"])) / W
        ax.plot(d, k["raw|cosine"]["s"], color=BLUE, label="cosine, raw code")
        ax.plot(d, k[f"{b}|pearson"]["s"], color=ORANGE, label=f"Pearson, {b} rates")
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=3.5, color=INK, label="ventral CA3 (Kjelstrup)")
        ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlim(0, 5)
        ax.set_title(f"{LABELS.get(pop, pop)} (W_A = {W:.3g} cells)")
        ax.set_xlabel("separation / mean field width")
    axs[0].set_ylabel("PV similarity")
    axs[0].legend(loc="upper right", frameon=False)
    fig.suptitle("PV similarity against separation along scaffold loops; dashed line at one field width",
                 fontsize=8.5, color=INK2)
    fig.savefig(out / "kernel_1d.png")
    plt.close(fig)


def fig_zero_curve(res, T, out):
    pops = [p for p in res if "fields_1d" in res[p]]
    fig, axs = grid_axes(len(pops))
    L = np.geomspace(2, 160, 100)
    rich = np.array([rich_zero_fraction(x, T) for x in L])
    for ax, pop in zip(axs, pops):
        R = res[pop]
        b = primary_basis(R)
        ax.fill_between(L, np.clip(rich - 0.155, 0, 1), np.clip(rich + 0.155, 0, 1),
                        color=GRID, lw=0)
        ax.plot(L, rich, color=MUTED, ls="--", label="Rich 2014 fit")
        for rate, col in (("0.5", AQUA), ("0.75", BLUE), ("1", ORANGE)):
            v = R["fields_1d"]["variants"].get(f"{b}|B{rate}", {})
            pts = sorted((x["length_W"], x["fit"]["frac_zero"]) for x in v.get("lengths", {}).values())
            if pts:
                a = np.array(pts)
                ax.plot(a[:, 0], a[:, 1], "o-", ms=3, color=col, label=f"B, {rate} Hz")
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(LABELS.get(pop, pop))
        ax.set_xlabel("track length / mean field width")
    axs[0].set_ylabel("units with no field")
    axs[0].legend(loc="lower left", frameon=False)
    fig.suptitle("Zero-field fraction against track length (definition B); band is Rich's curve +- 0.155",
                 fontsize=8.5, color=INK2)
    fig.savefig(out / "zero_curve.png")
    plt.close(fig)


def fig_counts(res, T, out):
    pops = [p for p in res if "fields_1d" in res[p]]
    fig, axs = grid_axes(len(pops), sharey=False)
    r0 = T["rich2014"]
    p_rich = r0["beta_m"] / (r0["beta_m"] + r0["track_m"])
    for ax, pop in zip(axs, pops):
        R = res[pop]
        b = primary_basis(R)
        x = R["fields_1d"]["variants"].get(f"{b}|B0.75", {}).get("lengths", {}).get("73.8")
        ax.set_title(LABELS.get(pop, pop))
        ax.set_xlabel("fields per unit, 74 widths")
        if x is None:
            ax.text(0.5, 0.5, "no fields", transform=ax.transAxes, ha="center", color=INK2)
            continue
        fit = x["fit"]
        mean = fit["mean"]
        kmax = int(max(10, np.ceil(mean * 3)))
        k = np.arange(kmax + 1)
        pois = stats.poisson.pmf(k, mean)
        nb = fit["negbin"]
        ax.plot(k, pois, color=AQUA, label="Poisson fit")
        if nb and nb.get("r"):
            ax.plot(k, stats.nbinom.pmf(k, nb["r"], nb["p"]), color=BLUE, label="neg. binomial fit")
        ax.plot(k, stats.nbinom.pmf(k, r0["r"], p_rich), color=MUTED, ls="--", label="Rich 2014")
        r_txt = "inf" if (nb or {}).get("r") is None else f"{nb['r']:.2f}"
        ax.text(0.97, 0.95, f"mean {mean:.2f}\nvar/mean {fit['var_over_mean']:.2f}\nr {r_txt}",
                transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=7)
    axs[0].set_ylabel("probability")
    axs[0].legend(loc="center right", frameon=False)
    fig.suptitle("Field-count fits on the 74-width track (definition B, 0.75 Hz); the histogram itself is in fields_1d.json",
                 fontsize=8.5, color=INK2)
    fig.savefig(out / "counts_74W.png")
    plt.close(fig)


def fig_area(res, T, out):
    pops = [p for p in res if "fields_2d" in res[p]]
    fig, axs = grid_axes(len(pops), sharey=False)
    for ax, pop in zip(axs, pops):
        R = res[pop]
        b = primary_basis(R)
        A0 = R["fields_2d"]["A0_cells"]
        for d, col in (("A", MUTED), ("Ap", AQUA), ("B0.75", BLUE)):
            rows = R["fields_2d"]["variants"].get(f"{b}|{d}", {}).get("rows", [])
            if rows:
                a = np.array([[r["area_cells"] / A0, r["fields_per_unit"]] for r in rows])
                ax.plot(a[:, 0], a[:, 1], "o-", ms=3, color=col, label=d)
        ax.set_title(LABELS.get(pop, pop))
        ax.set_xlabel("arena area / A0")
    axs[0].set_ylabel("fields per unit")
    axs[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Fields per unit against arena area (A0 = pi/4 W_A^2); Tanni 2022 and Harland 2021 find it linear",
                 fontsize=8.5, color=INK2)
    fig.savefig(out / "area_scaling.png")
    plt.close(fig)


def fig_basis(res, T, out):
    """Single-unit measures in the native basis against three random rotations."""
    pops = [p for p in res if res[p]["meta"]["signed"] and "fields_1d" in res[p]]
    if not pops:
        return
    measures = [("var/mean, 74 W (B)", "B0.75", "73.8", ("fit", "var_over_mean")),
                ("gap CV, 74 W (B)", "B0.75", "73.8", ("layout", "gap_cv_median")),
                ("field size CV, 74 W (B)", "B0.75", "73.8", ("field_size", "cv")),
                ("active fraction, 34 W (Ap)", "Ap", "33.8", ("active_frac",)),
                ("single-field share, 34 W (Ap)", "Ap", "33.8", ("single_field_frac_active",))]
    fig, axs = plt.subplots(1, len(measures), figsize=(2.0 * len(measures), 0.45 * len(pops) + 0.9),
                            sharey=True)
    for ax, (name, d, L, path) in zip(axs, measures):
        for i, pop in enumerate(pops):
            R = res[pop]
            vs = R["fields_1d"]["variants"]
            def get(key):
                x = vs.get(key, {}).get("lengths", {}).get(L)
                for p in path:
                    x = None if x is None else x.get(p)
                return x
            nat = get(f"rectify|{d}")
            rots = [get(f"rot{s}|rectify|{d}") for s in R["meta"]["params"]["rot_seeds"]]
            rots = [r for r in rots if r is not None]
            if rots:
                ax.plot(rots, [i] * len(rots), "o", ms=4, mfc="none", color=ORANGE,
                        label="rotated" if i == 0 else None)
            if nat is not None:
                ax.plot([nat], [i], "o", ms=5, color=BLUE, label="native" if i == 0 else None)
        ax.set_title(name)
        ax.set_yticks(range(len(pops)))
        ax.set_yticklabels([LABELS.get(p, p) for p in pops])
        ax.set_ylim(-0.6, len(pops) - 0.4)
    axs[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Single-unit measures depend on the basis: rectified units, native against three random rotations",
                 fontsize=8.5, color=INK2)
    fig.savefig(out / "basis_dependence.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", default="v1")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    root = a.root or (results_dir() / "coding_comparison" / a.tag)
    out = a.out or (figures_dir() / "coding_comparison" / a.tag)
    out.mkdir(parents=True, exist_ok=True)
    T = load_targets()
    res = {pop: load_results(root / pop) for pop in POP_ORDER if (root / pop / "meta.json").exists()}
    for f in (fig_kernel, fig_zero_curve, fig_counts, fig_area, fig_basis):
        f(res, T, out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
