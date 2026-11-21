"""Figures and summary numbers: the trained encoder (att0.5) against the ideal, from the spectra
saved by trained_spectrum.py.

    python analysis/hopfield_probe/figures/trained_spectrum_figs.py [DATA_DIR]

S(n) is each encoder's share of power on the plane wave n = (n1, n2) over the 1716 x 1716 torus;
the position-averaged kernel is k(D) = L^2 ifft2(S). The target is the ideal encoder's
expected spectrum P(n) at r = 16: each axis of n is round(Normal(0, L / (2 pi r))).

Writes, next to this script (core Helvetica/Symbol PDF and PNG):
  spectrum_main         a kernel against distance, b power per wave against |n|,
                        c power by module 13's harmonic |j13| (x component of n). A shift of
                        792 cells moves module 13 by one cell and leaves 11 and 12 in place, so
                        k(792, 0) = sum_n S(n) cos(2 pi j13(n1) / 13) exactly
  spectrum_maps         S(n) for |n| <= 80, and the on-axis spectrum (sum over n2)
  spectrum_kernel_maps  k(D) near the goal and over the whole arena
  spectrum_translation  z(x).z(x+D) across 20,000 positions x, at fixed D
and spectrum_summary.json (numbers quoted in the text).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, NullLocator
from scipy.special import ndtr

import style
from style import INK, INK_2, INK_MUTED, tint

style.apply()
HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/spectrum")
L, R = 1716, 16.0
LAMS = (11, 12, 13)
INV = (6, -1, 7)                         # j_m = INV_m * n mod lambda_m (CRT harmonics)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = {                               # label: (name, colour, line width)
    "ideal": ("ideal (512 waves)", BLUE, 1.2),
    "att42": ("trained att0.5", ORANGE, 1.2),
    "att43": ("trained att0.5, seed 43", tint(ORANGE, 0.45), 0.8),
    "gwn": ("grid wave network", AQUA, 1.0),
}
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", tint(BLUE, 0.6), BLUE, "#0d2f5c"])

n = np.arange(L); n = np.where(n > L // 2, n - L, n)            # signed index, n or D
N1, N2 = np.meshgrid(n, n, indexing="ij")
RAD = np.hypot(N1, N2)
RBIN = np.rint(RAD).astype(int)


def load(label):
    p = DATA / f"S_{label}.npy"
    return np.load(p) if p.exists() else None


def target(r=R):
    s = L / (2 * np.pi * r)
    p1 = ndtr((n + 0.5) / s) - ndtr((n - 0.5) / s)
    P = np.outer(p1, p1)
    return P / P.sum()


def kernel(S):
    return np.real(np.fft.ifft2(S)) * L * L


def ring(values, how, rmax):
    m = RBIN <= rmax
    b, v = RBIN[m], values[m]
    if how == "mean":
        return np.bincount(b, v, rmax + 1) / np.maximum(np.bincount(b, None, rmax + 1), 1)
    out = np.full(rmax + 1, -np.inf)
    np.maximum.at(out, b, v)
    return out


def order_axis(k):
    o = np.zeros_like(k)
    for lam, inv in zip(LAMS, INV):
        j = (inv * k) % lam
        j = np.where(j > lam // 2, lam - j, j)                  # |signed harmonic|
        o = np.maximum(o, j)
    return o


ORDER = np.maximum(order_axis(N1), order_axis(N2))              # highest harmonic used


def half_width(k):
    row = k[0, :200]
    return float(np.interp(0.5, row[::-1], np.arange(200)[::-1]))


def summary(label, S):
    k = kernel(S)
    nz = S.copy(); nz[0, 0] = 0
    ti = np.load(DATA / f"ti_{label}.npz")
    shifts = [tuple(s) for s in ti["shifts"]]
    sd = {f"{a},{b}": float(ti["dots"][i].std()) for i, (a, b) in enumerate(shifts)}
    rng = {f"{a},{b}": [float(np.percentile(ti["dots"][i], 1)), float(np.percentile(ti["dots"][i], 99))]
           for i, (a, b) in enumerate(shifts)}
    share = np.bincount(ORDER.ravel(), nz.ravel(), 7) / nz.sum()
    return dict(
        constant_share=float(S[0, 0]),
        effective_waves=float(nz.sum() ** 2 / (nz ** 2).sum()),
        half_width=half_width(k),
        avg_kernel_max_beyond_3r=float(k[RAD > 3 * R].max()),
        avg_kernel_argmax_beyond_3r=[int(v) for v in np.unravel_index(
            np.argmax(np.where(RAD > 3 * R, k, -9)), k.shape)],
        power_share_by_order=[round(float(x), 4) for x in share],
        power_share_within_r2=float(nz[RAD <= 2 * L / (2 * np.pi * R)].sum() / nz.sum()),
        translation_sd=sd, translation_p1_p99=rng,
        unit_purity_median=float(np.median(ti["purity"])),
        unit_purity_p10_p90=[float(np.percentile(ti["purity"], 10)),
                             float(np.percentile(ti["purity"], 90))],
    )


def plain_ticks(axis, ticks, fmt="{:g}"):
    axis.set_major_locator(FixedLocator(ticks)); axis.set_minor_locator(NullLocator())
    axis.set_ticklabels([fmt.format(t) for t in ticks])


def lg(v, floor=1e-12):
    """log10, with rounding-level values (no power there) left out."""
    v = np.asarray(v, float)
    return np.where(v > floor, np.log10(np.maximum(v, floor)), np.nan)


def sq_axis(ax, ticks):
    ax.set_xscale("function", functions=(lambda x: np.sign(x) * x ** 2,
                                         lambda x: np.sqrt(np.abs(x))))
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=7, handlelength=1.6, borderaxespad=0.2, **kw)


def fig_main(S, P):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.84, wspace=0.42)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), label="ideal, expected (Gaussian, r = 16)"),
               Line2D([], [], color=BLUE, lw=1.2, label="ideal, one draw of 512 waves"),
               Line2D([], [], color=ORANGE, lw=1.2, label="trained att0.5"),
               Line2D([], [], color=tint(ORANGE, 0.45), lw=1.0, label="seed 43"),
               Line2D([], [], color=AQUA, lw=1.2, label="grid wave network"),
               Line2D([], [], color=INK_MUTED, lw=3, label="grid code input")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=6, frameon=False,
               fontsize=7, handlelength=1.8, columnspacing=1.2)

    # a: kernel against distance, ring mean (solid) and ring max (thin)
    ax = axs[0]
    RM = 858
    d = np.arange(RM + 1)
    ax.plot(d, np.exp(-d ** 2 / (2 * R * R)), color=INK, lw=0.9, ls=(0, (3, 2)), label="Gaussian, r = 16")
    for lab in ("ideal", "att42", "gwn"):
        if S.get(lab) is None:
            continue
        name, col, lw = SERIES[lab]
        k = kernel(S[lab])
        ax.plot(d, ring(k, "max", RM), color=col, lw=0.6, alpha=0.9)
        ax.plot(d, ring(k, "mean", RM), color=col, lw=lw, label=name)
    ax.set_yscale("log"); ax.set_ylim(3e-3, 1.1); ax.set_xlim(0, RM)
    plain_ticks(ax.yaxis, [0.01, 0.1, 1])
    ax.set_xscale("function", functions=(lambda x: np.sqrt(np.abs(x)), lambda x: x ** 2))
    ax.xaxis.set_major_locator(FixedLocator([0, 16, 48, 143, 400, 792]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("distance from the goal (cells)", fontsize=8)
    ax.set_ylabel("average similarity", fontsize=8)

    # b: power per wave, averaged over rings of |n|
    ax = axs[1]
    NM = 120
    nn = np.arange(NM + 1)
    t = ring(P, "mean", NM)
    ax.plot(nn, lg(t), color=INK, lw=0.9, ls=(0, (3, 2)), label="ideal, expected")
    for lab in ("ideal", "att42", "att43", "gwn"):
        if S.get(lab) is None:
            continue
        name, col, lw = SERIES[lab]
        v = ring(S[lab], "mean", NM)
        v[0] = np.nan                                          # constant wave shown in text
        ax.plot(nn, lg(v), color=col, lw=lw if lab != "ideal" else 0.8,
                label=name if lab != "ideal" else "ideal, one draw")
    ax.set_ylim(-9, -2.5)
    sq_axis(ax, [0, 40, 60, 80, 100, 120]); ax.set_xlim(0, NM)
    ax.set_xlabel("frequency |n| (cycles across the arena)", fontsize=8)
    ax.set_ylabel("power per wave (log10)", fontsize=8)

    # c: share of power (x component) by module 13's harmonic |j13|
    ax = axs[2]
    labs = [l for l in ("target", "att42", "gwn", "input") if l == "target" or S.get(l) is not None]
    cols = {"target": INK, "input": INK_MUTED}
    names = {"target": "ideal, expected", "input": "grid code input"}
    j13 = (7 * n) % 13
    j13 = np.minimum(j13, 13 - j13)
    w = 0.8 / len(labs)
    at792 = {}
    for i, lab in enumerate(labs):
        A = P if lab == "target" else S[lab]
        Sx = A.sum(1)
        share = np.bincount(j13, Sx, 7)
        at792[lab] = float((Sx * np.cos(2 * np.pi * j13 / 13)).sum())
        col = cols.get(lab, SERIES.get(lab, ("", INK_MUTED, 1))[1])
        ax.bar(np.arange(7) + (i - (len(labs) - 1) / 2) * w, share, width=w * 0.9, color=col,
               label=names.get(lab, SERIES.get(lab, (lab,))[0]))
    ax.set_xticks(range(7))
    ax.set_xlabel("module 13 harmonic |j| of the wave", fontsize=8)
    ax.set_ylabel("share of power", fontsize=8)
    ax.text(0.98, 0.97, "similarity 792 cells away\n" + "\n".join(
        f"{names.get(l, SERIES.get(l, (l,))[0])}  {at792[l]:.2f}" for l in labs),
        transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=INK_2, linespacing=1.3)

    for ax, letter in zip(axs, "abc"):
        ax.tick_params(labelsize=7)
        ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
    style.save(fig, "spectrum_main", pad=0.03)


def fig_maps(S, P):
    labs = [l for l in ("target", "ideal", "att42", "gwn") if l == "target" or S.get(l) is not None]
    fig = plt.figure(figsize=(7.0, 3.6))
    W = 80
    sl = np.r_[L - W:L, 0:W + 1]
    vmax = 3e-3
    for i, lab in enumerate(labs):
        ax = fig.add_axes([0.02 + i * 0.245, 0.47, 0.21, 0.48])
        A = (P if lab == "target" else S[lab]).copy(); A[0, 0] = 0
        im = ax.imshow(np.log10(np.maximum(A[np.ix_(sl, sl)].T, 1e-12)), origin="lower", cmap=SEQ,
                       vmin=-8, vmax=-2.5, extent=[-W - .5, W + .5, -W - .5, W + .5],
                       interpolation="nearest")
        ax.set_title({"target": "ideal, expected"}.get(lab, SERIES.get(lab, (lab,))[0]), fontsize=8)
        ax.set_xticks([-80, 0, 80]); ax.set_yticks([-80, 0, 80]); ax.tick_params(labelsize=6.5)
        if i == 0:
            ax.set_ylabel("n2", fontsize=7.5, style="italic")
        ax.set_xlabel("n1", fontsize=7.5, style="italic", labelpad=1)
    cax = fig.add_axes([0.985, 0.5, 0.008, 0.42])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
    plain_ticks(cb.ax.yaxis, [-8, -6, -4]); cb.set_label("log10 power", fontsize=6.5)

    ax = fig.add_axes([0.07, 0.08, 0.9, 0.27])
    NM = 120
    k1 = np.arange(1, NM + 1)
    tm = P.sum(1)
    ax.plot(k1, lg(tm[k1] + tm[-k1]), color=INK, lw=0.9, ls=(0, (3, 2)), label="ideal, expected")
    for lab, dx in (("att42", 0.0), ("ideal", 0.25)):
        if S.get(lab) is None:
            continue
        m = S[lab].sum(1)
        v = m[k1] + m[-k1]
        ax.vlines(k1 + dx, -9, lg(v), color=SERIES[lab][1], lw=0.9,
                  label=SERIES[lab][0])
    for lam, inv, y in ((13, 7, 0.1), (11, 6, -0.6)):
        jj = (inv * k1) % lam
        low = np.minimum(jj, lam - jj) <= 1
        ax.plot(k1[low], np.full(low.sum(), y), "v", ms=2.4, color=INK_2, mew=0)
        ax.text(NM + 2, y, f"module {lam}, |j| = 0 or 1", fontsize=6.5, va="center", color=INK_2,
                clip_on=False)
    ax.set_ylim(-7, 0.4); ax.set_xlim(0, NM + 1)
    ax.set_xlabel("n1 (cycles across the arena along x, summed over n2)", fontsize=7.5)
    ax.set_ylabel("log10 power", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    legend(ax, loc="lower right", ncol=3)
    style.save(fig, "spectrum_maps", pad=0.03)


def fig_kernel_maps(S):
    labs = [l for l in ("ideal", "att42", "gwn") if S.get(l) is not None]
    fig = plt.figure(figsize=(7.0, 4.2))
    W = 64
    sl = np.r_[L - W:L, 0:W + 1]
    for i, lab in enumerate(labs):
        k = kernel(S[lab])
        ax = fig.add_axes([0.06 + i * 0.31, 0.55, 0.26, 0.38])
        im1 = ax.imshow(k[np.ix_(sl, sl)].T, origin="lower", cmap=SEQ, vmin=0, vmax=1,
                        extent=[-W - .5, W + .5, -W - .5, W + .5], interpolation="nearest")
        ax.set_title(SERIES[lab][0], fontsize=8)
        ax.set_xticks([-64, 0, 64]); ax.set_yticks([-64, 0, 64]); ax.tick_params(labelsize=6.5)
        if i == 0:
            ax.set_ylabel("near the goal (cells)", fontsize=7.5)
        ax = fig.add_axes([0.06 + i * 0.31, 0.06, 0.26, 0.38])
        im2 = ax.imshow(np.fft.fftshift(np.maximum(k, 1e-3)).T, origin="lower", cmap=SEQ,
                        norm=LogNorm(0.03, 1), extent=[-L / 2, L / 2, -L / 2, L / 2],
                        interpolation="antialiased")
        ax.set_xticks([-792, 0, 792]); ax.set_yticks([-792, 0, 792]); ax.tick_params(labelsize=6.5)
        if i == 0:
            ax.set_ylabel("whole arena, log colour (cells)", fontsize=7.5)
    for im, y in ((im1, 0.57), (im2, 0.08)):
        cax = fig.add_axes([0.975, y, 0.008, 0.34])
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
        plain_ticks(cb.ax.yaxis, [0, 0.5, 1] if im is im1 else [0.03, 0.1, 0.3, 1])
    style.save(fig, "spectrum_kernel_maps", pad=0.03)


def fig_translation():
    show = [(5, 0), (19, 0), (50, 0), (792, 0)]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.8))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.25, top=0.85, wspace=0.3)
    data = {}
    for lab in ("ideal", "att42", "gwn"):
        p = DATA / f"ti_{lab}.npz"
        if p.exists():
            data[lab] = np.load(p)
    for ax, sh in zip(axs, show):
        for lab, ti in data.items():
            i = [tuple(s) for s in ti["shifts"]].index(sh)
            v = ti["dots"][i]
            name, col, _ = SERIES[lab]
            if v.std() < 1e-4:
                ax.axvline(v.mean(), color=col, lw=1.2, label=name)
            else:
                ax.hist(v, bins=60, color=col, alpha=0.55, label=name, density=True)
        ax.set_title(f"{sh[0]} cells apart", fontsize=8)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("similarity", fontsize=7.5)
    legend(axs[0], loc="upper left")
    style.save(fig, "spectrum_translation", pad=0.03)


def main():
    P = target()
    S = {lab: load(lab) for lab in ("ideal", "att42", "att43", "att42u", "att43u", "gwn", "input")}
    out = {"target": {"effective_waves": float((P.sum() - P[0, 0]) ** 2 / ((P ** 2).sum() - P[0, 0] ** 2)),
                      "power_share_by_order": [round(float(x), 4) for x in
                                               np.bincount(ORDER.ravel(), P.ravel(), 7) / P.sum()]}}
    for lab, A in S.items():
        if A is not None:
            out[lab] = summary(lab, A)
            print(lab, json.dumps(out[lab]), flush=True)
    (HERE / "spectrum_summary.json").write_text(json.dumps(out, indent=1))
    fig_main(S, P)
    fig_maps(S, P)
    fig_kernel_maps(S)
    fig_translation()
    print("saved")


if __name__ == "__main__":
    main()
