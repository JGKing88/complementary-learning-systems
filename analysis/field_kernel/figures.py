"""Figure for docs/EXPERIMENTS_FIELD_KERNEL.md.

    python -m analysis.field_kernel.figures      # -> $FIG/field_kernel/tradeoff.pdf and .png

a  Pearson kernels: baseline, two-scale codes, and a single-scale code with mouse-sized fields.
b  Field width against the fast share, for each kind of fast input (1024 units).
c  Direction error of the fast test (noise only, 2-cell centred readout) against unit count.
d  The real probe: reach against field width for the readouts and unit counts that were run.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from analysis.field_kernel import law
from analysis.field_kernel import summary as SM

plt.rcParams.update({"pdf.use14corefonts": True, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.default": "regular", "axes.linewidth": 0.6})
RES = "/orcd/pool/003/jackking/cls_runs/results/field_kernel"
FIG = "/orcd/pool/003/jackking/cls_runs/figures/field_kernel"
LAM = 12.0
C = {"base": "#222222", "a": "#1f77b4", "b": "#d95f02", "single": "#999999", "grid": "#7b3294"}


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 2.7))

    # a: kernels
    a = json.load(open(f"{RES}/measure/set_a.json"))
    c = json.load(open(f"{RES}/measure/extra_c.json"))
    by = {m["spec"]: m for m in a + c}

    def prof(key):
        m = next(v for k, v in by.items() if key(k))
        return np.array(m["profile"]["pearson"][:81]), m["loop"]["W_A"]

    rows = [("one scale, r 23.2", lambda k: k.endswith("active=0.1") and "r=23.2" in k, C["base"], "-"),
            ("two scales, 30% fast", lambda k: "fast=0.3,r_fast=3" in k and "r=23.2" in k, C["a"], "-"),
            ("two scales, 50% fast", lambda k: "fast=0.5,r_fast=3" in k and "r=23.2" in k, C["b"], "-"),
            ("one scale, r 4.2", lambda k: "r=4.2" in k, C["single"], "--")]
    for name, key, col, ls in rows:
        p, W = prof(key)
        ax[0].plot(np.arange(81), p, color=col, ls=ls, lw=1.2, label=f"{name} (field {W:.0f})")
    ax[0].set_xlabel("separation (cells)")
    ax[0].set_ylabel("population similarity (Pearson)")
    ax[0].legend(frameon=False, fontsize=6.5)
    ax[0].set_title("a  kernel", loc="left", fontsize=8)

    # b: field width against fast share
    R = [r for r in law.rows() if r["D"] == 1024]
    base = next(r for r in R if r["alpha"] == 0)
    for kind, rs, col, lab in (("gauss", 1.0, "#9ecae1", "fast scale 1"), ("gauss", 2.0, "#4292c6", "fast scale 2"),
                               ("gauss", 3.0, "#08519c", "fast scale 3"), ("grid", 0.0, C["grid"], "direct grid input")):
        rr = sorted([r for r in R if r["kind"] == kind and r["r_fast"] == rs and r["alpha"] > 0],
                    key=lambda r: r["alpha"])
        ax[1].plot([0] + [r["alpha"] for r in rr], [base["W_A"] / LAM] + [r["W_A"] / LAM for r in rr],
                   "o-", ms=3, lw=1, color=col, label=lab)
    ax[1].axhspan(0.4, 0.5, color="#fdd49e", lw=0)
    ax[1].text(0.01, 0.44, "mouse", ha="left", fontsize=6.5)
    ax[1].set_yscale("log")
    ax[1].set_yticks([0.4, 0.5, 1, 2, 2.5])
    ax[1].set_yticklabels(["0.4", "0.5", "1", "2", "2.5"])
    ax[1].yaxis.set_minor_formatter(NullFormatter())
    ax[1].set_xlabel("share of input variance that is fast")
    ax[1].set_ylabel("field width / grid period")
    ax[1].legend(frameon=False, fontsize=6.5)
    ax[1].set_title("b  field width", loc="left", fontsize=8)

    # c: error against units
    R = [r for r in law.rows(kind="gauss") if r["r_fast"] == 3.0 or r["alpha"] == 0]
    for al, col in ((0.0, C["base"]), (0.1, "#9ecae1"), (0.3, C["a"]), (0.5, C["b"])):
        rr = sorted([r for r in R if r["alpha"] == al], key=lambda r: r["D"])
        if rr:
            ax[2].plot([r["D"] for r in rr], [r["err2"] for r in rr], "o-", ms=3, lw=1, color=col,
                       label=f"{int(al * 100)}% fast" if al else "one scale")
    d = np.array([1024, 16384])
    ax[2].plot(d, 40 * np.sqrt(1024 / d), color="#bbbbbb", lw=0.8, ls=":")
    ax[2].text(16384, 40 * 0.25 * 1.1, "1/sqrt(units)", ha="right", fontsize=6.5, color="#888888")
    ax[2].set_xscale("log", base=2)
    ax[2].set_yscale("log")
    ax[2].set_xticks([1024, 4096, 16384])
    ax[2].set_xticklabels(["1k", "4k", "16k"])
    ax[2].set_yticks([2, 5, 10, 20, 40])
    ax[2].set_yticklabels(["2", "5", "10", "20", "40"])
    ax[2].yaxis.set_minor_formatter(NullFormatter())
    ax[2].set_xlabel("units")
    ax[2].set_ylabel("direction error (deg), fast test")
    ax[2].legend(frameon=False, fontsize=6.5)
    ax[2].set_title("c  readout noise", loc="left", fontsize=8)

    # d: the real probe
    W = {"base": 30.1, "gauss 0.05": 14.0, "gauss 0.1": 10.3, "rf3 0.3": 8.0, "rf3 0.5": 6.2,
         "rf2 0.5": 5.1, "single r4.2": 6.0, "gauss fast 0.05": 14.0, "gauss fast 0.1": 10.3,
         "gauss fast 0.3": 6.5, "gauss 0.3": 6.5, "base p10 r23.2": 30.1}
    groups = {"forward 1 cell, 1k": ("#bbbbbb", "s"), "centred 1 cell, 1k": ("#9ecae1", "o"),
              "centred 1 cell, 4k": ("#4292c6", "o"), "centred 4 cells, 1k": ("#fdae6b", "^"),
              "centred 4 cells, 4k": ("#d95f02", "^"), "centred 4 cells, 8k": ("#7f2704", "^")}
    pts = {g: [] for g in groups}
    for s in "abcd":
        for d_ in SM.load(s):
            if "probe" not in d_ or "grid" in d_["label"] or "dense" in d_["label"] or "r16" in d_["label"] \
                    or "r32" in d_["label"] or "single" in d_["label"]:
                continue
            lab = d_["label"]
            name = next((k for k in sorted(W, key=len, reverse=True) if lab.startswith(k)), None)
            if name is None:
                continue
            units = "8k" if "D8192" in lab else "4k" if "D4096" in lab else "1k"
            if " c1" in lab:
                g = f"centred 1 cell, {units}"
            elif "step 4" in lab:
                g = f"centred 4 cells, {units}"
            elif "step" in lab or " c2" in lab:
                continue
            else:
                g = f"forward 1 cell, {units}"
            if g in pts:
                pts[g].append((W[name] / LAM, d_["probe"]["cont"]))
    for g, (col, mk) in groups.items():
        if pts[g]:
            x, y = zip(*sorted(pts[g]))
            ax[3].plot(x, y, mk + "-", ms=4, lw=0.8, color=col, label=g)
    ax[3].axvspan(0.4, 0.5, color="#fdd49e", lw=0)
    ax[3].set_xscale("log")
    ax[3].set_xticks([0.5, 1, 2.5])
    ax[3].set_xticklabels(["0.5", "1", "2.5"])
    ax[3].xaxis.set_minor_formatter(NullFormatter())
    ax[3].set_ylim(0, 1.03)
    ax[3].set_xlabel("field width / grid period")
    ax[3].set_ylabel("goals reached (probe)")
    ax[3].legend(frameon=False, fontsize=6)
    ax[3].set_title("d  navigation probe", loc="left", fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/tradeoff.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.03)
    print(f"{FIG}/tradeoff.pdf")


if __name__ == "__main__":
    main()
