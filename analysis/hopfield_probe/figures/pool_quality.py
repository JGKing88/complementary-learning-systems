"""Figure: pool size vs kernel quality for the depression learner.

    python analysis/hopfield_probe/figures/pool_quality.py     # reads pool_quality.json

a: largest alias (largest similarity beyond 3r, whole 1716 x 1716 arena) against pool size,
   for r = 8, 16, 32; dashed: prediction sqrt(2 ln(L/r) / n_eff) with n_eff = P / (pi r^2).
b: the same points against measured effective units; all widths fall on one curve.
Writes pool_quality.pdf (core Helvetica/Symbol) and pool_quality.png here.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

import style
from style import INK, INK_2, INK_MUTED, rich

style.apply()
HERE = Path(__file__).resolve().parent
L = 1716
COLS = {8.0: "#2a78d6", 16.0: "#eb6834", 32.0: "#1baf7a"}
TRAINED = (400_000, 0.195)          # trained grid wave network: pool, largest alias beyond 3r
IDEAL512 = 0.125                    # ideal encoder with 512 frequencies (0.11-0.14 over draws)

recs = json.loads((HERE / "pool_quality.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55), gridspec_kw=dict(wspace=0.32))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.2, top=0.88)

ALIAS_TICKS = [0.02, 0.05, 0.1, 0.2, 0.5]


def title(ax, runs):
    t = style.Rich(0.0, 1.07, runs, ax.transAxes, ha="left", size=8.5)
    t.set_clip_on(False)
    ax.add_artist(t)


for ax in axes:
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.012, 1.0)
    ax.yaxis.set_major_locator(FixedLocator(ALIAS_TICKS))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{t:g}" for t in ALIAS_TICKS])
    ax.axhline(IDEAL512, color=INK_MUTED, lw=0.7, ls=(0, (1, 2)), zorder=0)

# a: alias vs pool
ax = axes[0]
POOL_TICKS = [25_000, 100_000, 400_000, 1_600_000, 6_400_000]
ax.xaxis.set_major_locator(FixedLocator(POOL_TICKS))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(["25k", "100k", "400k", "1.6M", "6.4M"])
ax.set_xlim(20_000, 14_000_000)
for r, col in COLS.items():
    pools = sorted({x["pool"] for x in recs if x["r"] == r})
    a = np.array([[x["alias"] for x in recs if x["r"] == r and x["pool"] == p] for p in pools])
    ax.vlines(pools, a.min(1), a.max(1), color=col, lw=1.0, zorder=2)
    ax.plot(pools, a.mean(1), "o-", color=col, ms=3.2, lw=1.3, zorder=3, label=f"r = {r:.0f}")
    P = np.geomspace(20_000, 7_500_000, 100)
    ax.plot(P, np.sqrt(2 * np.log(L / r) / (P / (np.pi * r * r))), color=col, lw=0.8, ls=(0, (3, 2)),
            zorder=1)
    ax.text(pools[-1] * 1.45, a.mean(1)[-1], f"r = {r:.0f}", va="center", ha="left", fontsize=7.5,
            color=INK_2)
ax.plot(*TRAINED, "D", mfc="white", mec=INK, mew=0.9, ms=4.2, zorder=4)
ax.annotate("trained network", TRAINED, xytext=(TRAINED[0] * 1.6, 0.36), fontsize=7.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.6, shrinkA=1, shrinkB=3))
ax.set_xlabel("pool size (random units built)")
ax.set_ylabel("largest alias")
title(ax, [("a", "b"), ("  depression learner, uniform random pool", "")])

# b: alias vs effective units
ax = axes[1]
N_TICKS = [30, 100, 300, 1000, 3000, 10000, 30000]
ax.xaxis.set_major_locator(FixedLocator(N_TICKS))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(["30", "100", "300", "1k", "3k", "10k", "30k"])
ax.set_xlim(20, 50_000)
for r, col in COLS.items():
    ne = np.array([x["n_eff"] for x in recs if x["r"] == r])
    al = np.array([x["alias"] for x in recs if x["r"] == r])
    ax.plot(ne, al, "o", color=col, ms=3.0, mew=0, alpha=0.9, zorder=3, label=f"r = {r:.0f}")
    N = np.geomspace(20, 50_000, 100)
    ax.plot(N, np.sqrt(2 * np.log(L / r) / N), color=col, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.set_xlabel("effective units")
title(ax, [("b", "b"), ("  the same points against effective units", "")])
ax.text(1500, IDEAL512 * 1.12, "ideal, 512 frequencies", fontsize=7, color=INK_2, va="bottom")
rich(ax, 600, 0.45, [("dashed: ", ""), ("√", "g"), ("(2 ln(", ""), ("L", "i"), ("/", ""), ("r", "i"),
                     (") / ", ""), ("n", "i"), ("eff", "_"), (")", "")], ha="left", size=7.5, color=INK_2)
rich(ax, 600, 0.3, [("n", "i"), ("eff", "_"), (" = pool / ", ""), ("π", "g"), ("r", "i"), ("2", "^")],
     ha="left", size=7.5, color=INK_2)
leg = ax.legend(loc="lower left", fontsize=7, frameon=False, handlelength=1.0, borderaxespad=0.2)

for ax in axes:
    ax.tick_params(labelsize=7.5)

style.save(fig, "pool_quality", pad=0.03)
print("saved")
