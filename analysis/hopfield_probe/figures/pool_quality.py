"""Figure: pool size vs largest alias for the depression learner (empirical).

    python analysis/hopfield_probe/figures/pool_quality.py     # reads pool_quality.json

Largest alias = largest similarity to the goal beyond 3r over the whole 1716 x 1716 arena,
for r = 8, 16, 32 (mean and range over 3 seeds). Dotted: the target Gaussian's similarity
at distance r and 2r from the goal (e^-1/2, e^-2); the networks match these on average.
Writes pool_quality.pdf (core Helvetica/Symbol) and pool_quality.png here.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

import style
from style import INK, INK_2, INK_MUTED

style.apply()
HERE = Path(__file__).resolve().parent
COLS = {8.0: "#2a78d6", 16.0: "#eb6834", 32.0: "#1baf7a"}
TRAINED = (400_000, 0.195)          # trained grid wave network (r = 16): pool, largest alias
REFS = [(np.exp(-0.5), ""), (np.exp(-2.0), "2")]

recs = json.loads((HERE / "pool_quality.json").read_text())
fig, ax = plt.subplots(figsize=(3.6, 2.7))
fig.subplots_adjust(left=0.16, right=0.86, bottom=0.17, top=0.95)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(20_000, 9_000_000)
ax.set_ylim(0.012, 1.0)
YT = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
ax.yaxis.set_major_locator(FixedLocator(YT)); ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels([f"{t:g}" for t in YT])
XT = [25_000, 100_000, 400_000, 1_600_000, 6_400_000]
ax.xaxis.set_major_locator(FixedLocator(XT)); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(["25k", "100k", "400k", "1.6M", "6.4M"])

for y, mult in REFS:
    ax.axhline(y, color=INK_MUTED, lw=0.7, ls=(0, (1, 2)), zorder=0)
    t = style.Rich(1.02, y, [("at ", ""), (mult, ""), ("r", "i")], ax.get_yaxis_transform(),
                   ha="left", size=7, color=INK_2)
    t.set_clip_on(False)
    ax.add_artist(t)

for r, col in COLS.items():
    pools = sorted({x["pool"] for x in recs if x["r"] == r})
    a = np.array([[x["alias"] for x in recs if x["r"] == r and x["pool"] == p] for p in pools])
    ax.vlines(pools, a.min(1), a.max(1), color=col, lw=1.0, zorder=2)
    ax.plot(pools, a.mean(1), "o-", color=col, ms=3.0, lw=1.3, zorder=3, label=f"r = {r:.0f} cells")

ax.plot(*TRAINED, "D", mfc="white", mec=INK, mew=0.9, ms=4.0, zorder=4)
ax.annotate("trained network", TRAINED, xytext=(1_300_000, 0.36), fontsize=7, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.6, shrinkA=1, shrinkB=3))
ax.legend(loc="lower left", fontsize=7, frameon=False, handlelength=1.2, borderaxespad=0.3)
ax.set_xlabel("pool size (random units built)", fontsize=8)
ax.set_ylabel("largest alias", fontsize=8)
ax.tick_params(labelsize=7.5)

style.save(fig, "pool_quality", pad=0.03)
print("saved")
