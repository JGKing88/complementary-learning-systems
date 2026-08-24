"""Plots for P1 — where, and how often, `q` fails to point at the goal.

Reads the `.npz` files written by `q_failure_map.py` and renders six figures.
Nothing is recomputed here that the compute step already printed, so a figure
and the summary it illustrates cannot disagree.

    python -m analysis.nav_p2.q_failure_plots \
        --npz results/nav_p2/qmap_seed*.npz --outdir results/nav_p2/figs

Encoding decisions worth stating, because they are the difference between a
figure that answers the question and one that decorates it:

  * **Envs are samples, not identities.** 64 of them get one muted hairline
    each plus an emphasized median -- not 64 hues. The spread IS the finding
    (phase 1 drew one world and generalized from it), so it has to be drawn,
    but no single env deserves a legend entry.
  * **Seeds are replicates.** Shown as separate points on the same series
    rather than as their own colors; three lines that agree say "reproducible",
    three colors say "three different things".
  * **Lock outcome is plotted as failure, not as share.** P(lock=goal) is 0.99
    at every distractor count, so a stacked bar would be one full bar eleven
    times over. The rare part is the whole story, so the rare part gets the
    axis and the 0.99 is stated as text.
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

LOCK_GOAL, LOCK_DISTRACTOR, LOCK_MIXTURE = 0, 1, 2

# Slots 1-3 of the reference categorical palette. `references/palette.md`
# records that these three validate on the all-pairs list in both modes
# (worst pair CVD dE 9.2 light, normal-vision 24.0), which is the list that
# governs scatter and small multiples -- so three is also the cap here.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e6e5e0"

# Sequential: one hue, light to dark. Never a rainbow.
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f2f6fc", C1, "#123a6b"])


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "medium",
        "axes.labelsize": 9,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.dpi": 160,
    })


def _clean(ax, ygrid=True):
    """Hairline, solid, recessive -- and only where a reader needs it."""
    ax.set_axisbelow(True)
    ax.grid(ygrid, axis="y")
    ax.grid(False, axis="x")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load(paths):
    """Every seed, plus the distractor axis they must all agree on."""
    ds = [np.load(p) for p in sorted(paths)]
    nd = ds[0]["n_distractors"]
    for d in ds[1:]:
        assert np.array_equal(d["n_distractors"], nd), "seeds differ in axis"
    return ds, nd


# --------------------------------------------------------------------------
# 1. How often does the recall lock onto the wrong thing?
# --------------------------------------------------------------------------
def fig_lock(ds, nd, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    dist = np.array([[float((d["lock"][i] == LOCK_DISTRACTOR).mean())
                      for i in range(len(nd))] for d in ds])
    mix = np.array([[float((d["lock"][i] == LOCK_MIXTURE).mean())
                     for i in range(len(nd))] for d in ds])

    for arr, c, label in ((dist, C1, "locked on a distractor"),
                          (mix, C2, "spurious mixture (neither)")):
        ax.plot(nd, arr.mean(0) * 100, color=c, lw=2, zorder=3, label=label)
        for row in arr:                       # seeds as replicate points
            ax.plot(nd, row * 100, "o", color=c, ms=4.5, alpha=0.55,
                    mec=SURFACE, mew=1.2, zorder=2)

    _clean(ax)
    ax.set_xlabel("distractors stored alongside the goal")
    ax.set_ylabel("% of arena cells")
    ax.set_xticks(nd)
    ax.set_ylim(bottom=0)
    ax.set_title("The recall almost never locks onto a distractor")
    ax.legend(loc="upper left")
    goal_rate = np.mean([(d["lock"] == LOCK_GOAL).mean() for d in ds]) * 100
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    # Below the axes, not inside them: at these rates the lines run through
    # the middle of the panel and any in-plot caption lands on the data.
    fig.text(0.012, 0.055,
             f"The remaining {goal_rate:.1f}% of cells lock onto the goal.",
             fontsize=8.5, color=INK3, ha="left")
    fig.text(0.012, 0.018,
             "Points are the 3 scaffold seeds; line is their mean. "
             "64 envs x 8 draws x 400 cells per seed.",
             fontsize=8.5, color=INK3, ha="left")
    fig.savefig(os.path.join(out, "p1_lock_outcome.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. The finding-19 plot: between-world spread
# --------------------------------------------------------------------------
def fig_spread(ds, nd, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    d = ds[0]
    dc = d["dir_cos"]                                   # (lvl, env, draw, cell)
    per_env = np.stack([[float(np.nanmean(dc[i, e] < 0.5))
                         for i in range(len(nd))]
                        for e in range(dc.shape[1])])   # (env, lvl)

    for row in per_env:                                 # one hairline per env
        ax.plot(nd, row * 100, color=C1, lw=0.6, alpha=0.18, zorder=1)
    med = np.median(per_env, axis=0) * 100
    ax.plot(nd, med, color=C1, lw=2.4, zorder=4, label="median env")
    ax.fill_between(nd, np.percentile(per_env, 10, axis=0) * 100,
                    np.percentile(per_env, 90, axis=0) * 100,
                    color=C1, alpha=0.13, lw=0, zorder=2,
                    label="10th-90th percentile of envs")

    for y, txt in ((1.8, "phase 1, world A: 1.8%"),
                   (23.3, "phase 1, world B: 23.3%")):
        ax.axhline(y, color=C2, lw=1.4, zorder=3)
        ax.text(nd[-1], y, "  " + txt, color=C2, fontsize=8.5,
                va="center", ha="left")

    _clean(ax)
    ax.set_xlabel("distractors stored alongside the goal")
    ax.set_ylabel("% of cells with cos(q, goal dir) < 0.5")
    ax.set_xticks(nd)
    ax.set_xlim(nd[0], nd[-1] + 3.4)
    ax.set_ylim(bottom=0)
    ax.set_title("Readout quality varies more between worlds than between\n"
                 "distractor counts")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "p1_between_world_spread.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. The decomposition: direction error given which attractor won
# --------------------------------------------------------------------------
def fig_decomp(ds, nd, out):
    """Cumulative, not a histogram.

    The first version binned `dir_cos` and was unreadable: 80% of the
    goal-locked mass falls in the single bin at 1.0, so the spike set the
    y-scale and the 1.4% tail that the whole question is about was invisible.
    A CDF puts that tail on the axis, which is also the only way to compare
    groups whose populations differ by two orders of magnitude.
    """
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    i = len(nd) - 1                                      # the hardest level
    dc = np.concatenate([d["dir_cos"][i].ravel() for d in ds])
    lk = np.concatenate([d["lock"][i].ravel() for d in ds])

    grid = np.linspace(-1, 1, 401)
    groups = ((LOCK_GOAL, C1, "locked on the goal"),
              (LOCK_MIXTURE, C2, "spurious mixture (near no stored pattern)"),
              (LOCK_DISTRACTOR, C3, "locked on a distractor"))
    for code, c, label in groups:
        v = dc[(lk == code) & np.isfinite(dc)]
        if v.size == 0:
            continue
        cdf = np.searchsorted(np.sort(v), grid, side="right") / v.size
        ax.plot(grid, cdf * 100, color=c, lw=2,
                label=f"{label}  (n={v.size:,})")

    _clean(ax)
    ax.set_xlabel(f"cos(q, true goal direction)   at {nd[i]} distractors")
    ax.set_ylabel("% of that group's cells at or below x")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 100)
    ax.set_title("Even a wrong lock usually still points roughly at the goal")
    ax.legend(loc="upper left")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    # The surprise worth stating: the two failure groups are NOT randomly
    # directed. A mixture sits near the goal pattern without reaching the 0.9
    # threshold, so it still carries most of the direction -- which is why the
    # catastrophic tail stays small even where the lock is technically wrong.
    frac_bad = float(np.nanmean(dc < 0.5)) * 100
    ax.text(-0.97, 62,
            "a randomly-directed q would trace the diagonal;\n"
            "all three groups hug the right edge instead",
            fontsize=8.5, color=INK3, va="top")
    fig.text(0.012, 0.055,
             f"Across all cells at {nd[i]} distractors, {frac_bad:.1f}% fall "
             f"below cos 0.5 -- the catastrophic tail.",
             fontsize=8.5, color=INK3, ha="left")
    fig.text(0.012, 0.018,
             "Groups are normalized separately; their populations differ ~100x.",
             fontsize=8.5, color=INK3, ha="left")
    fig.savefig(os.path.join(out, "p1_decomposition.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Where in the arena does it break?
# --------------------------------------------------------------------------
def fig_arena(ds, nd, out):
    d = ds[0]
    i = len(nd) - 1
    dc, cells = d["dir_cos"][i], d["cells"]
    size = int(d["size"])
    bad = np.array([float(np.nanmean(dc[e] < 0.5)) for e in range(dc.shape[1])])
    picks = [("best env", int(bad.argmin())),
             ("median env", int(np.argsort(bad)[len(bad) // 2])),
             ("worst env", int(bad.argmax()))]

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2))
    for ax, (name, e) in zip(axes, picks):
        m = np.full((size, size), np.nan)
        vals = np.nanmean(dc[e], axis=0)                  # over draws
        m[cells[:, 0], cells[:, 1]] = vals
        # 98.6% of cells sit above cos 0.5, so a -1..1 ramp spends half its
        # range on empty space and renders every panel a flat dark block.
        im = ax.imshow(m.T, origin="lower", cmap=SEQ, vmin=0.5, vmax=1.0,
                       interpolation="nearest")
        g = d["goals"][e]
        ax.plot(g[0], g[1], marker="*", ms=13, color=C2, mec=SURFACE, mew=1.2,
                zorder=3)
        ax.set_title(f"{name}\n{bad[e] * 100:.1f}% of cells below cos 0.5",
                     fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02, extend="min")
    cb.set_label("cos(q, goal direction)", color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)
    fig.suptitle(f"Where q fails, at {nd[i]} distractors — orange star is the goal",
                 fontsize=10.5, color=INK, y=1.0)
    fig.savefig(os.path.join(out, "p1_arena_maps.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Does it break near the goal, or near a wall?
# --------------------------------------------------------------------------
def fig_geometry(ds, nd, out):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    i = len(nd) - 1
    for ax, key, xlabel in (
            (axes[0], "goal_dist", "distance from the goal (cells)"),
            (axes[1], "wall_dist", "distance from the nearest wall (cells)")):
        xs, ys = [], []
        for d in ds:
            g = np.repeat(d[key][None, :, None, :], 1, 0)[0]   # (env, 1, cell)
            g = np.broadcast_to(d[key][:, None, :], d["dir_cos"][i].shape)
            xs.append(g.ravel()); ys.append(d["dir_cos"][i].ravel())
        x = np.concatenate(xs); y = np.concatenate(ys)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        edges = np.arange(0, np.ceil(x.max()) + 1.0, 1.0)
        idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
        ctr, med, lo, hi = [], [], [], []
        for b in range(len(edges) - 1):
            v = y[idx == b]
            if v.size < 50:
                continue
            ctr.append((edges[b] + edges[b + 1]) / 2)
            med.append(np.median(v))
            lo.append(np.percentile(v, 10)); hi.append(np.percentile(v, 90))
        ax.fill_between(ctr, lo, hi, color=C1, alpha=0.15, lw=0)
        ax.plot(ctr, med, color=C1, lw=2)
        _clean(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.28, 1.02)
    axes[0].set_ylabel(f"cos(q, goal dir), {nd[i]} distractors")
    axes[0].set_title("median, with 10th-90th percentile band", loc="left",
                      fontsize=9, color=INK2)
    fig.suptitle("The readout degrades close to the goal, not close to a wall",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "p1_geometry.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. |q| separability — the explore side of the same measurement
# --------------------------------------------------------------------------
def fig_qnorm(ds, nd, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key, c, label in (("qnorm_goal", C1, "goal stored (exploit memory)"),
                          ("qnorm_absent", C2, "goal absent (explore memory)")):
        # The goal-absent memory at zero distractors is EMPTY, so its |q| is
        # zero by construction rather than by any failure to separate. Plotting
        # it would read as "perfect separability at d=0", which is backwards.
        keep = np.arange(len(nd)) if key == "qnorm_goal" else np.arange(1, len(nd))
        x = nd[keep]
        stat = lambda f: np.array(
            [f(np.concatenate([d[key][i].ravel() for d in ds])) for i in keep])
        med, lo, hi = (stat(np.median),
                       stat(lambda v: np.percentile(v, 25)),
                       stat(lambda v: np.percentile(v, 75)))
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(x, med, color=c, lw=2, label=label)

    _clean(ax)
    ax.set_xlabel("distractors stored alongside the goal")
    ax.set_ylabel("median |q|   (IQR band)")
    ax.set_xticks(nd)
    ax.set_ylim(bottom=0)
    ax.set_title("Magnitude still separates the two regimes at ten distractors")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "p1_qnorm_separability.png"))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", nargs="+", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    paths = [q for pat in args.npz for q in glob.glob(pat)]
    if not paths:
        raise SystemExit(f"no .npz matched {args.npz}")
    os.makedirs(args.outdir, exist_ok=True)
    _style()
    ds, nd = load(paths)
    print(f"{len(ds)} seed(s): {[os.path.basename(q) for q in sorted(paths)]}")
    print(f"envs/seed {ds[0]['dir_cos'].shape[1]}  draws {ds[0]['dir_cos'].shape[2]}"
          f"  cells {ds[0]['dir_cos'].shape[3]}  levels {len(nd)}")

    for fn in (fig_lock, fig_spread, fig_decomp, fig_arena, fig_geometry,
               fig_qnorm):
        fn(ds, nd, args.outdir)
        print(f"  {fn.__name__}")
    print(f"\nwrote {len(os.listdir(args.outdir))} files to {args.outdir}")


if __name__ == "__main__":
    main()
