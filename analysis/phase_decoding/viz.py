"""Plot helpers for Exp 1 (bars) and Exp 2 (PCA scatter, trajectory PCA)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .collect_trajectory import TrajectoryDataset


# ---------------------------------------------------------------------------
# Exp 1: bar plots
# ---------------------------------------------------------------------------

_SPLIT_ORDER = ["Within-arena", "LOO", "Random 80/20",
                "Quadrant 1v3", "Quadrant 3v1"]
# One color per metric (uniform across split families within a panel).
_METRIC_COLORS = {"parallelism": "#7EA6D8", "decodability": "#6CC58A"}


def _strip_chrome(ax) -> None:
    """Remove top + right spines and grid lines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _adapt_ylim_parallelism(values: list[float]) -> tuple[float, float]:
    """Adapt parallelism panel y-lim to the data range, clamped to max=1.05.

    Lower bound = min(observed - 0.05, 0) so 0 (chance) is always in view; if
    any value is negative we extend further down.
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (-1.05, 1.05)
    lo = float(min(arr.min() - 0.05, 0.0))
    lo = max(lo, -1.05)
    return (lo, 1.05)


def _summarize(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _save_png_and_pdf(fig: plt.Figure, out_path: str | Path, *, dpi: int = 500) -> None:
    """Save figure in both PNG and PDF formats using the same base name."""
    base = Path(out_path)
    fig.savefig(base.with_suffix(".png"), dpi=dpi)
    fig.savefig(base.with_suffix(".pdf"), dpi=dpi)


def _lighten_colors(colors: np.ndarray, amount: float = 0.35) -> np.ndarray:
    """Blend RGB colors toward white by `amount`."""
    rgb = np.asarray(colors, dtype=float)[..., :3]
    return rgb + (1.0 - rgb) * amount


_METRICS_SPEC = [
    ("parallelism", "Parallelism", 0.0),
    ("decodability", "Decodability", 0.5),
]


def plot_bars(results: dict, out_path: str | Path) -> None:
    """Single-run bars. results = {split_name: [{"parallelism": float, ...}]}.

    Two stacked panels (parallelism, decodability) with one bar per split
    family. Error = SD over folds. All bars in a panel share a color.
    """
    out_path = Path(out_path)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

    names = [n for n in _SPLIT_ORDER if n in results]

    for ax, (key, ylabel, chance) in zip(axes, _METRICS_SPEC):
        means, stds = [], []
        for n in names:
            m, s = _summarize([fold.get(key, float("nan")) for fold in results[n]])
            means.append(m); stds.append(s)
        xs = np.arange(len(names))
        ax.bar(xs, means, yerr=stds, capsize=4, color=_METRIC_COLORS[key],
               edgecolor="black", linewidth=0.6)
        ax.axhline(chance, color="k", lw=0.7, ls="--", alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(names)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        if key == "parallelism":
            # Adapt: drop the empty negative space when no fold went negative.
            data_for_ylim = [m + s for m, s in zip(means, stds)] + \
                            [m - s for m, s in zip(means, stds)] + means
            ax.set_ylim(*_adapt_ylim_parallelism(data_for_ylim))
        else:
            ax.set_ylim(0.0, 1.05)
        _strip_chrome(ax)
        for i, (m, s, n_folds) in enumerate(zip(
                means, stds, [len(results[n]) for n in names])):
            label = f"{m:.2f}±{s:.2f}\n(n={n_folds})"
            ax.text(xs[i], m + (0.04 if m >= 0 else -0.06),
                    label, ha="center", va="bottom" if m >= 0 else "top",
                    fontsize=11)
    fig.tight_layout()
    _save_png_and_pdf(fig, out_path, dpi=500)
    plt.close(fig)


def plot_bars_grouped(
    runs: list[tuple[str, dict]],
    out_path: str | Path,
) -> None:
    """Multi-run grouped bars.

    Args:
        runs: list of (label, results) tuples, where results has the same
            shape as plot_bars's `results` arg.
        out_path: PNG path.
    """
    out_path = Path(out_path)
    fig_w = max(6.6, 1.2 * len(runs) + 4.5)
    fig_h = fig_w + 0.8
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h),
                             sharex=True)

    # Use the union of split families across runs, ordered canonically.
    all_names: list[str] = []
    for _, res in runs:
        for n in res:
            if n not in all_names:
                all_names.append(n)
    names = [n for n in _SPLIT_ORDER if n in all_names] + \
            [n for n in all_names if n not in _SPLIT_ORDER]

    n_runs = len(runs)
    bar_w = 0.8 / max(n_runs, 1)
    # Per-run colors: use a categorical cmap so any number of runs looks ok.
    run_colors = _lighten_colors(plt.get_cmap("tab10")(np.arange(n_runs) % 10),
                                 amount=0.35)

    for ax, (key, ylabel, chance) in zip(axes, _METRICS_SPEC):
        xs_center = np.arange(len(names))
        all_extents: list[float] = []
        for ri, (label, res) in enumerate(runs):
            means, stds, n_folds = [], [], []
            for n in names:
                vals = [fold.get(key, float("nan")) for fold in res.get(n, [])]
                m, s = _summarize(vals)
                means.append(m); stds.append(s)
                n_folds.append(len([v for v in vals if v == v]))
            offset = (ri - (n_runs - 1) / 2) * bar_w
            xs = xs_center + offset
            ax.bar(xs, means, width=bar_w * 0.95, yerr=stds, capsize=2.5,
                   color=run_colors[ri], edgecolor="black", linewidth=0.5,
                   label=label)
            for i, (m, s, k) in enumerate(zip(means, stds, n_folds)):
                if not np.isfinite(m):
                    continue
                ax.text(xs[i], m + (0.03 if m >= 0 else -0.05),
                        f"{m:.2f}\n±{s:.2f}",
                        ha="center", va="bottom" if m >= 0 else "top",
                        fontsize=10)
            all_extents.extend(means)
            all_extents.extend([m + s for m, s in zip(means, stds)])
            all_extents.extend([m - s for m, s in zip(means, stds)])
        ax.axhline(chance, color="k", lw=0.7, ls="--", alpha=0.6)
        ax.set_xticks(xs_center)
        ax.set_xticklabels(names)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        if key == "parallelism":
            ax.set_ylim(*_adapt_ylim_parallelism(all_extents))
        else:
            ax.set_ylim(0.0, 1.05)
        _strip_chrome(ax)
    axes[0].legend(loc="lower right", fontsize=12, ncol=min(n_runs, 3),
                   frameon=False)
    fig.tight_layout()
    _save_png_and_pdf(fig, out_path, dpi=500)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 2 part 1: regular PCA scatter
# ---------------------------------------------------------------------------

def plot_pca_scatter(
    emb: np.ndarray,
    phase: np.ndarray,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    explore = phase == 0
    exploit = phase == 1
    ax.scatter(emb[explore, 0], emb[explore, 1], s=4, alpha=0.3,
               c="#4477AA", label=f"explore (n={int(explore.sum())})")
    ax.scatter(emb[exploit, 0], emb[exploit, 1], s=4, alpha=0.3,
               c="#EE6677", label=f"exploit (n={int(exploit.sum())})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", markerscale=3, fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_png_and_pdf(fig, out_path, dpi=500)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 2 part 2: trajectory PCA (individual + switch-aligned mean)
# ---------------------------------------------------------------------------

def _split_per_traj(emb: np.ndarray, traj_id: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for k in np.unique(traj_id):
        out[int(k)] = emb[traj_id == k]
    return out


def _switch_aligned_mean(
    emb_per_traj: dict[int, np.ndarray],
    trajs: TrajectoryDataset,
    max_window: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a (2W+1, 2) mean trajectory aligned to switch_t per traj.

    Returns (offsets, mean_xy, n_per_offset)."""
    W = max_window
    offsets = np.arange(-W, W + 1)
    sums = np.zeros((2 * W + 1, 2), dtype=np.float64)
    counts = np.zeros(2 * W + 1, dtype=np.int64)
    for k, traj in enumerate(trajs.trajectories):
        if k not in emb_per_traj:
            continue
        e = emb_per_traj[k]
        sw = traj.switch_t
        T = e.shape[0]
        for off in offsets:
            t = sw + off
            if 0 <= t < T:
                sums[off + W] += e[t]
                counts[off + W] += 1
    mean_xy = np.full((2 * W + 1, 2), np.nan, dtype=np.float64)
    nz = counts > 0
    mean_xy[nz] = sums[nz] / counts[nz, None]
    return offsets, mean_xy, counts


def plot_trajectory_pca(
    emb: np.ndarray,
    traj_id: np.ndarray,
    trajs: TrajectoryDataset,
    out_path: str | Path,
    *,
    max_individuals: int = 60,
    max_window: int = 80,
) -> None:
    """Each trajectory drawn as a thin line: pre-switch blue, post-switch red.
    Mean trajectory (aligned to switch event across all trajectories) drawn as a
    thick black line on top, with a marker at offset 0.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    emb_per_traj = _split_per_traj(emb, traj_id)
    keys = sorted(emb_per_traj.keys())
    rng = np.random.RandomState(0)
    if len(keys) > max_individuals:
        keys = sorted(rng.choice(keys, size=max_individuals, replace=False))

    for k in keys:
        e = emb_per_traj[k]
        traj = trajs.trajectories[k]
        sw = traj.switch_t
        if sw > 0:
            ax.plot(e[:sw, 0], e[:sw, 1], color="#4477AA", linewidth=0.5,
                    alpha=0.4)
        if sw < e.shape[0]:
            ax.plot(e[sw:, 0], e[sw:, 1], color="#EE6677", linewidth=0.5,
                    alpha=0.4)
        ax.scatter([e[sw, 0]] if sw < e.shape[0] else [],
                   [e[sw, 1]] if sw < e.shape[0] else [],
                   marker="*", s=14, c="black", zorder=4, alpha=0.6)

    offsets, mean_xy, counts = _switch_aligned_mean(
        _split_per_traj(emb, traj_id), trajs, max_window=max_window,
    )
    valid = ~np.isnan(mean_xy[:, 0])
    if valid.any():
        m_xy = mean_xy[valid]
        m_off = offsets[valid]
        # Pre-switch part of mean (offset < 0) blue; post-switch (offset >= 0) red.
        pre = m_off < 0
        post = m_off >= 0
        if pre.sum() >= 2:
            ax.plot(m_xy[pre, 0], m_xy[pre, 1], color="#222244",
                    linewidth=2.4, alpha=0.95, label="mean (pre-switch)")
        if post.sum() >= 2:
            ax.plot(m_xy[post, 0], m_xy[post, 1], color="#882222",
                    linewidth=2.4, alpha=0.95, label="mean (post-switch)")
        zero = np.where(m_off == 0)[0]
        if zero.size:
            i = int(zero[0])
            ax.scatter([m_xy[i, 0]], [m_xy[i, 1]], marker="*", s=320,
                       c="yellow", edgecolors="black", linewidths=1.0,
                       zorder=10, label="switch (oracle store)")

    # Phase legend (in addition to mean lines).
    from matplotlib.lines import Line2D
    extra = [
        Line2D([0], [0], color="#4477AA", lw=1.2, label="individual: explore"),
        Line2D([0], [0], color="#EE6677", lw=1.2, label="individual: exploit"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=extra + handles, fontsize=10, loc="best")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_png_and_pdf(fig, out_path, dpi=500)
    plt.close(fig)
