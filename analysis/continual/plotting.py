"""Renderer for paper-ready continual-learning figures.

Reads a history JSON written by ``baseline.py`` (RNN baseline) or ``agenthash.py``
(Hopfield agent) and writes figure files as both PNG and PDF:

  - ``<prefix>_forgetting.png``     — per-env reached-bit over the protocol
  - ``<prefix>_steps_to_goal.png``  — per-env steps-to-reach over the protocol

The legacy helpers ``save_forgetting_plot`` / ``save_steps_to_goal_plot`` are
kept here (moved out of ``train_rnn.py``) so ``train_rnn.train_sequential``'s
inline plot path keeps working unchanged.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _parse_bool_arg(value: str) -> bool:
    """Parse common CLI truthy/falsy strings into a bool."""
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


# ---------------------------------------------------------------------------
# Smoothing + generic per-env plot
# ---------------------------------------------------------------------------

def _smooth_with_nans(values: list[float], window: int) -> np.ndarray:
    """Centered rolling mean of size ``window``. NaN-aware:

      - If the input at position ``i`` is NaN, the output at ``i`` is NaN.
        (Prevents the curve from leaking into pre-block regions where the
        env hadn't been introduced yet — without this, a centered window
        peeks forward into real data and plots a value before the block
        actually starts.)
      - Otherwise output the mean of non-NaN values in the window.

    ``window <= 1`` returns the input unchanged.
    """
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size == 0:
        return arr
    half = window // 2
    out = np.empty_like(arr)
    for i in range(arr.size):
        if np.isnan(arr[i]):
            out[i] = float("nan")
            continue
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        chunk = arr[lo:hi]
        out[i] = float("nan") if np.all(np.isnan(chunk)) else float(np.nanmean(chunk))
    return out


def _detect_n_iters(trace: list, metric_key: str) -> int:
    """Inspect a trace to determine num_full_iters.

    Returns 1 if the metric is stored as a scalar (legacy or single-iter format),
    otherwise the length of the first list-valued metric encountered.
    """
    for _, _, metrics in trace:
        for m in metrics.values():
            if m is None:
                continue
            val = m.get(metric_key)
            if val is None:
                continue
            return len(val) if isinstance(val, list) else 1
    return 1


def _plot_metric_per_env(
    history: dict,
    n_envs: int,
    metric_key: str,
    ylabel: str,
    out_path: str,
    ylim: tuple[float, float] | None = None,
    label_y_frac: float = 0.04,
    label_va: str = "bottom",
    smooth_window: int = 1,
    show_std: bool = False,
    nan_fill: float | None = None,
) -> None:
    """Generic per-env trace plot.

    X = cumulative step; Y = ``metric_key`` value from each step's metrics
    dict. One line per env. Training blocks shaded by which env was active.
    NaN values (or ``None`` from JSON) appear as gaps. ``smooth_window > 1``
    applies a centered NaN-aware rolling mean before plotting.

    Multi-iter histories (where each metric is a list of length num_full_iters)
    are aggregated as: smooth each iter's curve, then nanmean / nanstd across
    iters at each step. The mean is plotted as a line; ±1σ is a shaded band.

    ``nan_fill``: when set, treat "env was introduced but metric value is
    null" (e.g. ``steps_to_goal`` on a failed trial) as ``nan_fill`` instead
    of a gap. The "env not yet introduced" gap (env missing from the inner
    dict entirely) is still preserved as NaN. Used for the steps-to-goal plot
    so failed trials render as the max_steps cap rather than vanishing from
    the curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _tick_fs = 13
    _axis_label_fs = 13
    _block_label_fs = 13

    trace = history["trace"]
    blocks = history["blocks"]
    if not trace:
        return

    steps = [t[0] for t in trace]
    n_steps = len(steps)
    n_iters = _detect_n_iters(trace, metric_key)

    # (n_envs, n_iters, n_steps), NaN-filled for missing values.
    per_env_iter_curves = np.full((n_envs, n_iters, n_steps), np.nan, dtype=float)
    for s_idx, (_, _, metrics) in enumerate(trace):
        for j in range(n_envs):
            m = metrics.get(j)
            if m is None:
                # Env not introduced at this step — leave NaN gap regardless of nan_fill.
                continue
            val = m.get(metric_key)
            if val is None:
                # Scalar legacy form: env was introduced but metric is null.
                if nan_fill is not None:
                    per_env_iter_curves[j, 0, s_idx] = float(nan_fill)
                continue
            if isinstance(val, list):
                for k, v in enumerate(val):
                    if v is not None:
                        per_env_iter_curves[j, k, s_idx] = float(v)
                    elif nan_fill is not None:
                        per_env_iter_curves[j, k, s_idx] = float(nan_fill)
            else:
                per_env_iter_curves[j, 0, s_idx] = float(val)

    fig, ax = plt.subplots(figsize=(max(8, n_envs * 1.6), 4.8))
    cmap = plt.cm.tab10
    for start, end, env_i in blocks:
        ax.axvspan(start - 0.5, end + 0.5,
                   color=cmap(env_i % 10), alpha=0.10, zorder=0)

    for j in range(n_envs):
        # Smooth each iter independently, then aggregate across iters.
        smoothed = np.array([
            _smooth_with_nans(per_env_iter_curves[j, k, :], smooth_window)
            for k in range(n_iters)
        ])  # (n_iters, n_steps)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(smoothed, axis=0)
            std = np.nanstd(smoothed, axis=0) / np.sqrt(n_iters)
        color = cmap(j % 10)
        ax.plot(steps, mean, color=color, linewidth=2.0)
        if n_iters > 1 and show_std:
            ax.fill_between(
                steps, mean - std, mean + std,
                color=color, alpha=0.15, linewidth=0,
            )

    if ylim is not None:
        ax.set_ylim(*ylim)
    for start, end, env_i in blocks:
        ax.text(
            (start + end) / 2.0, label_y_frac, f"train env {env_i}",
            ha="center", va=label_va, fontsize=_block_label_fs,
            color=cmap(env_i % 10),
            transform=ax.get_xaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
    ax.set_xlabel("Step", fontsize=_axis_label_fs)
    ax.set_ylabel(ylabel, fontsize=_axis_label_fs, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=_tick_fs)
    ax.axhline(1, linestyle=":", linewidth=1.0, color="black", alpha=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_root, _ = os.path.splitext(out_path)
    fig.savefig(f"{out_root}.png", dpi=500)
    fig.savefig(f"{out_root}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Legacy wrappers (consumed by train_rnn.train_sequential)
# ---------------------------------------------------------------------------

def save_forgetting_plot(
    history: dict, n_envs: int, out_path: str, smooth_window: int = 1,
) -> None:
    """nav_det per env over training (forgetting curve). Train_rnn schema."""
    _plot_metric_per_env(
        history, n_envs,
        metric_key="nav_det",
        ylabel="Nav_det",
        out_path=out_path,
        ylim=(-0.02, 1.02),
        smooth_window=smooth_window,
    )


def save_steps_to_goal_plot(
    history: dict, n_envs: int, out_path: str, smooth_window: int = 1,
) -> None:
    """mean_steps_to_goal per env. Train_rnn schema (NaN where 0 trials succeeded)."""
    _plot_metric_per_env(
        history, n_envs,
        metric_key="mean_steps_to_goal",
        ylabel="Mean_steps_to_goal",
        out_path=out_path,
        ylim=None,
        smooth_window=smooth_window,
    )


# ---------------------------------------------------------------------------
# CLI: read a final_plotting history JSON, render both figures
# ---------------------------------------------------------------------------

def _load_history(path: str) -> tuple[dict, dict]:
    """Load a history JSON; cast string env-keys back to int.

    Returns (history_for_plotter, metadata).
    """
    with open(path) as f:
        raw = json.load(f)
    md = raw["metadata"]
    trace = []
    for entry in raw["trace"]:
        step, train_env_idx, inner_str = entry
        inner = {int(k): v for k, v in inner_str.items()}
        trace.append((int(step), int(train_env_idx), inner))
    blocks = [(int(b[0]), int(b[1]), int(b[2])) for b in raw["blocks"]]
    return {"trace": trace, "blocks": blocks}, md


def _render(history: dict, md: dict, out_prefix: str, smooth: int, show_std: bool) -> None:
    n_envs = int(md["n_envs"])
    x_axis = md.get("x_axis_label", "step")
    max_steps = md.get("max_steps")  # used as nan_fill for the steps-to-goal plot

    fwd_path = f"{out_prefix}_forgetting.png"
    steps_path = f"{out_prefix}_steps_to_goal.png"
    path_path = f"{out_prefix}_path_to_goal.png"

    _plot_metric_per_env(
        history, n_envs,
        metric_key="reached",
        ylabel="Reached",
        out_path=fwd_path,
        ylim=(-0.02, 1.02),
        smooth_window=smooth,
        show_std=show_std,
    )
    _plot_metric_per_env(
        history, n_envs,
        metric_key="steps_to_goal",
        ylabel="Steps_to_goal",
        out_path=steps_path,
        ylim=(0.0, float(max_steps)) if max_steps is not None else None,
        smooth_window=smooth,
        show_std=show_std,
        nan_fill=float(max_steps) if max_steps is not None else None,
    )
    _plot_metric_per_env(
        history, n_envs,
        metric_key="path_to_goal",
        ylabel="Path_to_goal",
        out_path=path_path,
        ylim=(0.0, float(max_steps)) if max_steps is not None else None,
        smooth_window=smooth,
        show_std=show_std,
        nan_fill=float(max_steps) if max_steps is not None else None,
    )

    print(f"Wrote {fwd_path}")
    print(f"Wrote {steps_path}")
    print(f"Wrote {path_path}")
    print(f"  x-axis is in units of: {x_axis}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", required=True,
                   help="History JSON written by baseline.py / agenthash.py")
    p.add_argument("--out_prefix", required=True,
                   help="Output path prefix (writes <prefix>_forgetting.png "
                        "and <prefix>_steps_to_goal.png)")
    p.add_argument("--smooth", type=int, default=1,
                   help="Centered NaN-aware MA window applied at plot time. "
                        "Default 1 (no smoothing).")
    p.add_argument("--show_std", type=_parse_bool_arg, default=False,
                   help="Show standard deviation as a shaded band (true/false).")
    args = p.parse_args()

    history, md = _load_history(args.history)
    _render(history, md, args.out_prefix, args.smooth, args.show_std)


if __name__ == "__main__":
    main()
