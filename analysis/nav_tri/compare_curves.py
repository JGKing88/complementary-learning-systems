"""Several runs' training curves on one figure -- one colour per run.

`training_curve.py` draws one run with its three metrics on one axis. This
puts RUNS side by side: three panels (success rate, path optimality, swept
coverage), one line per run, d=0 dotted and d=10 solid, on the same x-axis
choice `training_curve` offers (update / episodes / env_steps). It is the
"is the one-env model as good as d0_base" picture, which the per-run figures
only answer by flipping between them.

    python -m analysis.nav_tri.compare_curves \\
        --log d0_base=$CLS_RUNS/logs/nav_p2_22133273.out \\
        --log "one env (K=4)"=$CLS_RUNS/logs/nav_p2_22756994.out \\
        --x episodes --out $CLS_RESULTS/nav_tri_probe/one_vs_d0_by_episodes.png

Reads the same lines `training_curve` reads (trainer logs or `reeval_series`
logs), so a sampled series can be overlaid on a deterministic one -- label
them.
"""
from __future__ import annotations

import argparse

import numpy as np

from .training_curve import X_LABELS, parse_log, series, x_axis

PANELS = (("success", "success rate"),
          ("optimality", "path optimality"),
          ("swept", "swept coverage"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", action="append", required=True,
                   help="label=path; repeatable, drawn in order")
    p.add_argument("--out", required=True, help="output image path")
    p.add_argument("--x", choices=tuple(X_LABELS), default="update")
    p.add_argument("--n_dist", type=int, nargs="+", default=[0, 10])
    p.add_argument("--goal_radius", type=float, default=1.0)
    p.add_argument("--xmax", type=float, default=None,
                   help="clip the x-axis (in --x units)")
    p.add_argument("--title", default=None)
    p.add_argument("--smooth", type=int, default=1,
                   help="centred moving-average window in evals (1 = none)")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    runs = []
    for spec in a.log:
        label, path = spec.split("=", 1)
        log = parse_log(path)
        if not log:
            raise SystemExit(f"{path}: no eval pairs")
        xmap, xlabel = x_axis(log, path, a.x)
        sets = {nd: series(log, nd, a.goal_radius) for nd in a.n_dist}
        runs.append((label, xmap, xlabel, sets))

    def _smooth(v):
        if a.smooth <= 1:
            return v
        k = a.smooth
        out = np.full_like(v, np.nan)
        for i in range(len(v)):
            lo, hi = max(0, i - k // 2), min(len(v), i + k // 2 + 1)
            w = v[lo:hi]
            w = w[np.isfinite(w)]
            out[i] = w.mean() if len(w) else np.nan
        return out

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = {0: (":", 1.4, 0.7), 5: ("--", 1.4, 0.8), 10: ("-", 2.0, 1.0)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, (key, name) in zip(axes, PANELS):
        for i, (label, xmap, xlabel, sets) in enumerate(runs):
            for nd, s in sets.items():
                x = np.array([xmap[int(u)] for u in s["u"] if int(u) in xmap])
                y = np.array([v for u, v in zip(s["u"], s[key]) if int(u) in xmap])
                if a.xmax is not None:
                    keep = x <= a.xmax
                    x, y = x[keep], y[keep]
                ls, lw, al = styles.get(nd, ("-", 1.6, 1.0))
                ax.plot(x, _smooth(y), ls, color=colors[i % len(colors)],
                        linewidth=lw, alpha=al,
                        label=f"{label} (d={nd})" if key == "success" else None)
        ax.set_title(name)
        ax.set_ylim(0.0, 1.03)
        ax.axhline(1.0, linestyle=":", linewidth=0.8, color="black", alpha=0.5)
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.set_xlabel(runs[0][2])
        if a.x != "update":
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else
                             (f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}")))
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.92, ncol=1)
    if a.title:
        fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
