"""Standalone plotter for exp1 metrics.json files.

Given one or more metrics.json paths (or run directories containing one),
re-renders the bar plot. With multiple runs, draws grouped bars per split
family with a per-run legend — useful for trained-vs-random-init control
overlays or trained-at-checkpoint-X-vs-Y comparisons.

Usage:
    # single-run re-plot
    python -m hopfield_nav.phase_decoding_v2.plot \
        --metrics RUN_DIR/metrics.json \
        --out RUN_DIR/bars.png

    # compare trained vs. random-init control
    python -m hopfield_nav.phase_decoding_v2.plot \
        --metrics TRAINED/metrics.json RANDOM/metrics.json \
        --labels trained random_init \
        --out trained_vs_random.png

You can pass a run directory in place of `metrics.json` and the script will
look for `<dir>/metrics.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .viz import plot_bars, plot_bars_grouped


def _resolve(p: str) -> Path:
    path = Path(p)
    if path.is_dir():
        path = path / "metrics.json"
    if not path.exists():
        raise SystemExit(f"[plot] not found: {path}")
    return path


def _load_results(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if "per_fold" in raw:
        return raw["per_fold"]
    # tolerate older layouts where the file IS the per_fold dict.
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics", required=True, type=str, nargs="+",
                    help="One or more metrics.json paths (or run dirs).")
    ap.add_argument("--labels", type=str, nargs="+", default=None,
                    help="Per-run legend labels. Default: parent dir name.")
    ap.add_argument("--out", required=True, type=str,
                    help="PNG path to write.")
    args = ap.parse_args()

    paths = [_resolve(m) for m in args.metrics]
    if args.labels is not None and len(args.labels) != len(paths):
        raise SystemExit(
            f"[plot] --labels has {len(args.labels)} entries but --metrics has "
            f"{len(paths)}; counts must match."
        )
    labels = args.labels if args.labels is not None else [
        p.parent.name for p in paths
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if len(paths) == 1:
        results = _load_results(paths[0])
        plot_bars(results, out)
        print(f"[plot] wrote {out} (single run: {labels[0]})", flush=True)
        return

    runs = [(label, _load_results(p)) for label, p in zip(labels, paths)]
    plot_bars_grouped(runs, out)
    print(f"[plot] wrote {out} ({len(runs)} runs: {', '.join(labels)})",
          flush=True)


if __name__ == "__main__":
    main()
