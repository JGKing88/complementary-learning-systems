"""Exp 1 entry point: collect explore/exploit trials → all splits →
parallelism + decodability bars.

Usage:
    python -m analysis.phase_decoding.exp1 \
        --ckpt CKPT --out_dir OUT \
        [--num_arenas 100] [--n_starts 100] [--max_steps 200] \
        [--n_dist_min 0] [--n_dist_max 5] \
        [--stochastic | --deterministic] \
        [--n_random_splits 20] [--test_frac 0.2] [--seed 0]
        [--trials_dir EXISTING/trials]   # resume: skip collection
        [--subsample_train 200000]       # cap LR train rows per fold

Outputs in OUT:
    trials/per_arena/<idx>.npz    (skipped when --trials_dir is given)
    trials/meta.json
    scaffold.json
    metrics.json     all per-fold parallelism + decodability scalars
    bars.png         the two-panel bar plot
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from .collect_trials import ExploreExploitCollector, TrialsDataset
from .metrics import decodability, parallelism_score, within_arena_baseline
from .rollout import RolloutEngine
from .splits import all_splits
from .viz import plot_bars


def _evaluate(
    h: np.ndarray,
    phase: np.ndarray,
    arena_id: np.ndarray,
    splits,
    *, subsample_train: int | None = None,
) -> dict:
    """Run all folds against a pre-pooled (h, phase, arena_id) triple."""
    out: dict = {}
    for split in splits:
        n_folds = len(split.folds)
        print(f"[exp1] split '{split.name}': {n_folds} folds", flush=True)
        t_split = time.perf_counter()
        rows = []
        for i, fold in enumerate(split.folds):
            t_fold = time.perf_counter()
            par = parallelism_score(h, phase, arena_id, fold.train, fold.test)
            dec = decodability(
                h, phase, arena_id, fold.train, fold.test,
                subsample_train=subsample_train,
            )
            rows.append({
                "fold": fold.name,
                "parallelism": par,
                "decodability": dec,
                "n_train_arenas": len(fold.train),
                "n_test_arenas": len(fold.test),
            })
            print(
                f"[exp1]   fold {i + 1}/{n_folds} {fold.name}: "
                f"parallelism={par:.3f} decodability={dec:.3f} "
                f"({time.perf_counter() - t_fold:.2f}s)",
                flush=True,
            )
            # Big sklearn objects can hold workspace; release between folds.
            gc.collect()
        accs = np.asarray([r["decodability"] for r in rows
                           if r["decodability"] == r["decodability"]])
        pars = np.asarray([r["parallelism"] for r in rows
                           if r["parallelism"] == r["parallelism"]])
        print(
            f"[exp1] split '{split.name}' done in "
            f"{time.perf_counter() - t_split:.1f}s | "
            f"parallelism mean={pars.mean():.3f} sd={pars.std(ddof=1) if pars.size > 1 else 0.0:.3f} | "
            f"decodability mean={accs.mean():.3f} sd={accs.std(ddof=1) if accs.size > 1 else 0.0:.3f}",
            flush=True,
        )
        out[split.name] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--encoder", default=None, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--trials_dir", default=None, type=str,
                    help="Path to an existing trials/ directory (from a prior "
                         "run). If given, skips collection.")
    ap.add_argument("--num_arenas", type=int, default=100)
    ap.add_argument("--n_starts", type=int, default=100,
                    help="Trials per condition per arena.")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--n_dist_min", type=int, default=0)
    ap.add_argument("--n_dist_max", type=int, default=5)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--stochastic", dest="deterministic", action="store_false")
    grp.add_argument("--deterministic", dest="deterministic", action="store_true")
    ap.set_defaults(deterministic=False)
    ap.add_argument("--n_random_splits", type=int, default=20)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--skip_loo", action="store_true", default=False,
                    help="Skip the LOO split family. LOO has one fold per "
                         "arena and each fold pools nearly all trials, so it "
                         "dominates eval time on big runs.")
    ap.add_argument("--subsample_train", type=int, default=None,
                    help="Cap on training rows per fold for the LR decoder. "
                         "Big LOO pools (~1.8M rows × 512 dims = 3.6 GB) can "
                         "OOM otherwise. Set e.g. 200000 to bound memory.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random_agent", action="store_true", default=False,
                    help="Control: use a freshly-initialized agent with the "
                         "same architecture as the ckpt (cfg loaded from "
                         "ckpt, weights NOT loaded). Tells you whether any "
                         "structure in the bars is learned vs. an artifact of "
                         "the input format.")
    ap.add_argument("--random_init_seed", type=int, default=0,
                    help="torch.manual_seed used for agent random init when "
                         "--random_agent is set.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[exp1] starting; out_dir={out_dir}", flush=True)
    t_total = time.perf_counter()

    quadrants_map: dict[int, int]
    arena_ids: list[int]

    if args.trials_dir is not None:
        print(f"[exp1] resume: loading trials from {args.trials_dir}",
              flush=True)
        data = TrialsDataset.load(args.trials_dir)
        quadrants_map = {a: td.quadrant for a, td in data.per_arena.items()}
        arena_ids = sorted(data.per_arena.keys())
        print(
            f"[exp1] loaded {len(arena_ids)} arenas; "
            f"quadrant counts {dict(sorted({q: sum(1 for v in quadrants_map.values() if v == q) for q in range(4)}.items()))}",
            flush=True,
        )
    else:
        engine = RolloutEngine(
            ckpt_path=args.ckpt, encoder_path=args.encoder,
            device=args.device, num_arenas=args.num_arenas,
            random_agent=args.random_agent,
            random_init_seed=args.random_init_seed,
        )
        bundle = engine.build_bundle()
        by_q = {q: 0 for q in range(4)}
        for q in bundle.quadrants:
            by_q[q] = by_q.get(q, 0) + 1
        print(f"[exp1] bundle ready: {len(bundle.envs)} arenas, "
              f"quadrant counts {by_q}", flush=True)

        collector = ExploreExploitCollector(engine)
        data = collector.collect(
            bundle,
            n_starts=args.n_starts, max_steps=args.max_steps,
            n_dist_min=args.n_dist_min, n_dist_max=args.n_dist_max,
            deterministic=args.deterministic,
            seed=args.seed,
        )
        print(f"[exp1] saving trials to {out_dir / 'trials'}", flush=True)
        data.save(out_dir / "trials")
        (out_dir / "scaffold.json").write_text(
            json.dumps(bundle.scaffold(), indent=2))

        # Free engine + bundle + GPU memory before the big eval pool allocation.
        quadrants_map = {a: bundle.quadrants[a] for a in bundle.arena_ids()}
        arena_ids = bundle.arena_ids()
        del engine, bundle, collector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Pool ONCE; drop per-arena memory before allocating the global array.
    print("[exp1] pooling all trials into a single (h, phase, arena_id) triple",
          flush=True)
    h, phase, arena_arr = data.pooled()
    print(
        f"[exp1] pooled shapes: h={h.shape} phase={phase.shape} "
        f"arena={arena_arr.shape}; memory ≈ "
        f"{h.nbytes / (1024**3):.2f} GB",
        flush=True,
    )
    # Release the per-arena dict now — we don't need it again.
    data.per_arena.clear()
    del data
    gc.collect()

    splits = all_splits(
        arena_ids, quadrants_map,
        n_random=args.n_random_splits, test_frac=args.test_frac, seed=args.seed,
    )
    if args.skip_loo:
        before = len(splits)
        splits = [s for s in splits if s.name != "LOO"]
        if len(splits) < before:
            print("[exp1] --skip_loo: dropping the LOO split family", flush=True)
    print(
        f"[exp1] computing metrics over {sum(len(s.folds) for s in splits)} "
        f"folds across {len(splits)} split families "
        + ", ".join(f"{s.name}={len(s.folds)}" for s in splits),
        flush=True,
    )

    results = _evaluate(
        h, phase, arena_arr, splits,
        subsample_train=args.subsample_train,
    )
    print(f"[exp1] computing within-arena baseline (one fold per arena, "
          f"{int(round((1 - args.test_frac) * 100))}/{int(round(args.test_frac * 100))} "
          "timestep split)", flush=True)
    t_within = time.perf_counter()
    within_rows = within_arena_baseline(
        h, phase, arena_arr,
        test_frac=args.test_frac, seed=args.seed,
    )
    n_skipped = sum(1 for r in within_rows if r.get("skipped", False))
    accs = np.asarray([r["decodability"] for r in within_rows
                       if r["decodability"] == r["decodability"]])
    pars = np.asarray([r["parallelism"] for r in within_rows
                       if r["parallelism"] == r["parallelism"]])
    print(
        f"[exp1] within-arena done in {time.perf_counter() - t_within:.1f}s "
        f"({n_skipped} skipped) | "
        f"parallelism mean={pars.mean():.3f} sd={pars.std(ddof=1) if pars.size > 1 else 0.0:.3f} | "
        f"decodability mean={accs.mean():.3f} sd={accs.std(ddof=1) if accs.size > 1 else 0.0:.3f}",
        flush=True,
    )
    # Match the per-fold dict shape used elsewhere.
    results["Within-arena"] = [
        {"fold": f"arena={r['arena_id']}",
         "parallelism": r["parallelism"],
         "decodability": r["decodability"],
         "n_train_arenas": 1, "n_test_arenas": 1,
         "n": r["n"]}
        for r in within_rows
    ]
    summary = {}
    for name, rows in results.items():
        summary[name] = {
            "n_folds": len(rows),
            "parallelism_mean": _mean([r["parallelism"] for r in rows]),
            "decodability_mean": _mean([r["decodability"] for r in rows]),
        }
    (out_dir / "metrics.json").write_text(json.dumps({
        "per_fold": results,
        "config": vars(args),
        "summary": summary,
    }, indent=2))

    print(f"[exp1] plotting bars to {out_dir / 'bars.png'}", flush=True)
    plot_bars(results, out_dir / "bars.png")
    print(
        f"[exp1] all done in {time.perf_counter() - t_total:.1f}s. "
        f"Wrote metrics.json + bars.png to {out_dir}",
        flush=True,
    )


def _mean(values) -> float:
    arr = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    return float(arr.mean()) if arr.size else float("nan")


if __name__ == "__main__":
    main()
