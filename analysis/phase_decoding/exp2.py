"""Exp 2 entry point: regular PCA scatter + trajectory PCA visualization.

Part 1 reuses the explore/exploit trial data collected by exp1 (or collects
fresh if --collect_trials_dir not given).
Part 2 collects two-episode trajectories with oracle store + no-reset teleport.

Usage:
    python -m analysis.phase_decoding.exp2 \
        --ckpt CKPT --out_dir OUT \
        [--trials_dir EXP1_OUT/trials]   # reuse existing if present
        [--num_arenas 100] [--n_starts 50]    # for fresh trials
        [--n_traj_per_arena 4] [--max_steps_ep1 200] [--max_steps_ep2 200]
        [--mlp_hidden 64] [--mlp_epochs 30]
        [--stochastic | --deterministic]
        [--seed 0]

Outputs:
    OUT/exp2_regular_pca.png
    OUT/exp2_trajectory_pca.png
    OUT/trajectories/...           per-trajectory npz + meta.json
    OUT/exp2_meta.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .classifier import extract_hidden, train_mlp
from .collect_trajectory import TwoEpisodeTrajectoryCollector
from .collect_trials import ExploreExploitCollector, TrialsDataset
from .rollout import RolloutEngine
from .viz import plot_pca_scatter, plot_trajectory_pca


def _pca_2d(X: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2), dtype=np.float32)
    return PCA(n_components=2, random_state=0).fit_transform(X).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--encoder", default=None, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--trials_dir", default=None, type=str,
                    help="Existing exp1 trials/ dir to reuse for Part 1. If "
                         "omitted, collects fresh trials with the params below.")
    ap.add_argument("--split", type=str, default="recorded",
                   help="Which validation arenas to decode from. 'recorded' "
                        "(default) is the run's own base_val from world.json. "
                        "Otherwise 'trait=level' pairs over place/wall/goal, "
                        "levels same | held_out | ood; unnamed traits default "
                        "to held_out. --split goal=ood asks whether phase is "
                        "decodable in arenas whose goal region the run never "
                        "trained on.")
    ap.add_argument("--val_seed", type=int, default=0,
                   help="Seed for minting a --split arena set.")
    ap.add_argument("--val_size", type=int, default=None,
                    help="Mint the arenas at this size -- the size-OOD "
                         "axis. Needs a minted --split; equal to the "
                         "training size it is a no-op.")
    ap.add_argument("--num_arenas", type=int, default=100)
    ap.add_argument("--n_starts", type=int, default=50,
                    help="If collecting fresh trials for Part 1.")
    ap.add_argument("--max_steps", type=int, default=200,
                    help="If collecting fresh trials for Part 1.")
    ap.add_argument("--n_dist_min", type=int, default=0)
    ap.add_argument("--n_dist_max", type=int, default=5)
    ap.add_argument("--n_traj_per_arena", type=int, default=4)
    ap.add_argument("--max_steps_ep1", type=int, default=200)
    ap.add_argument("--max_steps_ep2", type=int, default=200)
    ap.add_argument("--mlp_hidden", type=int, default=64)
    ap.add_argument("--mlp_epochs", type=int, default=30)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_batch", type=int, default=256)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--stochastic", dest="deterministic", action="store_false")
    grp.add_argument("--deterministic", dest="deterministic", action="store_true")
    ap.set_defaults(deterministic=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random_agent", action="store_true", default=False,
                    help="Control: use a freshly-initialized agent with the "
                         "same architecture as the ckpt (cfg loaded from "
                         "ckpt, weights NOT loaded).")
    ap.add_argument("--random_init_seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[exp2] starting; out_dir={out_dir}", flush=True)
    t_total = time.perf_counter()

    engine = RolloutEngine(
        ckpt_path=args.ckpt, encoder_path=args.encoder,
        device=args.device, num_arenas=args.num_arenas,
        random_agent=args.random_agent,
        random_init_seed=args.random_init_seed,
        split=args.split, val_seed=args.val_seed,
        val_size=args.val_size,
    )
    bundle = engine.build_bundle()

    # ---- Part 1: regular PCA on Exp-1-style trial data ------------------------
    print("[exp2] === part 1: regular PCA ===", flush=True)
    if args.trials_dir is not None:
        print(f"[exp2] loading trials from {args.trials_dir}", flush=True)
        trials = TrialsDataset.load(args.trials_dir)
        print(f"[exp2] loaded {len(trials.per_arena)} arenas", flush=True)
    else:
        trials = ExploreExploitCollector(engine).collect(
            bundle,
            n_starts=args.n_starts, max_steps=args.max_steps,
            n_dist_min=args.n_dist_min, n_dist_max=args.n_dist_max,
            deterministic=args.deterministic,
            seed=args.seed,
        )
        print(f"[exp2] saving trials to {out_dir / 'trials'}", flush=True)
        trials.save(out_dir / "trials")

    h_p1, phase_p1, _ = trials.pooled()
    print(f"[exp2] part 1 pooled (h, phase): {h_p1.shape}", flush=True)
    mlp_p1, m_p1 = train_mlp(
        h_p1, phase_p1,
        hidden_dim=args.mlp_hidden, epochs=args.mlp_epochs,
        lr=args.mlp_lr, batch_size=args.mlp_batch,
        seed=args.seed, device=device,
        log_label="part1",
    )
    print("[exp2] part 1 extracting hidden activations + PCA", flush=True)
    hidden_p1 = extract_hidden(mlp_p1, h_p1, m_p1["scaler"], device=device)
    pca_p1 = _pca_2d(hidden_p1)
    plot_pca_scatter(pca_p1, phase_p1, out_dir / "exp2_regular_pca.png")
    print(f"[exp2] part 1 wrote {out_dir / 'exp2_regular_pca.png'}", flush=True)

    # ---- Part 2: trajectory PCA -----------------------------------------------
    print("[exp2] === part 2: trajectory PCA ===", flush=True)
    trajs = TwoEpisodeTrajectoryCollector(engine).collect(
        bundle,
        n_traj_per_arena=args.n_traj_per_arena,
        max_steps_ep1=args.max_steps_ep1,
        max_steps_ep2=args.max_steps_ep2,
        n_dist_min=args.n_dist_min, n_dist_max=args.n_dist_max,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    print(f"[exp2] saving trajectories to {out_dir}", flush=True)
    trajs.save(out_dir)

    if not trajs.trajectories:
        print("[exp2] no trajectories collected (every ep1 timed out). "
              "Skipping trajectory PCA.", flush=True)
    else:
        h_p2, phase_p2, traj_id = trajs.pooled()
        print(f"[exp2] part 2 pooled (h, phase): {h_p2.shape}", flush=True)
        mlp_p2, m_p2 = train_mlp(
            h_p2, phase_p2,
            hidden_dim=args.mlp_hidden, epochs=args.mlp_epochs,
            lr=args.mlp_lr, batch_size=args.mlp_batch,
            seed=args.seed, device=device,
            log_label="part2",
        )
        print("[exp2] part 2 extracting hidden activations + PCA", flush=True)
        hidden_p2 = extract_hidden(mlp_p2, h_p2, m_p2["scaler"], device=device)
        pca_p2 = _pca_2d(hidden_p2)
        plot_trajectory_pca(
            pca_p2, traj_id, trajs,
            out_dir / "exp2_trajectory_pca.png",
        )
        print(f"[exp2] part 2 wrote {out_dir / 'exp2_trajectory_pca.png'}",
              flush=True)

    (out_dir / "exp2_meta.json").write_text(json.dumps({
        "config": vars(args),
        "trials_meta": trials.meta,
        "trajectories_meta": trajs.meta,
        "mlp_part1": {
            "val_balanced_acc": m_p1["val_balanced_acc"],
            "hidden_dim": m_p1["hidden_dim"],
        },
    }, indent=2, default=str))
    print(f"[exp2] all done in {time.perf_counter() - t_total:.1f}s. "
          f"Wrote PCA plots to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
