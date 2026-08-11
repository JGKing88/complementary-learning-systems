"""Is the coverage uniform over the grid, or concentrated?

`mean_coverage` says how MANY cells a rollout visits. It says nothing about
WHICH, and those come apart badly in this lineage: a policy that orbits the
perimeter and a policy that sweeps evenly can report the same 0.52 while being
qualitatively different agents. The perimeter-orbit basin found in v36 is
exactly this failure, so the number is worth having rather than assuming.

This re-runs `evaluate_exploration`'s trials and keeps the per-cell visit
counts it discards, then reports aggregate statistics -- deliberately not a
trajectory plot, which is not evidence.

    python -m hopfield_nav.explore_min_uniformity --ckpt <ckpt> [--n_distractors 0]

Reported per distractor level:

  occupancy        per cell, the fraction of trials that ever stood on it.
                   Its mean IS mean_coverage -- averaging over cells instead
                   of over trials -- so this is a decomposition of the headline
                   number, not a different measurement.
  ring profile     occupancy averaged over cells at each Chebyshev distance
                   from the wall. Ring 0 is the 76-cell perimeter, ring 9 the
                   4-cell centre on a 20x20. A perimeter-orbiting policy shows
                   a steep monotone falloff here and nothing else does.
  edge/centre      mean occupancy of rings 0-1 against rings 5+, and their
                   ratio. 1.0 is uniform; >1 is edge-biased.
  cold fraction    cells with occupancy below 0.10, i.e. ground that nearly
                   every rollout misses. Distinguishes "covers 52% evenly"
                   from "covers one 52% region reliably and the rest never".
  CV, entropy      coefficient of variation of occupancy, and Shannon entropy
                   of the normalized occupancy as a fraction of log(n_cells).
                   Entropy 1.0 is exactly uniform.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.batched import batched_exploration_trials
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint, load_agent,
)
from hopfield_nav.evaluation.metrics import random_start
from hopfield_nav.rollout.distractors import sample_distractors
from hopfield import Hopfield


@torch.no_grad()
def occupancy_grid(ckpt_path, device, n_dist, num_trials, max_steps,
                   num_val_envs, seed=42):
    """Per-cell visit counts, summed over trials and envs.

    Mirrors `evaluate_exploration`'s setup exactly -- same RNG order, same
    fresh-Hopfield-per-trial, same inert goal -- so the mean of the returned
    occupancy reproduces its `mean_coverage`. That equality is asserted by the
    caller and is the check that this is decomposing the real number.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    if num_val_envs is not None:
        cfg.num_val_envs = int(num_val_envs)

    encoder, enc_cfg, enc_gain = load_encoder(cfg.encoder_checkpoint, str(device))
    embed_dim = enc_cfg.out_dim
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(enc_gain)

    torch.manual_seed(0)
    np.random.seed(0)
    val_envs, vh, val_idxs = build_eval_world(cfg, encoder, str(device))
    agent = load_agent(cfg, ck["agent_state_dict"], embed_dim, device)
    agent.eval()

    size = cfg.env.size
    counts = np.zeros((size, size), dtype=np.int64)
    rng = np.random.RandomState(seed)
    n_trials_total = 0
    per_trial_cells = []

    for local_idx, env in enumerate(val_envs):
        env_offset = vh.env_offsets[val_idxs[local_idx]]
        goal = env.goal_location

        hopfields, starts = [], []
        for _ in range(num_trials):
            distractors = sample_distractors(vh, env_offset, size, n_dist, rng)
            hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            for pat in distractors:
                hop.input_memory(torch.from_numpy(pat).float())
            hopfields.append(hop)
            starts.append(random_start(size, goal, rng))

        visited, _found, _steps = batched_exploration_trials(
            agent=agent, env=env, env_offset=env_offset, vectorhash=vh,
            hopfields=hopfields, cfg=cfg, device=device, starts=starts,
            max_steps=max_steps, deterministic=True,
        )
        for cells in visited:
            per_trial_cells.append(len(cells))
            for (x, y) in cells:
                counts[y, x] += 1
        n_trials_total += len(visited)

    return counts, n_trials_total, float(np.mean(per_trial_cells)) / (size * size)


def ring_index(size):
    """Chebyshev distance from the wall, per cell. 0 is the perimeter."""
    ys, xs = np.mgrid[0:size, 0:size]
    return np.minimum(np.minimum(xs, ys), np.minimum(size - 1 - xs, size - 1 - ys))


def report(counts, n_trials, size, mean_cov_direct, label):
    occ = counts / max(n_trials, 1)          # P(a trial visits this cell)
    rings = ring_index(size)

    print(f"\n=== {label} ===")
    print(f"  mean occupancy over cells : {occ.mean():.4f}")
    print(f"  mean_coverage over trials : {mean_cov_direct:.4f}   "
          f"(must match; delta {abs(occ.mean() - mean_cov_direct):.5f})")

    print("\n  ring   cells   mean occupancy   (0 = perimeter)")
    for d in range(rings.max() + 1):
        m = rings == d
        print(f"  {d:>4}   {m.sum():>5}   {occ[m].mean():>14.4f}")

    edge = occ[rings <= 1].mean()
    centre = occ[rings >= 5].mean()
    cold = float((occ < 0.10).mean())
    cv = float(occ.std() / occ.mean()) if occ.mean() > 0 else float("nan")
    p = occ / occ.sum()
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(occ.size))

    print(f"\n  edge (rings 0-1)   {edge:.4f}")
    print(f"  centre (rings 5+)  {centre:.4f}")
    print(f"  edge/centre ratio  {edge / centre:.3f}"
          if centre > 0 else "  edge/centre ratio  inf")
    print(f"  cold cells (<0.10) {cold:.3f}")
    print(f"  CV of occupancy    {cv:.3f}")
    print(f"  entropy / uniform  {ent:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_distractors", type=int, nargs="+", default=[0, 10])
    p.add_argument("--num_trials", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--num-val-envs", type=int, default=10)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    label = args.label or args.ckpt.split("/")[-2]

    for n_dist in args.n_distractors:
        counts, n_trials, mean_cov = occupancy_grid(
            args.ckpt, device, n_dist, args.num_trials, args.max_steps,
            args.num_val_envs,
        )
        size = counts.shape[0]
        report(counts, n_trials, size, mean_cov, f"{label}  n_dist={n_dist}")


if __name__ == "__main__":
    main()
