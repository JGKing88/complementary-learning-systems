"""The similarity structure that makes the matched filter work.

Everything in §5.3-5.7 says the recall is one product, `W x`, with

    W x = (1/D) * sum_k xi_k (xi_k . x)

— a sum of stored patterns weighted by how similar each is to where the agent
is standing. So the entire retrieval rests on one property of the *encoder*:
that `xi_goal . x` (same env) is large while `xi_distractor . x` (other env) is
small. If that gap is real, the goal term dominates the sum and one product
returns the goal. If it is not, nothing else in the pipeline can rescue it.

This measures the gap directly, which nothing so far has:

  * `xi(p) . xi(goal)` as a function of grid distance within the env — the
    signal term, and how far it reaches
  * `xi(p) . xi(distractor)` for patterns drawn from other envs — the
    cross-talk term
  * the resulting matched-filter margin, goal weight against the largest
    distractor weight
  * and the same quantities *after* the tangent projection, which is where the
    explore/exploit magnitude separation comes from: a displacement to another
    env's pattern is an essentially unrelated direction in D dimensions, so it
    keeps only ~sqrt(2/D) of its norm when projected onto a 2-D plane.

    python -m analysis.nav_p2.why_it_works --ckpt <any nav ckpt>
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint,
)
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_distractors", type=int, default=10)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = args.envs
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device),
        getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    D = enc_cfg.out_dim
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    envs, vh, offsets = build_eval_world(cfg, encoder, str(device), ckpt_path=None)
    size = envs[0].size
    gx, gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.int32)

    rng = np.random.RandomState(args.seed)
    same, cross, dists, margin = [], [], [], []
    qn_goal, qn_dist = [], []

    for ei, env in enumerate(envs):
        off, goal = offsets[ei], env.goal_location
        emb_np = vh.get_encoded_state(cells, off)
        emb = F.normalize(torch.from_numpy(emb_np).float().to(device), dim=-1)
        W = vh.gram_schmidt_projection(cells, off)
        d = np.linalg.norm(cells - np.asarray(goal, float), axis=1)

        g = F.normalize(torch.from_numpy(goal_encoding(vh, off, goal))
                        .float().to(device), dim=-1)
        sg = (emb @ g).cpu().numpy()
        same.append(sg); dists.append(d)

        # Tangent projection of the displacement to the goal pattern.
        qg = vh.project_displacement(emb_np, np.repeat(g.cpu().numpy()[None, :],
                                                       len(cells), 0), W)
        qn_goal.append(np.linalg.norm(qg, axis=1))

        for _ in range(args.draws):
            dp = np.stack(sample_distractors(vh, off, size,
                                             args.n_distractors, rng))
            Dt = F.normalize(torch.from_numpy(dp).float().to(device), dim=-1)
            sd = (emb @ Dt.T).cpu().numpy()                 # (n_cell, n_dist)
            cross.append(sd.ravel())
            margin.append(sg - sd.max(axis=1))
            qd = vh.project_displacement(
                emb_np, dp[sd.argmax(axis=1)], W)
            qn_dist.append(np.linalg.norm(qd, axis=1))

    same = np.concatenate(same); dists = np.concatenate(dists)
    cross = np.concatenate(cross); margin = np.concatenate(margin)
    qn_goal = np.concatenate(qn_goal); qn_dist = np.concatenate(qn_dist)

    print(f"{len(envs)} envs x {args.draws} draws x {len(cells)} cells, "
          f"D={D}, {args.n_distractors} distractors\n")

    print("1. THE SIGNAL: similarity to the goal pattern, by grid distance")
    print(f"{'distance':>12} {'median xi.xi_goal':>19} {'p10':>8} {'p90':>8}")
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 30)):
        m = (dists >= lo) & (dists < hi)
        if m.sum() < 20:
            continue
        print(f"{f'{lo}-{hi}':>12} {np.median(same[m]):>19.4f} "
              f"{np.percentile(same[m], 10):>8.4f} {np.percentile(same[m], 90):>8.4f}")

    print("\n2. THE CROSS-TALK: similarity to patterns from OTHER envs")
    print(f"     median {np.median(cross):.4f}   p90 {np.percentile(cross, 90):.4f}"
          f"   p99 {np.percentile(cross, 99):.4f}   max {cross.max():.4f}")

    print("\n3. THE MARGIN that makes one product enough")
    print(f"     xi.xi_goal  -  max_k xi.xi_distractor")
    print(f"     median {np.median(margin):.4f}   p10 {np.percentile(margin, 10):.4f}"
          f"   frac <= 0 : {(margin <= 0).mean() * 100:.2f}%")
    print("     (fraction <= 0 is where a distractor out-weighs the goal in the sum)")

    print("\n4. AFTER THE TANGENT PROJECTION -- the explore/exploit separation")
    print(f"     |q| to the goal pattern       median {np.median(qn_goal):.4f}")
    print(f"     |q| to the nearest distractor median {np.median(qn_dist):.4f}")
    print(f"     ratio {np.median(qn_goal) / max(np.median(qn_dist), 1e-9):.2f}x")
    print(f"     random-direction prediction sqrt(2/D) = {np.sqrt(2 / D):.4f} "
          "of the displacement norm")


if __name__ == "__main__":
    main()
