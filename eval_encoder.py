#!/usr/bin/env python
"""
Standalone navigation evaluation for a saved encoder checkpoint.

Usage:
    python eval_encoder.py <run_name>/encoder_final.pt [--n_envs 20] [--eval_env_size 20] ...

Loads the encoder from the checkpoint, builds the full grid encoding,
samples random (non-overlapping) eval environments, and runs the Hopfield
navigation evaluation.  No train/test split — all envs are sampled uniformly
at random from the full grid.
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

# Ensure project root on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cls.utils.GridUtils import smooth_gbook, RHCEncoder
from cls.vectorhash.assoc_utils_np_2D import gen_gbook_2d
from cls.eval.nav_eval import run_navigation_eval, _rects_overlap

# Re-use the existing load_encoder from train_dist_encoder
from notebooks.train_dist_encoder import load_encoder


# ---------------------------------------------------------------------------
# Environment sampling (no train/test split)
# ---------------------------------------------------------------------------

def sample_random_envs(grid_Nx, grid_Ny, eval_env_size, n_envs, rng,
                       max_attempts=10000):
    """Sample non-overlapping eval envs placed randomly on the full grid.

    A 1-cell border from the grid edge is enforced for Gram-Schmidt neighbors.
    Coordinates use (gx, gy) convention: dim0 = x, dim1 = y.

    Returns:
        List of (gx0, gy0) top-left corners.
    """
    margin = 1
    placements = []
    attempts = 0

    while len(placements) < n_envs and attempts < max_attempts:
        gx0 = rng.randint(margin, grid_Nx - eval_env_size - margin + 1)
        gy0 = rng.randint(margin, grid_Ny - eval_env_size - margin + 1)
        if all(not _rects_overlap(gx0, gy0, eval_env_size, ex, ey, eval_env_size)
               for ex, ey in placements):
            placements.append((gx0, gy0))
        attempts += 1

    if len(placements) < n_envs:
        print(f"WARNING: could only place {len(placements)}/{n_envs} eval envs")
    return placements


# ---------------------------------------------------------------------------
# Full-grid encoding
# ---------------------------------------------------------------------------

def encode_full_grid(encoder, config, device, batch_size=1000):
    """Encode every position on the full grid → (H, W, embed_dim) numpy array."""
    lambdas = config["model_params"]["lambdas"]
    full_Npos = int(np.prod(lambdas))
    input_type = config["model_params"].get("input_type", "smoothed")
    gain = config["model_params"]["gain"]

    parts = []
    was_training = encoder.training
    encoder.eval()
    N_total = full_Npos * full_Npos

    with torch.no_grad():
        if input_type == "smoothed":
            Ng = sum(l * l for l in lambdas)
            gbook = gen_gbook_2d(lambdas, Ng, full_Npos)
            Phi_full = torch.tensor(smooth_gbook(gbook, lambdas, 0.25),
                                    dtype=torch.float32)
            del gbook
            flat = Phi_full.reshape(Phi_full.shape[0], -1).T  # (N, code_dim)
            for i in range(0, N_total, batch_size):
                chunk = flat[i:i + batch_size].to(device)
                parts.append(encoder(chunk, gain=gain).cpu())
        elif input_type == "rhc":
            rhc_D = config["model_params"]["rhc_D"]
            rhc_encoder = RHCEncoder(lambdas, rhc_D)
            all_positions = np.mgrid[0:full_Npos, 0:full_Npos].reshape(2, -1)
            all_ys, all_xs = all_positions[0], all_positions[1]
            for i in range(0, N_total, batch_size):
                chunk = torch.tensor(
                    rhc_encoder.encode_positions(all_xs[i:i + batch_size],
                                                 all_ys[i:i + batch_size]),
                    dtype=torch.float32,
                ).to(device)
                parts.append(encoder(chunk, gain=gain).cpu())
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

    if was_training:
        encoder.train()

    return torch.cat(parts, 0).reshape(full_Npos, full_Npos, -1).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved encoder on Hopfield navigation.")
    parser.add_argument("model", type=str,
                        help="Encoder checkpoint name (relative to encoders dir), "
                             "e.g. 'cool-run-42/encoder_final.pt'")
    # nav_eval_params — same keys as in training config
    parser.add_argument("--n_envs", type=int, default=5,
                        help="Number of eval environments PER Hopfield network")
    parser.add_argument("--eval_env_size", type=int, default=20,
                        help="Side length of each square eval environment")
    parser.add_argument("--num_hopfields", type=int, default=5,
                        help="Number of Hopfield networks (envs split across them)")
    parser.add_argument("--n_starts_per_env", type=int, default=100,
                        help="Target number of start positions per env")
    parser.add_argument("--max_steps_mult", type=int, default=3,
                        help="max_steps = max_steps_mult * eval_env_size")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Step size scale")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Normalize steps to unit length (default: True)")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable step normalization")
    parser.add_argument("--platform_radius", type=float, default=1.0,
                        help="Goal-reached threshold (Euclidean)")
    parser.add_argument("--recompute_interval", type=int, default=1,
                        help="How often to recompute projection matrix W")
    parser.add_argument("--hopfield_alpha", type=float, default=0.8,
                        help="Hopfield recall alpha (mixing coefficient)")
    parser.add_argument("--save_heatmaps", action="store_true",
                        help="Save per-env heatmap PNGs")
    parser.add_argument("--heatmap_dir", type=str, default=None,
                        help="Directory for heatmap PNGs (default: next to checkpoint)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for env placement and goals")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for encoding the full grid")
    args = parser.parse_args()

    normalize = not args.no_normalize

    # --- Load encoder ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading encoder: {args.model}")
    encoder, config = load_encoder(args.model, device=device)
    print(f"  input_type={config['model_params'].get('input_type', 'smoothed')}  "
          f"encoder_type={config['model_params'].get('encoder_type', 'cnn')}  "
          f"out_dim={config['model_params']['out_dim']}  "
          f"gain={config['model_params']['gain']}")

    lambdas = config["model_params"]["lambdas"]
    full_Npos = int(np.prod(lambdas))
    gain = config["model_params"]["gain"]

    # --- Encode full grid ---
    print(f"Encoding full {full_Npos}x{full_Npos} grid ...")
    encoded_Phi = encode_full_grid(encoder, config, device,
                                   batch_size=args.batch_size)
    print(f"  encoded_Phi shape: {encoded_Phi.shape}")

    # --- Sample environments ---
    # n_envs is per-Hopfield (same convention as training config)
    n_total = args.num_hopfields * args.n_envs
    rng = np.random.RandomState(args.seed)
    placements = sample_random_envs(full_Npos, full_Npos,
                                    args.eval_env_size, n_total, rng)
    print(f"Placed {len(placements)}/{n_total} eval envs "
          f"({args.num_hopfields} Hopfields x {args.n_envs} envs/hop, "
          f"size {args.eval_env_size})")

    if not placements:
        print("No environments placed — exiting.")
        return

    # --- Run navigation eval ---
    heatmap_dir = args.heatmap_dir
    if args.save_heatmaps and heatmap_dir is None:
        # Default: put heatmaps next to the checkpoint
        encoder_dir = os.path.join("/home/jackking/cls/encoders",
                                   os.path.dirname(args.model))
        heatmap_dir = os.path.join(encoder_dir, "eval_heatmaps")

    print(f"\n--- Navigation eval: {len(placements)} envs, "
          f"{args.num_hopfields} Hopfields ---")
    results = run_navigation_eval(
        encoded_Phi, placements, args.eval_env_size, gain,
        hopfield_alpha=args.hopfield_alpha,
        n_starts_per_env=args.n_starts_per_env,
        max_steps_mult=args.max_steps_mult,
        scale=args.scale, normalize=normalize,
        platform_radius=args.platform_radius,
        recompute_interval=args.recompute_interval,
        rng=rng,
        num_hopfields=args.num_hopfields,
        save_heatmaps=args.save_heatmaps,
        heatmap_dir=heatmap_dir,
    )

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"  Model:          {args.model}")
    print(f"  Envs:           {len(placements)}")
    print(f"  Accuracy:       {results['accuracy']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"  Mean steps:     {results['mean_steps']:.1f}")
    print(f"  Mean speed:     {results['mean_speed']:.3f}")
    print(f"  Dir acc (succ): {results['mean_dir_acc_success']:.3f}")
    print(f"  Dir acc (fail): {results['mean_dir_acc_fail']:.3f}")
    print(f"  Dist (succ):    {results['mean_dist_success']:.3f}")
    print(f"  Dist (fail):    {results['mean_dist_fail']:.3f}")
    print(f"{'='*60}")

    if args.save_heatmaps:
        print(f"  Heatmaps saved to: {heatmap_dir}")


if __name__ == "__main__":
    main()
