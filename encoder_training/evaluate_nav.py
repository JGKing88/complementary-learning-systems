"""Standalone navigation evaluator for a trained encoder.

Usage:
    python -m encoder_training.evaluate_nav --ckpt path/to/encoder_best.pt \\
        [--env_size 20] [--n_val_envs 5] [--num_hopfields 20] [--n_starts 100]

Outputs val_nav/{accuracy, mean_steps, mean_speed}. If --train_eval is given,
also runs eval on training patches.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json

import numpy as np
import torch

from .config import NavEvalConfig
from .data import build_full_grid
from .evaluate import encode_grid, run_nav_eval
from .train import load_encoder


def main():
    p = argparse.ArgumentParser(description="Run nav eval on a trained encoder")
    p.add_argument("--ckpt", required=True, help="Path to encoder_{best,final}.pt")
    p.add_argument("--device", default="cuda")

    # Override training envs (optional): by default we use those saved in the ckpt
    p.add_argument("--train_eval", action="store_true",
                   help="Also run on training patches")

    # NavEvalConfig overrides
    p.add_argument("--env_size", type=int, default=None)
    p.add_argument("--n_train_envs", type=int, default=None)
    p.add_argument("--n_val_envs", type=int, default=None)
    p.add_argument("--num_hopfields", type=int, default=None)
    p.add_argument("--n_starts_per_env", type=int, default=None)
    p.add_argument("--platform_radius", type=float, default=None)
    p.add_argument("--max_steps_mult", type=int, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--normalize", type=int, default=None,
                   help="0 or 1 (overrides stored setting)")
    p.add_argument("--recompute_interval", type=int, default=None)
    p.add_argument("--hopfield_alpha", type=float, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true",
                   help="Emit one JSON line with results on stdout")
    args = p.parse_args()

    # Load encoder
    encoder, ckpt = load_encoder(args.ckpt, device=args.device)
    device = next(encoder.parameters()).device
    gain = float(ckpt["gain"])
    lambdas = ckpt["model_config"]["lambdas"]
    train_cfg = ckpt.get("train_config", {})
    fwhm_ratio = train_cfg.get("fwhm_ratio", 0.25) if train_cfg else 0.25

    # Recover training patches (used only as 'avoid' regions when placing val
    # envs). Older checkpoints don't save y0s/x0s/sizes — in that case use an
    # empty avoid-list so val envs are placed anywhere on the grid.
    has_patches = all(k in ckpt for k in ("y0s", "x0s", "sizes"))
    if has_patches:
        y0s = ckpt["y0s"]; x0s = ckpt["x0s"]; sizes = ckpt["sizes"]
        print(f"Loaded {args.ckpt}")
    else:
        y0s, x0s, sizes = [], [], []
        print(f"Loaded {args.ckpt}")
        print(f"  [no patches in ckpt — val envs placed anywhere on the grid]")

    print(f"  lambdas={lambdas}  gain={gain}  n_envs={len(sizes)}  "
          f"sizes={sorted(set(sizes))}")
    if "val_nav_acc" in ckpt:
        print(f"  saved peak val_nav_acc = {ckpt['val_nav_acc']:.3f}")

    # Build nav cfg — start from stored config if available
    stored = (train_cfg.get("nav_eval") if train_cfg else None) or {}
    cfg = NavEvalConfig(**stored) if stored else NavEvalConfig()
    overrides = {
        "env_size": args.env_size, "n_train_envs": args.n_train_envs,
        "n_val_envs": args.n_val_envs, "num_hopfields": args.num_hopfields,
        "n_starts_per_env": args.n_starts_per_env,
        "platform_radius": args.platform_radius,
        "max_steps_mult": args.max_steps_mult, "scale": args.scale,
        "recompute_interval": args.recompute_interval,
        "hopfield_alpha": args.hopfield_alpha,
    }
    if args.normalize is not None:
        overrides["normalize"] = bool(args.normalize)
    for k, v in overrides.items():
        if v is not None:
            cfg = replace(cfg, **{k: v})
    print(f"Nav eval config: {asdict(cfg)}")

    # Build grid codes and encode
    Phi_full, full_Npos = build_full_grid(lambdas, fwhm_ratio)
    print(f"Encoding full grid ({full_Npos}x{full_Npos})...")
    encoded = encode_grid(encoder, Phi_full, gain, device)

    results = {}
    rng = np.random.RandomState(args.seed)
    val = run_nav_eval(encoded, y0s, x0s, sizes, full_Npos, gain, cfg,
                       rng=rng, split="val")
    results["val"] = val
    print(f"Val nav: acc={val['accuracy']:.3f} | "
          f"steps={val['mean_steps']:.1f} | speed={val['mean_speed']:.3f}")

    if args.train_eval:
        rng_t = np.random.RandomState(args.seed + 1)
        tr = run_nav_eval(encoded, y0s, x0s, sizes, full_Npos, gain, cfg,
                          rng=rng_t, split="train")
        results["train"] = tr
        print(f"Train nav: acc={tr['accuracy']:.3f} | "
              f"steps={tr['mean_steps']:.1f} | speed={tr['mean_speed']:.3f}")

    if args.json:
        # Strip non-serializable fields
        clean = {split: {k: v for k, v in d.items() if not k == "figures"}
                 for split, d in results.items()}
        print("JSON:", json.dumps(clean, default=float))


if __name__ == "__main__":
    main()
