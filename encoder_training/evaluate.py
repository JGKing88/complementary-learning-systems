"""Navigation-evaluation wrapper.

Replaces the old pearson/triplet metrics with the real nav-eval pipeline
from `cls.eval.nav_eval` — that is what we actually care about.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from cls.eval.nav_eval import (
    encode_full_grid as _encode_full_grid,
    sample_train_eval_envs,
    sample_val_eval_envs,
    run_navigation_eval,
)

from .config import NavEvalConfig


def encode_grid(encoder, Phi_full: torch.Tensor, gain: float,
                device: torch.device, batch_size: int = 1000) -> np.ndarray:
    """Encode the full grid; returns (full_Npos, full_Npos, embed_dim) np.ndarray."""
    Phi_np = Phi_full.numpy() if torch.is_tensor(Phi_full) else Phi_full
    return _encode_full_grid(encoder, Phi_np, gain, device, batch_size=batch_size)


def run_nav_eval(
    encoded: np.ndarray,
    y0s: list[int],
    x0s: list[int],
    sizes: list[int],
    full_Npos: int,
    gain: float,
    cfg: NavEvalConfig,
    rng: Optional[np.random.RandomState] = None,
    split: str = "val",
) -> dict:
    """Run navigation eval on either 'train' (inside training patches) or
    'val' (outside training patches) environments.
    """
    rng = rng or np.random.RandomState(42)
    if isinstance(sizes, list):
        max_patch = max(sizes) if sizes else 0
    else:
        max_patch = sizes
    n_total = cfg.num_hopfields * (cfg.n_train_envs if split == "train"
                                   else cfg.n_val_envs)

    if split == "train":
        placements = sample_train_eval_envs(
            y0s, x0s, sizes, cfg.env_size, n_total, rng)
    else:
        placements = sample_val_eval_envs(
            full_Npos, full_Npos, y0s, x0s, max_patch, cfg.env_size, n_total, rng)

    if not placements:
        return {"accuracy": float("nan"), "mean_steps": float("nan"),
                "mean_speed": float("nan"), "n_envs": 0}

    return run_navigation_eval(
        encoded, placements, cfg.env_size, gain,
        hopfield_alpha=cfg.hopfield_alpha,
        n_starts_per_env=cfg.n_starts_per_env,
        max_steps_mult=cfg.max_steps_mult,
        scale=cfg.scale, normalize=cfg.normalize,
        platform_radius=cfg.platform_radius,
        recompute_interval=cfg.recompute_interval,
        rng=rng, num_hopfields=cfg.num_hopfields,
    )
