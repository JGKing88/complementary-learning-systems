from __future__ import annotations
from tkinter import N

"""Train an MLP to classify the action given (start_state, next_state).

We build a dataset by sampling random grid positions, headings, and actions
from a single `WMEnv`. For each sample we compute:
  x = concat(obs(start_pos, start_heading), obs(next_pos, next_heading))
  y = action index in {0:N, 1:E, 2:S, 3:W}

Then we train an MLP classifier (via the existing Agent with MLP backbone)
to predict the action index from x.
"""

import argparse
import os
from collections import defaultdict
from typing import List, Tuple, Optional
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from cls.envs.environments import WMEnv, GridWMEnv
from cls.models import Agent
from cls.utils.GridUtils import VectorHash, smooth_gbook, overlaps

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CARDINAL_ACTIONS: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W


def displacement_to_label(dx: int, dy: int) -> int:
    """Convert a displacement (dx, dy) to an action label.
    
    The label represents the "first step" direction based on:
    - Larger magnitude axis determines direction
    - Ties broken by clockwise priority: E > S > W > N
    
    Args:
        dx: Displacement in x (positive = East)
        dy: Displacement in y (positive = North)
        
    Returns:
        Action index: 0=N, 1=E, 2=S, 3=W
    """
    if dx == 0 and dy == 0:
        raise ValueError("Zero displacement has no direction")
    
    abs_dx, abs_dy = abs(dx), abs(dy)
    
    if abs_dx > abs_dy:
        # X-axis dominates
        return 1 if dx > 0 else 3  # East or West
    elif abs_dy > abs_dx:
        # Y-axis dominates
        return 0 if dy > 0 else 2  # North or South
    else:
        # Tie: use clockwise priority E > S > W > N
        # Determine which directions are "valid" based on signs
        # Clockwise priority order: E(1), S(2), W(3), N(0)
        if dx > 0:
            return 1  # East (highest priority when tie with North)
        elif dy < 0:
            return 2  # South (higher priority than West)
        elif dx < 0:
            return 3  # West (higher priority than North)
        else:
            return 0  # North (lowest priority, only if dy > 0 and dx == 0)


def generate_shell_displacements(min_step: int, max_step: int):
    """Generate (dx, dy) displacement pairs for a shell [min_step, max_step).
    
    A "shell" consists of all displacements where min_step <= max(|dx|, |dy|) < max_step.
    This traces the perimeter of square rings without filtering.
    
    Args:
        min_step: Minimum magnitude (inclusive)
        max_step: Maximum magnitude (exclusive)
        
    Yields:
        Tuples of (dx, dy) displacements
        
    Example:
        generate_shell_displacements(1, 2) yields single-step moves: (0,1), (1,1), (1,0), ...
        generate_shell_displacements(2, 4) yields moves with max magnitude 2 or 3
    """
    for m in range(min_step, max_step):
        if m == 0:
            continue  # Skip zero magnitude
        # Top edge: dy = m, dx from -m to m
        for dx in range(-m, m + 1):
            yield (dx, m)
        # Right edge: dx = m, dy from m-1 down to -m (excluding top corner)
        for dy in range(m - 1, -m - 1, -1):
            yield (m, dy)
        # Bottom edge: dy = -m, dx from m-1 down to -m (excluding right corner)
        for dx in range(m - 1, -m - 1, -1):
            yield (dx, -m)
        # Left edge: dx = -m, dy from -m+1 to m-1 (excluding corners)
        for dy in range(-m + 1, m):
            yield (-m, dy)


def load_grid_encoder(
    encoder_weights: str,
    lambdas: list[int] | None,
    device: str,
) -> torch.nn.Module:
    """Load a pretrained GridEncoder from checkpoint.
    
    Args:
        encoder_weights: Filename of encoder weights in encoders/ directory
        lambdas: Grid cell module periods (for computing input dimension)
        device: Device to load encoder onto
        
    Returns:
        Loaded GridEncoder model in eval mode
    """
    from cls.encoder import GridEncoderCNN
    
    encoder_path = os.path.join(os.getcwd(), "encoders", encoder_weights)
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder weights not found: {encoder_path}")
    
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    config = config["model_params"]
    
    # Get encoder architecture from checkpoint
    g_hot_dim = config.get("in_dim") or sum(l**2 for l in (config.get("lambdas")))
    hidden_dim = config.get("hidden_dim")
    hidden_channels = config.get("hidden_channels")
    num_conv_layers = config.get("num_layers")
    out_dim = config.get("out_dim")
    nonlinearity = config.get("nonlinearity", "relu")
    output_nonlinearity = config.get("output_nonlinearity", "none")
    encoder_gain = config.get("gain", 5.0)
    
    if hidden_dim is None or out_dim is None:
        raise ValueError(f"Encoder checkpoint missing 'hidden' or 'out_dim' in config: {encoder_path}")
        
    grid_encoder = GridEncoderCNN(
        lambdas=lambdas,
        hidden_channels=hidden_channels,
        hidden_dim=hidden_dim,
        num_conv_layers=num_conv_layers,
        out_dim=out_dim,
        nonlinearity=nonlinearity,
        output_nonlinearity=output_nonlinearity,
        gain=encoder_gain,
    )

    # grid_encoder = GridEncoder(
    #     in_dim=g_hot_dim,
    #     hidden=hidden,
    #     out_dim=out_dim,
    #     nonlinearity=nonlinearity,
    #     output_nonlinearity=output_nonlinearity,
    #     gain=encoder_gain,
    # )
    
    if "state_dict" in checkpoint:
        grid_encoder.load_state_dict(checkpoint["state_dict"])
    else:
        grid_encoder.load_state_dict(checkpoint)
    
    print(f"Loaded GridEncoder from {encoder_path}: {g_hot_dim} -> {hidden_dim} -> {out_dim}")
    grid_encoder.to(device)
    grid_encoder.eval()
    
    return grid_encoder


def create_environments(
    size: int,
    speed: int,
    observation_size: int,
    seed: int,
    num_train_envs: int,
    use_grid: bool,
    input_type: str | None,
    Np: int | None,
    Npos: int | None,
    lambdas: list[int] | None,
    thresh: float,
    fwhm_ratio: float,
    grid_encoder: torch.nn.Module | None,
    device: str,
    use_headings: bool,
    shared_vectorhash: bool,
) -> Tuple[List[WMEnv | GridWMEnv], WMEnv | GridWMEnv, VectorHash | None]:
    """Create training and validation environments.
    
    Returns:
        Tuple of (train_envs, val_env, vectorhash)
    """
    all_envs: List[WMEnv | GridWMEnv] = []
    train_envs: List[WMEnv | GridWMEnv] = []
    vh = None
    
    if use_grid:
        for i in range(num_train_envs):
            env_i = GridWMEnv(
                size=size, speed=speed, seed=seed + i, observation_size=observation_size,
                input_type=input_type or "g_idx",
                fwhm_ratio=fwhm_ratio,
                encoder=grid_encoder,
                use_headings=use_headings,
            )
            if grid_encoder is not None:
                env_i._encoder_device = torch.device(device)
            train_envs.append(env_i)
            all_envs.append(env_i)

            if not shared_vectorhash:
                vh_i = VectorHash(
                    Np=Np,
                    lambdas=lambdas,
                    Npos=Npos,
                    size=size,
                    thresh=thresh,
                    use_headings=use_headings,
                )
                vh_i.initiate_vectorhash([env_i])
        
        if shared_vectorhash:
            vh = VectorHash(
                Np=Np,
                lambdas=lambdas,
                Npos=Npos,
                size=size,
                thresh=thresh,
                use_headings=use_headings,
            )
        
        # Create validation environment
        env_new = GridWMEnv(
            size=size, speed=speed, seed=seed + num_train_envs + 100, observation_size=observation_size,
            input_type=input_type or "g_idx",
            fwhm_ratio=fwhm_ratio,
            encoder=grid_encoder,
            use_headings=use_headings,
        )
        all_envs.append(env_new)
        if grid_encoder is not None:
            env_new._encoder_device = torch.device(device)
        
        if not shared_vectorhash:
            vh_new = VectorHash(
                Np=Np,
                lambdas=lambdas,
                Npos=Npos,
                size=size,
                thresh=thresh,
                use_headings=use_headings,
            )
            vh_new.initiate_vectorhash([env_new])
        else:
            # Initialize VectorHash with all environments
            vh.initiate_vectorhash(all_envs)
    else:
        for i in range(num_train_envs):
            env_i = WMEnv(size=size, speed=speed, seed=seed + i, observation_size=observation_size)
            train_envs.append(env_i)
        env_new = WMEnv(size=size, speed=speed, seed=seed + 1, observation_size=observation_size)
    
    return train_envs, env_new, vh


def plot_environment_layout(
    vh: VectorHash,
    all_envs: List[WMEnv | GridWMEnv],
    num_train_envs: int,
    size: int,
    lambdas: list[int] | None,
    plot_dir: str = "displacement_plots",
) -> None:
    """Plot environment locations in global space."""
    os.makedirs(plot_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("gainsboro")
    
    # Get Npos from VectorHash
    Npos = vh.Npos if hasattr(vh, 'Npos') else max(l for l in lambdas) * 4 if lambdas else 44
    
    # Plot each environment as a colored rectangle
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_envs)))
    for i, (cx, cy) in enumerate(vh.env_locations):
        label = f"Train {i+1}" if i < num_train_envs else "Val (new)"
        rect = Rectangle(
            (cx - 0.5, cy - 0.5), size, size,
            linewidth=2, edgecolor=colors[i], facecolor=colors[i], alpha=0.3
        )
        ax.add_patch(rect)
        ax.text(cx + size/2, cy + size/2, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=colors[i])
    
    ax.set_xlim(-1, Npos + 1)
    ax.set_ylim(-1, Npos + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (global)')
    ax.set_ylabel('Y (global)')
    ax.set_title(f'Environments in global space (Npos={Npos}, size={size})')
    ax.grid(True, alpha=0.3)
    
    env_plot_path = os.path.join(plot_dir, "environments_in_space.png")
    plt.savefig(env_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved environment layout to {env_plot_path}")


def build_all_unique_samples(
    env: WMEnv | GridWMEnv,
    tuples_list: list[Tuple[Tuple[int, int], Tuple[int, int], int]],
    input_type: str | None,
    use_displacement: bool,
    env_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Build exactly one sample per (start_pos, end_pos, label) tuple.
    
    Args:
        env: Environment to generate observations from
        tuples_list: List of (start_pos, end_pos, label) tuples
        input_type: Type of input representation
        use_displacement: Whether to use displacement (next - start) or concatenation
        env_idx: Index of this environment (for metadata)
        
    Returns:
        Tuple of (X, y, metadata) where metadata is list of (env_idx, start_pos, end_pos)
    """
    use_euclidean = (input_type == "euclidean")
    
    if use_euclidean:
        F = 2  # Just (x, y)
    elif isinstance(env, GridWMEnv):
        F = env.get_input_size()
    else:
        F = env._observation_size
    
    n = len(tuples_list)
    X_dim = F if use_displacement else 2 * F
    X = np.zeros((n, X_dim), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    meta = []
    
    for i, (start_pos, end_pos, label) in enumerate(tuples_list):
        heading = (1, 0)  # Fixed heading (doesn't affect g_idx)
        # Use displacement direction as next heading
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        if dx != 0 or dy != 0:
            # Normalize to unit vector for heading
            next_heading = (1 if dx > 0 else (-1 if dx < 0 else 0),
                           1 if dy > 0 else (-1 if dy < 0 else 0))
        else:
            next_heading = heading
        
        if use_euclidean:
            start_obs = np.array([start_pos[0], start_pos[1]], dtype=np.float32)
            next_obs = np.array([end_pos[0], end_pos[1]], dtype=np.float32)
        elif isinstance(env, GridWMEnv):
            start_obs = env.convert_obs(env._code_for(start_pos, heading))
            next_obs = env.convert_obs(env._code_for(end_pos, next_heading))
        else:
            start_obs = env._code_for(start_pos, heading)
            next_obs = env._code_for(end_pos, next_heading)
        
        if use_displacement:
            X[i] = (next_obs - start_obs).astype(np.float32)
        else:
            X[i] = np.concatenate([start_obs, next_obs]).astype(np.float32)
        y[i] = label
        meta.append((env_idx, start_pos, end_pos))
    
    return X, y, meta


def build_training_data_with_scaffold(
    train_envs: List[WMEnv | GridWMEnv],
    val_env: WMEnv | GridWMEnv,
    size: int,
    val_fraction: float,
    input_type: str | None,
    use_displacement: bool,
    rng: np.random.RandomState,
    max_steps: Tuple[int, int] = (1, 2),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list, list]:
    """Build training and validation datasets from environments.
    
    Args:
        train_envs: List of training environments
        val_env: Validation environment (new env)
        size: Grid size
        val_fraction: Fraction of data to use for validation
        input_type: Type of input representation
        use_displacement: Whether to use displacement or concatenation
        rng: Random state for shuffling
        max_steps: Displacement range [min, max) - shell of valid magnitudes
    
    Returns:
        Tuple of (X_tr, y_tr, X_val, y_val, X_val_new, y_val_new, meta_tr, meta_val, meta_val_new)
    """
    num_train_envs = len(train_envs)
    min_step, max_step = max_steps
    
    # Build ALL unique (start_pos, end_pos, label) tuples
    # Uses shell generator for efficient iteration
    print(f"Building unique (start, end, label) tuples with step range [{min_step}, {max_step})...")
    check_time = time.time()
    all_tuples = []
    for sx in range(size):
        for sy in range(size):
            start_pos = (sx, sy)
            for dx, dy in generate_shell_displacements(min_step, max_step):
                ex, ey = sx + dx, sy + dy
                # Check bounds
                if 0 <= ex < size and 0 <= ey < size:
                    end_pos = (ex, ey)
                    label = displacement_to_label(dx, dy)
                    all_tuples.append((start_pos, end_pos, label))
    print(f"Total unique tuples: {len(all_tuples)}")
    print(f"Time taken to build unique tuples: {time.time() - check_time:.2f} seconds")
    check_time = time.time()
    
    # Select one environment to hold out for val_new (before building training data)
    val_new_env_idx = rng.randint(0, num_train_envs)
    val_new_env = train_envs[val_new_env_idx]
    print(f"Selected training environment {val_new_env_idx} for val_new (will be excluded from training)")
    
    print(f"Building training and validation samples...")
    # Build one sample per (env, tuple) pair
    # Each env gets its OWN random train/val split so all positions are seen in training
    X_tr_list, y_tr_list, X_val_list, y_val_list = [], [], [], []
    meta_tr_list, meta_val_list = [], []
    n_val_tuples = max(1, int(val_fraction * len(all_tuples)))
    
    for i, env_i in enumerate(train_envs):
        # Skip the environment selected for val_new
        if i == val_new_env_idx:
            continue
        # Shuffle tuples independently for each env
        env_tuples = all_tuples.copy()
        rng.shuffle(env_tuples)
        
        # Split for this env
        val_tuples_i = env_tuples[:n_val_tuples]
        train_tuples_i = env_tuples[n_val_tuples:]
        
        X_tr_i, y_tr_i, meta_tr_i = build_all_unique_samples(
            env_i, train_tuples_i, input_type, use_displacement, env_idx=i
        )
        X_val_i, y_val_i, meta_val_i = build_all_unique_samples(
            env_i, val_tuples_i, input_type, use_displacement, env_idx=i
        )
        
        X_tr_list.append(X_tr_i)
        y_tr_list.append(y_tr_i)
        meta_tr_list.extend(meta_tr_i)
        X_val_list.append(X_val_i)
        y_val_list.append(y_val_i)
        meta_val_list.extend(meta_val_i)
        
        if i == 0:
            print(f"Per-env split: {len(train_tuples_i)} train, {len(val_tuples_i)} val tuples")
    
    X_tr = np.concatenate(X_tr_list, axis=0)
    y_tr = np.concatenate(y_tr_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    
    # Shuffle training data (and metadata in sync)
    perm = rng.permutation(len(X_tr))
    X_tr, y_tr = X_tr[perm], y_tr[perm]
    meta_tr = [meta_tr_list[i] for i in perm]
    
    n_train_envs_used = num_train_envs - 1  # One environment held out for val_new
    n_train_tuples = len(all_tuples) - n_val_tuples
    print(f"Train samples: {len(X_tr)} ({n_train_envs_used} envs × ~{n_train_tuples} tuples each)")
    print(f"Val samples: {len(X_val)} ({n_train_envs_used} envs × ~{n_val_tuples} tuples each)")
    print(f"Note: Each env has DIFFERENT train/val split - all positions seen in training!")
    print(f"Time taken to build training and validation samples: {time.time() - check_time:.2f} seconds")
    check_time = time.time()

    # val_new: use the environment that was held out
    print(f"Building val_new samples...")
    X_val_new, y_val_new, meta_val_new = build_all_unique_samples(
        val_new_env, all_tuples, input_type, use_displacement, env_idx=val_new_env_idx
    )
    print(f"Built val_new samples in {time.time() - check_time:.2f} seconds")
    
    return X_tr, y_tr, X_val, y_val, X_val_new, y_val_new, meta_tr, meta_val_list, meta_val_new

def build_training_data_without_scaffold(
    size: int,
    n_envs: int,
    max_steps: Tuple[int, int],
    val_fraction: float,
    input_type: str | None,
    use_displacement: bool,
    rng: np.random.RandomState,
    Np: int,
    lambdas: list[int],
    Npos: int,
    thresh: float,
    use_headings: bool,
    grid_encoder: torch.nn.Module | None = None,
    fwhm_ratio: float = 0.0,
    num_val_new_envs: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list, list]:
    """Build training and validation datasets using shared scaffold (gbook) without creating env objects.
    
    This version uses a pre-computed gbook and places virtual environments at different offsets,
    then indexes directly into gbook for observations. More efficient than creating full env objects.
    
    Args:
        size: Grid size
        n_envs: Number of training environments to place
        max_steps: Displacement range [min, max) - shell of valid magnitudes
        val_fraction: Fraction of data to use for validation
        input_type: Type of input representation (should be "g_idx" or "encoded_g")
        use_displacement: Whether to use displacement or concatenation
        rng: Random state for shuffling
        Np: Number of place cells
        lambdas: Grid cell module periods
        Npos: Global space size
        thresh: VectorHash nonlinearity threshold
        use_headings: Whether to use heading-dependent observations
        grid_encoder: Optional encoder to apply to gbook (for encoded_g)
        fwhm_ratio: Ratio of FWHM to lambda (e.g., 0.25 means FWHM is 1/4 of lambda)
    Returns:
        Tuple of (X_tr, y_tr, X_val, y_val, X_val_new, y_val_new, meta_tr, meta_val, meta_val_new)
    """
    # Setup scaffold and get gbook
    vh = VectorHash(
        Np=Np,
        lambdas=lambdas,
        Npos=Npos,
        size=size,
        thresh=thresh,
        use_headings=use_headings,
    )
    _, _, gbook, _, _, _, _, _ = vh.setup_scaffold(Np, lambdas, thresh)

    #smooth gbook
    gbook = smooth_gbook(gbook, lambdas, fwhm_ratio)
    
    # Apply encoder if needed (gbook shape: (Ng, Npos, Npos))
    if input_type == "encoded_g":
        if grid_encoder is None:
            raise ValueError("grid_encoder required for input_type='encoded_g'")
        # Convert gbook to torch tensor, encode, convert back
        # gbook is (Ng, Npos, Npos), need to reshape for batch processing
        device = next(grid_encoder.parameters()).device
        gbook_torch = torch.from_numpy(gbook).float().to(device)  # (Ng, Npos, Npos)
        # Reshape to (Npos*Npos, Ng) for batch encoding, then reshape back
        Ng, H, W = gbook_torch.shape
        gbook_flat = gbook_torch.permute(1, 2, 0).reshape(H * W, Ng)  # (Npos*Npos, Ng)
        with torch.no_grad():
            encoded_flat = grid_encoder(gbook_flat.unsqueeze(1))  # (Npos*Npos, 1, out_dim)
            encoded_flat = encoded_flat.squeeze(1)  # (Npos*Npos, out_dim)
        encoded = encoded_flat.reshape(H, W, -1).permute(2, 0, 1)  # (out_dim, Npos, Npos)
        gbook = encoded.cpu().numpy()
    
    print(f"gbook shape: {gbook.shape}")
    idx = gbook.shape[1] // 2
    phix = gbook[:, idx, idx]

    #vectorized version
    # Compute cosine similarity between phix (F,) and gbook (F, H, W) for each (H, W)
    # We do this by taking the dot product along F for each position, and normalizing
    num = np.tensordot(phix, gbook, axes=(0, 0))  # shape: (H, W)
    denom = np.linalg.norm(phix) * np.linalg.norm(gbook, axis=0)  # shape: (H, W)
    cosine_sims = num / denom

    plt.imshow(cosine_sims)
    plt.savefig("cosine_sims.png")
    plt.close()
    
    # Get feature dimension
    F = gbook.shape[0]  # Ng or encoded_dim
    
    min_step, max_step = max_steps
    
    # Build ALL unique (start_pos, end_pos, label) tuples
    print(f"Building unique (start, end, label) tuples with step range [{min_step}, {max_step})...")
    check_time = time.time()
    all_tuples = []
    for sx in range(size):
        for sy in range(size):
            start_pos = (sx, sy)
            for dx, dy in generate_shell_displacements(min_step, max_step):
                ex, ey = sx + dx, sy + dy
                # Check bounds
                if 0 <= ex < size and 0 <= ey < size:
                    end_pos = (ex, ey)
                    label = displacement_to_label(dx, dy)
                    all_tuples.append((start_pos, end_pos, label))
    print(f"Total unique tuples: {len(all_tuples)}")
    print(f"Time taken to build unique tuples: {time.time() - check_time:.2f} seconds")
    check_time = time.time()
    
    # Place environments at non-overlapping offsets
    print(f"Placing {n_envs} environments in global space...")
    used = []  # store placed (x, y)
    C_pairs = []
    max_tries = 10_000  # guard against infinite loops
    touch_ok = True      # set False to forbid touching
    tries = 0
    while len(C_pairs) < n_envs and tries < max_tries:
        x = rng.randint(0, Npos - size + 1)
        y = rng.randint(0, Npos - size + 1)
        if all(not overlaps(x, y, px, py, size, touch_ok) for (px, py) in used):
            used.append((x, y))
            C_pairs.append((x, y))
        tries += 1

    if len(C_pairs) < n_envs:
        raise RuntimeError(f"Could only place {len(C_pairs)}/{n_envs} squares; try fewer envs or smaller size.")
    
    print(f"Placed environments at offsets: {C_pairs}")
    print(f"Time taken to place environments: {time.time() - check_time:.2f} seconds")
    check_time = time.time()
    
    # Select environments to hold out for val_new (before building training data)
    if n_envs < num_val_new_envs + 1:
        raise ValueError(f"Need at least {num_val_new_envs + 1} environments (have {n_envs}), but only {n_envs} provided")
    
    val_new_env_indices = sorted(rng.choice(n_envs, size=num_val_new_envs, replace=False))
    val_new_offsets = [C_pairs[idx] for idx in val_new_env_indices]
    print(f"Selected {num_val_new_envs} environments {val_new_env_indices} at offsets {val_new_offsets} for val_new (will be excluded from training)")
    
    # Build training and validation samples (excluding the val_new environments)
    print(f"Building training and validation samples...")
    X_tr_list, y_tr_list, X_val_list, y_val_list = [], [], [], []
    meta_tr_list, meta_val_list = [], []
    n_val_tuples = max(1, int(val_fraction * len(all_tuples)))
    
    for env_idx in range(n_envs):
        # Skip the environments selected for val_new
        if env_idx in val_new_env_indices:
            continue
        C_X, C_Y = C_pairs[env_idx]
        
        # Shuffle tuples independently for each env
        env_tuples = all_tuples.copy()
        rng.shuffle(env_tuples)
        
        # Split for this env
        val_tuples_i = env_tuples[:n_val_tuples]
        train_tuples_i = env_tuples[n_val_tuples:]
        
        # Build samples for this environment
        X_dim = F if use_displacement else 2 * F
        
        # Training samples
        n_train = len(train_tuples_i)
        X_tr_i = np.zeros((n_train, X_dim), dtype=np.float32)
        y_tr_i = np.zeros((n_train,), dtype=np.int64)
        meta_tr_i = []
        
        for t_idx, (start_pos, end_pos, label) in enumerate(train_tuples_i):
            # Convert local positions to global positions
            global_start = (start_pos[0] + C_X, start_pos[1] + C_Y)
            global_end = (end_pos[0] + C_X, end_pos[1] + C_Y)
            
            # Index into gbook (shape: (F, Npos, Npos))
            start_obs = gbook[:, global_start[0], global_start[1]]  # (F,)
            next_obs = gbook[:, global_end[0], global_end[1]]  # (F,)
            
            if use_displacement:
                X_tr_i[t_idx] = (next_obs - start_obs).astype(np.float32)
            else:
                X_tr_i[t_idx] = np.concatenate([start_obs, next_obs]).astype(np.float32)
            y_tr_i[t_idx] = label
            meta_tr_i.append((env_idx, start_pos, end_pos))
        
        # Validation samples
        n_val = len(val_tuples_i)
        X_val_i = np.zeros((n_val, X_dim), dtype=np.float32)
        y_val_i = np.zeros((n_val,), dtype=np.int64)
        meta_val_i = []
        
        for t_idx, (start_pos, end_pos, label) in enumerate(val_tuples_i):
            # Convert local positions to global positions
            global_start = (start_pos[0] + C_X, start_pos[1] + C_Y)
            global_end = (end_pos[0] + C_X, end_pos[1] + C_Y)
            
            # Index into gbook
            start_obs = gbook[:, global_start[0], global_start[1]]
            next_obs = gbook[:, global_end[0], global_end[1]]
            
            if use_displacement:
                X_val_i[t_idx] = (next_obs - start_obs).astype(np.float32)
            else:
                X_val_i[t_idx] = np.concatenate([start_obs, next_obs]).astype(np.float32)
            y_val_i[t_idx] = label
            meta_val_i.append((env_idx, start_pos, end_pos))
        
        X_tr_list.append(X_tr_i)
        y_tr_list.append(y_tr_i)
        meta_tr_list.extend(meta_tr_i)
        X_val_list.append(X_val_i)
        y_val_list.append(y_val_i)
        meta_val_list.extend(meta_val_i)
        
        if env_idx == 0:
            print(f"Per-env split: {len(train_tuples_i)} train, {len(val_tuples_i)} val tuples")
    
    X_tr = np.concatenate(X_tr_list, axis=0)
    y_tr = np.concatenate(y_tr_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    
    # Shuffle training data (and metadata in sync)
    perm = rng.permutation(len(X_tr))
    X_tr, y_tr = X_tr[perm], y_tr[perm]
    meta_tr = [meta_tr_list[i] for i in perm]
    
    n_train_envs_used = n_envs - num_val_new_envs  # 4 environments held out for val_new
    n_train_tuples = len(all_tuples) - n_val_tuples
    print(f"Train samples: {len(X_tr)} ({n_train_envs_used} envs × ~{n_train_tuples} tuples each)")
    print(f"Val samples: {len(X_val)} ({n_train_envs_used} envs × ~{n_val_tuples} tuples each)")
    print(f"Time taken to build training and validation samples: {time.time() - check_time:.2f} seconds")
    check_time = time.time()
    
    # val_new: use all 4 environments that were held out
    print(f"Building val_new samples from {num_val_new_envs} held-out environments...")
    n_val_new = len(all_tuples) * num_val_new_envs  # One set of tuples per val_new environment
    X_dim = F if use_displacement else 2 * F
    X_val_new = np.zeros((n_val_new, X_dim), dtype=np.float32)
    y_val_new = np.zeros((n_val_new,), dtype=np.int64)
    
    meta_val_new = []
    sample_idx = 0
    for val_env_idx in val_new_env_indices:
        C_X_val, C_Y_val = C_pairs[val_env_idx]
        for t_idx, (start_pos, end_pos, label) in enumerate(all_tuples):
            global_start = (start_pos[0] + C_X_val, start_pos[1] + C_Y_val)
            global_end = (end_pos[0] + C_X_val, end_pos[1] + C_Y_val)
            
            start_obs = gbook[:, global_start[0], global_start[1]]
            next_obs = gbook[:, global_end[0], global_end[1]]
            
            if use_displacement:
                X_val_new[sample_idx] = (next_obs - start_obs).astype(np.float32)
            else:
                X_val_new[sample_idx] = np.concatenate([start_obs, next_obs]).astype(np.float32)
            y_val_new[sample_idx] = label
            meta_val_new.append((val_env_idx, start_pos, end_pos))
            sample_idx += 1
    
    print(f"Built val_new samples in {time.time() - check_time:.2f} seconds")
    
    return X_tr, y_tr, X_val, y_val, X_val_new, y_val_new, meta_tr, meta_val_list, meta_val_new


def build_training_data_global_space(
    size: int,
    max_steps: Tuple[int, int],
    val_fraction: float,
    input_type: str | None,
    use_displacement: bool,
    rng: np.random.RandomState,
    Np: int,
    lambdas: list[int],
    Npos: int,
    thresh: float,
    use_headings: bool,
    grid_encoder: torch.nn.Module | None = None,
    fwhm_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list, list]:
    """Build train/val data by sampling transitions from the entire global space (no environments).

    Conceptually this treats the whole Npos×Npos grid as a single environment. We still create a
    train/val split over (start,end,label) pairs, but there is no separate val_new set.
    """
    # Setup scaffold and get gbook (features over global space)
    vh = VectorHash(
        Np=Np,
        lambdas=lambdas,
        Npos=Npos,
        size=size,
        thresh=thresh,
        use_headings=use_headings,
    )
    _, _, gbook, _, _, _, _, _ = vh.setup_scaffold(Np, lambdas, thresh)

    gbook = smooth_gbook(gbook, lambdas, fwhm_ratio)

    if input_type == "encoded_g":
        if grid_encoder is None:
            raise ValueError("grid_encoder required for input_type='encoded_g'")
        device = next(grid_encoder.parameters()).device
        gbook_torch = torch.from_numpy(gbook).float().to(device)  # (Ng, Npos, Npos)
        Ng, H, W = gbook_torch.shape
        gbook_flat = gbook_torch.permute(1, 2, 0).reshape(H * W, Ng)  # (Npos*Npos, Ng)
        with torch.no_grad():
            encoded_flat = grid_encoder(gbook_flat.unsqueeze(1)).squeeze(1)  # (Npos*Npos, out_dim)
        encoded = encoded_flat.reshape(H, W, -1).permute(2, 0, 1)  # (out_dim, Npos, Npos)
        gbook = encoded.cpu().numpy()

    F = gbook.shape[0]
    X_dim = F if use_displacement else 2 * F

    min_step, max_step = max_steps

    # Build all valid global tuples (start,end,label) across Npos×Npos
    print(f"Building global (start, end, label) tuples over Npos={Npos} with step range [{min_step}, {max_step})...")
    check_time = time.time()
    all_tuples: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    disps = list(generate_shell_displacements(min_step, max_step))
    for sx in range(Npos):
        for sy in range(Npos):
            for dx, dy in disps:
                ex, ey = sx + dx, sy + dy
                if 0 <= ex < Npos and 0 <= ey < Npos:
                    all_tuples.append(((sx, sy), (ex, ey), displacement_to_label(dx, dy)))
    print(f"Total global tuples: {len(all_tuples)}")
    print(f"Time taken to build global tuples: {time.time() - check_time:.2f} seconds")

    rng.shuffle(all_tuples)
    n_val = max(1, int(val_fraction * len(all_tuples)))
    val_tuples = all_tuples[:n_val]
    train_tuples = all_tuples[n_val:]

    # Build train
    X_tr = np.zeros((len(train_tuples), X_dim), dtype=np.float32)
    y_tr = np.zeros((len(train_tuples),), dtype=np.int64)
    meta_tr: list = []
    for i, (start_pos, end_pos, label) in enumerate(train_tuples):
        start_obs = gbook[:, start_pos[0], start_pos[1]]
        next_obs = gbook[:, end_pos[0], end_pos[1]]
        if use_displacement:
            X_tr[i] = (next_obs - start_obs).astype(np.float32)
        else:
            X_tr[i] = np.concatenate([start_obs, next_obs]).astype(np.float32)
        y_tr[i] = label
        meta_tr.append((0, start_pos, end_pos))

    # Build val
    X_val = np.zeros((len(val_tuples), X_dim), dtype=np.float32)
    y_val = np.zeros((len(val_tuples),), dtype=np.int64)
    meta_val: list = []
    for i, (start_pos, end_pos, label) in enumerate(val_tuples):
        start_obs = gbook[:, start_pos[0], start_pos[1]]
        next_obs = gbook[:, end_pos[0], end_pos[1]]
        if use_displacement:
            X_val[i] = (next_obs - start_obs).astype(np.float32)
        else:
            X_val[i] = np.concatenate([start_obs, next_obs]).astype(np.float32)
        y_val[i] = label
        meta_val.append((0, start_pos, end_pos))

    # No val_new in this mode
    X_val_new = np.zeros((0, X_dim), dtype=np.float32)
    y_val_new = np.zeros((0,), dtype=np.int64)
    meta_val_new: list = []

    # Shuffle training (meta in sync)
    perm = rng.permutation(len(X_tr))
    X_tr, y_tr = X_tr[perm], y_tr[perm]
    meta_tr = [meta_tr[j] for j in perm]

    return X_tr, y_tr, X_val, y_val, X_val_new, y_val_new, meta_tr, meta_val, meta_val_new

def verify_label_correctness(
    y: np.ndarray,
    meta: list,
    dataset_name: str = "dataset",
    sample_size: int = 1000,
) -> None:
    """Verify that labels match the displacements from metadata.
    
    Args:
        y: Label array
        meta: Metadata list of (env_idx, start_pos, end_pos) tuples
        dataset_name: Name of dataset for error messages
        sample_size: Number of samples to check (0 = check all)
    """
    n_check = min(sample_size, len(y)) if sample_size > 0 else len(y)
    errors = []
    
    for i in range(n_check):
        env_idx, start_pos, end_pos = meta[i]
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        expected_label = displacement_to_label(dx, dy)
        actual_label = y[i]
        
        if actual_label != expected_label:
            errors.append((i, start_pos, end_pos, dx, dy, actual_label, expected_label))
    
    if errors:
        print(f"❌ Label correctness check FAILED for {dataset_name}:")
        print(f"   Found {len(errors)} errors out of {n_check} checked samples")
        print(f"   First few errors:")
        for i, start_pos, end_pos, dx, dy, actual, expected in errors[:5]:
            print(f"     Sample {i}: start={start_pos}, end={end_pos}, dx={dx}, dy={dy}")
            print(f"       Got label {actual} ({['N','E','S','W'][actual]}), expected {expected} ({['N','E','S','W'][expected]})")
        raise ValueError(f"Label correctness check failed: {len(errors)}/{n_check} samples have incorrect labels")
    else:
        print(f"✓ Label correctness check passed for {dataset_name} ({n_check} samples checked)")


def verify_data_leakage(
    meta_tr: list,
    meta_val_new: list,
    dataset_name: str = "datasets",
) -> None:
    """Verify that val_new environments are not in training data.
    
    Args:
        meta_tr: Training metadata list of (env_idx, start_pos, end_pos) tuples
        meta_val_new: Validation new metadata list of (env_idx, start_pos, end_pos) tuples
        dataset_name: Name for error messages
    """
    if not meta_val_new:
        print("⚠ Warning: meta_val_new is empty, skipping data leakage check")
        return
    
    # Get environment indices from training
    train_env_indices = set(meta_tr[i][0] for i in range(len(meta_tr)))
    
    # Get environment indices from val_new
    val_new_env_indices = set(meta_val_new[i][0] for i in range(len(meta_val_new)))
    
    # Check for overlap
    overlap = val_new_env_indices & train_env_indices
    
    if overlap:
        print(f"❌ Data leakage check FAILED for {dataset_name}:")
        print(f"   val_new environments {sorted(overlap)} appear in training data!")
        print(f"   Training environments: {sorted(train_env_indices)}")
        print(f"   Val_new environments: {sorted(val_new_env_indices)}")
        raise ValueError(f"Data leakage detected: val_new envs {sorted(overlap)} are in training")
    else:
        print(f"✓ Data leakage check passed for {dataset_name}")
        print(f"   Training uses {len(train_env_indices)} environments: {sorted(train_env_indices)}")
        print(f"   Val_new uses {len(val_new_env_indices)} environments: {sorted(val_new_env_indices)}")


def print_data_diagnostics(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    use_displacement: bool,
) -> None:
    """Print diagnostic information about training data."""
    print(f"X_tr shape: {X_tr.shape}, y_tr shape: {y_tr.shape}")
    print(f"X_tr stats: min={X_tr.min():.4f}, max={X_tr.max():.4f}, mean={X_tr.mean():.4f}, std={X_tr.std():.4f}")
    print(f"y_tr label distribution: {np.bincount(y_tr, minlength=4)}")
    print(f"X_tr first sample: {X_tr[0]}")
    print(f"X_tr has NaN: {np.isnan(X_tr).any()}, has Inf: {np.isinf(X_tr).any()}")
    
    # Check if all inputs are identical
    unique_rows = len(np.unique(X_tr, axis=0))
    print(f"Unique input rows: {unique_rows}/{len(X_tr)}")
    
    # Show mean displacement per action (if using displacement)
    if use_displacement:
        print("Mean displacement per action:")
        for a in range(4):
            mask = y_tr == a
            mean_disp = X_tr[mask].mean(axis=0)
            print(f"  Action {a} ({['N','E','S','W'][a]}): norm={np.linalg.norm(mean_disp):.4f}, sum={mean_disp.sum():.4f}")
    
    # Check for label ambiguity
    disp_to_labels = defaultdict(set)
    for x, y_label in zip(X_tr, y_tr):
        key = tuple(np.round(x, 6))
        disp_to_labels[key].add(y_label)
    ambiguous = sum(1 for labels in disp_to_labels.values() if len(labels) > 1)
    print(f"Label ambiguity: {ambiguous}/{len(disp_to_labels)} unique inputs map to multiple labels")


def plot_displacement_examples(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    meta_tr: list,
    lambdas: list[int] | None,
    use_displacement: bool,
    env_offsets: list[Tuple[int, int]],
    plot_dir: str = "displacement_plots",
    n_examples: int = 3,
) -> None:
    """Plot example displacement/observation patterns for each action."""
    os.makedirs(plot_dir, exist_ok=True)
    
    action_names = ['North', 'East', 'South', 'West']
    dim = X_tr.shape[1]
    ls = lambdas if lambdas is not None else [11, 12]
    expected_single = sum(l**2 for l in ls)
    
    print("\n=== Plotted example locations ===")
    
    if dim == expected_single:  # Displacement mode
        n_modules = len(ls)
        fig, axes = plt.subplots(4, n_examples * n_modules, figsize=(3*n_examples*n_modules, 14))
        
        for a_idx in range(4):
            mask = y_tr == a_idx
            mask_indices = np.where(mask)[0]
            example_indices = mask_indices[:n_examples]
            examples = X_tr[mask][:n_examples]
            
            for ex_idx, (global_idx, ex) in enumerate(zip(example_indices, examples)):
                env_idx, start_pos, next_pos = meta_tr[global_idx]
                offset = env_offsets[env_idx] if env_idx < len(env_offsets) else (0, 0)
                global_start = (start_pos[0] + offset[0], start_pos[1] + offset[1])
                global_next = (next_pos[0] + offset[0], next_pos[1] + offset[1])
                print(f"  {action_names[a_idx]} {ex_idx+1}: env{env_idx}, local {start_pos}→{next_pos}, global {global_start}→{global_next}")
                
                vmax = max(abs(ex.min()), abs(ex.max()), 0.1)
                start_idx = 0
                for m_idx, l in enumerate(ls):
                    col = ex_idx * n_modules + m_idx
                    ax = axes[a_idx, col]
                    
                    module_data = ex[start_idx:start_idx + l*l].reshape(l, l)
                    start_idx += l*l
                    
                    ax.imshow(module_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
                    if m_idx == 0:
                        ax.set_title(f"{action_names[a_idx]} {ex_idx+1} (λ={l})\nenv{env_idx}: {start_pos}→{next_pos}\nglobal: {global_start}→{global_next}", fontsize=7)
                    else:
                        ax.set_title(f"λ={l}", fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.suptitle("Displacement per module (red=+, blue=-)", fontsize=12)
        
    elif dim == 2 * expected_single:  # Concatenated (start, next)
        n_modules = len(ls)
        fig, axes = plt.subplots(4, n_examples * n_modules * 2, figsize=(2.5*n_examples*n_modules*2, 14))
        
        for a_idx in range(4):
            mask = y_tr == a_idx
            mask_indices = np.where(mask)[0]
            example_indices = mask_indices[:n_examples]
            examples = X_tr[mask][:n_examples]
            
            for ex_idx, (global_idx, ex) in enumerate(zip(example_indices, examples)):
                env_idx, start_pos, next_pos = meta_tr[global_idx]
                offset = env_offsets[env_idx] if env_idx < len(env_offsets) else (0, 0)
                global_start = (start_pos[0] + offset[0], start_pos[1] + offset[1])
                global_next = (next_pos[0] + offset[0], next_pos[1] + offset[1])
                print(f"  {action_names[a_idx]} {ex_idx+1}: env{env_idx}, local {start_pos}→{next_pos}, global {global_start}→{global_next}")
                
                start_obs = ex[:expected_single]
                next_obs = ex[expected_single:]
                vmax = max(abs(ex.min()), abs(ex.max()), 0.1)
                
                start_idx = 0
                for m_idx, l in enumerate(ls):
                    col_s = ex_idx * n_modules * 2 + m_idx * 2
                    ax_s = axes[a_idx, col_s]
                    mod_s = start_obs[start_idx:start_idx + l*l].reshape(l, l)
                    ax_s.imshow(mod_s, cmap='hot', vmin=0, vmax=vmax, aspect='equal')
                    if m_idx == 0:
                        ax_s.set_title(f"{action_names[a_idx]} {ex_idx+1} start (λ={l})\nenv{env_idx}: {start_pos}→{next_pos}\nglobal: {global_start}→{global_next}", fontsize=6)
                    else:
                        ax_s.set_title(f"start (λ={l})", fontsize=8)
                    ax_s.set_xticks([])
                    ax_s.set_yticks([])
                    
                    col_n = col_s + 1
                    ax_n = axes[a_idx, col_n]
                    mod_n = next_obs[start_idx:start_idx + l*l].reshape(l, l)
                    ax_n.imshow(mod_n, cmap='hot', vmin=0, vmax=vmax, aspect='equal')
                    ax_n.set_title(f"next (λ={l})", fontsize=8)
                    ax_n.set_xticks([])
                    ax_n.set_yticks([])
                    
                    start_idx += l*l
        
        plt.suptitle("Start and Next observations per module", fontsize=12)
        
    else:
        # Fallback: 1D bar plot
        fig, axes = plt.subplots(4, n_examples, figsize=(4*n_examples, 12))
        for a_idx in range(4):
            mask = y_tr == a_idx
            examples = X_tr[mask][:n_examples]
            for ex_idx, ex in enumerate(examples):
                ax = axes[a_idx, ex_idx]
                ax.bar(range(len(ex)), ex, width=1.0)
                ax.set_title(f"{action_names[a_idx]} #{ex_idx+1}")
                ax.set_ylim(-1.5, 1.5)
                if ex_idx == 0:
                    ax.set_ylabel(action_names[a_idx])
        plt.suptitle("1D representation", fontsize=12)
    
    print("=================================\n")
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "displacement_examples.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved displacement examples to {plot_path}")


def create_model(
    input_size: int,
    hidden_size: int,
    num_model_layers: int,
    dropout: float,
    model_type: str,
    input_type: str | None,
    lambdas: list[int] | None,
    use_displacement: bool,
    device: str,
) -> Tuple[Agent, str]:
    """Create and configure the Agent model.
    
    Returns:
        Tuple of (model, actual_model_type)
    """
    print(f"Model input_size: {input_size}")
    print(f"Model type: {model_type}")
    
    # CNN only works with g_hot
    actual_model_type = model_type
    if model_type == "cnn" and input_type != "g_hot":
        print(f"Warning: CNN only works with g_hot, but input_type={input_type}. Falling back to MLP.")
        actual_model_type = "mlp"
    
    model = Agent(
        input_size=input_size,
        hidden_size=hidden_size,
        num_model_layers=num_model_layers,
        num_actions=4,
        dropout=dropout,
        model_class=actual_model_type.upper(),
        encoder_dim=None,
        num_encoder_layers=0,
        lambdas=lambdas if actual_model_type == "cnn" else None,
    )
    
    if actual_model_type == "cnn":
        n_channels = 2 * len(lambdas) if not use_displacement else len(lambdas)
        print(f"Agent+CNN: {n_channels} channels ({'start+next' if not use_displacement else 'displacement'}), max_λ={max(lambdas)}, hidden={hidden_size}")
    else:
        print(f"Agent+MLP: input_dim={input_size}, hidden={hidden_size}, layers={num_model_layers}")
    
    model.to(device)
    return model, actual_model_type


def run_epoch(
    model: Agent,
    X: np.ndarray,
    y: np.ndarray,
    train: bool,
    optimizer: optim.Optimizer | None,
    criterion: nn.Module,
    batch_size: int,
    device: str,
) -> Tuple[float, float]:
    """Run one epoch of training or evaluation.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train(mode=train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    
    for start in range(0, X.shape[0], batch_size):
        xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]
        t = torch.from_numpy(yb).to(device).long()
        
        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        
        # Agent expects (batch, seq, dim) input, returns (logits, values, h_next)
        x = torch.from_numpy(xb).to(device).float().view(xb.shape[0], 1, -1)
        logits, values, _ = model(x, None)
        logits = logits[:, 0, :]  # (batch, num_actions)
        
        loss = criterion(logits, t)
        if train and optimizer is not None:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            total_correct += int((pred == t).sum().item())
            total += int(t.numel())
            total_loss += float(loss.item()) * int(t.numel())
    
    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc


def evaluate_detailed(
    model: Agent,
    X: np.ndarray,
    y_true: np.ndarray,
    meta: list,
    batch_size: int,
    device: str,
    size: int,
) -> dict:
    """Evaluate model accuracy broken down by starting position, direction, distance, and interactions.
    
    Args:
        model: Trained Agent model
        X: Input features (n_samples, feature_dim)
        y_true: True labels (n_samples,)
        meta: Metadata list of (env_idx, start_pos, end_pos) tuples
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        size: Grid size (for position analysis)
        
    Returns:
        Dictionary containing:
        - overall_accuracy: Overall accuracy
        - by_start_pos: Dict mapping (x, y) -> accuracy
        - by_direction: Dict mapping direction (0-3) -> accuracy
        - by_distance: Dict mapping distance -> accuracy
        - by_position_direction: Dict mapping (x, y, direction) -> accuracy
        - by_direction_distance: Dict mapping (direction, distance) -> accuracy
        - by_position_distance: Dict mapping (x, y, distance) -> accuracy
        - by_position_direction_distance: Dict mapping (x, y, direction, distance) -> accuracy
        - counts: Sample counts for each breakdown
    """
    model.eval()
    action_names = ['N', 'E', 'S', 'W']
    
    # Get predictions
    predictions = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = X[start : start + batch_size]
            x = torch.from_numpy(xb).to(device).float().view(xb.shape[0], 1, -1)
            logits, _, _ = model(x, None)
            logits = logits[:, 0, :]  # (batch, num_actions)
            pred = logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(pred)
    
    predictions = np.array(predictions)
    
    # Compute displacement and distance for each sample
    distances = []
    for env_idx, start_pos, end_pos in meta:
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        # Chebyshev distance (max of absolute displacements)
        distance = max(abs(dx), abs(dy))
        distances.append(distance)
    
    distances = np.array(distances)
    
    # Initialize breakdown dictionaries
    by_start_pos = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_direction = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_distance = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_position_direction = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_direction_distance = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_position_distance = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_position_direction_distance = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Compute accuracy for each sample
    correct = 0
    total = 0
    
    for i in range(len(X)):
        pred = predictions[i]
        true_label = y_true[i]
        env_idx, start_pos, end_pos = meta[i]
        distance = distances[i]
        direction = true_label  # Direction is the label itself
        
        is_correct = (pred == true_label)
        if is_correct:
            correct += 1
        total += 1
        
        # By starting position
        key_pos = tuple(start_pos)
        by_start_pos[key_pos]['correct'] += is_correct
        by_start_pos[key_pos]['total'] += 1
        
        # By direction
        by_direction[direction]['correct'] += is_correct
        by_direction[direction]['total'] += 1
        
        # By distance
        by_distance[distance]['correct'] += is_correct
        by_distance[distance]['total'] += 1
        
        # By position × direction
        key_pos_dir = (start_pos[0], start_pos[1], direction)
        by_position_direction[key_pos_dir]['correct'] += is_correct
        by_position_direction[key_pos_dir]['total'] += 1
        
        # By direction × distance
        key_dir_dist = (direction, distance)
        by_direction_distance[key_dir_dist]['correct'] += is_correct
        by_direction_distance[key_dir_dist]['total'] += 1
        
        # By position × distance
        key_pos_dist = (start_pos[0], start_pos[1], distance)
        by_position_distance[key_pos_dist]['correct'] += is_correct
        by_position_distance[key_pos_dist]['total'] += 1
        
        # By position × direction × distance
        key_pos_dir_dist = (start_pos[0], start_pos[1], direction, distance)
        by_position_direction_distance[key_pos_dir_dist]['correct'] += is_correct
        by_position_direction_distance[key_pos_dir_dist]['total'] += 1
    
    # Convert to accuracy dictionaries
    def to_accuracy(d):
        return {k: d[k]['correct'] / max(1, d[k]['total']) for k in d}
    
    def to_counts(d):
        return {k: d[k]['total'] for k in d}
    
    results = {
        'overall_accuracy': correct / max(1, total),
        'by_start_pos': to_accuracy(by_start_pos),
        'by_direction': to_accuracy(by_direction),
        'by_distance': to_accuracy(by_distance),
        'by_position_direction': to_accuracy(by_position_direction),
        'by_direction_distance': to_accuracy(by_direction_distance),
        'by_position_distance': to_accuracy(by_position_distance),
        'by_position_direction_distance': to_accuracy(by_position_direction_distance),
        'counts': {
            'by_start_pos': to_counts(by_start_pos),
            'by_direction': to_counts(by_direction),
            'by_distance': to_counts(by_distance),
            'by_position_direction': to_counts(by_position_direction),
            'by_direction_distance': to_counts(by_direction_distance),
            'by_position_distance': to_counts(by_position_distance),
            'by_position_direction_distance': to_counts(by_position_direction_distance),
        }
    }
    
    return results


def print_evaluation_report(
    results: dict,
    dataset_name: str = "Evaluation",
) -> None:
    """Print a formatted evaluation report.
    
    Args:
        results: Results dictionary from evaluate_detailed
        dataset_name: Name of the dataset being evaluated
    """
    action_names = ['N', 'E', 'S', 'W']
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} - Detailed Accuracy Breakdown")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    
    # By direction
    print(f"\n--- By Direction ---")
    for direction in sorted(results['by_direction'].keys()):
        acc = results['by_direction'][direction]
        count = results['counts']['by_direction'][direction]
        print(f"  {action_names[direction]}: {acc:.4f} ({count} samples)")
    
    # By distance
    print(f"\n--- By Distance ---")
    for distance in sorted(results['by_distance'].keys()):
        acc = results['by_distance'][distance]
        count = results['counts']['by_distance'][distance]
        print(f"  Distance {distance}: {acc:.4f} ({count} samples)")
    
    # By starting position (show as grid)
    print(f"\n--- By Starting Position (Grid) ---")
    start_pos_acc = results['by_start_pos']
    start_pos_counts = results['counts']['by_start_pos']
    
    # Find grid bounds
    if start_pos_acc:
        max_x = max(pos[0] for pos in start_pos_acc.keys())
        max_y = max(pos[1] for pos in start_pos_acc.keys())
        
        print("  Accuracy grid (rows=y, cols=x):")
        for y in range(max_y + 1):
            row = []
            for x in range(max_x + 1):
                key = (x, y)
                if key in start_pos_acc:
                    acc = start_pos_acc[key]
                    count = start_pos_counts[key]
                    row.append(f"{acc:.3f}({count})")
                else:
                    row.append("  -  ")
            print(f"    y={y:2d}: {'  '.join(row)}")
    
    # By direction × distance
    print(f"\n--- By Direction × Distance ---")
    dir_dist_acc = results['by_direction_distance']
    dir_dist_counts = results['counts']['by_direction_distance']
    
    for (direction, distance) in sorted(dir_dist_acc.keys()):
        acc = dir_dist_acc[(direction, distance)]
        count = dir_dist_counts[(direction, distance)]
        print(f"  {action_names[direction]} @ dist {distance}: {acc:.4f} ({count} samples)")
    
    # By position × direction (summary statistics)
    print(f"\n--- By Position × Direction (Summary) ---")
    pos_dir_acc = results['by_position_direction']
    pos_dir_counts = results['counts']['by_position_direction']
    
    # Group by position, show direction breakdown
    pos_to_dirs = defaultdict(list)
    for (x, y, direction) in pos_dir_acc.keys():
        pos_to_dirs[(x, y)].append(direction)
    
    # Show a few example positions
    example_positions = sorted(pos_to_dirs.keys())[:5]
    for pos in example_positions:
        print(f"  Position {pos}:")
        for direction in sorted(pos_to_dirs[pos]):
            key = (pos[0], pos[1], direction)
            acc = pos_dir_acc[key]
            count = pos_dir_counts[key]
            print(f"    {action_names[direction]}: {acc:.4f} ({count} samples)")
    
    # Statistics
    print(f"\n--- Statistics ---")
    all_accs = list(results['by_start_pos'].values())
    if all_accs:
        print(f"  Position accuracy: min={min(all_accs):.4f}, max={max(all_accs):.4f}, mean={np.mean(all_accs):.4f}, std={np.std(all_accs):.4f}")
    
    all_dir_accs = list(results['by_direction'].values())
    if all_dir_accs:
        print(f"  Direction accuracy: min={min(all_dir_accs):.4f}, max={max(all_dir_accs):.4f}, mean={np.mean(all_dir_accs):.4f}, std={np.std(all_dir_accs):.4f}")
    
    all_dist_accs = list(results['by_distance'].values())
    if all_dist_accs:
        print(f"  Distance accuracy: min={min(all_dist_accs):.4f}, max={max(all_dist_accs):.4f}, mean={np.mean(all_dist_accs):.4f}, std={np.std(all_dist_accs):.4f}")
    
    print(f"{'='*60}\n")


def train_loop(
    model: Agent,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_val_new: np.ndarray,
    y_val_new: np.ndarray,
    meta_val: list,
    meta_val_new: list,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    use_wandb: bool,
    size: int,
    eval_every: int = 20,
    regenerate_envs_per_epoch: bool = False,
    data_generator: callable = None,
    scheduler_type: str = "none",
    scheduler_step_size: int = 30,
    scheduler_gamma: float = 0.1,
    scheduler_T_max: int = 100,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    warmup_epochs: int = 0,
    warmup_start_lr: float = 0.0,
    save_every: int = 0,
    save_dir: str = "checkpoints",
) -> None:
    """Run the training loop.
    
    Args:
        regenerate_envs_per_epoch: If True, regenerate environment locations and data before each epoch
        data_generator: Function that returns (X_tr, y_tr, X_val, y_val, X_val_new, y_val_new, meta_val, meta_val_new)
                        Called with epoch number as argument. Only used if regenerate_envs_per_epoch=True.
        scheduler_type: Type of learning rate scheduler ("none", "step", "cosine", "plateau")
        scheduler_step_size: Step size for StepLR (epochs between LR reductions)
        scheduler_gamma: Multiplicative factor for StepLR
        scheduler_T_max: Maximum iterations for CosineAnnealingLR
        scheduler_patience: Patience for ReduceLROnPlateau
        scheduler_factor: Factor for ReduceLROnPlateau
        warmup_epochs: Number of epochs for linear warmup (0 = no warmup)
        warmup_start_lr: Starting learning rate for warmup (default: 0.0)
        save_every: Save model checkpoint every N epochs (0 = no saving during training, only final)
        save_dir: Directory to save checkpoints
    """
    print(f"Starting training loop with {n_epochs} epochs...")
    if regenerate_envs_per_epoch:
        print("Environment locations will be regenerated before each epoch")
    
    # Setup save directory
    if save_every > 0 or save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Model checkpoints will be saved to {save_dir}/")
        if save_every > 0:
            print(f"  Saving every {save_every} epochs")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Create learning rate scheduler
    scheduler = None
    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        print(f"Using StepLR scheduler: step_size={scheduler_step_size}, gamma={scheduler_gamma}")
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_T_max)
        print(f"Using CosineAnnealingLR scheduler: T_max={scheduler_T_max}")
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )
        print(f"Using ReduceLROnPlateau scheduler: factor={scheduler_factor}, patience={scheduler_patience}")
    elif scheduler_type != "none":
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Setup warmup
    if warmup_epochs > 0:
        print(f"Using linear warmup: {warmup_epochs} epochs from {warmup_start_lr:.6f} to {lr:.6f}")
        # Set initial LR to warmup_start_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_start_lr
        warmup_step = (lr - warmup_start_lr) / warmup_epochs
    else:
        warmup_step = 0.0
    
    # Store initial data (will be overwritten each epoch if regenerate_envs_per_epoch=True)
    current_X_tr, current_y_tr = X_tr, y_tr
    current_X_val, current_y_val = X_val, y_val
    current_X_val_new, current_y_val_new = X_val_new, y_val_new
    current_meta_val, current_meta_val_new = meta_val, meta_val_new
    
    for epoch in range(1, n_epochs + 1):
        # Regenerate data if flag is set
        if regenerate_envs_per_epoch and data_generator is not None:
            print(f"Regenerating environment locations for epoch {epoch}...")
            current_X_tr, current_y_tr, current_X_val, current_y_val, current_X_val_new, current_y_val_new, _, current_meta_val, current_meta_val_new = data_generator(epoch)
        
        tr_loss, tr_acc = run_epoch(
            model, current_X_tr, current_y_tr, train=True,
            optimizer=optimizer, criterion=criterion, batch_size=batch_size, device=device
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model, current_X_val, current_y_val, train=False,
                optimizer=None, criterion=criterion, batch_size=batch_size, device=device
            )
            has_val_new = (current_X_val_new is not None) and (len(current_X_val_new) > 0)
            if has_val_new:
                val_new_loss, val_new_acc = run_epoch(
                    model, current_X_val_new, current_y_val_new, train=False,
                    optimizer=None, criterion=criterion, batch_size=batch_size, device=device
                )
            else:
                val_new_loss, val_new_acc = float("nan"), float("nan")
        
        # Handle warmup and scheduler
        current_lr = optimizer.param_groups[0]['lr']
        
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            # Linear warmup: gradually increase LR from warmup_start_lr to target lr
            new_lr = warmup_start_lr + epoch * warmup_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            if epoch == warmup_epochs:
                print(f"  Warmup complete: LR = {new_lr:.6f}")
                # Ensure we're at target LR (should be, but double-check)
                for param_group in optimizer.param_groups:
                    if abs(param_group['lr'] - lr) > 1e-6:
                        param_group['lr'] = lr
        elif scheduler is not None:
            # Apply regular scheduler after warmup
            if scheduler_type == "plateau":
                scheduler.step(val_loss)  # ReduceLROnPlateau steps on metric
            else:
                scheduler.step()  # Other schedulers step on epoch
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"  Learning rate changed: {current_lr:.6f} -> {new_lr:.6f}")
        else:
            new_lr = current_lr
        
        print(
            f"epoch {epoch:04d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            + f"val {val_loss:.4f}/{val_acc:.3f} | "
            + (f"val(newenv) {val_new_loss:.4f}/{val_new_acc:.3f} | " if has_val_new else "")
            + f"lr {current_lr:.6f}"
        )
        
        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": float(tr_loss),
                "train/acc": float(tr_acc),
                "val/loss": float(val_loss),
                "val/acc": float(val_acc),
                "learning_rate": float(current_lr),
            }
            wandb.log(log_dict, step=epoch)
            if has_val_new:
                wandb.log(
                    {"val_new/loss": float(val_new_loss), "val_new/acc": float(val_new_acc)},
                    step=epoch,
                )
        
        # Detailed evaluation every eval_every epochs or on last epoch
        if epoch % eval_every == 0 or epoch == n_epochs:
            val_results = evaluate_detailed(
                model, current_X_val, current_y_val, current_meta_val, batch_size, device, size
            )
            
            print_evaluation_report(val_results, f"Val (epoch {epoch})")
            if has_val_new:
                val_new_results = evaluate_detailed(
                    model, current_X_val_new, current_y_val_new, current_meta_val_new, batch_size, device, size
                )
                print_evaluation_report(val_new_results, f"Val New (epoch {epoch})")
            
            if use_wandb:
                # Log detailed metrics
                for direction, acc in val_results['by_direction'].items():
                    wandb.log({f"val/acc_dir_{['N','E','S','W'][direction]}": acc}, step=epoch)
                for distance, acc in val_results['by_distance'].items():
                    wandb.log({f"val/acc_dist_{distance}": acc}, step=epoch)
                if has_val_new:
                    for direction, acc in val_new_results['by_direction'].items():
                        wandb.log({f"val_new/acc_dir_{['N','E','S','W'][direction]}": acc}, step=epoch)
                    for distance, acc in val_new_results['by_distance'].items():
                        wandb.log({f"val_new/acc_dist_{distance}": acc}, step=epoch)
        
        # Save checkpoint
        if save_every > 0 and epoch % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:04d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': tr_loss,
                'train_acc': tr_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    if save_dir:
        final_path = os.path.join(save_dir, "model_final.pt")
        torch.save({
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, final_path)
        print(f"Saved final model to {final_path}")


def train_classifier(
    size: int,
    speed: int,
    observation_size: int,
    shared_vectorhash: bool,
    seed: int,
    val_fraction: float,
    hidden_size: int,
    num_model_layers: int,
    batch_size: int,
    n_epochs: int,
    lr: float,
    device: str,
    use_grid: bool,
    input_type: str | None,
    Np: int | None,
    Npos: int | None,
    lambdas: list[int] | None,
    fwhm_ratio: float = 0.0,
    encoder_weights: str | None = None,
    thresh: float = 0.5,
    num_train_envs: int = 1,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    model_type: str = "mlp",
    use_wandb: bool = False,
    wandb_project: str = "cls_action_classifier",
    use_displacement: bool = False,
    use_headings: bool = False,
    max_steps: Tuple[int, int] = (1, 2),
    full_scaffold: bool = False,
    regenerate_envs_per_epoch: bool = False,
    scheduler_type: str = "none",
    scheduler_step_size: int = 30,
    scheduler_gamma: float = 0.1,
    scheduler_T_max: int = 100,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    warmup_epochs: int = 0,
    warmup_start_lr: float = 0.0,
    num_val_new_envs: int = 4,
    global_space: bool = False,
    save_every: int = 0,
    save_dir: str = "checkpoints",
):
    """Main training function that orchestrates all components."""
    check_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Load encoder if needed
    print(f"Loading encoder...")
    grid_encoder = None
    if use_grid and input_type == "encoded_g":
        if encoder_weights is None:
            raise ValueError("--encoder_weights required for input_type='encoded_g'")
        grid_encoder = load_grid_encoder(encoder_weights, lambdas, device)
    print(f"Loaded encoder in {time.time() - check_time:.2f} seconds")
    check_time = time.time()

    if full_scaffold:
        # Create environments
        print(f"Creating environments...")
        train_envs, val_env, vh = create_environments(
            size=size,
            speed=speed,
            observation_size=observation_size,
            seed=seed,
            num_train_envs=num_train_envs,
            use_grid=use_grid,
            input_type=input_type,
            Np=Np,
            lambdas=lambdas,
            Npos=Npos,
            thresh=thresh,
            fwhm_ratio=fwhm_ratio,
            grid_encoder=grid_encoder,
            device=device,
            use_headings=use_headings,
            shared_vectorhash=shared_vectorhash,
        )
        print(f"Created environments in {time.time() - check_time:.2f} seconds")
        check_time = time.time()

        print(f"Plotting environment layout...")
        # Plot environment layout (grid mode only)
        if use_grid and vh is not None:
            all_envs = train_envs + [val_env]
            plot_environment_layout(vh, all_envs, num_train_envs, size, lambdas)
        print(f"Plotted environment layout in {time.time() - check_time:.2f} seconds")
        check_time = time.time()
        
        # Build training data
        print(f"Building training data...")
        X_tr, y_tr, X_val, y_val, X_val_new, y_val_new, meta_tr, meta_val, meta_val_new = build_training_data_with_scaffold(
            train_envs=train_envs,
            val_env=val_env,
            size=size,
            val_fraction=val_fraction,
            input_type=input_type,
            use_displacement=use_displacement,
            rng=rng,
            max_steps=max_steps,
        )
        print(X_tr.shape)
        print(f"Built training data in {time.time() - check_time:.2f} seconds")
        check_time = time.time()
        
        # Verify data integrity
        print(f"Verifying data integrity...")
        verify_label_correctness(y_tr, meta_tr, "training", sample_size=1000)
        verify_label_correctness(y_val, meta_val, "validation", sample_size=500)
        verify_label_correctness(y_val_new, meta_val_new, "val_new", sample_size=500)
        verify_data_leakage(meta_tr, meta_val_new, "scaffold mode")
        print(f"Data integrity checks completed in {time.time() - check_time:.2f} seconds")
        check_time = time.time()

        # Get environment offsets for plotting
        if use_grid and vh is not None:
            env_offsets = vh.env_locations
        else:
            env_offsets = [(0, 0)] * (num_train_envs + 1)
    else:
        # Build training data without creating environment objects (uses shared scaffold)
        if not use_grid:
            raise ValueError("build_training_data_without_scaffold requires use_grid=True")
        if Np is None or lambdas is None or Npos is None:
            raise ValueError("Np, lambdas, and Npos required for build_training_data_without_scaffold")
        if global_space and regenerate_envs_per_epoch:
            raise ValueError("regenerate_envs_per_epoch is not supported when global_space=True (no environments to regenerate)")
        
        print(f"Building training data without scaffold...")
        if global_space:
            X_tr, y_tr, X_val, y_val, X_val_new, y_val_new, meta_tr, meta_val, meta_val_new = build_training_data_global_space(
                size=size,
                max_steps=max_steps,
                val_fraction=val_fraction,
                input_type=input_type,
                use_displacement=use_displacement,
                rng=rng,
                Np=Np,
                lambdas=lambdas,
                Npos=Npos,
                thresh=thresh,
                use_headings=use_headings,
                grid_encoder=grid_encoder,
                fwhm_ratio=fwhm_ratio,
            )
        else:
            X_tr, y_tr, X_val, y_val, X_val_new, y_val_new, meta_tr, meta_val, meta_val_new = build_training_data_without_scaffold(
                size=size,
                n_envs=num_train_envs,
                max_steps=max_steps,
                val_fraction=val_fraction,
                input_type=input_type,
                use_displacement=use_displacement,
                rng=rng,
                Np=Np,
                lambdas=lambdas,
                Npos=Npos,
                thresh=thresh,
                use_headings=use_headings,
                grid_encoder=grid_encoder,
                fwhm_ratio=fwhm_ratio,
                num_val_new_envs=num_val_new_envs,
            )
        print(f"Built training data in {time.time() - check_time:.2f} seconds")
        check_time = time.time()
        
        # Verify data integrity
        print(f"Verifying data integrity...")
        verify_label_correctness(y_tr, meta_tr, "training", sample_size=1000)
        verify_label_correctness(y_val, meta_val, "validation", sample_size=500)
        if len(y_val_new) > 0:
            verify_label_correctness(y_val_new, meta_val_new, "val_new", sample_size=500)
            verify_data_leakage(meta_tr, meta_val_new, "without_scaffold mode")
        print(f"Data integrity checks completed in {time.time() - check_time:.2f} seconds")
        check_time = time.time()
        
        # For plotting (not used in without_scaffold mode, but needed for compatibility)
        env_offsets = [(0, 0)] * (num_train_envs + 1)

    # Create model
    print(f"Creating model...")
    print(f"X_val shape: {X_val.shape}")
    input_size = X_tr.shape[1]
    print(input_size)
    model, actual_model_type = create_model(
        input_size=input_size,
        hidden_size=hidden_size,
        num_model_layers=num_model_layers,
        dropout=dropout,
        model_type=model_type,
        input_type=input_type,
        lambdas=lambdas,
        use_displacement=use_displacement,
        device=device,
    )
    print(f"Created model in {time.time() - check_time:.2f} seconds")
    check_time = time.time()

    # Initialize W&B
    if use_wandb:
        cfg = dict(
            size=size,
            speed=speed,
            observation_size=observation_size,
            seed=seed,
            train_samples=len(X_tr),
            val_samples=len(X_val),
            val_new_samples=len(X_val_new),
            val_fraction=val_fraction,
            hidden_size=hidden_size,
            num_model_layers=num_model_layers,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            device=device,
            vectorhash=use_grid,
            Np=int(Np) if Np is not None else None,
            lambdas=lambdas,
            input_type=input_type,
            fwhm_ratio=fwhm_ratio,
            encoder_weights=encoder_weights,
            thresh=thresh,
            use_displacement=use_displacement,
            dropout=dropout,
            weight_decay=weight_decay,
            model_type=actual_model_type,
            max_steps=list(max_steps),
            full_scaffold=full_scaffold,
            num_train_envs=num_train_envs,
            regenerate_envs_per_epoch=regenerate_envs_per_epoch,
            scheduler_type=scheduler_type,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            scheduler_T_max=scheduler_T_max,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
            num_val_new_envs=num_val_new_envs,
            Npos=Npos,
            global_space=global_space,
            save_every=save_every,
            save_dir=save_dir,
        )
        wandb.init(project=wandb_project, config=cfg)
    
    # Determine actual save directory (include wandb run name or timestamp)
    if use_wandb and wandb.run is not None:
        run_save_dir = os.path.join(save_dir, wandb.run.name)
    else:
        # Use timestamp-based name if not using wandb
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_save_dir = os.path.join(save_dir, f"run_{timestamp}")
    
    # Create data generator function if regenerate_envs_per_epoch is enabled
    data_generator = None
    if regenerate_envs_per_epoch:
        if full_scaffold:
            raise ValueError("regenerate_envs_per_epoch is only supported with full_scaffold=False")
        if not use_grid:
            raise ValueError("regenerate_envs_per_epoch requires use_grid=True")
        if global_space:
            raise ValueError("regenerate_envs_per_epoch is not supported when global_space=True (no environments to regenerate)")
        
        def generate_data_for_epoch(epoch: int):
            """Generate data for a specific epoch with new environment locations."""
            epoch_rng = np.random.RandomState(seed + epoch * 1000)  # Different seed per epoch
            # Use the without_scaffold version
            return build_training_data_without_scaffold(
                size=size,
                n_envs=num_train_envs,
                max_steps=max_steps,
                val_fraction=val_fraction,
                input_type=input_type,
                use_displacement=use_displacement,
                rng=epoch_rng,
                Np=Np,
                lambdas=lambdas,
                Npos=Npos,
                thresh=thresh,
                use_headings=use_headings,
                grid_encoder=grid_encoder,
                fwhm_ratio=fwhm_ratio,
                num_val_new_envs=num_val_new_envs,
            )
        data_generator = generate_data_for_epoch
    
    # Train
    print(f"Training...")
    check_time = time.time()
    train_loop(
        model=model,
        X_tr=X_tr,
        y_tr=y_tr,
        X_val=X_val,
        y_val=y_val,
        X_val_new=X_val_new,
        y_val_new=y_val_new,
        meta_val=meta_val,
        meta_val_new=meta_val_new,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        use_wandb=use_wandb,
        size=size,
        eval_every=5,
        regenerate_envs_per_epoch=regenerate_envs_per_epoch,
        data_generator=data_generator,
        scheduler_type=scheduler_type,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        scheduler_T_max=scheduler_T_max,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        warmup_epochs=warmup_epochs,
        warmup_start_lr=warmup_start_lr,
        save_every=save_every,
        save_dir=run_save_dir,
    )
    print(f"Trained in {time.time() - check_time:.2f} seconds")

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--observation_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_model_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Grid / Vectorhash options
    parser.add_argument("--vectorhash", action="store_true", default=False)
    parser.add_argument("--shared_vectorhash", action="store_true", default=False)
    parser.add_argument("--full_scaffold", action="store_true", default=False)
    parser.add_argument("--Np", type=int, default=1600)
    parser.add_argument("--lambdas", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--Npos", type=int, default=None)
    parser.add_argument("--input_type", type=str, default="g_idx",
                        help="g_idx, g_hot, encoded_g, s, p, or euclidean (raw x,y coords)")
    parser.add_argument("--use_displacement", action="store_true", default=False,
                        help="Use displacement to generate training data")
    parser.add_argument("--max_steps", type=int, nargs=2, default=[1, 2],
                        help="Displacement range [min, max) (e.g., '1 2' for single step, '3 5' for shell)")
    parser.add_argument("--fwhm_ratio", type=float, default=0.0,
                        help="Gaussian smoothing FWHM ratio for g_hot/encoded_g (0 = no smoothing)")
    parser.add_argument("--encoder_weights", type=str, default=None,
                        help="Encoder weights file in encoders/ directory (required for encoded_g)")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="VectorHash nonlinearity threshold (default 0.5, was 2.0)")
    parser.add_argument("--num_train_envs", type=int, default=1,
                        help="Number of training environments (more = more offset diversity)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate for regularization")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "cnn"],
                        help="Model type: 'mlp' or 'cnn' (CNN uses 2D structure of g_hot)")
    parser.add_argument("--use_headings", action="store_true", default=False,
                        help="Use heading-dependent observations (default: heading-invariant)")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="cls_action_classifier")
    parser.add_argument("--regenerate_envs_per_epoch", action="store_true", default=False,
                        help="Regenerate environment locations and data before each epoch (only works with full_scaffold=False)")
    parser.add_argument("--global_space", action="store_true", default=False,
                        help="Drop environments and sample transitions from the entire Npos×Npos space. Disables val_new.")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "cosine", "plateau"],
                        help="Learning rate scheduler type")
    parser.add_argument("--scheduler_step_size", type=int, default=30,
                        help="Step size for StepLR scheduler (epochs between LR reductions)")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1,
                        help="Multiplicative factor for StepLR scheduler")
    parser.add_argument("--scheduler_T_max", type=int, default=100,
                        help="Maximum iterations for CosineAnnealingLR scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=10,
                        help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_factor", type=float, default=0.5,
                        help="Factor for ReduceLROnPlateau scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Number of epochs for linear learning rate warmup (0 = no warmup)")
    parser.add_argument("--warmup_start_lr", type=float, default=0.0,
                        help="Starting learning rate for warmup (default: 0.0)")
    parser.add_argument("--num_val_new_envs", type=int, default=4,
                        help="Number of environments to hold out for val_new (only for full_scaffold=False)")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save model checkpoint every N epochs (0 = only save final model)")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parser.parse_args()

    train_classifier(
        size=args.size,
        speed=args.speed,
        observation_size=args.observation_size,
        shared_vectorhash=args.shared_vectorhash,
        seed=args.seed,
        val_fraction=args.val_fraction,
        hidden_size=args.hidden_size,
        num_model_layers=args.num_model_layers,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        use_grid=args.vectorhash,
        input_type=(args.input_type if args.vectorhash else None),
        Np=(args.Np if args.vectorhash else None),
        lambdas=(args.lambdas if args.vectorhash else None),
        Npos=(args.Npos if args.vectorhash else None),
        fwhm_ratio=args.fwhm_ratio,
        encoder_weights=args.encoder_weights,
        thresh=args.thresh,
        num_train_envs=args.num_train_envs,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        model_type=args.model_type,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        use_displacement=args.use_displacement,
        use_headings=args.use_headings,
        max_steps=tuple(args.max_steps),
        full_scaffold=args.full_scaffold,
        regenerate_envs_per_epoch=args.regenerate_envs_per_epoch,
        global_space=args.global_space,
        scheduler_type=args.scheduler,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_T_max=args.scheduler_T_max,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
        num_val_new_envs=args.num_val_new_envs,
        save_every=args.save_every,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
