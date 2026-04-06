"""
Navigation evaluation for distance encoders.

Tests how well an encoder supports Hopfield-based navigation by simulating
continuous trajectories across randomly placed environments.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cls.hopfield import Hopfield


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_full_grid(encoder, Phi, gain, device, batch_size=1000):
    """Encode the full Npos x Npos grid through the encoder.

    Args:
        encoder: GridEncoderCNN (or compatible) on *device*, in eval mode.
        Phi: np.ndarray of shape (code_dim, Npos, Npos) — grid codes
            (dim0 = x, dim1 = y, matching gen_gbook_2d convention).
        gain: float — encoder gain parameter.
        device: torch.device.
        batch_size: int — batch size for encoding.

    Returns:
        encoded_Phi: np.ndarray of shape (Npos, Npos, embed_dim).
            Indexed as encoded_Phi[gx, gy].
    """
    code_dim, Npos_x, Npos_y = Phi.shape
    flat = Phi.reshape(code_dim, Npos_x * Npos_y).T.astype(np.float32)  # (N, code_dim)

    parts = []
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            chunk = torch.from_numpy(flat[start:start + batch_size]).to(device)
            enc = encoder(chunk, gain=gain).cpu().numpy()
            parts.append(enc)
    if was_training:
        encoder.train()

    encoded_flat = np.concatenate(parts, axis=0)  # (N, embed_dim)
    embed_dim = encoded_flat.shape[1]
    return encoded_flat.reshape(Npos_x, Npos_y, embed_dim)


# ---------------------------------------------------------------------------
# Environment placement
# ---------------------------------------------------------------------------

def _rects_overlap(y0a, x0a, size_a, y0b, x0b, size_b):
    """Check if two axis-aligned rectangles overlap (touching = no overlap).

    Note: parameter names use (y0, x0) for compatibility with train_dist_encoder,
    but in grid convention these are (dim0, dim1) = (gx, gy).
    """
    return not (y0a + size_a <= y0b or y0b + size_b <= y0a or
                x0a + size_a <= x0b or x0b + size_b <= x0a)


def sample_train_eval_envs(train_y0s, train_x0s, train_patch_size,
                           eval_env_size, n_envs, rng, max_attempts=10000):
    """Sample eval envs that fit entirely within training patches.

    Each eval env is placed so its bounding box (plus 1-cell border for
    Gram-Schmidt neighbors) lies inside one of the training patches.

    Returns:
        List of (y0, x0) global coords for each eval env.
    """
    # Need border of 1 for neighbor lookups
    margin = 1
    inner = train_patch_size - eval_env_size - 2 * margin
    if inner < 0:
        print(f"WARNING: eval_env_size ({eval_env_size}) + border doesn't fit in "
              f"train_patch_size ({train_patch_size}). Skipping train nav eval.")
        return []

    n_patches = len(train_y0s)
    placements = []
    attempts = 0

    while len(placements) < n_envs and attempts < max_attempts:
        # Pick a random training patch
        pi = rng.randint(0, n_patches)
        py0, px0 = train_y0s[pi], train_x0s[pi]
        # Sample position within patch (with margin)
        y0 = py0 + margin + rng.randint(0, inner + 1)
        x0 = px0 + margin + rng.randint(0, inner + 1)
        # Check no overlap with existing placements
        if all(not _rects_overlap(y0, x0, eval_env_size, ey, ex, eval_env_size)
               for ey, ex in placements):
            placements.append((y0, x0))
        attempts += 1

    if len(placements) < n_envs:
        print(f"WARNING: could only place {len(placements)}/{n_envs} train eval envs")
    return placements


def sample_val_eval_envs(grid_H, grid_W, train_y0s, train_x0s, train_patch_size,
                         eval_env_size, n_envs, rng, max_attempts=10000):
    """Sample eval envs that do NOT overlap any training patch.

    Also ensures eval envs don't overlap each other and have a 1-cell border
    from the grid edge for Gram-Schmidt neighbor lookups.

    Returns:
        List of (y0, x0) global coords for each eval env.
    """
    margin = 1  # border for neighbor lookups
    placements = []
    attempts = 0

    while len(placements) < n_envs and attempts < max_attempts:
        y0 = rng.randint(margin, grid_H - eval_env_size - margin + 1)
        x0 = rng.randint(margin, grid_W - eval_env_size - margin + 1)
        # Check no overlap with any training patch
        overlaps_train = any(
            _rects_overlap(y0, x0, eval_env_size, ty, tx, train_patch_size)
            for ty, tx in zip(train_y0s, train_x0s)
        )
        if overlaps_train:
            attempts += 1
            continue
        # Check no overlap with existing val placements
        overlaps_val = any(
            _rects_overlap(y0, x0, eval_env_size, ey, ex, eval_env_size)
            for ey, ex in placements
        )
        if overlaps_val:
            attempts += 1
            continue
        placements.append((y0, x0))
        attempts += 1

    if len(placements) < n_envs:
        print(f"WARNING: could only place {len(placements)}/{n_envs} val eval envs")
    return placements


# ---------------------------------------------------------------------------
# Trajectory simulation
# ---------------------------------------------------------------------------

def _compute_projection_matrix(encoded_Phi, gx, gy):
    """Compute Gram-Schmidt projection matrix at (gx, gy).

    Axes follow the grid codebook convention: dim0 = x (East), dim1 = y (North).

    Returns:
        W: (2, embed_dim) — row 0 = East basis, row 1 = North basis.
        current: (embed_dim,) — encoded state at (gx, gy).
    """
    Nx, Ny = encoded_Phi.shape[:2]
    gx = np.clip(gx, 1, Nx - 2)
    gy = np.clip(gy, 1, Ny - 2)

    current = encoded_Phi[gx, gy]
    # North neighbor: (gx, gy+1) — +dim1, East neighbor: (gx+1, gy) — +dim0
    d_forward = encoded_Phi[gx, gy + 1] - current
    d_right = encoded_Phi[gx + 1, gy] - current

    # Gram-Schmidt: e1 = normalize(d_forward), e2 = orthogonalize(d_right, e1)
    norm_f = np.linalg.norm(d_forward)
    e1 = d_forward / max(norm_f, 1e-12)
    e2 = d_right - np.dot(d_right, e1) * e1
    norm_r = np.linalg.norm(e2)
    e2 = e2 / max(norm_r, 1e-12)

    W = np.stack([e2, e1], axis=0)  # (2, embed_dim): row0=East, row1=North
    return W, current


def _continuous_step(W, current_np, next_state_np, scale, normalize):
    """Compute continuous movement from Hopfield displacement.

    Returns:
        (dgx, dgy): step in grid coordinates (dim0 = x/East, dim1 = y/North).
        magnitude: raw magnitude of projected vector.
    """
    d_query = next_state_np - current_np
    q = W @ d_query  # q[0] = East component, q[1] = North component

    magnitude = np.linalg.norm(q)
    if magnitude < 1e-8:
        return (0.0, 0.0), 0.0

    if normalize:
        q_scaled = (q / magnitude) * scale
    else:
        q_scaled = q * scale

    # q[0] = East = +dim0 (+x), q[1] = North = +dim1 (+y)
    dgx = q_scaled[0]  # East  → +dim0
    dgy = q_scaled[1]  # North → +dim1

    return (dgx, dgy), magnitude


def simulate_trajectory(encoded_Phi, hopfield, start_gx, start_gy, goal_loc,
                        gain, max_steps, scale, normalize,
                        platform_radius, recompute_interval, alpha):
    """Simulate one continuous trajectory.

    Positions are (gx, gy) in global grid coords (dim0 = x, dim1 = y).

    Args:
        encoded_Phi: (Npos, Npos, embed_dim) array.
        hopfield: Hopfield network with stored goals.
        start_gx, start_gy: global starting position.
        goal_loc: (gx, gy) global goal position.
        gain: Hopfield recall beta.
        max_steps: maximum number of steps.
        scale: step scale (with normalize=True, each step is unit * scale).
        normalize: whether to normalize steps to unit length.
        platform_radius: goal reached threshold.
        recompute_interval: how often to recompute projection matrix W.
        alpha: Hopfield recall alpha (mixing coefficient).

    Returns:
        trajectory: np.ndarray of shape (T, 2) — positions visited as [gx, gy].
    """
    Nx, Ny = encoded_Phi.shape[:2]
    min_gx, max_gx = 1, Nx - 2
    min_gy, max_gy = 1, Ny - 2

    position = np.array([float(start_gx), float(start_gy)])
    trajectory = [position.copy()]

    grid_pos = np.clip(np.round(position).astype(int), [min_gx, min_gy], [max_gx, max_gy])
    W, _ = _compute_projection_matrix(encoded_Phi, grid_pos[0], grid_pos[1])
    steps_since_recompute = 0

    for _ in range(max_steps):
        grid_pos = np.clip(np.round(position).astype(int), [min_gx, min_gy], [max_gx, max_gy])
        state = torch.from_numpy(encoded_Phi[grid_pos[0], grid_pos[1]].copy()).float()
        current_np = state.numpy()

        next_state, _ = hopfield.recall(state, steps=1, beta=gain, alpha=alpha,
                                        use_tanh=True, normalize_each=True)

        if recompute_interval > 0 and steps_since_recompute >= recompute_interval:
            W, _ = _compute_projection_matrix(encoded_Phi, grid_pos[0], grid_pos[1])
            steps_since_recompute = 0

        (dgx, dgy), mag = _continuous_step(W, current_np, next_state.numpy(), scale, normalize)
        steps_since_recompute += 1

        if mag < 1e-6:
            break

        new_position = position + np.array([dgx, dgy])
        new_position[0] = np.clip(new_position[0], min_gx, max_gx)
        new_position[1] = np.clip(new_position[1], min_gy, max_gy)

        if np.linalg.norm(new_position - position) < 1e-6:
            break

        position = new_position
        trajectory.append(position.copy())

        if np.linalg.norm(position - np.array(goal_loc, dtype=float)) < platform_radius:
            break

    return np.array(trajectory)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def _eval_single_hopfield(encoded_Phi, chunk_placements, goal_locations, hopfield,
                          eval_env_size, gain, local_vals,
                          max_steps, scale, normalize, platform_radius,
                          recompute_interval, hopfield_alpha,
                          save_heatmaps, heatmap_dir, env_idx_offset):
    """Evaluate a single Hopfield network across its environments.

    Placements and goals use (gx, gy) convention: dim0 = x, dim1 = y.
    """
    per_env_results = []
    figures = []

    for local_idx in range(len(chunk_placements)):
        global_idx = env_idx_offset + local_idx
        gx0, gy0 = chunk_placements[local_idx]
        goal_loc = goal_locations[local_idx]
        goal_local = (goal_loc[0] - gx0, goal_loc[1] - gy0)

        starts = [(sx, sy) for sx in local_vals for sy in local_vals
                  if (sx, sy) != goal_local]

        successes = 0
        steps_list = []
        speed_list = []
        dist_success = []
        dist_fail = []
        dir_acc_success = []
        dir_acc_fail = []
        success_indices = set()

        for i, (sx, sy) in enumerate(starts):
            traj = simulate_trajectory(
                encoded_Phi, hopfield, gx0 + sx, gy0 + sy, goal_loc,
                gain, max_steps, scale, normalize,
                platform_radius, recompute_interval, hopfield_alpha,
            )

            start_pos = traj[0]
            start_dist = np.linalg.norm(start_pos - np.array(goal_loc, dtype=float))
            final_pos = traj[-1]
            final_dist = np.linalg.norm(final_pos - np.array(goal_loc, dtype=float))
            n_steps = len(traj) - 1

            if n_steps > 0:
                dists_along = np.linalg.norm(
                    traj - np.array(goal_loc, dtype=float), axis=1)
                da = float(np.sum(dists_along[1:] < dists_along[:-1])) / n_steps
            else:
                da = float('nan')

            if final_dist < platform_radius:
                successes += 1
                steps_list.append(n_steps)
                speed_list.append(start_dist / n_steps if n_steps > 0 else float('inf'))
                dist_success.append(final_dist)
                if not np.isnan(da):
                    dir_acc_success.append(da)
                success_indices.add(i)
            else:
                dist_fail.append(final_dist)
                if not np.isnan(da):
                    dir_acc_fail.append(da)

        accuracy = successes / len(starts) if starts else 0.0
        env_result = {
            'env_idx': global_idx,
            'placement': (gx0, gy0),
            'goal': goal_loc,
            'n_starts': len(starts),
            'accuracy': accuracy,
            'mean_steps': float(np.mean(steps_list)) if steps_list else float('nan'),
            'mean_speed': float(np.mean(speed_list)) if speed_list else float('nan'),
            'mean_dist_success': float(np.mean(dist_success)) if dist_success else float('nan'),
            'mean_dist_fail': float(np.mean(dist_fail)) if dist_fail else float('nan'),
            'mean_dir_acc_success': float(np.mean(dir_acc_success)) if dir_acc_success else float('nan'),
            'mean_dir_acc_fail': float(np.mean(dir_acc_fail)) if dir_acc_fail else float('nan'),
        }
        per_env_results.append(env_result)

        print(f"  Env {global_idx:2d} | acc={accuracy:.2f} | "
              f"dir_acc_succ={env_result['mean_dir_acc_success']:.2f} | "
              f"dir_acc_fail={env_result['mean_dir_acc_fail']:.2f} | "
              f"mean_steps={env_result['mean_steps']:5.1f} | "
              f"speed={env_result['mean_speed']:.3f} | "
              f"dist_succ={env_result['mean_dist_success']:.2f} | "
              f"dist_fail={env_result['mean_dist_fail']:.2f}")

        if save_heatmaps:
            heatmap = np.full((eval_env_size, eval_env_size), np.nan)
            for i, (sx, sy) in enumerate(starts):
                heatmap[sx, sy] = 1.0 if i in success_indices else 0.0

            fig, ax = plt.subplots(figsize=(7, 7))
            im = ax.imshow(heatmap.T, origin='lower', cmap='RdYlGn', vmin=0, vmax=1,
                           extent=[0, eval_env_size, 0, eval_env_size])
            local_gx = goal_loc[0] - gx0
            local_gy = goal_loc[1] - gy0
            ax.scatter(local_gx + 0.5, local_gy + 0.5, s=200, c='blue', marker='*',
                       edgecolors='white', linewidths=1.5, zorder=6, label='Goal')
            ax.set_xlabel('Local X')
            ax.set_ylabel('Local Y')
            ax.set_title(f'Env {global_idx} | acc={accuracy:.2f} | '
                         f'dir_acc S={env_result["mean_dir_acc_success"]:.2f} '
                         f'F={env_result["mean_dir_acc_fail"]:.2f}',
                         fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax, shrink=0.7, label='Success')
            ax.legend(loc='upper right', fontsize=9)
            plt.tight_layout()
            figures.append(fig)

            if heatmap_dir:
                os.makedirs(heatmap_dir, exist_ok=True)
                fig.savefig(os.path.join(heatmap_dir, f"heatmap_env{global_idx}.png"),
                            dpi=100, bbox_inches='tight')
            plt.close(fig)

    return per_env_results, figures


def run_navigation_eval(encoded_Phi, env_placements, eval_env_size, gain,
                        hopfield_alpha=0.8,
                        n_starts_per_env=100, max_steps_mult=3,
                        scale=1.0, normalize=True, platform_radius=1.0,
                        recompute_interval=1, rng=None,
                        num_hopfields=1,
                        save_heatmaps=False, heatmap_dir=None):
    """Run Hopfield navigation evaluation across multiple environments.

    Splits env_placements into `num_hopfields` groups, builds a separate
    Hopfield network for each group, and evaluates all environments.
    This avoids Hopfield capacity issues when testing many environments.

    Coordinates use (gx, gy) convention: dim0 = x (East), dim1 = y (North).

    Args:
        encoded_Phi: np.ndarray (Npos, Npos, embed_dim) — full encoded grid,
            indexed as encoded_Phi[gx, gy].
        env_placements: list of (gx0, gy0) — top-left corners of eval envs.
            Length should be num_hopfields * envs_per_hopfield.
        eval_env_size: int — side length of each eval env.
        gain: float — used as Hopfield beta (from model config gain).
        hopfield_alpha: float — Hopfield recall mixing coefficient.
        n_starts_per_env: int — target number of starting positions per env.
        max_steps_mult: int — max_steps = max_steps_mult * eval_env_size.
        scale: float — step size scale.
        normalize: bool — normalize steps to unit length.
        platform_radius: float — goal reached threshold (Euclidean).
        recompute_interval: int — how often to recompute W.
        rng: np.random.RandomState — random state.
        num_hopfields: int — number of separate Hopfield networks to use.
        save_heatmaps: bool — whether to save heatmap PNGs.
        heatmap_dir: str — directory for heatmap PNGs (required if save_heatmaps).

    Returns:
        dict with aggregate metrics and per_env_results list.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n_total = len(env_placements)
    if n_total == 0:
        return {'accuracy': float('nan'), 'per_env_results': []}

    embed_dim = encoded_Phi.shape[2]
    max_steps = max_steps_mult * eval_env_size

    # Evenly spaced starts (shared across all envs)
    n_per_side = int(np.ceil(np.sqrt(n_starts_per_env)))
    local_vals = np.linspace(0, eval_env_size - 1, n_per_side).astype(int)

    # Chunk placements into groups for separate Hopfield networks
    envs_per_hop = max(1, n_total // num_hopfields)
    chunks = []
    for i in range(0, n_total, envs_per_hop):
        chunks.append(env_placements[i:i + envs_per_hop])

    all_per_env_results = []
    all_figures = []
    env_idx_offset = 0

    for hop_idx, chunk in enumerate(chunks):
        print(f"  Hopfield {hop_idx} ({len(chunk)} envs):")

        # Pick goals and build Hopfield for this chunk
        goal_locations = []
        for gx0, gy0 in chunk:
            goal_gx = gx0 + rng.randint(0, eval_env_size)
            goal_gy = gy0 + rng.randint(0, eval_env_size)
            goal_locations.append((goal_gx, goal_gy))

        hopfield = Hopfield(num_units=embed_dim, beta=gain, device="cpu")
        for goal_gx, goal_gy in goal_locations:
            goal_enc = torch.from_numpy(encoded_Phi[goal_gx, goal_gy].copy()).float()
            hopfield.input_memory(goal_enc)

        chunk_results, chunk_figs = _eval_single_hopfield(
            encoded_Phi, chunk, goal_locations, hopfield,
            eval_env_size, gain, local_vals,
            max_steps, scale, normalize, platform_radius,
            recompute_interval, hopfield_alpha,
            save_heatmaps, heatmap_dir, env_idx_offset,
        )
        all_per_env_results.extend(chunk_results)
        all_figures.extend(chunk_figs)
        env_idx_offset += len(chunk)

    # --- Aggregate across all envs ---
    accs = [r['accuracy'] for r in all_per_env_results]
    steps_all = [r['mean_steps'] for r in all_per_env_results if not np.isnan(r['mean_steps'])]
    speeds_all = [r['mean_speed'] for r in all_per_env_results if not np.isnan(r['mean_speed'])]
    ds = [r['mean_dist_success'] for r in all_per_env_results if not np.isnan(r['mean_dist_success'])]
    df = [r['mean_dist_fail'] for r in all_per_env_results if not np.isnan(r['mean_dist_fail'])]
    das = [r['mean_dir_acc_success'] for r in all_per_env_results if not np.isnan(r['mean_dir_acc_success'])]
    daf = [r['mean_dir_acc_fail'] for r in all_per_env_results if not np.isnan(r['mean_dir_acc_fail'])]

    result = {
        'accuracy': float(np.mean(accs)) if accs else float('nan'),
        'accuracy_std': float(np.std(accs)) if accs else float('nan'),
        'mean_steps': float(np.mean(steps_all)) if steps_all else float('nan'),
        'mean_speed': float(np.mean(speeds_all)) if speeds_all else float('nan'),
        'mean_dist_success': float(np.mean(ds)) if ds else float('nan'),
        'mean_dist_fail': float(np.mean(df)) if df else float('nan'),
        'mean_dir_acc_success': float(np.mean(das)) if das else float('nan'),
        'mean_dir_acc_fail': float(np.mean(daf)) if daf else float('nan'),
        'per_env_results': all_per_env_results,
    }
    if save_heatmaps:
        result['figures'] = all_figures

    return result
