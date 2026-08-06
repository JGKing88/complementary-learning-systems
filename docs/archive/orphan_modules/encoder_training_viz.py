"""Visualization helpers for trained encoders.

- encode_grid_np: full-grid embedding (wraps cls.eval.nav_eval.encode_full_grid)
- plot_similarity_heatmap: cosine similarity of a ref point vs the full grid
- unique_radius_per_theta / plot_unique_radius_polar: how far you can walk
  in each direction before the similarity becomes non-monotonic (same
  method as vector_hash_playground.ipynb).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Unique radius (ported from test_placefield.compute_unique_radius)
# ---------------------------------------------------------------------------

def compute_unique_radius(xs_1d: np.ndarray, ref_ix: int,
                          similarity: np.ndarray) -> float:
    """Farthest distance along a 1D line from ref_ix at which similarity is
    strictly greater than all farther values.

    When both flanks exist (interior reference on the 1D sample line), returns
    min(left, right). If the reference sits at an end of the sampled segment
    (grid edge along that ray), only the available flank is used — otherwise
    ``min(0, r)`` would incorrectly zero the radius.
    """
    sim = np.asarray(similarity)
    n = len(sim)
    suff = np.maximum.accumulate(sim[::-1])[::-1]
    acc = np.maximum.accumulate(sim)

    right_r = 0.0
    for i in range(1, n - ref_ix):
        val = sim[ref_ix + i]
        k = ref_ix + i + 1
        max_beyond = suff[k] if k < n else -np.inf
        if val > max_beyond:
            right_r = xs_1d[ref_ix + i] - xs_1d[ref_ix]
        else:
            break

    left_r = 0.0
    for i in range(1, ref_ix + 1):
        val = sim[ref_ix - i]
        ub = ref_ix - i
        max_beyond = acc[ub - 1] if ub > 0 else -np.inf
        if val > max_beyond:
            left_r = xs_1d[ref_ix] - xs_1d[ref_ix - i]
        else:
            break

    if ref_ix == 0:
        return float(right_r)
    if ref_ix == n - 1:
        return float(left_r)
    return float(min(left_r, right_r))


def unique_radius_per_theta(
    cos_map: np.ndarray,
    ref_x: float, ref_y: float,
    n_rays: int = 72,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each direction θ, compute the unique radius along that ray.

    Uses only the outward half-line in direction θ (positive t along
    ``(cos θ, sin θ)``). The opposite direction is a different θ and is
    computed separately — we do **not** take ``min(left, right)`` on the same
    undirected line, which would force 180° symmetry in the polar plot.

    Returns (thetas, widths) arrays of length n_rays.
    """
    H, W = cos_map.shape
    interp = RegularGridInterpolator(
        (np.arange(H), np.arange(W)), cos_map,
        bounds_error=False, fill_value=np.nan,
    )
    xmin, xmax, ymin, ymax = 0, W - 1, 0, H - 1

    thetas = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    widths = np.zeros(n_rays)
    for i, theta in enumerate(thetas):
        c, s = np.cos(theta), np.sin(theta)

        ts_pos: list[float] = []
        t = 0.0
        while True:
            t += dt
            x, y = ref_x + t * c, ref_y + t * s
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                break
            if not np.isfinite(float(interp((y, x)))):
                break
            ts_pos.append(t)

        # ref_ix == 0 ⇒ ``compute_unique_radius`` uses only the +t flank
        # (no min with the opposite direction on the same line).
        all_t = np.concatenate([np.array([0.0]), np.array(ts_pos)])
        ref_ix = 0
        sim = np.array([
            float(interp((ref_y + t * s, ref_x + t * c))) for t in all_t])
        widths[i] = compute_unique_radius(all_t, ref_ix, sim)
    return thetas, widths


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def similarity_map(Z_grid: np.ndarray, ref_x: int, ref_y: int) -> np.ndarray:
    """Cosine-sim map of a reference point vs all positions.

    Z_grid: (H, W, D) L2-normalized embedding grid.
    Returns (H, W) cosine similarity map.
    """
    z_ref = Z_grid[ref_y, ref_x]
    return (Z_grid @ z_ref)


def plot_similarity_heatmap(
    Z_grid: np.ndarray,
    ref_x: int, ref_y: int,
    y0s: Sequence[int] | None = None,
    x0s: Sequence[int] | None = None,
    sizes: Sequence[int] | None = None,
    zoom: int = 100,
    title: str = "",
    savepath: str | None = None,
) -> np.ndarray:
    """Plot similarity heatmap with full view + zoom view. Returns cos_map."""
    H, W, _ = Z_grid.shape
    cos_map = similarity_map(Z_grid, ref_x, ref_y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    im1 = ax1.imshow(cos_map, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(im1, ax=ax1, label="cosine sim")
    if y0s:
        for y0, x0, s in zip(y0s, x0s, sizes):
            ax1.add_patch(plt.Rectangle(
                (x0, y0), s, s, fill=False,
                edgecolor="white", linewidth=0.6, alpha=0.7))
    ax1.plot(ref_x, ref_y, "r*", markersize=15,
             markeredgecolor="white", markeredgewidth=1)
    ax1.set_title("full grid")

    y0z = max(0, int(ref_y) - zoom); y1z = min(H, int(ref_y) + zoom)
    x0z = max(0, int(ref_x) - zoom); x1z = min(W, int(ref_x) + zoom)
    crop = cos_map[y0z:y1z, x0z:x1z]
    vmin_z, vmax_z = float(np.nanmin(crop)), float(np.nanmax(crop))
    im2 = ax2.imshow(crop, cmap="viridis", vmin=vmin_z, vmax=vmax_z,
                     extent=[x0z, x1z, y1z, y0z])
    plt.colorbar(im2, ax=ax2, label=f"cos sim ({vmin_z:.3f}..{vmax_z:.3f})")
    ax2.plot(ref_x, ref_y, "r*", markersize=20,
             markeredgecolor="white", markeredgewidth=1)
    ax2.set_title(f"zoom ±{zoom} cells")

    fig.suptitle(title)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
    return cos_map


def plot_unique_radius_polar(
    thetas: np.ndarray, widths: np.ndarray,
    title: str = "",
    savepath: str | None = None,
) -> None:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(thetas, widths, "o-", color="#2563eb",
            linewidth=1.5, markersize=4)
    ax.fill(thetas, widths, alpha=0.25, color="#2563eb")
    ax.set_title(title, pad=20)
    ax.text(0.5, -0.1,
            f"min: {widths.min():.1f} | median: {np.median(widths):.1f} | "
            f"max: {widths.max():.1f}",
            transform=ax.transAxes, ha="center")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
