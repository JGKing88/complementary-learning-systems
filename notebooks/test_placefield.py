"""
Place Cell Unique Coding Radius — Hyperparameter Sweep
=======================================================
- **Unique radius:** 1D linear track (same as before), mean over repeats.
- **Peak width (directed):** 2D map; **global minimum** over a directed **2π**
  sweep of cosine flanks where similarity **strictly decreases** along each
  ray (matches `vector_hash_playground` omni cell).
- **2D unique radius:** same **2D** map; on each ray, sample cosine **±** from
  the reference and apply the **same `compute_unique_radius` rule** as the 1D
  track; **global minimum** over **2π** (matches `vector_hash_playground` 2D
  unique-radius cell). Mean over repeats. Coarser 2D grid (`resolution_2d`);
  units cm like `map_2d_size`.
"""

import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# ---- 1D track (unique coding radius) ----
track_length = 1200
resolution = 1
xs = np.arange(0, track_length, resolution)
nx = len(xs)
ref_idx = nx // 2

# ---- 2D map (directed peak-width min over 2π) ----
# Same linear extent (cm) as track; larger `resolution_2d` = fewer pixels, faster & less RAM.
map_2d_size = 1200
resolution_2d = 8
n_rays_peak = 72

# ---- Hyperparameter grid ----
field_widths_list = [
    (5, 10),
    (10, 20),
    (20, 40),
    (30, 60),
    (40, 80),
    (10, 100),
]

Np_list = [100, 500, 1000, 2000, 4000, 8000]

fields_per_cell_list = [1, 3, 5, 10, 15, 20]

n_repeats = 10


def compute_unique_radius(xs_1d, ref_ix, similarity):
    """O(n) walks; suffix/prefix maxima avoid np.max on slices each step."""
    sim = np.asarray(similarity)
    n = len(sim)
    suff = np.maximum.accumulate(sim[::-1])[::-1]
    acc = np.maximum.accumulate(sim)

    right_r = 0
    for i in range(1, n - ref_ix):
        val = sim[ref_ix + i]
        k = ref_ix + i + 1
        max_beyond = suff[k] if k < n else -np.inf
        if val > max_beyond:
            right_r = xs_1d[ref_ix + i] - xs_1d[ref_ix]
        else:
            break

    left_r = 0
    for i in range(1, ref_ix + 1):
        val = sim[ref_ix - i]
        ub = ref_ix - i
        max_beyond = acc[ub - 1] if ub > 0 else -np.inf
        if val > max_beyond:
            left_r = xs_1d[ref_ix] - xs_1d[ref_ix - i]
        else:
            break
    return min(left_r, right_r)


def directed_peak_global_min_2d(cos_map, xs_g, ys_g, ref_x, ref_y):
    """
    For each θ in [0, 2π), walk outward from (ref_x, ref_y) along +θ while
    cosine strictly decreases; return min of those flank lengths (map units = cm).
    """
    dt = float(xs_g[1] - xs_g[0]) if len(xs_g) > 1 else 1.0
    xmin, xmax = float(xs_g.min()), float(xs_g.max())
    ymin, ymax = float(ys_g.min()), float(ys_g.max())
    interp = RegularGridInterpolator(
        (xs_g, ys_g),
        cos_map,
        bounds_error=False,
        fill_value=np.nan,
    )

    def _strict_flank(c, s, sign):
        sim0 = float(interp((ref_x, ref_y)))
        if not np.isfinite(sim0):
            return 0.0
        sims = [sim0]
        ts = [0.0]
        t = 0.0
        while True:
            t += sign * dt
            x = ref_x + t * c
            y = ref_y + t * s
            if x < xmin or x > xmax or y < ymin or y > ymax:
                break
            v = float(interp((x, y)))
            if not np.isfinite(v) or not (v < sims[-1]):
                break
            sims.append(v)
            ts.append(t)
        i = 0
        while i + 1 < len(sims) and sims[i + 1] < sims[i]:
            i += 1
        return abs(ts[i] - ts[0])

    def directed_width(theta):
        c, s = np.cos(theta), np.sin(theta)
        return _strict_flank(c, s, +1)

    thetas = np.linspace(0, 2 * np.pi, n_rays_peak, endpoint=False)
    return float(min(directed_width(th) for th in thetas))


def unique_radius_global_min_2d(cos_map, xs_g, ys_g, ref_x, ref_y):
    """
    For each θ in [0, 2π), sample cosine along the line through (ref_x, ref_y)
    in ±θ (grid step dt), run compute_unique_radius on that 1D profile;
    return min over θ (map units = cm).
    """
    dt = float(xs_g[1] - xs_g[0]) if len(xs_g) > 1 else 1.0
    xmin, xmax = float(xs_g.min()), float(xs_g.max())
    ymin, ymax = float(ys_g.min()), float(ys_g.max())
    interp = RegularGridInterpolator(
        (xs_g, ys_g),
        cos_map,
        bounds_error=False,
        fill_value=np.nan,
    )

    def unique_radius_along_theta(theta):
        c, s = np.cos(theta), np.sin(theta)
        ts_neg = []
        t = 0.0
        while True:
            t -= dt
            x = ref_x + t * c
            y = ref_y + t * s
            if x < xmin or x > xmax or y < ymin or y > ymax:
                break
            v = float(interp((x, y)))
            if not np.isfinite(v):
                break
            ts_neg.append(t)
        ts_neg.reverse()

        ts_pos = []
        t = 0.0
        while True:
            t += dt
            x = ref_x + t * c
            y = ref_y + t * s
            if x < xmin or x > xmax or y < ymin or y > ymax:
                break
            v = float(interp((x, y)))
            if not np.isfinite(v):
                break
            ts_pos.append(t)

        all_t = np.concatenate(
            [
                np.array(ts_neg, dtype=np.float64),
                np.array([0.0], dtype=np.float64),
                np.array(ts_pos, dtype=np.float64),
            ]
        )
        ref_ix = len(ts_neg)
        sim = np.array(
            [float(interp((ref_x + t * c, ref_y + t * s))) for t in all_t]
        )
        return compute_unique_radius(all_t, ref_ix, sim)

    thetas = np.linspace(0, 2 * np.pi, n_rays_peak, endpoint=False)
    return float(min(unique_radius_along_theta(th) for th in thetas))


def run_one(Np, field_width_range, fields_per_cell, seed, progress=False):
    rng = np.random.RandomState(seed)

    # ----- 1D track: unique radius -----
    pop = np.zeros((nx, Np), dtype=np.float32)
    cell_iter = range(Np)
    if progress:
        cell_iter = tqdm(cell_iter, desc="    1D place cells", leave=False, unit="cell")
    for cell in cell_iter:
        n_fields = max(1, rng.poisson(fields_per_cell))
        for _ in range(n_fields):
            center = rng.uniform(0, track_length)
            sigma = rng.uniform(field_width_range[0], field_width_range[1])
            peak = rng.uniform(0.5, 1.0)
            dists = np.abs(xs - center)
            pop[:, cell] += peak * np.exp(-dists**2 / (2 * sigma**2))

    ref_vec = pop[ref_idx, :]
    ref_norm = np.linalg.norm(ref_vec)

    dots = pop @ ref_vec
    norms = np.linalg.norm(pop, axis=1)
    cos_sim_1d = np.zeros(nx, dtype=np.float64)
    if ref_norm > 0:
        active = norms > 0
        cos_sim_1d[active] = dots[active] / (ref_norm * norms[active])

    r = compute_unique_radius(xs, ref_idx, cos_sim_1d)

    # ----- 2D map: directed peak min over 2π -----
    centers = []
    cell_iter2 = range(Np)
    if progress:
        cell_iter2 = tqdm(cell_iter2, desc="    2D field sites", leave=False, unit="cell")
    for cell in cell_iter2:
        n_fields = max(1, rng.poisson(fields_per_cell))
        for _ in range(n_fields):
            cx = rng.uniform(0, map_2d_size)
            cy = rng.uniform(0, map_2d_size)
            sigma = rng.uniform(field_width_range[0], field_width_range[1])
            peak = rng.uniform(0.5, 1.0)
            centers.append((cell, cx, cy, sigma, peak))
    centers = np.array(centers)

    xs_g = np.arange(0, map_2d_size, resolution_2d, dtype=np.float64)
    ys_g = np.arange(0, map_2d_size, resolution_2d, dtype=np.float64)
    nx_g, ny_g = len(xs_g), len(ys_g)
    pop2 = np.zeros((nx_g, ny_g, Np), dtype=np.float32)

    gauss_iter = centers
    if progress and len(centers) > 0:
        gauss_iter = tqdm(centers, desc="    2D Gaussian bump", leave=False, unit="field")
    for cell_idx, cx, cy, sigma, peak in gauss_iter:
        cell = int(cell_idx)
        radius = 4 * float(sigma)
        ix_min = max(0, int((cx - radius) / resolution_2d))
        ix_max = min(nx_g, int((cx + radius) / resolution_2d) + 1)
        iy_min = max(0, int((cy - radius) / resolution_2d))
        iy_max = min(ny_g, int((cy + radius) / resolution_2d) + 1)
        xs_sub = xs_g[ix_min:ix_max, np.newaxis]
        ys_sub = ys_g[iy_min:iy_max, np.newaxis].T
        contrib = peak * np.exp(
            -((xs_sub - cx) ** 2 + (ys_sub - cy) ** 2) / (2 * sigma**2)
        ).astype(np.float32)
        pop2[ix_min:ix_max, iy_min:iy_max, cell] += contrib

    ref_x2 = map_2d_size // 2
    ref_y2 = map_2d_size // 2
    ref_ix2 = int(np.argmin(np.abs(xs_g - ref_x2)))
    ref_iy2 = int(np.argmin(np.abs(ys_g - ref_y2)))
    ref_vec2 = pop2[ref_ix2, ref_iy2, :].astype(np.float64)
    ref_norm2 = np.linalg.norm(ref_vec2)
    pop64 = pop2.astype(np.float64)
    dot_m = np.tensordot(pop64, ref_vec2, axes=([2], [0]))
    norm_v = np.linalg.norm(pop64, axis=2)
    cos_map = np.zeros((nx_g, ny_g), dtype=np.float64)
    if ref_norm2 > 0:
        m = norm_v > 0
        cos_map[m] = dot_m[m] / (ref_norm2 * norm_v[m])

    rx = float(xs_g[ref_ix2])
    ry = float(ys_g[ref_iy2])
    peak_w = directed_peak_global_min_2d(cos_map, xs_g, ys_g, rx, ry)
    ur_2d = unique_radius_global_min_2d(cos_map, xs_g, ys_g, rx, ry)

    return r, peak_w, ur_2d


if __name__ == "__main__":
    # ---- Run sweep ----
    results = []
    total_configs = len(field_widths_list) * len(Np_list) * len(fields_per_cell_list)

    sweep = itertools.product(field_widths_list, Np_list, fields_per_cell_list)
    pbar_main = tqdm(
        sweep,
        total=total_configs,
        desc="Hyperparameter sweep",
        unit="config",
    )
    for fw, Np, fpc in pbar_main:
        pbar_main.set_postfix(
            sigma=f"{fw[0]}-{fw[1]}",
            Np=Np,
            fpc=fpc,
            refresh=False,
        )
        radii = []
        peak_mins = []
        ur2d_list = []
        rep_iter = range(n_repeats)
        if n_repeats > 1:
            rep_iter = tqdm(
                rep_iter,
                desc="    repeats",
                leave=False,
                unit="run",
            )
        for rep in rep_iter:
            seed = rep * 1000 + hash((fw, Np, fpc)) % 10000
            r, pw, ur2 = run_one(Np, fw, fpc, seed=seed, progress=True)
            radii.append(r)
            peak_mins.append(pw)
            ur2d_list.append(ur2)
        mean_r = np.mean(radii)
        mean_pw = np.mean(peak_mins)
        mean_ur2d = np.mean(ur2d_list)
        results.append({
            'field_width': fw,
            'Np': Np,
            'fields_per_cell': fpc,
            'unique_radius_cm': mean_r,
            'peak_width_cm': mean_pw,
            'unique_radius_2d_cm': mean_ur2d,
        })

    # ---- Print table ----
    print(
        f"\n{'Field Width':>15}  {'Np':>6}  {'Fields/Cell':>11}  "
        f"{'Unique R 1D':>12}  {'Dir. peak':>12}  {'Unique R 2D':>12}"
    )
    print("-" * 88)
    for r in results:
        fw_str = f"{r['field_width'][0]}-{r['field_width'][1]}"
        print(
            f"{fw_str:>15}  {r['Np']:>6}  {r['fields_per_cell']:>11}  "
            f"{r['unique_radius_cm']:>12.1f}  {r['peak_width_cm']:>12.1f}  "
            f"{r['unique_radius_2d_cm']:>12.1f}"
        )

    # ---- CSV ----
    csv_path = 'unique_radius_sweep.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'field_width_min', 'field_width_max', 'Np', 'fields_per_cell',
            'unique_radius_cosine_cm', 'peak_width_cosine_cm',
            'unique_radius_2d_cm',
        ])
        for r in results:
            writer.writerow([
                r['field_width'][0], r['field_width'][1], r['Np'],
                r['fields_per_cell'],
                round(r['unique_radius_cm'], 2),
                round(r['peak_width_cm'], 2),
                round(r['unique_radius_2d_cm'], 2),
            ])
    print(f"\nResults saved to {csv_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, len(Np_list), figsize=(5 * len(Np_list), 5), sharey=True)

    for ax_idx, Np in enumerate(Np_list):
        ax = axes[ax_idx]
        mat = np.zeros((len(field_widths_list), len(fields_per_cell_list)))
        for i, fw in enumerate(field_widths_list):
            for j, fpc in enumerate(fields_per_cell_list):
                match = [r for r in results
                        if r['field_width'] == fw and r['Np'] == Np and r['fields_per_cell'] == fpc]
                mat[i, j] = match[0]['unique_radius_cm']

        im = ax.imshow(mat, aspect='auto', cmap='viridis', origin='lower',
                    vmin=0, vmax=track_length // 2)
        ax.set_xticks(range(len(fields_per_cell_list)))
        ax.set_xticklabels(fields_per_cell_list)
        ax.set_xlabel('Fields per cell')
        if ax_idx == 0:
            ax.set_yticks(range(len(field_widths_list)))
            ax.set_yticklabels([f"{fw[0]}-{fw[1]}" for fw in field_widths_list])
            ax.set_ylabel('Field width σ (cm)')
        else:
            ax.set_yticks(range(len(field_widths_list)))
            ax.set_yticklabels([])
        ax.set_title(f'Np = {Np}')

        for i in range(len(field_widths_list)):
            for j in range(len(fields_per_cell_list)):
                ax.text(j, i, f'{mat[i, j]:.0f}', ha='center', va='center',
                        color='white' if mat[i, j] < track_length // 4 else 'black', fontsize=9)

    plt.colorbar(im, ax=axes, label='Unique radius (cm)', shrink=0.8)
    fig.suptitle(f'Unique Coding Radius — Cosine Similarity — Track = {track_length} cm', fontsize=14)
    plt.tight_layout()
    plt.savefig('unique_radius_sweep.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nDone.")
