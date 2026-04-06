"""
Place Cell Unique Coding Radius — Hyperparameter Sweep
=======================================================
Cosine similarity only. Reports unique coding radius over
combinations of field_width, Np, and fields_per_cell.
"""

import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt

# ---- Environment ----
track_length = 1200
resolution = 1
xs = np.arange(0, track_length, resolution)
nx = len(xs)
ref_idx = nx // 2

# ---- Hyperparameter grid ----
field_widths_list = [
    (5, 10),
    (10, 20),
    (20, 40),
    (30, 60),
    (40, 80),
]

Np_list = [100, 500, 1000, 2000, 4000, 6000]

fields_per_cell_list = [1, 3, 5, 10, 15, 20]

n_repeats = 20


def compute_unique_radius(xs, ref_idx, similarity):
    # Walk right
    right_r = 0
    for i in range(1, len(xs) - ref_idx):
        val = similarity[ref_idx + i]
        max_beyond = np.max(similarity[ref_idx + i + 1:]) if ref_idx + i + 1 < len(xs) else -np.inf
        if val > max_beyond:
            right_r = xs[ref_idx + i] - xs[ref_idx]
        else:
            break
    # Walk left
    left_r = 0
    for i in range(1, ref_idx + 1):
        val = similarity[ref_idx - i]
        max_beyond = np.max(similarity[:ref_idx - i]) if ref_idx - i > 0 else -np.inf
        if val > max_beyond:
            left_r = xs[ref_idx] - xs[ref_idx - i]
        else:
            break
    return min(left_r, right_r)


def run_one(Np, field_width_range, fields_per_cell, seed):
    rng = np.random.RandomState(seed)

    pop = np.zeros((nx, Np), dtype=np.float32)
    for cell in range(Np):
        n_fields = max(1, rng.poisson(fields_per_cell))
        for _ in range(n_fields):
            center = rng.uniform(0, track_length)
            sigma = rng.uniform(field_width_range[0], field_width_range[1])
            peak = rng.uniform(0.5, 1.0)
            dists = np.abs(xs - center)
            pop[:, cell] += peak * np.exp(-dists**2 / (2 * sigma**2))

    ref_vec = pop[ref_idx, :]
    ref_norm = np.linalg.norm(ref_vec)

    cos_sim = np.zeros(nx)
    for i in range(nx):
        v = pop[i, :]
        n = np.linalg.norm(v)
        if ref_norm > 0 and n > 0:
            cos_sim[i] = np.dot(ref_vec, v) / (ref_norm * n)

    r = compute_unique_radius(xs, ref_idx, cos_sim)
    return r

if __name__ == "__main__":
    # ---- Run sweep ----
    results = []
    total = len(field_widths_list) * len(Np_list) * len(fields_per_cell_list)
    count = 0

    for fw, Np, fpc in itertools.product(field_widths_list, Np_list, fields_per_cell_list):
        radii = []
        for rep in range(n_repeats):
            r = run_one(Np, fw, fpc, seed=rep * 1000 + hash((fw, Np, fpc)) % 10000)
            radii.append(r)
        mean_r = np.mean(radii)
        results.append({
            'field_width': fw,
            'Np': Np,
            'fields_per_cell': fpc,
            'unique_radius_cm': mean_r,
        })
        count += 1
        if count % 10 == 0:
            print(f"  {count}/{total} done...")

    print(f"  {total}/{total} done.")

    # ---- Print table ----
    print(f"\n{'Field Width':>15}  {'Np':>6}  {'Fields/Cell':>11}  {'Cosine Radius (cm)':>18}")
    print("-" * 58)
    for r in results:
        fw_str = f"{r['field_width'][0]}-{r['field_width'][1]}"
        print(f"{fw_str:>15}  {r['Np']:>6}  {r['fields_per_cell']:>11}  {r['unique_radius_cm']:>18.1f}")

    # ---- CSV ----
    csv_path = 'unique_radius_sweep.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['field_width_min', 'field_width_max', 'Np', 'fields_per_cell', 'unique_radius_cosine_cm'])
        for r in results:
            writer.writerow([r['field_width'][0], r['field_width'][1], r['Np'],
                            r['fields_per_cell'], round(r['unique_radius_cm'], 2)])
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