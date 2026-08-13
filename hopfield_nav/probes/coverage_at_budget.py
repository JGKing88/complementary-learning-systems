"""Reference families at the eval's exact budget: 20x20, 400 steps, stride 1.

The 0.56 memoryless figure in section 0 was computed at a different budget, and
section 3t now leans on it as the target for a per-trial number measured at 400
steps. Same simulator, matched budget, so the comparison is exact.
"""
import numpy as np
from hopfield_nav.world.walks import family_positions, REFERENCE_FAMILIES, lawnmower_coverage

SIZE, STEPS, B = 20, 400, 256
rng = np.random.RandomState(0)
print(f"{SIZE}x{SIZE} = {SIZE*SIZE} cells, {STEPS} steps, stride 1, B={B}\n")
print(f"{'family':>12} {'coverage':>10} {'cells/step':>11}")
for fam in REFERENCE_FAMILIES:
    if fam == "lawnmower":
        cov = lawnmower_coverage(SIZE, STEPS)
    else:
        pos = family_positions(fam, B, SIZE, STEPS, 1.0, 0.5, rng)
        cells = {tuple(np.round(p).astype(int)) for _ in [0] for p in []}
        # coverage per trial = unique visited cells / total cells
        q = np.round(pos).astype(int)          # (B, T, 2)
        cov = np.mean([len(set(map(tuple, q[b]))) / (SIZE * SIZE) for b in range(q.shape[0])])
    print(f"{fam:>12} {cov:>10.3f} {cov*SIZE*SIZE/STEPS:>11.3f}")
