"""If the untrained encoder collapsed, why did its local oracle score 8.5 deg?

Collapse is a statement about the *global* map. A smooth untrained MLP can
still have a well-defined local Jacobian, so neighbouring cells differ in a
consistent direction even when far cells are indistinguishable. If so, the
neighbour-difference norm will be an appreciable fraction of the far-pair
difference norm -- the chart is locally fine and globally flat.
"""
import sys
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

R = "/orcd/pool/003/jackking/cls_runs"
ENCODERS = [
    ("v35 (gain 3.70)", f"{R}/encoders/run_20260422_185816/encoder_best.pt", None),
    ("L7-s42 (gain 100)",
     f"{R}/sweeps/w53_attract_knee/004_att16_seed=42/encoder_final.pt", None),
    ("untrained (gain 5)", f"{R}/encoders/untrained_mlp.pt", 0.25),
]
NPOS, N = 1716, 200
rng = np.random.RandomState(1)

print(f"{'encoder':22s} {'||dz|| 1 cell':>14s} {'||dz|| far':>11s} "
      f"{'ratio':>8s} {'cos(dN,dE)':>11s}")
print("-" * 72)
for name, path, fb in ENCODERS:
    enc, cfg, gain, fwhm, _ = load_probe_encoder(path, fwhm_fallback=fb)
    field = Field(enc, list(cfg.lambdas), fwhm, gain, NPOS)

    gx = rng.randint(2, NPOS - 2, size=N)
    gy = rng.randint(2, NPOS - 2, size=N)
    z = field.encode(gx, gy).astype(np.float64)
    zN = field.encode(gx, gy + 1).astype(np.float64)
    zE = field.encode(gx + 1, gy).astype(np.float64)

    dN, dE = zN - z, zE - z
    local = np.linalg.norm(dN, axis=1)

    # Far pairs: shuffle so each row is compared against an unrelated cell.
    far = np.linalg.norm(z - z[rng.permutation(N)], axis=1)

    # Are the two axis directions distinguishable? A collapsed-but-smooth map
    # still has a Jacobian, but if dN and dE are parallel the local frame is
    # degenerate and the projection would be reading one axis twice.
    c = np.sum(dN * dE, axis=1) / (np.linalg.norm(dN, axis=1)
                                   * np.linalg.norm(dE, axis=1) + 1e-30)

    print(f"{name:22s} {local.mean():14.5f} {far.mean():11.5f} "
          f"{local.mean() / far.mean():8.4f} {np.median(c):11.4f}")

print()
print("ratio ~1: local steps as big as global -- a real chart.")
print("ratio ~0: globally collapsed. cos(dN,dE) ~0 means the local frame still")
print("separates the two axes, which is all the tangent projection needs.")
