"""Why does a near-exact corner still fail to be a fixed point?

For unit-norm patterns, `W z_1 = (1/D)[z_1 + sum_{k!=1} c_k z_k] - diag term`.
If the self term dominates coordinate-wise then sign(W z_1) = sign(z_1) and a
corner pattern is a fixed point of the saturated dynamics. Measured at gain
10000 the pattern is at cos 0.9988 to its binarisation and still lands at 0.15,
so the cross-talk term must be winning. This measures it directly.
"""
import sys
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np
import torch

from analysis.hopfield_probe.encode import grid_codes
from analysis.hopfield_probe.harness import load_probe_encoder

R = "/orcd/pool/003/jackking/cls_runs"
enc, cfg_m, g0, fwhm, _ = load_probe_encoder(
    f"{R}/sweeps/w53_attract_knee/004_att16_seed=42/encoder_final.pt")
rng = np.random.RandomState(0)
gx, gy = rng.randint(0, 1716, 25), rng.randint(0, 1716, 25)
codes = torch.from_numpy(grid_codes(list(cfg_m.lambdas), gx, gy, fwhm))

print(f"{'gain':>8s} {'pairwise cos':>23s} {'eff rank':>9s} {'sign kept':>10s}")
print(f"{'':8s} {'mean':>7s} {'median':>7s} {'max':>7s}")
print("-" * 56)
for g in (100.0, 300.0, 1000.0, 3000.0, 10000.0):
    with torch.no_grad():
        z = torch.nn.functional.normalize(torch.tanh(g * enc.net(codes)),
                                          dim=-1)
    Z = z.numpy().astype(np.float64)
    G = Z @ Z.T
    off = G[np.triu_indices(25, 1)]
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    eff = (ev.sum() ** 2) / (ev ** 2).sum()

    W = (Z.T @ Z) / Z.shape[1]
    np.fill_diagonal(W, 0.0)
    # Fraction of coordinates where one application preserves the sign. That is
    # exactly what a saturated step needs in order to return the same corner.
    kept = np.mean([np.mean(np.sign(W @ Z[k]) == np.sign(Z[k]))
                    for k in range(25)])
    print(f"{g:8.0f} {off.mean():7.4f} {np.median(off):7.4f} {off.max():7.4f} "
          f"{eff:9.2f} {kept:10.3f}")

print("\n'sign kept' is the fraction of coordinates where sign(Wz) = sign(z).")
print("A saturated step returns sign(Wz)/sqrt(D), so the fixed-point cosine is")
print("about 2*kept - 1. Anything near 0.5 is a coin flip and there is no")
print("fixed point no matter how exact the corner.")
