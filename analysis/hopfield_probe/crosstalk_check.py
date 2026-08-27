"""Is the untrained encoder's "fixed point" real, or is it one vector stored 25x?

A pattern set that has collapsed to a single direction is trivially made of
fixed points: W is then rank-1 along that direction and every pattern IS the
top eigenvector. That would make cos(iterate(p), p) ~ 1 for a reason that has
nothing to do with memory.
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
NPOS, M = 1716, 25
rng = np.random.RandomState(0)

print(f"{'encoder':22s} {'pairwise cos':>26s} {'top-eig':>9s} {'rank':>6s} "
      f"{'cos(p, v1)':>11s}")
print(f"{'':22s} {'mean':>8s} {'median':>8s} {'max':>8s} {'lam1/sum':>9s} "
      f"{'eff':>6s} {'median':>11s}")
print("-" * 80)

for name, path, fb in ENCODERS:
    enc, cfg, gain, fwhm, _ = load_probe_encoder(path, fwhm_fallback=fb)
    field = Field(enc, list(cfg.lambdas), fwhm, gain, NPOS)
    gx = rng.randint(0, NPOS, size=M)
    gy = rng.randint(0, NPOS, size=M)
    Z = field.encode(gx, gy).astype(np.float64)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)

    G = Z @ Z.T
    iu = np.triu_indices(M, 1)
    off = G[iu]

    # Spectrum of the Gram matrix: how many directions the pattern set spans.
    ev = np.linalg.eigvalsh(G)[::-1]
    ev = np.clip(ev, 0, None)
    frac1 = ev[0] / ev.sum()
    eff_rank = (ev.sum() ** 2) / (ev ** 2).sum()

    # Where each pattern sits relative to the top eigenvector of W.
    w, v = np.linalg.eigh(Z.T @ Z)
    v1 = v[:, -1]
    cos_v1 = np.abs(Z @ v1)

    print(f"{name:22s} {off.mean():8.4f} {np.median(off):8.4f} "
          f"{off.max():8.4f} {frac1:9.4f} {eff_rank:6.2f} "
          f"{np.median(cos_v1):11.4f}")

print()
print("A collapsed set reads: pairwise cos ~1, lam1/sum ~1, eff rank ~1,")
print("cos(p, v1) ~1 -- every pattern IS the top eigenvector, so 'fixed point'")
print("is vacuous. A healthy set reads: pairwise cos ~0, eff rank ~M.")
