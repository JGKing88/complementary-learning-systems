"""Does pushing attract_lambda 16 -> 32 buy radius at the cost of orthogonality?

That is the question the probe run cannot answer on its own. Sec 4 and Sec 7 of
the results doc argue cross-talk is what binds retrieval, and `unique_radius`
sees it only weakly through the alias ceiling -- so an arm can improve r_min and
still be worse for the readout.

Also reports the stored 20-reference radius so both quantities sit together.
"""
import sys
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np
import torch

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
ENC = [
    ("att16-s42", f"{S}/w53_attract_knee/004_att16_seed=42/encoder_final.pt"),
    ("att16-s43", f"{S}/w53_attract_knee/005_att16_seed=43/encoder_final.pt"),
    ("att32-s42", f"{S}/w54_attract_far/000_att32_seed=42/encoder_final.pt"),
    ("att32-s43", f"{S}/w54_attract_far/001_att32_seed=43/encoder_final.pt"),
]
NPOS, M = 1716, 25
rng = np.random.RandomState(0)
gx, gy = rng.randint(0, NPOS, M), rng.randint(0, NPOS, M)

print("Pattern geometry, 25 random scaffold cells, D=1024\n")
print(f"{'encoder':11s} {'pairwise: mean':>15s} {'median':>8s} {'worst':>7s} "
      f"{'eff rank':>9s} {'cos_bin':>8s} | {'r_min':>6s} {'r_med':>6s} "
      f"{'alias':>6s}")
print("-" * 92)
for lab, p in ENC:
    enc, cfg, gain, fwhm, _ = load_probe_encoder(p)
    field = Field(enc, list(cfg.lambdas), fwhm, gain, NPOS)
    Z = field.encode(gx, gy).astype(np.float64)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)

    G = Z @ Z.T
    off = G[np.triu_indices(M, 1)]
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    eff = (ev.sum() ** 2) / (ev ** 2).sum()

    B = np.sign(Z) / np.sqrt(Z.shape[1])
    cb = float(np.mean(np.sum(Z * B, 1) / (np.linalg.norm(Z, axis=1)
                                           * np.linalg.norm(B, axis=1))))

    ur = torch.load(p, map_location="cpu", weights_only=False)["unique_radius"]
    print(f"{lab:11s} {off.mean():15.4f} {np.median(off):8.4f} "
          f"{off.max():7.4f} {eff:9.2f} {cb:8.4f} | "
          f"{ur['r_min']:6.1f} {ur['r_median']:6.1f} "
          f"{ur['alias_ceiling_max']:6.3f}")

print("\nr_* here are the 20-reference numbers stored in the checkpoints, which")
print("EXPERIMENTS_UNIQUE_RADIUS.md says have been overturned four times. They")
print("rank arms; they are not the 100-reference headline.")
