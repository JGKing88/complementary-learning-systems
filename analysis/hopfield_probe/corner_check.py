"""Are the L7 (gain 100) embeddings already near hypercube corners?

nav_p2 5.7 found two independent conditions for attractor retrieval:

  loop gain  beta*S/D > 1   -- beta OR storage norm, compensable
  near-corner patterns      -- storage gain g only, NOT compensable, turn-on g~100

and left one question explicitly open: at g=100 the pattern sits at cos 0.954 to
its own binarization, so "whether the tangent projection still decodes direction
from a saturated pattern is an entirely separate question, and it is the one
that matters".

The L7 encoders run at gain 100 and z = normalize(tanh(gain * net(x))), which is
structurally nav_p2's p = tanh(g * xi). So the corner condition may already be
met in production. This measures both halves on the real encoders.
"""
import sys
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np
import torch

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

R = "/orcd/pool/003/jackking/cls_runs"
ENCODERS = [
    ("v35 (gain 3.70)", f"{R}/encoders/run_20260422_185816/encoder_best.pt", None),
    ("L7-s42 (gain 100)",
     f"{R}/sweeps/w53_attract_knee/004_att16_seed=42/encoder_final.pt", None),
    ("untrained (gain 5)", f"{R}/encoders/untrained_mlp.pt", 0.25),
]
NPOS, M = 1716, 25            # 25 stored patterns, as in nav_p2's mid-range
rng = np.random.RandomState(0)


def iterate(W, x, beta, steps=20, use_tanh=True, normalize=True):
    for _ in range(steps):
        h = W @ x
        x = np.tanh(beta * h) if use_tanh else h
        if normalize:
            n = np.linalg.norm(x)
            x = x / n if n > 1e-12 else x
    return x


for name, path, fb in ENCODERS:
    enc, cfg, gain, fwhm, _hdr = load_probe_encoder(path, fwhm_fallback=fb)
    field = Field(enc, list(cfg.lambdas), fwhm, gain, NPOS)
    gx = rng.randint(0, NPOS, size=M)
    gy = rng.randint(0, NPOS, size=M)
    Z = field.encode(gx, gy).astype(np.float64)          # (M, D), unit rows
    D = Z.shape[1]

    # --- corner condition ---------------------------------------------------
    # A true corner has every |coordinate| equal. Two readings of that: how
    # close the pattern is to its own binarization, and how concentrated the
    # coordinate magnitudes are.
    B = np.sign(Z) / np.sqrt(D)
    cos_bin = np.sum(Z * B, axis=1) / (np.linalg.norm(Z, axis=1)
                                       * np.linalg.norm(B, axis=1))
    absz = np.abs(Z)
    sat = (absz > 0.5 * absz.max(axis=1, keepdims=True)).mean(axis=1)
    # participation ratio / D: 1.0 = perfectly flat (a corner), ~0 = sparse
    pr = ((absz ** 2).sum(1) ** 2 / (absz ** 4).sum(1)) / D

    print("=" * 68)
    print(f"{name}   D={D}")
    print(f"  cos to own binarization   {cos_bin.mean():.4f}   "
          f"(nav_p2 turn-on: 0.954 at g=100)")
    print(f"  coords > half-max          {sat.mean():.3f}   "
          f"(nav_p2: 83% saturated at g=100)")
    print(f"  participation ratio / D    {pr.mean():.4f}   (1.0 = flat corner)")

    # --- loop gain and fixed points ----------------------------------------
    # Production stores at unit norm, so S=1 and the loop gain is beta/D.
    print(f"  production loop gain beta*S/D = {gain:.4g}/{D} = "
          f"{gain / D:.4f}   (needs > 1)")

    print(f"  {'scale':>10s} {'beta':>9s} {'loop gain':>10s} "
          f"{'cos(iter(p),p)':>15s}")
    for scale_name, scale in (("1/D", 1.0 / D), ("1", 1.0)):
        for beta in (gain, 1.0, 3.0, 10.0):
            W = scale * (Z.T @ Z)
            np.fill_diagonal(W, 0.0)
            cs = [np.dot(iterate(W, Z[k].copy(), beta), Z[k]) for k in range(M)]
            loop = beta * scale * D          # beta * S/D with S=1, scale=s
            print(f"  {scale_name:>10s} {beta:9.4g} {loop:10.3f} "
                  f"{np.median(cs):15.4f}")
