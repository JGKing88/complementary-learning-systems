"""Are the stored patterns attractors? The test that decides the label.

Jack's question: why call this a linear associative memory rather than an
attractor network? A linear map on the sphere, `x → normalize(W x)`, does have
attractors — so "it is linear" is not by itself an answer, and the distinction
has to be made on something measurable.

The criterion that matters for memory is **whether each stored pattern is
itself a stable fixed point with a basin around it.** That is what lets a
noisy or partial cue flow to the pattern it came from, which is the entire
functional point of an attractor memory. A network storing 11 patterns should
have at least 11 stable states, one per pattern.

Three tests, none of which depends on the word "linear":

  1. **Stability.** Start exactly at a stored pattern and iterate. A stable
     fixed point stays put. Anything else is not an attractor.
  2. **Pattern completion.** Start at a corrupted stored pattern. Does one step
     move it *toward* the clean pattern? Do further steps keep improving it?
     Cleanup that happens once and then reverses is a filter, not a basin.
  3. **How many attractors are there?** Iterate from many random starts and
     count distinct limits, up to sign. An attractor memory with 11 patterns
     should show ~11 (plus spurious mixtures). One limit means one attractor,
     and then the memory cannot live in the dynamics.

    python -m analysis.nav_p2.attractor_test --ckpt <any nav ckpt>
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint,
)
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors


def step(hop, X, beta, use_tanh=True):
    H = X @ hop.W.T
    return F.normalize(torch.tanh(beta * H) if use_tanh else H, dim=-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_distractors", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = 1
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    D, beta = enc_cfg.out_dim, cfg.hopfield.beta
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    envs, vh, offsets = build_eval_world(cfg, encoder, str(device), ckpt_path=None)
    env, off, goal, size = envs[0], offsets[0], envs[0].goal_location, envs[0].size

    rng = np.random.RandomState(args.seed)
    g_pat = goal_encoding(vh, off, goal)
    d_pats = sample_distractors(vh, off, size, args.n_distractors, rng)
    stored_np = np.stack([g_pat] + list(d_pats))
    hop = Hopfield(D, beta=beta, device=str(device))
    for pat in stored_np:
        hop.input_memory(torch.from_numpy(pat).float())
    P = F.normalize(torch.from_numpy(stored_np).float().to(device), dim=-1)  # (M,D)
    M = P.shape[0]
    print(f"{M} stored patterns, D={D}, beta={beta}")
    ov = (P @ P.T).abs()
    ov.fill_diagonal_(0)
    print(f"max |overlap| between stored patterns: {ov.max():.4f}\n")

    # --- 1. Are the stored patterns stable? -------------------------------
    print("TEST 1 -- start exactly AT each stored pattern and iterate")
    print(f"{'steps':>7} {'median cos to its own start':>29}")
    X = P.clone()
    for s in range(1, 201):
        X = step(hop, X, beta)
        if s in (1, 2, 5, 20, 50, 200):
            print(f"{s:>7} {(X * P).sum(-1).abs().median():>29.4f}")
    print("  a stable fixed point would hold at 1.0000\n")

    # --- 2. Does it complete a corrupted pattern? -------------------------
    print("TEST 2 -- start at a CORRUPTED stored pattern (cos 0.70 to clean)")
    noise = F.normalize(torch.randn(M, D, device=device), dim=-1)
    noise = F.normalize(noise - (noise * P).sum(-1, keepdim=True) * P, dim=-1)
    X = F.normalize(0.70 * P + np.sqrt(1 - 0.70 ** 2) * noise, dim=-1)
    print(f"{'steps':>7} {'median cos to the clean pattern':>33}")
    print(f"{0:>7} {(X * P).sum(-1).abs().median():>33.4f}")
    for s in range(1, 201):
        X = step(hop, X, beta)
        if s in (1, 2, 3, 5, 20, 200):
            print(f"{s:>7} {(X * P).sum(-1).abs().median():>33.4f}")
    print("  a basin would climb toward 1.0 and stay\n")

    # --- 3. How many distinct attractors are there? -----------------------
    print("TEST 3 -- 512 random starts, iterate 400 steps, count distinct limits")
    X = F.normalize(torch.randn(512, D, device=device), dim=-1)
    for _ in range(400):
        X = step(hop, X, beta)
    G = (X @ X.T).abs()                       # up to sign
    seen, reps = [], []
    for i in range(X.shape[0]):
        if not any(G[i, j] > 0.99 for j in reps):
            reps.append(i)
    print(f"  distinct limits found: {len(reps)}")
    print(f"  stored patterns:       {M}")
    w, V = torch.linalg.eigh(hop.W.double())
    v1 = V[:, -1].float().to(device)
    print(f"  median |cos(limit, top eigenvector of W)| : "
          f"{(X @ v1).abs().median():.6f}")
    print(f"  median max |cos(limit, any stored pattern)| : "
          f"{(X @ P.T).abs().max(dim=1).values.median():.4f}")


if __name__ == "__main__":
    main()
