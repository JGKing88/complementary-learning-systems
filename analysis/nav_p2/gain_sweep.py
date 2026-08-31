"""Why are there no basins — not even spurious ones? Gain, or normalization?

Jack's question. Two candidate causes, and they are separable by experiment.

**Gain.** A continuous Hopfield network has a classic phase transition in the
neuron gain (Hopfield 1984). Below a critical gain the only fixed point is the
trivial one; above it, the memory states nucleate and acquire basins. Basins in
a Hopfield net are *made by saturation* — `sign()` in the binary case, a
saturating `tanh` in the continuous one — because saturation is what maps a
whole neighbourhood onto the same corner. Spurious mixture states are made the
same way: `sign(ξ¹+ξ²+ξ³)` is a fixed point only because `sign` maps it to
itself. With the nonlinearity inert there is no mechanism to create *any*
additional stable state, memory or spurious. A symmetric linear map has exactly
one stable direction, its top eigenvector, and every other eigenvector is a
saddle. §5.4 measured the tanh argument here at ~1e-4 against the order-1 it
needs, so this network sits well below the transition.

**Normalization.** This is the other suspect, and it does something real, but
something different. Without it, `x ← W x` with `λ₁ ≈ 1.4e-3 < 1` sends every
state to **zero** — the origin is the only attractor and the state decays.
Normalizing keeps the state on the sphere, so the degenerate outcome becomes a
*direction* (the top eigenvector) instead of nothing. On this reading
normalization does not destroy basins; it changes what the single attractor is
when there are no basins to begin with.

The prediction, then: raise the gain enough and basins appear **with**
normalization still on. If instead basins only appear once normalization is
removed, the diagnosis is wrong.

Sweeps beta over orders of magnitude, with normalization on and off, measuring
at each point:

  * whether a stored pattern is a fixed point (cos to its own start after 200)
  * how many distinct attractors 256 random starts find
  * whether a cue corrupted to cos 0.70 is restored, and stays restored
  * the elementwise |beta * W x|, so the saturation onset is visible

    python -m analysis.nav_p2.gain_sweep --ckpt <any nav ckpt>
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


def iterate(W, X, beta, steps, normalize):
    for _ in range(steps):
        X = torch.tanh(beta * (X @ W.T))
        if normalize:
            X = F.normalize(X, dim=-1)
    return X


def n_distinct(X, tol=0.99):
    """Distinct limits up to sign, or -1 if the state collapsed to the origin.

    Without normalization and below the gain transition every state decays to
    zero. `F.normalize` maps zero to zero, so every pair has cosine 0 and each
    of the N starts counts as its own "attractor" -- an artifact that reads as
    a rich attractor landscape when it is the exact opposite. Detect the
    collapse and say so instead.
    """
    if X.norm(dim=-1).median() < 1e-6:
        return -1
    G = (F.normalize(X, dim=-1) @ F.normalize(X, dim=-1).T).abs()
    reps: list[int] = []
    for i in range(X.shape[0]):
        if not any(G[i, j] > tol for j in reps):
            reps.append(i)
        if len(reps) > 60:
            break
    return len(reps)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_distractors", type=int, default=10)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--starts", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = 1
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device),
        getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    D = enc_cfg.out_dim
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    envs, vh, offsets = build_eval_world(cfg, encoder, str(device), ckpt_path=None)
    env, off, goal, size = envs[0], offsets[0], envs[0].goal_location, envs[0].size

    rng = np.random.RandomState(args.seed)
    g_pat = goal_encoding(vh, off, goal)
    d_pats = sample_distractors(vh, off, size, args.n_distractors, rng)
    stored = np.stack([g_pat] + list(d_pats))
    hop = Hopfield(D, beta=cfg.hopfield.beta, device=str(device))
    for pat in stored:
        hop.input_memory(torch.from_numpy(pat).float())
    W = hop.W
    P = F.normalize(torch.from_numpy(stored).float().to(device), dim=-1)
    M = P.shape[0]

    noise = F.normalize(torch.randn(M, D, device=device), dim=-1)
    noise = F.normalize(noise - (noise * P).sum(-1, keepdim=True) * P, dim=-1)
    CORRUPT = F.normalize(0.70 * P + np.sqrt(1 - 0.49) * noise, dim=-1)
    R0 = F.normalize(torch.randn(args.starts, D, device=device), dim=-1)

    print(f"{M} stored patterns, D={D}, trainer beta={cfg.hopfield.beta}, "
          f"1/D={1/D:.3e}")
    print(f"max |overlap| between stored: {(P @ P.T).abs().fill_diagonal_(0).max():.4f}\n")

    betas = [5.0, 5e1, 5e2, 2e3, 5e3, 2e4, 1e5, 1e6]
    for norm in (True, False):
        print(f"=== normalize_each = {norm} " + "=" * 44)
        print(f"{'beta':>9} {'|b*Wx| med':>11} {'stored is':>10} "
              f"{'#attractors':>12} {'corrupted':>10} {'cos to':>8}")
        print(f"{'':>9} {'':>11} {'fixed pt':>10} {'from 256':>12} "
              f"{'restored?':>10} {'nearest':>8}")
        for b in betas:
            pre = (b * (P @ W.T)).abs()
            stab = iterate(W, P.clone(), b, args.steps, norm)
            stab_cos = (F.normalize(stab, dim=-1) * P).sum(-1).abs().median()
            lim = iterate(W, R0.clone(), b, args.steps, norm)
            k = n_distinct(lim)
            fix = iterate(W, CORRUPT.clone(), b, args.steps, norm)
            fix_cos = (F.normalize(fix, dim=-1) * P).sum(-1).abs().median()
            near = (F.normalize(lim, dim=-1) @ P.T).abs().max(1).values.median()
            kd = "collapsed" if k < 0 else str(k)
            print(f"{b:>9.0f} {pre.median():>11.3e} {stab_cos:>10.4f} "
                  f"{kd:>12} {fix_cos:>10.4f} {near:>8.4f}")
        print()

    print("Reading: 'stored is fixed pt' near 1.0 and '#attractors' near "
          f"{M}+ means basins exist.")
    print("A single attractor with 'stored is fixed pt' low means no basins.")


if __name__ == "__main__":
    main()
