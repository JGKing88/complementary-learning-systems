"""Is the Hopfield net broken, or is it being asked for the wrong thing?

Jack: "so classical continuous Hopfield networks just don't work?"

The finding in §5.6 was that no gain makes *these* patterns attractors, because
a saturating tanh has its fixed points at hypercube corners while the stored
patterns are continuous encoder outputs. That is a claim about a **mismatch**,
not about Hopfield networks, and the difference is testable: change only the
pattern type, or only the architecture, and see which one repairs it.

Three conditions, same 11 patterns, same D, same code path where possible:

  A. **classical + binarized patterns** — store `sign(ξ)` instead of `ξ`, at a
     gain high enough to saturate. If the corner story is right, the stored
     patterns become fixed points and basins appear. This is the classical
     model used as designed, and it is the control that decides whether the
     network is broken or merely mismatched.

  B. **classical + continuous patterns** — the current setup at high gain. §5.6
     says this fails; repeated here as the baseline.

  C. **modern / dense associative memory** — `ξ ← Xᵀ softmax(β X ξ)`
     (Ramsauer et al. 2020). Its fixed points are the stored patterns *by
     construction*, for continuous patterns, with exponential capacity. If this
     retrieves cleanly where A and B differ, then the architecture, not the
     idea of associative memory, is what is limiting this project.

Also reports the classical capacity reference: ~0.138·D for uncorrelated binary
patterns is 141 here, so 11 patterns is far under capacity — whatever is going
wrong, it is not capacity.

    python -m analysis.nav_p2.architecture_test --ckpt <any nav ckpt>
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


def n_distinct(X, tol=0.99, cap=60):
    if X.norm(dim=-1).median() < 1e-6:
        return -1
    G = (F.normalize(X, dim=-1) @ F.normalize(X, dim=-1).T).abs()
    reps: list[int] = []
    for i in range(X.shape[0]):
        if not any(G[i, j] > tol for j in reps):
            reps.append(i)
        if len(reps) > cap:
            break
    return len(reps)


def classical(P, beta, X, steps, normalize=True):
    """X <- normalize(tanh(beta W X)), W = (1/D) sum p p^T, zero diagonal."""
    D = P.shape[1]
    W = (P.T @ P) / D
    W.fill_diagonal_(0.0)
    for _ in range(steps):
        X = torch.tanh(beta * (X @ W.T))
        if normalize:
            X = F.normalize(X, dim=-1)
    return X


def modern(P, beta, X, steps):
    """X <- P^T softmax(beta P X) -- fixed points are the stored patterns."""
    for _ in range(steps):
        X = torch.softmax(beta * (X @ P.T), dim=-1) @ P
        X = F.normalize(X, dim=-1)
    return X


def report(name, run, P, betas, corrupt, rand):
    print(f"=== {name} " + "=" * (58 - len(name)))
    print(f"{'beta':>10} {'stored is':>10} {'#attractors':>12} "
          f"{'corrupted 0.70':>15} {'limit vs':>9}")
    print(f"{'':>10} {'fixed pt':>10} {'from starts':>12} "
          f"{'restored to':>15} {'nearest':>9}")
    for b in betas:
        stab = run(P, b, P.clone(), 200)
        s = (F.normalize(stab, dim=-1) * P).sum(-1).abs().median()
        fix = run(P, b, corrupt.clone(), 200)
        f = (F.normalize(fix, dim=-1) * P).sum(-1).abs().median()
        lim = run(P, b, rand.clone(), 200)
        k = n_distinct(lim)
        near = (F.normalize(lim, dim=-1) @ P.T).abs().max(1).values.median()
        kd = "collapsed" if k < 0 else str(k)
        print(f"{b:>10.4g} {s:>10.4f} {kd:>12} {f:>15.4f} {near:>9.4f}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_distractors", type=int, default=10)
    p.add_argument("--starts", type=int, default=256)
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
    D = enc_cfg.out_dim
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    envs, vh, offsets = build_eval_world(cfg, encoder, str(device), ckpt_path=None)
    env, off, goal, size = envs[0], offsets[0], envs[0].goal_location, envs[0].size

    rng = np.random.RandomState(args.seed)
    g_pat = goal_encoding(vh, off, goal)
    d_pats = sample_distractors(vh, off, size, args.n_distractors, rng)
    stored = np.stack([g_pat] + list(d_pats))
    P = F.normalize(torch.from_numpy(stored).float().to(device), dim=-1)
    M = P.shape[0]
    Pb = F.normalize(torch.sign(P), dim=-1)           # binarized

    ovc = (P @ P.T).abs().clone(); ovc.fill_diagonal_(0)
    ovb = (Pb @ Pb.T).abs().clone(); ovb.fill_diagonal_(0)
    print(f"{M} patterns, D={D}")
    print(f"classical capacity reference 0.138*D = {0.138 * D:.0f} patterns "
          f"-- {M} is far under it, so capacity is not the issue")
    print(f"max |overlap|  continuous {ovc.max():.4f}   binarized {ovb.max():.4f}")
    print(f"median cos(pattern, sign(pattern)) = "
          f"{(P * Pb).sum(-1).median():.4f}   <- how far binarizing moves them\n")

    def mk(Q):
        noise = F.normalize(torch.randn(M, D, device=device), dim=-1)
        noise = F.normalize(noise - (noise * Q).sum(-1, keepdim=True) * Q, dim=-1)
        return F.normalize(0.70 * Q + np.sqrt(0.51) * noise, dim=-1)

    R = F.normalize(torch.randn(args.starts, D, device=device), dim=-1)
    hi = [5.0, 5e3, 1e5, 1e6, 1e7]

    report("B. classical + CONTINUOUS patterns (the current setup)",
           classical, P, hi, mk(P), R)
    report("A. classical + BINARIZED patterns (the model as designed)",
           classical, Pb, hi, mk(Pb), R)
    report("C. modern / dense associative memory + continuous patterns",
           modern, P, [1.0, 8.0, 32.0, 128.0, 512.0], mk(P), R)

    print(f"Reading: 'stored is fixed pt' ~1.0 with ~{M} attractors and a "
          "corrupted cue restored\nto ~1.0 means the architecture stores these "
          "patterns as attractors.")


if __name__ == "__main__":
    main()
