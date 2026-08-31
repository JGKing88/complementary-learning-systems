"""Why doesn't the recall settle? It does — as power iteration, very slowly.

Jack's objection: a symmetric-weight Hopfield has an energy function, so it has
to reach a fixed point (or a 2-cycle under synchronous update). Measuring only
12 steps and reporting "it never converges" was the wrong conclusion from the
right data.

The suspicion this tests. `Hopfield` builds `W = (1/D) Σ ξξᵀ` with unit-norm
patterns and `zero_diag`, and updates `X ← normalize(tanh(β W X))` with β = 5,
D = 1024. For a cue near a stored pattern, `‖W x‖ ≈ 1/D ≈ 0.001`, so the tanh
argument is ≈ 0.005 — deep in the **linear** region of tanh, where
`tanh(u) ≈ u`. If that holds, the update reduces to `X ← normalize(W X)`, which
is **power iteration**, and the nonlinearity that makes a Hopfield network a
Hopfield network is inert.

That would explain everything measured: with ~11 near-orthogonal patterns the
top eigenvalues of W are near-degenerate, so power iteration converges at rate
(λ₂/λ₁)^k with λ₂/λ₁ ≈ 1 — a slow monotone drift away from the cue and toward
the leading eigenvector, which is a blend. It converges; twelve steps is
nowhere near enough to see it.

Four checks, each of which could falsify it:

  1. the tanh pre-activation magnitude — is it actually in the linear region?
  2. tanh vs no-tanh trajectories — if the nonlinearity is inert they coincide
  3. run to many steps — does it settle, and onto the top eigenvector of W?
  4. the spectrum — is λ₂/λ₁ close to 1, and does it predict the residual decay?

    python -m analysis.nav_p2.recall_dynamics --ckpt <any nav ckpt>
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint,
)
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_distractors", type=int, default=10)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--cells", type=int, default=256)
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
    D, beta = enc_cfg.out_dim, cfg.hopfield.beta
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    envs, vh, offsets = build_eval_world(cfg, encoder, str(device), ckpt_path=None)

    env, off, goal, size = envs[0], offsets[0], envs[0].goal_location, envs[0].size
    rng = np.random.RandomState(args.seed)
    cellsx = rng.randint(0, size, args.cells)
    cellsy = rng.randint(0, size, args.cells)
    cells = np.stack([cellsx, cellsy], 1).astype(np.int32)
    emb = torch.from_numpy(vh.get_encoded_state(cells, off)).float().to(device)

    g_pat = goal_encoding(vh, off, goal)
    d_pats = sample_distractors(vh, off, size, args.n_distractors, rng)
    hop = Hopfield(D, beta=beta, device=str(device))
    for pat in [g_pat] + list(d_pats):
        hop.input_memory(torch.from_numpy(pat).float())
    g_t = torch.nn.functional.normalize(
        torch.from_numpy(g_pat).float().to(device), dim=-1)

    print(f"D={D}  beta={beta}  scale=1/D={hop.scale:.3g}  "
          f"zero_diag={hop.zero_diag}  patterns={hop.num_memories}\n")

    # --- check 1: is tanh in its linear region? ---------------------------
    pre = beta * (emb @ hop.W.T)
    a = pre.abs()
    print("CHECK 1 -- tanh pre-activation |beta * W x|")
    print(f"  median {a.median():.3e}   p99 {a.flatten().quantile(0.99):.3e}   "
          f"max {a.max():.3e}")
    lin_err = (torch.tanh(pre) - pre).abs().max() / pre.abs().max()
    print(f"  max relative deviation of tanh(u) from u: {lin_err:.3e}")
    print(f"  -> tanh is {'INERT (linear regime)' if lin_err < 1e-3 else 'ACTIVE'}\n")

    # --- check 2: tanh vs no tanh -----------------------------------------
    def run(use_tanh, steps):
        X = emb.clone()
        for _ in range(steps):
            H = X @ hop.W.T
            X = torch.tanh(beta * H) if use_tanh else H
            X = torch.nn.functional.normalize(X, dim=-1)
        return X
    A, B = run(True, 12), run(False, 12)
    print("CHECK 2 -- tanh vs no-tanh after 12 steps")
    print(f"  median cos(with, without) = "
          f"{(torch.nn.functional.normalize(A,dim=-1) * torch.nn.functional.normalize(B,dim=-1)).sum(-1).median():.8f}\n")

    # --- check 4 first: the spectrum predicts the rate ---------------------
    evals = torch.linalg.eigvalsh(hop.W.double())
    top = evals.flip(0)[:14].cpu().numpy()
    print("CHECK 4 -- top eigenvalues of W")
    print("  " + "  ".join(f"{v:.5e}" for v in top[:6]))
    ratio = top[1] / top[0]
    print(f"  lambda2/lambda1 = {ratio:.6f}   -> power-iteration error decays "
          f"like {ratio:.4f}^k")
    print(f"  steps for a 10x reduction: {np.log(0.1)/np.log(ratio):.0f}\n")

    # --- check 3: run long, does it settle, and onto what? ----------------
    w, V = torch.linalg.eigh(hop.W.double())
    v1 = V[:, -1].float().to(device)
    print("CHECK 3 -- long run")
    print(f"{'step':>7} {'cos to goal':>12} {'cos to top eigvec':>18} {'residual':>12}")
    X = emb.clone()
    prev = X.clone()
    for s in range(1, args.steps + 1):
        H = X @ hop.W.T
        X = torch.nn.functional.normalize(torch.tanh(beta * H), dim=-1)
        if s in (1, 2, 5, 12, 50, 200, 800, 2000, args.steps):
            cg = (torch.nn.functional.normalize(X, dim=-1) @ g_t).median()
            cv = (torch.nn.functional.normalize(X, dim=-1) @ v1).abs().median()
            r = (X - prev).norm(dim=-1).median()
            print(f"{s:>7} {cg:>12.4f} {cv:>18.4f} {r:>12.3e}")
        prev = X.clone()


if __name__ == "__main__":
    main()
