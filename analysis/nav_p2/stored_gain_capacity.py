"""Store tanh(g * xi) instead of xi. At which gain and capacity are they fixed points?

Jack's experiment. Everything before this swept the gain of the *dynamics* and
stored the raw encoded patterns. This applies the nonlinearity **before**
storage — `p = tanh(g * xi)` — and asks for which storage gain `g` and how many
memories `M` those stored patterns are fixed points of the recall.

Why it should work where the earlier sweep did not. §5.6 found that a
saturating tanh has its fixed points at hypercube corners, and the raw encoded
patterns are continuous vectors that are not corners, so no dynamics gain could
make them fixed points. Pre-saturating the patterns moves them *to* corners:
`tanh(g * xi) -> sign(xi)` as `g` grows. So `g` interpolates continuously
between "the encoder's real vector" and "its binarization", and the question is
how little of the encoder's structure has to be given up before the memory
becomes a genuine attractor.

There is a scale subtlety that matters more than the gain. `Hopfield.input_memory`
normalizes each pattern to **unit norm** before the Hebbian outer product, so
components are ~1/sqrt(D) and `W x` comes out ~1/D — which is precisely why the
dynamics sits in tanh's linear region (§5.4). Classical Hopfield stores +-1
patterns with `W = (1/N) sum p p^T`, so `W p ~ p` with O(1) components and a
gain of order 1 saturates. This module therefore stores the tanh'd patterns at
their **natural scale**, unnormalized, which is the classical convention, and
reports the saturation level so the two regimes are visible.

Sweeps `g` x `M`, reporting whether the stored patterns are fixed points and
whether a corrupted cue returns to the right one. Classical capacity for
uncorrelated binary patterns is 0.138*D = 141 at D=1024, which the M axis
brackets.

    python -m analysis.nav_p2.stored_gain_capacity --ckpt <any nav ckpt>
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint,
)


def build_W(P: torch.Tensor) -> torch.Tensor:
    """Hebbian, (1/D) sum p p^T, zero diagonal -- as Hopfield.input_memory does."""
    D = P.shape[1]
    W = (P.T @ P) / D
    W.fill_diagonal_(0.0)
    return W


def iterate(W, X, beta, steps, normalize):
    for _ in range(steps):
        X = torch.tanh(beta * (X @ W.T))
        if normalize:
            X = F.normalize(X, dim=-1)
    return X


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--beta", type=float, default=1.0,
                   help="gain of the RECALL dynamics (classical convention is ~1 "
                        "once patterns are stored at +-1 scale)")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--normalize", action="store_true",
                   help="normalize each dynamics step, as the codebase does")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = 1
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device))
    D = enc_cfg.out_dim
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    envs, vh, offsets = build_eval_world(cfg, encoder, str(device), ckpt_path=None)

    # Encoded points from random scaffold positions -- the same population
    # distractors are drawn from.
    rng = np.random.RandomState(args.seed)
    MAXM = 400
    pos = np.stack([rng.randint(0, vh.Npos, MAXM),
                    rng.randint(0, vh.Npos, MAXM)], 1)
    raw = torch.from_numpy(
        np.stack([vh.encoded_Phi[a, b] for a, b in pos])).float().to(device)
    raw = F.normalize(raw, dim=-1)                        # unit-norm encodings
    comp = raw.abs().median()
    print(f"D={D}  {MAXM} encoded points  median |component| = {comp:.5f}  "
          f"(1/sqrt(D) = {1/np.sqrt(D):.5f})")
    print(f"recall dynamics: x <- tanh({args.beta} * W x)"
          f"{' , normalized each step' if args.normalize else ' , unnormalized'}")
    print(f"classical capacity 0.138*D = {0.138 * D:.0f} patterns\n")

    gains = [1, 3, 10, 30, 100, 300, 1000]
    Ms = [5, 11, 25, 50, 100, 141, 200, 400]

    print("SATURATION of the stored patterns, mean |tanh(g*xi)| "
          "(1.0 = fully binarized)")
    for g in gains:
        t = torch.tanh(g * raw)
        print(f"   g={g:>5}  mean|p| {t.abs().mean():.4f}   "
              f"||p||^2 {t.pow(2).sum(-1).mean():>7.1f}   "
              f"cos(p, sign) {(F.normalize(t, dim=-1) * F.normalize(torch.sign(raw), dim=-1)).sum(-1).median():.4f}")

    # The dynamics gain must clear a threshold that DEPENDS on the storage gain.
    # With p = tanh(g*xi) of squared norm S, W p ~ (S/D) p, so the loop gain is
    # beta*S/D and a nonzero fixed point needs beta > D/S -- 1024 at g=1, ~1.3
    # at g=100. A single beta cannot test this grid: the first run used
    # beta=1.0, sat below threshold in every cell, and every state decayed to
    # zero. beta is therefore set per cell as a multiple of its own threshold.
    MULTS = (1.2, 2.0, 5.0, 20.0, 100.0)

    def probe(g, M, corrupt_first):
        P = torch.tanh(g * raw[:M])
        S = P.pow(2).sum(-1).mean().item()
        W = build_W(P)
        Pn = F.normalize(P, dim=-1)
        if corrupt_first:
            n = F.normalize(torch.randn(M, D, device=device), dim=-1)
            n = F.normalize(n - (n * Pn).sum(-1, keepdim=True) * Pn, dim=-1)
            X0 = (0.70 * Pn + np.sqrt(0.51) * n) * P.norm(dim=-1, keepdim=True)
        else:
            X0 = P.clone()
        best, best_m = 0.0, 0.0
        for m in MULTS:
            beta = m * D / max(S, 1e-9)
            out = iterate(W, X0.clone(), beta, args.steps, args.normalize)
            c = (F.normalize(out, dim=-1) * Pn).sum(-1).median().item()
            if c > best:
                best, best_m = c, m
        return best, best_m

    hdr = f"{'gain / M':>10}" + "".join(f"{m:>8}" for m in Ms)
    for label, cf in (("A. stored patterns are fixed points?  best median "
                       "cos(iterate(p), p) over the beta sweep", False),
                      ("B. cue corrupted to cos 0.70 returns?  best median "
                       "cos(iterate(cue), p)", True)):
        print(f"\n{label}")
        print(hdr)
        for g in gains:
            row = f"{g:>10}"
            for M in Ms:
                c, _ = probe(g, M, cf)
                row += f"{c:>8.3f}"
            print(row)

    print("\nWhich beta multiple of the D/S threshold won, table A")
    print(hdr)
    for g in gains:
        row = f"{g:>10}"
        for M in Ms:
            _, bm = probe(g, M, False)
            row += f"{bm:>8.1f}"
        print(row)

    print("\nReading: >= 0.99 means the stored patterns ARE fixed points (A) "
          "/ are recovered (B).")
    print("The g at which a column turns on is the storage gain needed; the M "
          "at which\na row turns off is the capacity at that gain.")


if __name__ == "__main__":
    main()
