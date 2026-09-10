"""Pre-diagnostic for the firing-rate-adaptation novelty signal (P2 doc §21).

The proposal: replace the oracle novelty reward -- which reads a ground-truth
`visited_cells` array the agent cannot see -- with a signal computed from the
agent's own place code, by analogy with repetition suppression:

    a_t = lambda * a_{t-1} + (1 - lambda) * phi_t        (adaptation trace)
    n_t = 1 - cos(phi_t, a_{t-1})                        (un-adapted response)

`phi_t` is the encoder output at the current position, already computed every
step at `collector.py:263`.

THE THING THAT COULD KILL IT. At `encoder_gain=100` the code is near-binary and
this project's unique-radius work records that the coding radii are SMALL. If
cos(phi(x), phi(y)) collapses to ~0 for anything but an adjacent cell, then
`n_t` is 1.0 almost everywhere and the signal degenerates to "binary novelty
with a decay" -- still an improvement (no oracle, recency-weighted) but not the
graded, distance-sensitive signal that would make it worth the change.

So this measures two things, offline, with no training:

  1. THE KERNEL. cos(phi(x), phi(y)) as a function of |x - y|. If it falls off
     a cliff at one cell, the signal cannot be graded, whatever lambda is.
  2. THE SIGNAL. Run the trace along real trajectories and ask whether n_t
     carries anything the binary first-visit flag does not -- regressing n_t on
     the binary flag and reporting the residual.

  python -m analysis.nav_tri.adaptation_probe --ckpt X.pt --envs 3 --trials 8
"""

import argparse
import json

import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    cfg_from_checkpoint, eval_env_set, load_agent,
)
from hopfield_nav.evaluation.metrics import random_start
from hopfield_nav.world import generate as gen

from analysis.nav_tri.behavior_probe import rollout


def _cos_rows(A, B):
    """Row-wise cosine between (N, D) and (N, D)."""
    an = np.linalg.norm(A, axis=-1)
    bn = np.linalg.norm(B, axis=-1)
    return np.sum(A * B, axis=-1) / np.maximum(an * bn, 1e-12)


def kernel(vh, offset, size):
    """cos(phi(x), phi(y)) against |x - y|, over every cell pair in one env."""
    cells = np.array([[x, y] for x in range(size) for y in range(size)],
                     dtype=np.int32)
    phi = np.asarray(vh.get_encoded_state(cells, offset), dtype=np.float64)
    n = phi / np.maximum(np.linalg.norm(phi, axis=1, keepdims=True), 1e-12)
    C = n @ n.T
    D = np.linalg.norm(cells[:, None, :] - cells[None, :, :], axis=-1)
    iu = np.triu_indices(len(cells), k=1)
    return D[iu], C[iu]


def adaptation_signal(phi_seq, lam):
    """n_t = 1 - cos(phi_t, a_{t-1}), with a the leaky trace."""
    T = len(phi_seq)
    a = phi_seq[0].astype(np.float64).copy()
    out = np.zeros(T)
    for t in range(1, T):
        out[t] = 1.0 - _cos_rows(phi_seq[t][None], a[None])[0]
        a = lam * a + (1.0 - lam) * phi_seq[t]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--envs", type=int, default=3)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--split", default="place=held_out")
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.9, 0.95, 0.99])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    device = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = a.envs
    encoder, enc_cfg, gain = load_encoder(
        cfg.encoder_checkpoint, str(device), getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    embed_dim = enc_cfg.out_dim
    torch.manual_seed(0)
    np.random.seed(0)
    es = eval_env_set(cfg, encoder, str(device), ckpt_path=a.ckpt,
                      levels=gen.parse_levels(a.split), val_seed=0,
                      n_envs=cfg.num_val_envs)
    envs, vh, offsets = es["envs"], es["field"], es["offsets"]
    agent = load_agent(cfg, ck["agent_state_dict"], embed_dim, device)
    size = cfg.env.size
    out = {"gain": float(gain), "size": size}

    # ---------------------------------------------------------- 1. the kernel
    print("=" * 70)
    print("1. CODE SIMILARITY KERNEL  cos(phi(x), phi(y)) vs |x-y|")
    print("   (if this falls off a cliff at 1 cell, the signal cannot be graded)")
    ds, cs = [], []
    for i, env in enumerate(envs):
        d, c = kernel(vh, offsets[i], size)
        ds.append(d); cs.append(c)
    d = np.concatenate(ds); c = np.concatenate(cs)
    bins = [0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0, 14.0, 30.0]
    print("\n   %8s %8s %8s %8s %9s" % ("dist", "mean", "sd", "p95", "n"))
    rows = []
    for lo, hi in zip([0.0] + bins[:-1], bins):
        m = (d > lo) & (d <= hi)
        if not m.any():
            continue
        print("   %3.0f-%-4.0f %8.4f %8.4f %8.4f %9d"
              % (lo, hi, c[m].mean(), c[m].std(), np.percentile(c[m], 95),
                 m.sum()))
        rows.append([lo, hi, float(c[m].mean()), float(c[m].std())])
    out["kernel"] = rows
    near = c[(d > 0.0) & (d <= 1.5)].mean()
    far = c[d > 7.0].mean()
    print("\n   adjacent (<=1.5) %.4f   far (>7) %.4f   contrast %.4f"
          % (near, far, near - far))
    print("   -> %s" % ("GRADED: similarity decays over several cells"
                        if near - far > 0.15 and c[(d > 2.5) & (d <= 3.5)].mean() - far > 0.05
                        else "CLIFF: effectively a lookup table, signal will be ~binary"))

    # ------------------------------------------------------- 2. on real paths
    print("\n" + "=" * 70)
    print("2. THE SIGNAL ON REAL TRAJECTORIES")
    rng = np.random.RandomState(a.seed)
    all_n = {lam: [] for lam in a.lambdas}
    all_new = []
    for i, env in enumerate(envs):
        hops, starts = [], []
        for _ in range(a.trials):
            hops.append(Hopfield(embed_dim, beta=cfg.hopfield.beta,
                                 device=str(device)))
            starts.append(random_start(env.size, env.goal_location, rng))
        rec = rollout(agent=agent, env=env, env_offset=offsets[i],
                      vectorhash=vh, hopfields=hops, cfg=cfg, device=device,
                      starts=starts, max_steps=a.max_steps,
                      ends_on_arrival=False, goal_in_memory=False)
        cells = rec["cell"]                                   # (T, B, 2)
        for b in range(a.trials):
            cb = cells[:, b].astype(np.int32)
            phi = np.asarray(vh.get_encoded_state(cb, offsets[i]),
                             dtype=np.float64)
            seen, new = set(), np.zeros(len(cb))
            for t, cc in enumerate(cb):
                k = (int(cc[0]), int(cc[1]))
                new[t] = 0.0 if k in seen else 1.0
                seen.add(k)
            all_new.append(new[1:])
            for lam in a.lambdas:
                all_n[lam].append(adaptation_signal(phi, lam)[1:])

    new = np.concatenate(all_new)
    print("\n   binary first-visit rate: %.3f of steps" % new.mean())
    print("\n   %6s %8s %8s %8s %8s %10s %9s"
          % ("lambda", "mean", "sd", "p05", "p95", "corr(new)", "R2(new)"))
    res = []
    for lam in a.lambdas:
        n = np.concatenate(all_n[lam])
        r = float(np.corrcoef(n, new)[0, 1])
        # variance of the adaptation signal NOT explained by the binary flag
        b1 = n[new > 0.5].mean() if (new > 0.5).any() else 0.0
        b0 = n[new < 0.5].mean() if (new < 0.5).any() else 0.0
        pred = np.where(new > 0.5, b1, b0)
        r2 = 1.0 - np.var(n - pred) / max(np.var(n), 1e-12)
        print("   %6.2f %8.4f %8.4f %8.4f %8.4f %10.3f %9.3f"
              % (lam, n.mean(), n.std(), np.percentile(n, 5),
                 np.percentile(n, 95), r, r2))
        res.append({"lambda": lam, "mean": float(n.mean()),
                    "sd": float(n.std()), "corr_new": r, "r2_new": float(r2)})
        print("        new-cell mean %.4f   revisit mean %.4f   gap %.4f"
              % (b1, b0, b1 - b0))
    out["signal"] = res
    print("\n   R2(new) near 1.0 means the adaptation signal is the binary flag")
    print("   in disguise. Well below 1.0 means it carries extra structure.")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
