"""Sensory-keyed goal memory with SINGLE live views, on distal-panorama envs.

The omni test (retrieval_test.py) keys and queries with the four-heading view,
which the agent never sees in one step. Here the query is one egocentric view
at a random continuous heading -- what the agent actually has -- under three
keys:

  A  single  -- key: one view at the goal, random heading (the arrival view)
  B  matched -- as A, but the query faces the key's heading (upper bound)
  C  omni    -- key: the four-heading view at the goal, re-indexed by absolute
                direction (180 two-degree slices, overlaps averaged); a query
                view is compared on the slices its 60 rays point into

Same two questions as before: argmax own-goal accuracy, and stored-vs-unstored
top-score AUC.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from hopfield_nav.config import EnvConfig
from hopfield_nav.world.env import (
    CARDINAL_RADIANS, PANORAMA_BINS, cone_offsets, make_env, panorama_bins,
    raycast_codes,
)
from .retrieval_test import D0_ENV, D0_WORLD, auc


def build(n_fresh, seed, amp):
    cfg = EnvConfig(**D0_ENV, distal_amp=amp)
    with open(D0_WORLD) as f:
        rec = json.load(f)["split"]["base_val"]
    specs = [(int(s["wall_seed"]), tuple(s["goal"])) for s in rec]
    rng = np.random.RandomState(seed)
    used = {w for w, _ in specs}
    while len(specs) < len(rec) + n_fresh:
        w = int(rng.randint(0, 10_000_000))
        if w not in used:
            used.add(w)
            specs.append((w, None))
    envs = []
    for w, g in specs:
        e = make_env(cfg, "continuous", seed=w)
        if g is not None:
            e.set_goal(g)
        envs.append(e)
    return envs


def views(env, xs, ys, psi):
    return raycast_codes(env._wall_code, env.size, xs, ys, psi,
                         env._observation_size, env.wall_resolution, env._panorama)


def omni_by_bin(env, pos):
    """Four-heading view at ``pos`` as a (PANORAMA_BINS,) function of direction."""
    n = env._observation_size
    ang = (CARDINAL_RADIANS[:, None] + cone_offsets(n)[None, :]).ravel()
    v = env.omni_obs_at(pos)
    b = panorama_bins(ang)
    out = np.zeros(PANORAMA_BINS)
    np.add.at(out, b, v)
    return out / np.bincount(b, minlength=PANORAMA_BINS)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def run(envs, idx, variant, rng):
    S = envs[0].size
    n = envs[0]._observation_size
    gx, gy = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    xs, ys = gx.ravel().astype(float), gy.ravel().astype(float)
    key_psi = rng.uniform(-np.pi, np.pi, len(idx))
    if variant == "omni":
        keys = np.stack([omni_by_bin(envs[e], envs[e]._goal) for e in idx])   # (N,180)
    else:
        keys = np.stack([views(envs[e], [envs[e]._goal[0]], [envs[e]._goal[1]],
                               [key_psi[k]])[0] for k, e in enumerate(idx)])  # (N,60)
    correct, top_s, top_u, own = [], [], [], []
    for k, e in enumerate(idx):
        psi = (np.full(len(xs), key_psi[k]) if variant == "matched"
               else rng.uniform(-np.pi, np.pi, len(xs)))
        q = views(envs[e], xs, ys, psi)                                       # (C,60)
        if variant == "omni":
            bins = panorama_bins(psi[:, None] + cone_offsets(n)[None, :])     # (C,60)
            kk = keys[:, bins]                                                # (N,C,60)
            sims = np.einsum("cr,ncr->cn", unit(q), unit(kk))
        else:
            sims = unit(q) @ unit(keys).T
            # For "matched" every other env's key has its own heading, so the
            # query is compared at the OWN key's heading only; foreign keys are
            # at unrelated headings, as they would be in the real memory.
        o = sims[:, k]
        oth = np.delete(sims, k, axis=1).max(1)
        correct.append(o > oth)
        own.append(o)
        top_s.append(np.maximum(o, oth))
        top_u.append(oth)
    c = lambda L: np.concatenate(L)
    return dict(acc=float(c(correct).mean()), auc=auc(c(top_s), c(top_u)),
                own=float(c(own).mean()), unstored=float(c(top_u).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amp", type=float, default=1.0)
    p.add_argument("--Ns", default="6,30,100")
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    envs = build(94, 0, a.amp)
    res = {}
    for variant in ("single", "matched", "omni"):
        for N in [int(x) for x in a.Ns.split(",")]:
            rng = np.random.RandomState(1)
            subsets = ([list(range(6))] if N == 6 else
                       [sorted(rng.choice(len(envs), N, replace=False).tolist())
                        for _ in range(1 if N == len(envs) else a.draws)])
            rs = [run(envs, idx, variant, np.random.RandomState(2 + i))
                  for i, idx in enumerate(subsets)]
            m = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            res[f"{variant}_N{N}"] = m
            print(f"amp={a.amp} {variant:>7} N={N:3d}  acc {m['acc']:.3f}  auc {m['auc']:.3f}"
                  f"  own {m['own']:.3f}  unstored_top {m['unstored']:.3f}", flush=True)
    with open(os.path.join(a.out, f"single_view_amp{a.amp}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
