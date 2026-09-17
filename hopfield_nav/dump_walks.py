"""Dump the decode's random walks as an encoder-training dataset (plan sec 6.4).

    python -m hopfield_nav.dump_walks --size 20 --steps 131072000 --out walks_sz20_131M.npz

The same world, walkers and step budget as `train_decode_walk` (64 scattered
arenas, 8 walkers per env, discrete unit steps, goals inert). Each walker's
visited cells become rows: the cell's grid code, its coordinates (global,
so distances between two rows of one walker are that walker's odometry),
the walker id and the env id. `encoder_training.train --walk_data` trains
the encoder package's own trainer on these rows unchanged, grouping by
walker (odometry labels: only a walker's own moments are related) or by env
(the encoder's true-coordinate labels).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .train_decode_walk import Walkers
from .train_encoder_walk import build_world


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--n_val_envs", type=int, default=16)
    p.add_argument("--walkers", type=int, default=8)
    p.add_argument("--steps", type=int, default=131_072_000, help="total env-steps to walk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--place_margin", type=int, default=20)
    p.add_argument("--out", type=str, required=True)
    a = p.parse_args()
    a.observation_size, a.lambdas, a.fwhm_ratio = 120, [11, 12, 13], 0.25
    a.place_region, a.wall_seeds, a.n_same_envs, a.lr, a.n_updates, a.eval_every, a.device = \
        "anywhere", "0,10000000", 4, 3e-4, 0, 0, "cpu"
    rng = np.random.RandomState(a.seed)
    t0 = time.time()
    cfg, train, heldout, same, split, vh = build_world(a, rng)
    print(f"world: {len(train)} train envs, size {a.size}; {time.time()-t0:.0f}s", flush=True)
    walkers = Walkers(train.envs, a.walkers, a.seed + 1000)
    T = 64
    per_update = len(train) * a.walkers * T
    n_updates = int(np.ceil(a.steps / per_update))
    visited = np.zeros((len(train), a.walkers, a.size * a.size), dtype=bool)
    steps = 0
    for u in range(n_updates):
        seg = walkers.segment(T)                                   # (E, W, T+1, 2)
        ids = seg[..., 0] * a.size + seg[..., 1]                   # (E, W, T+1)
        for e in range(len(train)):
            for w in range(a.walkers):
                visited[e, w, ids[e, w]] = True
        steps += per_update
        if (u + 1) % 200 == 0:
            print(f"  {steps:,} steps; visited cells per walker {visited.sum(-1).mean():.0f} / {a.size**2}", flush=True)
    rows_e, rows_w, rows_c = np.where(visited)
    phi = np.concatenate([train.tensors[e].gbook[rows_c[rows_e == e]] for e in range(len(train))])
    order = np.argsort(rows_e, kind="stable")                      # phi above is grouped by env; align the rest
    rows_e, rows_w, rows_c = rows_e[order], rows_w[order], rows_c[order]
    offs = np.array(train.offsets, dtype=np.int64)
    coords = offs[rows_e] + np.stack([rows_c // a.size, rows_c % a.size], 1)
    walker_id = rows_e * a.walkers + rows_w
    np.savez_compressed(a.out, phi=phi.astype(np.float32), coords=coords.astype(np.float32),
                        walker=walker_id.astype(np.int64), env=rows_e.astype(np.int64),
                        offsets=offs, size=a.size, steps=steps, seed=a.seed, n_envs=len(train), walkers=a.walkers)
    print(f"wrote {a.out}: {len(phi):,} rows ({visited.sum(-1).mean():.0f} cells per walker), "
          f"{steps:,} env-steps, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
