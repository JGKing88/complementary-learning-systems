"""Ceiling for idea 2's memory network (plan §6.4).

    python -m analysis.idea2.ceiling [--walks 1] [--hidden 256]

Fits N envs' goals offline, all envs interleaved, with Adam for as many steps
as it likes, then decodes the goal from fresh views at uniform random cells
and headings. Nothing is written in sequence, so nothing is forgotten. This is
the best any learning rule can do with the same network and the same views
per env; if it falls short, meta-learning cannot close the gap.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from hopfield_nav.memory.plastic import decode_cells, forward, init_params, phase_loss
from analysis.idea2.lifetimes import LAMBDAS, EnvPool, Placed


def T_(x):
    return torch.from_numpy(np.asarray(x, np.float32))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--walks", type=int, default=1, help="40-step walks per env")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--N", type=int, nargs="+", default=[6, 30, 100])
    a = p.parse_args()
    pool = EnvPool(300, seed=2)                    # meta_train's eval pool at seed 0
    for N in a.N:
        rng = np.random.RandomState(10_000 + N)
        envs = [Placed(pool, i, rng) for i in rng.choice(len(pool), size=N, replace=False)]
        X = torch.cat([T_(e.visit(rng, 40)) for e in envs for _ in range(a.walks)])
        Y = torch.cat([T_(e.target).expand(40, -1) for e in envs for _ in range(a.walks)])
        torch.manual_seed(0)
        params = [t.requires_grad_(True) for t in init_params(X.shape[1], a.hidden, a.layers)]
        opt = torch.optim.Adam(params, lr=1e-3)
        for _ in range(a.steps):
            b = torch.randint(len(X), (512,))
            loss = phase_loss(forward(params, X[b])[0], Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = []
        with torch.no_grad():
            train = float(phase_loss(forward(params, X)[0], Y))
            for e in envs:
                x, _ = e.test_views(rng, 64)
                cell = decode_cells(forward(params, T_(x))[0].numpy(), e.cells_global, LAMBDAS)
                acc.append(float((np.linalg.norm(e.cells_local[cell] - e.goal, axis=1) <= 1.5).mean()))
        print(f"walks/env {a.walks} hidden {a.hidden} N={N:3d}: train phase_loss {train:.3f}, "
              f"test acc {np.mean(acc):.3f}", flush=True)


if __name__ == "__main__":
    main()
