"""Sec 21 follow-up: L1 learning error against the network's sampling noise.

A learned L1 lists its pairs in its own order, so ``random_triples(seed 0)``
picks a different set of frequencies for each learner, and the 400k network's
alias moves for that reason alone. Here each route-1 learner (1b, 1c; 100k
steps, the same walks and seeds as ``gwn_l1_online``) is built twice:

* learned: its own pairs (as ``gwn_l1_online``; the numbers should reproduce);
* twin: the same triples, with every learned pair replaced by the exact pair
  of its dominant pattern. Twin minus learned is the L1 error alone.

The exact network over triple seeds 0-7 gives the sampling spread.

    python -m analysis.hopfield_probe.gwn_l1_twin --out DIR
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .gwn_check import DEV, LAMBDAS, NU, SMOKE, World, _dump, _line, build
from .gwn_encoder import L1, exact_l1, random_triples
from .gwn_l1_learn import OFFS, _assemble, codes_at
from .gwn_l1_online import (CONDS, T_CHECK, cond_name, lms_shift, local_slowness, make_steps,
                            route1, unit_purity)

SIDES = (14, 20, 40, "scat")
SEEDS = range(2 if SMOKE else 8)


def network(l1, steps, fwhm, noise, W, tag, seed=0):
    tr = random_triples(l1, NU, seed=seed)
    v, pw = local_slowness(l1, tr, steps, fwhm, noise)
    enc, info = build(l1, tr, v, pw, fwhm)
    m = W.metrics(enc)
    _line(tag, info, m)
    return {**info, **m}


def twin_l1(per, fwhm):
    """Exact pairs of the learned pairs' dominant patterns, in the learned order."""
    ex = exact_l1(fwhm)
    cols = []
    for m, l in enumerate(LAMBDAS):
        w = np.stack([c + 1j * s for c, s in per[m]], 1)
        _, j = unit_purity(w, l)
        cols.append(OFFS[m] + j)
    cols = torch.tensor(np.concatenate(cols))
    return L1(ex.Wc[:, cols], ex.Ws[:, cols], ex.module[cols], ex.pattern[cols], LAMBDAS).to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    worlds = {f: World(fwhm=f) for f in (0.10, 0.25)}
    res = {}
    for f, W in worlds.items():
        ex = exact_l1(f).to(DEV)
        res[f"exact_fwhm{f:g}"] = [network(ex, W.steps, f, 0.0, W, f"exact bump {f:g} triple seed {s}", s)
                                   for s in SEEDS]
    Tmax = max(T_CHECK)
    for task, (side, fwhm, noise) in enumerate(CONDS):
        if side not in SIDES:
            continue
        W = worlds[fwhm]
        name = cond_name(side, fwhm, noise)
        IN, OUT, ax, (p0, p1) = make_steps(side, fwhm, noise, Tmax, W, 1000 + task)
        r1 = route1(IN, OUT, ax, Tmax, lms_shift(IN, OUT, ax, {Tmax})[Tmax])
        del IN, OUT
        if side == "scat":
            vis = np.argwhere(W.mask)
            cells = vis[np.random.default_rng(3).permutation(len(vis))[:5_000 if SMOKE else 50_000]]
            gsteps = W.steps
        else:
            cells = np.unique(np.concatenate([p0, p1[-1:]]), axis=0)
            idx = np.linspace(0, Tmax - 1, 512 if SMOKE else 8192).astype(int)
            gsteps = (p0[idx], p1[idx])
        codes = codes_at(cells, fwhm)
        for k in ("1b", "1c"):
            per = [p for p, _ in r1[k]]
            l1, _ = _assemble(per, codes, fwhm)
            res[f"{name}_{k}"] = {
                "learned": network(l1, gsteps, fwhm, noise, W, f"{name} {k} learned"),
                "twin": network(twin_l1(per, fwhm), gsteps, fwhm, noise, W, f"{name} {k} twin")}
        _dump(os.path.join(a.out, "twin.json"), res)
    _dump(os.path.join(a.out, "twin.json"), res)
    print("DONE twin", flush=True)


if __name__ == "__main__":
    main()
