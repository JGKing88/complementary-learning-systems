"""Sec 21 follow-up: how even must the number of L1 units per pattern be?

A lock-in L1 with random intrinsic turns holds each pattern j with a random
number of units. Random wiring then picks pattern j for a triple with
probability count(j) / sum, so a frequency's weight in the kernel is scaled by
the product of its three patterns' relative counts. That distortion factorises
by module and lands on the grid code's self-similar displacements: rule 2c's
exact, fully covering L1 aliased at 0.30-0.34 with every unit wired in, and
0.09 with one unit per pattern (``gwn_l1_lockin2``).

Here exact L1 with simulated counts: u(j) ~ Poisson(m / 2) units on pattern j,
each also emitting its conjugate, so count(j) = u(j) + u(-j) with mean m
(6 l^2 units per module, as in Sec 21, is m = 12). Triples are drawn with
probability proportional to the counts (the same as wiring uniformly over
duplicated pairs); gains and metrics as ``gwn_l1_twin``. Bump 0.10, no noise.

    python -m analysis.hopfield_probe.gwn_l1_mult --out DIR
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .gwn_check import DEV, LAMBDAS, NU, SMOKE, World, _dump, _line, build
from .gwn_encoder import exact_l1
from .gwn_l1_online import conj_index, local_slowness

FWHM = 0.10
MEANS = (2, 12) if SMOKE else (2, 6, 12, 50, 200, 1000)
SEEDS = (0,) if SMOKE else (0, 1)


def counts(m, l, rng):
    u = rng.poisson(m / 2, l * l)
    j = np.arange(l * l)
    return u + u[conj_index(j, l)]


def weighted_triples(l1, cnt, n, seed):
    g = torch.Generator().manual_seed(seed)
    cols = []
    for mi in range(len(LAMBDAS)):
        idx = l1.pairs_of(mi).cpu()
        p = torch.tensor(cnt[mi], dtype=torch.float64)
        cols.append(idx[torch.multinomial(p, n, replacement=True, generator=g)])
    return torch.stack(cols, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    W = World(fwhm=FWHM)
    ex = exact_l1(FWHM).to(DEV)
    res = {}
    for m in (None,) + MEANS:
        for s in SEEDS:
            rng = np.random.default_rng(100 + s)
            cnt = ([np.ones(l * l) for l in LAMBDAS] if m is None else
                   [counts(m, l, rng) for l in LAMBDAS])
            tr = weighted_triples(ex, cnt, NU, s)
            v, pw = local_slowness(ex, tr, W.steps, FWHM)
            enc, info = build(ex, tr, v, pw, FWHM)
            met = W.metrics(enc)
            tag = "one per pattern" if m is None else f"Poisson mean {m}"
            cv = float(np.mean([c.std() / c.mean() for c in cnt]))
            miss = float(np.mean([(c == 0).mean() for c in cnt]))
            _line(f"{tag} seed {s} (count CV {cv:.2f}, missing {miss:.3f})", info, met)
            res[f"{'uniform' if m is None else m}_seed{s}"] = {"count_cv": cv, "missing": miss, **info, **met}
            _dump(os.path.join(a.out, "mult.json"), res)
    print("DONE mult", flush=True)


if __name__ == "__main__":
    main()
