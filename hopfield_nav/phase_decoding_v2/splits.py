"""Arena split generators for cross-arena evaluation.

Four split families:
  - LOO          : train on N-1 arenas, test on 1 (every arena)
  - Random 80/20 : random arena permutation, first 20% as test (n_random folds)
  - Quadrant 1v3 : train on arenas in 1 goal-quadrant, test on the other 3
  - Quadrant 3v1 : train on arenas in 3 goal-quadrants, test on the held-out 1
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    train: frozenset[int]
    test: frozenset[int]
    name: str


@dataclass
class Split:
    name: str
    folds: list[Fold]


def split_loo(arena_ids: list[int]) -> Split:
    arenas = sorted(arena_ids)
    folds: list[Fold] = []
    for held in arenas:
        train = frozenset(a for a in arenas if a != held)
        test = frozenset({held})
        if not train:
            continue
        folds.append(Fold(train=train, test=test, name=f"held_arena={held}"))
    return Split(name="LOO", folds=folds)


def split_random(
    arena_ids: list[int], *,
    n_splits: int = 20, test_frac: float = 0.2, seed: int = 0,
) -> Split:
    arenas = np.asarray(sorted(arena_ids), dtype=np.int64)
    rng = np.random.RandomState(seed)
    n_test = max(1, int(round(test_frac * arenas.size)))
    folds: list[Fold] = []
    for k in range(n_splits):
        order = rng.permutation(arenas)
        test = frozenset(int(a) for a in order[:n_test])
        train = frozenset(int(a) for a in order[n_test:])
        if not train or not test:
            continue
        folds.append(Fold(train=train, test=test, name=f"split_{k}"))
    return Split(name="Random 80/20", folds=folds)


def _arenas_by_quadrant(quadrants: dict[int, int]) -> dict[int, list[int]]:
    by_q: dict[int, list[int]] = {q: [] for q in range(4)}
    for a, q in quadrants.items():
        by_q.setdefault(int(q), []).append(int(a))
    return {q: sorted(v) for q, v in by_q.items()}


def split_quadrant_one_vs_rest(quadrants: dict[int, int]) -> Split:
    by_q = _arenas_by_quadrant(quadrants)
    folds: list[Fold] = []
    all_arenas = set().union(*by_q.values())
    for q in sorted(by_q.keys()):
        train_set = frozenset(by_q[q])
        test_set = frozenset(all_arenas - train_set)
        if not train_set or not test_set:
            continue
        folds.append(Fold(train=train_set, test=test_set,
                          name=f"train_quadrant={q}"))
    return Split(name="Quadrant 1v3", folds=folds)


def split_quadrant_three_vs_one(quadrants: dict[int, int]) -> Split:
    by_q = _arenas_by_quadrant(quadrants)
    folds: list[Fold] = []
    all_arenas = set().union(*by_q.values())
    for q in sorted(by_q.keys()):
        test_set = frozenset(by_q[q])
        train_set = frozenset(all_arenas - test_set)
        if not train_set or not test_set:
            continue
        folds.append(Fold(train=train_set, test=test_set,
                          name=f"held_quadrant={q}"))
    return Split(name="Quadrant 3v1", folds=folds)


def all_splits(
    arena_ids: list[int],
    quadrants: dict[int, int],
    *,
    n_random: int = 20,
    test_frac: float = 0.2,
    seed: int = 0,
) -> list[Split]:
    return [
        split_loo(arena_ids),
        split_random(arena_ids, n_splits=n_random, test_frac=test_frac, seed=seed),
        split_quadrant_one_vs_rest(quadrants),
        split_quadrant_three_vs_one(quadrants),
    ]
