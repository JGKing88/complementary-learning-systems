"""Stratified patch placement: it must place, not overlap, and close holes.

The point of the jittered lattice (§5.4 step 3) is one measurable quantity:
the largest distance from an arena point to the nearest training patch. §4.6
found that distance correlates -0.47 with a reference's radius, so it is the
mechanism by which placement could matter at 10% coverage. If stratifying does
not shrink it, the arm is not worth running -- hence the test.
"""
from __future__ import annotations

import numpy as np
import torch

from encoder_training.data import _grid_dims, sample_nonoverlapping_patches

ARENA = 1716
SEEDS = (42, 43, 44, 45)

MIXES = {
    "lo_big":    [200] * 7,
    "lo_mixtop": [200] * 5 + [150] * 3 + [100] * 3,
    "lo_mix2":   [200] * 4 + [100] * 12,
    "lo_many":   [100] * 29,
    "lo_tail":   [200] * 3 + [150] * 4 + [100] * 6 + [70] * 8,
}


def _no_overlap(y0s, x0s, sizes) -> bool:
    n = len(sizes)
    for i in range(n):
        for j in range(i + 1, n):
            if not (y0s[i] + sizes[i] <= y0s[j] or y0s[j] + sizes[j] <= y0s[i]
                    or x0s[i] + sizes[i] <= x0s[j]
                    or x0s[j] + sizes[j] <= x0s[i]):
                return False
    return True


def _coverage_radius(y0s, x0s, sizes, step: int = 12) -> float:
    """Max over a coarse arena lattice of the distance to the nearest patch."""
    g = np.arange(0, ARENA, step)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    best = np.full(yy.shape, np.inf)
    for y0, x0, s in zip(y0s, x0s, sizes):
        dy = np.maximum.reduce([y0 - yy, yy - (y0 + s - 1),
                                np.zeros_like(yy)])
        dx = np.maximum.reduce([x0 - xx, xx - (x0 + s - 1),
                                np.zeros_like(xx)])
        best = np.minimum(best, np.hypot(dy, dx))
    return float(best.max())


def test_grid_dims_prefers_square_cells_over_tight_packing():
    # 7 patches: 2x4 wastes one cell but is 2:1 elongated; 3x3 wastes two and
    # is square. Squareness wins, because it is what bounds the hole size.
    assert _grid_dims(7, ARENA, ARENA) == (3, 3)
    assert sorted(_grid_dims(11, ARENA, ARENA)) == [3, 4]
    assert sorted(_grid_dims(29, ARENA, ARENA)) == [5, 6]


def test_stratified_places_every_mix_at_every_seed():
    for name, sizes in MIXES.items():
        for seed in SEEDS:
            torch.manual_seed(seed)
            y0s, x0s, out = sample_nonoverlapping_patches(
                ARENA, ARENA, sizes, placement="stratified")
            assert out == sizes, name
            assert len(y0s) == len(sizes)
            assert all(0 <= y <= ARENA - s for y, s in zip(y0s, sizes)), name
            assert all(0 <= x <= ARENA - s for x, s in zip(x0s, sizes)), name
            assert _no_overlap(y0s, x0s, sizes), f"{name} seed {seed}"


def test_stratified_shrinks_the_worst_hole():
    """The claim the arm rests on, checked before any GPU time is spent."""
    for name, sizes in MIXES.items():
        rand, strat = [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            rand.append(_coverage_radius(*sample_nonoverlapping_patches(
                ARENA, ARENA, sizes, placement="random")))
            torch.manual_seed(seed)
            strat.append(_coverage_radius(*sample_nonoverlapping_patches(
                ARENA, ARENA, sizes, placement="stratified")))
        assert np.mean(strat) < np.mean(rand), (
            f"{name}: stratified {np.mean(strat):.0f} "
            f"vs random {np.mean(rand):.0f}")


def test_random_placement_is_unchanged():
    """Every number in §1-§4 came from this path; it must not have moved."""
    torch.manual_seed(42)
    y0s, x0s, sizes = sample_nonoverlapping_patches(
        ARENA, ARENA, MIXES["lo_mixtop"])
    assert _no_overlap(y0s, x0s, sizes)
    torch.manual_seed(42)
    again = sample_nonoverlapping_patches(ARENA, ARENA, MIXES["lo_mixtop"],
                                          placement="random")
    assert (y0s, x0s, sizes) == again


def test_unknown_placement_rejected():
    try:
        sample_nonoverlapping_patches(ARENA, ARENA, [100] * 4,
                                      placement="lattice")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown placement")
