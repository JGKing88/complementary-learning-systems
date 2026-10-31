"""Idea 2's memory parts (``hopfield_nav/memory/plastic.py``).

1. The canvas puts each ray at its absolute-direction slot, with a mask.
2. Decoding true phases recovers the exact cell, anywhere on the scaffold.
3. A write lowers its own loss and is differentiable back to the initial
   weights and the learning rates (what the meta-learner needs).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from hopfield_nav.memory.plastic import (
    canvas, decode_cells, forward, goal_phases, init_params, write, write_loss)
from hopfield_nav.world.env import PANORAMA_BINS, cone_offsets, panorama_bins

LAMBDAS = (11, 12, 13)


def test_canvas_places_rays_by_absolute_direction():
    rng = np.random.RandomState(0)
    views = rng.randn(5, 60).astype(np.float32)
    psi = rng.uniform(-np.pi, np.pi, 5)
    c = canvas(views, psi)
    assert c.shape == (5, 2 * PANORAMA_BINS)
    bins = panorama_bins(psi[:, None] + cone_offsets(60)[None, :])
    for b in range(5):
        assert c[b, PANORAMA_BINS:].sum() == 60          # 60 distinct slots
        assert np.allclose(c[b, bins[b]], views[b])


def test_decode_recovers_the_exact_cell():
    rng = np.random.RandomState(1)
    for _ in range(20):
        off = rng.randint(0, 1716 - 20, size=2)
        gx, gy = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
        cells = np.stack([gx.ravel(), gy.ravel()], 1) + off
        i = rng.randint(len(cells), size=8)
        pred = goal_phases(cells[i, 0], cells[i, 1], LAMBDAS)
        assert np.array_equal(decode_cells(pred, cells, LAMBDAS), i)


def test_write_lowers_its_loss_and_is_differentiable():
    torch.manual_seed(0)
    theta = [p.requires_grad_(True) for p in init_params(2 * PANORAMA_BINS, 32, 2)]
    log_alpha = [torch.full_like(p, math.log(0.05)).requires_grad_(True) for p in theta]
    x = torch.randn(16, 2 * PANORAMA_BINS)
    target = torch.from_numpy(goal_phases([100], [200], LAMBDAS))
    before = float(write_loss(theta, x, target))
    new = write(theta, log_alpha, x, target, k=3, create_graph=True)
    after = write_loss(new, x, target)
    assert float(after) < before
    after.backward()
    for t in theta + log_alpha:
        assert t.grad is not None and torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0


def test_head_only_write_keeps_the_representation():
    torch.manual_seed(0)
    theta = [p.requires_grad_(True) for p in init_params(2 * PANORAMA_BINS, 32, 2)]
    log_alpha = [torch.full_like(p, math.log(0.5)).requires_grad_(True) for p in theta]
    x = torch.randn(16, 2 * PANORAMA_BINS)
    target = torch.from_numpy(goal_phases([100], [200], LAMBDAS))
    before = float(write_loss(theta, x, target, normalize=True))
    new = write(theta, log_alpha, x, target, k=3, n_plastic=2, normalize=True)
    assert all(n is t for n, t in zip(new[:-2], theta[:-2]))
    assert not torch.equal(new[-2], theta[-2])
    after = write_loss(new, x, target, normalize=True)
    assert float(after) < before
    after.backward()
    # The fixed layers still get a meta-gradient through the head's write.
    for t in theta + log_alpha[-2:]:
        assert t.grad is not None and t.grad.abs().sum() > 0


def test_forward_shapes_and_unit_phases():
    theta = init_params(2 * PANORAMA_BINS, 16, 1)
    ph, c = forward(theta, torch.randn(4, 2 * PANORAMA_BINS))
    assert ph.shape == (4, 12) and c.shape == (4,)
    assert torch.allclose(ph.reshape(4, 6, 2).norm(dim=2), torch.ones(4, 6), atol=1e-4)
