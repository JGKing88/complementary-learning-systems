"""An empty memory reads q = 0, also when other rows of the batch have one.

Until 2026-10-01, ``hopfield_signal_at`` / ``multistep_q`` on a list of
per-trajectory Hopfields projected *every* row with recall = 0 for the empty
ones, giving ``project(0 - x)`` -- a position-only vector -- and masked only the
normalized signal. Raw q (every Agent-HaSH run feeds it) therefore carried that
vector for an empty trajectory whenever another one in the batch held a memory:
the task regime, and evaluate_task at 0 distractors once any trajectory had
stored. When the whole batch was empty the early return gave 0, so the bug only
showed in mixed batches. GRID_MLP_NAV_PLAN §9.4.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield import Hopfield
from hopfield_nav.rollout import signal
from hopfield_nav.tests.fixtures import StubVectorHash, make_stub_cfg


def _batch(populated_rows, B=5):
    cfg = make_stub_cfg(movement_mode="continuous", input_hopfield_raw=True,
                        input_hopfield_multistep=(1, 2))
    vh = StubVectorHash(16, 8)
    pos = np.array([[1, 1], [2, 3], [4, 4], [3, 1], [5, 2]])[:B]
    emb = vh.get_encoded_state(pos, (0, 0))
    hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(B)]
    for b in populated_rows:
        hops[b].input_memory(torch.from_numpy(vh.encoded_Phi[6, 5 + b]).float())
    return cfg, vh, pos, emb, hops


@pytest.mark.parametrize("populated", [(0,), (0, 3), (1, 2, 4)])
def test_empty_rows_read_zero_q(populated):
    cfg, vh, pos, emb, hops = _batch(populated)
    sig, q, mask, W = signal.hopfield_signal_at(
        vh, cfg, emb, torch.from_numpy(emb), pos, (0, 0), hops, False,
        torch.device("cpu"), 8)
    empty = np.array([b not in populated for b in range(len(hops))])
    assert not q[empty].any(), "empty memory must read q = 0"
    assert not sig.numpy()[empty].any()
    assert (np.linalg.norm(q[~empty], axis=1) > 0).all()
    assert mask.numpy().tolist() == (~empty).tolist()
    ms = signal.multistep_q(vh, cfg, emb, torch.from_numpy(emb), hops, False,
                            W, [1, 2], 8, torch.device("cpu"))
    for s, q_s in ms.items():
        assert not q_s[empty].any(), f"multistep {s}: empty memory must read 0"
        assert (np.linalg.norm(q_s[~empty], axis=1) > 0).all()


def test_populated_rows_unchanged_by_their_neighbours():
    """A row's q depends only on its own memory, not on who else is empty."""
    cfg, vh, pos, emb, hops = _batch((0, 3))
    _, q_mixed, _, _ = signal.hopfield_signal_at(
        vh, cfg, emb, torch.from_numpy(emb), pos, (0, 0), hops, False,
        torch.device("cpu"), 8)
    cfg, vh, pos, emb, hops_all = _batch((0, 1, 2, 3, 4))
    _, q_full, _, _ = signal.hopfield_signal_at(
        vh, cfg, emb, torch.from_numpy(emb), pos, (0, 0), hops_all, False,
        torch.device("cpu"), 8)
    assert np.allclose(q_mixed[[0, 3]], q_full[[0, 3]])
