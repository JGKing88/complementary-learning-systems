"""Hopfield signal: one implementation, and B=1 agrees with B>1.

``eval.agent_step`` used to recompute recall, projection and direction-shaping
inline for a single env, alongside the collector's batched version. They now
share ``signal.py``; these tests pin the properties that made the duplication
dangerous, so a future divergence is caught as a divergence rather than as a
drop in eval numbers.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav.rollout import signal as signal_ops
from hopfield_nav.world.memory import Hopfield
from hopfield_nav.tests.fixtures import StubVectorHash, make_stub_cfg

EMBED_DIM = 8


def _hopfield(vh, cells, beta=1.0):
    h = Hopfield(EMBED_DIM, beta=beta, device="cpu")
    for gx, gy in cells:
        h.input_memory(torch.from_numpy(vh.encoded_Phi[gx, gy]).float())
    return h


@pytest.mark.parametrize("movement_mode", ["discrete", "continuous"])
def test_single_env_matches_the_same_row_of_a_batch(movement_mode):
    """The B=1 eval path is the batched path, not a parallel implementation.

    Agreement is to float32 precision, not bit-exact: batched matmuls
    accumulate in a different order, which moves the last ULP (observed
    0.17286253 vs 0.17286251). That is why the evaluator goldens pin per-trial
    outcome records rather than aggregate floats -- the records are stable
    under rebatching and the aggregates are not.
    """
    cfg = make_stub_cfg(movement_mode=movement_mode)
    vh = StubVectorHash(Npos=16, embed_dim=EMBED_DIM)
    device = torch.device("cpu")
    positions = np.array([[1, 1], [3, 2], [5, 4]], dtype=np.int32)
    emb_np = vh.get_encoded_state(positions, (0, 0))
    emb = torch.from_numpy(emb_np).float()
    hop = _hopfield(vh, [(7, 7), (2, 9)])

    sig_b, q_b, mask_b, _ = signal_ops.hopfield_signal_at(
        vh, cfg, emb_np, emb, positions, (0, 0), hop, True, device, EMBED_DIM)

    for row in range(positions.shape[0]):
        p1 = positions[row:row + 1]
        e1_np = vh.get_encoded_state(p1, (0, 0))
        e1 = torch.from_numpy(e1_np).float()
        sig_1, q_1, mask_1, _ = signal_ops.hopfield_signal_at(
            vh, cfg, e1_np, e1, p1, (0, 0), hop, True, device, EMBED_DIM)
        assert torch.allclose(sig_1[0], sig_b[row], atol=1e-6), f"signal row {row}"
        assert np.allclose(q_1[0], q_b[row], atol=1e-6), f"q row {row}"
        assert bool(mask_1[0]) == bool(mask_b[row]), f"mask row {row}"


def test_empty_hopfield_gives_zero_signal_and_no_basis():
    """Nothing recalled means no direction -- and no Gram-Schmidt basis."""
    cfg = make_stub_cfg()
    vh = StubVectorHash(Npos=16, embed_dim=EMBED_DIM)
    positions = np.array([[1, 1], [3, 2]], dtype=np.int32)
    emb_np = vh.get_encoded_state(positions, (0, 0))
    emb = torch.from_numpy(emb_np).float()
    hop = Hopfield(EMBED_DIM, beta=1.0, device="cpu")

    sig, q, mask, W = signal_ops.hopfield_signal_at(
        vh, cfg, emb_np, emb, positions, (0, 0), hop, True,
        torch.device("cpu"), EMBED_DIM)
    assert torch.all(sig == 0)
    assert np.all(q == 0)
    assert not mask.any()
    assert W is None


def test_per_env_rows_without_memory_get_zero_signal():
    """A batch may mix envs that have stored with envs that have not.

    The *signal* is masked to zero for memoryless rows. ``q`` is NOT -- see
    test_q_is_not_masked_for_memoryless_rows below, which pins that as the
    current (pre-existing) behavior.
    """
    cfg = make_stub_cfg()
    vh = StubVectorHash(Npos=16, embed_dim=EMBED_DIM)
    positions = np.array([[1, 1], [3, 2], [5, 4]], dtype=np.int32)
    emb_np = vh.get_encoded_state(positions, (0, 0))
    emb = torch.from_numpy(emb_np).float()
    hops = [_hopfield(vh, [(7, 7)]),
            Hopfield(EMBED_DIM, beta=1.0, device="cpu"),   # empty
            _hopfield(vh, [(2, 9)])]

    sig, q, mask, _ = signal_ops.hopfield_signal_at(
        vh, cfg, emb_np, emb, positions, (0, 0), hops, False,
        torch.device("cpu"), EMBED_DIM)
    assert mask.tolist() == [True, False, True]
    assert torch.all(sig[1] == 0)
    assert torch.any(sig[0] != 0) and torch.any(sig[2] != 0)


def test_q_is_not_masked_for_memoryless_rows():
    """CHARACTERIZATION -- pins a pre-existing quirk, not an endorsement.

    For a row whose Hopfield is empty, the recalled vector is left as zeros and
    then projected anyway, so q = W @ (0 - embedding): a nonzero direction
    derived from nothing. Consumers that gate on memory_mask (auto-nav, the BC
    teacher's trust_hop) never see it.

    One consumer does not gate: with ``input_hopfield_raw`` set, both callers
    feed raw q to the policy in place of the masked signal, so a memoryless env
    receives this spurious direction where it would otherwise receive zeros.
    train_phase_a_only defaults --input_hopfield_raw to True, and the empty
    Hopfield is the normal case during phase-A explore, so this was live.

    Left as-is deliberately: changing it changes training results, and this
    phase is behavior-preserving.
    """
    cfg = make_stub_cfg()
    vh = StubVectorHash(Npos=16, embed_dim=EMBED_DIM)
    positions = np.array([[1, 1], [3, 2], [5, 4]], dtype=np.int32)
    emb_np = vh.get_encoded_state(positions, (0, 0))
    emb = torch.from_numpy(emb_np).float()
    hops = [_hopfield(vh, [(7, 7)]),
            Hopfield(EMBED_DIM, beta=1.0, device="cpu"),   # empty
            _hopfield(vh, [(2, 9)])]

    _sig, q, mask, W = signal_ops.hopfield_signal_at(
        vh, cfg, emb_np, emb, positions, (0, 0), hops, False,
        torch.device("cpu"), EMBED_DIM)
    assert not mask[1]
    assert not np.allclose(q[1], 0.0), (
        "q for a memoryless row is expected to be nonzero here; if this now "
        "passes, the quirk was fixed -- update the docstring and check whether "
        "the input_hopfield_raw path changed")
    # It is exactly the projection of the negated embedding.
    expected = vh.project_displacement(
        emb_np[1:2], np.zeros((1, EMBED_DIM), dtype=np.float32), W[1:2])
    assert np.allclose(q[1], expected[0], atol=1e-6)


@pytest.mark.parametrize("movement_mode,width", [("discrete", 4), ("continuous", 2)])
def test_signal_shape_and_normalization(movement_mode, width):
    cfg = make_stub_cfg(movement_mode=movement_mode)
    q = np.array([[3.0, 4.0], [-1.0, 0.0]], dtype=np.float32)
    sig = signal_ops.q_to_signal(q, cfg.agent)
    assert sig.shape == (2, width)
    if movement_mode == "discrete":
        assert np.all(sig.sum(axis=1) == 1)         # one-hot
    else:
        assert np.allclose(np.linalg.norm(sig, axis=1), 1.0)   # unit vectors


def test_q_to_signal_ignores_input_hopfield_raw():
    """Raw-vs-normalized is a call-site choice, not part of shaping the signal.

    Both callers keep the normalized signal for direction classification and
    substitute raw q only in the policy input, so folding the flag in here
    would silently change what teacher-forcing consumes.
    """
    cfg = make_stub_cfg(movement_mode="continuous", input_hopfield_raw=True)
    q = np.array([[3.0, 4.0]], dtype=np.float32)
    sig = signal_ops.q_to_signal(q, cfg.agent)
    assert np.allclose(np.linalg.norm(sig, axis=1), 1.0)
    assert not np.allclose(sig, q)


def test_oracle_points_at_the_goal():
    """The oracle signal is the direction a perfect recall would give."""
    cfg = make_stub_cfg(movement_mode="continuous")
    vh = StubVectorHash(Npos=16, embed_dim=EMBED_DIM)
    positions = np.array([[2, 2]], dtype=np.int32)
    emb_np = vh.get_encoded_state(positions, (0, 0))

    sig, q = signal_ops.oracle_signal_at(
        vh, emb_np, positions, (0, 0), (2, 2), cfg.agent)
    # Standing on the goal: zero displacement, so no direction.
    assert np.allclose(q, 0.0, atol=1e-6)

    sig, q = signal_ops.oracle_signal_at(
        vh, emb_np, positions, (0, 0), (4, 2), cfg.agent)
    assert not np.allclose(q, 0.0)
    assert np.allclose(np.linalg.norm(sig, axis=1), 1.0)


def test_multistep_returns_zeros_without_a_basis():
    cfg = make_stub_cfg(movement_mode="continuous")
    vh = StubVectorHash(Npos=16, embed_dim=EMBED_DIM)
    positions = np.array([[1, 1], [3, 2]], dtype=np.int32)
    emb_np = vh.get_encoded_state(positions, (0, 0))
    emb = torch.from_numpy(emb_np).float()
    out = signal_ops.multistep_q(
        vh, cfg, emb_np, emb, _hopfield(vh, [(7, 7)]), True, None, [1, 2],
        EMBED_DIM, torch.device("cpu"))
    assert sorted(out) == [1, 2]
    assert all(np.all(v == 0) for v in out.values())


def test_multistep_matches_single_step_at_one_iteration():
    """Step 1 of the trajectory is the same projection the signal path makes.

    Only holds at cfg.hopfield.steps == 1, which make_stub_cfg sets; it is the
    check that the two recall entry points agree about what "one step" means.
    """
    cfg = make_stub_cfg(movement_mode="continuous")
    assert cfg.hopfield.steps == 1
    vh = StubVectorHash(Npos=16, embed_dim=EMBED_DIM)
    positions = np.array([[1, 1], [3, 2]], dtype=np.int32)
    emb_np = vh.get_encoded_state(positions, (0, 0))
    emb = torch.from_numpy(emb_np).float()
    hop = _hopfield(vh, [(7, 7), (2, 9)])
    device = torch.device("cpu")

    _sig, q, _mask, W = signal_ops.hopfield_signal_at(
        vh, cfg, emb_np, emb, positions, (0, 0), hop, True, device, EMBED_DIM)
    out = signal_ops.multistep_q(
        vh, cfg, emb_np, emb, hop, True, W, [1], EMBED_DIM, device)
    assert np.allclose(out[1], q, atol=1e-6)
