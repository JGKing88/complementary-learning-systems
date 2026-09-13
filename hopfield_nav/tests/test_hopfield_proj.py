"""`Hopfield(storage_rule="proj")` -- the incremental projection store.

What the rule promises (docs/THEORY_ENCODER_HOPFIELD.md, the `proj` rows, and
analysis/hopfield_probe/graded_fixedpoint_check.py --incremental): stored
patterns are EXACT fixed points, the one-at-a-time form equals the batch
projector Z^T (Z Z^T)^+ Z, and the result does not depend on storage order.
And what must not move: the Hebbian path, which every recorded run used.
"""
import numpy as np
import pytest
import torch

from hopfield import Hopfield

D, K = 64, 12


def _patterns(seed=0, k=K, d=D):
    rng = np.random.RandomState(seed)
    Z = rng.randn(k, d).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return torch.from_numpy(Z)


def _batch_projector(Z):
    Zn = torch.nn.functional.normalize(Z, dim=1).double()
    return Zn.T @ torch.linalg.pinv(Zn @ Zn.T) @ Zn


def test_hebb_is_the_default_and_is_unchanged():
    Z = _patterns()
    hop = Hopfield(D, beta=100.0)
    assert hop.storage_rule == "hebb" and hop.zero_diag is True
    hop.input_memory(Z[0]); hop.input_memory(Z[1])
    W = hop.scale * (torch.outer(Z[0], Z[0]) + torch.outer(Z[1], Z[1]))
    W.fill_diagonal_(0.0)
    assert torch.allclose(hop.W, W, atol=1e-7)
    assert hop.num_memories == 2


def test_proj_stored_patterns_are_exact_fixed_points():
    Z = _patterns()
    hop = Hopfield(D, beta=100.0, storage_rule="proj")
    for z in Z:
        hop.input_memory(z)
    for z in Z:
        # Linear recall: W z = scale * z exactly, so normalize(W z) = z.
        out = hop.recall(z, steps=1, use_tanh=False)
        assert torch.dot(out, z).item() > 1 - 1e-5
        # Production recall (tanh, beta=100, scale=1/D) is in its linear regime.
        out = hop.recall(z, steps=1)
        assert torch.dot(out, z).item() > 0.999


def test_proj_is_the_batch_projector():
    Z = _patterns()
    hop = Hopfield(D, storage_rule="proj")
    for z in Z:
        hop.input_memory(z)
    P = hop.projector().double()
    assert torch.allclose(P, _batch_projector(Z), atol=1e-4)
    assert torch.allclose(P @ P, P, atol=1e-4)          # idempotent
    assert abs(float(torch.trace(P)) - K) < 1e-3          # rank K


def test_proj_is_order_independent():
    Z = _patterns()
    a, b = Hopfield(D, storage_rule="proj"), Hopfield(D, storage_rule="proj")
    for z in Z:
        a.input_memory(z)
    for i in np.random.RandomState(1).permutation(K):
        b.input_memory(Z[int(i)])
    assert torch.allclose(a.W, b.W, atol=1e-5)


def test_proj_pattern_already_in_span_is_counted_but_leaves_W_alone():
    Z = _patterns()
    hop = Hopfield(D, storage_rule="proj")
    hop.input_memory(Z[0]); hop.input_memory(Z[1])
    W_before = hop.W.clone()
    hop.input_memory(Z[0] + Z[1])          # in span(z0, z1): novelty ~0
    assert torch.allclose(hop.W, W_before, atol=1e-6)
    assert hop.num_memories == 3


def test_proj_refuses_zero_diag_but_accepts_explicit_false():
    with pytest.raises(ValueError, match="zero_diag=True is incompatible"):
        Hopfield(D, storage_rule="proj", zero_diag=True)
    hop = Hopfield(D, storage_rule="proj", zero_diag=False)
    assert hop.zero_diag is False
    with pytest.raises(ValueError, match="storage_rule"):
        Hopfield(D, storage_rule="soft")


def test_proj_survives_clone_and_reset():
    Z = _patterns()
    hop = Hopfield(D, storage_rule="proj")
    hop.input_memory(Z[0])
    c = hop.clone()
    assert c.storage_rule == "proj" and torch.equal(c.W, hop.W)
    c.input_memory(Z[1])
    assert not torch.equal(c.W, hop.W)                 # independent W
    hop.reset()
    assert hop.num_memories == 0 and float(hop.W.abs().sum()) == 0.0
    hop.input_memory(Z[2])
    assert torch.dot(hop.recall(Z[2], steps=1, use_tanh=False), Z[2]).item() > 1 - 1e-5


def test_proj_recall_batch_matches_recall():
    Z = _patterns()
    hop = Hopfield(D, beta=100.0, storage_rule="proj")
    for z in Z:
        hop.input_memory(z)
    single = torch.stack([hop.recall(z, steps=1) for z in Z])
    assert torch.allclose(hop.recall_batch(Z, steps=1), single, atol=1e-6)
