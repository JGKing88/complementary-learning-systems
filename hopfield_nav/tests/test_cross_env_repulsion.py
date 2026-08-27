"""Where cross-environment repulsion actually comes from.

There is no dedicated cross-environment term in the loss. ``near`` is
``(dist < radius) & same_env`` and *everything else off the diagonal* is
repelled, so in a mixed batch every cross-env pair is pushed toward cosine 0
simply by not being near. That is the whole mechanism, and it is why
``single_env_batch=True`` removes it: a batch holding one environment contains
no cross-env pairs to repel.

But that flag changes two things at once — which pairs exist in the loss, and
which environments a gradient step is computed from. ``exclude_cross_env_pairs``
withholds only the pairs, keeping mixed batches, so the two can be told apart.
These tests pin that distinction.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from encoder_training.losses import mse_attract_repel
from encoder_training.train import _build_near_mask


def _batch(n_per_env=4, n_env=3, spacing=100.0):
    """Points from several envs; within an env they are 1 cell apart."""
    env_ids, coords = [], []
    for e in range(n_env):
        for i in range(n_per_env):
            env_ids.append(e)
            coords.append([e * spacing + i, 0.0])
    return (torch.arange(n_per_env * n_env),
            torch.tensor(env_ids), torch.tensor(coords, dtype=torch.float32))


def test_near_is_within_radius_and_same_env():
    idx, env_ids, coords = _batch()
    near, same_env, _ = _build_near_mask(idx, env_ids, coords, None, local_radius=2.0)

    # points 0 and 1 are 1 cell apart in env 0
    assert near[0, 1] and near[1, 0]
    # 0 and 3 are 3 cells apart — same env, too far
    assert not near[0, 3]
    # 0 and 4 are in different envs; never near however close the coords
    assert not near[0, 4]
    assert not near.diagonal().any()
    assert same_env[0, 1] and not same_env[0, 4]


def test_cross_env_pairs_are_repelled_by_default():
    """The default far set is "not near", which sweeps in every cross-env pair.

    This is the mechanism single_env_batch=True removes, so it must be pinned.
    """
    idx, env_ids, coords = _batch()
    near, _, _ = _build_near_mask(idx, env_ids, coords, None, local_radius=2.0)

    # a batch where every cross-env pair is maximally aliased (cos 1) but all
    # near pairs are already perfect: any loss must come from the cross-env
    # pairs alone.
    B = len(idx)
    K = torch.zeros(B, B)
    K[near] = 1.0
    K.fill_diagonal_(1.0)
    cross = env_ids[:, None] != env_ids[None, :]
    K[cross] = 1.0

    loss = mse_attract_repel(K, near, attract_lambda=2.0, repel_weight=5.0)
    assert loss > 0.0, "cross-env aliasing must be penalised by default"


def test_excluding_cross_env_pairs_stops_penalising_them():
    """With the narrower far mask the identical batch is now loss-free."""
    idx, env_ids, coords = _batch()
    near, same_env, _ = _build_near_mask(idx, env_ids, coords, None, local_radius=2.0)

    B = len(idx)
    K = torch.zeros(B, B)
    K[near] = 1.0
    K.fill_diagonal_(1.0)
    K[env_ids[:, None] != env_ids[None, :]] = 1.0

    far = ~near & same_env
    loss = mse_attract_repel(K, near, attract_lambda=2.0, repel_weight=5.0,
                             far_mask=far)
    assert loss == pytest.approx(0.0, abs=1e-6)


def test_exclusion_leaves_within_env_repulsion_intact():
    """Only *cross*-env pairs are withheld; distant same-env pairs still repel.

    Note `~near & same_env` *does* carry the diagonal, because `near` excludes
    it and `same_env` does not. Callers are not expected to strip it — the loss
    intersects any supplied `far_mask` with `~eye`, which
    ``test_far_mask_never_includes_the_diagonal`` pins.
    """
    idx, env_ids, coords = _batch()
    near, same_env, _ = _build_near_mask(idx, env_ids, coords, None, local_radius=2.0)
    far = ~near & same_env

    # 0 and 3 are same env and beyond the radius, so they must still be in far
    assert far[0, 3]
    # 0 and 4 are cross-env, so they must not be
    assert not far[0, 4]


def test_far_mask_never_includes_the_diagonal():
    """A self-pair has cosine 1 by construction; repelling it is meaningless."""
    idx, env_ids, coords = _batch()
    near, same_env, _ = _build_near_mask(idx, env_ids, coords, None, local_radius=2.0)

    B = len(idx)
    K = torch.eye(B)
    sloppy = torch.ones(B, B, dtype=torch.bool)          # includes the diagonal
    loss = mse_attract_repel(K, near, attract_lambda=0.0, repel_weight=1.0,
                             far_mask=sloppy)
    # diagonal entries are 1.0; if they leaked into `far` the loss would be
    # dragged toward 1/B rather than reflecting the off-diagonal zeros only
    assert loss == pytest.approx(0.0, abs=1e-6)


def test_single_env_batch_has_no_cross_env_pairs_to_repel():
    """The two settings coincide when the batch holds one environment.

    Which is the point: under single_env_batch=True the exclusion flag is a
    no-op, so any difference it makes in a mixed batch is attributable to the
    pairs and not to the batching.
    """
    idx, env_ids, coords = _batch(n_per_env=6, n_env=1)
    near, same_env, _ = _build_near_mask(idx, env_ids, coords, None, local_radius=2.0)

    B = len(idx)
    K = torch.rand(B, B).clamp(-1.0, 1.0)
    K = (K + K.T) / 2

    default = mse_attract_repel(K, near, repel_weight=5.0)
    excluded = mse_attract_repel(K, near, repel_weight=5.0,
                                 far_mask=~near & same_env)
    assert default == pytest.approx(float(excluded), abs=1e-6)


def test_build_near_mask_returns_a_triple():
    """Guards the signature: two masks and the pairwise distances.

    The third element was added when the loss gained a distance-graded target,
    which needs the distances the mask was built from rather than recomputing
    them. It is None on the radius<=0 path, where no distance matrix is built
    -- see test_zero_radius_still_returns_both_masks.
    """
    idx, env_ids, coords = _batch()
    out = _build_near_mask(idx, env_ids, coords, None, local_radius=2.0)
    assert isinstance(out, tuple) and len(out) == 3
    near, same_env, dist = out
    assert near.dtype == torch.bool and same_env.dtype == torch.bool
    assert dist is not None and dist.dtype.is_floating_point
    assert dist.shape == near.shape


def test_zero_radius_still_returns_both_masks():
    """The radius<=0 early-return path must not drop the same_env mask."""
    idx, env_ids, coords = _batch()
    near, same_env, dist = _build_near_mask(idx, env_ids, coords, None,
                                            local_radius=0.0)
    assert near.shape == same_env.shape
    assert not near.diagonal().any()
    # radius<=0 means "same env" is the near set
    assert near[0, 1] and not near[0, 4]
    # and no distance matrix is needed on this path, so none is built
    assert dist is None
