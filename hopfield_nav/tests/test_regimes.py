"""The rollout regimes: what memory each env gets, and when it is rebuilt.

The property pinned here is that an env's Hopfield is *derived from its current
state*, not cached at construction. The exploit regime stores
``encoded_Phi[goal + offset]``, and both of those are things a refresh moves. A
pooled memory would survive the move and go on pointing at the old cell -- while
the reward still fires at the real goal, so PPO would learn that the recall
channel does not pay. That is the opposite of what the regime exists to teach,
and nothing would raise.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield import Hopfield
from hopfield_nav.tests.fixtures import StubVectorHash, make_stub_cfg
from hopfield_nav.training.exploit import ExploitRegime
from hopfield_nav.training.explore import ExploreRegime
from hopfield_nav.training.stages import Knobs
from hopfield_nav.world.env import make_env
from hopfield_nav.world.world import World

EMBED = 8
DEVICE = torch.device("cpu")


def _knobs(**over):
    base = dict(lr=1e-4, empty_frac=0.0, novelty=0.0, eps=0.0, dist_min=0,
                dist_max=0, emp_dist_min=0, emp_dist_max=0)
    base.update(over)
    return Knobs(**base)


def _world(goal=(1, 1), offset=(0, 0)):
    cfg = make_stub_cfg(movement_mode="discrete")
    cfg.hopfield.beta = 1.0
    vh = StubVectorHash(Npos=16, embed_dim=EMBED)
    env = make_env(cfg.env, "discrete", seed=7)
    env.set_goal(goal)
    return cfg, vh, env, World(envs=[env], offsets=[offset], field=vh, assoc=None)


def _reference_W(vh, cell) -> torch.Tensor:
    """The weight matrix a Hopfield holding exactly ``encoded_Phi[cell]`` has."""
    hop = Hopfield(EMBED, beta=1.0, device="cpu")
    hop.input_memory(torch.from_numpy(vh.encoded_Phi[cell[0], cell[1]]).float())
    return hop.W


def test_exploit_memory_is_the_current_goal_cell():
    cfg, vh, env, world = _world(goal=(1, 1), offset=(0, 0))
    regime = ExploitRegime(cfg, EMBED, DEVICE, np.random.RandomState(0))
    hop = regime.spec(0, world, 0, env, (0, 0), _knobs()).hop
    assert hop.num_memories == 1
    assert torch.allclose(hop.W, _reference_W(vh, (1, 1)))


def test_exploit_memory_follows_a_moved_goal():
    """A goal refresh must move the stored pattern with it."""
    cfg, vh, env, world = _world(goal=(1, 1), offset=(0, 0))
    regime = ExploitRegime(cfg, EMBED, DEVICE, np.random.RandomState(0))
    before = regime.spec(0, world, 0, env, (0, 0), _knobs()).hop
    assert torch.allclose(before.W, _reference_W(vh, (1, 1)))

    env.set_goal((3, 3))                       # what a refresh tick does
    after = regime.spec(0, world, 0, env, (0, 0), _knobs()).hop

    assert torch.allclose(after.W, _reference_W(vh, (3, 3)))
    assert not torch.allclose(after.W, before.W), (
        "the memory still points at the old goal cell -- a pooled Hopfield "
        "would train the agent to distrust its own recall")


def test_exploit_memory_follows_a_moved_offset():
    """A placement refresh moves the goal's *global* cell, so the pattern too."""
    cfg, vh, env, world = _world(goal=(1, 1), offset=(0, 0))
    regime = ExploitRegime(cfg, EMBED, DEVICE, np.random.RandomState(0))
    before = regime.spec(0, world, 0, env, (0, 0), _knobs()).hop

    world.offsets = [(5, 5)]                   # what a place refresh does
    after = regime.spec(0, world, 0, env, (5, 5), _knobs()).hop

    assert torch.allclose(after.W, _reference_W(vh, (6, 6)))   # goal + offset
    assert not torch.allclose(after.W, before.W)


def test_each_rollout_gets_its_own_hopfield_object():
    """Distinct objects, so a write in one rollout cannot bleed into the next.

    Sharing was previously sound only because the collector suppresses stores
    when the Hopfield is not a list. Deriving removes the dependence on that.
    """
    cfg, vh, env, world = _world()
    regime = ExploitRegime(cfg, EMBED, DEVICE, np.random.RandomState(0))
    a = regime.spec(0, world, 0, env, (0, 0), _knobs()).hop
    b = regime.spec(0, world, 0, env, (0, 0), _knobs()).hop
    assert a is not b
    assert torch.allclose(a.W, b.W)            # same content, different object


def test_explore_memory_is_empty_without_distractors():
    cfg, vh, env, world = _world()
    regime = ExploreRegime(cfg, EMBED, DEVICE, np.random.RandomState(0))
    hop = regime.spec(0, world, 0, env, (0, 0), _knobs()).hop
    assert hop.num_memories == 0
    assert torch.count_nonzero(hop.W) == 0


@pytest.mark.parametrize("regime_cls, knob", [
    (ExploitRegime, "dist"), (ExploreRegime, "emp_dist"),
])
def test_no_randomness_is_drawn_when_distractors_are_off(regime_cls, knob):
    """An unconstrained run must consume no distractor RNG.

    The `use_distractors` flag is decided once per run precisely so a rollout
    that happens to sample zero distractors still advances the stream like its
    neighbours. Deriving the memory must not quietly change that.
    """
    cfg, vh, env, world = _world()
    rng = np.random.RandomState(0)
    regime = regime_cls(cfg, EMBED, DEVICE, rng, use_distractors=False)
    before = rng.get_state()[2]
    for _ in range(5):
        regime.spec(0, world, 0, env, (0, 0), _knobs(**{f"{knob}_max": 3}))
    assert rng.get_state()[2] == before, "distractor RNG advanced unexpectedly"
