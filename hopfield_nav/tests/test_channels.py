"""The policy input layout: one definition, and it rejects mistakes loudly.

The goldens pin the assembled tensors. These tests pin the assembler: that the
canonical order is what the docstring claims, that widths agree with what the
agent is constructed for, and -- the point of the module -- that a mis-supplied
channel raises instead of silently producing a correctly-shaped, wrongly-ordered
observation.
"""
from __future__ import annotations

import pytest
import torch

from hopfield_nav.policy import channels
from hopfield_nav.policy.agent import compute_input_dim
from hopfield_nav.tests.fixtures import make_stub_cfg


def _full_cfg():
    return make_stub_cfg(
        movement_mode="continuous", input_hopfield_signal=True,
        input_encoded_state=True, input_prev_action=True,
        input_prev_reward=True, input_sensory=True,
        input_goal_in_memory=True, input_hopfield_multistep=[1, 2],
    )


def test_canonical_order():
    cfg = _full_cfg()
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    assert [s.name for s in specs] == [
        "current_reward", "prev_reward", "encoded_state", "hopfield_signal",
        "hopfield_multistep_1", "hopfield_multistep_2", "prev_action",
        "sensory", "goal_in_memory",
    ]


def test_current_reward_is_always_first_and_always_present():
    """It carries the at-goal indicator the store head keys on."""
    for kwargs in ({}, dict(input_hopfield_signal=False),
                   dict(input_encoded_state=True, input_sensory=True)):
        cfg = make_stub_cfg(**kwargs)
        specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
        assert specs[0].name == "current_reward"
        assert specs[0].width == 1


def test_input_dim_is_the_sum_of_the_specs():
    cfg = _full_cfg()
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    total = sum(s.width for s in specs)
    assert channels.input_dim(cfg.agent, 8, cfg.env.observation_size) == total
    # compute_input_dim is the name the rest of the codebase imports; it must
    # stay in agreement, since it is what the agent's first layer is sized by.
    assert compute_input_dim(cfg.agent, 8, cfg.env.observation_size) == total


@pytest.mark.parametrize("movement_mode,expected", [("discrete", 4), ("continuous", 2)])
def test_signal_and_prev_action_widths_follow_mode(movement_mode, expected):
    cfg = make_stub_cfg(movement_mode=movement_mode)
    assert channels.signal_width(cfg.agent) == expected
    assert channels.prev_action_width(cfg.agent) == expected


def test_multistep_channels_only_in_continuous_mode():
    """Discrete mode has no projected-q channels, whatever the config says."""
    cfg = make_stub_cfg(movement_mode="discrete", input_hopfield_multistep=[1, 2])
    names = [s.name for s in channels.channel_specs(cfg.agent, 8, 12)]
    assert not any(n.startswith("hopfield_multistep") for n in names)


def test_build_produces_the_declared_width():
    cfg = _full_cfg()
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    values = {s.name: torch.zeros(3, s.width) for s in specs}
    out = channels.build_policy_input(specs, values, batch_size=3)
    assert out.shape == (3, sum(s.width for s in specs))


def test_channels_land_at_their_declared_offsets():
    """A channel's values appear exactly where `describe` says they do."""
    cfg = _full_cfg()
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    values = {s.name: torch.full((2, s.width), float(i))
              for i, s in enumerate(specs)}
    out = channels.build_policy_input(specs, values, batch_size=2)
    offset = 0
    for i, s in enumerate(specs):
        assert torch.all(out[:, offset:offset + s.width] == float(i)), s.name
        offset += s.width


def test_missing_enabled_channel_raises():
    cfg = _full_cfg()
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    values = {s.name: torch.zeros(3, s.width) for s in specs}
    del values["sensory"]
    with pytest.raises(KeyError, match="sensory"):
        channels.build_policy_input(specs, values, batch_size=3)


def test_wrong_width_raises():
    """The failure this module exists to prevent: same total, wrong channel."""
    cfg = _full_cfg()
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    values = {s.name: torch.zeros(3, s.width) for s in specs}
    values["hopfield_signal"] = torch.zeros(3, 4)   # discrete width in continuous mode
    with pytest.raises(ValueError, match="width 2"):
        channels.build_policy_input(specs, values, batch_size=3)


def test_wrong_batch_raises():
    cfg = _full_cfg()
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    values = {s.name: torch.zeros(3, s.width) for s in specs}
    values["prev_action"] = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="batch 2"):
        channels.build_policy_input(specs, values, batch_size=3)


def test_unbatched_channel_raises():
    cfg = _full_cfg()
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    values = {s.name: torch.zeros(3, s.width) for s in specs}
    values["prev_reward"] = torch.zeros(3)          # (B,) instead of (B, 1)
    with pytest.raises(ValueError, match=r"must be \(B, 1\)"):
        channels.build_policy_input(specs, values, batch_size=3)


def test_disabled_channels_may_be_supplied_and_are_ignored():
    """Callers compute the signal whether or not it ends up in the input."""
    cfg = _full_cfg()
    rich = {s.name: torch.zeros(3, s.width)
            for s in channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)}
    lean = make_stub_cfg(input_hopfield_signal=False, input_encoded_state=False)
    lean_specs = channels.channel_specs(lean.agent, 8, lean.env.observation_size)
    out = channels.build_policy_input(lean_specs, rich, batch_size=3)
    assert out.shape == (3, 1)      # current_reward only


def test_describe_reports_offsets():
    cfg = make_stub_cfg(movement_mode="continuous")
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    assert channels.describe(specs) == "current_reward[0:1] hopfield_signal[1:3]"
