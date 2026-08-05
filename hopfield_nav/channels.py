"""The policy input layout, defined once.

Before this module the observation vector was assembled in three places -- the
rollout main loop, the rollout truncation bootstrap, and ``eval.agent_step`` --
each with its own copy of the same ``if cfg.agent.input_x: parts.append(x)``
ladder, and a fourth copy of the *widths* in ``compute_input_dim``. They agreed,
but nothing made them agree, and the way that fails is silent: the tensor keeps
its shape while a channel moves, so the policy reads sensory values as a
prev-action one-hot and the only symptom is worse behavior.

Now there is one ordered list of channels, one function that builds a tensor
from it, and one function that sums its widths. A channel that is enabled but
not supplied is an error rather than a shape mismatch fifty lines later.

Channel order is a compatibility surface
----------------------------------------
Every saved checkpoint's first layer was trained against this order:

    current_reward, prev_reward, encoded_state, hopfield_signal,
    hopfield_multistep_<s>..., prev_action, sensory, goal_in_memory

Reordering it silently invalidates every checkpoint. New channels append at the
end. ``tests/golden/observations.npz`` pins the assembled tensors.

(The RNN baseline has its own, separate layout in ``rollout_rnn.py``. It is a
control model that deliberately shares only the environment, so the two are not
merged here.)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import AgentConfig


@dataclass(frozen=True)
class ChannelSpec:
    """One contiguous slice of the policy input."""
    name: str
    width: int


def signal_width(cfg: AgentConfig) -> int:
    """Width of the Hopfield direction channel: one-hot (discrete) or vector."""
    return 4 if cfg.hopfield_mode == "discrete" else 2


def prev_action_width(cfg: AgentConfig) -> int:
    return 4 if cfg.movement_mode == "discrete" else 2


def multistep_name(step: int) -> str:
    return f"hopfield_multistep_{step}"


def channel_specs(
    cfg: AgentConfig,
    embed_dim: int,
    sensory_dim: int = 0,
) -> list[ChannelSpec]:
    """The channels this config enables, in canonical order.

    ``current_reward`` is always first and always present: it carries the
    at-goal indicator the store head keys on, so its position is load-bearing.
    """
    specs = [ChannelSpec("current_reward", 1)]
    if cfg.input_prev_reward:
        specs.append(ChannelSpec("prev_reward", 1))
    if cfg.input_encoded_state:
        specs.append(ChannelSpec("encoded_state", embed_dim))
    if cfg.input_hopfield_signal:
        specs.append(ChannelSpec("hopfield_signal", signal_width(cfg)))
    if cfg.input_hopfield_multistep and cfg.hopfield_mode == "continuous":
        # Each snapshot of the recall trajectory contributes a projected q.
        for step in cfg.input_hopfield_multistep:
            specs.append(ChannelSpec(multistep_name(step), 2))
    if cfg.input_prev_action:
        specs.append(ChannelSpec("prev_action", prev_action_width(cfg)))
    if cfg.input_sensory:
        specs.append(ChannelSpec("sensory", sensory_dim))
    if cfg.input_goal_in_memory:
        specs.append(ChannelSpec("goal_in_memory", 1))
    return specs


def input_dim(cfg: AgentConfig, embed_dim: int, sensory_dim: int = 0) -> int:
    """Total policy input width: the sum of the enabled channels."""
    return sum(s.width for s in channel_specs(cfg, embed_dim, sensory_dim))


def build_policy_input(
    specs: list[ChannelSpec],
    values: dict[str, torch.Tensor],
    *,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Concatenate the enabled channels into a (B, D) policy input.

    ``values`` maps channel name to a (B, width) tensor. Supplying a channel
    that is not enabled is fine and ignored -- callers compute the Hopfield
    signal whether or not it ends up in the input. Omitting an enabled one, or
    supplying it at the wrong width, raises here rather than corrupting the
    layout downstream.
    """
    parts = []
    for spec in specs:
        value = values.get(spec.name)
        if value is None:
            raise KeyError(
                f"policy input channel {spec.name!r} is enabled but was not "
                f"supplied (have: {sorted(values)})"
            )
        if value.dim() != 2:
            raise ValueError(
                f"channel {spec.name!r} must be (B, {spec.width}), "
                f"got shape {tuple(value.shape)}"
            )
        if value.shape[1] != spec.width:
            raise ValueError(
                f"channel {spec.name!r} must have width {spec.width}, "
                f"got {value.shape[1]}"
            )
        if batch_size is not None and value.shape[0] != batch_size:
            raise ValueError(
                f"channel {spec.name!r} has batch {value.shape[0]}, "
                f"expected {batch_size}"
            )
        parts.append(value)
    return torch.cat(parts, dim=-1)


def describe(specs: list[ChannelSpec]) -> str:
    """`current_reward[0:1] hopfield_signal[1:3] ...` -- for logs and errors."""
    out, offset = [], 0
    for s in specs:
        out.append(f"{s.name}[{offset}:{offset + s.width}]")
        offset += s.width
    return " ".join(out)


__all__ = [
    "ChannelSpec",
    "build_policy_input",
    "channel_specs",
    "describe",
    "input_dim",
    "multistep_name",
    "prev_action_width",
    "signal_width",
]
