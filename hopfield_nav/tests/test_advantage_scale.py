"""`advantage_scale_by_group` measures the thing a pooled normalizer hides.

PPO here normalizes advantages once over the whole pooled buffer. When one
update mixes an exploit regime -- whose reward is a `goal_reward`-sized spike --
with an explore regime paid in novelty an order of magnitude smaller, the shared
divisor is set by the exploit rows and the explore objective silently runs at a
fraction of its strength. These tests pin that the diagnostic reports the
fraction, and that it reads 1.0 when there is nothing to hide.
"""
from __future__ import annotations

import torch

from hopfield_nav.rollout.types import RolloutBatch
from hopfield_nav.updates.ppo import advantage_scale_by_group

GAMMA, LAM = 0.99, 0.95


def _rollout(rewards: torch.Tensor) -> RolloutBatch:
    """A rollout with the given rewards and a value head pinned at zero.

    Values at zero make the advantage a pure discounted reward sum, so the
    group scales below are set by the rewards alone.
    """
    B, T = rewards.shape
    z2 = torch.zeros(B, T)
    return RolloutBatch(
        obs=torch.zeros(B, T, 1), move_actions=torch.zeros(B, T, 2),
        store_actions=z2.clone(), move_log_probs=z2.clone(),
        store_log_probs=z2.clone(), values=z2.clone(), rewards=rewards,
        bootstrap_value=torch.zeros(B), goal_reached=z2.clone(),
        explore_mask=torch.ones(B, T),
    )


def _spiky(B: int, T: int, amplitude: float, period: int) -> torch.Tensor:
    """An exploit-shaped reward: a small constant cost, plus rare big spikes."""
    r = torch.full((B, T), -0.05)
    r[:, ::period] = amplitude
    return r


def _dense(B: int, T: int, scale: float, seed: int = 0) -> torch.Tensor:
    """An explore-shaped reward: small, dense, and always present."""
    g = torch.Generator().manual_seed(seed)
    return -0.05 + scale * torch.randint(0, 2, (B, T), generator=g).float()


def test_a_single_group_is_never_rescaled():
    rollouts = [_rollout(_dense(4, 32, 0.3, seed=i)) for i in range(3)]
    out = advantage_scale_by_group(rollouts, ["emp"] * 3, GAMMA, LAM)
    # With one group the pooled divisor IS that group's own std.
    assert out["emp_share"] == 1.0
    assert out["emp_std"] == out["pooled"]


def test_a_large_goal_reward_shrinks_the_explore_gradient():
    """The mechanism, at the reward scales this project actually runs."""
    pre = [_rollout(_spiky(8, 64, amplitude=5.0, period=20)) for _ in range(4)]
    emp = [_rollout(_dense(8, 64, 0.3, seed=i)) for i in range(4)]
    out = advantage_scale_by_group(pre + emp, ["pre"] * 4 + ["emp"] * 4,
                                   GAMMA, LAM)
    # The spiky group dominates the shared divisor, so it keeps most of its
    # gradient while the dense group loses most of its.
    assert out["pre_share"] > 0.9
    assert out["emp_share"] < 0.2
    assert out["pre_std"] > 5 * out["emp_std"]


def test_matched_reward_scales_restore_a_fair_split():
    """Shrinking goal_reward is what makes the two objectives comparable.

    Same schedule, same rollout counts -- only the spike amplitude changes,
    and both groups come back to within a factor of ~2 of an even split. This
    is why `goal_reward` is a schedule knob and not just a reward knob.
    """
    pre = [_rollout(_spiky(8, 64, amplitude=0.5, period=20)) for _ in range(4)]
    emp = [_rollout(_dense(8, 64, 0.3, seed=i)) for i in range(4)]
    out = advantage_scale_by_group(pre + emp, ["pre"] * 4 + ["emp"] * 4,
                                   GAMMA, LAM)
    assert 0.4 < out["emp_share"] < 1.6
    assert 0.4 < out["pre_share"] < 1.6


def test_labels_must_cover_every_rollout():
    rollouts = [_rollout(_dense(2, 8, 0.3))] * 2
    try:
        advantage_scale_by_group(rollouts, ["emp"], GAMMA, LAM)
    except ValueError as exc:
        assert "1 group labels for 2 rollouts" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError")
