"""`advantage_scale_by_group` measures the thing a pooled normalizer hides.

PPO here normalizes advantages once over the whole pooled buffer. When one
update mixes an exploit regime -- whose reward is a `goal_reward`-sized spike --
with an explore regime paid in novelty, the shared divisor belongs to neither,
and each objective silently runs at a fraction of its own strength.

The fraction is not a one-way effect, which is the part worth pinning: a large
`goal_reward` attenuates the explore objective, a small one attenuates the
exploit objective, and somewhere between them the mixing is neutral. That is
what makes `goal_reward` a schedule knob and not just a reward knob, and it is
why the trainer logs the shares per update rather than assuming a value.
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


def _mixed(amplitude: float) -> dict[str, float]:
    """Four exploit-shaped rollouts against four explore-shaped ones."""
    pre = [_rollout(_spiky(8, 64, amplitude=amplitude, period=20))
           for _ in range(4)]
    emp = [_rollout(_dense(8, 64, 0.3, seed=i)) for i in range(4)]
    return advantage_scale_by_group(pre + emp, ["pre"] * 4 + ["emp"] * 4,
                                    GAMMA, LAM)


def test_a_single_group_is_never_rescaled():
    rollouts = [_rollout(_dense(4, 32, 0.3, seed=i)) for i in range(3)]
    out = advantage_scale_by_group(rollouts, ["emp"] * 3, GAMMA, LAM)
    # With one group the pooled divisor IS that group's own std.
    assert out["emp_share"] == 1.0
    assert out["emp_std"] == out["pooled"]


def test_a_large_goal_reward_attenuates_the_explore_objective():
    """v35 ran goal_reward=5 against a novelty of 0.3. This is that ratio."""
    big, moderate = _mixed(5.0), _mixed(1.0)
    assert big["pre_std"] > 2 * big["emp_std"]
    # The explore rows keep less of their gradient the louder exploit gets.
    assert big["emp_share"] < moderate["emp_share"]
    assert big["emp_share"] < 0.5


def test_a_small_goal_reward_attenuates_the_exploit_objective():
    """The same effect with the roles swapped, which is the part that makes
    this a balance rather than a ceiling."""
    tiny, moderate = _mixed(0.05), _mixed(2.0)
    assert tiny["pre_share"] < moderate["pre_share"]
    assert tiny["pre_share"] < 0.5


def test_some_amplitude_makes_the_mixing_neutral():
    """Between the two failure directions, both objectives run at full weight.

    The value here is a property of these synthetic reward scales, not a
    recommendation for `goal_reward` -- the real scales depend on how often the
    policy actually reaches a goal, which is why the trainer measures this per
    update instead of assuming it.
    """
    out = _mixed(2.0)
    assert 0.8 < out["pre_share"] < 1.2
    assert 0.8 < out["emp_share"] < 1.2


def test_labels_must_cover_every_rollout():
    rollouts = [_rollout(_dense(2, 8, 0.3))] * 2
    try:
        advantage_scale_by_group(rollouts, ["emp"], GAMMA, LAM)
    except ValueError as exc:
        assert "1 group labels for 2 rollouts" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError")
