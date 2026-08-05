"""Assert the live code still reproduces the golden fixtures bit-for-bit.

The fixtures in ``tests/golden/`` were generated from the pre-refactor code
(see ``gen_golden.py``). Phases 4-6 of the 2026-08 refactor rewrite the policy
input assembly, batch the evaluators, and move most of the package; none of
that is allowed to change a single number here.

A failure means one of two things:

  * the refactor changed behavior -- read the reported diff before anything
    else, especially in the observation goldens, where the dangerous failure
    is "right shape, wrong channel"; or
  * the change was intended -- regenerate with
    ``python -m hopfield_nav.tests.gen_golden`` and say so, with the diff,
    in the commit message.

Never regenerate to turn a red test green without reading the diff. That is
the single thing these files exist to prevent.
"""
from __future__ import annotations

import numpy as np
import pytest

from hopfield_nav.tests import gen_golden
from hopfield_nav.tests.fixtures import make_collector, make_stub_cfg


@pytest.mark.parametrize("name", sorted(gen_golden.GENERATORS))
def test_golden_matches(name):
    """Every stored array is reproduced exactly by the current code."""
    fresh = gen_golden.GENERATORS[name]()
    problems = gen_golden.compare(gen_golden.golden_path(name), fresh)
    assert not problems, (
        f"{len(problems)} golden difference(s) in {name}.npz:\n  "
        + "\n  ".join(problems)
        + "\n\nIf this change is intended, regenerate with:\n"
          "  python -m hopfield_nav.tests.gen_golden\n"
          "and include the diff in the commit message."
    )


def test_golden_files_are_present():
    """A missing golden must fail loudly, not silently skip coverage."""
    missing = [n for n in gen_golden.GENERATORS
               if not gen_golden.golden_path(n).exists()]
    assert not missing, f"missing golden files: {missing}"


# ---------------------------------------------------------------------------
# Channel layout
# ---------------------------------------------------------------------------
#
# The goldens pin the assembled tensors; these pin the *rule* that produced
# them, so a layout change is reported as a layout change rather than as 88
# unrelated array diffs.

CANONICAL_ORDER = [
    "current_reward",       # always present, width 1
    "prev_reward",
    "encoded_state",
    "hopfield_signal",
    "hopfield_multistep",
    "prev_action",
    "sensory",
    "goal_in_memory",
]


def _expected_width(cfg, embed_dim: int) -> int:
    """Input width implied by the canonical channel order, computed by hand.

    Deliberately independent of compute_input_dim: if the two disagree, one of
    them is wrong, and this test says which channel.
    """
    a = cfg.agent
    signal_dim = 4 if a.hopfield_mode == "discrete" else 2
    prev_action_dim = 4 if a.movement_mode == "discrete" else 2
    width = 1                                            # current_reward
    width += 1 if a.input_prev_reward else 0
    width += embed_dim if a.input_encoded_state else 0
    width += signal_dim if a.input_hopfield_signal else 0
    if a.input_hopfield_multistep and a.hopfield_mode == "continuous":
        width += 2 * len(a.input_hopfield_multistep)
    width += prev_action_dim if a.input_prev_action else 0
    width += cfg.env.observation_size if a.input_sensory else 0
    width += 1 if a.input_goal_in_memory else 0
    return width


@pytest.mark.parametrize("name,kwargs", gen_golden.OBS_CONFIGS)
def test_input_width_matches_channel_order(name, kwargs):
    from hopfield_nav.agent import compute_input_dim
    cfg = make_stub_cfg(**kwargs)
    expected = _expected_width(cfg, gen_golden.EMBED_DIM)
    actual = compute_input_dim(cfg.agent, gen_golden.EMBED_DIM,
                               cfg.env.observation_size)
    assert actual == expected, (
        f"{name}: compute_input_dim says {actual}, canonical channel order "
        f"says {expected}")


@pytest.mark.parametrize("name,kwargs", gen_golden.OBS_CONFIGS)
def test_observation_width_matches_agent_input_dim(name, kwargs):
    """The tensor the collector builds is the width the agent was built for."""
    saved = np.load(gen_golden.golden_path("observations"), allow_pickle=False)
    cfg = make_stub_cfg(**kwargs)
    expected = _expected_width(cfg, gen_golden.EMBED_DIM)
    for populated in ("empty", "populated"):
        obs = saved[f"obs__{name}__{populated}"]
        assert obs.shape[-1] == expected, (
            f"{name}/{populated}: golden observation width {obs.shape[-1]} "
            f"!= canonical {expected}")


def test_current_reward_is_channel_zero():
    """current_reward leads every observation, in every configuration.

    It is the only always-on channel, and the at-goal indicator the agent uses
    to decide when to store rides on it -- so its position is load-bearing.
    """
    saved = np.load(gen_golden.golden_path("observations"), allow_pickle=False)
    cfg = make_stub_cfg()
    time_penalty = cfg.env.time_penalty
    goal_reward = cfg.env.goal_reward
    for key in saved.files:
        if not key.startswith("obs__"):
            continue
        ch0 = saved[key][..., 0]
        # Channel 0 only ever takes the two reward values.
        assert np.all(np.isclose(ch0, -time_penalty) | np.isclose(ch0, goal_reward)), (
            f"{key}: channel 0 holds values other than "
            f"{{-{time_penalty}, {goal_reward}}}; it is not current_reward")


def test_bootstrap_observation_has_same_layout_as_main_loop():
    """The truncation bootstrap builds its own input; same width, same channel 0."""
    saved = np.load(gen_golden.golden_path("observations"), allow_pickle=False)
    for name, _ in gen_golden.OBS_CONFIGS:
        for populated in ("empty", "populated"):
            main = saved[f"obs__{name}__{populated}"]
            boot = saved[f"bootstrap_obs__{name}__{populated}"]
            assert boot.shape[-1] == main.shape[-1], (
                f"{name}/{populated}: bootstrap width {boot.shape[-1]} != "
                f"main-loop width {main.shape[-1]}")


def test_eval_observation_has_same_layout_as_rollout():
    """eval.agent_step's third assembly site agrees on width with the collector.

    Values legitimately differ (different trajectory, B=1), but a width or
    channel-count divergence between the training and eval paths is the bug
    class phase 4 exists to remove.
    """
    obs = np.load(gen_golden.golden_path("observations"), allow_pickle=False)
    ev = np.load(gen_golden.golden_path("eval_observations"), allow_pickle=False)
    for name, _ in gen_golden.OBS_CONFIGS:
        for populated in ("empty", "populated"):
            w_train = obs[f"obs__{name}__{populated}"].shape[-1]
            w_eval = ev[f"evalobs__{name}__{populated}"].shape[-1]
            assert w_eval == w_train, (
                f"{name}/{populated}: eval input width {w_eval} != "
                f"rollout input width {w_train}")
