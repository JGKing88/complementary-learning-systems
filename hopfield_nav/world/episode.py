"""The at-goal contract: five clauses, one implementation, declared per site.

When the agent occupies the goal at the start of a step, five things may or may
not happen. They are independent:

    C1 reward                the current_reward channel reads goal_reward
    C2 store_target_is_goal  a store fired here writes the GOAL cell's pattern,
                             not the pattern of whatever cell the agent snapped
                             to (only distinguishable at goal_radius > 0.5)
    C3 move_ignored          the movement action sampled here is discarded
    C4 teleport              the agent is relocated to a fresh start afterwards
    C5 reset_state           the RNN hidden state and prev_reward / prev_action
                             are zeroed, since the post-teleport agent has no
                             valid "previous step"

Before this module there was no switch between them. Two stepping primitives
had opposite hardcoded answers -- ``VecEnv.step_batch`` did C1+C3+C4 bundled
under ``goals_active``, and ``GridEnv.step`` did none of them, reaching the goal
not being an event at all -- while the evaluators got C1 anyway because
``agent_step`` reads the reward channel from ``GridEnv.reward()`` rather than
from the step. So a site's contract was decided by which env class it happened
to construct, and turning the teleport off (``goals_active=False``) silently
took the goal reward with it, which is the at-goal indicator the store head
keys on.

That is why three evaluators cannot be batched onto ``VecEnv`` as-is: they need
C1 without C4, a combination the old bundling could not express.

``resolve_at_goal`` is the whole decision, as a pure function over masks. Both
stepping paths call it, so there is one place where the contract lives and each
caller states which one it wants.

See ``tests/test_goal_contract.py`` for the specification and the per-site
justifications.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class GoalContract:
    """What a step that begins at the goal does.

    ``ends_on_arrival`` is a property of the *caller's loop*, not of the step:
    a site that terminates the episode the moment the agent arrives never takes
    a step from the goal, so C3-C5 are unreachable there. It is recorded here so
    that "does not teleport" and "never gets the chance to" stay distinguishable.
    """
    reward: bool = True                  # C1
    store_target_is_goal: bool = True    # C2
    move_ignored: bool = True            # C3
    teleport: bool = True                # C4
    reset_state: bool = True             # C5
    ends_on_arrival: bool = False

    def __post_init__(self) -> None:
        if self.reset_state and not self.teleport:
            raise ValueError(
                "reset_state without teleport: there is nothing to reset if the "
                "agent did not move to a new episode segment"
            )


#: The distribution the policy is trained on. Everything else is judged
#: against this.
TRAINING = GoalContract()

#: Reward the goal, but let the agent keep walking. What the coverage-style
#: evaluators need and what ``goals_active=False`` could not express, since that
#: would drop C1 as well.
OBSERVE = GoalContract(
    reward=True, store_target_is_goal=True,
    move_ignored=False, teleport=False, reset_state=False,
)

#: The episode ends when the agent arrives, so C3-C5 never apply.
ENDS_ON_ARRIVAL = GoalContract(
    reward=True, store_target_is_goal=True,
    move_ignored=False, teleport=False, reset_state=False,
    ends_on_arrival=True,
)

#: No goals in this world at all: no reward, no boundary. Corresponds to
#: ``goals_active=False`` -- pure-explore phase A.
NO_GOALS = GoalContract(
    reward=False, store_target_is_goal=False,
    move_ignored=False, teleport=False, reset_state=False,
)


#: What each at-goal call site is entitled to. Mirrors -- and is checked
#: against -- SITE_CONTRACTS in tests/test_goal_contract.py, which carries the
#: justification for each row. Changing a row is a decision about experiment
#: semantics, not a refactor.
SITE_CONTRACTS: dict[str, GoalContract] = {
    "training_rollout": TRAINING,
    "evaluate_realistic": TRAINING,
    "evaluate_repeat": TRAINING,
    "evaluate_navigation": ENDS_ON_ARRIVAL,
    "evaluate_sequential_episodes": ENDS_ON_ARRIVAL,
    "agenthash_mini_episode": ENDS_ON_ARRIVAL,
    "phase_decoding_v2_rollout": ENDS_ON_ARRIVAL,
    "evaluate_goal_discovery": OBSERVE,
    "evaluate_exploration": OBSERVE,
    "evaluate_union_coverage": OBSERVE,
    "visualize_trajectories_combined": TRAINING,
}


def contract_for(site: str) -> GoalContract:
    """The declared contract for a call site. Unknown sites are an error.

    Call this instead of hardcoding a contract, so that adding an at-goal site
    forces an entry in the table above rather than an implicit choice.
    """
    try:
        return SITE_CONTRACTS[site]
    except KeyError:
        raise KeyError(
            f"no at-goal contract declared for {site!r}. Add a row to "
            f"world/episode.SITE_CONTRACTS and to the table in "
            f"tests/test_goal_contract.py, with a justification."
        ) from None


def contract_from_goals_active(goals_active: bool) -> GoalContract:
    """The contract the pre-2026-08 bundling implied.

    ``goals_active`` was a single boolean standing in for C1+C3+C4 together.
    This is the compatibility shim that reproduces it, so that callers which
    have not yet been given an explicit contract behave exactly as before.
    """
    return TRAINING if goals_active else NO_GOALS


@dataclass(frozen=True)
class AtGoalResolution:
    """Per-row consequences of a step, given who started it at the goal."""
    rewards: np.ndarray         # (n,) float32
    at_goal: np.ndarray         # (n,) bool -- rows that started at the goal
    apply_move: np.ndarray      # (n,) bool -- rows whose action is applied
    teleport: np.ndarray        # (n,) bool -- rows to relocate afterwards
    reset_state: np.ndarray     # (n,) bool -- rows whose recurrent state resets


def resolve_at_goal(
    at_goal_mask: np.ndarray,
    contract: GoalContract,
    *,
    goal_reward: float,
    time_penalty: float,
) -> AtGoalResolution:
    """Decide what a step does, for each row, from the pre-step at-goal mask.

    Pure: no environment, no side effects. This is the entire at-goal policy,
    so a caller can be read alongside the contract it passed and the two cannot
    drift.

    ``at_goal_mask`` is the *pre-step* mask -- whether the agent occupied the
    goal when it chose its action. That is the convention the whole codebase
    uses, and it is what gives the agent one observable step at the goal in
    which to fire store.
    """
    at_goal_mask = np.asarray(at_goal_mask, dtype=bool).reshape(-1)
    n = at_goal_mask.shape[0]

    rewards = np.full(n, -float(time_penalty), dtype=np.float32)
    if contract.reward:
        rewards[at_goal_mask] = float(goal_reward)

    # A row's action is skipped only where the contract says to ignore it AND
    # the row is actually at the goal.
    ignored = at_goal_mask & contract.move_ignored
    teleport = at_goal_mask & contract.teleport
    reset = teleport & contract.reset_state

    return AtGoalResolution(
        rewards=rewards,
        at_goal=at_goal_mask,
        apply_move=~ignored,
        teleport=teleport,
        reset_state=reset,
    )


def require_single_env_support(contract: GoalContract, site: str) -> GoalContract:
    """Guard for loops that step a bare ``GridEnv`` one env at a time.

    ``GridEnv.step`` applies the move and never teleports -- reaching the goal
    is not an event there -- so it can only honour contracts with C3 and C4 off.
    A site whose declared contract needs either must be routed through a
    contract-aware stepper first.

    This is what stops the table and the code drifting apart: flip a row to
    TRAINING and the site raises here rather than quietly continuing to apply
    the move and stay put.
    """
    if contract.move_ignored or contract.teleport:
        raise NotImplementedError(
            f"{site} declares move_ignored={contract.move_ignored} / "
            f"teleport={contract.teleport}, but it steps a GridEnv directly, "
            f"which implements neither. Either give it a contract-aware "
            f"stepper, or handle the boundary explicitly at the call site the "
            f"way evaluate_realistic does."
        )
    return contract


def with_overrides(contract: GoalContract, **changes) -> GoalContract:
    """A copy with individual clauses changed, e.g. for an ablation."""
    return replace(contract, **changes)


__all__ = [
    "AtGoalResolution",
    "ENDS_ON_ARRIVAL",
    "GoalContract",
    "NO_GOALS",
    "OBSERVE",
    "SITE_CONTRACTS",
    "TRAINING",
    "contract_for",
    "require_single_env_support",
    "contract_from_goals_active",
    "resolve_at_goal",
    "with_overrides",
]
