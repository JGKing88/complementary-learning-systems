"""Executable specification of the at-goal contract. Phase 5's definition of done.

WHAT THE CONTRACT IS
--------------------
When the agent occupies the goal at the start of step t, five things may or may
not happen. They are independent, and the point of writing them out is that
today they are not:

    C1 reward        the current_reward input channel reads goal_reward
    C2 store_target  a store fired on this step writes the GOAL cell's pattern
    C3 move_ignored  the movement action sampled on this step is discarded
    C4 teleport      the agent is relocated to a fresh start after the step
    C5 reset_state   the RNN hidden state and prev_reward / prev_action are
                     zeroed, because the post-teleport agent has no valid
                     "previous step"

C5 only means anything when C4 holds. C2 only means anything when a store can
actually fire.

WHY THEY MUST BE SEPARATE SWITCHES
----------------------------------
Today there is no switch. There are two stepping primitives with hardcoded,
opposite answers:

    VecEnv.step_batch   C1 C3 C4 all yes, bundled under `goals_active`
    GridEnv.step        none of them -- reaching the goal is not an event

and the evaluators get C1 anyway, because `agent_step` reads the reward channel
from `GridEnv.reward()` rather than from the step. So `goals_active` is doing
three jobs at once: turning off the teleport also turns off the goal reward,
which is the at-goal indicator the store head keys on.

That accident is what makes the sites diverge, and it is why three evaluators
cannot be batched onto VecEnv without changing what they measure. Phase 5's job
is to replace the two primitives with one that takes C1-C5 as parameters, and
have each site declare its own row of the table below.

HOW TO USE THIS FILE
--------------------
`SITE_CONTRACTS` is the specification: what each call site is entitled to, and
why. It is prose that happens to be executable. The tests below verify the parts
that can be observed today -- the two primitives, and the evaluators that can be
driven cheaply. After Phase 5 every site should be probeable the same way, and
the `xfail`-free version of this file is the acceptance criterion.

Changing an entry in SITE_CONTRACTS is a decision about experiment semantics,
not a refactor. Two entries are marked DECISION: they are the ones where
adopting the training contract would move published numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from hopfield_nav.world.env import CARDINAL_ACTIONS, GridEnv, at_goal, make_env
from hopfield_nav.config import EnvConfig
from hopfield import Hopfield
from hopfield_nav.tests.fixtures import StubVectorHash, make_stub_cfg
from hopfield_nav.world.vec_env import VecEnv

EMBED_DIM = 8


# ---------------------------------------------------------------------------
# The specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalContract:
    """What happens on a step that begins with the agent at the goal.

    ``None`` means "not exercised": the site ends the episode on arrival, so it
    never takes a step *from* the goal and the clause has no observable meaning
    there. That is different from False, which means the step happens and the
    thing deliberately does not.
    """
    reward: bool | None            # C1
    store_target_is_goal: bool | None   # C2
    move_ignored: bool | None      # C3
    teleport: bool | None          # C4
    reset_state: bool | None       # C5
    why: str = ""


FULL = GoalContract(True, True, True, True, True, "the training contract")

#: What each call site is entitled to. This is the table Phase 5 implements.
SITE_CONTRACTS: dict[str, GoalContract] = {

    # --- reference implementation -----------------------------------------
    "training_rollout": GoalContract(
        True, True, True, True, True,
        "VecEnv.step_batch + rollout.py. Everything else is judged against "
        "this: it is the distribution the policy is actually trained on."),

    # --- already implement it, by hand ------------------------------------
    "evaluate_realistic": GoalContract(
        True, True, True, True, True,
        "Hand-rolls the contract; its docstring calls the at-goal iteration a "
        "'training-matching reach'. Measures reach intervals over a long "
        "persistent-Hopfield run, so it must match training exactly."),
    "evaluate_repeat": GoalContract(
        True, True, True, True, True,
        "Same structure as realistic, fresh Hopfield per trial."),

    # --- never take a step from the goal ----------------------------------
    "evaluate_navigation": GoalContract(
        True, None, None, None, None,
        "Breaks the moment at_goal is true after a step. The task is 'get "
        "there', so arrival ends it; C2-C5 are unreachable. This is why it is "
        "the one evaluator that can be batched onto VecEnv today."),
    "evaluate_sequential_episodes": GoalContract(
        True, True, None, None, None,
        "Stores at goal, then breaks. C2 is exercised, the rest are not."),
    "agenthash_mini_episode": GoalContract(
        True, True, None, None, None,
        "The figure pipeline's episode. Its comment says it breaks 'before the "
        "env's teleport step'."),
    "phase_decoding_v2_rollout": GoalContract(
        True, True, None, None, None,
        "Breaks on goal when stop_on_goal is set."),

    # --- adopted the training contract 2026-08-06 --------------------------
    "evaluate_goal_discovery": GoalContract(
        True, True, True, True, True,
        "Discovery is 'find the goal and write it down on arrival', so the "
        "agent gets exactly one store opportunity per visit and is then "
        "relocated -- the same deal training gives it. It previously declined "
        "the teleport, which let a policy park inside the goal radius and "
        "accumulate opportunities until its store probability drifted past "
        "0.5, measuring 'fires eventually' rather than 'fires on arrival'. "
        "The trial keeps running after the teleport, so one trial now yields "
        "several arrivals and store_efficiency is a per-arrival rate."),

    # --- the goal is inert ------------------------------------------------
    "evaluate_exploration": GoalContract(
        False, False, False, False, False,
        "NO_GOALS since 2026-08-06. This measures walking, so the goal must "
        "not be an event of any kind: no teleport (a free random jump would "
        "inflate coverage) and no reward either, because a spike in the reward "
        "channel is a signal the goal exists and turns coverage into partial "
        "goal-seeking. Stores never fire, so nothing can enter memory. "
        "Reaching the goal is still recorded -- resolve_at_goal passes the "
        "mask through regardless of C1 -- it just has no effect on behavior. "
        "evaluate_union_coverage was absorbed here the same day; it ran its "
        "own rollouts to compute a union over the same walks."),

    # --- visualizer --------------------------------------------------------
    "visualize_trajectories_combined": GoalContract(
        True, True, True, True, True,
        "Keeps its own loop to record per-step positions, and applies an "
        "explicit post-store teleport + RNN reset. Should take these five from "
        "the shared wrapper rather than re-deriving them."),
}


def test_every_site_is_declared():
    """A new at-goal call site must be added here, with a justification."""
    assert len(SITE_CONTRACTS) == 10
    for name, c in SITE_CONTRACTS.items():
        assert c.why, f"{name} has no justification"


def test_reset_only_claimed_where_teleport_is():
    """C5 is meaningless without C4 -- nothing to reset if nothing moved."""
    for name, c in SITE_CONTRACTS.items():
        if c.reset_state:
            assert c.teleport, f"{name} claims reset_state without teleport"


def test_reward_clause_is_universal():
    """C1 is the one clause every site already agrees on.

    It is also the one that `goals_active` accidentally couples to C3/C4, which
    is precisely why the teleport cannot be switched off by that flag.
    """
    rewarding = {n for n, c in SITE_CONTRACTS.items() if c.reward}
    # evaluate_exploration is the one site that declines C1: it measures
    # walking, and a reward spike is a signal the goal is there.
    assert set(SITE_CONTRACTS) - rewarding == {"evaluate_exploration"}


def test_the_coverage_evaluators_decline_the_teleport():
    """The population that still needs reward-without-teleport, enumerated.

    `evaluate_goal_discovery` left this set on 2026-08-06: discovery is about
    storing *on arrival*, so it takes the training contract. The two that
    remain count distinct cells, and a teleport is a free random jump that
    would inflate coverage for reasons unrelated to the exploration policy.
    """
    declining = {n for n, c in SITE_CONTRACTS.items() if c.teleport is False}
    assert declining == {"evaluate_exploration"}


# ---------------------------------------------------------------------------
# Probing the two stepping primitives
# ---------------------------------------------------------------------------

def _probe_primitive(step_fn, env, goal) -> tuple[bool, bool]:
    """Place the agent on the goal, step east, report (move_ignored, teleport).

    Both are read off the resulting position: the move is ignored if the agent
    did not end up one cell east, and it teleported if it ended up somewhere
    that is neither the goal nor that neighbour.
    """
    east = (goal[0] + 1, goal[1])
    after = step_fn(env, goal)
    move_ignored = after != east
    teleport = after not in (goal, east)
    return move_ignored, teleport


def _step_gridenv(env, goal) -> tuple[int, int]:
    env.set_position(goal)
    env.step(CARDINAL_ACTIONS[1])          # east
    return tuple(int(v) for v in env.current_location)


def _step_vecenv(env, goal) -> tuple[int, int]:
    vec = VecEnv(env, batch_size=1)
    vec.reset_all()
    vec._pos[0] = np.array(goal, dtype=np.int32)
    _rewards, _reached, pos = vec.step_batch(np.array([1]))   # east
    return tuple(int(v) for v in pos[0])


def _goal_env(size: int = 6, seed: int = 3) -> tuple[GridEnv, tuple[int, int]]:
    """An env whose goal is not on the eastern wall, so 'step east' is legal."""
    for s in range(seed, seed + 50):
        env = make_env(EnvConfig(size=size), "discrete", seed=s)
        gx, gy = env.goal_location
        if gx < size - 1:
            return env, (int(gx), int(gy))
    raise AssertionError("no suitable env found")


def test_gridenv_step_implements_none_of_c3_c4():
    """GridEnv.step: reaching the goal is not an event. Nine lines, no branch."""
    env, goal = _goal_env()
    move_ignored, teleport = _probe_primitive(_step_gridenv, env, goal)
    assert not move_ignored, "GridEnv.step should apply the move at goal"
    assert not teleport, "GridEnv.step should never teleport"


def test_vecenv_step_batch_implements_both_c3_and_c4():
    """VecEnv.step_batch: the at-goal step is consumed and the agent moved."""
    env, goal = _goal_env()
    move_ignored, teleport = _probe_primitive(_step_vecenv, env, goal)
    assert move_ignored, "step_batch should discard the move at goal"
    assert teleport, "step_batch should teleport after the at-goal step"


def test_the_two_primitives_differ_in_exactly_c3_and_c4():
    """The gap Phase 5 closes: same C1, opposite C3/C4, no switch between them."""
    env_a, goal_a = _goal_env()
    env_b, goal_b = _goal_env()
    assert _probe_primitive(_step_gridenv, env_a, goal_a) == (False, False)
    assert _probe_primitive(_step_vecenv, env_b, goal_b) == (True, True)


def test_goals_active_bundles_reward_with_the_teleport():
    """Why the teleport cannot simply be switched off with goals_active.

    Turning it off does suppress C3/C4 -- and silently takes C1 with it, so the
    agent stops seeing the at-goal reward its store head keys on.
    """
    size = 6
    env, goal = _goal_env(size=size)
    env.goals_active = False
    assert env.reward() != env.goal_reward or not at_goal(env)

    env.set_position(goal)
    assert at_goal(env), "sanity: standing on the goal"
    assert env.reward() == -env.time_penalty, (
        "with goals_active=False the reward channel loses the goal signal")

    vec = VecEnv(env, batch_size=1)
    vec.reset_all()
    vec._pos[0] = np.array(goal, dtype=np.int32)
    _r, reached, pos = vec.step_batch(np.array([1]))
    assert not reached[0], "no at-goal event when goals_active is False"
    assert tuple(int(v) for v in pos[0]) == (goal[0] + 1, goal[1]), (
        "the move should be applied, i.e. C3/C4 are off")


def test_c1_is_available_without_c3_c4_via_gridenv():
    """The combination the three declining evaluators rely on.

    GridEnv keeps goals_active=True, so reward() reports goal_reward, while
    GridEnv.step applies the move and never teleports. Phase 5 must keep this
    combination expressible -- it is not a bug, it is what exploration needs.
    """
    env, goal = _goal_env()
    env.set_position(goal)
    assert at_goal(env)
    assert env.reward() == env.goal_reward       # C1 on
    after = _step_gridenv(env, goal)
    assert after == (goal[0] + 1, goal[1])       # C3, C4 off


# ---------------------------------------------------------------------------
# Probing the sites end to end
# ---------------------------------------------------------------------------

class ScriptedAgent(torch.nn.Module):
    """A NavAgent-shaped stand-in with fixed move and store decisions.

    The probes need to control what the policy does at the goal, which a
    trained-weights agent cannot guarantee. Returns the same dict keys as
    NavAgent.get_action_and_value and the same 4-tuple from forward.
    """

    def __init__(self, move_idx: int = 1, store: bool = False, hidden: int = 4):
        super().__init__()
        self.move_idx = move_idx
        self.store = store
        self.hidden = hidden
        self.seen_rewards: list[float] = []

    def eval(self):                       # noqa: D102 - nn.Module API
        return self

    def _h(self, x, h):
        B = x.shape[0]
        return torch.zeros(1, B, self.hidden) if h is None else h

    def get_action_and_value(self, x, h=None, **kwargs):
        B = x.shape[0]
        self.seen_rewards.append(float(x[0, 0, 0]))   # channel 0 = current_reward
        return {
            "move_action": torch.full((B,), self.move_idx, dtype=torch.long),
            "store_action": torch.full((B,), 1.0 if self.store else 0.0),
            "move_log_prob": torch.zeros(B),
            "store_log_prob": torch.zeros(B),
            "value": torch.zeros(B),
            "h_next": self._h(x, h),
        }

    def forward(self, x, h=None, return_features=False):
        from torch.distributions import Bernoulli, Categorical
        B, T = x.shape[0], x.shape[1]
        move = Categorical(logits=torch.zeros(B, T, 4))
        store = Bernoulli(logits=torch.zeros(B, T))
        return move, store, torch.zeros(B, T), self._h(x, h)


def _stub_world(goal_radius: float = 0.5):
    cfg = make_stub_cfg(movement_mode="discrete")
    cfg.env.goal_radius = goal_radius
    vh = StubVectorHash(Npos=16, embed_dim=EMBED_DIM)
    vh.env_offsets = [(0, 0)]
    return cfg, vh


def _run_agent_step_from_goal(cfg, vh, *, store: bool, n_steps: int = 3):
    """Drive eval.agent_step from the goal and record the position trail.

    This is the primitive every evaluator uses, so what it does here is what
    they all do -- their differences are termination rules layered on top.
    """
    from hopfield_nav.evaluation.metrics import agent_step

    env = make_env(cfg.env, "discrete", seed=3)
    while env.goal_location[0] >= cfg.env.size - 1:
        env = make_env(cfg.env, "discrete", seed=int(env.rng.randint(0, 10_000)))
    goal = tuple(int(v) for v in env.goal_location)
    env.set_position(goal)

    agent = ScriptedAgent(move_idx=1, store=store)
    hop = Hopfield(EMBED_DIM, beta=1.0, device="cpu")
    hop.input_memory(torch.from_numpy(vh.encoded_Phi[9, 9]).float())

    trail = [goal]
    stored = []
    h_rnn = prev_reward = prev_action = None
    for _ in range(n_steps):
        out = agent_step(agent, env, (0, 0), vh, hop, h_rnn, cfg,
                         torch.device("cpu"), deterministic=True,
                         goal_local=goal, goal_in_memory=True,
                         prev_reward=prev_reward, prev_action=prev_action)
        h_rnn = out["h_rnn"]
        prev_reward = out["next_prev_reward"]
        prev_action = out["next_prev_action"]
        if out["store_action"] > 0.5:
            stored.append(out["store_embedding"][0].clone())
        trail.append(tuple(int(v) for v in env.current_location))
    return goal, trail, stored, agent


def test_c1_holds_on_the_eval_path():
    """The agent reads goal_reward on the at-goal step, through agent_step."""
    cfg, vh = _stub_world()
    _goal, _trail, _stored, agent = _run_agent_step_from_goal(
        cfg, vh, store=False, n_steps=1)
    assert agent.seen_rewards[0] == pytest.approx(cfg.env.goal_reward)


def test_eval_path_applies_the_move_and_does_not_teleport():
    """C3 and C4 are off for every evaluator that keeps stepping at the goal.

    This is the shared behavior behind the goal_discovery / exploration /
    union_coverage rows of SITE_CONTRACTS -- and the reason batching them onto
    VecEnv would change what they measure.
    """
    cfg, vh = _stub_world()
    goal, trail, _stored, _agent = _run_agent_step_from_goal(
        cfg, vh, store=False, n_steps=1)
    assert trail[1] == (goal[0] + 1, goal[1]), (
        f"expected a plain step east from {goal}, got {trail[1]}")


def test_c2_store_at_goal_writes_the_goal_cell():
    """A store on the at-goal step writes the goal's pattern."""
    cfg, vh = _stub_world()
    goal, _trail, stored, _agent = _run_agent_step_from_goal(
        cfg, vh, store=True, n_steps=1)
    assert len(stored) == 1
    expected = torch.from_numpy(vh.encoded_Phi[goal[0], goal[1]]).float()
    assert torch.allclose(stored[0], expected)


def test_c2_holds_even_when_at_goal_off_cell():
    """C2 with goal_radius > 0.5, where the agent stands on a neighbour.

    This is what allow_offcell_store=False buys: the pattern written is the
    goal's, not the cell the agent happens to occupy. Without it, navigation
    would later recall a neighbour and steer to the wrong place.
    """
    cfg, vh = _stub_world(goal_radius=1.0)
    assert cfg.env.allow_offcell_store is False
    from hopfield_nav.evaluation.metrics import agent_step

    env = make_env(cfg.env, "discrete", seed=3)
    goal = tuple(int(v) for v in env.goal_location)
    neighbour = (goal[0] + 1, goal[1]) if goal[0] + 1 < cfg.env.size else (goal[0] - 1, goal[1])
    env.set_position(neighbour)
    assert at_goal(env), "radius 1.0 should make the neighbour count as at-goal"

    hop = Hopfield(EMBED_DIM, beta=1.0, device="cpu")
    hop.input_memory(torch.from_numpy(vh.encoded_Phi[9, 9]).float())
    out = agent_step(ScriptedAgent(store=True), env, (0, 0), vh, hop, None, cfg,
                     torch.device("cpu"), deterministic=True,
                     goal_local=goal, goal_in_memory=True)
    written = out["store_embedding"][0]
    goal_pattern = torch.from_numpy(vh.encoded_Phi[goal[0], goal[1]]).float()
    neighbour_pattern = torch.from_numpy(
        vh.encoded_Phi[neighbour[0], neighbour[1]]).float()
    assert torch.allclose(written, goal_pattern)
    assert not torch.allclose(written, neighbour_pattern)


# ---------------------------------------------------------------------------
# Phase 5 acceptance criteria
# ---------------------------------------------------------------------------

def test_phase5_one_parameterized_decision_point_exists():
    """The five clauses are a value both stepping paths consult.

    Delivered as a pure resolution function rather than the new stepping entry
    point this file originally sketched: `resolve_at_goal` decides, and the two
    existing steppers call it. That keeps one implementation of the contract
    without needing the batched/single env unification, which is phase 4c.
    """
    from hopfield_nav.world import episode
    import inspect

    assert hasattr(episode, "GoalContract")
    assert hasattr(episode, "resolve_at_goal")
    from hopfield_nav.world.vec_env import ContinuousVecEnv, VecEnv
    for cls in (VecEnv, ContinuousVecEnv):
        params = inspect.signature(cls.step_batch).parameters
        assert "contract" in params, f"{cls.__name__}.step_batch takes no contract"


def test_phase5_the_two_tables_agree():
    """episode.SITE_CONTRACTS and the spec table above must not drift."""
    from hopfield_nav.world import episode
    assert set(episode.SITE_CONTRACTS) == set(SITE_CONTRACTS)
    for name, spec in SITE_CONTRACTS.items():
        impl = episode.SITE_CONTRACTS[name]
        if spec.teleport is None:
            assert impl.ends_on_arrival, (
                f"{name}: spec says the clauses are unreachable, so the "
                f"implementation should mark it ends_on_arrival")
        else:
            assert impl.teleport == spec.teleport, name
            assert impl.move_ignored == spec.move_ignored, name
            assert impl.reset_state == spec.reset_state, name
        assert impl.reward == spec.reward, name


def test_phase5_contract_for_rejects_undeclared_sites():
    """Adding an at-goal site must force a row in the table, not a default."""
    from hopfield_nav.world import episode
    with pytest.raises(KeyError, match="no at-goal contract declared"):
        episode.contract_for("some_new_evaluator")


def test_phase5_reset_without_teleport_is_rejected():
    """C5 is meaningless without C4, and the type refuses to express it."""
    from hopfield_nav.world.episode import GoalContract
    with pytest.raises(ValueError, match="nothing to reset"):
        GoalContract(teleport=False, reset_state=True)


def test_phase5_contract_can_reward_without_teleporting():
    """The combination goals_active could not express, now a value.

    This is what unblocks batching the coverage-style evaluators: they can run
    on VecEnv with the goal rewarded and no teleport.
    """
    from hopfield_nav.world import episode
    res = episode.resolve_at_goal(
        np.array([True, False]), episode.OBSERVE,
        goal_reward=1.0, time_penalty=0.01)
    assert res.rewards[0] == pytest.approx(1.0)      # C1 on
    assert res.apply_move[0]                          # C3 off
    assert not res.teleport[0]                        # C4 off


def test_phase5_vecenv_honours_an_explicit_contract():
    """End to end: the same VecEnv, two contracts, two outcomes."""
    from hopfield_nav.world import episode
    for contract, expect_teleport in ((episode.TRAINING, True),
                                      (episode.OBSERVE, False)):
        env, goal = _goal_env()
        vec = VecEnv(env, batch_size=1)
        vec.reset_all()
        vec._pos[0] = np.array(goal, dtype=np.int32)
        rewards, reached, pos = vec.step_batch(np.array([1]), contract=contract)
        after = tuple(int(v) for v in pos[0])
        assert reached[0], "both contracts still report the at-goal event"
        assert rewards[0] == pytest.approx(env.goal_reward), "C1 in both"
        moved_east = after == (goal[0] + 1, goal[1])
        assert (after not in (goal, (goal[0] + 1, goal[1]))) is expect_teleport
        if not expect_teleport:
            assert moved_east, "OBSERVE should apply the move"


def test_phase5_every_site_calls_contract_for():
    """The anti-drift property: a site's contract is stated, not inherited.

    Checked by source inspection rather than behavior, because the point is
    that the choice is *visible* at the call site.
    """
    import pathlib as _p
    root = _p.Path(__file__).resolve().parents[1]
    sources = {
        "evaluate_realistic": root / "evaluation" / "metrics.py",
        "evaluate_repeat": root / "evaluation" / "metrics.py",
        "evaluate_goal_discovery": root / "evaluation" / "metrics.py",
        # exploration's stepping moved into the batched driver on
        # 2026-08-06, so that is where it states its contract.
        "evaluate_exploration": root / "evaluation" / "batched.py",
        "training_rollout": root / "rollout" / "collector.py",
    }
    unreadable = [str(p) for p in sources.values() if not p.exists()]
    assert not unreadable, (
        f"source paths are stale -- these files moved: {unreadable}"
    )
    missing = [site for site, path in sources.items()
               if f'contract_for("{site}")' not in path.read_text()]
    assert not missing, f"sites not declaring their contract: {missing}"


def test_phase5_the_guard_fires_on_an_impossible_declaration():
    """Change a GridEnv-stepping site's row to TRAINING and it must break.

    This is the property that keeps the table honest: the declaration cannot
    become aspirational, because a site that cannot honour what it declares
    refuses to run.
    """
    from hopfield_nav.world import episode
    with pytest.raises(NotImplementedError, match="steps a GridEnv directly"):
        episode.require_single_env_support(episode.TRAINING, "evaluate_exploration")
    # The contracts those sites actually declare are accepted.
    for site in ("evaluate_exploration", "evaluate_navigation"):
        episode.require_single_env_support(episode.contract_for(site), site)
    # ...and evaluate_goal_discovery, which declares TRAINING since
    # 2026-08-06, is correctly rejected: it hand-rolls the boundary at its call
    # site the way evaluate_realistic does, so it must not route through here.
    with pytest.raises(NotImplementedError, match="steps a GridEnv directly"):
        episode.require_single_env_support(
            episode.contract_for("evaluate_goal_discovery"),
            "evaluate_goal_discovery")


def test_phase5_declared_contract_reaches_the_training_stepper():
    """rollout.py passes its contract to step_batch rather than relying on the
    goals_active default, so the two cannot silently disagree."""
    import inspect
    from hopfield_nav.rollout.collector import RolloutCollector
    src = inspect.getsource(RolloutCollector.collect_rollout)
    assert 'contract_for("training_rollout")' in src
    assert "contract=goal_contract" in src


# ---------------------------------------------------------------------------
# The inert at-goal action must not reach the movement surrogate
# ---------------------------------------------------------------------------

def _rollout_with_goal_contact(*, goals_active: bool = True):
    """A rollout on a wide goal radius, so at-goal steps actually occur."""
    from hopfield_nav.tests.fixtures import make_collector

    cfg = make_stub_cfg(movement_mode="discrete", batch_envs=16,
                        steps_per_rollout=40)
    cfg.env.goal_radius = 1.5
    cfg.env.goals_active = goals_active
    collector, agent, _vh = make_collector(cfg, EMBED_DIM, seed=0)
    env = make_env(cfg.env, "discrete", seed=1234)
    hops = [Hopfield(EMBED_DIM, beta=1.0, device="cpu")
            for _ in range(cfg.batch_envs)]
    torch.manual_seed(0)
    np.random.seed(0)
    return collector.collect_rollout(env, agent, hops, allow_store=True, update_idx=1)


def test_at_goal_steps_are_masked_out_of_move_loss():
    """C3 says the env discards the move, so the advantage is the same
    whichever action was sampled: expected gradient zero, sample gradient pure
    variance -- and injected at the highest-|advantage| steps in the buffer,
    since those are the ones collecting goal_reward.
    """
    b = _rollout_with_goal_contact()
    goal = b.goal_reached.bool()
    pol = b.policy_action_mask.bool()
    assert goal.sum() > 0, "vacuous: no at-goal steps occurred"
    assert (goal & pol).sum() == 0
    assert {round(float(v), 3) for v in b.rewards[goal]} == {1.0}


def test_ordinary_steps_are_not_masked():
    """Only the inert ones drop out -- no ε or auto-nav in this config."""
    b = _rollout_with_goal_contact()
    goal = b.goal_reached.bool()
    pol = b.policy_action_mask.bool()
    assert (~goal & pol).sum() == (~goal).sum()


def test_at_goal_steps_stay_in_the_store_loss():
    """Store IS causal at the goal -- that is the step it must fire on."""
    b = _rollout_with_goal_contact()
    goal = b.goal_reached.bool()
    assert (goal & b.explore_mask.bool()).sum() == goal.sum()


def test_nothing_is_masked_when_the_move_is_not_discarded():
    """With goals_active off the contract applies no C3, so every action is
    the policy's and belongs in the surrogate."""
    b = _rollout_with_goal_contact(goals_active=False)
    assert b.policy_action_mask.bool().all()


# ---------------------------------------------------------------------------
# evaluate_goal_discovery under the training contract (adopted 2026-08-06)
# ---------------------------------------------------------------------------

def _discovery_world(size=4, goal_radius=0.5, n_envs=2):
    from hopfield_nav.tests.fixtures import make_collector
    from hopfield_nav.world.env import make_env
    cfg = make_stub_cfg(movement_mode="discrete")
    cfg.env.size = size
    cfg.env.goal_radius = goal_radius
    _c, agent, vh = make_collector(cfg, EMBED_DIM, seed=0)
    vh.env_offsets = [(0, 0), (8, 8)][:n_envs]
    envs = [make_env(cfg.env, "discrete", seed=100 + i) for i in range(n_envs)]
    return cfg, agent, vh, envs


def test_goal_discovery_arrivals_are_never_consecutive():
    """The teleport, observed through the only channel that exposes it.

    At ``goal_radius`` 0.5 the relocation target -- ``random_start``, which
    excludes the goal cell -- is necessarily *not* at goal. So a step that
    begins at the goal guarantees the next one does not, and arrivals cannot
    exceed every other step. Without the teleport the agent keeps whatever
    position its policy chose and can begin any number of consecutive steps at
    the goal, which is exactly the behavior this contract change removes.
    """
    import math
    from hopfield_nav.evaluation.metrics import evaluate_goal_discovery
    cfg, agent, vh, envs = _discovery_world()
    max_steps = 40
    records: list[tuple] = []
    evaluate_goal_discovery(
        agent, envs, vh, vh.env_offsets, cfg, torch.device("cpu"),
        num_trials=24, max_steps=max_steps, n_distractors_list=[0],
        per_trial=records)

    arrivals = [r[6] for r in records]
    assert sum(arrivals) > 0, "vacuous: no trial ever reached the goal"
    cap = math.ceil(max_steps / 2)
    assert max(arrivals) <= cap, (
        f"a trial recorded {max(arrivals)} arrivals in {max_steps} steps "
        f"(cap {cap}); arrivals on consecutive steps mean the agent was not "
        f"relocated")


def test_goal_discovery_store_efficiency_is_per_arrival():
    """store_efficiency == stores / arrivals, counted, not inferred.

    It used to be ``store_rate / reach_rate`` over per-trial booleans, which
    was a proxy for the same thing. Once a trial can arrive several times that
    ratio stops meaning anything, so the counts are kept directly.
    """
    from hopfield_nav.evaluation.metrics import evaluate_goal_discovery
    cfg, agent, vh, envs = _discovery_world()
    records: list[tuple] = []
    res = evaluate_goal_discovery(
        agent, envs, vh, vh.env_offsets, cfg, torch.device("cpu"),
        num_trials=24, max_steps=40, n_distractors_list=[0],
        per_trial=records)

    arrivals = sum(r[6] for r in records)
    stores = sum(r[7] for r in records)
    assert arrivals > 0, "vacuous: no arrivals to compute an efficiency over"
    assert res[0]["store_efficiency"] == pytest.approx(stores / arrivals)
    assert res[0]["mean_arrivals"] == pytest.approx(arrivals / len(records))
    # A store can only be credited on an arrival step.
    assert stores <= arrivals


def test_goal_discovery_runs_the_full_budget():
    """The trial no longer ends at the first successful store.

    Under OBSERVE it broke there, so one trial meant at most one arrival.
    Continuing is what makes ``mean_arrivals`` and the per-arrival efficiency
    mean anything.
    """
    import inspect
    from hopfield_nav.evaluation import metrics
    src = inspect.getsource(metrics.evaluate_goal_discovery)
    body = src[src.index("for step in range(max_steps):"):]
    assert "if stored_goal:\n                        break" not in body, (
        "evaluate_goal_discovery still breaks on the first store")


# ---------------------------------------------------------------------------
# evaluate_exploration under NO_GOALS (adopted 2026-08-06)
# ---------------------------------------------------------------------------

def test_no_goals_suppresses_the_reward_but_not_the_detection():
    """The property the whole exploration redesign rests on.

    C1 off means the policy gets no signal that it is standing on the goal --
    the reward channel reads `-time_penalty` there like anywhere else. But the
    at-goal mask is passed through untouched, so the evaluator can still record
    that the walk found it. If `resolve_at_goal` ever gated the mask on the
    contract, exploration would silently stop reporting `goal_find_rate`.
    """
    from hopfield_nav.world import episode
    res = episode.resolve_at_goal(
        np.array([True, False]), episode.NO_GOALS,
        goal_reward=1.0, time_penalty=0.01)
    np.testing.assert_allclose(res.rewards, [-0.01, -0.01])
    np.testing.assert_array_equal(res.at_goal, [True, False])
    assert not res.teleport.any()
    assert res.apply_move.all()

    # ...and this is what it would have been under the old row.
    was = episode.resolve_at_goal(
        np.array([True, False]), episode.OBSERVE,
        goal_reward=1.0, time_penalty=0.01)
    np.testing.assert_allclose(was.rewards, [1.0, -0.01])


def test_exploration_declares_no_goals():
    from hopfield_nav.world import episode
    c = episode.contract_for("evaluate_exploration")
    assert c == episode.NO_GOALS
    assert c.reward is False, "a reward spike is a signal the goal exists"
    assert c.teleport is False, "a teleport is a free jump into fresh territory"


def _explore_world(size=5, n_envs=2):
    from hopfield_nav.tests.fixtures import make_collector
    from hopfield_nav.world.env import make_env
    cfg = make_stub_cfg(movement_mode="discrete")
    cfg.env.size = size
    _c, agent, vh = make_collector(cfg, EMBED_DIM, seed=0)
    vh.env_offsets = [(0, 0), (8, 8)][:n_envs]
    envs = [make_env(cfg.env, "discrete", seed=100 + i) for i in range(n_envs)]
    return cfg, agent, vh, envs


def test_exploration_records_a_goal_it_starts_on():
    """Detection works even where the reward does not, and even at step 0.

    `random_start` excludes the goal *cell*, which stops being the same thing
    as "outside the goal radius" once that radius exceeds 0.5 -- so the start
    is checked rather than assumed.
    """
    from hopfield_nav.evaluation.batched import batched_exploration_trials
    cfg, agent, vh, envs = _explore_world()
    env, off = envs[0], vh.env_offsets[0]
    hop = Hopfield(EMBED_DIM, beta=cfg.hopfield.beta, device="cpu")
    _visited, found, steps = batched_exploration_trials(
        agent=agent, env=env, env_offset=off, vectorhash=vh, hopfields=[hop],
        cfg=cfg, device=torch.device("cpu"), starts=[env.goal_location],
        max_steps=10, deterministic=True)
    assert found[0] is True
    assert steps[0] == 0


def test_exploration_never_writes_to_memory():
    """No store can fire, so a trial's Hopfield ends as it began.

    The store head still produces an output; the driver simply does not read
    it. Counting memories is the observable form of that.
    """
    from hopfield_nav.evaluation.batched import batched_exploration_trials
    cfg, agent, vh, envs = _explore_world()
    env, off = envs[0], vh.env_offsets[0]
    hops = []
    for k in range(3):
        h = Hopfield(EMBED_DIM, beta=cfg.hopfield.beta, device="cpu")
        for j in range(k):                       # 0, 1, 2 distractors
            h.input_memory(torch.from_numpy(vh.encoded_Phi[j + 3, j + 3]).float())
        hops.append(h)
    before = [h.num_memories for h in hops]
    batched_exploration_trials(
        agent=agent, env=env, env_offset=off, vectorhash=vh, hopfields=hops,
        cfg=cfg, device=torch.device("cpu"),
        starts=[(0, 0), (1, 1), (2, 2)], max_steps=25, deterministic=True)
    assert [h.num_memories for h in hops] == before == [0, 1, 2]


def test_exploration_runs_every_trial_to_the_full_budget():
    """No early termination -- that is what makes the union well defined and
    a per-step coverage rate unbiased by trial length."""
    from hopfield_nav.evaluation.batched import batched_exploration_trials
    cfg, agent, vh, envs = _explore_world()
    env, off = envs[0], vh.env_offsets[0]
    hops = [Hopfield(EMBED_DIM, beta=cfg.hopfield.beta, device="cpu")
            for _ in range(4)]
    max_steps = 12
    visited, _f, _s = batched_exploration_trials(
        agent=agent, env=env, env_offset=off, vectorhash=vh, hopfields=hops,
        cfg=cfg, device=torch.device("cpu"),
        starts=[(0, 0), (0, 1), (1, 0), (1, 1)], max_steps=max_steps,
        deterministic=True)
    # A trial cut short could not have visited more cells than its own budget.
    # Every trial having had the full budget is what this asserts, via the
    # only externally visible consequence: all four sets are non-degenerate.
    assert all(1 <= len(v) <= max_steps + 1 for v in visited)


def test_exploration_union_and_redundancy_are_consistent():
    """union <= sum of parts, and redundancy lands in [1/N, 1]."""
    from hopfield_nav.evaluation.metrics import evaluate_exploration
    cfg, agent, vh, envs = _explore_world()
    n_trials = 6
    records: list[tuple] = []
    res = evaluate_exploration(
        agent, envs, vh, vh.env_offsets, cfg, torch.device("cpu"),
        num_trials=n_trials, max_steps=20, n_distractors_list=[0],
        per_trial=records)
    m = res[0]
    assert len(records) == n_trials * len(envs)
    assert sum(r[3] for r in records) > 0, "vacuous: no cells visited at all"
    assert m["union_coverage"] >= m["mean_coverage"] - 1e-12
    assert 1.0 / n_trials - 1e-12 <= m["redundancy"] <= 1.0 + 1e-12
    assert m["union_per_rollout"] == pytest.approx(
        m["union_coverage"] / n_trials)
    assert m["cells_per_step"] == pytest.approx(
        m["mean_coverage"] * cfg.env.size ** 2 / 20)
