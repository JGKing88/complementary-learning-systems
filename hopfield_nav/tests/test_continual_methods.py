"""Behavioural tests for the continual-learning methods.

"It ran without crashing" is worth almost nothing for this suite: a replay
buffer that silently never samples, or an EWC penalty that is always zero,
produces a perfectly clean run whose curve is identical to the naive baseline
-- and would be reported as "the method does not help." Every test here checks
that the method does the specific thing it is supposed to do, and several are
written so that the obvious way of breaking the method fails them.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hopfield_nav.config import (
    EnvConfig, RNNAgentConfig, RNNBCConfig, RNNTrainConfig)
from hopfield_nav.continual.base import (
    NoMethod, build_method, parse_method_args)
from hopfield_nav.continual.regularize import OnlineEWC
from hopfield_nav.continual.replay import ExperienceReplay
from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from hopfield_nav.training.rnn_sequential import run_sequential_blocks
from hopfield_nav.world.env import GridEnv

OBS = 16
SIZE = 6


def _rollout(b: int = 2, t: int = 5, d: int = OBS, tag: float = 0.0):
    """A stand-in RNNRolloutBatch. `tag` marks it so identity is checkable."""
    from hopfield_nav.rollout.rnn import RNNRolloutBatch
    return RNNRolloutBatch(
        obs=torch.full((b, t, d), tag),
        teacher_move_action=torch.zeros((b, t, 2)),
        move_label_mask=torch.ones((b, t)),
        rewards=torch.zeros((b, t)),
        goal_reached=torch.zeros((b, t)),
        student_move_action=torch.zeros((b, t, 2)),
    )


def _agent(hidden: int = 8, **kw) -> RNNAgent:
    cfg = RNNAgentConfig(hidden_size=hidden, movement_mode="continuous",
                         init_log_std=-1.0, freeze_log_std=True, **kw)
    return RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS))


# ===========================================================================
# registry / arg parsing
# ===========================================================================

def test_parse_method_args_returns_raw_strings():
    """Parsing does not coerce; `build_method` does, because coercion needs the
    target type. See `test_coercion_is_directed_by_the_target_type`."""
    got = parse_method_args("buffer_size=inf,replay_batches=2,lam=1e3,"
                            "sampling=balanced,normalize_fisher=true")
    assert got == {"buffer_size": "inf", "replay_batches": "2", "lam": "1e3",
                   "sampling": "balanced", "normalize_fisher": "true"}


def test_coercion_is_directed_by_the_target_type():
    """The bug this exists to prevent: `fisher=true` names the *string* "true"
    -- one of two allowed estimators -- while `normalize_fisher=true` means the
    boolean. A parser guessing from the text alone turns the first into `True`
    and the method rejects it, which is what happened the first time Wave 1 ran.
    """
    m = build_method("online_ewc", **parse_method_args(
        "fisher=true,normalize_fisher=true,lam=1e3,fisher_trajectories=4"))
    assert m.fisher == "true" and isinstance(m.fisher, str)
    assert m.normalize_fisher is True
    assert m.lam == 1000.0 and isinstance(m.lam, float)
    assert m.fisher_trajectories == 4 and isinstance(m.fisher_trajectories, int)


def test_coercion_handles_inf_and_ints():
    er = build_method("er", **parse_method_args(
        "buffer_size=inf,replay_batches=3,sampling=reservoir"))
    assert er.buffer_size == float("inf")
    assert er.replay_batches == 3 and isinstance(er.replay_batches, int)
    assert er.sampling == "reservoir"
    er2 = build_method("er", **parse_method_args("buffer_size=200"))
    assert er2.buffer_size == 200


def test_coercion_rejects_a_non_boolean_for_a_boolean():
    with pytest.raises(ValueError, match="expected a boolean"):
        build_method("online_ewc", **parse_method_args("normalize_fisher=maybe"))


def test_parse_method_args_empty():
    assert parse_method_args(None) == {} and parse_method_args("") == {}


def test_build_method_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown continual method"):
        build_method("definitely_not_a_method")


def test_build_method_rejects_unknown_arg():
    """A typo in a sweep script must crash at launch, not run at the default.

    This is the failure that would otherwise be invisible: `--method_args
    lambda=1e3` (instead of `lam`) would silently give lam=100.0 and the run
    would look like a tuned result.
    """
    with pytest.raises(ValueError, match="unknown args"):
        build_method("online_ewc", lambda_=1e3)


def test_seed_forwarded_only_where_it_exists():
    """Callers pass seed unconditionally; methods without an RNG must not break."""
    assert isinstance(build_method("er", seed=7), ExperienceReplay)
    assert isinstance(build_method("online_ewc", seed=7), OnlineEWC)
    assert isinstance(build_method("none", seed=7), NoMethod)


# ===========================================================================
# Experience Replay
# ===========================================================================

def test_er_empty_buffer_returns_nothing():
    er = ExperienceReplay()
    assert er.extra_batches(_rollout(), 0) == []


def test_er_samples_only_strictly_older_data():
    """`extra_batches` runs before `after_update`, so the current rollout can
    never come back as its own replay. Reversing that order is the natural
    implementation mistake and this test is what catches it."""
    er = ExperienceReplay(replay_batches=4, seed=0)
    for step in range(6):
        cur = _rollout(tag=float(step))
        sampled = er.extra_batches(cur, block=0)
        tags = {float(s.obs[0, 0, 0]) for s in sampled}
        assert float(step) not in tags, (
            f"update {step} replayed itself; tags={sorted(tags)}")
        assert all(t < step for t in tags)
        er.after_update(cur, 0, agent=None)


def test_er_unbounded_buffer_keeps_everything():
    er = ExperienceReplay(buffer_size=float("inf"))
    for i in range(25):
        er.after_update(_rollout(tag=float(i)), i % 3, agent=None)
    assert len(er._buf) == 25
    assert er._seen == 25


def test_er_bounded_buffer_stays_bounded():
    er = ExperienceReplay(buffer_size=5, seed=0)
    for i in range(60):
        er.after_update(_rollout(tag=float(i)), i % 3, agent=None)
    assert len(er._buf) == 5
    assert er._seen == 60


def test_er_reservoir_eviction_is_not_just_the_first_k():
    """A buffer that fills and then ignores the stream is a real bug that a
    size check alone would pass."""
    er = ExperienceReplay(buffer_size=5, sampling="reservoir", seed=3)
    for i in range(200):
        er.after_update(_rollout(tag=float(i)), 0, agent=None)
    tags = sorted(float(r.obs[0, 0, 0]) for r in er._buf)
    assert max(tags) >= 5, f"buffer never accepted anything after filling: {tags}"


def test_er_balanced_sampling_spreads_across_blocks():
    """The point of balanced sampling: a draw late in training must not be
    dominated by the most recent env just because it contributed most items."""
    er = ExperienceReplay(sampling="balanced", seed=0)
    for _ in range(3):
        er.after_update(_rollout(), 0, agent=None)      # env 0: 3 items
    for _ in range(97):
        er.after_update(_rollout(), 1, agent=None)      # env 1: 97 items

    idx = er._sample(400)
    blocks = [er._block_of[i] for i in idx]
    frac0 = blocks.count(0) / len(blocks)
    assert 0.35 < frac0 < 0.65, (
        f"balanced sampling gave env 0 a share of {frac0:.2f}; a stream-uniform "
        "draw would give ~0.03")


def test_er_reservoir_sampling_follows_the_stream():
    """The contrast case, so the previous test is measuring the sampler and not
    an accident of the fixture."""
    er = ExperienceReplay(sampling="reservoir", seed=0)
    for _ in range(3):
        er.after_update(_rollout(), 0, agent=None)
    for _ in range(97):
        er.after_update(_rollout(), 1, agent=None)

    idx = er._sample(90)
    frac0 = [er._block_of[i] for i in idx].count(0) / 90
    assert frac0 < 0.2, f"reservoir draw should track the stream, got {frac0:.2f}"


def test_er_replay_batches_zero_disables_replay():
    er = ExperienceReplay(replay_batches=0)
    for i in range(5):
        er.after_update(_rollout(), 0, agent=None)
    assert er.extra_batches(_rollout(), 0) == []


def test_er_state_bytes_tracks_the_buffer():
    er = ExperienceReplay()
    assert er.state_bytes() == 0
    er.after_update(_rollout(), 0, agent=None)
    one = er.state_bytes()
    assert one > 0
    er.after_update(_rollout(), 0, agent=None)
    assert er.state_bytes() == pytest.approx(2 * one)


def test_er_declares_it_needs_no_task_signal():
    """Load-bearing for the results table's 'boundary-free' column."""
    er = ExperienceReplay()
    assert er.needs_task_boundaries is False
    assert er.needs_task_id is False


# ===========================================================================
# Online EWC
# ===========================================================================

def test_ewc_no_penalty_before_any_block_ends():
    assert OnlineEWC().penalty(_agent()) is None


def _trainable(agent):
    return {n: p for n, p in agent.named_parameters() if p.requires_grad}


def test_ewc_penalty_matches_the_closed_form():
    """0.5 * lam * sum_k F_k (theta_k - theta*_k)^2, over TRAINABLE params only.

    `freeze_log_std=True` leaves `movement_log_std` in `parameters()` with
    requires_grad False. Penalising it would be meaningless -- it cannot move --
    and the implementation correctly skips it, so the closed form has to skip it
    too or the test is asserting the wrong number.
    """
    agent = _agent()
    ewc = OnlineEWC(lam=3.0)
    ewc._fisher = {n: torch.full_like(p, 2.0) for n, p in _trainable(agent).items()}
    ewc._anchor = {n: p.detach().clone() for n, p in _trainable(agent).items()}

    with torch.no_grad():
        for p in agent.parameters():
            p.add_(0.5)

    n_params = sum(p.numel() for p in _trainable(agent).values())
    expected = 0.5 * 3.0 * 2.0 * (0.5 ** 2) * n_params
    assert float(ewc.penalty(agent).detach()) == pytest.approx(expected, rel=1e-5)


def test_ewc_penalty_skips_frozen_parameters():
    """The reason the closed form above excludes them -- pinned explicitly so a
    change to that behaviour is a deliberate one."""
    agent = _agent()
    frozen = [n for n, p in agent.named_parameters() if not p.requires_grad]
    assert frozen, "fixture no longer has a frozen parameter to test with"

    ewc = OnlineEWC(lam=1.0)
    ewc._fisher = {n: torch.full_like(p, 1.0) for n, p in agent.named_parameters()}
    ewc._anchor = {n: torch.zeros_like(p) for n, p in agent.named_parameters()}
    with torch.no_grad():
        for n, p in agent.named_parameters():
            p.fill_(1.0 if n in frozen else 0.0)
    # Every trainable parameter sits exactly on its anchor, so anything nonzero
    # could only have come from the frozen one.
    assert float(ewc.penalty(agent).detach()) == pytest.approx(0.0, abs=1e-9)


def test_ewc_penalty_is_zero_at_the_anchor():
    agent = _agent()
    ewc = OnlineEWC(lam=10.0)
    ewc._fisher = {n: torch.ones_like(p) for n, p in agent.named_parameters()}
    ewc._anchor = {n: p.detach().clone() for n, p in agent.named_parameters()}
    assert float(ewc.penalty(agent).detach()) == pytest.approx(0.0, abs=1e-9)


def test_ewc_penalty_tracks_current_parameters():
    """`penalty` is called inside the optimisation loop and must be recomputed
    from live parameters. A cached scalar would stop constraining after the
    first minibatch step -- and would still make every other test pass."""
    agent = _agent()
    ewc = OnlineEWC(lam=1.0)
    ewc._fisher = {n: torch.ones_like(p) for n, p in agent.named_parameters()}
    ewc._anchor = {n: p.detach().clone() for n, p in agent.named_parameters()}

    with torch.no_grad():
        for p in agent.parameters():
            p.add_(0.1)
    near = float(ewc.penalty(agent).detach())
    with torch.no_grad():
        for p in agent.parameters():
            p.add_(0.9)
    far = float(ewc.penalty(agent).detach())
    assert far > near * 50, (near, far)


def test_ewc_penalty_is_differentiable_wrt_the_parameters():
    agent = _agent()
    ewc = OnlineEWC(lam=1.0)
    ewc._fisher = {n: torch.ones_like(p) for n, p in agent.named_parameters()}
    ewc._anchor = {n: (p.detach() + 1.0) for n, p in agent.named_parameters()}
    pen = ewc.penalty(agent)
    pen.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in agent.parameters())


def test_ewc_fisher_is_estimated_and_non_negative():
    agent = _agent()
    ewc = OnlineEWC(fisher_trajectories=4)
    for _ in range(3):
        ewc.after_update(_rollout(b=2, t=4), 0, agent)
    ewc.on_block_end(0, agent, envs=[])

    assert ewc._fisher, "no Fisher was produced"
    assert ewc._anchor, "no anchor was set"
    assert ewc._blocks_consolidated == 1
    for f in ewc._fisher.values():
        assert torch.all(f >= 0), "a squared gradient came out negative"
    assert any(float(f.sum()) > 0 for f in ewc._fisher.values()), \
        "the Fisher is identically zero, so the penalty can never bind"


def test_ewc_gamma_accumulates_across_blocks():
    agent = _agent()
    ewc = OnlineEWC(gamma=1.0, fisher_trajectories=4)
    for _ in range(2):
        ewc.after_update(_rollout(b=2, t=4), 0, agent)
    ewc.on_block_end(0, agent, envs=[])
    after_one = {n: f.clone() for n, f in ewc._fisher.items()}

    for _ in range(2):
        ewc.after_update(_rollout(b=2, t=4), 1, agent)
    ewc.on_block_end(1, agent, envs=[])

    grew = [float(ewc._fisher[n].sum()) >= float(v.sum()) - 1e-9
            for n, v in after_one.items()]
    assert all(grew), "gamma=1.0 must accumulate, not overwrite"


def test_ewc_gamma_zero_forgets_previous_fisher():
    """The contrast case for the accumulation test.

    Covers every key, including one the new estimate will not contain (the
    frozen parameter): decaying only the keys present in the new estimate would
    leave that one pinned at 99 forever.
    """
    agent = _agent()
    ewc = OnlineEWC(gamma=0.0, fisher_trajectories=4)
    ewc._fisher = {n: torch.full_like(p, 99.0)
                   for n, p in agent.named_parameters()}
    for _ in range(2):
        ewc.after_update(_rollout(b=2, t=4), 1, agent)
    ewc.on_block_end(1, agent, envs=[])
    assert all(float(f.max()) < 99.0 for f in ewc._fisher.values())


def test_ewc_true_and_empirical_fisher_differ():
    """The plan's central claim about doing EWC properly. If these agreed, the
    'true vs empirical Fisher' distinction would be rhetoric."""
    torch.manual_seed(0)
    rollouts = [_rollout(b=4, t=6) for _ in range(3)]

    def fisher_for(kind: str):
        torch.manual_seed(0)
        agent = _agent()
        ewc = OnlineEWC(fisher=kind, fisher_trajectories=8)
        for r in rollouts:
            ewc.after_update(r, 0, agent)
        ewc.on_block_end(0, agent, envs=[])
        return ewc._fisher

    true_f, emp_f = fisher_for("true"), fisher_for("empirical")
    diffs = [float((true_f[n] - emp_f[n]).abs().max()) for n in true_f]
    assert max(diffs) > 1e-8, \
        "true and empirical Fisher came out identical; one of them is wrong"


def test_ewc_declares_it_needs_boundaries_but_not_task_id():
    ewc = OnlineEWC()
    assert ewc.needs_task_boundaries is True
    assert ewc.needs_task_id is False


def test_ewc_state_bytes_counts_fisher_and_anchor():
    agent = _agent()
    ewc = OnlineEWC(fisher_trajectories=2)
    assert ewc.state_bytes() == 0
    ewc.after_update(_rollout(b=1, t=3), 0, agent)
    ewc.on_block_end(0, agent, envs=[])
    n_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    assert ewc.state_bytes() == pytest.approx(2 * n_params * 4, rel=0.05)


# ===========================================================================
# end to end through the driver
# ===========================================================================

def _tiny_cfg(n_envs: int, updates: int) -> RNNTrainConfig:
    return RNNTrainConfig(
        env=EnvConfig(size=SIZE, observation_size=OBS,
                      movement_mode="continuous", goal_radius=0.5),
        agent=RNNAgentConfig(hidden_size=8, movement_mode="continuous",
                             init_log_std=-1.0, freeze_log_std=True),
        bc=RNNBCConfig(lr=1e-2, epochs=1, n_minibatches=1),
        n_envs=n_envs, updates_per_env=updates,
        batch_envs=2, steps_per_rollout=6, eval_max_steps=6,
    )


def _run(method, n_envs=2, updates=3):
    cfg = _tiny_cfg(n_envs, updates)
    torch.manual_seed(0)
    agent = RNNAgent(cfg.agent, compute_rnn_input_dim(cfg.agent, OBS))
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.bc.lr)
    envs = [GridEnv(size=SIZE, observation_size=OBS, seed=s)
            for s in range(n_envs)]
    seen: list[dict] = []
    blocks = run_sequential_blocks(
        cfg=cfg, agent=agent, optimizer=opt, envs=envs,
        device=torch.device("cpu"), n_eval_trials=1,
        on_update=lambda u: seen.append(u.losses), method=method,
    )
    return agent, blocks, seen


def test_driver_default_is_naive_sgd():
    """No method argument must reproduce the floor exactly."""
    _, blocks, seen = _run(None)
    assert len(blocks) == 2
    assert all(l["n_replay_batches"] == 0.0 for l in seen)
    assert all("penalty" not in l for l in seen)


def test_driver_feeds_replay_batches_into_the_update():
    er = ExperienceReplay(replay_batches=2, seed=0)
    _, _, seen = _run(er)
    assert seen[0]["n_replay_batches"] == 0.0, "nothing to replay on update 1"
    assert seen[-1]["n_replay_batches"] == 2.0
    assert er.state_bytes() > 0


def test_driver_applies_the_ewc_penalty_after_the_first_block():
    ewc = OnlineEWC(lam=10.0, fisher_trajectories=4)
    _, _, seen = _run(ewc, n_envs=2, updates=3)
    assert "penalty" not in seen[0], "penalised before any block ended"
    assert seen[-1].get("penalty", 0.0) > 0.0, "penalty never became active"
    assert ewc._blocks_consolidated == 2


def test_large_lambda_actually_restrains_drift():
    """The functional claim, not just the plumbing: a big lam must leave the
    parameters nearer the block-0 anchor than a zero lam does.

    Two things this test has to control for, and both bit the first version:

    *The comparison is lam=0, not `None`.* Estimating the Fisher samples actions
    from the model, which consumes the global torch RNG. A naive run therefore
    diverges from an EWC run for reasons that have nothing to do with the
    penalty. `OnlineEWC(lam=0.0)` walks exactly the same code path and draws
    exactly the same random numbers, so the only surviving difference is the
    penalty itself.

    *The distance is measured from the anchor, not from init.* The penalty
    cannot bind during block 0 -- there is no Fisher yet -- so block-0 drift is
    identical by construction and including it only dilutes the signal.
    """
    class _SnapAtBlock0(OnlineEWC):
        """`_anchor` is rewritten at every block end, so by the time a run
        finishes it holds the end of block 1, not the point block 1 was
        supposed to stay near. Snapshot the one we actually want."""
        block0_anchor: dict | None = None

        def on_block_end(self, block, agent, envs):
            super().on_block_end(block, agent, envs)
            if block == 0:
                self.block0_anchor = {n: t.clone()
                                      for n, t in self._anchor.items()}

    def run(lam):
        torch.manual_seed(0)
        ewc = _SnapAtBlock0(lam=lam, fisher_trajectories=8)
        agent, _, _ = _run(ewc, n_envs=2, updates=6)
        return agent, ewc

    free_agent, free_ewc = run(0.0)
    held_agent, held_ewc = run(1e6)

    # Block 0 cannot be penalised -- there is no Fisher yet -- so the two runs
    # must reach the same anchor. If they do not, the lam=0 control is not
    # controlling and the comparison below would be meaningless.
    for n, a in free_ewc.block0_anchor.items():
        assert torch.allclose(a, held_ewc.block0_anchor[n], atol=1e-6), \
            f"block 0 diverged at {n}; the lam=0 control is not controlling"

    def dist(agent, anchor):
        return math.sqrt(sum(
            float((p - anchor[n]).detach().pow(2).sum())
            for n, p in agent.named_parameters() if n in anchor))

    d_free = dist(free_agent, free_ewc.block0_anchor)
    d_held = dist(held_agent, free_ewc.block0_anchor)
    assert d_held < d_free, (
        f"lam=1e6 drifted {d_held:.5f} from the block-0 anchor vs "
        f"{d_free:.5f} at lam=0; the penalty is not constraining anything")


# ===========================================================================
# Wave 2: SI, LwF, CLEAR, DER++
# ===========================================================================

from hopfield_nav.continual.distill import CLEAR, DERpp, LwF          # noqa: E402
from hopfield_nav.continual.regularize import SynapticIntelligence     # noqa: E402


def _nudge(agent, eps=0.05):
    with torch.no_grad():
        for p in agent.parameters():
            if p.requires_grad:
                p.add_(eps)


# --- Synaptic Intelligence -------------------------------------------------

def test_si_registered_and_declares_its_needs():
    si = build_method("si", lam=2.0)
    assert isinstance(si, SynapticIntelligence)
    assert si.needs_task_boundaries is True and si.needs_task_id is False


def test_si_no_penalty_before_any_block_ends():
    agent = _agent()
    si = SynapticIntelligence()
    si.on_block_start(0, agent, [])
    assert si.penalty(agent) is None


def test_si_path_integral_accumulates_on_each_step():
    """`after_step` is the only place the per-step gradient and delta exist. If
    the driver stops calling it, omega stays zero and SI silently becomes a
    no-op that still looks like a method in the history."""
    agent = _agent()
    si = SynapticIntelligence()
    si.on_block_start(0, agent, [])
    assert all(float(v.abs().sum()) == 0 for v in si._omega.values())

    for _ in range(3):
        for p in agent.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, 0.1)
        _nudge(agent, 0.01)
        si.after_step(agent)

    assert any(float(v.abs().sum()) > 0 for v in si._omega.values()), \
        "the path integral never accumulated"


def test_si_importance_is_non_negative_and_penalty_binds():
    agent = _agent()
    si = SynapticIntelligence(lam=1.0)
    si.on_block_start(0, agent, [])
    for _ in range(3):
        for p in agent.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, -0.1)   # descending
        _nudge(agent, 0.01)
        si.after_step(agent)
    si.on_block_end(0, agent, [])

    assert si._importance and si._anchor
    for v in si._importance.values():
        assert torch.all(v >= 0), "importance must be clamped non-negative"

    at_anchor = float(si.penalty(agent).detach())
    _nudge(agent, 0.5)
    moved = float(si.penalty(agent).detach())
    assert at_anchor == pytest.approx(0.0, abs=1e-9)
    assert moved > at_anchor


def test_si_end_to_end_through_the_driver():
    si = SynapticIntelligence(lam=1.0)
    _, _, seen = _run(si, n_envs=2, updates=3)
    assert si._blocks_consolidated == 2
    assert seen[-1].get("penalty", 0.0) > 0.0, "SI penalty never became active"


# --- LwF -------------------------------------------------------------------

def test_lwf_stores_no_data():
    """The cheapest point on the memory axis: a model copy, and nothing else."""
    lwf = LwF()
    assert lwf.state_bytes() == 0
    agent = _agent()
    lwf.on_block_start(1, agent, [])
    params = sum(p.numel() * p.element_size() for p in agent.parameters())
    assert lwf.state_bytes() == pytest.approx(params, rel=0.01)


def test_lwf_is_inactive_in_the_first_block():
    """Nothing to preserve yet; snapshotting at block 0 would only pin the
    policy to its initialisation."""
    agent = _agent()
    lwf = LwF()
    lwf.on_block_start(0, agent, [])
    assert lwf._old is None
    assert lwf.aux_loss(agent, _rollout(), []) is None


def test_lwf_kl_is_zero_against_an_unchanged_model():
    """KL(p || p) = 0. If this is nonzero the distillation term is measuring
    something other than divergence from the snapshot."""
    agent = _agent()
    lwf = LwF(alpha=1.0)
    lwf.on_block_start(1, agent, [])
    loss = lwf.aux_loss(agent, _rollout(b=2, t=4), [])
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)


def test_lwf_kl_grows_as_the_model_moves():
    agent = _agent()
    lwf = LwF(alpha=1.0)
    lwf.on_block_start(1, agent, [])
    r = _rollout(b=2, t=4)
    _nudge(agent, 0.05)
    lwf.after_update(r, 1, agent)          # clears the cached snapshot outputs
    near = float(lwf.aux_loss(agent, r, []).detach())
    _nudge(agent, 0.5)
    lwf.after_update(r, 1, agent)
    far = float(lwf.aux_loss(agent, r, []).detach())
    assert 0.0 < near < far, (near, far)


def test_lwf_caches_the_frozen_outputs_but_not_across_updates():
    """The snapshot's outputs are constant within an update, so they are
    computed once -- but the cache must not survive into the next update, or a
    stale target would be distilled against new data."""
    agent = _agent()
    lwf = LwF()
    lwf.on_block_start(1, agent, [])
    r = _rollout(b=1, t=3)
    lwf.aux_loss(agent, r, [])
    assert lwf._cache
    lwf.after_update(r, 1, agent)
    assert not lwf._cache


# --- CLEAR -----------------------------------------------------------------

def test_clear_is_replay_plus_distillation_and_needs_no_boundaries():
    c = build_method("clear", buffer_size="inf", replay_batches="2")
    assert isinstance(c, CLEAR) and isinstance(c, ExperienceReplay)
    assert c.needs_task_boundaries is False and c.needs_task_id is False


def test_clear_distillation_is_inactive_until_a_block_ends():
    agent = _agent()
    c = CLEAR(replay_batches=1)
    c.after_update(_rollout(), 0, agent)
    extra = c.extra_batches(_rollout(), 0)
    assert extra, "fixture should have something to replay"
    assert c.aux_loss(agent, _rollout(), extra) is None


def test_clear_distillation_activates_and_grows_with_drift():
    agent = _agent()
    c = CLEAR(replay_batches=1, clone_coef=1.0)
    for _ in range(3):
        c.after_update(_rollout(b=2, t=4), 0, agent)
    c.on_block_end(0, agent, [])
    extra = c.extra_batches(_rollout(b=2, t=4), 1)

    at_snapshot = float(c.aux_loss(agent, _rollout(b=2, t=4), extra).detach())
    assert at_snapshot == pytest.approx(0.0, abs=1e-6)
    _nudge(agent, 0.3)
    c.after_update(_rollout(b=2, t=4), 1, agent)
    moved = float(c.aux_loss(agent, _rollout(b=2, t=4), extra).detach())
    assert moved > at_snapshot


def test_clear_state_bytes_counts_buffer_and_snapshot():
    agent = _agent()
    c = CLEAR()
    c.after_update(_rollout(), 0, agent)
    buf_only = c.state_bytes()
    c.on_block_end(0, agent, [])
    assert c.state_bytes() > buf_only


# --- DER++ -----------------------------------------------------------------

def test_derpp_targets_stay_aligned_with_the_buffer():
    """The invariant the whole method rests on. If they drift apart, DER++
    distils each replayed trajectory against another trajectory's target and
    every test that only checks 'the loss is positive' still passes."""
    agent = _agent()
    d = DERpp(buffer_size=5, seed=0)
    for i in range(40):
        d.after_update(_rollout(b=1, t=3, tag=float(i)), i % 3, agent)
        assert len(d._targets) == len(d._buf), (
            f"after {i + 1} inserts: {len(d._targets)} targets vs "
            f"{len(d._buf)} buffer entries")


def test_derpp_target_is_zero_error_against_an_unchanged_model():
    agent = _agent()
    d = DERpp(replay_batches=1, alpha=1.0)
    d.after_update(_rollout(b=2, t=4), 0, agent)
    extra = d.extra_batches(_rollout(b=2, t=4), 0)
    loss = d.aux_loss(agent, _rollout(b=2, t=4), extra)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)


def test_derpp_error_grows_as_the_model_moves():
    agent = _agent()
    d = DERpp(replay_batches=1, alpha=1.0)
    d.after_update(_rollout(b=2, t=4), 0, agent)
    extra = d.extra_batches(_rollout(b=2, t=4), 0)
    _nudge(agent, 0.3)
    moved = float(d.aux_loss(agent, _rollout(b=2, t=4), extra).detach())
    assert moved > 1e-6


def test_derpp_declares_it_needs_no_task_signal():
    d = DERpp()
    assert d.needs_task_boundaries is False and d.needs_task_id is False


# --- all of them, through the driver ---------------------------------------

@pytest.mark.parametrize("name,kwargs", [
    ("er", {}),
    ("clear", {"replay_batches": 1}),
    ("derpp", {"replay_batches": 1}),
    ("online_ewc", {"lam": 10.0, "fisher_trajectories": 4}),
    ("si", {"lam": 1.0}),
    ("lwf", {"alpha": 1.0}),
])
def test_every_method_runs_end_to_end(name, kwargs):
    m = build_method(name, seed=0, **kwargs)
    agent, blocks, seen = _run(m, n_envs=2, updates=3)
    assert len(blocks) == 2
    assert len(seen) == 6
    assert m.state_bytes() >= 0
    d = m.describe()
    assert d["method"] == name
    assert "state_bytes" in d


# ===========================================================================
# --freeze_trunk (plan section 3.2 P4)
# ===========================================================================

from analysis.continual.baseline import (                      # noqa: E402
    HEAD_PREFIXES, freeze_trunk_params)


def test_freeze_trunk_holds_the_trunk_and_frees_the_head():
    agent = _agent(hidden=16)
    n_frozen, n_trainable = freeze_trunk_params(agent)
    assert n_frozen > 0 and n_trainable > 0
    for name, p in agent.named_parameters():
        if name.startswith(HEAD_PREFIXES):
            # log_std may be frozen by its own flag; the point is the head is
            # not frozen *by this function*.
            continue
        assert not p.requires_grad, f"{name} should have been held"


def test_freeze_trunk_counts_are_exact():
    """A GRU(16->8) trunk plus a Linear(8->2) head, in continuous mode."""
    agent = _agent(hidden=8)
    n_frozen, n_trainable = freeze_trunk_params(agent)
    head = sum(p.numel() for n, p in agent.named_parameters()
               if n.startswith(HEAD_PREFIXES))
    trunk = sum(p.numel() for n, p in agent.named_parameters()
                if not n.startswith(HEAD_PREFIXES))
    assert n_frozen == trunk
    # log_std is frozen by the fixture's freeze_log_std, so the trainable count
    # is the head minus whatever the fixture already held.
    assert n_trainable <= head and n_trainable > 0


def test_freeze_trunk_actually_stops_the_trunk_moving():
    """The behavioural claim. A run with the trunk held must leave every trunk
    tensor bit-identical, and must still move the head -- a "fix" that froze
    everything would pass the first half and fail the second."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(2, 3)
    agent = RNNAgent(cfg.agent, compute_rnn_input_dim(cfg.agent, OBS))
    before = {n: p.detach().clone() for n, p in agent.named_parameters()}
    freeze_trunk_params(agent)
    opt = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad],
                           lr=1e-2)
    envs = [GridEnv(size=SIZE, observation_size=OBS, seed=s) for s in range(2)]
    run_sequential_blocks(
        cfg=cfg, agent=agent, optimizer=opt, envs=envs,
        device=torch.device("cpu"), n_eval_trials=1,
    )

    moved_head = False
    for n, p in agent.named_parameters():
        if n.startswith(HEAD_PREFIXES):
            if not torch.equal(p.detach(), before[n]):
                moved_head = True
        else:
            assert torch.equal(p.detach(), before[n]), \
                f"held parameter {n} moved anyway"
    assert moved_head, "nothing in the head moved; the run learned nothing"


def test_freeze_trunk_refuses_to_leave_nothing_trainable():
    """If HEAD_PREFIXES ever stops matching -- a rename, a new architecture --
    the run must fail loudly rather than train zero parameters and report a
    plausible-looking flat curve."""
    class _NoHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.Linear(4, 4)

    with pytest.raises(RuntimeError, match="nothing trainable"):
        freeze_trunk_params(_NoHead())
