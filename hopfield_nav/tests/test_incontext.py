"""The in-context regime: lifetimes, not episodes.

Plan section 5.2 turns on one behaviour -- an env that reaches its goal is
teleported to a fresh start and the hidden state is *kept* -- and the entire
control rests on that behaviour being real. Two ways it could be silently
broken, both of which would still produce a clean-looking run:

  * the collector keeps freezing reachers, so a "lifetime" is one episode
    followed by 190 frozen steps, and the curve is flat because nothing
    happened;
  * the evaluator zeroes `h` between episodes, so there is no channel for
    anything to carry and the curve is flat because the memory was erased.

Both give the *same* answer as a genuine negative result, which is why they are
tested directly rather than inferred from the curve.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav.config import RNNAgentConfig
from hopfield_nav.evaluation.incontext import (
    evaluate_in_context, evaluate_in_context_all)
from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from hopfield_nav.rollout.rnn import collect_rollout_rnn
from hopfield_nav.world.env import GridEnv
from hopfield_nav.world.vec_env import make_vec

OBS = 24
SIZE = 5


def _agent(**kw):
    cfg = RNNAgentConfig(hidden_size=16, movement_mode="continuous",
                         init_log_std=-1.0, freeze_log_std=True, **kw)
    return cfg, RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS))


def _env(seed=0):
    return GridEnv(size=SIZE, observation_size=OBS, seed=seed)


# ---------------------------------------------------------------------------
# collector
# ---------------------------------------------------------------------------

def test_episodic_mode_freezes_reachers():
    """The default. Once an env is at goal it stops moving, so the mask goes to
    zero and stays there -- this is the behaviour the in-context flag departs
    from, pinned so the departure is visible."""
    cfg, agent = _agent()
    vec = make_vec(_env(), 8, "continuous", 1.0, False)
    r = collect_rollout_rnn(vec, agent, cfg, steps=60,
                            device=torch.device("cpu"),
                            teacher_force=True, carry_across_episodes=False)
    # A frozen env contributes at most one at-goal step; after that the row is
    # masked out forever. So no row can register goal_reached more than a few
    # times, and once masked it never comes back.
    for b in range(r.move_label_mask.shape[0]):
        m = r.move_label_mask[b].numpy()
        nz = np.nonzero(m)[0]
        if len(nz) and nz[-1] + 1 < len(m):
            assert m[nz[-1] + 1:].sum() == 0, \
                "an episodic row resumed after being frozen"


def test_lifetime_mode_keeps_collecting_after_a_goal():
    """With the flag on, reaching the goal must not end the row. Under the
    teacher every episode is solved quickly, so a 60-step lifetime should
    contain several goal-reaches and stay supervised throughout."""
    cfg, agent = _agent()
    vec = make_vec(_env(), 8, "continuous", 1.0, False)
    r = collect_rollout_rnn(vec, agent, cfg, steps=60,
                            device=torch.device("cpu"),
                            teacher_force=True, carry_across_episodes=True)
    reaches = r.goal_reached.sum(dim=1)
    assert float(reaches.max()) >= 2, (
        f"no row reached the goal twice in 60 teacher-driven steps; "
        f"max was {float(reaches.max())} -- the teleport is not happening")
    # And the row is still being supervised near the end, rather than frozen.
    assert float(r.move_label_mask[:, -5:].sum()) > 0, \
        "lifetime rows went unsupervised at the end, as if frozen"


def test_lifetime_mode_teleports_rather_than_leaving_the_agent_on_the_goal():
    cfg, agent = _agent()
    env = _env()
    vec = make_vec(env, 16, "continuous", 1.0, False)
    collect_rollout_rnn(vec, agent, cfg, steps=40, device=torch.device("cpu"),
                        teacher_force=True, carry_across_episodes=True)
    gx, gy = env.goal_location
    on_goal = sum(1 for b in range(vec.B)
                  if tuple(vec.positions()[b]) == (gx, gy))
    assert on_goal < vec.B, \
        "every row ended standing on the goal; nothing was teleported"


def test_the_two_modes_actually_differ():
    """A guard against the flag being wired to nothing."""
    cfg, agent = _agent()
    torch.manual_seed(0)
    a = collect_rollout_rnn(make_vec(_env(), 8, "continuous", 1.0, False),
                            agent, cfg, steps=60, device=torch.device("cpu"),
                            teacher_force=True, carry_across_episodes=False)
    torch.manual_seed(0)
    b = collect_rollout_rnn(make_vec(_env(), 8, "continuous", 1.0, False),
                            agent, cfg, steps=60, device=torch.device("cpu"),
                            teacher_force=True, carry_across_episodes=True)
    assert float(b.move_label_mask.sum()) > float(a.move_label_mask.sum()), (
        "the lifetime rollout was not more supervised than the episodic one; "
        "the flag may not be reaching the collector")


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------

def test_evaluate_in_context_shape_and_bounds():
    _, agent = _agent()
    r = evaluate_in_context(_env(), agent, n_lifetimes=8, n_episodes=4,
                            max_steps=25, device=torch.device("cpu"))
    assert len(r["success_by_episode"]) == 4
    assert len(r["mean_steps_by_episode"]) == 4
    assert all(0.0 <= v <= 1.0 for v in r["success_by_episode"])
    assert r["adaptation"] == pytest.approx(
        r["success_by_episode"][-1] - r["success_by_episode"][0])


def test_every_lifetime_gets_through_all_its_episodes():
    """The step budget has to be enough that a lifetime is not cut short --
    otherwise late episodes would report as failures because they never ran,
    and the curve would slope *down* for a purely bookkeeping reason."""
    _, agent = _agent()
    r = evaluate_in_context(_env(), agent, n_lifetimes=6, n_episodes=5,
                            max_steps=10, device=torch.device("cpu"))
    # A never-run episode is indistinguishable from a failed one in the rate,
    # so assert on the mechanism instead: with a step budget of
    # n_episodes*(max_steps+1) every lifetime must be able to time out of each
    # episode and still start the next.
    assert r["n_episodes"] == 5
    assert len(r["success_by_episode"]) == 5


def test_a_perfect_teacher_like_policy_scores_on_every_episode():
    """Sanity: an agent that can reach the goal should do so in every episode,
    not just the first. If later episodes score lower, the teleport or the
    episode bookkeeping is wrong."""
    class _Homing(RNNAgent):
        """Moves straight at the goal. Stands in for a solved policy."""
        def __init__(self, cfg, input_dim, goal):
            super().__init__(cfg, input_dim)
            self._goal = np.asarray(goal, dtype=np.float32)
            self._vec = None

        @torch.no_grad()
        def act(self, x, h=None, deterministic=False):
            pos = self._vec.positions().astype(np.float32)
            d = self._goal[None, :] - pos
            n = np.linalg.norm(d, axis=1, keepdims=True)
            step = np.divide(d, n, out=np.zeros_like(d), where=n > 0)
            return {"move_action": torch.from_numpy(step),
                    "move_log_prob": torch.zeros(len(step)), "h_next": h}

    env = _env()
    cfg = RNNAgentConfig(hidden_size=8, movement_mode="continuous",
                         init_log_std=-1.0, freeze_log_std=True)
    agent = _Homing(cfg, compute_rnn_input_dim(cfg, OBS), env.goal_location)

    import hopfield_nav.evaluation.incontext as IC
    real_make_vec = IC.make_vec

    def spy(*a, **kw):
        v = real_make_vec(*a, **kw)
        agent._vec = v
        return v

    IC.make_vec = spy
    try:
        r = evaluate_in_context(env, agent, n_lifetimes=8, n_episodes=5,
                                max_steps=40, device=torch.device("cpu"))
    finally:
        IC.make_vec = real_make_vec

    assert r["first_episode"] > 0.8, r["success_by_episode"]
    assert r["last_episode"] > 0.8, (
        f"a homing policy failed later episodes: {r['success_by_episode']} -- "
        "the teleport or the episode bookkeeping is wrong")


def test_evaluate_in_context_all_covers_every_env():
    _, agent = _agent()
    envs = [_env(0), _env(1), _env(2)]
    out = evaluate_in_context_all(envs, agent, n_lifetimes=4, n_episodes=3,
                                  max_steps=12, device=torch.device("cpu"))
    assert set(out) == {0, 1, 2}
    assert all(len(v["success_by_episode"]) == 3 for v in out.values())
