"""memory_lift must separate an agent that remembers from one that cannot.

Writing the positive control took two attempts, and the first failure is worth
recording: the evaluator detects a goal-reach and *teleports* before it calls
`act`, so a scripted agent watching its own position never sees itself on the
goal and never arms. It looked like the metric had missed an obvious memory
when in fact the fixture had never demonstrated one.

The agent below instead detects the teleport itself -- a position jump larger
than any single step, immediately after standing next to the goal -- which is
exactly the event "I just scored". That is the caricature of a memory: once a
lifetime has found the goal, go straight back to it. If `memory_lift` cannot
see this, it cannot see anything, and the null it reports on the real policies
would be worthless.
"""
import numpy as np
import torch

from hopfield_nav.config import RNNAgentConfig
from hopfield_nav.evaluation import incontext as IC
from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from hopfield_nav.world.env import GridEnv

OBS, SIZE = 24, 5


class Scripted(RNNAgent):
    def __init__(self, cfg, input_dim, goal, remembers):
        super().__init__(cfg, input_dim)
        self._goal = np.asarray(goal, dtype=np.float32)
        self._remembers = remembers
        self._vec = None
        self._armed = None
        self._prev = None
        self._rng = np.random.RandomState(0)

    def reset_state(self, B):
        self._armed = np.zeros(B, dtype=bool)
        self._prev = None

    @torch.no_grad()
    def act(self, x, h=None, deterministic=False):
        pos = self._vec.positions().astype(np.float32)
        B = pos.shape[0]
        if self._armed is None or len(self._armed) != B:
            self.reset_state(B)

        if self._remembers and self._prev is not None:
            jumped = np.linalg.norm(pos - self._prev, axis=1) > 2.0
            was_at_goal = np.linalg.norm(
                self._prev - self._goal[None, :], axis=1) < 1.5
            self._armed |= (jumped & was_at_goal)
        self._prev = pos.copy()

        d = self._goal[None, :] - pos
        n = np.linalg.norm(d, axis=1, keepdims=True)
        home = np.divide(d, n, out=np.zeros_like(d), where=n > 0)
        wander = self._rng.randn(B, 2).astype(np.float32) * 0.6
        step = np.where(self._armed[:, None], home, wander)
        return {"move_action": torch.from_numpy(step.astype(np.float32)),
                "move_log_prob": torch.zeros(B), "h_next": h}


def run(remembers):
    env = GridEnv(size=SIZE, observation_size=OBS, seed=3)
    cfg = RNNAgentConfig(hidden_size=8, movement_mode="continuous",
                         init_log_std=-1.0, freeze_log_std=True)
    agent = Scripted(cfg, compute_rnn_input_dim(cfg, OBS),
                     env.goal_location, remembers)
    real = IC.make_vec

    def spy(*a, **kw):
        v = real(*a, **kw)
        agent._vec = v
        agent.reset_state(v.B)
        return v

    IC.make_vec = spy
    try:
        return IC.evaluate_in_context(env, agent, n_lifetimes=96,
                                      n_episodes=8, max_steps=40,
                                      device=torch.device("cpu"))
    finally:
        IC.make_vec = real




def test_memory_lift_sees_an_agent_that_remembers():
    r = run(remembers=True)
    assert r["memory_lift"] > 0.4, (
        f"memory_lift={r['memory_lift']:+.3f} on an agent that provably "
        "remembers -- the metric cannot see memory, so any null it reports "
        "on a real policy would be worthless")
    assert r["success_by_episode"][-1] > r["success_by_episode"][0] + 0.2, \
        "the lifetime mechanics did not let a remembering agent improve"


def test_memory_lift_does_not_see_memory_in_a_blind_agent():
    r = run(remembers=False)
    assert r["memory_lift"] < 0.2, (
        f"memory_lift={r['memory_lift']:+.3f} on a blind agent -- the metric "
        "reports memory that is not there")
