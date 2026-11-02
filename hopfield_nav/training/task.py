"""The task-faithful regime: one rollout is the task itself.

`docs/TASK_FAITHFUL_PLAN.md`. The agent is introduced to an arena whose memory
holds distractors only, searches, and on its first goal touch an oracle writes
the goal cell's pattern into *that trajectory's* Hopfield -- once, never
anywhere else, and never through the agent's store head. It is teleported and
keeps going; further touches pay and teleport. Explore rewards (novelty,
epsilon) apply until the write, exploit rewards after; wall / persistence /
time / goal are the same throughout, exactly as they are across the two
regimes in `explore.py` / `exploit.py`.

Unlike those two, this regime hands the collector a *list* of B Hopfields:
the write is per trajectory, so a shared instance would let one trajectory's
find plant the goal for the other B-1. The collector refuses a shared
instance whenever stores are allowed (`collector.py`).

Visits (`Knobs.visits` > 1): the composer keeps each trajectory's Hopfield
and its "goal stored" flag across the next `visits - 1` rollouts of the same
env while the recurrent state resets each rollout -- the continual
protocol's revisit trial, where the agent comes back with its memory intact
and no reward yet seen. `new_memory` builds the fresh list at the start of a
visit sequence; the composer passes the previous rollout's list and flags
for the later visits.
"""
from __future__ import annotations

import numpy as np
import torch

from hopfield import Hopfield
from ..config import TrainConfig
from ..memory import backend as memory_backend
from ..rollout.distractors import sample_distractors
from ..rollout.types import RolloutBatch
from .stages import Knobs, RolloutSpec


class TaskRegime:
    # The write happens (an oracle at the goal), so the collector must be
    # told stores are allowed; the agent's head is masked inside the
    # collector under `task_mode`.
    allows_store = True

    def __init__(self, cfg: TrainConfig, embed_dim: int,
                 device: torch.device, dist_rng: np.random.RandomState, *,
                 use_distractors: bool, batch_size: int):
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.device = device
        self.dist_rng = dist_rng
        # Per-run decision, as in ExploitRegime: a sampled count of 0 must
        # still consume the same randomness as its neighbours.
        self.use_distractors = use_distractors
        self.B = int(batch_size)

    def new_memory(self, vh, env, env_offset, knobs: Knobs) -> list[Hopfield]:
        """B fresh Hopfields, each with its own distractor draw
        (n ~ U[dist_min, dist_max], from cells outside this arena).

        Under ``memory_backend='sensory_kv'`` the B memories are sensory
        key-value stores and a distractor is a foreign arena's goal key paired
        with an outside-arena cell's grid code (``memory/backend.py``).

        Under ``memory_backend='none'`` (idea 3) there is no memory: B Nones,
        and the recurrent state is what the composer carries across visits."""
        if memory_backend.is_none(self.cfg):
            return [None] * self.B
        if memory_backend.is_kv(self.cfg):
            mems = []
            for _ in range(self.B):
                n_dist = (int(self.dist_rng.randint(knobs.dist_min, knobs.dist_max + 1))
                          if self.use_distractors else 0)
                mems.append(memory_backend.new_kv_memory(
                    self.cfg, env_offset, env.size, n_dist, self.dist_rng))
            return mems
        hops = []
        for _ in range(self.B):
            hop = Hopfield(self.embed_dim, beta=self.cfg.hopfield.beta,
            storage_rule=self.cfg.hopfield.storage_rule,
                           device=str(self.device))
            if self.use_distractors:
                n_dist = int(self.dist_rng.randint(
                    knobs.dist_min, knobs.dist_max + 1))
                if n_dist > 0:
                    # `env.size`, not `cfg.env.size`: see ExploitRegime.
                    for pat in sample_distractors(
                            vh, env_offset, env.size, n_dist, self.dist_rng):
                        hop.input_memory(torch.from_numpy(pat).float())
            hops.append(hop)
        return hops

    def spec(self, w_idx: int, world, local_idx: int, env, env_offset,
             knobs: Knobs, *, hops: list[Hopfield] | None = None,
             store_fired: np.ndarray | None = None) -> RolloutSpec:
        """`hops` / `store_fired` come from the previous visit; None starts a
        new sequence with a fresh memory."""
        if hops is None:
            hops = self.new_memory(world.field, env, env_offset, knobs)
            store_fired = None
        return RolloutSpec(
            hop=hops,
            allow_store=self.allows_store,
            novelty_reward=knobs.novelty,
            goals_active=True,
            epsilon=knobs.eps,
            # Never an oracle bit: the agent must infer that memory became
            # trustworthy from prev_reward and the readout.
            goal_in_memory_init=False,
            ends_on_goal=False,
            task_mode=True,
            store_fired_init=store_fired,
        )


def merge_visits(rollouts: list[RolloutBatch]) -> RolloutBatch:
    """One visit sequence as one trajectory: its rollouts end to end in time.

    Idea 3 (memory_backend "none") carries the recurrent state from one visit
    to the next, so PPO has to replay the whole sequence from its start for
    the re-forwarded states to match the collected ones, and GAE has to run
    across the visit boundary (the sequence is one episode; only its last
    visit's bootstrap value is a truncation). Only the fields PPO reads are
    joined; the per-visit diagnostics stay on the original rollouts, which is
    where the trainer logs them from.
    """
    if len(rollouts) == 1:
        return rollouts[0]
    for r in rollouts:
        if (r.alive_mask is not None or r.visited_targets is not None
                or r.teacher_move_action is not None):
            raise ValueError("merge_visits joins plain PPO task rollouts only "
                             "(no ends_on_goal, aux visited head or BC labels)")

    def cat(name):
        return torch.cat([getattr(r, name) for r in rollouts], dim=1)

    return RolloutBatch(
        obs=cat("obs"),
        move_actions=cat("move_actions"),
        store_actions=cat("store_actions"),
        move_log_probs=cat("move_log_probs"),
        store_log_probs=cat("store_log_probs"),
        values=cat("values"),
        rewards=cat("rewards"),
        bootstrap_value=rollouts[-1].bootstrap_value,
        goal_reached=cat("goal_reached"),
        explore_mask=cat("explore_mask"),
        policy_action_mask=cat("policy_action_mask"),
        store_fired_final=rollouts[-1].store_fired_final,
        h_final=rollouts[-1].h_final,
    )


__all__ = ["TaskRegime", "merge_visits"]
