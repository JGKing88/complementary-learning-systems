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
from ..rollout.distractors import sample_distractors
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
        (n ~ U[dist_min, dist_max], from cells outside this arena)."""
        hops = []
        for _ in range(self.B):
            hop = Hopfield(self.embed_dim, beta=self.cfg.hopfield.beta,
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


__all__ = ["TaskRegime"]
