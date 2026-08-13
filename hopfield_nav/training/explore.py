"""The explore regime: envs whose Hopfield starts empty of the goal.

The agent has nothing useful to recall, so coverage is the job and novelty is
what pays for it. Optionally the goal reward is switched off entirely
(`goals_off`), which makes the regime purely about covering ground rather than
stumbling onto the goal.

Re-drawing the goal to break the memorization shortcut -- where a fixed sensory
codebook lets the agent learn "in env X, go to position Y" without exploring --
used to live here as `randomize_goal`. It is now `--refresh_goal`
(`training/refresh.py`), which draws from the declared train cell partition
rather than uniformly over the arena, so a refreshed goal cannot land on a cell
reserved for validation.

Epsilon-greedy is applied here and only here -- see `exploit.py` for why.

The Hopfield may still be preloaded with *distractors*: patterns from cells
outside this env's region, with no goal among them. That is what teaches the
explore policy to ignore a recall signal rather than chase it, so eval-time
distractors don't trigger spurious follow behavior.
"""
from __future__ import annotations

import numpy as np
import torch

from hopfield import Hopfield
from ..config import TrainConfig
from ..rollout.distractors import sample_distractors
from .stages import Knobs, RolloutSpec


class ExploreRegime:
    """Per-world pools of empty Hopfields, plus the per-rollout choice.

    Same pooling rule as `ExploitRegime`: shared and reused when the contents
    are fixed, freshly built per rollout when distractors are resampled.
    """

    # Declared here rather than implied by the type of `hop`: this regime
    # never writes, and that is a decision, not a consequence.
    allows_store = False

    def __init__(self, cfg: TrainConfig, embed_dim: int,
                 device: torch.device, dist_rng: np.random.RandomState, *,
                 goals_off: bool = False, use_distractors: bool = False,
                 ends_on_goal: bool = True):
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.device = device
        self.dist_rng = dist_rng
        self.goals_off = goals_off
        self.ends_on_goal = ends_on_goal
        # See ExploitRegime for why this is a per-run decision, not per-update.
        self.use_distractors = use_distractors

    def _build_hop(self, vh, env, env_offset, knobs: Knobs) -> Hopfield:
        """The env's memory for this rollout: empty, plus any distractors.

        Derived rather than pooled, for the same reason as `ExploitRegime` --
        see its `_build_hop`. Nothing this regime stores depends on the env's
        goal or offset, so a stale pool would be harmless here; deriving anyway
        keeps one rule ("the memory is built from the env's current state")
        instead of one rule and an exception.
        """
        hop = Hopfield(self.embed_dim, beta=self.cfg.hopfield.beta,
                       device=str(self.device))
        if self.use_distractors:
            n_dist = int(self.dist_rng.randint(
                knobs.emp_dist_min, knobs.emp_dist_max + 1))
            if n_dist > 0:
                # `env.size`, not `cfg.env.size`: see ExploitRegime.
                for pat in sample_distractors(
                        vh, env_offset, env.size, n_dist, self.dist_rng):
                    hop.input_memory(torch.from_numpy(pat).float())
        return hop

    def spec(self, w_idx: int, world, local_idx: int, env, env_offset,
             knobs: Knobs) -> RolloutSpec:
        hop = self._build_hop(world.field, env, env_offset, knobs)
        return RolloutSpec(
            hop=hop,
            # This regime is scored on coverage, not on remembering anything, and
            # `hop` is one object shared by all B trajectories.
            allow_store=self.allows_store,
            novelty_reward=knobs.novelty,
            goals_active=not self.goals_off,
            epsilon=knobs.eps,
            goal_in_memory_init=False,
            # A found goal ends the episode here rather than teleporting the
            # agent onward. Vacuous under `goals_off`, where there is no goal
            # event to end on.
            ends_on_goal=(not self.goals_off) and self.ends_on_goal,
        )


__all__ = ["ExploreRegime"]
