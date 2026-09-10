"""Experience Replay over whole trajectories.

The plan's section 0.1 argues that replay is the method to beat here and that
it will probably win on retention, because the entire 5-env x 200-update stream
is ~192 MB and an unbounded buffer is therefore free. This is that method, and
the `buffer_size=inf` setting is deliberately the *default*: the interesting
result is not whether a memory-limited buffer degrades gracefully, it is where
perfect memory sits on the cost frontier against a store that keeps no data at
all.

Two design decisions worth stating, because both could reasonably have gone the
other way:

**A buffer item is a whole trajectory, not a timestep.** `bc_rnn_update` runs
the trunk over `(B, T, D)` and the recurrent state is what carries the agent's
localisation, so a replayed timestep torn out of its trajectory would be
supervised in a context the agent could never have been in. Trajectories also
make the memory accounting honest: what is stored is exactly what was
collected.

**Sampling is per-env balanced by default.** A reservoir over the stream is the
textbook choice, but the stream is *ordered by env*, so a uniform draw late in
training is dominated by recent envs -- which is the same recency bias the
method exists to fix. Balanced sampling costs nothing here and is uniformly
better; `sampling="reservoir"` keeps the textbook version available as an
ablation rather than as the default.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .base import ContinualMethod


def _batch_bytes(rollout) -> int:
    """Bytes a stored rollout actually occupies."""
    total = 0
    for name in ("obs", "teacher_move_action", "move_label_mask",
                 "rewards", "goal_reached", "student_move_action"):
        t = getattr(rollout, name, None)
        if torch.is_tensor(t):
            total += t.numel() * t.element_size()
    return total


class ExperienceReplay(ContinualMethod):
    """Sample `replay_batches` stored trajectories per update; train on them too.

    Needs no task boundaries and no task identity -- it only ever sees a stream
    of rollouts. `block` is accepted so the balanced sampler can stratify, and
    that is a convenience the method could recover by clustering observations;
    it is not privileged information the way a task ID at *eval* time would be.
    """

    name = "er"
    needs_task_boundaries = False
    needs_task_id = False

    def __init__(
        self,
        buffer_size: float = float("inf"),
        replay_batches: int = 1,
        sampling: str = "balanced",
        seed: int = 0,
    ) -> None:
        if replay_batches < 0:
            raise ValueError(f"replay_batches must be >= 0, got {replay_batches}")
        if sampling not in ("balanced", "reservoir"):
            raise ValueError(
                f"sampling must be 'balanced' or 'reservoir', got {sampling!r}")
        self.buffer_size = buffer_size
        self.replay_batches = int(replay_batches)
        self.sampling = sampling
        self.rng = np.random.RandomState(seed)

        self._buf: list = []
        self._block_of: list[int] = []
        #: How many items have ever been offered. Reservoir needs it; it is also
        #: the honest denominator for "what fraction of the stream was kept".
        self._seen = 0

    # -- hooks ------------------------------------------------------------

    def extra_batches(self, rollout, block: int) -> list:
        """Sampled BEFORE `after_update` stores the new rollout, so a replayed
        trajectory is always genuinely older than the one driving the update."""
        if not self._buf or self.replay_batches == 0:
            return []
        idx = self._sample(self.replay_batches)
        return [self._buf[i] for i in idx]

    def after_update(self, rollout, block: int, agent) -> None:
        self._insert(rollout, block)

    # -- buffer -----------------------------------------------------------

    def _sample(self, k: int) -> list[int]:
        n = len(self._buf)
        k = min(k, n)
        if self.sampling == "reservoir":
            return list(self.rng.choice(n, size=k, replace=False))

        # Balanced: pick blocks uniformly, then an item uniformly within the
        # block. Without replacement across the whole draw would be nicer but
        # would bias toward blocks with few items once one is exhausted; the
        # per-draw form is what "uniform over envs" actually means.
        by_block: dict[int, list[int]] = {}
        for i, b in enumerate(self._block_of):
            by_block.setdefault(b, []).append(i)
        blocks = sorted(by_block)
        out: list[int] = []
        for _ in range(k):
            b = blocks[self.rng.randint(len(blocks))]
            pool = by_block[b]
            out.append(pool[self.rng.randint(len(pool))])
        return out

    def _insert(self, rollout, block: int) -> None:
        self._seen += 1
        if math.isinf(self.buffer_size) or len(self._buf) < self.buffer_size:
            self._buf.append(rollout)
            self._block_of.append(block)
            return
        # Reservoir eviction: keep a uniform sample of everything ever offered.
        j = self.rng.randint(self._seen)
        if j < len(self._buf):
            self._buf[j] = rollout
            self._block_of[j] = block

    # -- reporting --------------------------------------------------------

    def state_bytes(self) -> int:
        return sum(_batch_bytes(r) for r in self._buf)

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "buffer_size": ("inf" if math.isinf(self.buffer_size)
                            else int(self.buffer_size)),
            "replay_batches": self.replay_batches,
            "sampling": self.sampling,
            "buffer_items": len(self._buf),
            "stream_items_seen": self._seen,
        })
        return d


__all__ = ["ExperienceReplay"]
