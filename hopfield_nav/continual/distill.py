"""Functional regularisers: anchor what the model *does*, not where it sits.

EWC and SI constrain parameters. These constrain outputs, which is a different
and often better-behaved thing: two very different parameter vectors can
implement the same policy, and a parameter-space penalty spends its budget
forbidding that, while an output-space penalty does not care.

    LwF     distil the model as it was at the START of this block, evaluated on
            THIS block's states. No buffer at all.
    CLEAR   distil the model as it was at the END of the previous block,
            evaluated on REPLAYED states. Needs a buffer; ER is its base.

The distinction between them is where the states come from, and it is the
interesting axis. LwF asks "do not change your behaviour on the data you are
looking at now", which costs no memory and is the cleanest measurement of what
pure functional regularisation buys at zero storage. CLEAR asks "do not change
your behaviour on the data you used to see", which needs the data but targets
the states that actually matter for retention.

Both use the closed-form KL between the old and new action distributions --
diagonal Gaussians in `continuous` mode, categoricals in `discrete` -- via
`torch.distributions.kl_divergence`, so neither needs a mode-specific branch.

**Why CLEAR is here rather than filed under continual RL.** Rolnick et al. 2019
framed it that way, but strip V-trace and the value-cloning term (there is no
value head on `RNNAgent`) and what remains is replay plus a supervised
distillation loss. It earns its place because the replay data here is
*self-generated*: DAgger collects under the student's own policy, so a replayed
state came from a policy that no longer exists, and anchoring to the old
policy's outputs rather than only to the oracle's labels is the standard fix.
"""
from __future__ import annotations

import copy

import torch

from .cost import COUNTER
from torch.distributions import kl_divergence

from .base import ContinualMethod
from .replay import ExperienceReplay


def _frozen_copy(agent):
    """An eval-mode, gradient-free snapshot of the policy."""
    snap = copy.deepcopy(agent)
    snap.eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    return snap


def _masked_kl(old_dist, new_dist, mask: torch.Tensor,
               movement_mode: str) -> torch.Tensor:
    """KL(old || new), summed over action dims, averaged over supervised steps."""
    kl = kl_divergence(old_dist, new_dist)
    if movement_mode == "continuous":
        kl = kl.sum(-1)                       # (B, T) after summing the 2 axes
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


class LwF(ContinualMethod):
    """Learning without Forgetting (Li & Hoiem 2016). No buffer.

    Snapshots the policy at the start of each block and penalises divergence
    from it on the *current* block's states. Repeatedly reported as a strong
    domain-incremental baseline despite its simplicity, and it is the only
    method in the suite that stores no data and no per-parameter state at all --
    which makes it the cheapest point on the memory axis by a wide margin.
    """

    name = "lwf"
    needs_task_boundaries = True     # to know when to re-snapshot
    needs_task_id = False

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self._old = None
        self._cache: dict[int, object] = {}

    def on_block_start(self, block: int, agent, envs) -> None:
        # From block 1 onward. At block 0 there is nothing to preserve, and
        # snapshotting there would only pin the policy to its initialisation.
        self._old = _frozen_copy(agent) if block > 0 else None
        self._cache = {}

    def aux_loss(self, agent, rollout, extra: list) -> torch.Tensor | None:
        if self._old is None or self.alpha == 0.0:
            return None
        key = id(rollout)
        if key not in self._cache:
            # The frozen model's outputs do not change within an update, but
            # `aux_loss` is called once per minibatch step -- so compute once.
            with torch.no_grad():
                COUNTER.add(rollout.obs, backward=False)
                self._cache = {key: self._old(rollout.obs)[0]}
        old_dist = self._cache[key]
        COUNTER.add(rollout.obs, backward=True)
        new_dist, _ = agent(rollout.obs)
        kl = _masked_kl(old_dist, new_dist, rollout.move_label_mask,
                        agent.cfg.movement_mode)
        return self.alpha * kl

    def after_update(self, rollout, block: int, agent) -> None:
        self._cache = {}

    def state_bytes(self) -> int:
        if self._old is None:
            return 0
        return sum(p.numel() * p.element_size() for p in self._old.parameters())

    def describe(self) -> dict:
        d = super().describe()
        d.update({"alpha": self.alpha})
        return d


class CLEAR(ExperienceReplay):
    """Rolnick et al. 2019, minus the parts that need an RL algorithm.

    Experience replay at a 50:50 new:replay ratio, plus behaviour cloning to
    the past self on replayed states. Rolnick report ~0.01 for the policy-
    cloning coefficient and note performance is insensitive to the exact ratio;
    both are exposed rather than hardcoded.

    The `needs_task_boundaries = False` here is real and load-bearing for the
    results table: CLEAR's snapshot can be taken on a fixed schedule, and the
    method never uses task identity. It is set at block ends only because this
    protocol happens to have them.
    """

    name = "clear"
    needs_task_boundaries = False
    needs_task_id = False

    def __init__(
        self,
        buffer_size: float = float("inf"),
        replay_batches: int = 1,
        sampling: str = "balanced",
        seed: int = 0,
        clone_coef: float = 0.01,
    ) -> None:
        super().__init__(buffer_size=buffer_size, replay_batches=replay_batches,
                         sampling=sampling, seed=seed)
        self.clone_coef = float(clone_coef)
        self._past = None
        self._cache: dict[int, object] = {}

    def on_block_end(self, block: int, agent, envs) -> None:
        self._past = _frozen_copy(agent)

    def aux_loss(self, agent, rollout, extra: list) -> torch.Tensor | None:
        if self._past is None or not extra or self.clone_coef == 0.0:
            return None
        total = None
        for r in extra:
            key = id(r)
            if key not in self._cache:
                with torch.no_grad():
                    COUNTER.add(r.obs, backward=False)
                    self._cache[key] = self._past(r.obs)[0]
            COUNTER.add(r.obs, backward=True)
            new_dist, _ = agent(r.obs)
            kl = _masked_kl(self._cache[key], new_dist, r.move_label_mask,
                            agent.cfg.movement_mode)
            total = kl if total is None else total + kl
        if total is None:
            return None
        return self.clone_coef * total / max(1, len(extra))

    def after_update(self, rollout, block: int, agent) -> None:
        super().after_update(rollout, block, agent)
        self._cache = {}

    def state_bytes(self) -> int:
        b = super().state_bytes()
        if self._past is not None:
            b += sum(p.numel() * p.element_size() for p in self._past.parameters())
        return b

    def describe(self) -> dict:
        d = super().describe()
        d.update({"clone_coef": self.clone_coef})
        return d


class DERpp(ExperienceReplay):
    """Dark Experience Replay++ (Buzzega et al. 2020).

    Replay with two terms: the oracle label (the usual BC loss, which the base
    class already contributes by handing the trajectories to the update) and an
    MSE against the model's *own output at the moment the sample was stored*.

    The instructive contrast with CLEAR is *when* the anchor is taken. DER++
    freezes a target the instant a trajectory enters the buffer, so different
    buffer entries are anchored to different, older versions of the policy --
    a spread of the optimisation trajectory rather than a single snapshot.
    CLEAR anchors everything to one converged past self. Both are boundary-free;
    they disagree about what "the past" means.

    Anchors are stored as distribution *parameters* rather than as a frozen
    network, which is why this method's state is the buffer plus a couple of
    small tensors per entry instead of a model copy.
    """

    name = "derpp"
    needs_task_boundaries = False
    needs_task_id = False

    def __init__(
        self,
        buffer_size: float = float("inf"),
        replay_batches: int = 1,
        sampling: str = "balanced",
        seed: int = 0,
        alpha: float = 0.5,
    ) -> None:
        super().__init__(buffer_size=buffer_size, replay_batches=replay_batches,
                         sampling=sampling, seed=seed)
        self.alpha = float(alpha)
        self._targets: list = []          # parallel to self._buf
        self._last_idx: list[int] = []

    # -- buffer, kept in step with the targets ----------------------------

    def extra_batches(self, rollout, block: int) -> list:
        if not self._buf or self.replay_batches == 0:
            return []
        self._last_idx = self._sample(self.replay_batches)
        return [self._buf[i] for i in self._last_idx]

    def after_update(self, rollout, block: int, agent) -> None:
        with torch.no_grad():
            dist, _ = agent(rollout.obs)
            target = self._stored_params(dist, agent.cfg.movement_mode)
        n_before = len(self._buf)
        super().after_update(rollout, block, agent)
        if len(self._buf) > n_before:
            self._targets.append(target)
        else:
            # Reservoir replaced an entry; find which one and keep the targets
            # aligned. `_insert` overwrote exactly the slot whose object is now
            # this rollout.
            for i, r in enumerate(self._buf):
                if r is rollout:
                    self._targets[i] = target
                    break

    @staticmethod
    def _live_params(dist, movement_mode: str):
        """The current output, still attached to the graph.

        Kept separate from `_stored_params` on purpose. A single helper that
        detached was used for both roles, which is correct for the target and
        silently fatal for the prediction: `aux_loss` then returned a nonzero
        number with `requires_grad=False`, so DER++ added a constant to the loss
        and contributed no gradient at all. It ran as plain ER, which is exactly
        what the results showed -- bit-identical across alpha spanning four
        orders of magnitude.
        """
        if movement_mode == "discrete":
            return dist.logits
        return (dist.mean, dist.stddev)

    @staticmethod
    def _stored_params(dist, movement_mode: str):
        """The anchor, frozen at insertion time. Detached by design."""
        if movement_mode == "discrete":
            return dist.logits.detach().clone()
        return (dist.mean.detach().clone(), dist.stddev.detach().clone())

    def aux_loss(self, agent, rollout, extra: list) -> torch.Tensor | None:
        if not extra or not self._last_idx or self.alpha == 0.0:
            return None
        movement_mode = agent.cfg.movement_mode
        total = None
        for r, i in zip(extra, self._last_idx):
            if i >= len(self._targets):
                continue
            COUNTER.add(r.obs, backward=True)
            dist, _ = agent(r.obs)
            cur = self._live_params(dist, movement_mode)
            tgt = self._targets[i]
            mask = r.move_label_mask.unsqueeze(-1)
            if movement_mode == "discrete":
                se = ((cur - tgt) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                se = (((cur[0] - tgt[0]) ** 2 + (cur[1] - tgt[1]) ** 2) * mask
                      ).sum() / mask.sum().clamp_min(1.0)
            total = se if total is None else total + se
        if total is None:
            return None
        return self.alpha * total / max(1, len(extra))

    def state_bytes(self) -> int:
        b = super().state_bytes()
        for t in self._targets:
            if isinstance(t, tuple):
                b += sum(x.numel() * x.element_size() for x in t)
            else:
                b += t.numel() * t.element_size()
        return b

    def describe(self) -> dict:
        d = super().describe()
        d.update({"alpha": self.alpha})
        return d


__all__ = ["LwF", "CLEAR", "DERpp"]
