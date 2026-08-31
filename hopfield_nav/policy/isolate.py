"""Parameter-isolation policies: a head per task, and gated units per task.

The third family in the plan (section 4.3). Where replay changes *what the
update is trained on* and regularisation changes *how far parameters may
move*, isolation changes *which parameters a task uses at all*. Both policies
here need to be told which task they are in, at training time and at
evaluation time, which is a real cost and is recorded as one -- the Hopfield
store needs neither.

  `MultiHeadRNNAgent`  one shared recurrent trunk, one movement head per task.
  `XdGRNNAgent`        one shared everything, with a fixed random subset of
                       hidden units active per task (Masse et al., PNAS 2018).

Neither is expected to win outright, and that is why they are worth running.
Multi-head bounds the family from one direction: its heads cannot interfere at
all, so whatever it fails to retain is forgetting in the shared trunk, and no
amount of head isolation will fix it. XdG bounds it from the other: it isolates
inside the trunk, at the price of giving each task a fraction of the units.
Between them they say how much of this problem is even addressable by allocating
parameters, which is the question the plan asks of the whole family.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from ..config import RNNAgentConfig
from .agent_rnn import RNNAgent, act_from_forward
from .recurrent import build_recurrent_core


class _MovementHead(nn.Module):
    """The output half of `RNNAgent`, on its own so there can be several.

    Deliberately mirrors `RNNAgent.__init__`'s head construction rather than
    generalising it: the parameter *names* have to stay `movement_head` /
    `movement_mean` / `movement_log_std` so a pretrained checkpoint's head
    tensors can be copied in by name, and so `HEAD_PREFIXES` in the driver keeps
    meaning what it says.
    """

    def __init__(self, cfg: RNNAgentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.movement_mode == "discrete":
            self.movement_head = nn.Linear(cfg.hidden_size, 4)
        else:
            self.movement_mean = nn.Linear(cfg.hidden_size, 2)
            self.movement_log_std = nn.Parameter(
                torch.full((2,), cfg.init_log_std))
            if cfg.freeze_log_std:
                self.movement_log_std.requires_grad = False

    def forward(self, features: torch.Tensor):
        if self.cfg.movement_mode == "discrete":
            return Categorical(logits=self.movement_head(features))
        mean = self.movement_mean(features)
        return Normal(mean, self.movement_log_std.exp().expand_as(mean))


def _head_tensors(state_dict: dict) -> dict:
    """The movement-head entries of an `RNNAgent` checkpoint.

    Raises rather than returning an empty dict, because an empty head loads
    without complaint into a `_MovementHead` with `strict=False` semantics
    nowhere in sight -- and a warm start that silently skipped the head would
    leave the policy at its random init while the run reported itself as
    pretrained.
    """
    head = {k: v for k, v in state_dict.items()
            if k.startswith(("movement_head", "movement_mean",
                             "movement_log_std"))}
    if not head:
        raise KeyError(
            "checkpoint has no movement head to copy; keys look like "
            f"{sorted(state_dict)[:6]}")
    return head


class _TaskConditioned(nn.Module):
    """Shared task bookkeeping: `set_task`, and the refusal to guess.

    Both policies below are useless without a task id and dangerous with the
    wrong one -- evaluating every env under task 0's head produces a curve that
    looks exactly like catastrophic forgetting, and nothing downstream could
    tell the difference. So there is no default: an agent asked to act before
    `set_task` raises.
    """

    def __init__(self, cfg: RNNAgentConfig, n_tasks: int) -> None:
        super().__init__()
        if n_tasks < 1:
            raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
        self.cfg = cfg
        self.n_tasks = int(n_tasks)
        self._task: int | None = None

    def set_task(self, task: int) -> None:
        if not 0 <= task < self.n_tasks:
            raise IndexError(
                f"task {task} out of range for {self.n_tasks} tasks")
        self._task = int(task)

    @property
    def task(self) -> int:
        if self._task is None:
            raise RuntimeError(
                f"{type(self).__name__} has no active task: call set_task(i) "
                "before acting. Defaulting to 0 would evaluate every env under "
                "one task's parameters, which is indistinguishable from "
                "forgetting.")
        return self._task

    @torch.no_grad()
    def act(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> dict:
        return act_from_forward(self, x, h, deterministic)


class MultiHeadRNNAgent(_TaskConditioned):
    """Shared GRU trunk, one movement head per task, selected by task id.

    The cheapest possible member of the isolation family, and the informative
    one precisely because it is partial. The heads are perfectly protected --
    task j's head receives no gradient while task i is training -- so any
    retention this fails to deliver is forgetting in the trunk, which every
    richer isolation scheme also has to solve. If a free task id plus perfect
    head isolation does not retain here, head-level isolation is not the answer
    and the family's remaining hope is isolation *inside* the trunk.

    The heads are tiny against the trunk: 258 parameters each against 73,728,
    so five tasks cost 2% more parameters than the baseline.
    """

    def __init__(self, cfg: RNNAgentConfig, input_dim: int,
                 n_tasks: int) -> None:
        super().__init__(cfg, n_tasks)
        self.rnn = build_recurrent_core(cfg, input_dim)
        self.heads = nn.ModuleList([_MovementHead(cfg) for _ in range(n_tasks)])

    #: For `freeze_trunk_params`, which otherwise looks for `movement_*` at the
    #: top level and would freeze every parameter this agent has.
    head_prefixes = ("heads.",)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple:
        features, h_next = self.rnn(x, h)
        return self.heads[self.task](features), h_next

    def warm_start_from(self, state_dict: dict) -> None:
        """Load an `RNNAgent` checkpoint: trunk as-is, head copied to all tasks.

        Every task therefore begins from the same pretrained policy, which is
        what the single-head controls do and the only way the arms compare.
        """
        # Head first: a checkpoint that is not an `RNNAgent`'s fails both
        # loads, and "no movement head" names the actual problem where a GRU
        # shape mismatch only describes a symptom of it.
        head = _head_tensors(state_dict)
        trunk = {k[len("rnn."):]: v for k, v in state_dict.items()
                 if k.startswith("rnn.")}
        self.rnn.load_state_dict(trunk)
        for h in self.heads:
            h.load_state_dict(head)

    def describe(self) -> dict:
        n_head = sum(p.numel() for p in self.heads[0].parameters())
        return {
            "arch": "multihead",
            "n_tasks": self.n_tasks,
            "trunk_params": sum(p.numel() for p in self.rnn.parameters()),
            "head_params_each": n_head,
            "trainable_params": sum(p.numel() for p in self.parameters()
                                    if p.requires_grad),
        }


class XdGRNNAgent(_TaskConditioned):
    """Context-dependent gating: a fixed random subset of units per task.

    Masse et al. (PNAS 2018). Each task draws one binary mask over the hidden
    units, fixed for the run, with `1 - gating` of them active; the mask is
    applied to the recurrent state at *every* timestep, so a task's units are
    the only ones that carry its state and the only ones its head reads. Masks
    are drawn independently per task, so they overlap by chance rather than
    being carved disjointly -- that is the published scheme, and the overlap is
    what makes it a soft isolation rather than a hard partition.

    Applying the mask inside the recurrence is what makes this XdG rather than
    a masked readout, and it costs a Python loop over timesteps, because a
    cuDNN GRU cannot be interrupted between steps. The cost lands only on the
    BC update: rollout and evaluation already step the recurrence one
    observation at a time, so their loop is the same length it always was.

    One caveat that belongs next to the numbers rather than in a footnote: the
    pretrained checkpoint was trained with every unit available, so gating away
    80% of them hands this arm a broken policy at step 0, where the single-head
    controls start from a working one. That is inherent to combining XdG with a
    warm start rather than a flaw in either, and it is why the gating fraction
    is swept from mild to severe instead of being set at the paper's value.
    """

    def __init__(self, cfg: RNNAgentConfig, input_dim: int, n_tasks: int,
                 *, gating: float = 0.8, seed: int = 0) -> None:
        super().__init__(cfg, n_tasks)
        if not 0.0 <= gating < 1.0:
            raise ValueError(
                f"gating is the fraction of units held OFF and must be in "
                f"[0, 1), got {gating}")
        self.gating = float(gating)
        self.rnn = build_recurrent_core(cfg, input_dim)
        self.head = _MovementHead(cfg)

        # A dedicated generator, so the masks are a function of `seed` alone.
        # Drawing from the global RNG would make them depend on how many draws
        # the env construction happened to make first, and two runs that differ
        # only in an unrelated flag would get different masks.
        g = torch.Generator().manual_seed(int(seed))
        h = cfg.hidden_size
        n_active = max(1, int(round((1.0 - self.gating) * h)))
        masks = torch.zeros(n_tasks, h)
        for t in range(n_tasks):
            idx = torch.randperm(h, generator=g)[:n_active]
            masks[t, idx] = 1.0
        self.register_buffer("masks", masks)
        self.n_active = n_active

    head_prefixes = ("head.",)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple:
        if x.dim() != 3:
            raise ValueError(
                f"XdGRNNAgent expects (B, T, input_dim), got {tuple(x.shape)}")
        mask = self.masks[self.task].view(1, 1, -1)     # broadcast over (L, B, H)
        B, T, _ = x.shape
        if h is None:
            h = x.new_zeros(self.rnn.num_layers, B, self.rnn.hidden_size)
        h = h * mask

        feats = []
        for t in range(T):
            _, h = self.rnn(x[:, t:t + 1], h)
            h = h * mask
            feats.append(h[-1])                         # top layer's state
        features = torch.stack(feats, dim=1)            # (B, T, hidden)
        return self.head(features), h

    def warm_start_from(self, state_dict: dict) -> None:
        """Load an `RNNAgent` checkpoint straight in; the mask does the rest."""
        head = _head_tensors(state_dict)
        trunk = {k[len("rnn."):]: v for k, v in state_dict.items()
                 if k.startswith("rnn.")}
        self.rnn.load_state_dict(trunk)
        self.head.load_state_dict(head)

    def describe(self) -> dict:
        overlap = float((self.masks @ self.masks.t()).fill_diagonal_(0).sum()
                        / max(1, self.n_tasks * (self.n_tasks - 1))
                        ) if self.n_tasks > 1 else 0.0
        return {
            "arch": "xdg",
            "n_tasks": self.n_tasks,
            "gating": self.gating,
            "units_active_per_task": self.n_active,
            "hidden_size": self.cfg.hidden_size,
            "mean_pairwise_overlap_units": overlap,
            "trainable_params": sum(p.numel() for p in self.parameters()
                                    if p.requires_grad),
        }


def warm_start(agent, state_dict: dict) -> None:
    """Load an `RNNAgent` checkpoint into whichever policy this is.

    `RNNAgent` takes it directly; the task-conditioned policies have to fan the
    head out or route around a hypernetwork, so they carry their own method.
    One function so the driver has one call site and cannot forget a case.
    """
    fn = getattr(agent, "warm_start_from", None)
    if fn is not None:
        fn(state_dict)
        return
    if isinstance(agent, RNNAgent):
        agent.load_state_dict(state_dict)
        return
    raise TypeError(
        f"{type(agent).__name__} has no warm_start_from and is not an "
        "RNNAgent; a checkpoint cannot be loaded into it.")


__all__ = ["MultiHeadRNNAgent", "XdGRNNAgent", "warm_start"]
