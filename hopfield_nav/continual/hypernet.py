"""The hypernetwork output regulariser (von Oswald et al., ICLR 2020).

The mechanism that makes a task-conditioned hypernetwork a continual-learning
method rather than just an unusual parameterisation. Without it, training on
task t moves the generator freely and the weights it produces for tasks
j < t move with it; the architecture alone protects nothing.

The regulariser fixes what the generator *outputs* for earlier tasks rather
than where its parameters sit:

    L_reg  =  beta / (t)  *  sum_{j < t}  || generate(e_j; theta) - w_j* ||^2

with `w_j*` the weights the generator produced for task j at the moment task
t began. This is the interesting part of the method and the reason it belongs
in this suite rather than beside EWC: EWC and SI both ask parameters to stay
put, which is a proxy for behaviour staying put and a poor one in a recurrent
network where a small weight change compounds over 200 timesteps. This asks
the *behaviour-determining weights themselves* to stay put, exactly, and lets
the generator's own parameters go wherever they need to.

Two consequences worth stating before the numbers arrive. It needs task
boundaries, to know when to re-snapshot. And it needs the task id at
evaluation time, because there is no policy at all until an embedding selects
one -- so it is scored as an upper bound on its family rather than as a peer
of the boundary-free methods. The Hopfield store needs neither.
"""
from __future__ import annotations

import copy

import torch

from .base import ContinualMethod


class HypernetOutputReg(ContinualMethod):
    """Pin the generator's output on every past task to what it used to be.

    `beta` is the only strength knob and is swept over decades, for the reason
    that cost this suite two waves already: a coefficient taken from a paper is
    calibrated to that paper's loss scale. The BC loss here is a Gaussian
    negative log-likelihood of order 10, not a cross-entropy of order 1, so a
    published beta means nothing until it is placed against this objective.
    `losses["penalty"]` carries the term's actual value per update, so the
    comparison can be made from the histories rather than assumed.

    `normalize` chooses the norm. The literal form in the paper sums squared
    differences over every generated weight; across 74k of them that makes the
    penalty's natural scale depend on the size of the target network, so the
    default here is the per-element mean and beta is scale-free. `False`
    recovers the paper's form exactly, for anyone comparing to it.
    """

    name = "hnet"
    needs_task_boundaries = True
    needs_task_id = True

    def __init__(self, beta: float = 1.0, normalize: bool = True) -> None:
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        self.beta = float(beta)
        self.normalize = bool(normalize)
        # A copy of the generator as it stood when the current block began.
        # Fixed size: this is the whole memory cost of the method, and the
        # reason it is on the frontier at all.
        self._snapshot = None
        # Targets for the current block, recomputed from `_snapshot` at each
        # boundary. A cache -- a deterministic function of the snapshot and the
        # task embeddings, holding nothing the snapshot does not already
        # determine -- so it is not counted in `state_bytes`.
        self._targets: dict[int, torch.Tensor] = {}

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _require_hypernet(agent):
        gen = getattr(agent, "generate", None)
        if gen is None or not hasattr(agent, "hyper"):
            raise TypeError(
                f"method 'hnet' regularises a hypernetwork's output, but the "
                f"agent is a {type(agent).__name__} with no generator. Pass "
                "--arch hnet, or pick a method that applies to a plain policy.")
        return agent

    def _snapshot_generate(self, task: int) -> torch.Tensor:
        """`generate(task)` under the snapshot's parameters, detached."""
        hyper, base, _ = self._snapshot
        with torch.no_grad():
            w = hyper(task)
            if base is not None:
                w = w + base
        return w

    # -- hooks ------------------------------------------------------------

    def on_block_start(self, block: int, agent, envs) -> None:
        """Recompute the targets for every task already seen.

        Nothing has trained since `on_block_end` took the snapshot, so these are
        the weights the agent would produce right now for tasks 0..block-1 --
        which is the definition of what must not change.
        """
        self._require_hypernet(agent)
        self._targets = {}
        if block == 0 or self._snapshot is None:
            return
        for j in range(block):
            self._targets[j] = self._snapshot_generate(j)

    def penalty(self, agent) -> torch.Tensor | None:
        if self.beta == 0.0 or not self._targets:
            return None
        total = None
        for j, target in self._targets.items():
            d = agent.generate(j) - target
            term = d.pow(2).mean() if self.normalize else d.pow(2).sum()
            total = term if total is None else total + term
        return self.beta * total / len(self._targets)

    def on_block_end(self, block: int, agent, envs) -> None:
        """Freeze a copy of the generator, so the next block has a reference."""
        self._require_hypernet(agent)
        hyper = copy.deepcopy(agent.hyper).eval()
        for p in hyper.parameters():
            p.requires_grad_(False)
        base = getattr(agent, "base", None)
        # A frozen base is a buffer that never moves, so the live tensor is
        # always equal to what a copy would hold. Borrow it for the arithmetic
        # and record that it is not ours -- `state_bytes` must not charge the
        # method for storage it does not need, or a frozen-base arm would be
        # placed on the frontier at twice its true memory cost.
        owned = base is not None and getattr(agent, "base_mode", None) == "learned"
        if owned:
            base = base.detach().clone()
        self._snapshot = (hyper, base, owned)

    # -- reporting --------------------------------------------------------

    def state_bytes(self) -> int:
        """The generator snapshot. Fixed size, independent of task count.

        The per-task targets held during a block are recomputed from this at
        every boundary and are a cache of it, not additional state -- so they
        are deliberately not counted. Storing them instead would be the same
        method with a memory cost that grows in the number of tasks, which is
        the cost this design exists to avoid.
        """
        if self._snapshot is None:
            return 0
        hyper, base, owned = self._snapshot
        n = sum(p.numel() * p.element_size() for p in hyper.parameters())
        if owned:
            n += base.numel() * base.element_size()
        return int(n)

    def describe(self) -> dict:
        d = super().describe()
        d.update({"beta": self.beta, "normalize": self.normalize,
                  "n_targets": len(self._targets)})
        return d


__all__ = ["HypernetOutputReg"]
