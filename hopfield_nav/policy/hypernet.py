"""A task-conditioned hypernetwork that generates the policy's weights.

The plan (docs/CONTINUAL_CONTROLS_PLAN.md section 4.3) names this the headline
competitor, for two reasons that both survive contact with this task.

The empirical one: Ehret et al. (ICLR 2021) benchmarked online EWC, SI,
masking, masking+SI, generative replay, coresets, multitask and from-scratch
across four *recurrent* continual-learning benchmarks, and hypernetworks beat
the weight-importance family consistently. Everything else in this suite was
selected on feedforward evidence; this policy is a GRU.

The structural one: it is the closest classical thing to what the Hopfield
agent does. Both keep a small addressable per-task code and recover behaviour
from it rather than overwriting one shared set of weights. The difference is
how the code gets written -- a gradient-descent inner loop over a whole block
here, against one Hebbian outer product there -- and that difference is
precisely what the cost frontier is meant to price.

Two pieces:

  `ChunkedHypernet`   embeddings -> a flat vector of target-network weights.
  `HyperRNNAgent`     an `RNNAgent`-shaped policy whose weights come from one.

The generated weights are pushed through a *template* `RNNAgent` with
`torch.func.functional_call`, so the forward pass is the baseline's own code
rather than a re-derivation of it. That matters more than it sounds: a
hand-written functional GRU would be a second implementation of the thing the
control is supposed to share with the baseline, and any drift between them
would show up as a method effect.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch.func import functional_call

from ..config import RNNAgentConfig
from .agent_rnn import RNNAgent, act_from_forward


class ChunkedHypernet(nn.Module):
    """`(task embedding, chunk embedding) -> a chunk of target weights`.

    Chunking is what keeps this from being absurd. The target network here has
    about 74k parameters; a hypernetwork mapping a 32-dim task embedding
    straight onto all of them needs a final layer of 74k x hidden, which is
    millions of parameters to generate thousands -- a "compression" scheme
    larger than the thing it compresses, and not a fair competitor on the
    parameter axis of the frontier.

    So the output is produced `chunk_dim` weights at a time by one small MLP,
    re-invoked with a different learned chunk embedding for each chunk (von
    Oswald et al., ICLR 2020). At the defaults the whole generator is roughly
    the size of the policy it generates, which is the point.

    `chunk_dim=0` selects the unchunked network instead -- one MLP straight to
    `out_dim`. Kept because it is the honest way to ask whether chunking costs
    anything here, and it is the variant that would be reached for if the
    target network were small.
    """

    def __init__(
        self,
        n_tasks: int,
        out_dim: int,
        *,
        emb_dim: int = 32,
        chunk_dim: int = 512,
        hidden: Sequence[int] = (100, 100),
        emb_init: float = 0.1,
        init_out_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if n_tasks < 1:
            raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        self.n_tasks = int(n_tasks)
        self.out_dim = int(out_dim)
        self.emb_dim = int(emb_dim)
        self.chunk_dim = int(chunk_dim)

        self.task_emb = nn.Parameter(torch.randn(n_tasks, emb_dim) * emb_init)

        if self.chunk_dim > 0:
            self.n_chunks = math.ceil(out_dim / self.chunk_dim)
            self.chunk_emb = nn.Parameter(
                torch.randn(self.n_chunks, emb_dim) * emb_init)
            in_dim, head_dim = 2 * emb_dim, self.chunk_dim
        else:
            self.n_chunks = 1
            # Registered so `state_dict` has the same keys either way; never
            # read in the unchunked path.
            self.register_parameter("chunk_emb", None)
            in_dim, head_dim = emb_dim, out_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, head_dim)

        # A small head at init means every task starts life generating nearly
        # the same weights. That is exactly what is wanted when a warm start
        # supplies those weights (see `HyperRNNAgent`): task 0 begins from the
        # pretrained policy rather than from a random one, which is the only
        # way this arm is comparable to the pretrained controls it is scored
        # against. The head is free to grow from there.
        if init_out_scale != 1.0:
            with torch.no_grad():
                self.head.weight.mul_(init_out_scale)
                self.head.bias.mul_(init_out_scale)

    def forward(self, task: int) -> torch.Tensor:
        """Flat `(out_dim,)` vector of generated weights for one task."""
        if not 0 <= task < self.n_tasks:
            raise IndexError(
                f"task {task} out of range for {self.n_tasks} task embeddings")
        e = self.task_emb[task]
        if self.chunk_dim > 0:
            z = torch.cat(
                [e.unsqueeze(0).expand(self.n_chunks, -1), self.chunk_emb],
                dim=-1)                                   # (n_chunks, 2*emb)
            w = self.head(self.body(z)).reshape(-1)       # (n_chunks*chunk_dim,)
            return w[: self.out_dim]                      # trailing chunk trimmed
        return self.head(self.body(e))


#: Where the warm start comes from, and what may move afterwards.
HNET_BASES: tuple[str, ...] = ("learned", "frozen", "none")


class HyperRNNAgent(nn.Module):
    """`RNNAgent`'s interface, with the weights generated per task.

        w_t = base + hypernet(task_embedding_t)

    and the policy is then the ordinary `RNNAgent` forward pass run with `w_t`.

    `base` is what makes this arm comparable to the rest of the suite. Every
    headline run in Waves 1 and 2 starts from the same pretrained checkpoint, so
    a hypernetwork starting from random weights would be scored against a
    different control and the comparison would be about pretraining rather than
    about hypernetworks. A chunked hypernetwork cannot be initialised to emit an
    arbitrary 74k-vector -- its output bias is `chunk_dim` wide and shared by
    every chunk -- so the checkpoint is held as an explicit base vector and the
    generator starts near zero on top of it. In the unchunked case this *is*
    the output bias, written down separately.

    Three settings, and they are genuinely different mechanisms:

      ``learned``  base is a free parameter, warm-started from the checkpoint.
                   Shared across tasks, so it can drift -- but the regulariser
                   sees the sum, so drift in the base is penalised exactly as
                   drift in the generator is.
      ``frozen``   base is fixed at the checkpoint forever; only the
                   task-conditioned part moves. Cannot forget through the base
                   at all, which makes it the interesting variant rather than
                   the conservative one.
      ``none``     no base. Pure von Oswald, random init, and the only setting
                   whose parameter count matches the baseline policy's.

    The task is selected by `set_task` and *persists*: the driver sets it at
    each block boundary and the evaluator sets it per env. An agent asked to act
    before any task is set raises rather than defaulting to task 0, because
    defaulting would silently evaluate every env under one task's weights and
    the result would look like catastrophic forgetting.
    """

    def __init__(
        self,
        cfg: RNNAgentConfig,
        input_dim: int,
        n_tasks: int,
        *,
        emb_dim: int = 32,
        chunk_dim: int = 512,
        hyper_hidden: Sequence[int] = (100, 100),
        base: str = "learned",
        init_out_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if base not in HNET_BASES:
            raise ValueError(
                f"unknown hnet base {base!r}; known: {list(HNET_BASES)}")
        if cfg.freeze_log_std:
            # log_std is one of the generated weights, so there is nothing here
            # for `requires_grad = False` to hold still. Refusing is the point:
            # a silently ignored freeze flag is a defect this project has
            # already paid for once.
            raise ValueError(
                "freeze_log_std is not meaningful for HyperRNNAgent: the "
                "movement log-std is generated by the hypernetwork like every "
                "other weight, so there is no leaf parameter to freeze. Drop "
                "the flag, or use --arch rnn.")
        self.cfg = cfg
        self.n_tasks = int(n_tasks)
        self.base_mode = base

        # The template supplies the forward pass and the parameter *shapes*;
        # its own values are never read, because `functional_call` replaces
        # every one of them. Held inside a list so `nn.Module.__setattr__` does
        # not register it -- its parameters must not appear in
        # `agent.parameters()`, or the optimiser would carry state for 74k
        # tensors that can never move and every parameter count in the
        # metadata would be double what it is.
        self._tmpl = [RNNAgent(cfg, input_dim)]
        self.spec = [(n, tuple(p.shape), p.numel())
                     for n, p in self._tmpl[0].named_parameters()]
        self.out_dim = sum(n for _, _, n in self.spec)

        flat = torch.cat([p.detach().reshape(-1)
                          for _, p in self._tmpl[0].named_parameters()])
        if base == "none":
            self.register_parameter("base", None)
        elif base == "learned":
            self.base = nn.Parameter(flat.clone())
        else:                                   # frozen
            self.register_buffer("base", flat.clone())

        self.hyper = ChunkedHypernet(
            n_tasks, self.out_dim, emb_dim=emb_dim, chunk_dim=chunk_dim,
            hidden=hyper_hidden,
            # With no base there is nothing to start from, so the generator has
            # to carry the whole init and must not be shrunk to nothing.
            init_out_scale=1.0 if base == "none" else init_out_scale,
        )

        self._task: int | None = None
        self._cache: dict | None = None
        self._cache_task: int | None = None

    # -- task selection ---------------------------------------------------

    def set_task(self, task: int) -> None:
        """Which task's weights to generate. Invalidates the weight cache."""
        if not 0 <= task < self.n_tasks:
            raise IndexError(
                f"task {task} out of range for {self.n_tasks} tasks")
        self._task = int(task)
        self._cache = None
        self._cache_task = None

    @property
    def task(self) -> int:
        if self._task is None:
            raise RuntimeError(
                "HyperRNNAgent has no active task: call set_task(i) before "
                "acting. Defaulting to 0 would evaluate every env under one "
                "task's weights, which is indistinguishable from forgetting.")
        return self._task

    # -- weight generation ------------------------------------------------

    def generate(self, task: int) -> torch.Tensor:
        """Flat weight vector for `task`, attached to the generator's graph."""
        w = self.hyper(task)
        if self.base is not None:
            w = w + self.base
        return w

    def unflatten(self, w: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flat vector -> the template's `{name: tensor}` parameter dict."""
        out: dict[str, torch.Tensor] = {}
        i = 0
        for name, shape, numel in self.spec:
            out[name] = w[i:i + numel].view(shape)
            i += numel
        return out

    def _params_for(self, task: int) -> dict[str, torch.Tensor]:
        """Generated parameters, cached whenever caching is provably safe.

        Rollout and evaluation call this once per environment step under
        `no_grad`, and regenerating 74k weights per step costs more than the
        policy forward it feeds. The cache is safe there because the parameters
        cannot have moved: an optimiser step is always preceded by a
        grad-enabled forward through this agent, and that path clears the cache
        before returning. `set_task` clears it too.
        """
        if torch.is_grad_enabled():
            self._cache = None
            self._cache_task = None
            return self.unflatten(self.generate(task))
        if self._cache is None or self._cache_task != task:
            self._cache = self.unflatten(self.generate(task))
            self._cache_task = task
        return self._cache

    # -- policy -----------------------------------------------------------

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple:
        """x: (B, T, input_dim), h: (num_layers, B, hidden) or None.

        Returns (move_dist, h_next), exactly as `RNNAgent.forward` does.
        """
        params = self._params_for(self.task)
        return functional_call(self._tmpl[0], params, (x, h))

    @torch.no_grad()
    def act(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> dict:
        return act_from_forward(self, x, h, deterministic)

    # -- housekeeping -----------------------------------------------------

    def _apply(self, *args, **kwargs):
        """Keep the unregistered template on the same device/dtype as the rest.

        `nn.Module._apply` is what `.to()`, `.float()` and `.cuda()` all route
        through, and it only visits registered children. The template is
        deliberately not one, so it would otherwise be left on the CPU while
        the generated weights moved -- which `functional_call` would not
        complain about until the first matmul.
        """
        out = super()._apply(*args, **kwargs)
        self._tmpl[0]._apply(*args, **kwargs)
        self._cache = None
        self._cache_task = None
        return out

    def warm_start_from(self, state_dict: dict) -> None:
        """Point the base at a pretrained `RNNAgent`'s weights.

        The generator's head was initialised small, so immediately after this
        every task generates approximately the checkpoint -- which is where the
        single-head pretrained controls start, and therefore the only place
        this arm can start if the two are to be compared.

        With ``base="none"`` there is nowhere to put them, and quietly ignoring
        a checkpoint would produce a run labelled "pretrained" that was not.
        """
        if self.base is None:
            raise ValueError(
                "hnet base='none' has no base vector to warm-start; it is the "
                "from-scratch variant. Use base='learned' or 'frozen' with a "
                "checkpoint, or drop --load_checkpoint.")
        missing = [n for n, _, _ in self.spec if n not in state_dict]
        if missing:
            raise KeyError(
                f"checkpoint is missing {len(missing)} target parameters, "
                f"e.g. {missing[:3]}; it does not describe this policy.")
        flat = torch.cat([state_dict[n].detach().reshape(-1).to(
            self.base.dtype) for n, _, _ in self.spec])
        with torch.no_grad():
            self.base.copy_(flat)
        self._cache = None
        self._cache_task = None

    def generator_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Everything the output regulariser has to pin: the generator itself.

        The base counts when it is learned, because the regularised quantity is
        the *sum*. A frozen base cannot move, and a snapshot of it would be
        constant overhead that nothing reads.
        """
        named = [(f"hyper.{n}", p) for n, p in self.hyper.named_parameters()]
        if self.base_mode == "learned":
            named.append(("base", self.base))
        return named

    @torch.no_grad()
    def task_divergence(self) -> dict:
        """How task-dependent the generated weights currently are.

        The quietest failure this architecture has. The generator's output
        layer starts small so that every task begins at the warm-started base
        -- which is what makes the arm comparable to the pretrained controls.
        If that output never grows, every task gets approximately the same 73k
        weights, every environment shares one policy, and the arm is the naive
        baseline wearing a hypernetwork's metadata. It would produce a
        completely ordinary run: sensible losses, a penalty that scales with
        beta, low retention, and the plausible, wrong conclusion that the
        method does not help here.

        No test of the mechanics catches it, because the mechanics would all be
        correct; it is a question about where optimisation ended up. So the
        answer is recorded beside every run.

        `pairwise` is the one that matters: mean over task pairs of
        ||w_i - w_j|| / ||w_i||, which is nonzero only if different tasks
        genuinely get different weights.
        """
        ws = [self.generate(t) for t in range(self.n_tasks)]
        pairs = [(i, j) for i in range(self.n_tasks)
                 for j in range(i + 1, self.n_tasks)]
        pairwise = (sum(float((ws[i] - ws[j]).norm() / ws[i].norm())
                        for i, j in pairs) / len(pairs)) if pairs else 0.0
        hyper_norm = sum(float(self.hyper(t).norm())
                         for t in range(self.n_tasks)) / self.n_tasks
        out = {"pairwise_divergence": pairwise, "hyper_norm": hyper_norm}
        if self.base is not None:
            out["base_norm"] = float(self.base.norm())
            out["conditioned_frac"] = hyper_norm / max(out["base_norm"], 1e-12)
        return out

    def describe(self) -> dict:
        """Parameter accounting, for the metadata and the frontier table."""
        n_hyper = sum(p.numel() for _, p in self.hyper.named_parameters())
        n_base = self.base.numel() if self.base is not None else 0
        return {
            **self.task_divergence(),
            "arch": "hnet",
            "n_tasks": self.n_tasks,
            "base": self.base_mode,
            "target_params": self.out_dim,
            "hyper_params": n_hyper,
            "base_params": n_base,
            "trainable_params": sum(p.numel() for p in self.parameters()
                                    if p.requires_grad),
            "emb_dim": self.hyper.emb_dim,
            "chunk_dim": self.hyper.chunk_dim,
            "n_chunks": self.hyper.n_chunks,
        }


__all__ = ["ChunkedHypernet", "HyperRNNAgent", "HNET_BASES"]
