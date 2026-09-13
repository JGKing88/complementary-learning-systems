"""Hopfield network for associative memory.

Stores patterns via Hebbian learning, recalls via iterative dynamics.

Two storage rules, chosen at construction and fixed for the life of the
memory (``storage_rule``):

    hebb    W += scale * z z^T, diagonal zeroed.  One-shot, online, D^2
            synapses. A stored pattern is only NEAR a fixed point, because
            the patterns overlap (docs/THEORY_ENCODER_HOPFIELD.md).

    proj    the projection rule (Personnaz et al.), W = scale * P with P the
            orthogonal projector onto span(Z), so that W z_i = scale * z_i
            EXACTLY and every stored pattern is a true fixed point of the
            normalized recall, graded or binary. Built one pattern at a time
            with no history -- the projector onto span(z_1..z_n) is the
            projector onto span(z_1..z_{n-1}) plus the projector onto the
            residual:

                r = z - P z              the part of z the memory does not know
                P <- P + r r^T / <z, r>  and <z, r> = ||r||^2, P a projector

            A delta rule -- Hebbian on the error rather than on z -- needing one
            forward pass. Verified equal to the batch projector Z^T (Z Z^T)^+ Z
            and order-independent (analysis/hopfield_probe/graded_fixedpoint_check
            --incremental). The diagonal is NOT zeroed: that would destroy the
            identity the rule exists to provide. W is kept scaled by ``scale``
            so that ``beta * W x`` lands in the same range the Hebbian rule
            produces and recall's tanh stays in the regime everything was
            calibrated in; the scale cannot move the fixed point, since
            normalize(c P x) = normalize(P x) for any c > 0.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F


class Hopfield:
    """Continuous Hopfield network with sequential memory storage.

    Attributes:
        num_units: Dimension of patterns.
        W: Weight matrix (num_units, num_units).
        num_memories: Number of stored patterns.
    """

    STORAGE_RULES = ("hebb", "proj")

    def __init__(
        self,
        num_units: int,
        beta: float = 2.0,
        scale: float | None = None,
        zero_diag: bool | None = None,
        device: torch.device | str | None = None,
        storage_rule: str = "hebb",
        proj_eps: float = 1e-6,
    ) -> None:
        if storage_rule not in self.STORAGE_RULES:
            raise ValueError(
                f"storage_rule={storage_rule!r} not in {self.STORAGE_RULES}")
        # `zero_diag` defaults per rule: the Hebbian self-term is noise and is
        # removed; the projector's diagonal is load-bearing. An explicit True
        # under `proj` is refused rather than ignored, because the caller is
        # asking for something the rule cannot honour.
        if zero_diag is None:
            zero_diag = storage_rule == "hebb"
        elif zero_diag and storage_rule == "proj":
            raise ValueError(
                "zero_diag=True is incompatible with storage_rule='proj': the "
                "projector's diagonal is what makes a stored pattern a fixed "
                "point (W z = z). Pass zero_diag=False or omit it.")
        self.num_units = num_units
        self.beta = beta
        self.scale = scale if scale is not None else 1.0 / num_units
        self.zero_diag = zero_diag
        self.storage_rule = storage_rule
        # A pattern whose squared novelty <z, r> = ||r||^2 (unit z) is below
        # this is already in the span -- an exact fixed point -- and updating
        # on it would divide by ~0. It is counted as stored and W is left alone.
        self.proj_eps = float(proj_eps)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.W = torch.zeros(num_units, num_units, device=self.device)
        self.num_memories: int = 0

    # ------------------------------------------------------------------
    # Memory storage
    # ------------------------------------------------------------------

    def input_memory(self, z: torch.Tensor, normalize: bool = True) -> None:
        """Store a single pattern under this memory's ``storage_rule``.

        hebb:  W += scale * z z^T (diagonal zeroed if ``zero_diag``).
        proj:  one incremental step of the projection rule -- see the module
               docstring. Returns nothing either way; ``num_memories`` counts
               every call, including a `proj` pattern already in the span.
        """
        z = z.to(self.device).view(-1)
        if z.numel() != self.num_units:
            raise ValueError(f"Pattern size {z.numel()} != num_units {self.num_units}")
        if normalize:
            z = F.normalize(z, dim=0)
        if self.storage_rule == "proj":
            self._store_proj(z)
        else:
            self.W.addmm_(z.unsqueeze(1), z.unsqueeze(0), alpha=self.scale)
            if self.zero_diag:
                self.W.fill_diagonal_(0.0)
        self.num_memories += 1

    def _store_proj(self, z: torch.Tensor) -> None:
        """P <- P + r r^T / <z, r>, with r = z - P z and W = scale * P.

        The residual is formed against the UNSCALED projector: W holds
        scale * P, so P z = (W z) / scale. Float32 at D=1024 keeps the
        projector idempotent to ~1e-6 per pattern; the harness's batch
        `pinv` form is the reference if that ever needs checking.
        """
        Pz = (self.W @ z) / self.scale
        r = z - Pz
        d = float(torch.dot(z, r))          # = ||r||^2 up to round-off
        if d <= self.proj_eps:
            return                          # already in span: exact fixed point
        self.W.addmm_(r.unsqueeze(1), r.unsqueeze(0), alpha=self.scale / d)

    def projector(self) -> torch.Tensor:
        """The unscaled projector P = W / scale (meaningful under `proj`)."""
        return self.W / self.scale

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        x0: torch.Tensor,
        steps: int = 15,
        beta: float | None = None,
        alpha: float = 1.0,
        use_tanh: bool = True,
        normalize_each: bool = True,
    ) -> torch.Tensor:
        """Recall from cue.  x_{t+1} = (1-a)x + a*tanh(b*W@x), then normalize.

        Returns the final state (num_units,).
        """
        beta = beta if beta is not None else self.beta
        x = x0.to(self.device).view(-1).clone()
        for _ in range(steps):
            h = self.W @ x
            delta = torch.tanh(beta * h) if use_tanh else h
            x = (1 - alpha) * x + alpha * delta
            if normalize_each:
                x = F.normalize(x, dim=0)
        return x

    def recall_batch(
        self,
        x0_batch: torch.Tensor,
        steps: int = 15,
        beta: float | None = None,
        alpha: float = 1.0,
        use_tanh: bool = True,
        normalize_each: bool = True,
    ) -> torch.Tensor:
        """Batched recall when W is shared across all cues.

        x0_batch: (B, num_units).  Returns (B, num_units).
        Only valid when the same W applies to every sample (i.e., no per-sample stores).
        """
        beta = beta if beta is not None else self.beta
        X = x0_batch.to(self.device).clone()            # (B, D)
        for _ in range(steps):
            H = X @ self.W.T                             # (B, D)
            delta = torch.tanh(beta * H) if use_tanh else H
            X = (1 - alpha) * X + alpha * delta
            if normalize_each:
                X = F.normalize(X, dim=-1)
        return X

    def recall_batch_trajectory(
        self,
        x0_batch: torch.Tensor,
        snapshot_steps: list[int],
        beta: float | None = None,
        alpha: float = 1.0,
        use_tanh: bool = True,
        normalize_each: bool = True,
    ) -> dict[int, torch.Tensor]:
        """Like recall_batch but returns intermediate states at requested steps.

        Lets the policy see the recall-convergence trajectory: clean memory
        attractors converge in 1-2 steps; diffuse landscapes wander.
        """
        if not snapshot_steps:
            return {}
        beta = beta if beta is not None else self.beta
        X = x0_batch.to(self.device).clone()
        snapshot_set = set(snapshot_steps)
        max_step = max(snapshot_steps)
        out: dict[int, torch.Tensor] = {}
        for s in range(1, max_step + 1):
            H = X @ self.W.T
            delta = torch.tanh(beta * H) if use_tanh else H
            X = (1 - alpha) * X + alpha * delta
            if normalize_each:
                X = F.normalize(X, dim=-1)
            if s in snapshot_set:
                out[s] = X.clone()
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored memories."""
        self.W.zero_()
        self.num_memories = 0

    def clone(self) -> Hopfield:
        """Deep copy (independent W matrix)."""
        return copy.deepcopy(self)

    def energy(self, x: torch.Tensor) -> float:
        """Hopfield energy E = -0.5 x^T W x."""
        x = x.to(self.device).view(-1)
        return -0.5 * (x @ self.W @ x).item()


def recall_per_env_batch_trajectory(
    x0_batch: torch.Tensor,
    W_batch: torch.Tensor,
    snapshot_steps: list[int],
    beta: float = 2.0,
    alpha: float = 1.0,
    use_tanh: bool = True,
    normalize_each: bool = True,
) -> dict[int, torch.Tensor]:
    """Per-env-W version of recall_batch_trajectory."""
    if not snapshot_steps:
        return {}
    X = x0_batch.clone()
    snapshot_set = set(snapshot_steps)
    max_step = max(snapshot_steps)
    out: dict[int, torch.Tensor] = {}
    for s in range(1, max_step + 1):
        H = torch.bmm(W_batch, X.unsqueeze(-1)).squeeze(-1)
        delta = torch.tanh(beta * H) if use_tanh else H
        X = (1 - alpha) * X + alpha * delta
        if normalize_each:
            X = F.normalize(X, dim=-1)
        if s in snapshot_set:
            out[s] = X.clone()
    return out


def recall_per_env_batch(
    x0_batch: torch.Tensor,
    W_batch: torch.Tensor,
    steps: int = 1,
    beta: float = 2.0,
    alpha: float = 1.0,
    use_tanh: bool = True,
    normalize_each: bool = True,
) -> torch.Tensor:
    """Batched recall across B envs each with its own W matrix.

    Args:
        x0_batch: (B, D) cues.
        W_batch: (B, D, D) per-env weight matrices.

    Returns (B, D).
    """
    X = x0_batch.clone()                                   # (B, D)
    for _ in range(steps):
        # Per-env matmul: H[b] = W_batch[b] @ X[b]. Use bmm with X as (B, D, 1).
        H = torch.bmm(W_batch, X.unsqueeze(-1)).squeeze(-1)  # (B, D)
        delta = torch.tanh(beta * H) if use_tanh else H
        X = (1 - alpha) * X + alpha * delta
        if normalize_each:
            X = F.normalize(X, dim=-1)
    return X
