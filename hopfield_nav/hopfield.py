"""Hopfield network for associative memory.

Stores patterns via Hebbian learning, recalls via iterative dynamics.
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

    def __init__(
        self,
        num_units: int,
        beta: float = 2.0,
        scale: float | None = None,
        zero_diag: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        self.num_units = num_units
        self.beta = beta
        self.scale = scale if scale is not None else 1.0 / num_units
        self.zero_diag = zero_diag
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.W = torch.zeros(num_units, num_units, device=self.device)
        self.num_memories: int = 0

    # ------------------------------------------------------------------
    # Memory storage
    # ------------------------------------------------------------------

    def input_memory(self, z: torch.Tensor, normalize: bool = True) -> None:
        """Store a single pattern via Hebbian update: W += scale * z z^T."""
        z = z.to(self.device).view(-1)
        if z.numel() != self.num_units:
            raise ValueError(f"Pattern size {z.numel()} != num_units {self.num_units}")
        if normalize:
            z = F.normalize(z, dim=0)
        self.W.addmm_(z.unsqueeze(1), z.unsqueeze(0), alpha=self.scale)
        if self.zero_diag:
            self.W.fill_diagonal_(0.0)
        self.num_memories += 1

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
