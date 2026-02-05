"""Hopfield network for associative memory.

This module provides a modern continuous Hopfield network that can:
- Sequentially store memories via Hebbian learning
- Recall patterns from partial/noisy cues
"""
from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F

class Hopfield:
    """Continuous Hopfield network with sequential memory storage.
    
    Stores patterns via outer-product Hebbian learning and recalls
    via iterative dynamics (synchronous or asynchronous).
    
    Attributes:
        num_units: Dimension of patterns
        W: Weight matrix (num_units, num_units)
        num_memories: Number of stored patterns
    """

    def __init__(
        self,
        num_units: int,
        beta: float = 2.0,
        scale: Optional[float] = None,
        zero_diag: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize Hopfield network.
        
        Args:
            num_units: Dimension of pattern vectors
            beta: Inverse temperature for tanh nonlinearity
            scale: Scaling factor for weights (default: 1/num_units)
            zero_diag: Whether to zero the diagonal of W
            device: Torch device for tensors
        """
        self.num_units = num_units
        self.beta = beta
        self.scale = scale if scale is not None else 1.0 / num_units
        self.zero_diag = zero_diag
        self.device = device or torch.device("cpu")
        
        # Initialize empty weight matrix
        self.W = torch.zeros(num_units, num_units, device=self.device)
        self.num_memories = 0

    def input_memory(self, z: torch.Tensor, normalize: bool = True) -> None:
        """Store a single memory pattern via Hebbian learning.
        
        Incrementally updates W += scale * z @ z.T
        
        Args:
            z: Pattern to store, shape (num_units,) or (1, num_units)
            normalize: Whether to L2-normalize the pattern before storing
        """
        z = z.to(self.device).view(-1)
        
        if z.numel() != self.num_units:
            raise ValueError(f"Pattern size {z.numel()} != num_units {self.num_units}")
        
        if normalize:
            z = F.normalize(z, dim=0)
        
        # Hebbian update: W += scale * z ⊗ z
        self.W += self.scale * torch.outer(z, z)
        
        if self.zero_diag:
            self.W.fill_diagonal_(0.0)
        
        self.num_memories += 1

    def recall(
        self,
        x0: torch.Tensor,
        steps: int = 15,
        beta: float = 1.0,
        alpha: float = 1.0,
        use_tanh: bool = True,
        normalize_each: bool = True,
        asynchronous: bool = False,
        update_order: str = "random",
        rng: Optional[torch.Generator] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Recall a pattern from an initial cue.
        
        Iteratively applies: x_{t+1} = (1-α)x_t + α·f(W @ x_t)
        where f is tanh(β·) or identity.
        
        Args:
            x0: Initial state / cue, shape (num_units,)
            steps: Number of update iterations
            beta: Inverse temperature (default: self.beta)
            alpha: Update mixing coefficient (1.0 = full update)
            use_tanh: Whether to apply tanh nonlinearity
            normalize_each: Whether to L2-normalize after each step
            asynchronous: Use coordinate-wise updates instead of synchronous
            update_order: "random" or "sequential" (for asynchronous)
            rng: Random generator for async random order
            target: Optional target pattern for tracking cosine similarity
            
        Returns:
            x: Final recalled state, shape (num_units,)
            cos_sims: List of cosine similarities to target at each step
                      (empty if target is None)
        """
        beta = beta if beta is not None else self.beta
        x = x0.to(self.device).view(-1).clone()
        cos_sims: List[float] = []
        
        for _ in range(steps):
            # Track similarity to target
            if target is not None:
                cos_sim = F.cosine_similarity(
                    x.unsqueeze(0), 
                    target.to(self.device).view(1, -1), 
                    dim=1
                ).item()
                cos_sims.append(cos_sim)
            
            if not asynchronous:
                # Synchronous update
                h = self.W @ x
                delta_x = torch.tanh(beta * h) if use_tanh else h
                x = (1 - alpha) * x + alpha * delta_x
            else:
                # Asynchronous coordinate-wise updates
                if update_order == "random":
                    if rng is not None:
                        order = torch.randperm(self.num_units, generator=rng, device=self.device)
                    else:
                        order = torch.randperm(self.num_units, device=self.device)
                else:
                    order = torch.arange(self.num_units, device=self.device)
                
                # Maintain local field and update incrementally
                h = self.W @ x
                for k in order.tolist():
                    hk = h[k]
                    new_val = torch.tanh(beta * hk) if use_tanh else hk
                    updated = (1 - alpha) * x[k] + alpha * new_val
                    delta = updated - x[k]
                    if delta != 0:
                        x[k] = updated
                        h = h + delta * self.W[:, k]
            
            if normalize_each:
                x = F.normalize(x, dim=0)
        
        # Final similarity
        if target is not None:
            cos_sim = F.cosine_similarity(
                x.unsqueeze(0), 
                target.to(self.device).view(1, -1), 
                dim=1
            ).item()
            cos_sims.append(cos_sim)
        
        return x, cos_sims

    def reset(self) -> None:
        """Clear all stored memories."""
        self.W.zero_()
        self.num_memories = 0

    def energy(self, x: torch.Tensor) -> float:
        """Compute Hopfield energy E = -0.5 * x.T @ W @ x.
        
        Args:
            x: State vector, shape (num_units,)
            
        Returns:
            Energy (scalar)
        """
        x = x.to(self.device).view(-1)
        return -0.5 * (x @ self.W @ x).item()
