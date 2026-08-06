"""Hopfield recall -> local frame -> direction signal, in one implementation.

The agent never sees a recalled memory directly. What reaches the policy is a
*direction*: recall from the current embedding, subtract the current embedding
from the result, and project that displacement onto the Gram-Schmidt basis of
the two local scaffold axes, giving a 2-D (East, North) vector ``q``. Discrete
agents get ``q`` classified to a cardinal one-hot; continuous agents get it
normalized (or raw, under ``input_hopfield_raw``).

This ran twice: batched over B envs in the rollout collector, and again inline
at B=1 in ``eval.agent_step`` -- with the eval copy recomputing the
Gram-Schmidt basis every step while the collector cached it for
``recompute_interval`` steps. Identical today only because that interval is 1
and no CLI can set it otherwise. Both now call the functions here.

Everything is batched. The single-env caller passes B=1.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import AgentConfig, TrainConfig
from ..world.memory import (
    Hopfield, recall_per_env_batch, recall_per_env_batch_trajectory,
)
from ..utils import classify_direction_batch, direction_to_onehot


def q_to_signal(q: np.ndarray, agent_cfg: AgentConfig) -> np.ndarray:
    """Turn a projected displacement into the agent's direction channel.

    Discrete: nearest cardinal, one-hot (B, 4). Continuous: unit vector (B, 2).

    This deliberately does *not* implement ``input_hopfield_raw``: that flag
    substitutes the unnormalized ``q`` for this signal in the policy input,
    while the normalized signal is still what teacher-forcing and the direction
    classifier consume. Callers choose which of the two to feed the policy.
    """
    if agent_cfg.hopfield_mode == "discrete":
        return direction_to_onehot(classify_direction_batch(q))
    mag = np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
    return (q / mag).astype(np.float32)


def project_to_signal(
    vectorhash,
    embeddings_np: np.ndarray,
    recalled_np: np.ndarray,
    positions: np.ndarray,
    env_offset: tuple[int, int],
    agent_cfg: AgentConfig,
    cached_W: np.ndarray | None = None,
    recompute_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (recalled - current) into the local frame and shape the signal.

    Returns ``(signal, q, W)``:
        signal: (B, 4) one-hot or (B, 2) unit vector
        q:      (B, 2) raw projected displacement, (East, North)
        W:      (B, 2, embed_dim) Gram-Schmidt basis. Callers hold this as the
                next call's ``cached_W`` so ``recompute_interval`` actually
                saves work -- otherwise W is rebuilt every step regardless of
                the config.
    """
    W = vectorhash.gram_schmidt_projection(
        positions, env_offset, cached_W=cached_W, recompute_mask=recompute_mask,
    )
    q = vectorhash.project_displacement(embeddings_np, recalled_np, W)
    q = q.astype(np.float32, copy=False)
    return q_to_signal(q, agent_cfg), q, W


def hopfield_signal_at(
    vectorhash,
    cfg: TrainConfig,
    embeddings_np: np.ndarray,
    embeddings: torch.Tensor,
    positions: np.ndarray,
    env_offset: tuple[int, int],
    hopfields,
    shared_hopfield: bool,
    device: torch.device,
    embed_dim: int,
    cached_W: np.ndarray | None = None,
    recompute_mask: np.ndarray | None = None,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray | None]:
    """Recall + project for a batch of states.

    ``hopfields`` is either one shared Hopfield (``shared_hopfield=True``) or a
    list of B per-env ones. Rows whose Hopfield holds no memory get a zero
    signal and a False memory mask -- there is nothing to recall, so there is no
    direction to report.

    Returns ``(signal_t, q, memory_mask_t, new_cached_W)``. ``new_cached_W`` is
    None when no recall happened, in which case the caller's existing cache is
    left alone.
    """
    B = positions.shape[0]
    signal_dim = 4 if cfg.agent.hopfield_mode == "discrete" else 2
    hopfield_signal = torch.zeros(B, signal_dim, device=device)
    q_full = np.zeros((B, 2), dtype=np.float32)
    memory_mask = torch.zeros(B, dtype=torch.bool, device=device)

    if recompute_mask is None:
        recompute_mask = np.ones(B, dtype=bool)

    if shared_hopfield:
        hop = hopfields
        if hop.num_memories == 0:
            return hopfield_signal, q_full, memory_mask, None
        recalled = hop.recall_batch(
            embeddings, steps=cfg.hopfield.steps,
            beta=cfg.hopfield.beta, alpha=cfg.hopfield.alpha,
        )
        sig_np, q_full, cached_W = project_to_signal(
            vectorhash, embeddings_np, recalled.cpu().numpy(),
            positions, env_offset, cfg.agent, cached_W, recompute_mask,
        )
        hopfield_signal = torch.from_numpy(sig_np).float().to(device)
        memory_mask[:] = True
        return hopfield_signal, q_full, memory_mask, cached_W

    # Per-env Hopfields: recall only the rows that have something to recall,
    # stacking their weight matrices into one batched pass.
    has_memory = [b for b in range(B) if hopfields[b].num_memories > 0]
    if not has_memory:
        return hopfield_signal, q_full, memory_mask, None

    idx = torch.as_tensor(has_memory, device=device, dtype=torch.long)
    W_stack = torch.stack([hopfields[b].W for b in has_memory], dim=0)
    recalled_stack = recall_per_env_batch(
        embeddings.index_select(0, idx), W_stack,
        steps=cfg.hopfield.steps,
        beta=cfg.hopfield.beta, alpha=cfg.hopfield.alpha,
    )
    recalled_np_full = np.zeros((B, embed_dim), dtype=np.float32)
    recalled_np_full[has_memory] = recalled_stack.cpu().numpy()
    sig, q_full, cached_W = project_to_signal(
        vectorhash, embeddings_np, recalled_np_full,
        positions, env_offset, cfg.agent, cached_W, recompute_mask,
    )
    memory_mask[idx] = True
    hopfield_signal = torch.where(
        memory_mask.unsqueeze(-1),
        torch.from_numpy(sig).float().to(device),
        hopfield_signal,
    )
    return hopfield_signal, q_full, memory_mask, cached_W


def oracle_signal_at(
    vectorhash,
    embeddings_np: np.ndarray,
    positions: np.ndarray,
    env_offset: tuple[int, int],
    goal_local: tuple[int, int],
    agent_cfg: AgentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """The signal a *perfect* recall would produce: goal embedding, no dynamics.

    Same Gram-Schmidt projection as the real path, but the displacement is
    ``goal_embedding - current_embedding`` rather than a recalled pattern.
    Answers "would a flawless directional cue fix this behavior?", isolating
    policy failures from associative-memory failures.

    Returns ``(signal, q)``.
    """
    goal_arr = np.array([goal_local], dtype=np.int32)
    goal_enc = vectorhash.get_encoded_state(goal_arr, env_offset)
    goal_enc = np.broadcast_to(goal_enc, embeddings_np.shape)
    signal, q, _ = project_to_signal(
        vectorhash, embeddings_np, goal_enc, positions, env_offset, agent_cfg,
    )
    return signal, q


def multistep_q(
    vectorhash,
    cfg: TrainConfig,
    embeddings_np: np.ndarray,
    embeddings: torch.Tensor,
    hopfields,
    shared_hopfield: bool,
    cached_W: np.ndarray | None,
    multistep_steps,
    embed_dim: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Project the recall *trajectory* at each requested iteration count.

    Lets the policy see where recall is heading, not just where it lands after
    ``cfg.hopfield.steps``. Returns ``{step: (B, 2) float32}``, zero-filled
    where memory is empty or no basis is available.
    """
    if not multistep_steps:
        return {}
    B = embeddings.shape[0]
    out = {s: np.zeros((B, 2), dtype=np.float32) for s in multistep_steps}
    if cached_W is None:
        return out

    if shared_hopfield:
        hop = hopfields
        if hop.num_memories == 0:
            return out
        traj = hop.recall_batch_trajectory(
            embeddings, multistep_steps,
            beta=cfg.hopfield.beta, alpha=cfg.hopfield.alpha,
        )
        for s, X_s in traj.items():
            out[s] = vectorhash.project_displacement(
                embeddings_np, X_s.cpu().numpy(), cached_W,
            ).astype(np.float32, copy=False)
        return out

    has_memory = [b for b in range(B) if hopfields[b].num_memories > 0]
    if not has_memory:
        return out
    idx_t = torch.as_tensor(has_memory, device=device, dtype=torch.long)
    W_stack = torch.stack([hopfields[b].W for b in has_memory], dim=0)
    traj = recall_per_env_batch_trajectory(
        embeddings.index_select(0, idx_t), W_stack, multistep_steps,
        beta=cfg.hopfield.beta, alpha=cfg.hopfield.alpha,
    )
    for s, X_s in traj.items():
        recalled_np_full = np.zeros((B, embed_dim), dtype=np.float32)
        recalled_np_full[has_memory] = X_s.cpu().numpy()
        out[s] = vectorhash.project_displacement(
            embeddings_np, recalled_np_full, cached_W,
        ).astype(np.float32, copy=False)
    return out


__all__ = [
    "hopfield_signal_at",
    "multistep_q",
    "oracle_signal_at",
    "project_to_signal",
    "q_to_signal",
]
