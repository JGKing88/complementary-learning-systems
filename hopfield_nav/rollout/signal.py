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
from hopfield import (
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


def chart_fraction(
    q: np.ndarray,
    recalled_np: np.ndarray,
    embeddings_np: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """``‖q‖ / ‖recall − x‖`` — how much of this recall the local chart explains.

    **The one genuinely missing input channel P3 found** (§7.7.2). The policy
    receives `q`, the recalled displacement projected into a local 2-D
    Gram-Schmidt frame — about 8 of its 74 input dims. The recall itself is
    1024-dimensional, so **1022 dimensions are projected away and never reach
    the policy**. This scalar is what they carried.

    §7.7.1 predicted, explicitly, that compressing the signal to one number
    would fail: "the 2-D frame is a much smaller subspace than the 64-dim
    chart, so its residual is dominated by in-chart directions the frame simply
    does not span, and the goal-present/absent contrast may wash out."

    That prediction was wrong. Measured (§7.7.2), at ten distractors:

        statistic                     P2 gain-5   w52 gain-100
        ‖q‖ (what the policy has)         0.698          0.930
        chart_frac (this)                 0.974          0.988
        d1_chart (needs a per-env fit)    0.942          0.972

    **It matches or beats the fitted 64-dim basis everywhere**, and beats ‖q‖
    by **+0.276 AUC** on the encoder §7 was measured on. The means say why: at
    ten distractors `frac_goal` is 0.638 against `frac_dist` 0.125, a 5× gap,
    with the goal-absent value near §5.8's √(2/D) ≈ 0.044 prediction for an
    unrelated direction in D = 1024.

    Free where the recall already exists: `W` is built every step by
    `project_to_signal` and the recall is already computed.

    Rows with no recall get 0.0 rather than a 0/0 — an empty memory means the
    quantity is *undefined*, and the same distinction the per-regime
    diagnostics needed (`cos_aq_frac`) applies here.
    """
    disp = np.asarray(recalled_np, dtype=np.float64) - np.asarray(
        embeddings_np, dtype=np.float64)
    den = np.linalg.norm(disp, axis=-1)
    num = np.linalg.norm(np.asarray(q, dtype=np.float64), axis=-1)
    out = np.zeros(den.shape, dtype=np.float32)
    ok = den > 1e-8
    if valid is not None:
        # A row with no memory has recalled == 0, so the displacement is -x
        # and the ratio is a perfectly finite NUMBER that means nothing. Gate
        # it explicitly rather than letting the arithmetic invent a value.
        ok = ok & np.asarray(valid).reshape(-1).astype(bool)
    out[ok] = (num[ok] / den[ok]).astype(np.float32)
    return out


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
    return_chart: bool = False,
) -> tuple:
    """Recall + project for a batch of states.

    ``hopfields`` is either one shared Hopfield (``shared_hopfield=True``) or a
    list of B per-env ones. Rows whose Hopfield holds no memory get a zero
    signal and a False memory mask -- there is nothing to recall, so there is no
    direction to report.

    Returns ``(signal_t, q, memory_mask_t, new_cached_W)``. ``new_cached_W`` is
    None when no recall happened, in which case the caller's existing cache is
    left alone.

    With ``return_chart=True`` a fifth element is appended: ``chart_frac``,
    the (B,) array from :func:`chart_fraction`. It is opt-in rather than always
    returned because fifteen call sites unpack the four-tuple, and a signature
    change to all of them is a large blast radius for a channel only one arm
    uses.
    """
    B = positions.shape[0]
    signal_dim = 4 if cfg.agent.hopfield_mode == "discrete" else 2
    hopfield_signal = torch.zeros(B, signal_dim, device=device)
    q_full = np.zeros((B, 2), dtype=np.float32)
    memory_mask = torch.zeros(B, dtype=torch.bool, device=device)

    if recompute_mask is None:
        recompute_mask = np.ones(B, dtype=bool)

    def _ret(sig, q, mask, W, recalled=None):
        if not return_chart:
            return sig, q, mask, W
        if recalled is None:
            chart = np.zeros(B, dtype=np.float32)
        else:
            chart = chart_fraction(q, recalled, embeddings_np,
                                   valid=mask.cpu().numpy())
        return sig, q, mask, W, chart

    if shared_hopfield:
        hop = hopfields
        if hop.num_memories == 0:
            return _ret(hopfield_signal, q_full, memory_mask, None)
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
        return _ret(hopfield_signal, q_full, memory_mask, cached_W,
                    recalled.cpu().numpy())

    # Per-env Hopfields: recall only the rows that have something to recall,
    # stacking their weight matrices into one batched pass.
    has_memory = [b for b in range(B) if hopfields[b].num_memories > 0]
    if not has_memory:
        return _ret(hopfield_signal, q_full, memory_mask, None)

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
    # `memory_mask` is now set, so _ret gates chart_frac on it: rows with no
    # memory get 0 rather than the finite-but-meaningless ratio that
    # recalled == 0 would otherwise produce.
    return _ret(hopfield_signal, q_full, memory_mask, cached_W,
                recalled_np_full)


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
    "chart_fraction",
    "hopfield_signal_at",
    "multistep_q",
    "oracle_signal_at",
    "project_to_signal",
    "q_to_signal",
]
