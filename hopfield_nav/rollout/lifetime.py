"""Short continual-protocol lifetimes for idea 3 (GRID_MLP_NAV_PLAN §7.3).

Idea 3 has no memory module: the GRU's hidden state is where every goal has to
live. Stage 2 trains it on a small copy of the continual protocol
(``analysis/continual/agenthash.py``): ``n_envs`` arenas introduced one at a
time, and in block i, ``iters`` passes over arenas 1..i, one trial each. A
trial starts at a random cell and ends at the goal or after the collector's
``T`` steps. Each of the collector's B rows is one lifetime: the rows share
the arenas and their goals but start each trial at their own cell.

The recurrent state runs through the whole lifetime. It is never reset, and a
row that finishes a trial early keeps its state unchanged while the others
finish (``freeze_done_state``). Search shaping (novelty, epsilon) applies in
an arena until the row has first found its goal there, as in the task regime.

For PPO a lifetime is one trajectory: each row's real steps, in order, packed
to the front and zero-padded behind. Re-forwarding the packed sequence from a
zero state therefore reproduces the collected states exactly. The lifetime is
one episode: returns run across trial boundaries and stop at its end.
"""
from __future__ import annotations

import numpy as np
import torch

from .types import RolloutBatch

# The fields PPO reads, all (B, T, ...) on the time axis.
_PACKED = ("obs", "move_actions", "store_actions", "move_log_probs",
           "store_log_probs", "values", "rewards", "goal_reached",
           "explore_mask", "policy_action_mask")


def protocol_order(n_envs: int, iters: int) -> list[int]:
    """Arena index of each trial: block i runs ``iters`` passes over 0..i."""
    order: list[int] = []
    for i in range(int(n_envs)):
        for _ in range(int(iters)):
            order.extend(range(i + 1))
    return order


def run_lifetime(collector, agent, envs, offsets, *, n_envs: int, iters: int,
                 epsilon: float, update_idx: int = 0, h0=None):
    """Run one lifetime per row over ``envs[:n_envs]``; returns
    ``(rollouts, order)``, one rollout per trial.

    The collector reads novelty off its own config, so the caller sets
    ``cfg.hopfield.novelty_reward`` before calling, as the composer does for
    every other regime.
    """
    if len(envs) < n_envs:
        raise ValueError(f"a lifetime of {n_envs} arenas needs as many envs, "
                         f"got {len(envs)}")
    B = collector.B
    order = protocol_order(n_envs, iters)
    fired = [None] * n_envs
    h = h0
    rollouts = []
    for j in order:
        env = envs[j]
        was_active = getattr(env, "goals_active", True)
        env.goals_active = True
        try:
            r = collector.collect_rollout(
                env, agent, [None] * B, allow_store=True, h_rnn=h,
                env_offset=offsets[j], update_idx=update_idx,
                epsilon_now=epsilon, goal_in_memory_init=False,
                ends_on_goal=True, task_mode=True, store_fired_init=fired[j],
                freeze_done_state=True)
        finally:
            env.goals_active = was_active
        h = r.h_final
        fired[j] = r.store_fired_final
        rollouts.append(r)
    return rollouts, order


def pack(rollouts: list[RolloutBatch]) -> RolloutBatch:
    """The lifetime as one trajectory per row: real steps first, in order.

    A stable sort on "not alive" moves each row's alive steps to the front
    without reordering them. The padding is zeroed and marked not alive, and
    the bootstrap is zero because the lifetime is the episode.
    """
    alive = torch.cat([r.alive_mask for r in rollouts], dim=1)       # (B, S)
    B, S = alive.shape
    L = int(alive.sum(1).max().item())
    key = (1.0 - alive) * S + torch.arange(S, device=alive.device).float()
    idx = torch.argsort(key, dim=1)[:, :L]                            # (B, L)
    keep = torch.gather(alive, 1, idx)
    out = {}
    for name in _PACKED:
        x = torch.cat([getattr(r, name) for r in rollouts], dim=1)
        ix = idx.view(B, L, *([1] * (x.dim() - 2))).expand(B, L, *x.shape[2:])
        g = torch.gather(x, 1, ix)
        m = keep.view(B, L, *([1] * (x.dim() - 2))).to(g.dtype)
        out[name] = g * m
    return RolloutBatch(
        **out,
        bootstrap_value=torch.zeros(B, device=alive.device),
        alive_mask=keep,
        h_final=rollouts[-1].h_final,
    )


def pad_common(batches: list[RolloutBatch]) -> list[RolloutBatch]:
    """Zero-pad packed lifetimes to one length (PPO pools them along rows)."""
    L = max(b.obs.shape[1] for b in batches)
    out = []
    for b in batches:
        n = L - b.obs.shape[1]
        if n == 0:
            out.append(b)
            continue

        def pad(x):
            z = x.new_zeros(x.shape[0], n, *x.shape[2:])
            return torch.cat([x, z], dim=1)

        out.append(RolloutBatch(
            **{name: pad(getattr(b, name)) for name in _PACKED},
            bootstrap_value=b.bootstrap_value,
            alive_mask=pad(b.alive_mask),
            h_final=b.h_final,
        ))
    return out


def trial_outcomes(rollouts: list[RolloutBatch], order: list[int]) -> dict:
    """Per row and trial: arena, reached, steps to the goal, and whether the
    row had found this arena's goal on an earlier trial (``known``), plus how
    many trials of other arenas it has done since it last reached that goal
    (``gap``; -1 when not known).

    All arrays are (B, n_trials).
    """
    B = rollouts[0].rewards.shape[0]
    n = len(rollouts)
    reached = np.zeros((B, n), dtype=bool)
    steps = np.full((B, n), np.nan)
    known = np.zeros((B, n), dtype=bool)
    gap = np.full((B, n), -1, dtype=np.int64)
    last_reach: dict[int, np.ndarray] = {}     # arena -> trial index, -1 never
    for k, (r, j) in enumerate(zip(rollouts, order)):
        g = r.goal_reached.cpu().numpy() > 0                         # (B, T)
        hit = g.any(1)
        reached[:, k] = hit
        steps[hit, k] = g[hit].argmax(1)
        lr = last_reach.setdefault(j, np.full(B, -1, dtype=np.int64))
        known[:, k] = lr >= 0
        # Trials of OTHER arenas since this arena's goal was last reached.
        others = np.array([sum(1 for kk in range(int(l) + 1, k) if order[kk] != j)
                           if l >= 0 else -1 for l in lr])
        gap[:, k] = others
        lr[hit] = k
    first = np.zeros(n, dtype=bool)
    seen = set()
    for k, j in enumerate(order):
        if j not in seen:
            first[k] = True
            seen.add(j)
    return {"order": np.asarray(order), "reached": reached, "steps": steps,
            "known": known, "gap": gap, "first": first}


def summarise(outcomes: list[dict]) -> dict[str, float]:
    """Pooled lifetime statistics, for logs and evals.

    ``first_found``: the first trial in each arena (a search). ``known_*``:
    trials where the row had already reached this arena's goal, the trials a
    memory is for. ``unknown_found``: later trials in an arena whose goal the
    row has never reached (a search again).
    """
    reached = np.concatenate([o["reached"].ravel() for o in outcomes])
    steps = np.concatenate([o["steps"].ravel() for o in outcomes])
    known = np.concatenate([o["known"].ravel() for o in outcomes])
    first = np.concatenate([np.broadcast_to(o["first"], o["reached"].shape).ravel()
                            for o in outcomes])

    def mean(x):
        return float(np.mean(x)) if x.size else float("nan")

    kr = known & reached
    return {
        "first_found": mean(reached[first]),
        "known_reached": mean(reached[known]),
        "known_steps": mean(steps[kr]),
        "unknown_found": mean(reached[~known & ~first]),
        "known_frac": mean(known[~first]),
    }


__all__ = ["pack", "pad_common", "protocol_order", "run_lifetime", "summarise", "trial_outcomes"]
