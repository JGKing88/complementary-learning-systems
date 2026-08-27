"""Test D -- does the field actually flow to the goal?

Angle error per cell is a local statistic. Whether the trajectories it induces
arrive is a global property of the vector field, and Test B cannot see it: a
field with 30 degrees of mean error and one sink in the wrong corner is a very
different object from one with 30 degrees of error and no spurious sinks.

No policy and no encoder calls -- this consumes the ``q`` field Test B already
built, which is why it runs by default.

Two variants, because they fail differently:

**discrete** steps to the neighbour ``classify_direction_batch`` picks, which is
literally what a perfect discrete agent would do. The field is then a
deterministic map on ``size^2`` states, so its whole structure is knowable
exactly: every trajectory ends in a cycle, and enumerating the cycles *is* the
sink inventory. No sampling, no tolerance.

**continuous** steps ``continuous_scale * q_hat`` from a float position and
snaps for lookup. It can stall in ways the discrete one cannot -- oscillating
across a cell boundary, or creeping at a fixed point of the snap.

Arrival is the goal *cell* (discrete) or within 0.5 of the goal point
(continuous). 0.5 is the snap-equality radius -- a geometric fact about
``round`` -- and deliberately not ``goal_radius``, which is a reward-shaping
knob that can change between training runs and must not gate an encoder metric.
"""
from __future__ import annotations

import numpy as np

from .harness import ProbeConfig, World, local_cells
from .stats import BinnedStat, Scalars

ARRIVAL_RADIUS = 0.5


def _unit_q(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return np.divide(q, n, out=np.zeros_like(q), where=n > 1e-12)


# ---------------------------------------------------------------------------
# Discrete: the field is a function on cells, so solve it exactly
# ---------------------------------------------------------------------------

def discrete_successor(q: np.ndarray, size: int) -> np.ndarray:
    """``next[i]`` for every cell ``i``, under wall clipping. ``(size^2,)``."""
    from hopfield_nav.utils import classify_direction_batch
    from hopfield_nav.world.env import CARDINAL_ACTIONS

    cells = local_cells(size)
    d = classify_direction_batch(q)
    step = np.array(CARDINAL_ACTIONS, dtype=np.int64)[d]

    # A zero q has no direction: treat it as "stay", which is what a policy
    # reading an all-zero channel would effectively do, and which shows up
    # honestly as a sink rather than as a spurious cardinal.
    step[np.linalg.norm(q, axis=1) <= 1e-12] = 0

    nxt = np.clip(cells + step, 0, size - 1)
    return nxt[:, 0] * size + nxt[:, 1]


def terminal_structure(nxt: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Resolve a functional graph: ``(terminal_id per node, cycles)``.

    Every node of a functional graph leads to exactly one cycle. A cycle of
    length 1 is a sink; longer is a limit cycle. This is the whole global
    structure of the discrete field, computed once in ``O(n)``.
    """
    n = nxt.shape[0]
    state = np.zeros(n, dtype=np.int8)      # 0 unvisited, 1 on stack, 2 done
    terminal = np.full(n, -1, dtype=np.int64)
    cycles: list[list[int]] = []

    for start in range(n):
        if state[start]:
            continue
        path: list[int] = []
        node = start
        while state[node] == 0:
            state[node] = 1
            path.append(node)
            node = int(nxt[node])
        if state[node] == 1:                # closed a new cycle
            at = path.index(node)
            cyc = path[at:]
            cid = len(cycles)
            cycles.append(cyc)
            for c in cyc:
                terminal[c] = cid
        tid = terminal[node]
        for p in reversed(path):
            if terminal[p] == -1:
                terminal[p] = tid
            state[p] = 2
    return terminal, cycles


def discrete_flow(q: np.ndarray, size: int, goal: tuple[int, int]) -> dict:
    """Reach-rate, steps-to-goal, sinks and limit cycles for one env.

    **The goal is absorbing.** An agent that arrives stops; it does not require
    the field to have a fixed point there. Without this the goal cell's own
    (near-zero, but not zero) ``q`` classifies to some cardinal and steps the
    agent straight back off the goal, so nothing is ever terminal at the goal
    and ``reach_rate`` collapses to ~0 no matter how good the field is. That is
    an artifact of modelling arrival as terminality, not a property of the
    encoder.
    """
    nxt = discrete_successor(q, size)
    goal_row = goal[0] * size + goal[1]
    nxt = nxt.copy()
    nxt[goal_row] = goal_row

    terminal, cycles = terminal_structure(nxt)
    goal_cycle = int(terminal[goal_row])
    reached = terminal == goal_cycle

    # Steps to arrival, walking the map (bounded by the state count).
    steps = np.full(size * size, -1, dtype=np.int64)
    for i in np.flatnonzero(reached):
        node, c = int(i), 0
        while node != goal_row and c <= size * size:
            node = int(nxt[node])
            c += 1
        steps[i] = c if node == goal_row else -1
    reached &= steps >= 0

    cells = local_cells(size)
    d0 = np.sqrt(((cells - np.array(goal)) ** 2).sum(1).astype(float))

    sinks, limit_cycles = [], []
    for cid, cyc in enumerate(cycles):
        if cid == goal_cycle:
            continue
        basin = int((terminal == cid).sum())
        rep = cyc[0]
        entry = {
            "cells": [[int(c // size), int(c % size)] for c in cyc],
            "basin": basin,
            "dist_from_goal": float(np.hypot(rep // size - goal[0],
                                             rep % size - goal[1])),
        }
        (sinks if len(cyc) == 1 else limit_cycles).append(entry)

    return {
        "reach_rate": float(reached.mean()),
        "mean_steps": float(steps[reached].mean()) if reached.any() else None,
        # Always beside mean_steps, never separated: mean_steps over a
        # shrinking success set is the classic trap -- a field that only
        # succeeds from nearby posts an excellent one.
        "n_success": int(reached.sum()),
        "n_starts": int(size * size),
        "sinks": sinks,
        "limit_cycles": limit_cycles,
        "reached": reached,
        "start_dist": d0,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Continuous
# ---------------------------------------------------------------------------

def continuous_flow(
    q: np.ndarray, size: int, goal: tuple[int, int], cfg: ProbeConfig,
) -> dict:
    """Follow ``continuous_scale * q_hat`` from every cell centre."""
    qh = _unit_q(q)
    p = local_cells(size).astype(float)
    n = p.shape[0]
    max_steps = cfg.flow_max_steps_factor * size

    arrived = np.zeros(n, dtype=bool)
    steps = np.full(n, -1, dtype=np.int64)
    goal_arr = np.array(goal, dtype=float)

    for t in range(max_steps):
        d = np.linalg.norm(p - goal_arr, axis=1)
        just = (~arrived) & (d <= ARRIVAL_RADIUS)
        steps[just] = t
        arrived |= just
        if arrived.all():
            break
        cx = np.clip(np.round(p[:, 0]), 0, size - 1).astype(np.int64)
        cy = np.clip(np.round(p[:, 1]), 0, size - 1).astype(np.int64)
        step = qh[cx * size + cy] * cfg.continuous_scale
        step[arrived] = 0.0
        p = np.clip(p + step, -0.5, size - 0.5)

    d = np.linalg.norm(p - goal_arr, axis=1)
    just = (~arrived) & (d <= ARRIVAL_RADIUS)
    steps[just] = max_steps
    arrived |= just

    d0 = np.sqrt(((local_cells(size) - np.array(goal)) ** 2)
                 .sum(1).astype(float))
    return {
        "reach_rate": float(arrived.mean()),
        "mean_steps": float(steps[arrived].mean()) if arrived.any() else None,
        "n_success": int(arrived.sum()),
        "n_starts": int(n),
        "reached": arrived,
        "start_dist": d0,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_test_d(
    field, worlds: list[World], cfg: ProbeConfig, *, progress=None,
) -> dict:
    """Test D over every world, ``K`` and step count."""
    from .harness import build_memory
    from .qfield import cell_q_field

    size = cfg.env_size
    d_edges = np.arange(0, int(np.ceil(np.hypot(size, size))) + 2, dtype=float)
    out: dict = {"config": cfg.to_json(), "arrival_radius": ARRIVAL_RADIUS,
                 "k": {}}

    for k in cfg.k_values:
        per_step = {}
        for s in cfg.steps:
            per_step[str(s)] = {
                "discrete": {"reach_by_dist": BinnedStat(d_edges, "frac"),
                             "scalars": Scalars()},
                "continuous": {"reach_by_dist": BinnedStat(d_edges, "frac"),
                               "scalars": Scalars()},
                "sinks": [], "limit_cycles": [],
            }

        for w in worlds:
            rng = np.random.RandomState(w.seed * 13 + k)
            mem = build_memory(field, w, k, cfg, rng)
            test_envs = (list(range(k)) if cfg.memory_mode == "multi_env_goals"
                         else [0])
            for j, e in enumerate(test_envs):
                qf, _c, _b = cell_q_field(field, w, e, mem, cfg)
                goal = w.specs[e].goal
                for s in cfg.steps:
                    P = per_step[str(s)]
                    dsc = discrete_flow(qf[s], size, goal)
                    cnt = continuous_flow(qf[s], size, goal, cfg)
                    for name, res in (("discrete", dsc), ("continuous", cnt)):
                        P[name]["reach_by_dist"].add(
                            res["start_dist"], res["reached"].astype(float))
                        P[name]["scalars"].add("reach_rate", res["reach_rate"])
                        if res["mean_steps"] is not None:
                            P[name]["scalars"].add("mean_steps",
                                                   res["mean_steps"])
                        P[name]["scalars"].add("n_success", res["n_success"])
                    if w.index < cfg.n_map_worlds and j < cfg.n_map_envs:
                        for sk in dsc["sinks"]:
                            P["sinks"].append({**sk, "world": w.index,
                                               "env": e})
                        for lc in dsc["limit_cycles"]:
                            P["limit_cycles"].append({**lc, "world": w.index,
                                                      "env": e})
            if progress:
                progress(f"D  k={k:>3} world={w.index}")

        out["k"][str(k)] = {
            str(s): {
                "discrete": {
                    "reach_by_dist": per_step[str(s)]["discrete"]
                    ["reach_by_dist"].to_json(),
                    "scalars": per_step[str(s)]["discrete"]
                    ["scalars"].to_json(),
                },
                "continuous": {
                    "reach_by_dist": per_step[str(s)]["continuous"]
                    ["reach_by_dist"].to_json(),
                    "scalars": per_step[str(s)]["continuous"]
                    ["scalars"].to_json(),
                },
                "sinks": sorted(per_step[str(s)]["sinks"],
                                key=lambda r: -r["basin"])[:64],
                "limit_cycles": sorted(per_step[str(s)]["limit_cycles"],
                                       key=lambda r: -r["basin"])[:32],
            } for s in cfg.steps
        }
    return out


__all__ = [
    "ARRIVAL_RADIUS", "continuous_flow", "discrete_flow",
    "discrete_successor", "run_test_d", "terminal_structure",
]
