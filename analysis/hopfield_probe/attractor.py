"""Test A -- are stored goals attractors, and how big is the basin?

Two questions that sound like one:

**Is a stored pattern a fixed point?** Recall from ``z_k`` itself and see
whether it stays. Swept over ``steps``, because "attractor" is a statement
about iterating the map, not about one application of it.

**Over what real-space disc does a cue relax to the goal?** The cue is not a
noised goal vector -- it is the embedding of *another cell*, which is what
makes the radius live in cells rather than in cosine.

Retrieval is decided against a bank of **cells**, not against the stored
patterns. "Nearer the goal than any other stored goal" is a K-way choice and
far too easy at small K; "nearer the goal cell than any other cell in the
world" is the question that makes this a position readout. So ``retrieved``
returns a *cell*, ``exact_hit`` is threshold-free, and there is no ``tau``
anywhere in this file.
"""
from __future__ import annotations

import warnings

import numpy as np

from .encode import Field
from .harness import (
    CellBank, Memory, NEAR_CELLS, OUTCOMES, ProbeConfig, World,
    build_cell_bank, build_memory, env_offset_distances, local_cells,
    recall_trajectory, tanh_argument,
)
from .stats import BinnedStat, CategoryByDist, Scalars

N_SECTORS = 8


# ---------------------------------------------------------------------------
# Radii
# ---------------------------------------------------------------------------

def first_failure_radius(
    hit: np.ndarray, dist: np.ndarray, frac: float = 1.0,
    max_r: int | None = None,
) -> float:
    """Largest integer ``r`` with ``mean(hit[dist <= r]) >= frac``, stopping at
    the **first** failure.

    First failure, not last success: the condition is not monotone in ``r`` --
    it can fail at 3 and hold again at 12 once the offending cell falls inside
    the disc -- and a radius only means anything if the guarantee nests all the
    way in. Same convention as ``encoder_training.unique_radius``, so the two
    are readable side by side.

    Returns ``-1.0`` when even ``r=0`` fails, i.e. the goal cell does not
    retrieve itself.
    """
    hit = np.asarray(hit, dtype=bool)
    dist = np.asarray(dist, dtype=float)
    if hit.size == 0:
        return float("nan")
    top = int(np.ceil(np.nanmax(dist))) if max_r is None else int(max_r)
    best = -1.0
    for r in range(0, top + 1):
        sel = dist <= r + 1e-9
        if not sel.any():
            continue
        if hit[sel].mean() + 1e-12 >= frac:
            best = float(r)
        else:
            break
    return best


def radius_by_direction(
    hit: np.ndarray, delta: np.ndarray, dist: np.ndarray, frac: float = 1.0,
) -> list[float]:
    """``first_failure_radius`` within each of ``N_SECTORS`` angular sectors.

    A radius averaged over directions hides a direction where it is zero, and
    anisotropy is a known live property of this scaffold.
    """
    ang = np.arctan2(delta[:, 1], delta[:, 0])
    sector = np.clip(
        np.floor((ang + np.pi) / (2 * np.pi / N_SECTORS)).astype(int),
        0, N_SECTORS - 1)
    out = []
    for s in range(N_SECTORS):
        m = (sector == s) & (dist > 0)
        out.append(first_failure_radius(hit[m], dist[m], frac) if m.any()
                   else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Fixed points
# ---------------------------------------------------------------------------

def fixed_point_probe(mem: Memory, cfg: ProbeConfig) -> dict:
    """Recall each stored pattern from itself. Sec 3.1."""
    traj = recall_trajectory(mem, mem.Z, cfg.steps, cfg)
    Zn = _unit(mem.Z)
    k = Zn.shape[0]
    out: dict[str, dict] = {}
    for s in cfg.steps:
        Xn = _unit(traj[s])
        cos_self = (Xn * Zn).sum(axis=1)
        top = (Xn @ Zn.T).argmax(axis=1)

        # Endpoint spread across *different* cues. If Sec 1.3 is right this
        # rises toward 1 with steps: every cue landing on one vector. One
        # number for "the dynamics have a single attractor, not K".
        if k > 1:
            g = Xn @ Xn.T
            iu = np.triu_indices(k, k=1)
            pairwise = float(np.mean(g[iu]))
        else:
            pairwise = float("nan")

        out[str(s)] = {
            "residual_mean": float(np.mean(1.0 - cos_self)),
            "cos_self_mean": float(np.mean(cos_self)),
            # -z is as much a fixed point as z under a symmetric W with
            # normalisation, and a flipped recall inverts q. Counted, never
            # absorbed into |cos|.
            "sign_flip_frac": float(np.mean(cos_self < 0)),
            "frac_self_consistent": float(np.mean(top == np.arange(k))),
            "mean_pairwise_cos": pairwise,
        }
    return out


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    X: np.ndarray, bank: CellBank, cfg: ProbeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest bank cell for each recall endpoint. Chunked over cues.

    The full ``(n_cues, M)`` cosine is over 3 GB at ``K=50``, so this walks the
    cues in blocks and keeps only the argmax and its value.
    """
    Xn = _unit(X)
    Bn = _unit(bank.Z)
    idx = np.empty(Xn.shape[0], dtype=np.int64)
    val = np.empty(Xn.shape[0], dtype=np.float32)
    for a in range(0, Xn.shape[0], cfg.cos_chunk):
        b = min(a + cfg.cos_chunk, Xn.shape[0])
        c = Xn[a:b] @ Bn.T
        idx[a:b] = c.argmax(axis=1)
        val[a:b] = c.max(axis=1)
    return idx, val


def classify_outcomes(
    ret_env: np.ndarray, ret_x: np.ndarray, ret_y: np.ndarray,
    test_env: int, goal: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """``(outcome_index, retr_dist)`` per cue. Sec 3.2.

    ``retr_dist`` is **NaN** whenever retrieval left the test env. Real-space
    distance to the goal is not a meaningful quantity across two different
    rooms, and averaging one in would manufacture a small, reassuring number.
    """
    n = ret_env.shape[0]
    same = ret_env == test_env
    dist = np.full(n, np.nan)
    dist[same] = np.sqrt((ret_x[same] - goal[0]) ** 2
                         + (ret_y[same] - goal[1]) ** 2)

    out = np.full(n, OUTCOMES.index("alias"), dtype=np.int8)
    out[ret_env >= 0] = OUTCOMES.index("other_env")
    out[same] = OUTCOMES.index("far_same_env")
    out[same & (dist <= NEAR_CELLS)] = OUTCOMES.index("near")
    out[same & (dist == 0.0)] = OUTCOMES.index("exact")
    return out, dist


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_test_a(
    field: Field, worlds: list[World], cfg: ProbeConfig, *, progress=None,
) -> dict:
    """Test A over every world and every ``K``, streamed into accumulators."""
    size = cfg.env_size
    cells = local_cells(size)
    d_edges = np.arange(0, int(np.ceil(np.hypot(size, size))) + 2, dtype=float)

    out: dict = {"config": cfg.to_json(), "outcomes": list(OUTCOMES), "k": {}}
    tanh_pool: list[np.ndarray] = []

    for k in cfg.k_values:
        acc = {
            str(s): {
                "cos_goal": BinnedStat(d_edges, "cos"),
                "exact": BinnedStat(d_edges, "frac"),
                "retr_dist": BinnedStat(d_edges, "cells"),
                "outcome": CategoryByDist(d_edges, len(OUTCOMES)),
                "scalars": Scalars(),
            } for s in cfg.steps
        }
        sector_pool = {str(s): [] for s in cfg.steps}
        confusion_pool = {str(s): np.zeros((k, k)) for s in cfg.steps}
        fixed_pool: dict[str, list[dict]] = {str(s): [] for s in cfg.steps}
        diag_fracs: list[float] = []
        maps: list[dict] = []

        for w in worlds:
            rng = np.random.RandomState(w.seed * 31 + k)
            mem = build_memory(field, w, k, cfg, rng)
            bank = build_cell_bank(field, w, k, cfg, rng)
            diag_fracs.append(mem.diag_frac)

            fp = fixed_point_probe(mem, cfg)
            for s in cfg.steps:
                fixed_pool[str(s)].append(fp[str(s)])
            if len(tanh_pool) < 8:
                tanh_pool.append(tanh_argument(mem, mem.Z, cfg))

            # multi_env_goals scores every env in the world against one W --
            # K measurements from one world, which is what makes K=50 cheap.
            # The other modes define a single test env, so they score one.
            test_envs = (list(range(k)) if cfg.memory_mode == "multi_env_goals"
                         else [0])

            cues = np.concatenate([
                field.encode(cells[:, 0] + w.specs[e].offset[0],
                             cells[:, 1] + w.specs[e].offset[1])
                for e in test_envs], axis=0)
            traj = recall_trajectory(mem, cues, cfg.steps, cfg)
            n_cell = size * size

            for s in cfg.steps:
                idx, _cos_top = retrieve(traj[s], bank, cfg)
                ret_env, ret_x, ret_y = bank.decode(idx)
                A = acc[str(s)]
                w_exact, w_r95, w_frac = [], [], []

                for j, e in enumerate(test_envs):
                    sl = slice(j * n_cell, (j + 1) * n_cell)
                    goal = w.specs[e].goal
                    delta = cells - np.array(goal)
                    d = np.sqrt((delta ** 2).sum(1).astype(float))

                    outcome, retr_d = classify_outcomes(
                        ret_env[sl], ret_x[sl], ret_y[sl], e, goal)
                    exact = outcome == OUTCOMES.index("exact")

                    gz = mem.Z[mem.owner == e]
                    cos_goal = (_unit(traj[s][sl]) @ gz[0] if gz.shape[0]
                                else np.full(n_cell, np.nan))

                    A["cos_goal"].add(d, cos_goal)
                    A["exact"].add(d, exact.astype(float))
                    A["retr_dist"].add(d, retr_d)
                    A["outcome"].add(d, outcome)

                    w_frac.append(float(exact.mean()))
                    w_exact.append(first_failure_radius(exact, d, 1.0))
                    w_r95.append(first_failure_radius(exact, d, 0.95))
                    sector_pool[str(s)].append(
                        radius_by_direction(exact, delta, d, 1.0))

                    om = outcome == OUTCOMES.index("other_env")
                    if om.any():
                        v = ret_env[sl][om]
                        v = v[(v >= 0) & (v < k)]
                        np.add.at(confusion_pool[str(s)], (e, v), 1)

                    if (w.index < cfg.n_map_worlds
                            and j < cfg.n_map_envs and s == cfg.steps[0]):
                        maps.append({
                            "world": w.index, "env": e, "steps": int(s),
                            "size": size,
                            "goal": [int(goal[0]), int(goal[1])],
                            "outcome": outcome.astype(int).tolist(),
                            "retr_dist": _nan_list(retr_d),
                            "cos_goal": _nan_list(cos_goal),
                        })

                A["scalars"].add("r_exact_all", float(np.mean(w_exact)))
                A["scalars"].add("r_exact_95", float(np.mean(w_r95)))
                A["scalars"].add("exact_frac", float(np.mean(w_frac)))

            if progress:
                progress(f"A  k={k:>3} world={w.index}")

        out["k"][str(k)] = {
            "diag_frac_mean": float(np.mean(diag_fracs)),
            "per_step": {
                str(s): {
                    "cos_goal": acc[str(s)]["cos_goal"].to_json(),
                    "exact": acc[str(s)]["exact"].to_json(),
                    "retr_dist": acc[str(s)]["retr_dist"].to_json(),
                    "outcome": acc[str(s)]["outcome"].to_json(),
                    "scalars": acc[str(s)]["scalars"].to_json(),
                    "fixed_point": _pool_dicts(fixed_pool[str(s)]),
                    "r_by_direction": _sector_summary(sector_pool[str(s)]),
                    "confusion": confusion_pool[str(s)].tolist(),
                } for s in cfg.steps
            },
            "offset_distances": (
                env_offset_distances(worlds[0], k).tolist()
                if worlds else []),
            "maps": maps,
        }

    if tanh_pool:
        allv = np.abs(np.concatenate(tanh_pool))
        edges = np.logspace(-9, 3, 49)
        out["tanh_arg"] = {
            "abs_mean": float(allv.mean()),
            "abs_p50": float(np.percentile(allv, 50)),
            "abs_p99": float(np.percentile(allv, 99)),
            "abs_max": float(allv.max()),
            "hist_edges": edges.tolist(),
            "hist": np.histogram(allv, bins=edges)[0].tolist(),
        }
    return out


def _nan_list(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a)]


def _pool_dicts(rows: list[dict]) -> dict:
    """Mean and per-world values for each key of a list of flat dicts."""
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows], dtype=float)
        fin = vals[np.isfinite(vals)]
        out[key] = {
            "mean": float(fin.mean()) if fin.size else None,
            "std": float(fin.std()) if fin.size else None,
            "values": [None if not np.isfinite(v) else float(v)
                       for v in vals],
        }
    return out


def _sector_summary(rows: list[list[float]]) -> dict:
    if not rows:
        return {"mean": [], "min": []}
    a = np.array(rows, dtype=float)
    # An all-NaN sector is legitimate: at small env sizes a sector can contain
    # no cell at all. nanmean warns rather than erroring, so silence it here
    # instead of leaving a warning that reads like a defect.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            "mean": [None if not np.isfinite(v) else float(v)
                     for v in np.nanmean(a, axis=0)],
            "min": [None if not np.isfinite(v) else float(v)
                    for v in np.nanmin(a, axis=0)],
        }


__all__ = [
    "classify_outcomes", "first_failure_radius", "fixed_point_probe",
    "radius_by_direction", "retrieve", "run_test_a",
]
