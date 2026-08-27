"""Tests B and C -- how far off does ``q`` point?

**One implementation, two position sources.** Test C differs from Test B only
in where the position comes from and what the true bearing is measured against;
forking them would guarantee they drift apart and stop being comparable, which
would destroy ``excess(d)`` -- the one number Test C exists to produce.

The conventions, and they are the easiest thing in this package to get
silently backwards:

    q            = W @ (recalled - current),  row 0 East, row 1 North
    theta_pred   = atan2(q_north, q_east)        # == (dy, dx)
    theta_true   = atan2(goal_y - y, goal_x - x)
    err          = wrap_to_pi(theta_pred - theta_true)

A transpose anywhere in there mirrors every angle while leaving every aggregate
looking plausible, which is why ``test_hopfield_probe.py`` pins a hand-built
due-East case.

``q`` at a continuous position depends on that position **only through
``snap(p)``**, so there are just ``size^2`` distinct values per
(env, K, steps). Test C computes those once and then costs one lookup per
sample -- which is why the sampling rate can be pushed as high as we like.
"""
from __future__ import annotations

import numpy as np

from .encode import Field
from .harness import (
    Memory, ProbeConfig, World, build_memory, local_cells,
    recall_trajectory, scored_envs,
)
from .stats import (
    BinnedStat, Map2D, Scalars, continuous_dist_edges, wrap_to_pi,
)

N_SECTORS = 8
CHANCE_ABS_DEG = 90.0
CHANCE_ACC45 = 0.25
CHANCE_ACC90 = 0.5


# ---------------------------------------------------------------------------
# The q field
# ---------------------------------------------------------------------------

def project_q(basis: np.ndarray, current: np.ndarray,
              recalled: np.ndarray) -> np.ndarray:
    """``VectorHash.project_displacement``: ``(B, 2)`` as (East, North)."""
    return np.einsum("bij,bj->bi", basis, recalled - current)


def bearing(delta_x: np.ndarray, delta_y: np.ndarray) -> np.ndarray:
    """Bearing of a displacement in the same frame ``q`` lives in."""
    return np.arctan2(delta_y, delta_x)


def q_error(q: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    """Signed angular error in radians, ``[-pi, pi)`` (see wrap_to_pi)."""
    return wrap_to_pi(np.arctan2(q[:, 1], q[:, 0]) - theta_true)


def cell_q_field(
    field: Field, world: World, env: int, mem: Memory, cfg: ProbeConfig,
    *, swap_gram_schmidt: bool = False,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """``q`` at every cell of one env, for every step count.

    Returns ``({steps: (n_cell, 2)}, cues, basis)``. The basis is computed
    **once** and reused across the whole recall trajectory, which is what the
    production ``multistep_q`` does -- recomputing it per step would describe a
    system that does not exist.
    """
    cells = local_cells(cfg.env_size)
    offset = world.specs[env].offset
    cues = field.encoded_state(cells, offset)
    basis = field.local_basis(cells, offset,
                              swap_gram_schmidt=swap_gram_schmidt)
    traj = recall_trajectory(mem, cues, cfg.steps, cfg)
    return ({s: project_q(basis, cues, traj[s]) for s in cfg.steps},
            cues, basis)


# ---------------------------------------------------------------------------
# Test B accumulation
# ---------------------------------------------------------------------------

class GridAcc:
    """Everything Test B reports, for one ``(K, steps)``."""

    def __init__(self, cfg: ProbeConfig):
        size = cfg.env_size
        d_edges = np.arange(0, int(np.ceil(np.hypot(size, size))) + 2,
                            dtype=float)
        self.abs_err = BinnedStat(d_edges, "angle_deg")
        self.acc45 = BinnedStat(d_edges, "frac")
        self.acc90 = BinnedStat(d_edges, "frac")
        self.qnorm = BinnedStat(d_edges, "qnorm")
        # Goal-relative: the goal moves between envs, so an env-absolute
        # per-cell average is a different quantity in every sample. Re-indexed
        # by (c - g), the goal is at the centre by construction.
        self.rel_signed = Map2D((2 * size - 1, 2 * size - 1))
        self.rel_abs = Map2D((2 * size - 1, 2 * size - 1))
        # Kept anyway: it answers a different question -- is q worse near walls
        # and in corners? -- that goal-relative coordinates average away.
        self.abs_map = Map2D((size, size))
        self.sector_signed = np.zeros(N_SECTORS)
        self.sector_n = np.zeros(N_SECTORS, dtype=np.int64)
        self.scalars = Scalars()
        self.size = size

    def add_env(self, q: np.ndarray, cells: np.ndarray,
                goal: tuple[int, int]) -> dict:
        size = self.size
        dx = goal[0] - cells[:, 0]
        dy = goal[1] - cells[:, 1]
        d = np.sqrt((dx ** 2 + dy ** 2).astype(float))

        theta_true = bearing(dx.astype(float), dy.astype(float))
        err = q_error(q, theta_true)
        deg = np.degrees(err)
        adeg = np.abs(deg)

        # c == g: q ~ 0 and theta_true is undefined. Reported as ||q|| only,
        # never as a 0-degree error and never as a NaN that silently averages.
        degenerate = d == 0
        adeg[degenerate] = np.nan
        deg_masked = np.where(degenerate, np.nan, deg)

        qn = np.linalg.norm(q, axis=1)

        self.abs_err.add(d, adeg)
        self.acc45.add(d, (adeg < 45.0).astype(float))
        self.acc90.add(d, (adeg < 90.0).astype(float))
        self.qnorm.add(d, qn)

        ix = (cells[:, 0] - goal[0]) + (size - 1)
        iy = (cells[:, 1] - goal[1]) + (size - 1)
        self.rel_signed.add(ix, iy, deg_masked)
        self.rel_abs.add(ix, iy, adeg)
        self.abs_map.add(cells[:, 0], cells[:, 1], adeg)

        sec = _sector(theta_true)
        ok = ~degenerate
        np.add.at(self.sector_signed, sec[ok], deg[ok])
        np.add.at(self.sector_n, sec[ok], 1)

        finite = np.isfinite(adeg)
        stats = {
            "abs_err_mean": float(np.nanmean(adeg)),
            "acc45": float(np.mean(adeg[finite] < 45.0)),
            "acc90": float(np.mean(adeg[finite] < 90.0)),
            "qnorm_mean": float(qn.mean()),
            "abs_err": adeg,
        }
        # Recorded here rather than by the caller: every caller wants them, and
        # a caller that forgets leaves an accumulator that silently reports an
        # empty spread instead of a wrong one.
        for key in ("abs_err_mean", "acc45", "acc90", "qnorm_mean"):
            self.scalars.add(key, stats[key])
        return stats

    def to_json(self) -> dict:
        with np.errstate(invalid="ignore", divide="ignore"):
            sec = np.where(self.sector_n > 0,
                           self.sector_signed / np.maximum(self.sector_n, 1),
                           np.nan)
        return {
            "abs_err": self.abs_err.to_json(),
            "acc45": self.acc45.to_json(),
            "acc90": self.acc90.to_json(),
            "qnorm": self.qnorm.to_json(),
            "map_goal_relative_signed": self.rel_signed.to_json(),
            "map_goal_relative_abs": self.rel_abs.to_json(),
            "map_env_absolute_abs": self.abs_map.to_json(),
            "sector_signed_deg": [None if not np.isfinite(v) else float(v)
                                  for v in sec],
            "sector_n": self.sector_n.tolist(),
            "scalars": self.scalars.to_json(),
        }


def _sector(theta: np.ndarray) -> np.ndarray:
    return np.clip(
        np.floor((theta + np.pi) / (2 * np.pi / N_SECTORS)).astype(int),
        0, N_SECTORS - 1)


# ---------------------------------------------------------------------------
# Test C accumulation
# ---------------------------------------------------------------------------

class ContinuousAcc:
    """Everything Test C reports, for one ``(K, steps)``."""

    def __init__(self, cfg: ProbeConfig):
        size = cfg.env_size
        edges = continuous_dist_edges(float(np.hypot(size, size)))
        self.abs_err = BinnedStat(edges, "angle_deg")
        self.err_geom = BinnedStat(edges, "angle_deg")
        self.excess = BinnedStat(edges, "excess_deg")
        self.acc45 = BinnedStat(edges, "frac")
        self.acc90 = BinnedStat(edges, "frac")
        n = cfg.subcell_bins
        self.subcell = Map2D((size * n, size * n))
        self.scalars = Scalars()
        self.cfg = cfg

    def add_env(self, q_cells: np.ndarray, p: np.ndarray,
                goal: tuple[int, int], cell_abs_err_deg: np.ndarray) -> dict:
        cfg = self.cfg
        size = cfg.env_size

        # The env's own snap, not a re-derivation of it: _pos = clip(round, ...).
        cx = np.clip(np.round(p[:, 0]), 0, size - 1).astype(np.int64)
        cy = np.clip(np.round(p[:, 1]), 0, size - 1).astype(np.int64)
        row = cx * size + cy
        q = q_cells[row]

        # True bearing from the CONTINUOUS position; q was read at the cell.
        theta_true = bearing(goal[0] - p[:, 0], goal[1] - p[:, 1])
        adeg = np.abs(np.degrees(q_error(q, theta_true)))
        d = np.sqrt((goal[0] - p[:, 0]) ** 2 + (goal[1] - p[:, 1]) ** 2)

        # The analytic snap ceiling: what a *perfect* readout at the snapped
        # cell would still get wrong, purely from being read one cell over.
        theta_cell = bearing((goal[0] - cx).astype(float),
                             (goal[1] - cy).astype(float))
        geom = np.abs(np.degrees(wrap_to_pi(theta_cell - theta_true)))

        # Snap-attributable error, per sample, with the Hopfield contribution
        # differenced out at the cell this sample actually snapped to. Can go
        # negative -- the snap sometimes helps -- and that is reported, not
        # clipped away.
        excess = adeg - cell_abs_err_deg[row]

        # A sample sitting in the goal cell has an undefined cell bearing when
        # the cell IS the goal; drop those from geom/excess, keep them in the
        # raw error, which is a real thing the agent experiences.
        at_goal_cell = (cx == goal[0]) & (cy == goal[1])
        geom = np.where(at_goal_cell, np.nan, geom)
        excess = np.where(at_goal_cell, np.nan, excess)

        self.abs_err.add(d, adeg)
        self.err_geom.add(d, geom)
        self.excess.add(d, excess)
        self.acc45.add(d, (adeg < 45.0).astype(float))
        self.acc90.add(d, (adeg < 90.0).astype(float))

        n = cfg.subcell_bins
        sx = np.clip(((p[:, 0] + 0.5) * n).astype(int), 0, size * n - 1)
        sy = np.clip(((p[:, 1] + 0.5) * n).astype(int), 0, size * n - 1)
        self.subcell.add(sx, sy, adeg)

        near = d < 2.0
        return {
            "abs_err_mean": float(np.nanmean(adeg)),
            "excess_near_mean": float(np.nanmean(excess[near]))
            if near.any() else float("nan"),
            "acc45": float(np.mean(adeg < 45.0)),
        }

    def to_json(self) -> dict:
        return {
            "abs_err": self.abs_err.to_json(),
            "err_geom": self.err_geom.to_json(),
            "excess": self.excess.to_json(),
            "acc45": self.acc45.to_json(),
            "acc90": self.acc90.to_json(),
            "map_subcell_abs": self.subcell.to_json(),
            "scalars": self.scalars.to_json(),
        }


def sample_continuous(
    cfg: ProbeConfig, goal: tuple[int, int], rng: np.random.RandomState,
) -> np.ndarray:
    """Uniform over the region the env clips to, plus a near-goal annulus.

    ``[-0.5, size-0.5]^2`` is exactly what ``clip(round(p), 0, size-1)`` maps
    onto the grid. The annulus exists because uniform-area sampling puts
    ``proportional to d`` mass per distance bin, so the near-goal bins that
    carry this test's headline would otherwise be its noisiest.
    """
    size = cfg.env_size
    lo, hi = -0.5, size - 0.5
    uni = rng.uniform(lo, hi, size=(cfg.n_cont_samples, 2))
    if cfg.n_cont_annulus <= 0:
        return uni
    r = cfg.annulus_radius * np.sqrt(rng.uniform(0, 1, cfg.n_cont_annulus))
    th = rng.uniform(0, 2 * np.pi, cfg.n_cont_annulus)
    ann = np.stack([goal[0] + r * np.cos(th), goal[1] + r * np.sin(th)], 1)
    ann = np.clip(ann, lo, hi)
    return np.concatenate([uni, ann], axis=0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_tests_bc(
    field: Field, worlds: list[World], cfg: ProbeConfig, *, progress=None,
) -> dict:
    """Tests B and C over every world, ``K`` and step count."""
    size = cfg.env_size
    cells = local_cells(size)
    out: dict = {
        "config": cfg.to_json(),
        "chance": {"abs_err_deg": CHANCE_ABS_DEG, "acc45": CHANCE_ACC45,
                   "acc90": CHANCE_ACC90},
        "k": {},
    }

    for k in cfg.k_values:
        gacc = {str(s): GridAcc(cfg) for s in cfg.steps}
        cacc = {str(s): ContinuousAcc(cfg) for s in cfg.steps}
        example_maps: list[dict] = []

        for w in worlds:
            rng = np.random.RandomState(w.seed * 17 + k)
            test_envs = scored_envs(cfg, k)
            mem = build_memory(field, w, k, cfg, rng)

            for j, e in enumerate(test_envs):
                if cfg.memory_mode != "multi_env_goals" and j > 0:
                    break
                qf, _cues, _basis = cell_q_field(field, w, e, mem, cfg)
                goal = w.specs[e].goal
                p = sample_continuous(cfg, goal, rng)

                for s in cfg.steps:
                    stats_b = gacc[str(s)].add_env(qf[s], cells, goal)
                    stats_c = cacc[str(s)].add_env(
                        qf[s], p, goal, stats_b["abs_err"])
                    cacc[str(s)].scalars.add("abs_err_mean",
                                             stats_c["abs_err_mean"])
                    cacc[str(s)].scalars.add("excess_near_mean",
                                             stats_c["excess_near_mean"])
                    cacc[str(s)].scalars.add("acc45", stats_c["acc45"])

                if w.index < cfg.n_map_worlds and j < cfg.n_map_envs:
                    s0 = cfg.steps[0]
                    dx = goal[0] - cells[:, 0]
                    dy = goal[1] - cells[:, 1]
                    d = np.sqrt((dx ** 2 + dy ** 2).astype(float))
                    err = np.degrees(q_error(
                        qf[s0], bearing(dx.astype(float), dy.astype(float))))
                    err[d == 0] = np.nan
                    example_maps.append({
                        "world": w.index, "env": e, "steps": int(s0),
                        "size": size, "goal": [int(goal[0]), int(goal[1])],
                        "signed_err_deg": [None if not np.isfinite(v)
                                           else float(v) for v in err],
                        "q": qf[s0].astype(float).tolist(),
                    })

            if progress:
                progress(f"BC k={k:>3} world={w.index}")

        out["k"][str(k)] = {
            "per_step": {
                str(s): {"grid": gacc[str(s)].to_json(),
                         "continuous": cacc[str(s)].to_json()}
                for s in cfg.steps
            },
            "example_maps": example_maps,
        }
    return out


__all__ = [
    "CHANCE_ABS_DEG", "CHANCE_ACC45", "CHANCE_ACC90", "ContinuousAcc",
    "GridAcc", "bearing", "cell_q_field", "project_q", "q_error",
    "run_tests_bc", "sample_continuous",
]
