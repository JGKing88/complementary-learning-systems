"""Grid environments for Hopfield navigation.

Env manages positions, codebook, stepping, goal detection.
No VectorHash recall, no encoding, no Hopfield — those are in the rollout collector.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..config import EnvConfig

CARDINAL_ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W

# Forward foveal cone: 120° total, centered on the agent's heading. Rays are
# evenly spaced in angle; θ is measured clockwise from forward.
FOVEAL_HALF_ANGLE_DEG = 60.0

# ---------------------------------------------------------------------------
# Heading
# ---------------------------------------------------------------------------
# Heading is a single angle ψ, radians, measured **clockwise from North**, so
# forward is (sin ψ, cos ψ). That is the same convention the ray angles already
# use, and it is the whole reason this is cheap: rotating the cone to face ψ is
# adding ψ to every ray angle. ψ = 0 is North, which is where the cone was
# hard-wired before headings existed -- so a fixed-heading env and an egocentric
# one see the same thing on step zero.
#
# Movement stays allocentric: actions are world-frame, and heading only ever
# follows from where the agent actually went.

# ψ for each entry of CARDINAL_ACTIONS. Discrete movement sets heading from this
# table rather than through atan2, so ψ lands *exactly* on a multiple of π/2 and
# `cardinal_index` below always resolves it -- which is what keeps discrete
# rollouts on the precomputed-codebook path instead of ray-casting every step.
CARDINAL_RADIANS = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float64)

_HEADING_VECTORS = np.array(CARDINAL_ACTIONS, dtype=np.float64)  # (4, 2)
N_HEADINGS = len(CARDINAL_ACTIONS)


def nearest_heading(disp):
    """Index of the cardinal a displacement points along. ``-1`` if it is still.

    A discrete cardinal step already *is* a heading, so this is exact there. A
    continuous step is not, and resolves to the cardinal it is most aligned
    with; exact diagonals tie to the lower index.

    ``-1`` means "did not move, keep the heading you had". Pass the *realized*
    (post-clip) displacement, not the requested action: that is what leaves the
    view unchanged when a step is clipped by a wall, in both movement modes.

    disp: (2,) -> Python int, or (N, 2) -> (N,) int64.
    """
    d = np.asarray(disp, dtype=np.float64)
    single = d.ndim == 1
    d = d.reshape(-1, 2)
    idx = (d @ _HEADING_VECTORS.T).argmax(axis=1)
    idx[np.linalg.norm(d, axis=1) < 1e-12] = -1
    return int(idx[0]) if single else idx


def cardinal_index(psi):
    """Which cardinal ψ sits exactly on, or ``-1`` for anything in between.

    ``k % 4`` *is* the ``CARDINAL_ACTIONS`` index: ψ=0 -> N=0, π/2 -> E=1,
    π -> S=2, -π/2 -> W=3, and the modulo absorbs ``atan2``'s (-π, π] range.

    Callers use this to decide whether an observation can be read straight out
    of the precomputed cardinal codebook instead of being ray-cast.

    psi: scalar -> Python int, or (N,) -> (N,) int64.
    """
    p = np.asarray(psi, dtype=np.float64)
    single = p.ndim == 0
    q = p.reshape(-1) / (np.pi / 2)
    k = np.rint(q)
    idx = np.where(np.abs(q - k) < 1e-12, k.astype(np.int64) % 4, -1)
    return int(idx[0]) if single else idx


def cone_offsets(n_rays: int) -> np.ndarray:
    """Ray angles within the cone, relative to forward: (n_rays,).

    Bin-centered: θ_i = -half + (i + 0.5) * (2*half / n_rays).
    """
    half = np.deg2rad(FOVEAL_HALF_ANGLE_DEG)
    return -half + (np.arange(n_rays) + 0.5) * (2 * half / n_rays)


def raycast_codes(wall_code: np.ndarray, size: int, xs, ys, psi,
                  n_rays: int, resolution: int = 1) -> np.ndarray:
    """Foveal view from each (x, y) facing each ψ: ``(N, n_rays)`` of ±1 codes.

    The vectorized form of the four plane intersections ``_raycast_segment_code``
    does one ray at a time. Walls live at half-integer planes x=-0.5, x=size-0.5,
    y=-0.5, y=size-0.5; wall order is 0=N, 1=E, 2=S, 3=W.

    ``resolution`` is how many wall segments span one grid cell, so ``wall_code``
    is ``(4, size * resolution)``. At 1 -- the default -- segment k is the unit
    interval centered on cell-index k, which is what the walls were before this
    was a parameter. Above 1, a segment boundary can fall *inside* a cell, which
    is the point: it is what lets two positions within the same cell read
    differently. See ``EnvConfig.wall_resolution``.

    Ties go to the earlier wall, matching the scalar version's ``t < best_t``
    scan in N, E, S, W order (``argmin`` returns the first minimum). A ray that
    hits nothing reads 0.0, as it did before -- unreachable from inside the box,
    but the box is not re-derived here.

    xs, ys, psi: (N,) broadcastable. Returns float32.
    """
    xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
    ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
    psi = np.atleast_1d(np.asarray(psi, dtype=np.float64))
    xs, ys, psi = np.broadcast_arrays(xs, ys, psi)

    # Rotating the cone to face ψ is adding ψ to every ray angle -- the one
    # place the clockwise-from-North convention pays for itself.
    angles = psi[:, None] + cone_offsets(n_rays)[None, :]        # (N, n_rays)
    dx, dy = np.sin(angles), np.cos(angles)
    cx, cy = xs[:, None], ys[:, None]

    hi = size - 0.5
    inf = np.inf

    def _plane(num, den, keep):
        """t along the ray to a plane, +inf where the ray cannot reach it.

        ``num`` is per-position (N, 1) and ``den`` per-ray (N, n_rays), so the
        result has to be allocated at the broadcast shape, not ``num``'s.
        """
        t = np.full(np.broadcast_shapes(num.shape, den.shape), inf,
                    dtype=np.float64)
        np.divide(num, den, out=t, where=keep)
        t[~keep | (t < 0.0)] = inf
        return t

    # Each wall: distance to its plane, then the coordinate the ray hits it at.
    t_n = _plane(hi - cy, dy, dy > 0.0)          # N: y = size-0.5
    t_e = _plane(hi - cx, dx, dx > 0.0)          # E: x = size-0.5
    t_s = _plane(-0.5 - cy, dy, dy < 0.0)        # S: y = -0.5
    t_w = _plane(-0.5 - cx, dx, dx < 0.0)        # W: x = -0.5

    hit_n = cx + t_n * dx
    hit_e = cy + t_e * dy
    hit_s = cx + t_s * dx
    hit_w = cy + t_w * dy

    # A plane the ray reaches outside the wall's extent is not a hit.
    for t, h in ((t_n, hit_n), (t_e, hit_e), (t_s, hit_s), (t_w, hit_w)):
        t[np.isfinite(t) & ((h < -0.5) | (h > hi))] = inf

    ts = np.stack([t_n, t_e, t_s, t_w], axis=-1)          # (N, n_rays, 4)
    hits = np.stack([hit_n, hit_e, hit_s, hit_w], axis=-1)
    wall = ts.argmin(axis=-1)                             # first min == earlier wall

    h = np.take_along_axis(hits, wall[..., None], axis=-1)[..., 0]
    # Scale the *continuous* hit coordinate before quantising. Quantising to a
    # cell first and multiplying up would only replicate the coarse code, which
    # is the whole thing resolution exists to avoid.
    fine = (np.where(np.isfinite(h), h, 0.0) + 0.5) * resolution
    seg = np.clip(np.floor(fine), 0, size * resolution - 1).astype(np.int64)

    codes = wall_code[wall, seg].astype(np.float32)
    return np.where(np.isfinite(ts).any(axis=-1), codes, 0.0).astype(np.float32)


def at_goal(env):
    """Is the agent within ``env.goal_radius`` (L2) of ``env._goal``?

    Single canonical at-goal predicate. Does NOT consult ``goals_active``
    — that gating stays at the call site.

    Always uses the env's *actual* position:
      - ContinuousVecEnv  → env._pos_f          (B, 2) float, returns ndarray[bool] (B,)
      - ContinuousGridEnv → env._continuous_pos (2,)   float, returns Python bool
      - VecEnv            → env._pos            (B, 2) int,   returns ndarray[bool] (B,)
      - GridEnv           → env._pos            (2,)   int,   returns Python bool

    ``goal`` and ``radius`` come from the env. The helper deliberately does
    not accept a raw position: in continuous mode, ``env.current_location``
    is the snap and silently using it would re-introduce the snap-square
    vs. L2-ball mismatch this helper exists to prevent.

    For raw (pos, goal, radius) checks (tests / non-env code) use
    :func:`_at_goal_l2` directly.
    """
    if hasattr(env, "_pos_f"):              # ContinuousVecEnv
        pos = env._pos_f
    elif hasattr(env, "_continuous_pos"):   # ContinuousGridEnv
        pos = env._continuous_pos
    else:                                   # GridEnv / VecEnv
        pos = env._pos
    return _at_goal_l2(pos, env._goal, env.goal_radius)


def _at_goal_l2(pos, goal, radius: float = 0.5):
    """L2 ball predicate on raw positions.

    pos:    (2,) or (B, 2) — int or float
    goal:   (gx, gy)
    radius: L2 distance threshold; inclusive (distance == radius counts).
    Returns: Python bool for (2,) input; ndarray[bool] (B,) for (B, 2).
    """
    pos_arr = np.asarray(pos)
    goal_arr = np.asarray(goal)
    r2 = float(radius) * float(radius)
    if pos_arr.ndim == 1:
        dx = float(pos_arr[0]) - float(goal_arr[0])
        dy = float(pos_arr[1]) - float(goal_arr[1])
        return bool(dx * dx + dy * dy <= r2)
    if pos_arr.ndim == 2:
        d2 = (pos_arr[:, 0] - goal_arr[0]) ** 2 + (pos_arr[:, 1] - goal_arr[1]) ** 2
        return d2 <= r2
    raise ValueError(f"pos must be (2,) or (B, 2), got shape {pos_arr.shape}")


def max_offcell_offset(goal_radius: float) -> int:
    """Largest per-axis cell offset between an at-goal position and the goal.

    ``at_goal`` is an L2 ball of radius ``goal_radius`` around the goal in float
    coordinates, but embeddings and sensory input are read at the *snapped*
    cell. A position lands on the cell ``k`` steps away along an axis once it
    reaches ``goal + k - 0.5``, so offset ``k`` is reachable when
    ``k - 0.5 < goal_radius``; the largest such ``k`` is
    ``ceil(goal_radius + 0.5) - 1``.

    0 means every at-goal position snaps to the goal cell itself. That is the
    case at the default radius 0.5, where the only positions that would snap
    elsewhere sit exactly on the boundary (measure zero, and resolved by
    round-half-to-even in favour of the goal cell).

        radius 0.5 -> 0     radius 1.0 -> 1     radius 2.0 -> 2
    """
    return int(np.ceil(float(goal_radius) + 0.5)) - 1


def warn_if_offcell_stores(env_cfg, *, where: str = "") -> None:
    """Report the at-goal store policy when the radius makes it observable.

    Silent unless ``goal_radius`` is large enough that an at-goal position can
    snap to some cell other than the goal's -- below that, the two policies are
    the same thing. Above it, warn when off-cell stores are allowed (memory
    receives a neighbour's embedding, which is then what navigation recalls) and
    note it when they are suppressed (the substitution is active and worth
    knowing about).
    """
    offset = max_offcell_offset(env_cfg.goal_radius)
    if offset == 0:
        return
    prefix = f"[{where}] " if where else ""
    if getattr(env_cfg, "allow_offcell_store", False):
        print(
            f"{prefix}WARNING: goal_radius={env_cfg.goal_radius} with "
            f"allow_offcell_store=True — a store fired at goal can write the "
            f"embedding of a cell up to {offset} cell(s) away from the goal "
            f"along either axis. Drop --allow_offcell_store to store the goal "
            f"cell's embedding instead."
        )
    else:
        print(
            f"{prefix}note: goal_radius={env_cfg.goal_radius} puts at-goal "
            f"positions up to {offset} cell(s) from the goal; stores fired "
            f"there write the goal cell's embedding, not the cell the agent "
            f"is standing on (allow_offcell_store=False)."
        )


class EnvState(NamedTuple):
    position: tuple[int, int]
    goal: tuple[int, int]
    obs: np.ndarray
    reward: float


class GridEnv:
    """Discrete grid environment with binary codebook observations.

    Goal stays fixed.  On goal-reach: call reset_position() to teleport.
    """

    def __init__(
        self,
        size: int,
        speed: int = 1,
        observation_size: int = 60,
        seed: int | None = None,
        time_penalty: float = 0.01,
        goals_active: bool = True,
        goal_reward: float = 1.0,
        goal_radius: float = 0.5,
        egocentric_heading: bool = True,
        wall_resolution: int = 1,
    ) -> None:
        self.size = size
        if int(wall_resolution) < 1:
            raise ValueError(
                f"wall_resolution must be >= 1, got {wall_resolution}")
        self.wall_resolution = int(wall_resolution)
        self.speed = speed
        self._observation_size = observation_size
        self.time_penalty = time_penalty
        self.goals_active = goals_active
        self.goal_reward = goal_reward
        self.goal_radius = goal_radius
        # False pins every observation to ψ=0 (North), reproducing the fixed
        # cone this env had before headings were wired up. Heading is still
        # tracked either way -- it just isn't seen.
        self.egocentric_heading = egocentric_heading
        # Kept so an env can say which seed built it. The wall code is a pure
        # function of (seed, size), so this is what lets a world be *recorded*
        # rather than replayed -- see docs/EVAL_SPLITS_DESIGN.md §1.4.
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Wall bar code: 4 walls (N, E, S, W), `size * wall_resolution` ±1
        # segments each. Walls sit at half-integer boundaries (y=size-0.5 N,
        # x=size-0.5 E, y=-0.5 S, x=-0.5 W).
        #
        # At wall_resolution=1 there is one segment per grid cell, aligned with
        # cell index k along that wall -- the original coarse barcode. Above 1
        # the segments subdivide each cell, so a stripe boundary can fall inside
        # a cell rather than only on its edge. That is what gives a ray
        # information about *where within* a cell it is looking from, which a
        # cell-aligned code cannot carry at all.
        self._wall_code = self.rng.choice(
            [-1.0, 1.0], size=(4, size * self.wall_resolution)
        ).astype(np.float32)

        # Cardinal sensory codebook: _codebook[x, y, h] is the foveal view from
        # cell (x, y) facing CARDINAL_ACTIONS[h] -- observation_size rays evenly
        # spaced in the cone, each outputting the ±1 code of the wall segment it
        # hits. No zeros (every ray hits a wall).
        #
        # This is NOT what the agent observes: heading is continuous, so a live
        # observation is ray-cast at the current ψ (see `obs_at`). The table
        # survives as the canonical per-cell artifact -- the scaffold's sbook
        # (world/scaffold.py) and the generator's env-identity check
        # (world/generate.py) are both defined on the four cardinal views -- and
        # as the fast path whenever ψ happens to be exactly cardinal, which in
        # discrete movement it always is.
        self._codebook = self._build_sensory_codebook(observation_size)

        # Pick random goal and start
        self._goal = self._random_position()
        self._pos = self._random_position(exclude=self._goal)
        self._heading_rad = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def goal_location(self) -> tuple[int, int]:
        return self._goal

    @property
    def current_location(self) -> tuple[int, int]:
        return self._pos

    @property
    def heading(self) -> float:
        """Facing, radians clockwise from North. Forward is (sin ψ, cos ψ)."""
        return float(self._heading_rad)

    @property
    def heading_vector(self) -> tuple[float, float]:
        """Facing as a unit (dx, dy)."""
        return (float(np.sin(self._heading_rad)), float(np.cos(self._heading_rad)))

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def obs(self) -> np.ndarray:
        """Current observation (observation_size,), from where the agent faces."""
        return self.obs_at(self.current_location)

    def obs_at(self, pos: tuple[int, int], psi: float | None = None) -> np.ndarray:
        """Foveal view from ``pos``, facing ``psi`` (default: the env's heading).

        Reads the precomputed table when ψ is exactly cardinal and ray-casts
        otherwise. The two agree by construction -- the table is built by the
        same function -- so this is a speed choice, not a behavioral one, and it
        keeps discrete rollouts on a pure array gather.
        """
        psi = self._obs_heading() if psi is None else float(psi)
        k = cardinal_index(psi)
        if k >= 0:
            return self._codebook[pos[0], pos[1], k].copy()
        return raycast_codes(self._wall_code, self.size, pos[0], pos[1], psi,
                             self._observation_size, self.wall_resolution)[0]

    def omni_obs_at(self, pos: tuple[int, int]) -> np.ndarray:
        """All four cardinal views from ``pos``, concatenated: (4*obs_size,).

        The scaffold's stand-in for a heading-invariant observation -- see
        ``world/scaffold.fit_env_assoc``, which explains why it is a patch.
        """
        return self._codebook[pos[0], pos[1]].reshape(-1).copy()

    def omni_obs_all(self) -> np.ndarray:
        """``omni_obs_at`` for every cell: (size, size, 4*obs_size)."""
        return self._codebook.reshape(self.size, self.size, -1).copy()

    def _obs_heading(self) -> float:
        """The ψ observations are read at: the live heading, or North if fixed."""
        return float(self._heading_rad) if self.egocentric_heading else 0.0

    def step(self, action: tuple[int, int]) -> EnvState:
        """Take a cardinal action (dx, dy).  Clips to grid bounds."""
        dx, dy = action[0] * self.speed, action[1] * self.speed
        nx = max(0, min(self.size - 1, self._pos[0] + dx))
        ny = max(0, min(self.size - 1, self._pos[1] + dy))
        # Heading follows the *realized* displacement, so a step clipped by a
        # wall leaves the agent facing where it already was. Cardinal moves take
        # ψ from the table, keeping it exactly on a multiple of π/2.
        k = nearest_heading((nx - self._pos[0], ny - self._pos[1]))
        if k >= 0:
            self._heading_rad = float(CARDINAL_RADIANS[k])
        self._pos = (nx, ny)
        return EnvState(self._pos, self._goal, self.obs(), self.reward())

    def reward(self) -> float:
        if self.goals_active and at_goal(self):
            return self.goal_reward
        return -self.time_penalty

    def reset(self) -> EnvState:
        """Reset position (keep goal fixed)."""
        self._pos = self._random_position(exclude=self._goal)
        self._heading_rad = 0.0
        return EnvState(self._pos, self._goal, self.obs(), self.reward())

    def set_position(self, pos: tuple[int, int]) -> None:
        """Place the agent at an externally-chosen cell.

        Used by eval to place the agent at a seeded random start without
        going through reset() (which samples from the env's internal RNG).
        Goal, codebook, and RNG are untouched.
        """
        x, y = int(pos[0]), int(pos[1])
        if not (0 <= x < self.size and 0 <= y < self.size):
            raise ValueError(f"position {(x, y)} out of bounds for size {self.size}")
        self._pos = (x, y)
        self._heading_rad = 0.0

    def reset_goal(self) -> None:
        """Pick a new random goal."""
        self._goal = self._random_position(exclude=self._pos)

    def set_goal(self, pos: tuple[int, int]) -> None:
        """Place the goal at an externally-chosen cell.

        Mirror of ``set_position``. The generator resolves goals from a declared
        domain, so the env has to accept one rather than draw its own. Nothing
        else is touched -- in particular the env's RNG is not consumed, so the
        goal drawn in ``__init__`` simply becomes dead entropy and ``_wall_code``
        stays bit-identical for a given seed.
        """
        x, y = int(pos[0]), int(pos[1])
        if not (0 <= x < self.size and 0 <= y < self.size):
            raise ValueError(f"goal {(x, y)} out of bounds for size {self.size}")
        self._goal = (x, y)

    def reset_position(self) -> EnvState:
        """Teleport to a random position (goal stays fixed)."""
        return self.reset()

    def best_action_to_goal(self, randomize: bool = False) -> tuple[int, int]:
        """Greedy best cardinal action toward goal."""
        dx = self._goal[0] - self._pos[0]
        dy = self._goal[1] - self._pos[1]
        candidates = []
        for a in CARDINAL_ACTIONS:
            nx = max(0, min(self.size - 1, self._pos[0] + a[0] * self.speed))
            ny = max(0, min(self.size - 1, self._pos[1] + a[1] * self.speed))
            dist = abs(nx - self._goal[0]) + abs(ny - self._goal[1])
            candidates.append((dist, a))
        candidates.sort(key=lambda x: x[0])
        best_dist = candidates[0][0]
        best = [c for c in candidates if c[0] == best_dist]
        if randomize:
            return best[self.rng.randint(len(best))][1]
        return best[0][1]

    # ------------------------------------------------------------------
    # Sensory (foveal ray-cast) codebook
    # ------------------------------------------------------------------

    def _build_sensory_codebook(self, n_rays: int) -> np.ndarray:
        """Precompute the (size, size, 4, n_rays) foveal view for every cell.

        One slab per cardinal heading. The cone spans
        θ ∈ [-FOVEAL_HALF_ANGLE_DEG, +FOVEAL_HALF_ANGLE_DEG] clockwise from
        forward, rays centered in equal-angle bins. Each ray traces from the
        cell center, intersects the nearest in-range wall, and reads that wall
        segment's ±1 code from ``self._wall_code``.
        """
        size = self.size
        gx, gy = np.meshgrid(np.arange(size, dtype=np.float64),
                             np.arange(size, dtype=np.float64), indexing="ij")
        # Flattened as [x][y][heading], matching the reshape below.
        xs = np.repeat(gx.ravel(), N_HEADINGS)
        ys = np.repeat(gy.ravel(), N_HEADINGS)
        psi = np.tile(CARDINAL_RADIANS, size * size)
        codes = raycast_codes(self._wall_code, size, xs, ys, psi, n_rays,
                              self.wall_resolution)
        return codes.reshape(size, size, N_HEADINGS, n_rays)

    def fully_explore_random(self) -> list[tuple[tuple[int, int], np.ndarray, tuple[int, int]]]:
        """Visit all positions with all 4 headings in random order.

        Returns list of (position, obs, heading) tuples. The four entries for a
        cell are four genuinely different views -- before headings were wired
        up they were four copies of the same North-facing one, which is why
        callers that wanted one view per cell used to filter on ``heading``.
        Use ``omni_obs_at`` for a single heading-free vector per cell instead.
        """
        items = []
        for x in range(self.size):
            for y in range(self.size):
                for h, a in enumerate(CARDINAL_ACTIONS):
                    items.append(((x, y), self._codebook[x, y, h].copy(), a))
        self.rng.shuffle(items)
        return items

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _random_position(self, exclude: tuple[int, int] | None = None) -> tuple[int, int]:
        while True:
            p = (int(self.rng.randint(0, self.size)), int(self.rng.randint(0, self.size)))
            if p != exclude:
                return p


class ContinuousGridEnv(GridEnv):
    """Grid env with floating-point positions.  Snaps to integer for obs lookup."""

    def __init__(self, *args, scale: float = 1.0, normalize: bool = True,
                 max_action_norm: float | None = None,
                 min_action_norm: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._continuous_pos = np.array(self._pos, dtype=np.float64)
        self.scale = scale
        self.normalize_step = normalize
        self.max_action_norm = max_action_norm
        self.min_action_norm = min_action_norm

    @property
    def current_location(self) -> tuple[int, int]:
        snapped = np.clip(np.round(self._continuous_pos), 0, self.size - 1).astype(int)
        return (int(snapped[0]), int(snapped[1]))

    def step(self, action: np.ndarray) -> EnvState:
        """Continuous action (dx, dy) as float array."""
        a = np.asarray(action, dtype=np.float64)
        if self.normalize_step and np.linalg.norm(a) > 1e-8:
            a = a / np.linalg.norm(a)
        else:
            n = np.linalg.norm(a)
            if self.max_action_norm is not None and n > self.max_action_norm:
                a = a * (self.max_action_norm / n)
            elif self.min_action_norm is not None and 1e-8 < n < self.min_action_norm:
                a = a * (self.min_action_norm / n)
        before = self._continuous_pos
        self._continuous_pos = np.clip(
            self._continuous_pos + a * self.scale,
            0, self.size - 1,
        )
        # Heading is the direction actually travelled, so a step absorbed by the
        # clip leaves the agent facing where it was. Continuous movement can face
        # any angle, so ψ comes from atan2 rather than the cardinal table --
        # atan2(dx, dy), in that argument order, is clockwise from North.
        disp = self._continuous_pos - before
        if float(np.linalg.norm(disp)) >= 1e-12:
            self._heading_rad = float(np.arctan2(disp[0], disp[1]))
        self._pos = self.current_location
        return EnvState(self._pos, self._goal, self.obs(), self.reward())

    def reset(self) -> EnvState:
        state = super().reset()
        self._continuous_pos = np.array(self._pos, dtype=np.float64)
        return state

    def set_position(self, pos: tuple[int, int]) -> None:
        super().set_position(pos)
        self._continuous_pos = np.array(self._pos, dtype=np.float64)

    def oracle_unit_toward_goal(self) -> np.ndarray:
        """Unit (dx, dy) toward the goal in grid coordinates; used by eval action oracle."""
        g = np.array(self._goal, dtype=np.float64)
        v = g - self._continuous_pos
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return np.array([0.0, 0.0], dtype=np.float32)
        return (v / n).astype(np.float32)


def make_env(env_cfg: EnvConfig, movement_mode: str, seed: int) -> GridEnv:
    """Single-env factory: picks GridEnv or ContinuousGridEnv based on movement_mode.

    Used by eval to build val envs that can natively handle the agent's action
    format via env.step(...) — no hand-rolled movement math needed.
    """
    if movement_mode == "continuous":
        return ContinuousGridEnv(
            size=env_cfg.size,
            speed=env_cfg.speed,
            observation_size=env_cfg.observation_size,
            seed=seed,
            time_penalty=env_cfg.time_penalty,
            scale=env_cfg.continuous_scale,
            normalize=env_cfg.continuous_normalize,
            max_action_norm=env_cfg.max_action_norm,
            min_action_norm=env_cfg.min_action_norm,
            goals_active=env_cfg.goals_active,
            goal_reward=env_cfg.goal_reward,
            goal_radius=env_cfg.goal_radius,
            egocentric_heading=env_cfg.egocentric_heading,
            wall_resolution=env_cfg.wall_resolution,
        )
    return GridEnv(
        size=env_cfg.size,
        speed=env_cfg.speed,
        observation_size=env_cfg.observation_size,
        seed=seed,
        time_penalty=env_cfg.time_penalty,
        goals_active=env_cfg.goals_active,
        goal_reward=env_cfg.goal_reward,
        goal_radius=env_cfg.goal_radius,
        egocentric_heading=env_cfg.egocentric_heading,
        wall_resolution=env_cfg.wall_resolution,
    )
