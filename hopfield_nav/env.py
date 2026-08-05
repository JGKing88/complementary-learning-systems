"""Grid environments for Hopfield navigation.

Env manages positions, codebook, stepping, goal detection.
No VectorHash recall, no encoding, no Hopfield — those are in the rollout collector.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .config import EnvConfig

CARDINAL_ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W

# Forward foveal cone (fixed North for now): 120° total, centered on +y.
# Rays are evenly spaced in angle; θ clockwise from North (sin θ, cos θ).
FOVEAL_HALF_ANGLE_DEG = 60.0


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
    ) -> None:
        self.size = size
        self.speed = speed
        self._observation_size = observation_size
        self.time_penalty = time_penalty
        self.goals_active = goals_active
        self.goal_reward = goal_reward
        self.goal_radius = goal_radius
        self.rng = np.random.RandomState(seed)

        # Wall bar code: 4 walls (N, E, S, W), `size` ±1 segments each, one
        # segment per grid cell along the wall. Walls sit at half-integer
        # boundaries (y=size-0.5 N, x=size-0.5 E, y=-0.5 S, x=-0.5 W); segment
        # k on each wall is aligned with cell index k along that wall.
        self._wall_code = self.rng.choice(
            [-1.0, 1.0], size=(4, size)
        ).astype(np.float32)

        # Sensory codebook: per-cell foveal view (120° cone, heading=North).
        # _codebook[x, y] is the vector the agent sees from cell (x, y):
        # observation_size rays evenly spaced in the cone, each outputs the
        # ±1 code of the wall segment it hits. No zeros (every ray hits a wall).
        self._codebook = self._build_sensory_codebook(observation_size)

        # Pick random goal and start
        self._goal = self._random_position()
        self._pos = self._random_position(exclude=self._goal)
        self._heading = (1, 0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def goal_location(self) -> tuple[int, int]:
        return self._goal

    @property
    def current_location(self) -> tuple[int, int]:
        return self._pos

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def obs(self) -> np.ndarray:
        """Current observation (observation_size,)."""
        return self._codebook[self._pos[0], self._pos[1]].copy()

    def obs_at(self, pos: tuple[int, int]) -> np.ndarray:
        return self._codebook[pos[0], pos[1]].copy()

    def step(self, action: tuple[int, int]) -> EnvState:
        """Take a cardinal action (dx, dy).  Clips to grid bounds."""
        dx, dy = action[0] * self.speed, action[1] * self.speed
        nx = max(0, min(self.size - 1, self._pos[0] + dx))
        ny = max(0, min(self.size - 1, self._pos[1] + dy))
        if (nx, ny) != self._pos:
            self._heading = (np.sign(nx - self._pos[0]), np.sign(ny - self._pos[1]))
        self._pos = (nx, ny)
        return EnvState(self._pos, self._goal, self.obs(), self.reward())

    def reward(self) -> float:
        if self.goals_active and at_goal(self):
            return self.goal_reward
        return -self.time_penalty

    def reset(self) -> EnvState:
        """Reset position (keep goal fixed)."""
        self._pos = self._random_position(exclude=self._goal)
        self._heading = (1, 0)
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
        self._heading = (1, 0)

    def reset_goal(self) -> None:
        """Pick a new random goal."""
        self._goal = self._random_position(exclude=self._pos)

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
        """Precompute (size, size, n_rays) foveal view for every cell.

        Heading is fixed North (+y). Cone spans θ ∈ [-FOVEAL_HALF_ANGLE_DEG,
        +FOVEAL_HALF_ANGLE_DEG] clockwise from North, with rays centered in
        equal-angle bins. Each ray: trace from cell center (x, y) along
        (sin θ, cos θ); intersect with the nearest in-range wall; read that
        wall segment's ±1 code from ``self._wall_code``.
        """
        size = self.size
        half = np.deg2rad(FOVEAL_HALF_ANGLE_DEG)
        # Bin-centered angles: θ_i = -half + (i + 0.5) * (2*half / n_rays).
        angles = -half + (np.arange(n_rays) + 0.5) * (2 * half / n_rays)
        sin_t = np.sin(angles).astype(np.float64)
        cos_t = np.cos(angles).astype(np.float64)

        codebook = np.zeros((size, size, n_rays), dtype=np.float32)
        for cx in range(size):
            for cy in range(size):
                for i in range(n_rays):
                    codebook[cx, cy, i] = self._raycast_segment_code(
                        cx, cy, sin_t[i], cos_t[i]
                    )
        return codebook

    def _raycast_segment_code(
        self, cx: float, cy: float, dx: float, dy: float,
    ) -> float:
        """Cast a ray from (cx, cy) in direction (dx, dy) and return the ±1
        code of the first wall segment hit.

        Walls live at half-integer planes x=-0.5, x=size-0.5, y=-0.5, y=size-0.5.
        Wall index order: 0=N, 1=E, 2=S, 3=W. Segment k on each wall is the
        unit interval centered on cell-index k.
        """
        size = self.size
        best_t = np.inf
        best_wall = -1
        best_seg = 0

        # Wall 0 (N): y = size - 0.5
        if dy > 0.0:
            t = (size - 0.5 - cy) / dy
            if 0.0 <= t < best_t:
                x_hit = cx + t * dx
                if -0.5 <= x_hit <= size - 0.5:
                    best_t, best_wall = t, 0
                    best_seg = int(np.clip(np.floor(x_hit + 0.5), 0, size - 1))
        # Wall 1 (E): x = size - 0.5
        if dx > 0.0:
            t = (size - 0.5 - cx) / dx
            if 0.0 <= t < best_t:
                y_hit = cy + t * dy
                if -0.5 <= y_hit <= size - 0.5:
                    best_t, best_wall = t, 1
                    best_seg = int(np.clip(np.floor(y_hit + 0.5), 0, size - 1))
        # Wall 2 (S): y = -0.5
        if dy < 0.0:
            t = (-0.5 - cy) / dy
            if 0.0 <= t < best_t:
                x_hit = cx + t * dx
                if -0.5 <= x_hit <= size - 0.5:
                    best_t, best_wall = t, 2
                    best_seg = int(np.clip(np.floor(x_hit + 0.5), 0, size - 1))
        # Wall 3 (W): x = -0.5
        if dx < 0.0:
            t = (-0.5 - cx) / dx
            if 0.0 <= t < best_t:
                y_hit = cy + t * dy
                if -0.5 <= y_hit <= size - 0.5:
                    best_t, best_wall = t, 3
                    best_seg = int(np.clip(np.floor(y_hit + 0.5), 0, size - 1))

        if best_wall < 0:
            return 0.0
        return float(self._wall_code[best_wall, best_seg])

    def fully_explore_random(self) -> list[tuple[tuple[int, int], np.ndarray, tuple[int, int]]]:
        """Visit all positions with all 4 headings in random order.

        Returns list of (position, obs, heading) tuples.
        """
        items = []
        for x in range(self.size):
            for y in range(self.size):
                for h in CARDINAL_ACTIONS:
                    items.append(((x, y), self._codebook[x, y].copy(), h))
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

    def obs(self) -> np.ndarray:
        pos = self.current_location
        return self._codebook[pos[0], pos[1]].copy()

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
        self._continuous_pos = np.clip(
            self._continuous_pos + a * self.scale,
            0, self.size - 1,
        )
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
    )
