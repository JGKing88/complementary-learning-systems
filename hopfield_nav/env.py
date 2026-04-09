"""Grid environments for Hopfield navigation.

Env manages positions, codebook, stepping, goal detection.
No VectorHash recall, no encoding, no Hopfield — those are in the rollout collector.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .config import EnvConfig

CARDINAL_ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W


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
        observation_size: int = 512,
        seed: int | None = None,
        time_penalty: float = 0.01,
    ) -> None:
        self.size = size
        self.speed = speed
        self._observation_size = observation_size
        self.time_penalty = time_penalty
        self.rng = np.random.RandomState(seed)

        # Generate binary codebook: one obs per position (heading-invariant)
        self._codebook = self.rng.randint(
            0, 2, size=(size, size, observation_size)
        ).astype(np.float32)

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
        return 1.0 if self._pos == self._goal else -self.time_penalty

    def reset(self) -> EnvState:
        """Reset position (keep goal fixed)."""
        self._pos = self._random_position(exclude=self._goal)
        self._heading = (1, 0)
        return EnvState(self._pos, self._goal, self.obs(), self.reward())

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

    def clone(self) -> GridEnv:
        """Shallow clone sharing codebook but with independent state."""
        new = GridEnv.__new__(GridEnv)
        new.size = self.size
        new.speed = self.speed
        new._observation_size = self._observation_size
        new.time_penalty = self.time_penalty
        new.rng = np.random.RandomState(self.rng.randint(0, 2**31))
        new._codebook = self._codebook  # shared
        new._goal = self._goal
        new._pos = self._random_position_with_rng(new.rng, exclude=self._goal)
        new._heading = (1, 0)
        return new

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _random_position(self, exclude: tuple[int, int] | None = None) -> tuple[int, int]:
        return self._random_position_with_rng(self.rng, exclude)

    @staticmethod
    def _random_position_with_rng(
        rng: np.random.RandomState,
        exclude: tuple[int, int] | None = None,
        size: int | None = None,
    ) -> tuple[int, int]:
        # size is inferred from the instance in normal usage
        raise NotImplementedError("Use instance method")

    def _random_position(self, exclude: tuple[int, int] | None = None) -> tuple[int, int]:
        while True:
            p = (int(self.rng.randint(0, self.size)), int(self.rng.randint(0, self.size)))
            if p != exclude:
                return p


class ContinuousGridEnv(GridEnv):
    """Grid env with floating-point positions.  Snaps to integer for obs lookup."""

    def __init__(self, *args, scale: float = 1.0, normalize: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._continuous_pos = np.array(self._pos, dtype=np.float64)
        self.scale = scale
        self.normalize_step = normalize

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
