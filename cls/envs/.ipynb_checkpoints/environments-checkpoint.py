from __future__ import annotations

from ..types import Position, FovFunction, Vector2
from ..utils import VectorHash

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


class WMEnv:
    """Discrete square grid environment with codebook-based observations.

    Coordinate system:
    - The grid is size x size with integer coordinates 0..size-1 on both axes.
    - Positions are tuples (x, y) with origin at bottom-left (0, 0).
    - Heading is a cardinal vector (dx, dy) derived from the last successful non-zero move.

    Movement semantics (vector actions):
    - Action is a vector (dx, dy) with integer steps. For now we expect
      components in {-1, 0, 1}. The environment scales by `speed`.
    - Movement proceeds up to `speed` sub-steps; if a boundary would be
      crossed, the agent stops at the last valid cell (partial movement).
      If no sub-step is possible, the agent stays.
    - If the vector is (0, 0), the agent stays and heading is unchanged.

    Observation (codebook):
    - At initialization, a binary code of length `observation_size` is generated
      for every (position, heading) pair.
    - Calling `obs()` returns the code corresponding to the current state.
    - No wall/FOV geometry is used; `fov_fn`, `fov_deg` are retained only for
      API compatibility and are unused.

    Step return: (current_location, goal_location, obs)

    Utility:
    - best_action_to_goal(randomize=False): choose among forward/right/left/back
      the action that minimizes distance to goal after moving this step. If
      multiple are equally good, tie-break order is forward > right > left > back
      unless randomize=True.
    """

    def __init__(
        self,
        size: int,
        speed: int,
        seed: Optional[int] = None,
        fov_fn: Optional[FovFunction] = None,
        max_fov: Optional[int] = None,
        observation_size: int = 64,
        fov_deg: float = 120.0,
        time_penalty: float = 0.01,
        start_type: Optional[str] = "random",
    ) -> None:
        if not isinstance(size, int) or size <= 1:
            raise ValueError("size must be an integer >= 2")
        if not isinstance(speed, int) or speed <= 0:
            raise ValueError("speed must be a positive integer")
        if max_fov is not None and (not isinstance(max_fov, int) or max_fov < 1):
            raise ValueError("max_fov must be a positive integer if provided")

        self.size: int = size
        self.speed: int = speed
        self._rng = random.Random(seed)

        self._fov_fn: FovFunction = fov_fn or (lambda d, s: max(1, d + 1))
        self._max_fov: Optional[int] = max_fov
        self._observation_size: int = observation_size
        self._fov_deg: float = fov_deg
        self._time_penalty: float = time_penalty
        # Initialize codebook as numpy array (size, size, 4, observation_size)
        rs = np.random.RandomState(self._rng.randrange(10_000_000))
        self._codebook = rs.randint(
            0,
            2,
            size=(self.size, self.size, 4, self._observation_size),
            dtype=np.int32,
        ).astype(np.float32)

        self._goal: Position = (self._rng.randrange(self.size), self._rng.randrange(self.size))
        if start_type == "random":
            self._start: Position = self._sample_start_not_equal_to(self._goal)
            self._heading: Vector2 = self._random_cardinal_heading()
        elif start_type == "corner":
            self._start = (0, 0)
            self._heading = (1, 0)
        self._pos: Position = self._start

    # --------------------- Public API ---------------------
    @property
    def current_location(self) -> Position:
        return self._pos

    @property
    def goal_location(self) -> Position:
        return self._goal

    @property
    def heading(self) -> Vector2:
        return self._heading

    def reset(self) -> Tuple[Position, Position, np.ndarray, float]:
        """Reset agent state (start, position, heading) but keep goal and walls.

        The goal remains fixed across resets. Use `reset_goal()` to change it.
        """
        self._start = self._sample_start_not_equal_to(self._goal)
        self._pos = self._start
        self._heading = self._random_cardinal_heading()
        return self.state()

    def reset_goal(self) -> Tuple[Position, Position, np.ndarray, float]:
        """Sample a new goal location (different from current start) and return observation."""
        while True:
            new_goal = (self._rng.randrange(self.size), self._rng.randrange(self.size))
            if new_goal != self._start:
                self._goal = new_goal
                break
        return self.state()

    def step(self, action: Vector2) -> Tuple[Position, Position, np.ndarray, float]:
        """Apply a vector action (dx, dy); returns (current_location, goal_location, obs).

        Movement allows partial progress up to the boundary. Heading updates only
        if any movement occurred this step (non-zero displacement).
        """
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("action must be a (dx, dy) tuple")

        dx, dy = action
        if dx == 0 and dy == 0:
            return self.state()

        # Normalize to a cardinal direction if both components non-zero (diagonal):
        # prefer the larger magnitude component; if equal, keep both to allow diagonal if desired.
        ndx, ndy = self._normalize_vector(dx, dy)
        new_pos = self._simulate_move(self._pos, (ndx, ndy), self.speed)
        moved = new_pos != self._pos
        self._pos = new_pos
        if moved:
            self._heading = (ndx, ndy)
        return self.state()

    def obs(self) -> np.ndarray:
        """Return the pre-generated binary code for (position, heading)."""
        return self._code_for(self._pos, self._heading)
    
    def obs_at_goal(self) -> np.ndarray:
        return self._code_for(self._goal, self._heading)

    def best_action_to_goal(self, randomize: bool = False) -> Vector2:
        """Return the best relative action toward the goal.

        Candidates are relative to a cardinal basis heading determined from the
        current heading (or goal direction if heading is zero): forward, right,
        left, back. The chosen action minimizes the distance to the goal after
        simulating this step (with partial movement allowed). If multiple are
        equally good, apply deterministic priority: forward > right > left > back,
        unless `randomize=True`, in which case a best action is chosen uniformly
        at random among the tied candidates.
        """
        # If already at goal, do not move
        if self._pos == self._goal:
            return (0, 0)
        base = self._cardinal_basis_heading()
        candidates = [
            base,
            self._rotate_right(base),
            self._rotate_left(base),
            (-base[0], -base[1]),
        ]

        best_dist2 = None
        best_indices = []
        for idx, vec in enumerate(candidates):
            next_pos = self._simulate_move(self._pos, vec, self.speed)
            dx = self._goal[0] - next_pos[0]
            dy = self._goal[1] - next_pos[1]
            dist2 = dx * dx + dy * dy
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best_indices = [idx]
            elif dist2 == best_dist2:
                best_indices.append(idx)

        if randomize and best_indices:
            choice_idx = self._rng.choice(best_indices)
        else:
            choice_idx = best_indices[0] if best_indices else 0

        return candidates[choice_idx]

    def fully_explore(self) -> list[tuple[Position, np.ndarray, Vector2]]:
        """Return a deterministic coverage of all positions from all headings.

        This method does not mutate environment state. It computes observations
        for each grid position under each of the four cardinal headings by
        stitching together four serpentine (lawnmower) passes:
        - Two horizontal passes to cover E and W at every (x, y)
        - Two vertical passes to cover N and S at every (x, y)

        Returns a list of (position, obs, heading) tuples.
        """
        results: list[tuple[Position, np.ndarray, Vector2]] = []

        def record_row(y_order: list[int], ltr_on_even: bool) -> None:
            for i, y in enumerate(y_order):
                even = (i % 2 == 0)
                if (even and ltr_on_even) or ((not even) and (not ltr_on_even)):
                    xs = range(0, self.size)
                    heading = (1, 0)
                else:
                    xs = range(self.size - 1, -1, -1)
                    heading = (-1, 0)
                for x in xs:
                    pos: Position = (x, y)
                    results.append((pos, self._code_for(pos, heading), heading))

        def record_col(x_order: list[int], btt_on_even: bool) -> None:
            for j, x in enumerate(x_order):
                even = (j % 2 == 0)
                if (even and btt_on_even) or ((not even) and (not btt_on_even)):
                    ys = range(0, self.size)
                    heading = (0, 1)
                else:
                    ys = range(self.size - 1, -1, -1)
                    heading = (0, -1)
                for y in ys:
                    pos: Position = (x, y)
                    results.append((pos, self._code_for(pos, heading), heading))

        # Horizontal passes: bottom->top then top->bottom with flipped parity
        record_row(list(range(0, self.size)), ltr_on_even=True)
        record_row(list(range(self.size - 1, -1, -1)), ltr_on_even=False)

        # Vertical passes: left->right then right->left with flipped parity
        record_col(list(range(0, self.size)), btt_on_even=True)
        record_col(list(range(self.size - 1, -1, -1)), btt_on_even=False)

        return results

    def fully_explore_random(self) -> list[tuple[Position, np.ndarray, Vector2]]:
        """Return all (position, obs, heading) tuples in random order.

        This method does not mutate environment state. It enumerates every grid
        position under each of the four cardinal headings and shuffles the
        resulting list using the environment's RNG.
        """
        results: list[tuple[Position, np.ndarray, Vector2]] = []
        headings: list[Vector2] = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]
        for x in range(self.size):
            for y in range(self.size):
                pos: Position = (x, y)
                for heading in headings:
                    results.append((pos, self._code_for(pos, heading), heading))
        self._rng.shuffle(results)
        return results        

    # --------------------- Internals ----------------------
    
    # --------------------- Cloning ------------------------
    def clone(self) -> "WMEnv":
        """Create a shallow clone that shares the codebook but has independent RNG/state.

        Note: The clone will be initialized with the same configuration and then
        its `_codebook` and `_goal` are set to share with the original. Call
        `reset()` on the clone to sample a new start/heading independent of the original.
        """
        new_env = type(self)(
            size=self.size,
            speed=self.speed,
            seed=None,
            fov_fn=self._fov_fn,
            max_fov=self._max_fov,
            observation_size=self._observation_size,
            fov_deg=self._fov_deg,
            time_penalty=self._time_penalty,
        )
        # Share large immutable arrays
        new_env._codebook = self._codebook
        # Keep same goal so best-action labels match semantics
        new_env._goal = self._goal
        return new_env

    def reward(self) -> float:
        return 1.0 if self._pos == self._goal else -self._time_penalty

    def state(self) -> Tuple[Position, Position, np.ndarray, float]:
        return self._pos, self._goal, self.obs(), self.reward()

    # Codebook helpers  
    def _heading_index(self, heading: Vector2) -> int:
        if heading == (0, 1):
            return 0
        if heading == (1, 0):
            return 1
        if heading == (0, -1):
            return 2
        if heading == (-1, 0):
            return 3
        # For diagonals or zero, project to dominant axis
        dx, dy = heading
        if abs(dx) >= abs(dy):
            return 1 if dx > 0 else 3
        return 0 if dy > 0 else 2

    def _code_for(self, pos: Position, heading: Vector2) -> np.ndarray:
        x, y = pos
        h = self._heading_index(heading)
        return self._codebook[x, y, h]

    def _sample_start_not_equal_to(self, goal: Position) -> Position:
        while True:
            start = (self._rng.randrange(self.size), self._rng.randrange(self.size))
            if start != goal:
                return start

    def _simulate_move(self, pos: Position, direction: Vector2, speed: int) -> Position:
        dx, dy = direction
        x, y = pos
        for _ in range(speed):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                break
            x, y = nx, ny
        return (x, y)

    # Remove wall-based geometry; observation is codebook lookup

    # --------------------- Helpers -----------------------
    def _rotate_right(self, vec: Vector2) -> Vector2:
        # (dx, dy) -> (dy, -dx)
        return (vec[1], -vec[0])

    def _rotate_left(self, vec: Vector2) -> Vector2:
        # (dx, dy) -> (-dy, dx)
        return (-vec[1], vec[0])

    def _cardinal_basis_heading(self) -> Vector2:
        """Return a cardinal unit vector representing the current facing basis.

        If current heading is diagonal, project to dominant axis. If heading is
        zero, choose the cardinal direction that points most toward the goal
        (tie-break prefers x axis).
        """
        hx, hy = self._heading
        if hx == 0 and hy == 0:
            gx = self._goal[0] - self._pos[0]
            gy = self._goal[1] - self._pos[1]
            if abs(gx) >= abs(gy):
                return (1, 0) if gx >= 0 else (-1, 0)
            return (0, 1) if gy >= 0 else (0, -1)
        if abs(hx) >= abs(hy):
            return (1, 0) if hx > 0 else (-1, 0)
        return (0, 1) if hy > 0 else (0, -1)

    def _normalize_vector(self, dx: int, dy: int) -> Vector2:
        if dx == 0 and dy == 0:
            return (0, 0)
        # Clamp to -1..1
        dx = 0 if dx == 0 else (1 if dx > 0 else -1)
        dy = 0 if dy == 0 else (1 if dy > 0 else -1)
        # If diagonal, keep as-is (allow diagonal) or choose dominant axis?
        # For now, keep diagonal to allow future extension; movement uses this vector.
        return (dx, dy)

    def _random_cardinal_heading(self) -> Vector2:
        return random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])


class GridWMEnv(WMEnv):
    def __init__(self, *args, **kwargs):
        self.input_type = kwargs.pop("input_type", "g")
        super().__init__(*args, **kwargs)
    
    @property
    def goal_location(self) -> Position:
        return super().goal_location

    def initiate_vectorhash(self, vectorhash: VectorHash):
        self.vectorhash = vectorhash
    
    def get_input_size(self) -> int:
        if self.input_type == "g_hot":
            return self.vectorhash.Ng
        elif self.input_type == "g_idx":
            return len(self.vectorhash.lambdas)*2
        elif self.input_type == "s":
            return self.vectorhash.Ns
        elif self.input_type == "p":
            return self.vectorhash.Np
        else:
            raise ValueError(f"Invalid input type: {self.input_type}")
    
    def convert_obs(self, obs: np.ndarray) -> np.ndarray:
        s,p,g = self.vectorhash.recall(obs.copy())

        if self.input_type == "g_hot":
            return np.asarray(g, dtype=np.float32)
        elif self.input_type == "g_idx":
            return np.asarray(self.vectorhash.grid_onehot_to_indices(g), dtype=np.float32)
        elif self.input_type == "s":
            return np.asarray(s, dtype=np.float32)
        elif self.input_type == "p":
            return np.asarray(p, dtype=np.float32)
        else:
            raise ValueError(f"Invalid input type: {self.input_type}")
    
    def obs(self) -> np.ndarray:
        obs = super().obs()
        return self.convert_obs(obs)
    
    def obs_at_goal(self) -> np.ndarray:
        obs = super().obs_at_goal()
        return self.convert_obs(obs)
    
    def clone(self) -> "GridWMEnv":
        new_env = type(self)(
            size=self.size,
            speed=self.speed,
            seed=None,
            observation_size=self._observation_size,
            fov_deg=self._fov_deg,
            time_penalty=self._time_penalty,
            input_type=self.input_type,
        )
        # Share codebook and vectorhash; copy goal
        new_env._codebook = self._codebook
        if hasattr(self, "vectorhash"):
            new_env.vectorhash = self.vectorhash
        new_env._goal = self._goal
        return new_env


class WMVecEnv:
    """Vectorized environment built from a base WMEnv clone.

    Maintains batched positions and headings and provides vectorized obs,
    best-action labeling, and stepping for cardinal action vectors.
    """

    def __init__(self, base_env: WMEnv, batch_size: int, seed: Optional[int] = None) -> None:
        self.size = base_env.size
        self.speed = base_env.speed
        self._time_penalty = base_env._time_penalty
        self._codebook = base_env._codebook  # shared
        self._goal = base_env._goal          # shared goal across batch (same semantics as clones)
        self._observation_size = base_env._observation_size
        self._rng = np.random.RandomState(seed if seed is not None else 0)
        # Batched state (B,)
        self.B = int(batch_size)
        self._pos = np.zeros((self.B, 2), dtype=np.int32)
        self._heading = np.zeros((self.B, 2), dtype=np.int32)
        self.reset_all()

    def reset_all(self) -> None:
        # Sample starts not equal to goal
        gx, gy = self._goal
        self._pos[:, 0] = self._rng.randint(0, self.size, size=self.B)
        self._pos[:, 1] = self._rng.randint(0, self.size, size=self.B)
        mask_eq = (self._pos[:, 0] == gx) & (self._pos[:, 1] == gy)
        while mask_eq.any():
            self._pos[mask_eq, 0] = self._rng.randint(0, self.size, size=int(mask_eq.sum()))
            self._pos[mask_eq, 1] = self._rng.randint(0, self.size, size=int(mask_eq.sum()))
            mask_eq = (self._pos[:, 0] == gx) & (self._pos[:, 1] == gy)
        # Random cardinal headings
        choices = np.array([[0,1],[1,0],[0,-1],[-1,0]], dtype=np.int32)
        idx = self._rng.randint(0, 4, size=self.B)
        self._heading = choices[idx]

    # --------- Vector helpers ---------
    @staticmethod
    def _heading_index_batch(heading: np.ndarray) -> np.ndarray:
        # heading shape: (B, 2)
        hx = heading[:, 0]
        hy = heading[:, 1]
        h_idx = np.zeros(hx.shape[0], dtype=np.int32)
        # Cardinal cases
        h_idx[(hy == 1) & (hx == 0)] = 0
        h_idx[(hx == 1) & (hy == 0)] = 1
        h_idx[(hy == -1) & (hx == 0)] = 2
        h_idx[(hx == -1) & (hy == 0)] = 3
        # Diagonal or zero -> dominant axis
        diag = ~(((hy == 1) & (hx == 0)) | ((hx == 1) & (hy == 0)) | ((hy == -1) & (hx == 0)) | ((hx == -1) & (hy == 0)))
        if diag.any():
            abshx = np.abs(hx[diag])
            abshy = np.abs(hy[diag])
            choose_x = abshx >= abshy
            # For x-dominant: 1 if hx>0 else 3; For y-dominant: 0 if hy>0 else 2
            idxs = np.where(diag)[0]
            x_part = idxs[choose_x]
            y_part = idxs[~choose_x]
            h_idx[x_part] = np.where(hx[x_part] > 0, 1, 3)
            h_idx[y_part] = np.where(hy[y_part] > 0, 0, 2)
        return h_idx

    def obs_batch(self, indices: List[int], input_addendum: Optional[str] = None) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        x = self._pos[idx, 0]
        y = self._pos[idx, 1]
        h_idx = self._heading_index_batch(self._heading[idx])
        obs = self._codebook[x, y, h_idx]  # (B, F)
        if input_addendum == "goal":
            gx = np.full_like(x, self._goal[0])
            gy = np.full_like(y, self._goal[1])
            goal_obs = self._codebook[gx, gy, h_idx]
            obs = np.concatenate([obs, goal_obs], axis=-1)
        elif input_addendum == "diff":
            gx = np.full_like(x, self._goal[0])
            gy = np.full_like(y, self._goal[1])
            goal_obs = self._codebook[gx, gy, h_idx]
            obs = goal_obs - obs
        return obs.astype(np.float32)

    def best_action_to_goal_batch(self, indices: List[int], randomize: bool = False) -> List[Tuple[int,int]]:
        idx = np.asarray(indices, dtype=np.int64)
        B = idx.shape[0]
        pos = self._pos[idx].astype(np.int32)
        # Compute basis heading (cardinal) per env
        hx = self._heading[idx, 0]
        hy = self._heading[idx, 1]
        base = np.zeros((B, 2), dtype=np.int32)
        # zero heading -> toward goal dominant axis
        zero = (hx == 0) & (hy == 0)
        if zero.any():
            gx = self._goal[0] - pos[zero, 0]
            gy = self._goal[1] - pos[zero, 1]
            choose_x = np.abs(gx) >= np.abs(gy)
            bx = np.where(choose_x, np.where(gx >= 0, 1, -1), 0)
            by = np.where(choose_x, 0, np.where(gy >= 0, 1, -1))
            base[zero, 0] = bx
            base[zero, 1] = by
        nonzero = ~zero
        if nonzero.any():
            hx_nz = hx[nonzero]
            hy_nz = hy[nonzero]
            choose_x = np.abs(hx_nz) >= np.abs(hy_nz)
            bx = np.where(choose_x, np.where(hx_nz > 0, 1, -1), 0)
            by = np.where(choose_x, 0, np.where(hy_nz > 0, 1, -1))
            base[nonzero, 0] = bx
            base[nonzero, 1] = by
        # Candidates: forward, right, left, back
        right = np.stack([base[:,1], -base[:,0]], axis=1)
        left = np.stack([-base[:,1], base[:,0]], axis=1)
        back = -base
        candidates = np.stack([base, right, left, back], axis=1)  # (B, 4, 2)
        # Simulate move for each candidate up to speed
        pos_rep = np.repeat(pos[:, None, :], 4, axis=1)
        new_pos = pos_rep.copy()
        for _ in range(self.speed):
            nx = new_pos[:, :, 0] + candidates[:, :, 0]
            ny = new_pos[:, :, 1] + candidates[:, :, 1]
            inside = (nx >= 0) & (nx < self.size) & (ny >= 0) & (ny < self.size)
            new_pos[:, :, 0] = np.where(inside, nx, new_pos[:, :, 0])
            new_pos[:, :, 1] = np.where(inside, ny, new_pos[:, :, 1])
        dx = self._goal[0] - new_pos[:, :, 0]
        dy = self._goal[1] - new_pos[:, :, 1]
        dist2 = dx * dx + dy * dy
        # Choose argmin with tie-break: forward > right > left > back
        # Implement by adding tiny increasing offsets so lower index preferred
        tiebreak = np.array([0.0, 1e-6, 2e-6, 3e-6], dtype=np.float64)
        dist2_tb = dist2.astype(np.float64) + tiebreak
        best_idx = np.argmin(dist2_tb, axis=1)
        if randomize:
            # find all ties and randomly choose among them
            chosen = best_idx.copy()
            for b in range(B):
                minv = dist2[b].min()
                ties = np.where(dist2[b] == minv)[0]
                if ties.size > 1:
                    chosen[b] = int(self._rng.choice(ties))
            best_idx = chosen
        actions = [tuple(candidates[i, best_idx[i]].tolist()) for i in range(B)]
        return actions

    def step_batch(self, indices: List[int], action_vectors: List[Tuple[int,int]]) -> tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[float], List[bool]]:
        idx = np.asarray(indices, dtype=np.int64)
        B = idx.shape[0]
        vec = np.asarray(action_vectors, dtype=np.int32)
        # Normalize to -1..1 per component
        vec = np.sign(vec)
        pos = self._pos[idx].astype(np.int32)
        new_pos = pos.copy()
        for _ in range(self.speed):
            nx = new_pos[:, 0] + vec[:, 0]
            ny = new_pos[:, 1] + vec[:, 1]
            inside = (nx >= 0) & (nx < self.size) & (ny >= 0) & (ny < self.size)
            new_pos[:, 0] = np.where(inside, nx, new_pos[:, 0])
            new_pos[:, 1] = np.where(inside, ny, new_pos[:, 1])
        moved = (new_pos[:, 0] != pos[:, 0]) | (new_pos[:, 1] != pos[:, 1])
        # Update batch state
        self._pos[idx] = new_pos
        # Update headings where moved
        vec_norm = vec.copy()
        # already -1/0/1 per component
        self._heading[idx[moved]] = vec_norm[moved]
        # Rewards and dones
        gx, gy = self._goal
        dones = (new_pos[:, 0] == gx) & (new_pos[:, 1] == gy)
        rewards = np.where(dones, 1.0, -self._time_penalty).astype(np.float32)
        # Outputs as lists for compatibility
        cur_list = [tuple(new_pos[i].tolist()) for i in range(B)]
        goal_list = [self._goal for _ in range(B)]
        reward_list = [float(r) for r in rewards.tolist()]
        done_list = [bool(d) for d in dones.tolist()]
        return cur_list, goal_list, reward_list, done_list


class GridWMVecEnv(WMVecEnv):
    def __init__(self, base_env: GridWMEnv, batch_size: int, seed: Optional[int] = None, use_preconv_codebook: bool = True) -> None:
        self.input_type = base_env.input_type
        if hasattr(base_env, "vectorhash"):
            self.vectorhash = base_env.vectorhash
        super().__init__(base_env=base_env, batch_size=batch_size, seed=seed)
        # Precompute converted codebook for fast batched lookup when possible
        self._preconv_codebook: Optional[np.ndarray] = None
        if use_preconv_codebook:
            try:
                self._preconv_codebook = self._build_preconv_codebook()
            except Exception:
                # Fallback to on-the-fly conversion path if anything goes wrong
                self._preconv_codebook = None

    def _convert_obs_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        # Fallback: per-row conversion via vectorhash.recall
        converted = []
        for i in range(obs_batch.shape[0]):
            s, p, g = self.vectorhash.recall(obs_batch[i].copy())
            if self.input_type == "g_hot":
                converted.append(np.asarray(g, dtype=np.float32))
            elif self.input_type == "g_idx":
                converted.append(np.asarray(self.vectorhash.grid_onehot_to_indices(g), dtype=np.float32))
            elif self.input_type == "s":
                converted.append(np.asarray(s, dtype=np.float32))
            elif self.input_type == "p":
                converted.append(np.asarray(p, dtype=np.float32))
            else:
                raise ValueError(f"Invalid input type: {self.input_type}")
        return np.stack(converted)

    def obs_batch(self, indices: List[int], input_addendum: Optional[str] = None) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        h_idx = self._heading_index_batch(self._heading[idx])
        if self._preconv_codebook is not None:
            x = self._pos[idx, 0]
            y = self._pos[idx, 1]
            conv = self._preconv_codebook[x, y, h_idx]
            if input_addendum == "goal":
                gx = np.full(x.shape[0], self._goal[0], dtype=np.int32)
                gy = np.full(y.shape[0], self._goal[1], dtype=np.int32)
                goal_conv = self._preconv_codebook[gx, gy, h_idx]
                return np.concatenate([conv, goal_conv], axis=-1)
            elif input_addendum == "diff":
                gx = np.full(x.shape[0], self._goal[0], dtype=np.int32)
                gy = np.full(y.shape[0], self._goal[1], dtype=np.int32)
                goal_conv = self._preconv_codebook[gx, gy, h_idx]
                return goal_conv - conv
            return conv
        # Fallback: convert on the fly
        raw = super().obs_batch(indices, input_addendum=None)
        conv = self._convert_obs_batch(raw)
        if input_addendum == "goal":
            gx = np.full(len(indices), self._goal[0], dtype=np.int32)
            gy = np.full(len(indices), self._goal[1], dtype=np.int32)
            goal_raw = self._codebook[gx, gy, h_idx]
            goal_conv = self._convert_obs_batch(goal_raw)
            return np.concatenate([conv, goal_conv], axis=-1)
        elif input_addendum == "diff":
            gx = np.full(len(indices), self._goal[0], dtype=np.int32)
            gy = np.full(len(indices), self._goal[1], dtype=np.int32)
            goal_raw = self._codebook[gx, gy, h_idx]
            goal_conv = self._convert_obs_batch(goal_raw)
            return goal_conv - conv
        return conv

    def _build_preconv_codebook(self) -> np.ndarray:
        """Precompute converted codebook (size, size, 4, Fconv) for current input_type.

        Falls back to per-row conversion if vectorhash is not available.
        """
        if not hasattr(self, "vectorhash"):
            raise RuntimeError("vectorhash not available for preconversion")
        size = self.size
        F = self._observation_size
        # Determine output feature size by converting one sample
        sample = self._codebook[0, 0, 0]
        s, p, g = self.vectorhash.recall(sample.copy())
        if self.input_type == "g_hot":
            out_dim = len(g)
        elif self.input_type == "g_idx":
            out_dim = len(self.vectorhash.grid_onehot_to_indices(g))
        elif self.input_type == "s":
            out_dim = len(s)
        elif self.input_type == "p":
            out_dim = len(p)
        else:
            raise ValueError(f"Invalid input type: {self.input_type}")
        pre = np.zeros((size, size, 4, out_dim), dtype=np.float32)
        for x in range(size):
            for y in range(size):
                for h in range(4):
                    code = self._codebook[x, y, h].copy()
                    s, p, g = self.vectorhash.recall(code)
                    if self.input_type == "g_hot":
                        pre[x, y, h] = np.asarray(g, dtype=np.float32)
                    elif self.input_type == "g_idx":
                        pre[x, y, h] = np.asarray(self.vectorhash.grid_onehot_to_indices(g), dtype=np.float32)
                    elif self.input_type == "s":
                        pre[x, y, h] = np.asarray(s, dtype=np.float32)
                    elif self.input_type == "p":
                        pre[x, y, h] = np.asarray(p, dtype=np.float32)
        return pre
