"""Vectorized batched environment for parallel rollouts.

Manages B parallel episodes sharing a single GridEnv's codebook and goal.
Returns raw binary observations — all neural processing is in the rollout collector.

Two variants:
  VecEnv — discrete (integer positions, cardinal actions)
  ContinuousVecEnv — continuous (float positions, (dx,dy) float actions, snap for lookup)
"""
from __future__ import annotations

import numpy as np

from .env import GridEnv, CARDINAL_ACTIONS, at_goal


class VecEnv:
    """Batched grid environment.

    All B episodes share the same codebook and goal from base_env.
    Each has independent position/heading state.
    """

    def __init__(self, base_env: GridEnv, batch_size: int) -> None:
        self.base_env = base_env
        self.B = batch_size
        self.size = base_env.size
        self.speed = base_env.speed
        self._codebook = base_env._codebook  # shared
        self._goal = base_env._goal
        self._obs_size = base_env._observation_size
        self.time_penalty = base_env.time_penalty
        self.goals_active = getattr(base_env, "goals_active", True)
        self.goal_reward = getattr(base_env, "goal_reward", 1.0)
        self.goal_radius = getattr(base_env, "goal_radius", 0.5)

        # Batched state: (B, 2) integer positions
        self._pos = np.zeros((batch_size, 2), dtype=np.int32)
        self._heading = np.zeros((batch_size, 2), dtype=np.int32)
        self._rng = np.random.RandomState(base_env.rng.randint(0, 2**31))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_all(self) -> None:
        """Random start positions for all episodes (all != goal)."""
        gx, gy = self._goal
        for b in range(self.B):
            while True:
                x = self._rng.randint(0, self.size)
                y = self._rng.randint(0, self.size)
                if (x, y) != (gx, gy):
                    break
            self._pos[b] = [x, y]
            self._heading[b] = [1, 0]

    def reset_indices(self, indices: np.ndarray) -> None:
        """Reset specific episodes to random positions (goal stays fixed)."""
        gx, gy = self._goal
        for b in indices:
            while True:
                x = self._rng.randint(0, self.size)
                y = self._rng.randint(0, self.size)
                if (x, y) != (gx, gy):
                    break
            self._pos[b] = [x, y]
            self._heading[b] = [1, 0]

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    def obs_batch(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Get observations for a batch of episodes.

        Returns (B, obs_size) or (len(indices), obs_size).
        """
        if indices is None:
            return self._codebook[self._pos[:, 0], self._pos[:, 1]]
        return self._codebook[self._pos[indices, 0], self._pos[indices, 1]]

    def positions(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Get current positions (B, 2) or (len(indices), 2)."""
        if indices is None:
            return self._pos.copy()
        return self._pos[indices].copy()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step_batch(
        self,
        actions: np.ndarray | list,
        indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step environments with cardinal action indices (0=N, 1=E, 2=S, 3=W).

        Semantics: if the agent's current (pre-action) position is the goal,
        this step consumes the at-goal turn — reward +1, teleport, and the move
        action is ignored. This gives the agent one observable step at goal.

        actions: (B,) int array of action indices, or list of action tuples.
        Returns (rewards, goal_reached, positions) all shape (B,) or (len(indices),).
        """
        if indices is None:
            indices = np.arange(self.B)

        n = len(indices)
        rewards = np.full(n, -self.time_penalty, dtype=np.float32)

        # Pre-action at-goal check — this step consumes the at-goal turn.
        # Skipped entirely when goals_active is False (pure-explore Phase A).
        # Computed once over the full batch via the centralized helper, then
        # sliced to `indices` for this step. Discrete envs use _pos (int).
        if self.goals_active:
            goal_reached = at_goal(self)[indices].copy()
        else:
            goal_reached = np.zeros(n, dtype=bool)
        rewards[goal_reached] = self.goal_reward

        for j, b in enumerate(indices):
            if goal_reached[j]:
                continue  # skip movement; teleport applied below

            if isinstance(actions[j], (int, np.integer)):
                dx, dy = CARDINAL_ACTIONS[int(actions[j])]
            else:
                dx, dy = int(actions[j][0]), int(actions[j][1])

            nx = max(0, min(self.size - 1, self._pos[b, 0] + dx * self.speed))
            ny = max(0, min(self.size - 1, self._pos[b, 1] + dy * self.speed))

            if (nx, ny) != (self._pos[b, 0], self._pos[b, 1]):
                self._heading[b] = [np.sign(nx - self._pos[b, 0]),
                                    np.sign(ny - self._pos[b, 1])]
            self._pos[b] = [nx, ny]

        # Teleport envs that were at goal.
        reached_b = indices[goal_reached]
        if len(reached_b) > 0:
            self.reset_indices(reached_b)

        return rewards, goal_reached, self._pos[indices].copy()

    def best_action_batch(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Greedy best action indices toward goal for each episode."""
        if indices is None:
            indices = np.arange(self.B)
        gx, gy = self._goal
        actions = np.zeros(len(indices), dtype=np.int32)
        for j, b in enumerate(indices):
            best_dist = float('inf')
            best_a = 0
            for a_idx, (dx, dy) in enumerate(CARDINAL_ACTIONS):
                nx = max(0, min(self.size - 1, self._pos[b, 0] + dx * self.speed))
                ny = max(0, min(self.size - 1, self._pos[b, 1] + dy * self.speed))
                dist = abs(nx - gx) + abs(ny - gy)
                if dist < best_dist:
                    best_dist = dist
                    best_a = a_idx
            actions[j] = best_a
        return actions


class ContinuousVecEnv:
    """Batched environment with float positions and continuous (dx, dy) actions.

    State:
        _pos_f: (B, 2) float64 — source of truth for movement.
        _pos:   (B, 2) int32   — snapped position, always == snap(_pos_f).

    Invariant: _pos is updated whenever _pos_f changes (via _update_snapped).
    All external reads (positions, obs, goal checks) use _pos.
    """

    def __init__(self, base_env: GridEnv, batch_size: int,
                 scale: float = 1.0, normalize: bool = False,
                 max_action_norm: float | None = None,
                 min_action_norm: float | None = None) -> None:
        self.base_env = base_env
        self.B = batch_size
        self.size = base_env.size
        self._codebook = base_env._codebook
        self._goal = base_env._goal
        self._obs_size = base_env._observation_size
        self.time_penalty = base_env.time_penalty
        self.goals_active = getattr(base_env, "goals_active", True)
        self.goal_reward = getattr(base_env, "goal_reward", 1.0)
        self.goal_radius = getattr(base_env, "goal_radius", 0.5)
        self.scale = scale
        # When True, each (dx, dy) action is L2-normalized to a unit vector
        # before * scale, matching ContinuousGridEnv.step's behavior. Train and
        # eval must agree on this flag — see PARAMETER_AUDIT bug #6.
        self.normalize_step = normalize
        # Soft cap on action magnitude (only honored when normalize=False).
        self.max_action_norm = max_action_norm
        # Soft floor on action magnitude (only honored when normalize=False).
        self.min_action_norm = min_action_norm

        self._pos_f = np.zeros((batch_size, 2), dtype=np.float64)
        self._pos = np.zeros((batch_size, 2), dtype=np.int32)
        self._rng = np.random.RandomState(base_env.rng.randint(0, 2**31))

    def _update_snapped(self, indices: np.ndarray | None = None) -> None:
        """Recompute _pos from _pos_f for given indices (or all)."""
        if indices is None:
            self._pos = np.clip(np.round(self._pos_f).astype(np.int32), 0, self.size - 1)
        else:
            self._pos[indices] = np.clip(
                np.round(self._pos_f[indices]).astype(np.int32), 0, self.size - 1)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_all(self) -> None:
        gx, gy = self._goal
        for b in range(self.B):
            while True:
                x = self._rng.randint(0, self.size)
                y = self._rng.randint(0, self.size)
                if (x, y) != (gx, gy):
                    break
            self._pos_f[b] = [float(x), float(y)]
        self._update_snapped()

    def reset_indices(self, indices: np.ndarray) -> None:
        gx, gy = self._goal
        for b in indices:
            while True:
                x = self._rng.randint(0, self.size)
                y = self._rng.randint(0, self.size)
                if (x, y) != (gx, gy):
                    break
            self._pos_f[b] = [float(x), float(y)]
        self._update_snapped(indices)

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    def positions(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Snapped integer positions (B, 2) int. Always consistent with _pos_f."""
        if indices is None:
            return self._pos.copy()
        return self._pos[indices].copy()

    def positions_continuous(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Raw float positions (B, 2) float."""
        if indices is None:
            return self._pos_f.copy()
        return self._pos_f[indices].copy()

    def obs_batch(self, indices: np.ndarray | None = None) -> np.ndarray:
        pos = self._pos if indices is None else self._pos[indices]
        return self._codebook[pos[:, 0], pos[:, 1]]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step_batch(
        self,
        actions: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step with continuous (dx, dy) actions.

        Semantics: if the agent's current (pre-action) position is the goal,
        this step consumes the at-goal turn — reward +1, teleport, and the move
        action is ignored. Otherwise apply the move and emit -time_penalty.

        This gives the agent exactly one observable step at the goal before
        teleportation, so `at_goal` checks computed from `positions()` pre-action
        are meaningful (and the agent can fire store on that step).

        actions: (B, 2) float array.
        Returns (rewards, goal_reached, snapped_positions).
        """
        if indices is None:
            indices = np.arange(self.B)

        n = len(indices)
        rewards = np.full(n, -self.time_penalty, dtype=np.float32)

        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 2)

        if self.normalize_step:
            norms = np.linalg.norm(actions, axis=-1, keepdims=True)
            # Avoid div-by-zero on zero-vector actions (leave them as-is).
            safe = np.where(norms > 1e-8, norms, 1.0)
            actions = actions / safe
        else:
            if self.max_action_norm is not None:
                norms = np.linalg.norm(actions, axis=-1, keepdims=True)
                scale_down = np.where(norms > self.max_action_norm,
                                      self.max_action_norm / np.maximum(norms, 1e-8),
                                      1.0)
                actions = actions * scale_down
            if self.min_action_norm is not None:
                norms = np.linalg.norm(actions, axis=-1, keepdims=True)
                # Only scale up when action is non-trivially nonzero (>1e-8) so
                # we don't synthesize a direction from numerical noise.
                scale_up = np.where(
                    (norms > 1e-8) & (norms < self.min_action_norm),
                    self.min_action_norm / np.maximum(norms, 1e-8),
                    1.0,
                )
                actions = actions * scale_up

        # Identify envs that start the step at goal — they reap +1 and teleport,
        # their movement is ignored. Skipped when goals_active is False (pure-explore).
        # The check is on the *continuous* pre-action position (_pos_f), via the
        # centralized at_goal helper which auto-resolves to _pos_f for ContinuousVecEnv.
        if self.goals_active:
            goal_reached = at_goal(self)[indices].copy()
        else:
            goal_reached = np.zeros(n, dtype=bool)
        rewards[goal_reached] = self.goal_reward

        # Apply movement only to envs not at goal.
        for j, b in enumerate(indices):
            if goal_reached[j]:
                continue
            self._pos_f[b] = np.clip(
                self._pos_f[b] + actions[j] * self.scale,
                0.0, float(self.size - 1),
            )

        self._update_snapped(indices)

        # Teleport envs that were at goal.
        reached_b = indices[goal_reached]
        if len(reached_b) > 0:
            self.reset_indices(reached_b)

        return rewards, goal_reached, self._pos[indices].copy()
