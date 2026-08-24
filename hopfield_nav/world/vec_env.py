"""Vectorized batched environment for parallel rollouts.

Manages B parallel episodes sharing a single GridEnv's codebook and goal.
Returns raw binary observations — all neural processing is in the rollout collector.

Two variants:
  VecEnv — discrete (integer positions, cardinal actions)
  ContinuousVecEnv — continuous (float positions, (dx,dy) float actions, snap for lookup)
"""
from __future__ import annotations

import numpy as np

from .env import (
    CARDINAL_RADIANS, CARDINAL_ACTIONS, GridEnv, at_goal, cardinal_index,
    nearest_heading, raycast_codes,
)
from . import episode
from .episode import GoalContract


def _views(vec, pos: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Foveal views for a batch of (cell, heading) pairs: (N, obs_size).

    Gathers from the cardinal codebook when every ψ is exactly cardinal --
    always true in discrete movement, and always true when the env's cone is
    pinned North -- and ray-casts otherwise. The two paths agree by
    construction (the codebook is built by the same function), so this is a
    speed choice rather than a behavioral one.
    """
    k = cardinal_index(psi)
    if k.size == 0 or (k >= 0).all():
        return vec._codebook[pos[:, 0], pos[:, 1], k]
    return raycast_codes(vec._wall_code, vec.size, pos[:, 0], pos[:, 1], psi,
                         vec._obs_size, vec.wall_resolution)


def _obs_psi(vec, indices: np.ndarray | None) -> np.ndarray:
    """The ψ observations are read at: live headings, or North when fixed."""
    psi = vec._heading_rad if indices is None else vec._heading_rad[indices]
    return psi if vec.egocentric_heading else np.zeros_like(psi)


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
        self._wall_code = base_env._wall_code  # shared; for off-cardinal casts
        self._goal = base_env._goal
        self._obs_size = base_env._observation_size
        self.time_penalty = base_env.time_penalty
        self.goals_active = getattr(base_env, "goals_active", True)
        self.goal_reward = getattr(base_env, "goal_reward", 1.0)
        self.goal_radius = getattr(base_env, "goal_radius", 0.5)
        self.egocentric_heading = getattr(base_env, "egocentric_heading", True)
        self.wall_resolution = getattr(base_env, "wall_resolution", 1)

        # Batched state: (B, 2) integer positions, (B,) heading angles.
        # Heading is radians clockwise from North; cardinal moves land it
        # exactly on a multiple of π/2, so obs stays on the codebook gather.
        self._pos = np.zeros((batch_size, 2), dtype=np.int32)
        self._heading_rad = np.zeros(batch_size, dtype=np.float64)
        # Displacement actually realized by the last step_batch, which is not
        # the action: the norm clamp and the arena clip both bite, and the
        # at-goal contract can suppress movement entirely. Kept here so every
        # consumer reads one number rather than recomputing it -- see
        # last_displacement().
        self._last_displacement = np.zeros((batch_size, 2), dtype=np.float64)
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
            self._heading_rad[b] = 0.0

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
            self._heading_rad[b] = 0.0

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    def obs_batch(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Observations from where each episode stands, facing where it faces.

        Returns (B, obs_size) or (len(indices), obs_size).
        """
        pos = self._pos if indices is None else self._pos[indices]
        return _views(self, pos, _obs_psi(self, indices))

    def last_displacement(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Displacement the last ``step_batch`` actually produced, (B, 2).

        Distinct from the action in three ways that all matter to the policy:
        the norm clamp rescales it, the arena clip truncates it at a wall, and
        the at-goal contract can suppress the move outright. Zero for episodes
        not included in that step's ``indices``, which did not move.

        The regime cues in EXPERIMENTS_NAV_P2 §7.2 compare how much ``q``
        changed against how far the agent went; fed the commanded action
        instead, they are wrong exactly at walls.
        """
        if indices is None:
            return self._last_displacement.copy()
        return self._last_displacement[indices].copy()

    def positions(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Get current positions (B, 2) or (len(indices), 2)."""
        if indices is None:
            return self._pos.copy()
        return self._pos[indices].copy()

    def set_positions(self, positions) -> None:
        """Place each episode at a caller-chosen cell.

        The batched counterpart of ``GridEnv.set_position``: evaluators seed
        their own start positions from a dedicated RNG rather than letting the
        env sample them, so that batching does not change which starts a trial
        gets. Heading is reset, matching set_position.
        """
        pos = np.asarray(positions, dtype=np.int32).reshape(-1, 2)
        if pos.shape[0] != self.B:
            raise ValueError(f"expected {self.B} positions, got {pos.shape[0]}")
        self._pos[:] = pos
        self._heading_rad[:] = 0.0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step_batch(
        self,
        actions: np.ndarray | list,
        indices: np.ndarray | None = None,
        contract: GoalContract | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step environments with cardinal action indices (0=N, 1=E, 2=S, 3=W).

        Semantics at the goal are governed by ``contract`` -- see
        ``world/episode.py``. The default reproduces the historical bundling:
        ``goals_active`` True means the at-goal step is consumed (reward
        ``goal_reward``, movement ignored, teleport), False means the goal is
        not an event at all. Pass a contract explicitly to decouple those, e.g.
        to reward the goal without teleporting away from it.

        actions: (B,) int array of action indices, or list of action tuples.
        Returns (rewards, goal_reached, positions) all shape (B,) or (len(indices),).
        """
        if indices is None:
            indices = np.arange(self.B)

        n = len(indices)
        if contract is None:
            contract = episode.contract_from_goals_active(self.goals_active)

        # Pre-action at-goal check — this is what gives the agent one
        # observable step at the goal. Computed over the full batch via the
        # centralized helper, then sliced to `indices`. Discrete envs use _pos.
        at_goal_mask = (at_goal(self)[indices].copy() if self.goals_active
                        else np.zeros(n, dtype=bool))
        res = episode.resolve_at_goal(
            at_goal_mask, contract,
            goal_reward=self.goal_reward, time_penalty=self.time_penalty,
        )
        rewards, goal_reached = res.rewards, res.at_goal

        moved = np.zeros((n, 2), dtype=np.int64)
        for j, b in enumerate(indices):
            if not res.apply_move[j]:
                continue  # movement ignored; teleport (if any) applied below

            if isinstance(actions[j], (int, np.integer)):
                dx, dy = CARDINAL_ACTIONS[int(actions[j])]
            else:
                dx, dy = int(actions[j][0]), int(actions[j][1])

            nx = max(0, min(self.size - 1, self._pos[b, 0] + dx * self.speed))
            ny = max(0, min(self.size - 1, self._pos[b, 1] + dy * self.speed))

            moved[j] = (nx - self._pos[b, 0], ny - self._pos[b, 1])
            self._pos[b] = [nx, ny]

        # Heading follows the *realized* displacement, so an episode whose step
        # was clipped by a wall (or ignored at the goal) keeps facing where it
        # was. Cardinal moves take ψ from the table, which keeps it exactly on a
        # multiple of π/2 and so keeps obs on the codebook gather.
        turned = nearest_heading(moved)
        spun = turned >= 0
        if spun.any():
            self._heading_rad[indices[spun]] = CARDINAL_RADIANS[turned[spun]]

        # Recorded before the teleport below, so it is the displacement the
        # action produced and not the jump that followed it.
        self._last_displacement[:] = 0.0
        self._last_displacement[indices] = moved

        # Relocate the envs the contract says to teleport.
        reached_b = indices[res.teleport]
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
        self._wall_code = base_env._wall_code  # shared; for off-cardinal casts
        self._goal = base_env._goal
        self._obs_size = base_env._observation_size
        self.time_penalty = base_env.time_penalty
        self.goals_active = getattr(base_env, "goals_active", True)
        self.goal_reward = getattr(base_env, "goal_reward", 1.0)
        self.goal_radius = getattr(base_env, "goal_radius", 0.5)
        self.egocentric_heading = getattr(base_env, "egocentric_heading", True)
        self.wall_resolution = getattr(base_env, "wall_resolution", 1)
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
        # See VecEnv.__init__ -- realized displacement of the last step.
        self._last_displacement = np.zeros((batch_size, 2), dtype=np.float64)
        # Heading, radians clockwise from North. Continuous movement can face
        # any angle, so unlike VecEnv this rarely lands on a cardinal and
        # observations are genuinely ray-cast each step.
        self._heading_rad = np.zeros(batch_size, dtype=np.float64)
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
            self._heading_rad[b] = 0.0
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
            self._heading_rad[b] = 0.0
        self._update_snapped(indices)

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    def last_displacement(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Displacement the last ``step_batch`` actually produced, (B, 2).

        The continuous copy of ``VecEnv.last_displacement`` -- these two
        classes do not share a base, so the method has to exist on both. This
        is the one that matters: the norm clamp only applies here.
        """
        if indices is None:
            return self._last_displacement.copy()
        return self._last_displacement[indices].copy()

    def positions(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Snapped integer positions (B, 2) int. Always consistent with _pos_f."""
        if indices is None:
            return self._pos.copy()
        return self._pos[indices].copy()

    def set_positions(self, positions) -> None:
        """Place each episode at a caller-chosen cell (float position = cell)."""
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        if pos.shape[0] != self.B:
            raise ValueError(f"expected {self.B} positions, got {pos.shape[0]}")
        self._pos_f[:] = pos
        self._heading_rad[:] = 0.0
        self._update_snapped()

    def positions_continuous(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Raw float positions (B, 2) float."""
        if indices is None:
            return self._pos_f.copy()
        return self._pos_f[indices].copy()

    def obs_batch(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Views from the snapped cell, facing the continuous heading."""
        pos = self._pos if indices is None else self._pos[indices]
        return _views(self, pos, _obs_psi(self, indices))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step_batch(
        self,
        actions: np.ndarray,
        indices: np.ndarray | None = None,
        contract: GoalContract | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step with continuous (dx, dy) actions.

        Semantics at the goal are governed by ``contract`` -- see
        ``world/episode.py``. The default reproduces the historical bundling
        via ``goals_active``: the at-goal step is consumed (reward
        ``goal_reward``, movement ignored, teleport), giving the agent exactly
        one observable step at the goal in which to fire store, so that
        pre-action ``at_goal`` checks are meaningful.

        actions: (B, 2) float array.
        Returns (rewards, goal_reached, snapped_positions).
        """
        if indices is None:
            indices = np.arange(self.B)

        n = len(indices)
        if contract is None:
            contract = episode.contract_from_goals_active(self.goals_active)

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

        # The pre-action check is on the *continuous* position (_pos_f), via the
        # centralized at_goal helper, which auto-resolves to _pos_f here. What
        # follows from it is the contract's decision, not this method's.
        at_goal_mask = (at_goal(self)[indices].copy() if self.goals_active
                        else np.zeros(n, dtype=bool))
        res = episode.resolve_at_goal(
            at_goal_mask, contract,
            goal_reward=self.goal_reward, time_penalty=self.time_penalty,
        )
        rewards, goal_reached = res.rewards, res.at_goal

        moved = np.zeros((n, 2), dtype=np.float64)
        for j, b in enumerate(indices):
            if not res.apply_move[j]:
                continue
            before = self._pos_f[b].copy()
            self._pos_f[b] = np.clip(
                self._pos_f[b] + actions[j] * self.scale,
                0.0, float(self.size - 1),
            )
            moved[j] = self._pos_f[b] - before

        # Heading is the direction actually travelled -- a step absorbed by the
        # clip leaves the agent facing where it was. Continuous movement can
        # face any angle, so ψ is atan2 of the displacement rather than a
        # cardinal; atan2(dx, dy), in that argument order, is clockwise from
        # North, matching the ray convention.
        spun = np.linalg.norm(moved, axis=1) >= 1e-12
        if spun.any():
            self._heading_rad[indices[spun]] = np.arctan2(
                moved[spun, 0], moved[spun, 1])

        self._update_snapped(indices)

        # Recorded before the teleport below, so it is the displacement the
        # action produced and not the jump that followed it.
        self._last_displacement[:] = 0.0
        self._last_displacement[indices] = moved

        # Relocate the envs the contract says to teleport.
        reached_b = indices[res.teleport]
        if len(reached_b) > 0:
            self.reset_indices(reached_b)

        return rewards, goal_reached, self._pos[indices].copy()


_INHERIT = object()


def make_vec(
    env: GridEnv,
    batch: int,
    movement_mode: str,
    continuous_scale: float = 1.0,
    continuous_normalize: bool = False,
    *,
    reset: bool = True,
    min_action_norm: float | None = _INHERIT,
    max_action_norm: float | None = _INHERIT,
) -> VecEnv | ContinuousVecEnv:
    """Build the batched env matching ``movement_mode``.

    Lives beside the two classes it constructs, so every layer above can reach
    it without an upward import. There were three copies of this: train_rnn,
    eval_rnn, and baseline importing train_rnn's private one -- and they were
    not identical. train_rnn's reset positions before returning; eval_rnn's did
    not, because its caller places the agent explicitly. That difference is now
    the ``reset`` argument rather than a discrepancy between two functions with
    the same name.

    The action-norm bounds default to **whatever the wrapped env carries**
    rather than to None. They used not to be forwarded at all, which made this
    function silently drop them: training builds ContinuousVecEnv directly and
    passes both, while every eval path comes through here, so setting the
    bounds clamped training and left every reported number unclamped. Reading
    them off ``env`` fixes that for all five call sites at once and, unlike an
    argument each caller must remember, cannot be forgotten by the next one --
    a batched env that moved differently from the single env it wraps would be
    a contradiction in terms. Pass an explicit value (including None) to
    override.
    """
    if min_action_norm is _INHERIT:
        min_action_norm = getattr(env, "min_action_norm", None)
    if max_action_norm is _INHERIT:
        max_action_norm = getattr(env, "max_action_norm", None)
    if movement_mode == "continuous":
        vec = ContinuousVecEnv(
            env, batch_size=batch, scale=continuous_scale,
            normalize=continuous_normalize,
            min_action_norm=min_action_norm,
            max_action_norm=max_action_norm,
        )
    else:
        vec = VecEnv(env, batch_size=batch)
    if reset:
        vec.reset_all()
    return vec
