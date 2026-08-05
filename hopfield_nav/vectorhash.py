"""VectorHash: grid/place/sensory scaffold + encoded_Phi + Gram-Schmidt.

No Hopfield management — Hopfield is dynamic agent state managed by the
rollout collector.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from cls.vectorhash.assoc_utils_np import (
    nonlin, train_pbook, train_gcpc, pseudotrain_Wsp, pseudotrain_Wps,
)
from cls.vectorhash.assoc_utils_np_2D import gen_gbook_2d

from .config import VectorHashConfig
from .utils import gram_schmidt_2d_batch, smooth_gbook

if TYPE_CHECKING:
    from .env import GridEnv

# Helpers from existing code ---------------------------------------------------

_randn = np.random.randn
_randint = np.random.randint


def _overlaps(x1, y1, x2, y2, size, touch_ok=True):
    gap = 0 if touch_ok else 1
    return not (x1 + size + gap <= x2 or x2 + size + gap <= x1 or
                y1 + size + gap <= y2 or y2 + size + gap <= y1)


# ------------------------------------------------------------------------------
# VectorHash
# ------------------------------------------------------------------------------

class VectorHash:
    """Grid/place/sensory scaffold with encoded_Phi and Gram-Schmidt projection.

    Lifecycle:
        1. __init__(cfg)
        2. build_scaffold()
        3. register_envs(envs)
        4. precompute_encoded_phi(encoder, ...)
        5. At runtime: recall(), gram_schmidt_projection(), get_encoded_state()

    If cfg.static_vectorhash: steps 2–3 build only gbook and env offsets (no
    pbook / Wgp / Wsp / self-test); recall() is unavailable. Sensory input
    (when enabled) is read directly from each env's own codebook, so no
    scaffold-sized sbook is needed here.
    """

    def __init__(self, cfg: VectorHashConfig, size: int) -> None:
        self.cfg = cfg
        self.lambdas = list(cfg.lambdas)
        self.Np = cfg.Np
        self.Ng = int(np.sum(np.square(cfg.lambdas)))
        self.Npos = cfg.Npos if cfg.Npos is not None else int(np.prod(cfg.lambdas))
        self.size = size
        self.thresh = cfg.thresh
        self.c = cfg.c

        self.env_offsets: list[tuple[int, int]] = []
        self.encoded_Phi: np.ndarray | None = None  # (Npos, Npos, embed_dim)

    # ------------------------------------------------------------------
    # 1. Scaffold
    # ------------------------------------------------------------------

    def build_scaffold(self) -> None:
        """Generate gbook, pbook, Wpg, Wgp (or only gbook when cfg.static_vectorhash)."""
        lambdas = self.lambdas
        print("  build_scaffold: gen_gbook_2d")
        self.gbook = gen_gbook_2d(lambdas, self.Ng, self.Npos)  # (Ng, Npos, Npos)
        self.module_sizes = [l ** 2 for l in lambdas]
        if self.cfg.static_vectorhash:
            print("  build_scaffold: static_vectorhash — skipping pbook / Wgp")
            return

        Wpg = _randn(self.Np, self.Ng)
        prune = int((1 - self.c) * self.Np * self.Ng)
        mask = np.ones((self.Np, self.Ng))
        mask[_randint(0, self.Np, prune), _randint(0, self.Ng, prune)] = 0
        Wpg = mask * Wpg

        print("  build_scaffold: train_pbook")
        self.pbook = nonlin(train_pbook(Wpg, self.gbook), thresh=self.thresh)
        self.Wpg = Wpg

        gbook_flat = self.gbook.reshape(self.Ng, -1)
        pbook_flat = self.pbook.reshape(self.Np, -1)

        print("  build_scaffold: train_gcpc (Wgp)")
        self.Wgp = train_gcpc(pbook_flat, gbook_flat, Npatts=self.Npos ** 2)

    # ------------------------------------------------------------------
    # 2. Register environments
    # ------------------------------------------------------------------

    def _random_offsets(self, n_envs: int, size: int) -> list[tuple[int, int]]:
        used: list[tuple[int, int]] = []
        pairs: list[tuple[int, int]] = []
        for _ in range(100_000):
            if len(pairs) >= n_envs:
                break
            x = np.random.randint(0, self.Npos - size + 1)
            y = np.random.randint(0, self.Npos - size + 1)
            if all(not _overlaps(x, y, px, py, size) for (px, py) in used):
                used.append((x, y))
                pairs.append((x, y))
        return pairs

    def _spread_offsets(
        self, n_envs: int, size: int, jitter: float = 0.4,
    ) -> list[tuple[int, int]]:
        """Roughly-uniform grid layout with random jitter within each cell.

        Places envs on a rows x cols lattice spanning [0, Npos - size] in
        each axis, then perturbs each offset by up to
            jitter * (spacing - size) / 2
        in each axis. With jitter in [0, 1] the post-jitter offsets are
        guaranteed non-overlapping. Uses global np.random (seeded by the
        caller), so results are deterministic per np seed.
        """
        max_off = self.Npos - size
        if n_envs <= 0:
            return []
        if n_envs == 1:
            return [(max_off // 2, max_off // 2)]

        # rows x cols layout covering n_envs slots, rows <= cols.
        rows = max(int(np.floor(np.sqrt(n_envs))), 1)
        cols = int(np.ceil(n_envs / rows))

        # Per-axis spacing of the ideal lattice (0 if only 1 row/col).
        sp_x = max_off / (rows - 1) if rows > 1 else 0.0
        sp_y = max_off / (cols - 1) if cols > 1 else 0.0

        # Max jitter per axis that keeps envs non-overlapping (half the slack,
        # scaled by the jitter fraction). Clamp to 0 if spacing <= size.
        j_x = max(0.0, (sp_x - size) / 2.0) * jitter
        j_y = max(0.0, (sp_y - size) / 2.0) * jitter

        # Shrink the lattice bounds by the jitter radius so jitter can't push
        # past [0, max_off].
        xs = (np.linspace(j_x, max_off - j_x, rows) if rows > 1
              else np.array([max_off / 2.0]))
        ys = (np.linspace(j_y, max_off - j_y, cols) if cols > 1
              else np.array([max_off / 2.0]))

        pairs: list[tuple[int, int]] = []
        for x in xs:
            for y in ys:
                if len(pairs) >= n_envs:
                    break
                dx = np.random.uniform(-j_x, j_x) if j_x > 0 else 0.0
                dy = np.random.uniform(-j_y, j_y) if j_y > 0 else 0.0
                pairs.append((int(round(x + dx)), int(round(y + dy))))
            if len(pairs) >= n_envs:
                break

        # Sanity: verify non-overlap. Guaranteed with jitter <= 1 and
        # spacing > size, but guard against degenerate Npos/size combos.
        for i, (xi, yi) in enumerate(pairs):
            for (xj, yj) in pairs[:i]:
                if _overlaps(xi, yi, xj, yj, size):
                    raise RuntimeError(
                        f"Spread placement produced overlapping envs "
                        f"(size={size}, Npos={self.Npos}, n_envs={n_envs}, "
                        f"jitter={jitter}). Scaffold too small."
                    )
        return pairs

    def register_envs(
        self, envs: list[GridEnv], placement: str = "random",
        spread_jitter: float = 0.4,
    ) -> None:
        """Explore envs, place them in the grid, build Wsp/Wps.

        placement:
          - "random": rejection-sampled non-overlapping offsets (default).
          - "spread": rows x cols lattice spanning the scaffold with random
            jitter per offset so envs are roughly uniformly distributed but
            not on a perfect grid. spread_jitter in [0, 1] scales the
            per-axis perturbation; 0 = exact lattice, 1 = maximum safe
            jitter (post-jitter envs still guaranteed non-overlapping).
        """
        n_envs = len(envs)
        size = self.size

        if placement == "spread":
            pairs = self._spread_offsets(n_envs, size, jitter=spread_jitter)
        elif placement == "random":
            pairs = self._random_offsets(n_envs, size)
        else:
            raise ValueError(f"Unknown placement: {placement!r}")
        if len(pairs) < n_envs:
            raise RuntimeError(f"Could only place {len(pairs)}/{n_envs} envs.")

        if self.cfg.static_vectorhash:
            self.env_offsets = [pairs[i] for i in range(n_envs)]
            print("  register_envs: static_vectorhash — skipping Wsp/Wps and scaffold test")
            return

        all_locs: list[np.ndarray] = []
        all_obs: list[np.ndarray] = []
        self.env_offsets = []

        for env_idx, env in enumerate(envs):
            pos_obs_head = env.fully_explore_random()
            # Heading-invariant: take one heading per position
            pos_obs_head = [p for p in pos_obs_head if p[2] == (1, 0)]

            locs = np.array([p[0] for p in pos_obs_head])
            obs = np.array([p[1] for p in pos_obs_head])
            C_X, C_Y = pairs[env_idx]
            self.env_offsets.append((C_X, C_Y))
            locs[:, 0] += C_X
            locs[:, 1] += C_Y
            all_locs.append(locs)
            all_obs.append(obs)

        all_locs_arr = np.concatenate(all_locs)
        all_obs_arr = np.concatenate(all_obs)
        sbook = all_obs_arr.T  # (Ns, Npatts)

        Npatts = len(all_locs_arr)
        path_pbook = np.zeros((self.Np, Npatts))
        path_gbook = np.zeros((self.Ng, Npatts))
        for k, loc in enumerate(all_locs_arr):
            path_pbook[:, k] = self.pbook[:, loc[0], loc[1]]
            path_gbook[:, k] = self.gbook[:, loc[0], loc[1]]

        print("  register_envs: pseudotrain_Wsp")
        self.Wsp = pseudotrain_Wsp(sbook, path_pbook, Npatts)
        print("  register_envs: pseudotrain_Wps")
        self.Wps = pseudotrain_Wps(path_pbook, sbook, Npatts)

        self.Ns = sbook.shape[0]

        # Test scaffold
        self._test_scaffold(sbook, path_gbook)

    # ------------------------------------------------------------------
    # 3. Recall
    # ------------------------------------------------------------------

    def recall(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single observation recall: obs -> (s, p, g).

        obs: (Ns,) binary observation.
        Returns (s_out, p_out, g_out) numpy arrays.
        """
        if self.cfg.static_vectorhash:
            raise RuntimeError("VectorHash.recall is unavailable when cfg.static_vectorhash is True")
        # No second nonlin: Wps reconstructs already-thresholded pbook values.
        # Re-thresholding destroys the signal (double threshold bug).
        pin = self.Wps @ obs
        gin = self.Wgp @ pin

        # Module-wise winner-take-all
        gout = np.zeros_like(gin)
        idx = 0
        for j in self.module_sizes:
            gmod = gin[idx:idx + j]
            gout[gmod.argmax() + idx] = 1
            idx += j

        pout = nonlin(self.Wpg @ gout, thresh=self.thresh)
        sout = (self.Wsp @ pout > 0).astype(float)
        return sout, pout, gout

    def recall_batch(self, obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched recall.  obs_batch: (B, Ns).  Returns (s, p, g) each (B, dim)."""
        if self.cfg.static_vectorhash:
            raise RuntimeError(
                "VectorHash.recall_batch is unavailable when cfg.static_vectorhash is True"
            )
        S = obs_batch.T  # (Ns, B)
        pin = self.Wps @ S  # no second nonlin
        gin = self.Wgp @ pin

        gout = np.zeros_like(gin)
        idx = 0
        B = S.shape[1]
        for j in self.module_sizes:
            gmod = gin[idx:idx + j]
            maxes = gmod.argmax(axis=0)
            gout[maxes + idx, np.arange(B)] = 1
            idx += j

        pout = nonlin(self.Wpg @ gout, thresh=self.thresh)
        sout = (self.Wsp @ pout > 0).astype(float)
        return sout.T, pout.T, gout.T  # (B, dim) each

    # ------------------------------------------------------------------
    # 4. Grid state → position
    # ------------------------------------------------------------------

    def g_to_position(self, g: np.ndarray) -> np.ndarray:
        """Convert grid one-hot (B, Ng) to global (gx, gy) positions (B, 2).

        Uses the first module's one-hot to determine (x % lambda, y % lambda),
        then looks up the full position from gbook.
        """
        B = g.shape[0]
        positions = np.zeros((B, 2), dtype=np.int32)
        for b in range(B):
            # Match against gbook columns
            diffs = np.abs(self.gbook.reshape(self.Ng, -1).T - g[b]).sum(axis=1)
            best = diffs.argmin()
            positions[b, 0] = best // self.Npos
            positions[b, 1] = best % self.Npos
        return positions

    def g_to_position_fast(self, g: np.ndarray) -> np.ndarray:
        """Fast version: decode each module's one-hot to get position mod lambda.

        Returns (B, 2) global positions.  Only exact for grid book positions.
        """
        B = g.shape[0]
        # Use first module to get (x mod lambda0, y mod lambda0)
        l0 = self.lambdas[0]
        n0 = l0 ** 2
        g0 = g[:, :n0].reshape(B, l0, l0)
        flat0 = g0.reshape(B, -1)
        active = flat0.argmax(axis=1)
        pos_x = active // l0  # x mod lambda0
        pos_y = active % l0   # y mod lambda0

        # For multi-module, we'd need CRT to recover full position.
        # For now, scan the gbook for exact match (still fast for small Npos).
        positions = np.zeros((B, 2), dtype=np.int32)
        for b in range(B):
            # Check all (x, y) where x mod l0 == pos_x[b] and y mod l0 == pos_y[b]
            for x in range(pos_x[b], self.Npos, l0):
                for y in range(pos_y[b], self.Npos, l0):
                    if np.array_equal(self.gbook[:, x, y], g[b]):
                        positions[b] = [x, y]
                        break
        return positions

    # ------------------------------------------------------------------
    # 5. Encoded space
    # ------------------------------------------------------------------

    def precompute_encoded_phi(
        self,
        encoder: torch.nn.Module,
        fwhm_ratio: float,
        device: str | torch.device = "cpu",
    ) -> None:
        """Encode all gbook positions → self.encoded_Phi (Npos, Npos, embed_dim)."""
        Npos = self.Npos
        if fwhm_ratio > 0:
            sgb = smooth_gbook(self.gbook, self.lambdas, fwhm_ratio)
        else:
            sgb = self.gbook.copy()

        flat = sgb.reshape(self.Ng, Npos * Npos).T.astype(np.float32)
        parts = []
        with torch.no_grad():
            for start in range(0, flat.shape[0], 1000):
                chunk = torch.from_numpy(flat[start:start + 1000]).to(device)
                parts.append(encoder(chunk).cpu().numpy())

        encoded = np.concatenate(parts, axis=0)
        self.encoded_Phi = encoded.reshape(Npos, Npos, -1)
        print(f"  precomputed encoded_Phi: {self.encoded_Phi.shape}")

    def get_encoded_state(
        self, positions: np.ndarray, env_offset: tuple[int, int]
    ) -> np.ndarray:
        """Look up encoded states for positions.

        positions: (B, 2) local env coords.
        Returns (B, embed_dim).
        """
        gx = np.clip(positions[:, 0] + env_offset[0], 0, self.Npos - 1)
        gy = np.clip(positions[:, 1] + env_offset[1], 0, self.Npos - 1)
        return self.encoded_Phi[gx, gy]

    def get_store_patterns(
        self,
        positions: np.ndarray,
        env_offset: tuple[int, int],
        *,
        at_goal_mask: np.ndarray | None = None,
        goal: tuple[int, int] | None = None,
        allow_offcell: bool = True,
    ) -> np.ndarray:
        """Patterns a store action writes, one row per batch element.

        Normally this is exactly ``get_encoded_state`` -- the encoded state of
        the cell the agent is standing on. It differs only in the off-cell case:
        with ``goal_radius > 0.5`` in continuous mode, ``at_goal`` tests the
        float position while embeddings are read at the snapped cell, so an
        agent can be "at goal" while standing on a neighbour, and a store there
        would write the neighbour's embedding as the goal memory.

        ``allow_offcell=True`` (the default, and every run through 2026-08)
        keeps that. ``allow_offcell=False`` substitutes ``encoded_Phi`` at the
        goal cell for the rows that are at goal, so the stored pattern is the
        one navigation will later recall.

        positions: (B, 2) local env coords.
        at_goal_mask: (B,) bool, or None to mean "no row is at goal".
        goal: local goal coords, required when suppressing off-cell stores.
        Returns (B, embed_dim).
        """
        patterns = self.get_encoded_state(positions, env_offset)
        if allow_offcell or at_goal_mask is None or goal is None:
            return patterns
        at_goal_mask = np.asarray(at_goal_mask, dtype=bool).reshape(-1)
        offcell = at_goal_mask & (
            (positions[:, 0] != goal[0]) | (positions[:, 1] != goal[1])
        )
        if not offcell.any():
            return patterns
        goal_arr = np.array([goal], dtype=np.int32)
        goal_pattern = self.get_encoded_state(goal_arr, env_offset)[0]
        patterns = patterns.copy()
        patterns[offcell] = goal_pattern
        return patterns

    def gram_schmidt_projection(
        self,
        positions: np.ndarray,
        env_offset: tuple[int, int],
        cached_W: np.ndarray | None = None,
        recompute_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute local 2D projection matrices at given positions.

        positions: (B, 2) local env coords.
        Returns W: (B, 2, embed_dim).
        """
        C_X, C_Y = env_offset
        Npos = self.encoded_Phi.shape[0]
        embed_dim = self.encoded_Phi.shape[2]
        B = positions.shape[0]

        gx = np.clip(positions[:, 0] + C_X, 1, Npos - 2)
        gy = np.clip(positions[:, 1] + C_Y, 1, Npos - 2)
        current = self.encoded_Phi[gx, gy]

        if cached_W is None:
            W = np.zeros((B, 2, embed_dim), dtype=np.float32)
            recompute_mask = np.ones(B, dtype=bool)
        else:
            W = cached_W.copy()
            if recompute_mask is None:
                recompute_mask = np.ones(B, dtype=bool)

        rc = np.where(recompute_mask)[0]
        if len(rc) > 0:
            d_fwd = self.encoded_Phi[gx[rc], gy[rc] + 1] - current[rc]
            d_rgt = self.encoded_Phi[gx[rc] + 1, gy[rc]] - current[rc]
            W[rc] = gram_schmidt_2d_batch(d_fwd, d_rgt)

        return W

    def project_displacement(
        self, current: np.ndarray, recalled: np.ndarray, W: np.ndarray
    ) -> np.ndarray:
        """Project (recalled - current) through W.

        current, recalled: (B, embed_dim).  W: (B, 2, embed_dim).
        Returns (B, 2).
        """
        displacement = recalled - current
        return np.einsum('bij,bj->bi', W, displacement)

    # ------------------------------------------------------------------
    # 6. Goal pattern storage (for pre_stored mode)
    # ------------------------------------------------------------------

    def get_goal_encodings(self, envs: list[GridEnv]) -> list[np.ndarray]:
        """Get encoded goal patterns for a list of environments.

        Returns list of (embed_dim,) numpy arrays.
        """
        patterns = []
        for env_idx, env in enumerate(envs):
            offset = self.env_offsets[env_idx]
            gx = min(max(env.goal_location[0] + offset[0], 0), self.Npos - 1)
            gy = min(max(env.goal_location[1] + offset[1], 0), self.Npos - 1)
            patterns.append(self.encoded_Phi[gx, gy])
        return patterns

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _test_scaffold(self, sbook: np.ndarray, path_gbook: np.ndarray) -> None:
        """Validate that obs → recall → g recovers the correct grid state."""
        Npatts = sbook.shape[1]
        correct = 0
        for k in range(Npatts):
            s, p, g = self.recall(sbook[:, k])
            if np.array_equal(g, path_gbook[:, k]):
                correct += 1
        accuracy = correct / Npatts
        print(f"  scaffold test: {correct}/{Npatts} grid recovery ({accuracy:.1%})")
        if accuracy < 0.95:
            raise RuntimeError(
                f"Grid recovery only {accuracy:.1%}, expected >95%. "
                "Try increasing Np or observation_size."
            )
        elif accuracy < 1.0:
            import warnings
            warnings.warn(f"Grid recovery {accuracy:.1%} (not perfect). {Npatts - correct}/{Npatts} failed.")
