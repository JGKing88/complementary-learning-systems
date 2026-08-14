"""VectorHash: the grid/place/sensory scaffold field, plus per-env-set placement.

The module holds two things that used to be one class, split along the boundary
its own methods already drew:

``VectorHash``
    The **field**: everything that is a pure function of
    ``(lambdas, Npos, fwhm_ratio, encoder)`` -- ``gbook``, ``encoded_Phi``, and
    (non-static) ``pbook`` / ``Wpg`` / ``Wgp``. It knows nothing about
    environments, so one instance is shared by every world and every split. That
    matters: ``encoded_Phi`` is 12 GB at ``Npos=1716, out_dim=1024``, and
    building one per world was duplicating it bit-for-bit.

``EnvAssoc``
    The sensory<->place weights ``Wsp`` / ``Wps`` fitted to **one** env set, plus
    the ``recall`` path they serve. Non-static mode only; ``fit_env_assoc``
    returns ``None`` under ``static_vectorhash``, where ``register_envs`` never
    assigned them in the first place.

Env *offsets* live on neither. They are a property of an env set and are passed
around as a plain ``list[tuple[int, int]]`` -- the convention
``evaluation/protocols.py`` and ``evaluation/rnn.py`` already used.

No Hopfield management -- Hopfield is dynamic agent state managed by the rollout
collector.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from gridcode.assoc import (
    nonlin, train_pbook, train_gcpc, pseudotrain_Wsp, pseudotrain_Wps,
)
from gridcode.codebook import gen_gbook_2d

from ..config import VectorHashConfig
from ..utils import gram_schmidt_2d_batch, smooth_gbook
from .env import CARDINAL_ACTIONS

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
# Placement
# ------------------------------------------------------------------------------
#
# Free functions, not methods: an offset is a property of an env set, not of the
# field it indexes into. Both take an explicit ``rng`` so a caller can pin the
# stream. Passing the ``np.random`` *module* reproduces the historical behavior
# exactly (these drew from the global stream when they were methods), which is
# what the offsets-as-lists refactor needs in order to change nothing.


def random_offsets(n_envs: int, size: int, Npos: int, rng) -> list[tuple[int, int]]:
    """Rejection-sampled non-overlapping offsets."""
    used: list[tuple[int, int]] = []
    pairs: list[tuple[int, int]] = []
    for _ in range(100_000):
        if len(pairs) >= n_envs:
            break
        x = rng.randint(0, Npos - size + 1)
        y = rng.randint(0, Npos - size + 1)
        if all(not _overlaps(x, y, px, py, size) for (px, py) in used):
            used.append((x, y))
            pairs.append((x, y))
    return pairs


def spread_offsets(
    n_envs: int, size: int, Npos: int, rng, jitter: float = 0.4,
) -> list[tuple[int, int]]:
    """Roughly-uniform grid layout with random jitter within each cell.

    Places envs on a rows x cols lattice spanning [0, Npos - size] in each axis,
    then perturbs each offset by up to
        jitter * (spacing - size) / 2
    in each axis. With jitter in [0, 1] the post-jitter offsets are guaranteed
    non-overlapping.
    """
    max_off = Npos - size
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
            dx = rng.uniform(-j_x, j_x) if j_x > 0 else 0.0
            dy = rng.uniform(-j_y, j_y) if j_y > 0 else 0.0
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
                    f"(size={size}, Npos={Npos}, n_envs={n_envs}, "
                    f"jitter={jitter}). Scaffold too small."
                )
    return pairs


def place_envs(
    n_envs: int, size: int, Npos: int, rng, *,
    placement: str = "random", spread_jitter: float = 0.4,
) -> list[tuple[int, int]]:
    """Dispatch to a placement strategy and check that it placed everything."""
    if placement == "spread":
        pairs = spread_offsets(n_envs, size, Npos, rng, jitter=spread_jitter)
    elif placement == "random":
        pairs = random_offsets(n_envs, size, Npos, rng)
    else:
        raise ValueError(f"Unknown placement: {placement!r}")
    if len(pairs) < n_envs:
        raise RuntimeError(f"Could only place {len(pairs)}/{n_envs} envs.")
    return pairs


# ------------------------------------------------------------------------------
# VectorHash: the shared field
# ------------------------------------------------------------------------------

class VectorHash:
    """Grid/place scaffold with encoded_Phi and Gram-Schmidt projection.

    Lifecycle:
        1. __init__(cfg)
        2. build_scaffold()
        3. precompute_encoded_phi(encoder, ...)
        4. At runtime: gram_schmidt_projection(), get_encoded_state()

    If cfg.static_vectorhash: step 2 builds only gbook (no pbook / Wgp), and
    ``fit_env_assoc`` returns None, so there is no recall path. Sensory input
    (when enabled) is read directly from each env's own codebook, so no
    scaffold-sized sbook is needed here.

    Nothing on this object depends on which environments exist, which is what
    lets one instance back every world and every split.
    """

    def __init__(self, cfg: VectorHashConfig) -> None:
        self.cfg = cfg
        self.lambdas = list(cfg.lambdas)
        self.Np = cfg.Np
        self.Ng = int(np.sum(np.square(cfg.lambdas)))
        self.Npos = cfg.Npos if cfg.Npos is not None else int(np.prod(cfg.lambdas))
        self.thresh = cfg.thresh
        self.c = cfg.c

        # Above prod(lambdas) the grid code repeats: two distinct positions get
        # identical activity in *every* module, so the scaffold aliases outright
        # and no downstream separation check can see it. Npos=None resolves to
        # exactly prod(lambdas), which is the boundary and is fine.
        prod_lambdas = int(np.prod(cfg.lambdas))
        if self.Npos > prod_lambdas:
            raise ValueError(
                f"Npos={self.Npos} exceeds prod(lambdas)={prod_lambdas}: distinct "
                f"scaffold positions would share an identical grid code in every "
                f"module. Lower Npos or add a module to lambdas={list(cfg.lambdas)}."
            )

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
    # 2. Grid state → position
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
    # 3. Encoded space
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
        del sgb

        # Written straight into the destination rather than collected into a
        # list and concatenated. At Npos=1716 the result is 12 GB, so holding
        # the parts and their concatenation at once doubled that, and peak RSS
        # was 44 GB for a 12 GB answer -- which on a memory-contended node is
        # the difference between scheduling and queueing. The first chunk sizes
        # the buffer because embed_dim is the encoder's business, not ours.
        with torch.no_grad():
            for start in range(0, flat.shape[0], 1000):
                chunk = torch.from_numpy(flat[start:start + 1000]).to(device)
                out = encoder(chunk).cpu().numpy()
                if start == 0:
                    encoded = np.empty((flat.shape[0], out.shape[1]),
                                       dtype=out.dtype)
                encoded[start:start + out.shape[0]] = out
        del flat

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


# ------------------------------------------------------------------------------
# EnvAssoc: sensory <-> place weights, fitted to one env set
# ------------------------------------------------------------------------------

class EnvAssoc:
    """``Wsp`` / ``Wps`` for one env set, and the recall path they serve.

    Built only in non-static mode -- ``register_envs`` never assigned these
    under ``static_vectorhash``, and no current run uses the non-static path
    (``pbook`` alone is 37.7 GB at ``Npos=1716``).

    One instance per env set, **never merged across sets**: the self-test in
    ``fit_env_assoc`` scales with the number of registered patterns, so fitting
    train and val envs together would face a harder disambiguation problem than
    fitting each alone and could fail where two separate fits pass.
    """

    def __init__(self, field: VectorHash, Wsp: np.ndarray, Wps: np.ndarray,
                 Ns: int) -> None:
        self.field = field
        self.Wsp = Wsp
        self.Wps = Wps
        self.Ns = Ns

    def recall(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single observation recall: obs -> (s, p, g).

        obs: (Ns,) binary observation.
        Returns (s_out, p_out, g_out) numpy arrays.
        """
        f = self.field
        # No second nonlin: Wps reconstructs already-thresholded pbook values.
        # Re-thresholding destroys the signal (double threshold bug).
        pin = self.Wps @ obs
        gin = f.Wgp @ pin

        # Module-wise winner-take-all
        gout = np.zeros_like(gin)
        idx = 0
        for j in f.module_sizes:
            gmod = gin[idx:idx + j]
            gout[gmod.argmax() + idx] = 1
            idx += j

        pout = nonlin(f.Wpg @ gout, thresh=f.thresh)
        sout = (self.Wsp @ pout > 0).astype(float)
        return sout, pout, gout

    def recall_batch(self, obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched recall.  obs_batch: (B, Ns).  Returns (s, p, g) each (B, dim)."""
        f = self.field
        S = obs_batch.T  # (Ns, B)
        pin = self.Wps @ S  # no second nonlin
        gin = f.Wgp @ pin

        gout = np.zeros_like(gin)
        idx = 0
        B = S.shape[1]
        for j in f.module_sizes:
            gmod = gin[idx:idx + j]
            maxes = gmod.argmax(axis=0)
            gout[maxes + idx, np.arange(B)] = 1
            idx += j

        pout = nonlin(f.Wpg @ gout, thresh=f.thresh)
        sout = (self.Wsp @ pout > 0).astype(float)
        return sout.T, pout.T, gout.T  # (B, dim) each


def fit_env_assoc(
    field: VectorHash,
    envs: list[GridEnv],
    offsets: list[tuple[int, int]],
) -> EnvAssoc | None:
    """Fit ``Wsp`` / ``Wps`` to this env set and validate grid recovery.

    Returns ``None`` under ``static_vectorhash`` -- there is no recall path in
    that mode, and the historical ``register_envs`` simply skipped this work.

    **The observation this fits is not the one the agent sees.** Under
    egocentric heading a cell has no single appearance -- the view rotates with
    the agent, continuously -- so "one sbook column per position" is no longer
    well defined, and one column per (position, heading) is unbounded. The patch
    is to give each position the concatenation of its four *cardinal* views:

        sbook[:, k] = concat(view(pos_k, N), view(pos_k, E),
                             view(pos_k, S), view(pos_k, W))

    which keeps ``Npatts`` at one per position and widens ``Ns`` to
    ``4 * observation_size``. It is a stand-in, and worth being honest about:
    the agent's real input is a *single* view of width ``observation_size`` at a
    continuous angle, so this vector is not something it can ever observe in one
    step. ``EnvAssoc.recall`` therefore takes the concatenation, not a live
    observation -- see ``GridEnv.omni_obs_at``. Nothing on the navigation path
    depends on this (the collector keys off positions, not observations); it is
    the scaffold's own sensory<->place association.

    A fixed-heading env keeps the historical North-only sbook, so that mode
    reproduces its previous fit exactly.

    **Why not map the four views to one place code instead?** That is the
    obvious alternative -- each cardinal view its own sbook column at the width
    the agent actually observes, all four targeting the same p/g column, a
    genuinely heading-invariant readout. It was measured, and it does not work:

        size=8, Np=1600, 4 envs, obs=60      grid recovery
          north-only baseline (Ns=60)             39.8%
          four views -> one state  (Ns=60)         7.3%
          concatenation            (Ns=240)       96.9%

    Nor is it a question of width -- at Ns=240, the *same* input width the
    concatenation gets, the shared-target fit still scores 12.5%. The obstruction
    is that a cell's four views have mean pairwise cosine +0.028 while views of
    *different* cells average +0.046: a cell's own views are no more alike than
    two random cells'. The cone is 120 deg, so headings 90 deg apart share 30 deg
    of arc and headings 180 deg apart share none -- opposite views have no ray in
    common and are looks at different walls, not one place seen twice. Asking a
    linear map to collapse them is asking it to annihilate 6 difference vectors
    per cell in a <=240-dimensional input space, which is over-constrained by
    roughly 6x, so the pseudoinverse averages and recovers nothing.

    Concatenation works precisely because it never asks the map to *discard*
    heading; it supplies every view at once, so heading is unambiguous rather
    than marginalized. A single-view invariant readout would need a cone wider
    than 180 deg (so any two headings overlap), an explicit heading input, or a
    nonlinear encoder -- none of which this linear scaffold has.
    """
    if field.cfg.static_vectorhash:
        print("  fit_env_assoc: static_vectorhash — skipping Wsp/Wps and scaffold test")
        return None

    all_locs: list[np.ndarray] = []
    all_obs: list[np.ndarray] = []

    for env_idx, env in enumerate(envs):
        # One entry per cell, in the env RNG's shuffled order. Any single
        # heading dedups the four entries a cell contributes; East is the one
        # the pre-heading code filtered on, and it is kept so that neither the
        # pattern ordering nor the env RNG's consumption moves here.
        pos_obs_head = [p for p in env.fully_explore_random()
                        if p[2] == CARDINAL_ACTIONS[1]]
        cells = [p[0] for p in pos_obs_head]

        if getattr(env, "egocentric_heading", True):
            obs = np.array([env.omni_obs_at(p) for p in cells])
        else:
            obs = np.array([env.obs_at(p, psi=0.0) for p in cells])

        locs = np.array(cells)
        C_X, C_Y = offsets[env_idx]
        locs[:, 0] += C_X
        locs[:, 1] += C_Y
        all_locs.append(locs)
        all_obs.append(obs)

    all_locs_arr = np.concatenate(all_locs)
    all_obs_arr = np.concatenate(all_obs)
    sbook = all_obs_arr.T  # (Ns, Npatts)

    Npatts = len(all_locs_arr)
    path_pbook = np.zeros((field.Np, Npatts))
    path_gbook = np.zeros((field.Ng, Npatts))
    for k, loc in enumerate(all_locs_arr):
        path_pbook[:, k] = field.pbook[:, loc[0], loc[1]]
        path_gbook[:, k] = field.gbook[:, loc[0], loc[1]]

    print("  fit_env_assoc: pseudotrain_Wsp")
    Wsp = pseudotrain_Wsp(sbook, path_pbook, Npatts)
    print("  fit_env_assoc: pseudotrain_Wps")
    Wps = pseudotrain_Wps(path_pbook, sbook, Npatts)

    assoc = EnvAssoc(field, Wsp, Wps, Ns=sbook.shape[0])
    _test_assoc(assoc, sbook, path_gbook)
    return assoc


def _test_assoc(assoc: EnvAssoc, sbook: np.ndarray, path_gbook: np.ndarray) -> None:
    """Validate that obs → recall → g recovers the correct grid state."""
    Npatts = sbook.shape[1]
    correct = 0
    for k in range(Npatts):
        s, p, g = assoc.recall(sbook[:, k])
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


# ------------------------------------------------------------------------------
# Goal patterns (for pre_stored mode)
# ------------------------------------------------------------------------------

def goal_encodings(
    field: VectorHash,
    envs: list[GridEnv],
    offsets: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Encoded goal pattern for each env.  Returns list of (embed_dim,) arrays."""
    patterns = []
    for env_idx, env in enumerate(envs):
        ox, oy = offsets[env_idx]
        gx = min(max(env.goal_location[0] + ox, 0), field.Npos - 1)
        gy = min(max(env.goal_location[1] + oy, 0), field.Npos - 1)
        patterns.append(field.encoded_Phi[gx, gy])
    return patterns


__all__ = [
    "EnvAssoc",
    "VectorHash",
    "fit_env_assoc",
    "goal_encodings",
    "place_envs",
    "random_offsets",
    "spread_offsets",
]
