"""Sensory-keyed argmax goal memory (idea 1, docs/GRID_MLP_NAV_PLAN.md §1–2).

Write (oracle store at the goal): key = the four-heading view at the goal,
re-indexed by *absolute direction* into ``PANORAMA_BINS`` slices (a cell's
view of the world as a function of compass direction; the cones' overlaps are
averaged); value = the goal's grid code.

Read (every step): the live single view, ray-cast at absolute heading ψ,
covers the slices its rays point into. Each stored key is compared with the
view on exactly those slices (cosine); the best match returns its grid code
and similarity ``s``. An empty memory returns zeros and ``s = 0``.

Only meaningful with the distal panorama on (``EnvConfig.distal_amp``): the
near walls alone decorrelate within one cell step, so no key matches away from
the goal (plan §2.1–2.3). Offline, at ``distal_amp = 1``, this read picks the
own env's goal from every cell at N = 100 stored goals with 96% accuracy, own
similarity ≈ 0.50 against ≈ 0.32 for the best foreign key (§2.4).
"""
from __future__ import annotations

import numpy as np

from ..world.env import CARDINAL_RADIANS, PANORAMA_BINS, cone_offsets, panorama_bins

_EPS = 1e-8


def omni_key(env, pos) -> np.ndarray:
    """The four-heading view at ``pos`` as a ``(PANORAMA_BINS,)`` function of
    absolute direction. Slices seen by two cones average their two readings."""
    n = env._observation_size
    bins = panorama_bins((CARDINAL_RADIANS[:, None] + cone_offsets(n)[None, :]).ravel())
    view = env.omni_obs_at(tuple(int(p) for p in pos))
    total = np.bincount(bins, weights=view, minlength=PANORAMA_BINS)
    count = np.bincount(bins, minlength=PANORAMA_BINS)
    return (total / np.maximum(count, 1)).astype(np.float32)


class SensoryKVMemory:
    """One trajectory's store: parallel lists of keys and grid-code values."""

    def __init__(self, key_dim: int = PANORAMA_BINS, value_dim: int | None = None) -> None:
        self.key_dim = int(key_dim)
        self.value_dim = value_dim
        self.keys: list[np.ndarray] = []
        self.values: list[np.ndarray] = []

    @property
    def num_memories(self) -> int:
        return len(self.keys)

    def write(self, key: np.ndarray, value: np.ndarray) -> None:
        key = np.asarray(key, dtype=np.float32).reshape(-1)
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if key.shape[0] != self.key_dim:
            raise ValueError(f"key width {key.shape[0]} != {self.key_dim}")
        if self.value_dim is None:
            self.value_dim = value.shape[0]
        elif value.shape[0] != self.value_dim:
            raise ValueError(f"value width {value.shape[0]} != {self.value_dim}")
        self.keys.append(key)
        self.values.append(value)

    def copy(self) -> "SensoryKVMemory":
        m = SensoryKVMemory(self.key_dim, self.value_dim)
        m.keys, m.values = list(self.keys), list(self.values)
        return m


def read_batch(memories: list[SensoryKVMemory], views: np.ndarray,
               psi: np.ndarray, value_dim: int):
    """Argmax read for B trajectories at once.

    views: (B, n_rays) live views; psi: (B,) absolute headings they were cast
    at. Returns ``(values (B, value_dim), s (B,), has_memory (B,) bool)``.
    """
    views = np.asarray(views, dtype=np.float32)
    B, n = views.shape
    counts = np.array([m.num_memories for m in memories])
    out_v = np.zeros((B, value_dim), dtype=np.float32)
    out_s = np.zeros(B, dtype=np.float32)
    has = counts > 0
    if not has.any():
        return out_v, out_s, has
    K = int(counts.max())
    keys = np.zeros((B, K, PANORAMA_BINS), dtype=np.float32)
    vals = np.zeros((B, K, value_dim), dtype=np.float32)
    valid = np.zeros((B, K), dtype=bool)
    for b, m in enumerate(memories):
        if m.num_memories:
            keys[b, :m.num_memories] = np.stack(m.keys)
            vals[b, :m.num_memories] = np.stack(m.values)
            valid[b, :m.num_memories] = True
    bins = panorama_bins(np.asarray(psi, dtype=np.float64)[:, None]
                         + cone_offsets(n)[None, :])                  # (B, n)
    k_seen = np.take_along_axis(keys, bins[:, None, :], axis=2)       # (B, K, n)
    q = views / (np.linalg.norm(views, axis=1, keepdims=True) + _EPS)
    k_seen = k_seen / (np.linalg.norm(k_seen, axis=2, keepdims=True) + _EPS)
    sims = np.einsum("bn,bkn->bk", q, k_seen)
    sims = np.where(valid, sims, -np.inf)
    best = sims.argmax(axis=1)
    rows = np.arange(B)
    out_s[has] = sims[rows, best][has]
    out_v[has] = vals[rows, best][has]
    return out_v, out_s, has
