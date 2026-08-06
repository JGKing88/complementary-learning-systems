"""Shared utilities: Gram-Schmidt, direction classification, smoothing."""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Gram-Schmidt projection
# ---------------------------------------------------------------------------

def gram_schmidt_2d_batch(
    d_forward: np.ndarray,
    d_right: np.ndarray,
) -> np.ndarray:
    """Orthonormalize two displacement vectors into a 2D basis via Gram-Schmidt.

    d_forward: (B, D) — typically the North neighbor displacement.
    d_right:   (B, D) — typically the East neighbor displacement.
    Returns W: (B, 2, D) where W[:,0,:] = East basis, W[:,1,:] = North basis.
    """
    # e1 = normalize(d_forward)  — North basis
    norm1 = np.linalg.norm(d_forward, axis=-1, keepdims=True).clip(1e-12)
    e1 = d_forward / norm1

    # e2 = normalize(d_right - proj(d_right, e1))  — East basis
    proj = np.sum(d_right * e1, axis=-1, keepdims=True) * e1
    e2_raw = d_right - proj
    norm2 = np.linalg.norm(e2_raw, axis=-1, keepdims=True).clip(1e-12)
    e2 = e2_raw / norm2

    # Stack: row 0 = East, row 1 = North
    return np.stack([e2, e1], axis=1)  # (B, 2, D)


def classify_direction_batch(q: np.ndarray) -> np.ndarray:
    """Classify 2D vectors into cardinal directions.

    q: (B, 2) where q[:,0] = East component, q[:,1] = North component.
    Returns direction indices (B,):  0=N, 1=E, 2=S, 3=W.
    """
    angles = np.arctan2(q[:, 1], q[:, 0])
    # Bin into 4 quadrants centered on each cardinal direction
    idx = np.full(angles.shape, 3, dtype=np.int32)           # default W
    idx[(-np.pi / 4 <= angles) & (angles < np.pi / 4)] = 1   # E
    idx[(np.pi / 4 <= angles) & (angles < 3 * np.pi / 4)] = 0  # N
    idx[(-3 * np.pi / 4 <= angles) & (angles < -np.pi / 4)] = 2  # S
    return idx


def direction_to_onehot(idx: np.ndarray) -> np.ndarray:
    """Convert direction indices (B,) to one-hot (B, 4)."""
    B = idx.shape[0]
    onehot = np.zeros((B, 4), dtype=np.float32)
    onehot[np.arange(B), idx] = 1.0
    return onehot


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
# Re-exported from gridcode so that `from hopfield_nav.utils import smooth_gbook`
# keeps working for the callers that already say it. Until phase 7 these were
# shims around `encoder_training.utils` -- a sideways reach into a sibling
# package for two functions that belong to neither.

from gridcode.smoothing import smooth_g, smooth_gbook  # noqa: F401
