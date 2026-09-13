"""The grid code at an arbitrary lattice orientation and scale (plan sec 4B.2).

`gen_gbook_2d` + `smooth_gbook` fix the lattice: module m's bump sits at the
integer phases `(X mod lambda_m, Y mod lambda_m)` of the global position.
`gbook_at` is the same code with the lattice rotated by `theta` and scaled by
`scale` before the phases are taken,

    (X', Y') = R_theta (X, Y) / scale
    phi_m    = (X' mod lambda_m, Y' mod lambda_m)        continuous
    block_m  = exp(-0.5 * d_torus(phi_m)^2 / sigma_m^2)   peak 1, unnormalised

and at (theta = 0, scale = 1) reproduces `smooth_gbook(gen_gbook_2d(...))`
gathered at the same positions to float precision -- gate B2-C1, which is
what pins the smoothing convention here so the two cannot drift. That
convention, read off `smoothing.smooth_gbook` and `codebook.gen_gbook_2d`:
per module, `sigma = fwhm / (2 sqrt(2 ln 2))` with `fwhm = fwhm_ratio *
lambda`; toroidal distance `min(d, lambda - d)` on each axis; the lambda x
lambda block flattened row-major with the X phase as the ROW (`flat // lambda`)
and the Y phase as the column (`flat % lambda`). Continuous phases replace
the integer ones and nothing else changes.

The decoding helpers below are what a reader of the code -- the scripted
estimator of plan sec 4B.5, or a test -- needs and nothing more: the bump's
centroid on the torus, the wrapped phase difference, and the Chinese-remainder
step that turns per-module phase differences back into a displacement.
Nothing here imports `hopfield_nav`; this is the bottom of the stack.
"""
from __future__ import annotations

import numpy as np

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def rotation(theta: float) -> np.ndarray:
    """`R_theta = [[cos, -sin], [sin, cos]]` acting on column vectors (X, Y)."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def module_phases(positions: np.ndarray, lambdas, theta: float = 0.0,
                  scale: float = 1.0, shift=(0.0, 0.0)) -> np.ndarray:
    """Continuous phases `(N, M, 2)` in `[0, lambda_m)` of the rotated, scaled, shifted positions.

    `shift` is a translation of the lattice, added after rotation and scaling
    (`R_theta P / s + shift`). Every module is shifted by the same amount, so
    a shift uniform over the combined period leaves the ABSOLUTE phases
    uniform whatever `theta` is, while phase DIFFERENCES between two
    positions -- everything a displacement decode uses -- are untouched.
    That is what makes it the ingredient that closes the weights' route to
    `theta` through memorised env positions (plan sec 4B.2).
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    xy = (pos @ rotation(theta).T) / float(scale)               # (N, 2) = R_theta (X, Y) / s
    xy = xy + np.asarray(shift, dtype=np.float64).reshape(1, 2)
    lam = np.asarray(lambdas, dtype=np.float64)                  # (M,)
    return np.mod(xy[:, None, :], lam[None, :, None])            # (N, M, 2)


def _module_bump(phase: np.ndarray, lam: int, sigma: float) -> np.ndarray:
    """`(N, lam * lam)` Gaussian bumps around continuous phases `(N, 2)`, row = X phase."""
    grid = np.arange(lam, dtype=np.float64)
    dx = np.abs(grid[None, :] - phase[:, 0:1])                    # (N, lam) along X (rows)
    dx = np.minimum(dx, lam - dx)
    dy = np.abs(grid[None, :] - phase[:, 1:2])                    # (N, lam) along Y (cols)
    dy = np.minimum(dy, lam - dy)
    e = -0.5 / (sigma * sigma)
    bump = np.exp(e * dx * dx)[:, :, None] * np.exp(e * dy * dy)[:, None, :]   # (N, lam, lam)
    return bump.reshape(bump.shape[0], -1)


def gbook_at(positions: np.ndarray, lambdas, fwhm_ratio: float, theta: float = 0.0,
             scale: float = 1.0, shift=(0.0, 0.0)) -> np.ndarray:
    """The smoothed grid code `(N, Ng)` float32 at global `positions (N, 2)`.

    `fwhm_ratio <= 0` gives the one-hot code (nearest integer phase), matching
    `smooth_gbook`'s early return.
    """
    phases = module_phases(positions, lambdas, theta, scale, shift)
    parts = []
    for m, lam in enumerate(lambdas):
        lam = int(lam)
        if fwhm_ratio <= 0:
            idx = np.mod(np.rint(phases[:, m]).astype(np.int64), lam)
            block = np.zeros((phases.shape[0], lam * lam), dtype=np.float64)
            block[np.arange(phases.shape[0]), idx[:, 0] * lam + idx[:, 1]] = 1.0
        else:
            sigma = fwhm_ratio * lam * _FWHM_TO_SIGMA
            block = _module_bump(phases[:, m], lam, sigma)
        parts.append(block)
    return np.concatenate(parts, axis=1).astype(np.float32)


def module_slices(lambdas) -> list[slice]:
    """Where each module's `lambda^2` block sits in the Ng-wide code."""
    out, off = [], 0
    for lam in lambdas:
        out.append(slice(off, off + int(lam) ** 2))
        off += int(lam) ** 2
    return out


# ---------------------------------------------------------------------------
# Reading the code back
# ---------------------------------------------------------------------------

def torus_centroid(block: np.ndarray, lam: int) -> np.ndarray:
    """Circular-mean centroid `(N, 2)` in `[0, lam)` of `(N, lam * lam)` bump weights.

    Per axis: the weights are marginalised onto that axis, mapped to angles
    `2 pi k / lam`, and the phase of the weighted resultant is read back in
    cells. Exact for a symmetric bump that does not overlap itself around the
    torus, and a fraction of a cell off otherwise; the wide-bump case is what
    the fwhm gate in the tests bounds.
    """
    w = np.asarray(block, dtype=np.float64).reshape(-1, lam, lam)
    ang = 2.0 * np.pi * np.arange(lam) / lam
    z = np.exp(1j * ang)
    wx = w.sum(axis=2)                                            # (N, lam) over X (rows)
    wy = w.sum(axis=1)                                            # (N, lam) over Y (cols)
    cx = np.angle(wx @ z) * lam / (2.0 * np.pi)
    cy = np.angle(wy @ z) * lam / (2.0 * np.pi)
    return np.mod(np.stack([cx, cy], axis=1), lam)


def code_phases(code: np.ndarray, lambdas) -> np.ndarray:
    """Centroids `(N, M, 2)` of every module of an `(N, Ng)` code."""
    code = np.asarray(code).reshape(-1, sum(int(l) ** 2 for l in lambdas))
    return np.stack([torus_centroid(code[:, sl], int(lam))
                     for sl, lam in zip(module_slices(lambdas), lambdas)], axis=1)


def wrapped_phase_diff(a: np.ndarray, b: np.ndarray, lam) -> np.ndarray:
    """`a - b` wrapped into `[-lam/2, lam/2)`; broadcasts, `lam` may be `(M,)` or `(M, 1)`."""
    lam = np.asarray(lam, dtype=np.float64)
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.mod(d + lam / 2.0, lam) - lam / 2.0


def crt_displacement(dphase: np.ndarray, lambdas, max_abs: float, k_range: int = 3) -> np.ndarray:
    """The displacement `(N, 2)` consistent with per-module phase differences `(N, M, 2)`.

    Axes are independent (square tori). Per axis, every candidate
    `dphase_0 + k * lambda_0` for `k in [-k_range, k_range]` is scored by its
    summed squared wrapped residual against the other modules and the best
    inside `[-max_abs, max_abs]` wins; the residual of the winner is the
    decode's own confidence. With three coprime moduli the combined period is
    `prod(lambdas)` (1716 for 11, 12, 13) so any window that size is unique,
    and `max_abs = S sqrt(2)` for one env is far inside it.
    """
    dp = np.asarray(dphase, dtype=np.float64)
    N, M, _ = dp.shape
    lam = np.asarray(lambdas, dtype=np.float64)
    ks = np.arange(-k_range, k_range + 1, dtype=np.float64)
    out = np.zeros((N, 2), dtype=np.float64)
    for ax in range(2):
        cand = dp[:, 0, ax][:, None] + ks[None, :] * lam[0]      # (N, K)
        res = np.zeros_like(cand)
        for m in range(1, M):
            res += wrapped_phase_diff(cand, dp[:, m, ax][:, None], lam[m]) ** 2
        res = np.where(np.abs(cand) <= max_abs, res, np.inf)
        best = np.argmin(res, axis=1)
        out[:, ax] = cand[np.arange(N), best]
    return out


__all__ = [
    "rotation", "module_phases", "gbook_at", "module_slices",
    "torus_centroid", "code_phases", "wrapped_phase_diff", "crt_displacement",
]
