"""Grid smoothing utilities for encoder training."""
from __future__ import annotations

import numpy as np


def onehot2d_to_gaussian(
    x: np.ndarray,
    sigma: float,
    sigmay: float | None = None,
    wrap: bool = True,
    normalize: str = "none",
) -> np.ndarray:
    """Turn a one-hot 2D map (H, W) into a wrapped 2D Gaussian bump."""
    x = x.astype(float, copy=False)
    H, W = x.shape
    if H == 0 or W == 0 or sigma <= 0:
        return np.zeros_like(x, dtype=float)

    pts = np.argwhere(x != 0)
    if pts.size == 0:
        return np.zeros_like(x, dtype=float)

    sx, sy = float(sigma), float(sigmay if sigmay is not None else sigma)
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    out = np.zeros((H, W), dtype=float)

    for cy, cx in pts:
        dy = np.abs(rows - cy)
        dx = np.abs(cols - cx)
        if wrap:
            dy = np.minimum(dy, H - dy)
            dx = np.minimum(dx, W - dx)
        out += x[cy, cx] * np.exp(-0.5 * ((dy / sy) ** 2 + (dx / sx) ** 2))

    if normalize == "max":
        m = out.max()
        if m > 0:
            out /= m
    elif normalize == "sum":
        s = out.sum()
        if s > 0:
            out /= s
    return out


def smooth_g(g: np.ndarray, lambdas: list[int], fwhm_ratio: float) -> np.ndarray:
    """Smooth a one-hot grid vector by converting each module to a Gaussian bump."""
    if fwhm_ratio <= 0:
        return g.copy()

    gout = np.zeros_like(g, dtype=np.float32)
    i = 0
    for l in lambdas:
        fwhm = l * fwhm_ratio
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        two_d = g[i:i + l ** 2].reshape(l, l)
        gout[i:i + l ** 2] = onehot2d_to_gaussian(two_d, sigma).flatten()
        i += l ** 2
    return gout


def smooth_gbook(gbook: np.ndarray, lambdas: list[int], fwhm_ratio: float) -> np.ndarray:
    """Smooth all gbook positions.  Vectorized over positions per module.

    gbook: (Ng, Npos, Npos)
    Returns smoothed array of same shape.
    """
    if fwhm_ratio <= 0:
        return gbook.copy()

    Ng, Npos1, Npos2 = gbook.shape
    out = np.zeros_like(gbook, dtype=np.float32)
    offset = 0

    for l in lambdas:
        n = l * l
        fwhm = l * fwhm_ratio
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        # (n, Npos1, Npos2) -> (N, l, l) where N = Npos1*Npos2
        block = gbook[offset:offset + n].reshape(l, l, -1).transpose(2, 0, 1)

        flat = block.reshape(block.shape[0], -1)
        active = np.argmax(flat, axis=1)
        cy, cx = active // l, active % l

        rows_1d = np.arange(l)
        cols_1d = np.arange(l)
        dy = np.abs(rows_1d[None, :, None] - cy[:, None, None])
        dy = np.minimum(dy, l - dy)
        dx = np.abs(cols_1d[None, None, :] - cx[:, None, None])
        dx = np.minimum(dx, l - dx)

        bumps = np.exp(-0.5 * ((dy / sigma) ** 2 + (dx / sigma) ** 2))
        out[offset:offset + n] = bumps.transpose(1, 2, 0).reshape(n, Npos1, Npos2)
        offset += n

    return out
