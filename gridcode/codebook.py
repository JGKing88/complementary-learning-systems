"""The grid codebook: one one-hot grid vector per 2-D position.

`gen_gbook_2d` is the entry point to everything downstream -- the scaffold, the
encoder's training data, and the sweeps all start from a `gbook`. Extracted
verbatim from `cls/vectorhash/assoc_utils_np_2D.py` in phase 7, which was 228
lines of which this was the only live one.
"""
from __future__ import annotations

import numpy as np


def gen_gbook_2d(lambdas, Ng, Npos):
    """
    Return grid codebook (grid activity vector for each position)

    Inputs:
        lambdas - list[int], grid periods
        Ng - int, number of grid cells
            should equal to sum of period squared
        Npos - int, number of spatial positions in each axis

    Outputs:
        gbook - np.array, size (Ng, Npos, Npos)
            gbook[:, a, b] = grid vector at position (a, b)
    """
    lambdas = np.asarray(lambdas, dtype=int)
    gbook = np.zeros((Ng, Npos, Npos))

    # Precompute module offsets
    module_offsets = np.zeros(len(lambdas), dtype=int)
    module_offsets[1:] = np.cumsum(lambdas[:-1] ** 2)

    # Vectorized: for each module, compute the active row index for all (x, y)
    xs = np.arange(Npos)
    ys = np.arange(Npos)

    for m, period in enumerate(lambdas):
        offset = int(module_offsets[m])
        # phi1[x] = x % period, phi2[y] = y % period
        phi1 = xs % period          # (Npos,)
        phi2 = ys % period          # (Npos,)
        # Flat index within module: phi1 * period + phi2 for all (x, y) pairs
        flat_idx = phi1[:, None] * period + phi2[None, :]  # (Npos, Npos)
        # Row in gbook = offset + flat_idx
        rows = offset + flat_idx    # (Npos, Npos)
        # Set using advanced indexing
        col_x, col_y = np.meshgrid(xs, ys, indexing='ij')
        gbook[rows, col_x, col_y] = 1.0

    return gbook


__all__ = [
    "gen_gbook_2d",
]
