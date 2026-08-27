"""Encoding scaffold positions without materialising the field.

``VectorHash.precompute_encoded_phi`` builds ``(Npos, Npos, out_dim)`` -- 12 GB
at ``Npos=1716, out_dim=1024``, on top of a ``(434, 1716, 1716)`` ``gbook`` --
and every probe in this package touches a few tens of thousands of those 2.94M
cells. So the grid code is built *per position* here instead.

That is exact, not an approximation. A position's grid code is a pure function
of ``(x mod lambda, y mod lambda)`` per module (``gridcode/codebook.py``), and
``smooth_gbook`` places a wrapped Gaussian bump at that phase independently in
each module (``gridcode/smoothing.py``), so the whole field factorises:

    smoothed[off_m + i*l + j, x, y]
        = exp(-0.5 * ((wrap(i - x mod l) / sigma_m)^2
                    + (wrap(j - y mod l) / sigma_m)^2))

with ``wrap(d) = min(|d|, l - |d|)`` and ``sigma_m = l * fwhm_ratio / (2 sqrt(2
ln 2))``. The index order -- ``i`` from the *x* phase, ``j`` from the *y* phase,
row ``i*l + j`` -- is the one ``smooth_gbook`` produces, and getting it backwards
would transpose every embedding without changing any aggregate enough to
notice. ``test_hopfield_probe.py`` pins this function against the real
``gen_gbook_2d`` + ``smooth_gbook`` pair, which is the only reason to trust it.

The float expressions below are written in ``smooth_gbook``'s exact form
(``(d/sigma)**2``, summed x-term first) rather than an algebraically equal one,
so the two agree bit-for-bit rather than to a tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def grid_codes(
    lambdas: list[int],
    xs: np.ndarray,
    ys: np.ndarray,
    fwhm_ratio: float,
) -> np.ndarray:
    """Smoothed grid code for each ``(x, y)``: ``(N, Ng)`` float32.

    ``fwhm_ratio <= 0`` returns the unsmoothed one-hot code, matching
    ``VectorHash.precompute_encoded_phi``'s ``gbook.copy()`` branch.
    """
    xs = np.asarray(xs, dtype=np.int64).reshape(-1)
    ys = np.asarray(ys, dtype=np.int64).reshape(-1)
    if xs.shape != ys.shape:
        raise ValueError(f"xs {xs.shape} and ys {ys.shape} must match")

    n_pos = xs.size
    n_g = int(sum(l * l for l in lambdas))
    # float32 out, float64 arithmetic: smooth_gbook computes np.exp in float64
    # and assigns into a float32 array, so this rounds at the same point.
    out = np.zeros((n_pos, n_g), dtype=np.float32)

    off = 0
    for l in lambdas:
        n = l * l
        idx = np.arange(l, dtype=np.int64)

        # Wrapped phase offsets. `d_x` indexes the module block's first axis
        # (the x phase), `d_y` its second -- see the module docstring.
        d_x = np.abs(idx[None, :] - (xs % l)[:, None])
        d_x = np.minimum(d_x, l - d_x)                        # (N, l)
        d_y = np.abs(idx[None, :] - (ys % l)[:, None])
        d_y = np.minimum(d_y, l - d_y)                        # (N, l)

        if fwhm_ratio > 0:
            sigma = (l * fwhm_ratio) / (2 * np.sqrt(2 * np.log(2)))
            block = np.exp(
                -0.5 * ((d_x[:, :, None] / sigma) ** 2
                        + (d_y[:, None, :] / sigma) ** 2)
            )                                                 # (N, l, l)
        else:
            block = ((d_x[:, :, None] == 0) & (d_y[:, None, :] == 0))
            block = block.astype(np.float64)

        out[:, off:off + n] = block.reshape(n_pos, n)
        off += n

    return out


def encode_positions(
    encoder: torch.nn.Module,
    lambdas: list[int],
    fwhm_ratio: float,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    chunk: int = 4096,
) -> np.ndarray:
    """Grid code -> encoder -> ``(N, out_dim)`` float32 unit vectors."""
    codes = grid_codes(lambdas, xs, ys, fwhm_ratio)
    parts = []
    with torch.no_grad():
        for start in range(0, codes.shape[0], chunk):
            batch = torch.from_numpy(codes[start:start + chunk]).to(device)
            parts.append(encoder(batch).cpu().numpy())
    if not parts:
        raise ValueError("encode_positions got zero positions")
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


@dataclass
class Field:
    """An encoder bound to a scaffold, with the two lookups the probes need.

    Holds no field array. ``Npos`` is carried because the two clip ranges below
    are part of the production semantics, not an implementation detail:
    ``get_encoded_state`` clips to ``[0, Npos-1]`` while
    ``gram_schmidt_projection`` clips to ``[1, Npos-2]``, and the difference is
    only invisible because envs are never placed hard against the edge.
    """

    encoder: torch.nn.Module
    lambdas: list[int]
    fwhm_ratio: float
    gain: float
    Npos: int
    device: str = "cpu"
    chunk: int = 4096

    @property
    def embed_dim(self) -> int:
        with torch.no_grad():
            probe = torch.zeros(
                1, int(sum(l * l for l in self.lambdas)), device=self.device)
            return int(self.encoder(probe).shape[-1])

    def encode(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """Encode *global* scaffold coordinates. No clipping -- callers clip."""
        return encode_positions(
            self.encoder, self.lambdas, self.fwhm_ratio, gx, gy,
            device=self.device, chunk=self.chunk,
        )

    def encoded_state(
        self, positions: np.ndarray, offset: tuple[int, int]
    ) -> np.ndarray:
        """``VectorHash.get_encoded_state``: local coords, clip ``[0, Npos-1]``."""
        positions = np.asarray(positions).reshape(-1, 2)
        gx = np.clip(positions[:, 0] + offset[0], 0, self.Npos - 1)
        gy = np.clip(positions[:, 1] + offset[1], 0, self.Npos - 1)
        return self.encode(gx, gy)

    def local_basis(
        self,
        positions: np.ndarray,
        offset: tuple[int, int],
        *,
        swap_gram_schmidt: bool = False,
    ) -> np.ndarray:
        """``VectorHash.gram_schmidt_projection``: ``(B, 2, D)``, row0 East.

        ``swap_gram_schmidt`` is control Sec 6.2: orthogonalise in the other
        order -- East kept exactly, North reduced -- while still emitting
        ``(East, North)`` rows, so ``q`` stays comparable.
        """
        from hopfield_nav.utils import gram_schmidt_2d_batch

        positions = np.asarray(positions).reshape(-1, 2)
        gx = np.clip(positions[:, 0] + offset[0], 1, self.Npos - 2)
        gy = np.clip(positions[:, 1] + offset[1], 1, self.Npos - 2)

        current = self.encode(gx, gy)
        d_fwd = self.encode(gx, gy + 1) - current           # North, +y
        d_rgt = self.encode(gx + 1, gy) - current           # East,  +x

        if not swap_gram_schmidt:
            return gram_schmidt_2d_batch(d_fwd, d_rgt)
        # Same routine, arguments swapped: it returns (reduced, kept) rows, so
        # the result is (North_reduced, East_kept) and the rows come back in
        # the opposite roles. Flip them so row0 is still East.
        swapped = gram_schmidt_2d_batch(d_rgt, d_fwd)
        return swapped[:, ::-1, :].copy()


__all__ = ["Field", "encode_positions", "grid_codes"]
