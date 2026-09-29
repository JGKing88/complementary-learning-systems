"""The smoothed grid code at any scaffold position, without the smoothed book.

``smooth_gbook(gen_gbook_2d(...))`` is ``(Ng, Npos, Npos)`` -- 5 GB at
``lambdas 11 12 13``, ``Npos 1716``. But each module's block depends on the
position only through its phase ``(x mod λ, y mod λ)``, and smoothing is a
wrapped Gaussian on that module's own λ×λ torus. So module ``m``'s smoothed
code is a ``(λ, λ, λ²)`` table indexed by phase, and the full code at
``(x, y)`` is the concatenation of three table rows. Bit-identical to the book
(``tests/test_grid_memory.py``).
"""
from __future__ import annotations

import numpy as np

from gridcode.codebook import gen_gbook_2d
from gridcode.smoothing import smooth_gbook


class SmoothedGridCode:
    """``at(gx, gy) -> (B, Ng)``: the smoothed grid code at global positions.

    Positions are clipped to ``[0, Npos - 1]``, the convention of
    ``rollout.rnn.grid_state_vec`` and ``VectorHash.get_encoded_state``.
    """

    def __init__(self, lambdas, fwhm_ratio: float, Npos: int) -> None:
        self.lambdas = [int(l) for l in lambdas]
        self.fwhm_ratio = float(fwhm_ratio)
        self.Npos = int(Npos)
        self.Ng = sum(l * l for l in self.lambdas)
        # One period of each module's smoothed book: positions 0..λ-1 on a
        # book of just that module. (λ, λ, λ²) indexed [x mod λ, y mod λ].
        self._tables = []
        for l in self.lambdas:
            book = smooth_gbook(gen_gbook_2d([l], l * l, l), [l], self.fwhm_ratio)
            self._tables.append(np.ascontiguousarray(book.transpose(1, 2, 0)))

    def at(self, gx, gy) -> np.ndarray:
        gx = np.clip(np.asarray(gx, dtype=np.int64), 0, self.Npos - 1)
        gy = np.clip(np.asarray(gy, dtype=np.int64), 0, self.Npos - 1)
        return np.concatenate([t[gx % l, gy % l] for l, t in
                               zip(self.lambdas, self._tables)], axis=-1).astype(np.float32)

    def at_local(self, positions: np.ndarray, env_offset) -> np.ndarray:
        """Local ``(B, 2)`` env cells plus the env's scaffold offset."""
        positions = np.asarray(positions)
        return self.at(positions[..., 0] + int(env_offset[0]),
                       positions[..., 1] + int(env_offset[1]))
