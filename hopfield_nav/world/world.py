"""A world: a set of environments, where they sit in the scaffold, and the
associative weights (if any) fitted to them.

This replaces the ``{"envs", "vectorhash", "env_indices"}`` dict that
``training/world_setup.setup_world`` used to return. Two things changed:

``offsets`` is here, not on the scaffold.
    An offset is a property of an env set, not of the field it indexes into.
    Keeping it on ``VectorHash`` was what forced a whole second 12 GB scaffold
    per world -- a second world needed its own ``env_offsets`` and the only way
    to get one was to build a second scaffold. With offsets here, ``field`` is
    shared by reference across every world and every split.

``env_indices`` is gone.
    It was ``list(range(n))`` at every construction site -- an identity map from
    a world's local env index to a global one, used only to index back into
    ``vh.env_offsets``. With offsets stored per world it has nothing to index.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import GridEnv
from .scaffold import EnvAssoc, VectorHash, fit_env_assoc, place_envs


@dataclass
class World:
    """One env set, placed in a scaffold.

    ``field`` is shared -- do not mutate it per world. ``assoc`` is ``None``
    under ``static_vectorhash``, which is every current run.
    """

    envs: list[GridEnv]
    offsets: list[tuple[int, int]]
    field: VectorHash
    assoc: EnvAssoc | None = None

    def __len__(self) -> int:
        return len(self.envs)


def build_world(
    field: VectorHash,
    envs: list[GridEnv],
    *,
    offsets: list[tuple[int, int]] | None = None,
    placement: str = "spread",
    spread_jitter: float = 0.4,
    rng=np.random,
    size: int | None = None,
) -> World:
    """Place ``envs`` in ``field`` and fit their associative weights.

    ``offsets`` given -> placement is skipped and they are used verbatim, which
    is how a generated ``EnvSpec`` list gets built. ``offsets=None`` -> sample
    them, reproducing what ``VectorHash.register_envs`` did.

    ``rng`` defaults to the ``np.random`` *module*, which is the stream the old
    method drew from; pass a ``RandomState`` to pin it.
    """
    if offsets is None:
        if size is None:
            size = envs[0].size
        offsets = place_envs(
            len(envs), size, field.Npos, rng,
            placement=placement, spread_jitter=spread_jitter,
        )
    elif len(offsets) != len(envs):
        raise ValueError(
            f"got {len(offsets)} offsets for {len(envs)} envs"
        )
    assoc = fit_env_assoc(field, envs, offsets)
    return World(envs=envs, offsets=list(offsets), field=field, assoc=assoc)


__all__ = ["World", "build_world"]
