"""The Hopfield associative memory, shared by both research stacks.

A Hebbian store over unit-norm embeddings and an iterative tanh recall:

    input_memory(z):  W += scale * z zᵀ          (zero diagonal)
    recall(x):        x <- (1-α)x + α·tanh(β·W x), then L2-normalize

This was `hopfield_nav/world/memory.py`, and there was a second, near-identical
copy at `cls/hopfield.py` that the encoder-side navigation eval used. Phase 7
made it one package rather than duplicating it, because the alternative --
`encoder_training` importing `hopfield_nav` -- is the one import direction the
layering test forbids outright: the encoder side has to stay runnable on nodes
that never build a `GridEnv`.

The two copies were verified equivalent before the `cls` one was deleted: over
54 cases (3 gains x 2 alphas x 3 memory counts x 3 recall lengths) the stored
weight matrices were identical and the recalled states agreed to 0.0. The
`cls` version additionally carried an asynchronous coordinate-wise update mode
and per-step cosine-similarity tracking against a target pattern; no caller ever
passed either, and `cls/nav.py` -- the only live call -- passed every other
argument explicitly, so the differing `beta` default never applied.

Layer 0: this imports nothing else in the repo, which is what lets both
`hopfield_nav` and `encoder_training` depend on it.
"""
from .core import (
    Hopfield,
    recall_per_env_batch,
    recall_per_env_batch_trajectory,
)

__all__ = [
    "Hopfield",
    "recall_per_env_batch",
    "recall_per_env_batch_trajectory",
]
