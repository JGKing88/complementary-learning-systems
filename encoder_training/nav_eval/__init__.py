"""Navigation evaluation for the encoder, on the raw scaffold.

Absorbed from `cls/eval/nav_eval.py` + `cls/nav.py` in phase 7 so `cls/` could
be deleted. This is the encoder side's own nav metric: it walks a Hopfield
recall gradient over `encoded_Phi` directly, with no policy, no `GridEnv` and no
`VectorHash` -- which is exactly why it must not reach into `hopfield_nav`.

The `Hopfield` it uses is now the shared one in the top-level `hopfield`
package, not the second copy that lived at `cls/hopfield.py`. The two were
verified equivalent before the copy was deleted; see `hopfield/__init__.py`.
"""
from .evaluate import (
    encode_full_grid,
    run_navigation_eval,
    sample_train_eval_envs,
    sample_val_eval_envs,
)

__all__ = [
    "encode_full_grid",
    "run_navigation_eval",
    "sample_train_eval_envs",
    "sample_val_eval_envs",
]
