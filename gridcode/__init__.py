"""Grid codes and the associative layers over them.

The live remnant of `cls/`, extracted in phase 7 so that package could be
deleted: `codebook.gen_gbook_2d` builds the one-hot grid book, `smoothing`
turns its columns into Gaussian bumps, and `assoc` holds the place-book and
pseudoinverse trainers `VectorHash` uses to build a scaffold.

This is the bottom of the stack. It imports nothing from `hopfield_nav`,
`encoder_training` or `analysis`, which is what lets both of the first two
depend on it -- and is why `smooth_g`/`smooth_gbook` live here rather than in
`encoder_training.utils`, where `hopfield_nav.utils` had been reaching sideways
to find them.

Everything was moved verbatim. `smoothing.py` is `encoder_training/utils.py`
unchanged; the other two are the transitive closure of the six functions the
live code imported, checked numerically against the `cls` originals before the
originals were deleted.
"""
