"""Analysis layer: figure and experiment pipelines built on the library.

Nothing in `hopfield_nav` or `encoder_training` may import from here. That is
the whole point of analysis being a separate top-level package rather than a
subpackage: an analysis import inside the library would make a figure script's
matplotlib backend a dependency of training.
"""
