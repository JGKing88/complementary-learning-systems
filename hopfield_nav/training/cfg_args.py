"""Reconciling a typed command line with a config inherited from a checkpoint.

`--load_checkpoint` makes the parent's config the base, so "the caller did not
mention this flag" has to mean "inherit it". A parsed ``Namespace`` cannot say
that -- it holds a value either way, and the value it holds is the *dataclass
default*, which is precisely the wrong answer. The information survives only in
``argv``.

Two entry points need this and they need it differently. `train_navigate` has a
hand-written flag-to-field table (`CFG_FIELDS`) because it applies flags onto an
already-built config. `train.py` builds its config in one nested constructor
call, and that call already *is* the mapping -- so rather than transcribe ninety
entries into a table that would silently rot as flags are added, `overlay_typed`
recovers the mapping by running the constructor twice: once for real, once with
every untyped flag replaced by a sentinel. Wherever a sentinel comes out the far
end, that field was never typed and the checkpoint's value stands.
"""
from __future__ import annotations

import argparse
import dataclasses


def explicit_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    """The dests of the flags the caller actually typed.

    ``--flag=value`` is split on '='. ``BooleanOptionalAction``'s ``--no-flag``
    needs no special case: argparse lists it in the same action's
    ``option_strings``. The parser must be built with ``allow_abbrev=False``, or
    a shortened flag reaches the Namespace while going unmatched here -- and
    then silently loses to the inherited value.
    """
    typed = {tok.split("=", 1)[0] for tok in argv if tok.startswith("-")}
    return {action.dest for action in parser._actions
            if any(opt in typed for opt in action.option_strings)}


def set_path(cfg, path: str, value) -> None:
    """Write a dotted path like ``env.size`` on a nested config."""
    obj = cfg
    *parents, leaf = path.split(".")
    for part in parents:
        obj = getattr(obj, part)
    setattr(obj, leaf, value)


class _Untyped:
    """Marks a config field whose flag the caller never mentioned."""

    __slots__ = ("dest",)

    def __init__(self, dest: str) -> None:
        self.dest = dest

    def __repr__(self) -> str:            # only ever seen while debugging
        return f"<untyped {self.dest}>"


def overlay_typed(base, args: argparse.Namespace, typed: set[str], build) -> None:
    """Write only the caller's typed flags from ``args`` onto ``base``.

    ``build(namespace) -> config`` is the entry point's own constructor, called
    twice more with sentinels standing in for arguments. A leaf is overwritten
    only when both probes agree it came from a flag the caller typed:

    ``reached``  every argument is a sentinel, so any leaf that is *not* one is
                 a field the constructor never feeds from argv at all -- most of
                 ``PPOConfig``, for instance. Those must be inherited whole, or
                 resuming silently resets the parent's ``ppo_epochs`` to the
                 dataclass default.
    ``chosen``   only the untyped arguments are sentinels, so a non-sentinel
                 leaf here came from something the caller actually typed.

    This also handles one flag feeding two fields (``movement_mode`` writes both
    ``env`` and ``agent``) without anyone having to remember it.

    ``base`` is mutated in place. ``build`` must pass its arguments through
    without arithmetic or validation, which is what lets a sentinel survive the
    trip; the configs have no ``__post_init__``, so it does.
    """
    def probe(keep: set[str]):
        return build(argparse.Namespace(**{
            k: (v if k in keep else _Untyped(k)) for k, v in vars(args).items()}))

    _walk(base, probe(set()), probe(typed), build(args))


def _walk(base, reached, chosen, typed_cfg) -> None:
    for f in dataclasses.fields(reached):
        r = getattr(reached, f.name)
        if dataclasses.is_dataclass(r) and not isinstance(r, type):
            _walk(getattr(base, f.name), r, getattr(chosen, f.name),
                  getattr(typed_cfg, f.name))
        elif isinstance(r, _Untyped) and not isinstance(
                getattr(chosen, f.name), _Untyped):
            setattr(base, f.name, getattr(typed_cfg, f.name))


__all__ = ["explicit_dests", "overlay_typed", "set_path"]
