"""The `ContinualMethod` interface, the no-op default, and the registry.

`run_sequential_blocks` already had the shape every continual-learning method
needs: a block loop with per-update callbacks. What it lacked was somewhere for
a method to *change the update*. That is all this interface is -- six hooks,
each corresponding to a place the literature's methods actually intervene:

    on_block_start   a task boundary is beginning        (masks, gating)
    extra_batches    add data to this update             (ER, CLEAR, DER++)
    penalty          add a term to the loss              (EWC, SI, LwF)
    aux_loss         add a term that needs the batches   (CLEAR, DER++ distill)
    after_update     the update happened                 (buffer insert, SI)
    on_block_end     a task boundary is ending           (Fisher, anchors)

plus `state_bytes`, which is not bookkeeping: the whole point of the control
suite is a cost-matched frontier (plan section 0.1), and a method's stored
bytes is one of its five axes. A method that does not report it cannot be
placed on the plot.

Deliberately *not* a `typing.Protocol`: methods share enough default behaviour
(most implement two hooks and want the other four to be no-ops) that a base
class with working defaults is less code at every call site than a protocol
plus six stubs per method.
"""
from __future__ import annotations

from typing import Any

import torch


class ContinualMethod:
    """No-op base. Subclasses override only the hooks they need.

    The driver calls these in this order, per update:

        extra_batches(rollout, block)      -> sampled BEFORE the new rollout is
                                              stored, so "replay" means old data
        [the BC update runs, adding penalty() to each minibatch loss]
        after_update(rollout, block, agent)

    and at block boundaries, `on_block_start` / `on_block_end`.
    """

    #: Registry key; also what lands in `history["metadata"]["method"]`.
    name: str = "none"

    #: Whether the method is told where task boundaries are. Recorded so the
    #: results table can carry a "boundary-free" column honestly -- several
    #: methods (ER, CLEAR, DER++) genuinely do not need them and should get
    #: credit for it.
    needs_task_boundaries: bool = False

    #: Whether the method needs to be told *which* task it is in at eval time.
    #: The Hopfield agent needs neither; anything that does is reported as an
    #: upper bound rather than a peer.
    needs_task_id: bool = False

    # -- hooks ------------------------------------------------------------

    def on_block_start(self, block: int, agent, envs) -> None:
        """A new env is about to be trained on."""

    def extra_batches(self, rollout, block: int) -> list:
        """Extra rollout batches to fold into this update's BC loss."""
        return []

    def penalty(self, agent) -> torch.Tensor | None:
        """A scalar added to the loss on every minibatch step.

        Called inside the optimisation loop, so it must be recomputed from the
        *current* parameters -- returning a cached tensor would detach the
        penalty from the thing it is supposed to constrain.
        """
        return None

    def aux_loss(self, agent, rollout, extra: list) -> torch.Tensor | None:
        """A scalar that needs the batches themselves (distillation terms).

        Separate from `penalty` because CLEAR and DER++ regularise the model's
        *outputs on specific states*, not its parameters, so they need the data
        that `penalty` never sees.
        """
        return None

    def after_update(self, rollout, block: int, agent) -> None:
        """The update has been applied. Buffer insertion goes here."""

    def on_block_end(self, block: int, agent, envs) -> None:
        """This env's block is finished. Fisher/anchor computation goes here."""

    # -- reporting --------------------------------------------------------

    def state_bytes(self) -> int:
        """Bytes of state the method carries: replay data, importances, masks.

        One of the five axes of the cost frontier. Count what the method would
        actually have to keep, not what Python happens to allocate.
        """
        return 0

    def describe(self) -> dict[str, Any]:
        """Everything about this method that belongs in the history metadata."""
        return {
            "method": self.name,
            "needs_task_boundaries": self.needs_task_boundaries,
            "needs_task_id": self.needs_task_id,
            "state_bytes": self.state_bytes(),
        }


class NoMethod(ContinualMethod):
    """Naive sequential SGD -- the floor the whole suite is measured against."""
    name = "none"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _registry() -> dict[str, type[ContinualMethod]]:
    """Imported lazily so `base` stays importable on its own.

    (`replay` and `regularize` both import from here; a module-scope import in
    the other direction would be a cycle.)
    """
    from .regularize import OnlineEWC
    from .replay import ExperienceReplay

    return {
        NoMethod.name: NoMethod,
        ExperienceReplay.name: ExperienceReplay,
        OnlineEWC.name: OnlineEWC,
    }


CONTINUAL_METHODS: tuple[str, ...] = ("none", "er", "online_ewc")


def build_method(name: str, seed: int | None = None, **kwargs) -> ContinualMethod:
    """`--method er --method_args buffer_size=inf,replay_batches=1` -> an object.

    Unknown names fail here with the list of known ones rather than surfacing
    two hours later as a silently-naive run that looks like a method result. An
    unknown *argument* fails the same way, so a typo in a sweep script is a
    crash at launch and not a config that quietly ran at the default.

    `seed` is forwarded only to methods that have their own RNG, so callers can
    pass it unconditionally without knowing which those are.
    """
    import inspect

    reg = _registry()
    if name not in reg:
        raise ValueError(
            f"unknown continual method {name!r}; known: {sorted(reg)}")
    cls = reg[name]

    params = inspect.signature(cls.__init__).parameters
    if seed is not None and "seed" in params:
        kwargs = {**kwargs, "seed": seed}
    unknown = sorted(set(kwargs) - set(params))
    if unknown:
        accepted = sorted(p for p in params if p != "self")
        raise ValueError(
            f"method {name!r} got unknown args {unknown}; accepts {accepted}")

    typed = {}
    for k, v in kwargs.items():
        default = params[k].default
        try:
            typed[k] = coerce_to(v, default)
        except ValueError as e:
            raise ValueError(
                f"method {name!r} arg {k}={v!r}: {e} "
                f"(default is {default!r})") from None
    return cls(**typed)


def parse_method_args(spec: str | None) -> dict[str, str]:
    """`"buffer_size=inf,replay_batches=2,lam=1e3"` -> `{key: raw string}`.

    Deliberately does **not** coerce. Coercion needs to know the target type,
    and only `build_method` does: `fisher=true` names the string `"true"`
    (one of two allowed Fisher estimators) while `normalize_fisher=true` means
    the boolean. A parser guessing from the text alone turns the first into
    `True` and the method rejects it -- which is exactly what happened the first
    time this ran.
    """
    out: dict[str, str] = {}
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"method arg {item!r} is not key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def coerce_to(value: Any, default: Any) -> Any:
    """Coerce a raw string to whatever `default` is. Non-strings pass through,
    so programmatic callers can hand over real types directly."""
    if not isinstance(value, str):
        return value
    low = value.lower()
    if isinstance(default, bool):
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"expected a boolean, got {value!r}")
    if isinstance(default, str):
        # A str default means the string IS the value -- do not helpfully turn
        # "true" into a bool. This is the case that broke `fisher=true`.
        return value
    if low in ("none", "null"):
        return None
    if low in ("inf", "infinity"):
        return float("inf")
    # bool is checked above, before int, because bool subclasses int.
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    # No usable default to steer by: fall back to the permissive guess.
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


__all__ = [
    "CONTINUAL_METHODS", "ContinualMethod", "NoMethod",
    "build_method", "parse_method_args", "coerce_to",
]
