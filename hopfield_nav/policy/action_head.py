"""Bounded action mean and state-dependent spread, shared by both agents.

Both live here rather than in each agent so the two policy classes cannot drift
apart -- `agent.py` and `agent_rnn.py` build the same head and must apply it the
same way.

**Why the squash is on the MEAN, not the sample.** Squashing the sample is a
change of variables: PPO's log-prob would need the Jacobian of the radial map,
`log|J| = log(r'/r) + log(dr'/dr)`, in the same way SAC's tanh-squash carries
`log(1 - tanh^2)`. Get it wrong and the importance ratio is silently biased.
Squashing the mean and adding noise afterwards leaves the distribution Gaussian,
so `log_prob`, the ratio, and the entropy term are all untouched.

**Why it is radial, not elementwise.** An elementwise tanh on a 2-D action
squashes x and y independently, which *changes the direction*: a vector along an
axis is squashed on one component while a diagonal one is squashed on both, so
the map biases toward the diagonals. The action here is a displacement and
direction is the part that must survive, so the squash acts on `||a||` and
leaves the unit vector alone.

**What it fixes.** Under a hard env clamp the gradient on `||mu||` past the cap
is one-directional -- samples that still fall under the cap keep pushing it up,
nothing pushes back -- so the commanded magnitude drifts without bound. Measured
at 8.18 against a cap of 2.0 (EXPERIMENTS_NAV_P2 section 8.2), which collapses
the effective angular noise `sigma/||mu||` to about 3.5 degrees. Nothing in the
objective can see that, because Gaussian entropy depends on sigma alone.

**What it does NOT fix.** `sigma/||mu||` still varies by the width of the
magnitude range: over [0.5, 2] that is a 4x channel, which is *larger* than the
2.2x state-dependent modulation the policy currently shows. So a policy could
keep modulating `||mu||` and leave a state-dependent sigma head flat. That is
detectable rather than preventable -- log both and look. Full decoupling needs a
polar parameterization, which is a larger change.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def squash_mean(mean: torch.Tensor, lo: float, hi: float,
                eps: float = 1e-8) -> torch.Tensor:
    """Map ``||mean||`` smoothly into ``[lo, hi]``, preserving direction.

    ``r = lo + (hi - lo) * tanh(r_raw / (hi - lo))`` -- monotone, equal to
    ``lo + r_raw`` for small inputs, asymptotic to ``hi``.

    **Two honest limits, both measured rather than assumed.**

    The derivative is *not* "never zero". In float32 it is 6.6e-01 at
    ``r_raw = 1``, 5.1e-03 at 5, 6.4e-06 at 10, and **exactly 0.0 by 15**, where
    tanh saturates. So this moves the dead region from ``hi`` out to ~15, it
    does not abolish it, and a sufficiently determined drift still gets there.

    That is survivable, and it is the real argument for this change: past the
    saturation point the *raw* parameter keeps drifting but the *effective*
    magnitude stays pinned at ``hi``, so the angular noise ``sigma/||mu||``
    stays at ``sigma/hi`` instead of collapsing without bound. Under the hard
    env clamp the raw drift was equally unbounded AND behaviourally live,
    because the clamp acted on the sample rather than the mean. Here the drift
    becomes behaviourally inert. The pathology is neutralized, not prevented.

    A zero mean maps to **zero**, not to ``lo``, because the direction of a
    zero vector is undefined and inventing one would be worse. It is harmless:
    the sample is ``mean + noise``, so the realized action is nonzero and the
    env's own ``min_action_norm`` floors it.
    """
    span = hi - lo
    r_raw = mean.norm(dim=-1, keepdim=True)
    r = lo + span * torch.tanh(r_raw / span)
    return mean * (r / r_raw.clamp_min(eps))


def build_log_std(cfg, hidden_size: int):
    """The std parameterization: a global parameter, or a per-state head.

    Returned as ``(param_or_None, head_or_None)`` and attached by the caller
    under the EXISTING attribute names, so an agent with the new flags off has
    byte-identical ``state_dict`` keys to one built before this module existed.
    Renaming them would break loading of every checkpoint in the project.

    The head is initialized to reproduce the global-sigma policy exactly: zero
    weights, ``init_log_std`` in the bias. So step one is the old behaviour and
    any state-dependence has to be learned rather than started from noise.
    """
    if getattr(cfg, "state_dependent_std", False):
        head = nn.Linear(hidden_size, 2)
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, cfg.init_log_std)
        return None, head
    param = nn.Parameter(torch.full((2,), cfg.init_log_std))
    if cfg.freeze_log_std:
        param.requires_grad = False
    return param, None


def movement_std(cfg, features: torch.Tensor, mean: torch.Tensor,
                 log_std, log_std_head) -> torch.Tensor:
    """Std for a batch, from the global parameter or the per-state head.

    Takes ``features`` because a state-dependent sigma is a function of the
    state, not of the mean -- conditioning it on the same features as the mean
    is what lets it read the Hopfield channels and be uncertain exactly where
    the readout is untrustworthy.

    The clamp matters more than it looks: a state-dependent sigma can collapse
    SELECTIVELY, to zero exactly where exploitation pays -- near the goal -- and
    that is much harder to notice than a global collapse.
    """
    if log_std_head is not None:
        lo = float(getattr(cfg, "log_std_min", -2.5))
        hi = float(getattr(cfg, "log_std_max", 0.5))
        return log_std_head(features).clamp(lo, hi).exp()
    return log_std.exp().expand_as(mean)


def action_bounds_from(env_cfg) -> tuple[float, float] | None:
    """``(min, max)`` action norm, or None if either is unset."""
    lo = getattr(env_cfg, "min_action_norm", None)
    hi = getattr(env_cfg, "max_action_norm", None)
    return None if lo is None or hi is None else (float(lo), float(hi))
