"""Polar action parameterization: heading x speed as separate distributions.

The Cartesian head in ``action_head.py`` bounds ``||mu||`` but does not
DECOUPLE direction from speed: with an isotropic Gaussian the effective
angular noise is ``sigma/||mu||``, so a policy can buy directional exploration
by slowing down. EXPERIMENTS_NAV_P2 section 9.3 measured exactly that -- the
state-dependent sigma head modulated 1.086x on distractor count and was FLAT
against distance to goal, while ``||mu||`` modulated 1.234x *without* the head
and 1.220x with it. The head displaced nothing; the magnitude channel kept
doing the work, and the residual channel is 4x wide over [0.5, 2] against the
2.2x modulation the policy actually shows.

Here the two are separate factors, so neither can pay for the other::

    theta ~ VonMises(theta_bar, kappa)      allocentric world-frame heading
    r     ~ lo + (hi - lo) * Beta(mu*nu, (1-mu)*nu)
    a     = r * (cos theta, sin theta)

**Allocentric, not egocentric.** ``q = W_x (recall(x) - x)`` is itself a
world-frame displacement, so the heading the policy is trying to match is
world-frame. An egocentric turn would make the policy re-derive
``target - previous heading`` from information it already has, and the
reference frame would drift whenever the env clamp made the realized heading
differ from the commanded one. It also lets ``movement_mean`` be reused
unchanged as the direction head.

**(mu, nu), not (alpha, beta).** Identical family -- this is only how alpha
and beta are computed. The reason is freezability: in (alpha, beta) there is
no spread parameter to freeze, because holding alpha fixed and learning beta
moves the mean AND the spread together. ``freeze_log_std`` has been
load-bearing through this project -- and silently did nothing on
train_navigate for the whole v35 lineage -- so a freeze needs to be a
``requires_grad`` flag on one named scalar, not a constraint holding a
combination of two heads fixed.

**Why nu >= 2.** A U-shape (mass spiking at BOTH ends) needs alpha < 1 and
beta < 1, i.e. ``nu < min(1/mu, 1/(1-mu)) <= 2``. So one CONSTANT floor
forbids U-shapes for every mu, with no restriction on mu at all -- which is
what keeps nu genuinely freezable and mu exactly the mean. The J-shapes that
survive (density piling at one end) are legitimate "go as fast as allowed"
policies; ``speed_mu_eps`` keeps them off the boundary where the Beta gradient
blows up.

**Jacobians.** ``log_prob`` is taken on ``(theta, u)`` and OMITS both the
polar->Cartesian term ``-log r`` and the affine rescale ``-log span``. Each
depends only on the sampled action, never on a parameter, so both cancel
exactly in the PPO importance ratio -- provided they are omitted consistently
at sampling and at re-evaluation. ``test_polar_head.py::TestPPODynamics`` pins
that against a numerically differentiated Cartesian reference.

**Entropy** is ``H(VonMises) + H(Beta)``: the polar entropy, not the Cartesian
one, which differs by ``E[log r]``. That difference is precisely the term that
would let an entropy bonus pay for directional randomness with speed, so
dropping it is the point rather than an approximation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, VonMises

_U_EPS = 1e-6          # keeps Beta.log_prob off the open interval's boundary


def vm_entropy(kappa: torch.Tensor) -> torch.Tensor:
    """``H = -kappa*I1/I0 + log(2*pi*I0(kappa))``.

    ``torch.distributions.VonMises`` raises NotImplementedError for entropy, so
    it is supplied here. Written with the exponentially SCALED Bessels --
    ``I0(k) = i0e(k)*e^k`` -- because ``I0(148)`` overflows float32 while
    ``i0e(148)`` is 0.033.
    """
    i0e = torch.special.i0e(kappa)
    i1e = torch.special.i1e(kappa)
    return -kappa * (i1e / i0e) + math.log(2.0 * math.pi) + i0e.log() + kappa


def circular_sd(kappa: torch.Tensor) -> torch.Tensor:
    """Circular standard deviation of a VonMises, in radians.

    ``sqrt(-2 ln R_bar)`` with ``R_bar = I1/I0``. This is the column that makes
    polar runs comparable to Cartesian ones: section 9.3's arm sat at 10.56
    degrees of ``sigma/||mu||``, which corresponds to kappa = 29.4 and reports
    back here as 10.66 degrees -- the two conventions agree to ~1%, so the two
    parameterizations plot on one axis.
    """
    r_bar = torch.special.i1e(kappa) / torch.special.i0e(kappa)
    return (-2.0 * r_bar.clamp(1e-7, 1.0 - 1e-7).log()).sqrt()


class PolarMove:
    """Heading x speed, packaged to stand in for ``Normal(mean, std)``.

    ``log_prob`` and ``entropy`` both return ``(..., 2)`` -- ``[heading,
    speed]`` -- so the ``.sum(-1)`` that every existing call site already
    applies to the Cartesian head produces the joint quantity here with no
    change. That is deliberate: it is why the rollout collector and the
    evaluators need no polar-specific branch.

    With ``speed_const`` set the speed factor is DELETED, not driven to a
    degenerate limit: its log-prob slot is exactly zero and its entropy slot is
    exactly zero. A zero-variance Normal or an infinite-concentration Beta
    would give log-prob -> +inf and entropy -> -inf; this gives neither, and
    the action simply lives on a circle of radius ``speed_const``.
    """

    def __init__(self, theta, kappa, speed_mu=None, speed_nu=None, *,
                 lo: float, hi: float, speed_const: float | None = None,
                 dir_norm=None):
        self.theta = theta
        # ALREADY the effective concentration: PolarHead shrinks it by
        # ||v||/sqrt(||v||^2 + dir_soft^2). Logging kappa rather than the head's
        # raw output is the honest choice -- kappa_eff is what governs behaviour.
        self.kappa = kappa
        self.dir_norm = dir_norm
        self.lo, self.hi = float(lo), float(hi)
        self.span = self.hi - self.lo
        self.speed_const = speed_const
        self.speed_mu = speed_mu
        self.speed_nu = speed_nu
        self._vm = VonMises(theta, kappa)
        self._beta = (None if speed_const is not None
                      else Beta(speed_mu * speed_nu, (1.0 - speed_mu) * speed_nu))

    # -- speed summaries ----------------------------------------------------

    @property
    def speed_mean(self) -> torch.Tensor:
        """Exactly the mean speed. ``mu`` IS the Beta's mean -- the readability
        half of why (mu, nu) was chosen over (alpha, beta)."""
        if self.speed_const is not None:
            return torch.full_like(self.theta, self.speed_const)
        return self.lo + self.span * self.speed_mu

    @property
    def speed_std(self) -> torch.Tensor:
        if self.speed_const is not None:
            return torch.zeros_like(self.theta)
        return self.span * self._beta.stddev

    # -- distribution interface --------------------------------------------

    def sample(self) -> torch.Tensor:
        theta = self._vm.sample()
        if self.speed_const is None:
            r = self.lo + self.span * self._beta.sample()
        else:
            r = torch.full_like(theta, self.speed_const)
        return torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """``(..., 2)``: ``[log p(theta), log p(u)]``.

        Recovers ``(theta, r)`` from the Cartesian action by ``atan2`` and
        ``norm``. Both are exact for actions this head produced -- ``||a||`` is
        the sampled speed by construction, so the round trip does not lose the
        magnitude the way inverting a squash would. Off-policy actions (epsilon
        -greedy, auto-nav override) can land outside the speed range; the clamp
        keeps that finite, and ``policy_action_mask`` already excludes them from
        the surrogate.
        """
        theta = torch.atan2(action[..., 1], action[..., 0])
        lp_theta = self._vm.log_prob(theta)
        if self.speed_const is None:
            u = (action.norm(dim=-1) - self.lo) / self.span
            lp_speed = self._beta.log_prob(u.clamp(_U_EPS, 1.0 - _U_EPS))
        else:
            lp_speed = torch.zeros_like(lp_theta)
        return torch.stack([lp_theta, lp_speed], dim=-1)

    def entropy(self) -> torch.Tensor:
        """``(..., 2)``: ``[H(VonMises), H(Beta)]``. See the module note on why
        this is the polar entropy rather than the Cartesian one."""
        h_theta = vm_entropy(self.kappa)
        h_speed = (torch.zeros_like(h_theta) if self.speed_const is not None
                   else self._beta.entropy())
        return torch.stack([h_theta, h_speed], dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """Cartesian mean action, for ``deterministic=True`` eval."""
        r = self.speed_mean
        return torch.stack([r * self.theta.cos(), r * self.theta.sin()], dim=-1)

    @property
    def stddev(self) -> torch.Tensor:
        """``(..., 2)`` as **(radial, tangential)**, not (x, y).

        A polar distribution has no meaningful per-axis std; these are the two
        physically distinct spreads, and quoting them in the natural frame is
        less misleading than rotating them into x/y. Callers wanting the
        logging columns should use :meth:`diag`.
        """
        return torch.stack(
            [self.speed_std, self.speed_mean * circular_sd(self.kappa)], dim=-1)

    def with_temperature(self, t: float) -> "PolarMove":
        """Scale both spreads by ``t`` -- the polar analogue of scaling sigma.

        ``sigma_ang ~ kappa^-1/2`` so ``kappa -> kappa/t^2``; Beta variance goes
        as ``1/(nu+1)`` so ``(nu+1) -> (nu+1)/t^2``. Frozen speed stays frozen:
        a deterministic factor has no spread to scale.
        """
        kappa = (self.kappa / (t * t)).clamp_min(1e-4)
        if self.speed_const is not None:
            return PolarMove(self.theta, kappa, lo=self.lo, hi=self.hi,
                             speed_const=self.speed_const)
        nu = ((self.speed_nu + 1.0) / (t * t) - 1.0).clamp_min(2.0)
        return PolarMove(self.theta, kappa, self.speed_mu, nu,
                         lo=self.lo, hi=self.hi)

    @torch.no_grad()
    def diag(self) -> dict[str, float]:
        """Per-update logging columns, chosen to line up with the Cartesian ones.

        ``mu_norm`` <- mean speed (vs Cartesian ``||mu||``), ``sigma`` <- speed
        sd (vs Cartesian radial noise), ``ang_noise`` <- circular sd in radians
        (vs Cartesian ``sigma/||mu||``). Plus ``kappa`` itself, which has no
        Cartesian counterpart.
        """
        out = {
            "mu_norm": float(self.speed_mean.mean()),
            "sigma": float(self.speed_std.mean()),
            "ang_noise": float(circular_sd(self.kappa).mean()),
            "kappa": float(self.kappa.mean()),
        }
        if self.dir_norm is not None:
            # The gauge freedom, logged so a drift toward the singular corner
            # is visible while it happens rather than found in a post-mortem.
            # Healthy is O(1); sustained values near dir_soft mean the heading
            # is being held near-uniform by a shrinking direction vector.
            out["dir_norm"] = float(self.dir_norm.mean())
        return out


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _spread_param(state_dependent: bool, frozen: bool, hidden_size: int,
                  init: float):
    """One scalar knob: a per-state head, or a global (optionally frozen) param.

    Head init is zero weights + ``init`` in the bias, so step one reproduces
    the global-parameter policy exactly and any state-dependence has to be
    learned rather than started from noise -- the same discipline the sigma
    head was built with.
    """
    if state_dependent:
        head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, init)
        return head, None
    param = nn.Parameter(torch.tensor(float(init)))
    if frozen:
        param.requires_grad = False
    return None, param


class PolarHead(nn.Module):
    """Concentration and speed parameters for :class:`PolarMove`.

    The heading MEAN is deliberately not here: it is the agent's existing
    ``movement_mean`` Linear, read as a direction and ``atan2``-ed. Reusing it
    means a checkpoint forked into a polar run keeps its learned direction and
    only the spread parameters start fresh.

    ``state_dependent_std`` and ``freeze_log_std`` are reused rather than
    duplicated: under polar they govern kappa and nu, which ARE the spreads.
    The speed mean ``mu`` is always learnable -- "freeze the spread, keep the
    mean" is exactly the case (alpha, beta) could not express.
    """

    def __init__(self, cfg, hidden_size: int, lo: float, hi: float) -> None:
        super().__init__()
        self.lo, self.hi = float(lo), float(hi)
        self.span = self.hi - self.lo
        self.log_kappa_min = float(getattr(cfg, "log_kappa_min", -1.0))
        self.log_kappa_max = float(getattr(cfg, "log_kappa_max", 5.0))
        self.nu_min = float(getattr(cfg, "speed_nu_min", 2.0))
        self.nu_max = float(getattr(cfg, "speed_nu_max", 200.0))
        self.mu_eps = float(getattr(cfg, "speed_mu_eps", 0.05))
        self.dir_soft = float(getattr(cfg, "dir_soft", 0.01))

        fs = getattr(cfg, "freeze_speed", None)
        if fs is not None and not (self.lo - 1e-9 <= float(fs) <= self.hi + 1e-9):
            raise ValueError(
                f"freeze_speed={fs} is outside the action bounds "
                f"[{self.lo}, {self.hi}]; the env would clamp it every step")
        self.speed_const = None if fs is None else float(fs)

        state = bool(getattr(cfg, "state_dependent_std", False))
        frozen = bool(getattr(cfg, "freeze_log_std", False))

        self.log_kappa_head, self.log_kappa = _spread_param(
            state, frozen, hidden_size,
            float(getattr(cfg, "init_log_kappa", 1.85)))

        if self.speed_const is None:
            init_mu = float(getattr(cfg, "init_speed_mu", 0.5))
            init_nu = float(getattr(cfg, "init_speed_nu", 3.0))
            if not 0.0 < init_mu < 1.0:
                raise ValueError("init_speed_mu is the NORMALIZED mean speed "
                                 "and must lie in (0, 1)")
            if init_nu <= self.nu_min:
                raise ValueError(
                    f"init_speed_nu={init_nu} must exceed speed_nu_min="
                    f"{self.nu_min}; below the floor a U-shaped speed density "
                    "is being requested and the floor would silently override")
            self.speed_mu_head, self.speed_mu = _spread_param(
                state, False, hidden_size, math.log(init_mu / (1.0 - init_mu)))
            self.speed_nu_head, self.speed_nu = _spread_param(
                state, frozen, hidden_size, _inv_softplus(init_nu - self.nu_min))
        else:
            self.speed_mu_head = self.speed_nu_head = None
            self.speed_mu = self.speed_nu = None

    @staticmethod
    def _read(head, param, features):
        if head is not None:
            return head(features).squeeze(-1)
        return param.expand(features.shape[:-1])

    def forward(self, features: torch.Tensor,
                direction: torch.Tensor) -> PolarMove:
        # ``atan2`` alone has gain ``1/||v||`` on the heading, and ``||v||`` is
        # a pure GAUGE FREEDOM -- theta is scale-invariant, so nothing in the
        # objective pressures the direction head's magnitude and it random
        # walks. As it shrinks the heading gradient diverges: measured, one
        # sample in 48 reached an importance ratio of 2.34 after a single
        # 1e-3 Adam step at ||v|| = 0.24, entirely from the heading factor
        # (the speed factor contributed 0.005). That is the same shape as the
        # origin singularity that killed the first p9_e_sq_std run at u120.
        #
        # Softening turns a short direction vector into a LOW EFFECTIVE
        # concentration, which is both bounded and the correct limit: no
        # preferred direction means an uncertain heading, exactly what the
        # von Mises natural parameterization ``eta = kappa * u`` would say.
        # For ||v|| >> dir_soft the shrink factor is 1 to within 0.1%, so the
        # decoupling is untouched in the operating regime -- the coupling
        # exists only in the degenerate corner where it should.
        # SCALE-FREE. `direction` comes off a vanilla ReLU RNN unrolled 200
        # steps, whose activations can grow like ||W_hh||^200, so ||v|| can be
        # enormous for a few samples -- and every form of this computation
        # SQUARES it. theta is scale-invariant so the forward stays perfectly
        # finite (atan2 of a huge vector is an ordinary angle, and shrink -> 1),
        # which is exactly why all four crash reports showed a clean forward
        # with a non-finite BACKWARD, and why the non-finite parameters were
        # movement_mean and log_kappa_head -- the two that consume this -- with
        # the Beta speed path and the value head untouched.
        #
        # Dividing by the max-abs component first makes every squared quantity
        # O(1). Mathematically identical: theta is unchanged because atan2 is
        # scale-invariant, and shrink is rewritten in the equivalent ratio form
        #     ||v||^2 / (||v||^2 + s^2)  ==  sq_n / (sq_n + (s/vmax)^2)
        # which needs no unnormalized square anywhere.
        vmax = direction.abs().amax(-1, keepdim=True).clamp_min(1e-20)
        vn = direction / vmax
        sq = vn.pow(2).sum(-1)                       # in [0, 2], cannot overflow
        # (s/vmax)^2 -> 0 for a huge ||v|| (shrink -> 1) and blows up for a tiny
        # one (shrink -> 0), both of which are the intended limits.
        soft_rel = (self.dir_soft / vmax.squeeze(-1)).pow(2)
        shrink = sq / (sq + soft_rel)
        # atan2 is likewise undefined at the origin with a NaN gradient. An
        # exactly-zero direction is measure-zero but not impossible, and one
        # NaN reaching clip_grad_norm_ zeroes the entire batch's update. The
        # replacement is a CONSTANT, so autograd routes zero gradient through
        # those entries -- the honest answer where the heading is undefined --
        # and `shrink` has already driven their concentration to uniform.
        # Degeneracy is a MAGNITUDE test, not an exact-zero test. The bound on
        # the heading gradient rests on kappa_eff shrinking as ||v||^2 while the
        # atan2 gain grows as 1/||v||; the kappa floor breaks that
        # proportionality, so any band where the floor binds while atan2 is
        # still live reopens the divergence -- measured at exactly 1000 with an
        # exact-zero test and ||v|| = 1e-9.
        #
        # A tenth of dir_soft is the natural threshold: there kappa_eff is
        # already ~1e-4 of nominal, so the heading is meaningless anyway, and
        # tying it to dir_soft rather than to a bare constant keeps the two
        # from being retuned apart. atan2 is undefined at the origin with a NaN
        # gradient; the replacement is a CONSTANT, so autograd routes zero
        # gradient through those entries, which is the honest answer where the
        # heading is undefined.
        degenerate = (vmax.squeeze(-1) < 0.1 * self.dir_soft).unsqueeze(-1)
        safe = torch.where(degenerate,
                           vn.new_tensor([1.0, 0.0]).expand_as(vn), vn)
        theta = torch.atan2(safe[..., 1], safe[..., 0])
        # The TRUE ||v||, reassembled for logging only. This is the gauge; the
        # softening bounds what happens when it decays but does not prevent it.
        dir_norm = (sq.detach().sqrt() * vmax.detach().squeeze(-1))
        # THE FLOOR IS 1e-2, AND THAT IS NOT AN ARBITRARY SMALL NUMBER.
        #
        # kappa = 0 is the uniform distribution on the circle -- the correct
        # limit for a zero direction vector -- but torch's VonMises rejects a
        # zero concentration, so some floor is required. It cannot be tiny:
        # `VonMises.log_prob`'s gradient w.r.t. concentration is **NaN for
        # kappa < 1e-5**, with a perfectly finite forward. torch's
        # `_log_modified_bessel_fn` evaluates BOTH branches and selects with
        # `torch.where`; the large-kappa branch computes `3.75/x`, which
        # overflows for tiny x, and `where`'s backward then does `inf * 0`.
        #
        # A 1e-6 floor put kappa exactly in that region and is what produced
        # the P10 crashes: non-finite gradients on log_kappa_head and
        # movement_mean with every loss finite, spreading to the RNN through
        # `features`. 1e-2 keeps three orders of magnitude of margin and is
        # still uniform for any purpose here -- circular sd 186.5 degrees, a
        # 2% density modulation between the modal and antimodal directions.
        kappa = (self._read(self.log_kappa_head, self.log_kappa, features).clamp(
            self.log_kappa_min, self.log_kappa_max).exp() * shrink).clamp_min(1e-2)
        if self.speed_const is not None:
            return PolarMove(theta, kappa, lo=self.lo, hi=self.hi,
                             speed_const=self.speed_const, dir_norm=dir_norm)
        mu = torch.sigmoid(
            self._read(self.speed_mu_head, self.speed_mu, features))
        mu = mu.clamp(self.mu_eps, 1.0 - self.mu_eps)
        nu = self.nu_min + F.softplus(
            self._read(self.speed_nu_head, self.speed_nu, features))
        return PolarMove(theta, kappa, mu, nu.clamp(max=self.nu_max),
                         lo=self.lo, hi=self.hi, dir_norm=dir_norm)
