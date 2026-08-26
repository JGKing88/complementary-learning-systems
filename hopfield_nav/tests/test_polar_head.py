"""The polar action head: heading x speed as separate distributions.

Three groups of properties, chosen for what fails SILENTLY:

* the log-prob contract, because a wrong Jacobian biases the PPO ratio without
  raising anything -- the run just learns the wrong thing slightly;
* the freeze contract, because ``--freeze_log_std`` already spent an entire
  run lineage silently doing nothing on train_navigate, and polar adds two more
  spread parameters that could go the same way;
* the numerical edges (kappa at its clamps, mu at its epsilon, off-policy
  actions outside the speed range), because a single NaN reaches
  ``clip_grad_norm_`` and zeroes the whole batch's update -- which is exactly
  how the first p9_e_sq_std run died at u120.
"""
from __future__ import annotations

import math

import pytest
import torch

from hopfield_nav.config import AgentConfig, PPOConfig
from hopfield_nav.policy.agent import NavAgent
from hopfield_nav.policy.polar_head import (
    PolarHead, PolarMove, circular_sd, vm_entropy,
)
from hopfield_nav.rollout.types import RolloutBatch
from hopfield_nav.training.world_setup import move_params, set_phase_freeze
from hopfield_nav.updates.ppo import ppo_update

LO, HI = 0.5, 2.0
BOUNDS = (LO, HI)


def _cfg(**kw):
    base = dict(movement_mode="continuous", hopfield_mode="continuous",
                hidden_size=16, init_log_std=-0.7, action_polar=True)
    base.update(kw)
    return AgentConfig(**base)


def _head(**kw):
    return PolarHead(_cfg(**kw), 16, LO, HI)


def _dist(feat=None, **kw):
    head = _head(**kw)
    feat = torch.randn(4, 6, 16) if feat is None else feat
    return head(feat, torch.randn(*feat.shape[:-1], 2)), head


def _agent(**kw):
    return NavAgent(_cfg(**kw), input_dim=8, action_bounds=BOUNDS)


# ---------------------------------------------------------------------------
# Bounds and shape
# ---------------------------------------------------------------------------

class TestSpeedBounds:

    def test_speed_is_exactly_inside_the_bounds(self):
        """The Beta's support IS the interval, so the env clamp never fires --
        unlike the Cartesian head, where the sample could land anywhere and the
        clamp had to catch it."""
        d, _ = _dist()
        for _ in range(20):
            r = d.sample().norm(dim=-1)
            assert float(r.min()) >= LO - 1e-5
            assert float(r.max()) <= HI + 1e-5

    def test_no_direction_reversal(self):
        """A Normal on the speed would put mass below zero -- 4.7% of it at
        mu=0.5, sigma=0.3 -- and a negative radius points BACKWARDS. The Beta's
        positive support is why this family was chosen over a plain Normal."""
        d, _ = _dist()
        s = d.sample()
        cos = torch.nn.functional.cosine_similarity(s, d.mean, dim=-1)
        # Samples spread around the mean heading but must never be antipodal
        # for a reason as dumb as a negative magnitude.
        assert float(s.norm(dim=-1).min()) > 0.0
        assert cos.shape == s.shape[:-1]

    def test_frozen_speed_is_exact(self):
        d, _ = _dist(freeze_speed=1.0)
        r = d.sample().norm(dim=-1)
        assert torch.allclose(r, torch.ones_like(r), atol=1e-5)
        assert torch.allclose(d.mean.norm(dim=-1), torch.ones_like(r), atol=1e-5)

    def test_frozen_speed_outside_bounds_raises(self):
        """A constant the env would clamp every single step is a
        misconfiguration, not a policy."""
        with pytest.raises(ValueError, match="outside the action bounds"):
            _head(freeze_speed=3.0)

    def test_u_shape_floor_is_enforced(self):
        with pytest.raises(ValueError, match="U-shaped"):
            _head(init_speed_nu=1.28)     # the value that would match Cartesian

    def test_log_prob_and_entropy_carry_a_factor_axis(self):
        """(..., 2) so the `.sum(-1)` every existing call site already applies
        to the Cartesian head yields the joint quantity with no new branch."""
        d, _ = _dist()
        s = d.sample()
        assert d.log_prob(s).shape == (4, 6, 2)
        assert d.entropy().shape == (4, 6, 2)
        assert d.mean.shape == (4, 6, 2)


# ---------------------------------------------------------------------------
# The log-prob contract -- what PPO actually consumes
# ---------------------------------------------------------------------------

class TestPPODynamics:

    def test_ratio_is_exactly_one_at_unchanged_parameters(self):
        d, _ = _dist()
        s = d.sample()
        lp = d.log_prob(s).sum(-1)
        ratio = torch.exp(d.log_prob(s).sum(-1) - lp)
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)

    def test_omitted_jacobians_cancel_in_the_ratio(self):
        """The load-bearing claim. ``log_prob`` drops both ``-log r`` and
        ``-log span``; each depends only on the action, so the ratio computed
        without them must equal the true CARTESIAN ratio computed with them.
        A bias here would never raise -- the run would just learn something
        slightly wrong."""
        feat = torch.randn(8, 5, 16)
        dirs = torch.randn(8, 5, 2)
        h_old, h_new = _head(), _head()
        with torch.no_grad():                      # perturb into a genuinely different policy
            for p_new, p_old in zip(h_new.parameters(), h_old.parameters()):
                p_new.copy_(p_old + 0.3 * torch.randn_like(p_old))
        d_old, d_new = h_old(feat, dirs), h_new(feat, dirs + 0.2)
        a = d_old.sample()

        r = a.norm(dim=-1)
        jac = -(r.log()) - math.log(HI - LO)       # log|d(theta,u)/d(ax,ay)|
        cart_old = d_old.log_prob(a).sum(-1) + jac
        cart_new = d_new.log_prob(a).sum(-1) + jac

        ours = torch.exp(d_new.log_prob(a).sum(-1) - d_old.log_prob(a).sum(-1))
        truth = torch.exp(cart_new - cart_old)
        assert torch.allclose(ours, truth, rtol=1e-5, atol=1e-6)

    def test_log_prob_round_trips_through_the_cartesian_action(self):
        """The buffer stores a Cartesian action, so log_prob must recover
        (theta, r) from it exactly. atan2 is exact and ||a|| IS the sampled
        speed -- unlike inverting a squash, which loses precision at
        saturation."""
        d, _ = _dist()
        a = d.sample()
        theta = torch.atan2(a[..., 1], a[..., 0])
        r = a.norm(dim=-1)
        rebuilt = torch.stack([r * theta.cos(), r * theta.sin()], -1)
        assert torch.allclose(a, rebuilt, atol=1e-5)

    def test_entropy_matches_monte_carlo(self):
        """-E[log p] under the polar measure. Catches a wrong Bessel identity
        in ``vm_entropy``, which torch does not provide and so is hand-rolled."""
        for kappa in (0.5, 2.0, 6.34, 29.4):
            k = torch.tensor(kappa)
            d = PolarMove(torch.zeros(200000), k.expand(200000),
                          torch.full((200000,), 0.4),
                          torch.full((200000,), 5.0), lo=LO, hi=HI)
            s = d.sample()
            mc = -d.log_prob(s).sum(-1).mean()
            assert float(d.entropy().sum(-1)[0]) == pytest.approx(float(mc), abs=0.02), kappa

    def test_entropy_has_no_monotone_speed_preference(self):
        """The reason entropy is taken in POLAR coordinates. The Cartesian
        entropy differs by E[log r], and a log-normal speed's entropy carries a
        bare ``+m`` -- both of which pay the policy to take bigger steps. The
        Beta's entropy does depend on mu, but SYMMETRICALLY about mid-speed, so
        it cannot be increased by going faster."""
        nu = torch.tensor(5.0)
        def H(mu):
            return float(PolarMove(torch.zeros(1), torch.tensor([6.0]),
                                   torch.tensor([mu]), nu.expand(1),
                                   lo=LO, hi=HI).entropy()[0, 1])
        assert H(0.3) == pytest.approx(H(0.7), abs=1e-5)
        assert H(0.5) > H(0.8) and H(0.5) > H(0.2)

    def test_ppo_update_runs_and_reports_polar_diagnostics(self):
        agent = _agent()
        stats = _run_ppo(agent)
        for k in ("mu_norm", "sigma", "ang_noise", "kappa", "move_loss",
                  "move_entropy"):
            assert math.isfinite(stats[k]), k
        assert LO <= stats["mu_norm"] <= HI
        assert stats["kappa"] > 0

    def test_cartesian_update_omits_kappa_entirely(self):
        """Absent, not 0.0 (which would plot as a real measurement on the
        shared axis) and not NaN (test_smoke_train asserts every per-update
        field is finite, and that invariant catches genuinely broken runs)."""
        agent = NavAgent(_cfg(action_polar=False), input_dim=8,
                         action_bounds=BOUNDS)
        assert "kappa" not in _run_ppo(agent)

    def test_ppo_update_moves_every_polar_parameter(self):
        """If the ratio were one factor short -- the ``isinstance(move_dist,
        Normal)`` the sum used to be gated on -- the speed parameters would sit
        exactly still while the run looked healthy."""
        agent = _agent()
        before = {n: p.detach().clone()
                  for n, p in agent.polar_head.named_parameters()}
        _run_ppo(agent, epochs=4)
        moved = {n for n, p in agent.polar_head.named_parameters()
                 if not torch.allclose(p.detach(), before[n])}
        assert moved == set(before), f"stuck: {set(before) - moved}"

    def test_ratios_stay_near_one_after_a_small_step(self):
        """A blown-up importance ratio is the classic way a new action
        parameterization destroys a PPO run without erroring.

        Bounded at the 99th percentile rather than the max: the ratio is
        heading-dominated (measured, the speed factor moves log-prob by 0.005
        against the heading's 0.283) and the max over a few hundred samples of
        a heavy-tailed quantity is seed noise. The tail itself is what PPO's
        clip exists for; a shifted BODY would be the real failure.
        """
        torch.manual_seed(0)
        agent = _agent()
        obs = torch.randn(32, 8, 8)
        with torch.no_grad():
            d0 = agent(obs)[0]
            a = d0.sample()
            lp0 = d0.log_prob(a).sum(-1)
        opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
        loss = -(agent(obs)[0].log_prob(a).sum(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ratio = torch.exp(agent(obs)[0].log_prob(a).sum(-1) - lp0).flatten()
        assert float(ratio.median()) == pytest.approx(1.0, abs=0.05)
        assert float(ratio.quantile(0.99)) < 1.6
        assert float(ratio.quantile(0.01)) > 0.6

    def test_heading_gradient_is_bounded_as_the_direction_vector_shrinks(self):
        """``||v||`` is a GAUGE FREEDOM -- atan2 is scale-invariant, so nothing
        in the objective pressures the direction head's magnitude and it random
        walks, while the heading gradient goes as kappa/||v||. Measured without
        the softening: 26 at ||v||=0.24, 636 at 0.01, 6360 at 0.001, unbounded
        at 0. One such sample reaching clip_grad_norm_ rescales the whole
        batch's update to nothing -- how the first p9_e_sq_std run died."""
        head = _head()
        worst = 0.0
        # The sweep deliberately straddles the degenerate guard (||v||=1e-3)
        # and the shrink knee (||v||=dir_soft): a floor/threshold pair chosen
        # apart would leave a divergent band BETWEEN the sampled points, which
        # is how the 1000 got through the first version of this test.
        for vn in (0.0, 1e-9, 1e-6, 5e-4, 1e-3, 2e-3, 0.005, 0.01, 0.02,
                   0.05, 0.071, 0.1, 0.24, 1.0):
            v = torch.tensor([[vn, 0.0]], requires_grad=True)
            d = head(torch.zeros(1, 16), v)
            d.log_prob(torch.tensor([[0.0, 1.0]])).sum().backward()
            assert torch.isfinite(v.grad).all(), vn
            worst = max(worst, float(v.grad.abs().max()))
        # Peaks at ||v|| = dir_soft; measured 318 there against unbounded
        # without the softening. The bound is generous enough to survive a
        # dir_soft retune but tight enough to fail if the softening is lost.
        assert worst < 500.0, worst

    def test_dir_soft_barely_touches_kappa_where_the_policy_operates(self):
        """The softening must be a BACKSTOP, not a live participant. It was
        0.05 until the first smoke run showed the real 1024-unit trunk emits
        ||v|| ~ 0.071 at init, where 0.05 cut kappa by a third and silently
        broke the calibration against the p9 arms."""
        head = _head()
        feat = torch.zeros(1, 16)
        far = head(feat, torch.tensor([[1.0, 0.0]])).kappa.detach()
        assert float(far) == pytest.approx(6.36, rel=0.005)
        # The measured init operating point: the distortion must be ~1%, not 30%.
        init = head(feat, torch.tensor([[0.071, 0.0]])).kappa.detach()
        assert float(init) == pytest.approx(6.36, rel=0.03)
        near = head(feat, torch.tensor([[0.002, 0.0]])).kappa.detach()
        assert float(near) < 0.15 * float(far)      # -> near-uniform heading

    def test_gradient_ascent_turns_the_heading_toward_a_target(self):
        """End-to-end: the score-function gradient through atan2 and
        VonMises.log_prob actually steers the policy. VonMises has no rsample,
        so if this works the gradient is genuinely flowing the PPO way."""
        torch.manual_seed(0)
        head = _head()
        feat = torch.ones(256, 16) * 0.1
        direction = torch.nn.Parameter(torch.tensor([[-1.0, 0.0]]).repeat(256, 1))
        opt = torch.optim.Adam([direction] + list(head.parameters()), lr=0.05)
        target = 0.0                                     # want heading -> +x
        for _ in range(150):
            d = head(feat, direction)
            a = d.sample().detach()
            adv = torch.cos(torch.atan2(a[..., 1], a[..., 0]) - target)
            loss = -(d.log_prob(a).sum(-1) * (adv - adv.mean())).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        final = torch.atan2(direction[:, 1], direction[:, 0]).mean().detach()
        assert abs(float(final)) < 0.5, f"heading stalled at {float(final):.3f} rad"


class TestUpdateSurvivesNonFiniteSamples:
    """Both P10 arms died in update 2 with EVERY entry of the heading NaN.

    The mechanism is generic to this PPO loop but polar is what made it live:
    a von Mises whose kappa reaches its ceiling can put ~2*kappa between two
    log-probs, and exp() of that is `inf` in float32. `inf` then meets a zero
    mask -- and `inf * 0` is NaN, so the mask that exists to REMOVE the
    offending steps is what converted them into a NaN. `clip_grad_norm_` then
    scaled by `max_norm / inf` = 0, writing NaN into the parameters for good.
    """

    def test_an_overflowing_ratio_on_a_masked_step_stays_finite(self):
        agent = _agent()
        B, T = 6, 8
        obs = torch.randn(B, T, 8)
        with torch.no_grad():
            md, sd, values, _ = agent(obs)
            a = md.sample()
            st = sd.sample()
            lp = md.log_prob(a).sum(-1)
        # One masked step whose stored log-prob is absurd -- exactly what an
        # epsilon action re-scored under a sharp policy looks like.
        lp = lp.clone()
        lp[0, 0] = -1e4
        mask = torch.ones(B, T)
        mask[0, 0] = 0.0
        batch = RolloutBatch(
            obs=obs, move_actions=a, store_actions=st, move_log_probs=lp,
            store_log_probs=sd.log_prob(st), values=values,
            rewards=torch.randn(B, T) * 0.1, bootstrap_value=torch.zeros(B),
            goal_reached=torch.zeros(B, T), explore_mask=torch.ones(B, T),
            policy_action_mask=mask)
        stats = ppo_update(agent, [batch], PPOConfig(ppo_epochs=1,
                                                     n_minibatches=1),
                           torch.optim.Adam(agent.parameters(), lr=1e-3))
        assert math.isfinite(stats["move_loss"])
        for n, p in agent.named_parameters():
            assert torch.isfinite(p).all(), n

    def test_a_nonfinite_gradient_skips_the_step_instead_of_poisoning_it(self):
        agent = _agent()
        before = {n: p.detach().clone() for n, p in agent.named_parameters()}
        B, T = 6, 8
        obs = torch.randn(B, T, 8)
        with torch.no_grad():
            md, sd, values, _ = agent(obs)
            a, st = md.sample(), sd.sample()
            lp = md.log_prob(a).sum(-1)
        rew = torch.randn(B, T) * 0.1
        rew[0, 0] = float("inf")            # -> inf return -> inf value loss
        batch = RolloutBatch(
            obs=obs, move_actions=a, store_actions=st, move_log_probs=lp,
            store_log_probs=sd.log_prob(st), values=values, rewards=rew,
            bootstrap_value=torch.zeros(B), goal_reached=torch.zeros(B, T),
            explore_mask=torch.ones(B, T))
        stats = ppo_update(agent, [batch], PPOConfig(ppo_epochs=1,
                                                     n_minibatches=1),
                           torch.optim.Adam(agent.parameters(), lr=1e-3))
        assert stats["nonfinite_steps"] >= 1.0
        for n, p in agent.named_parameters():
            assert torch.isfinite(p).all(), n
            assert torch.allclose(p.detach(), before[n]), f"{n} was stepped"


def _run_ppo(agent, epochs=1):
    B, T = 6, 8
    obs = torch.randn(B, T, 8)
    with torch.no_grad():
        md, sd, values, _ = agent(obs)
        move_a = md.sample()
        store_a = sd.sample()
        lp = md.log_prob(move_a)
        lp = lp.sum(-1) if lp.dim() > 2 else lp
        batch = RolloutBatch(
            obs=obs, move_actions=move_a, store_actions=store_a,
            move_log_probs=lp, store_log_probs=sd.log_prob(store_a),
            values=values, rewards=torch.randn(B, T) * 0.1,
            bootstrap_value=torch.zeros(B),
            goal_reached=torch.zeros(B, T),
            explore_mask=torch.ones(B, T))
    cfg = PPOConfig(ppo_epochs=epochs, n_minibatches=2)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    return ppo_update(agent, [batch], cfg, opt)


# ---------------------------------------------------------------------------
# Freezing -- the contract (alpha, beta) could not express
# ---------------------------------------------------------------------------

class TestFreezing:

    def test_frozen_speed_contributes_nothing_to_log_prob_or_entropy(self):
        """Deleted, not driven to a degenerate limit: a zero-variance Normal
        would give +inf log-prob and -inf entropy, and both would poison the
        surrogate."""
        d, _ = _dist(freeze_speed=1.0)
        s = d.sample()
        assert torch.equal(d.log_prob(s)[..., 1], torch.zeros(4, 6))
        assert torch.equal(d.entropy()[..., 1], torch.zeros(4, 6))
        assert torch.isfinite(d.log_prob(s)).all()
        assert torch.isfinite(d.entropy()).all()

    def test_frozen_speed_builds_no_speed_parameters_at_all(self):
        head = _head(freeze_speed=1.0)
        assert head.speed_mu is None and head.speed_nu is None
        assert not any("speed" in n for n, _ in head.named_parameters())

    def test_freeze_log_std_freezes_the_spreads_and_not_the_mean(self):
        """The whole point of (mu, nu): 'freeze the spread, keep the mean' is
        inexpressible in (alpha, beta), where holding alpha fixed and learning
        beta moves both."""
        head = _head(freeze_log_std=True)
        assert head.log_kappa.requires_grad is False
        assert head.speed_nu.requires_grad is False
        assert head.speed_mu.requires_grad is True

    def test_set_phase_freeze_does_not_silently_unfreeze_the_spreads(self):
        """--freeze_log_std was a no-op on train_navigate for the whole v35
        lineage because unfreezing the movement head re-enabled it. Polar adds
        two more parameters that could go the same way."""
        agent = _agent(freeze_log_std=True)
        set_phase_freeze(agent, freeze_move=False, freeze_store=True,
                         freeze_value=False, freeze_rnn=False)
        assert agent.polar_head.log_kappa.requires_grad is False
        assert agent.polar_head.speed_nu.requires_grad is False
        assert agent.polar_head.speed_mu.requires_grad is True

    def test_move_params_collects_the_polar_parameters(self):
        """Omitting them would leave the whole spread parameterization frozen
        -- the same class of bug as the `[None]` this function was fixed for."""
        agent = _agent()
        names = {id(p) for p in move_params(agent)}
        for p in agent.polar_head.parameters():
            assert id(p) in names

    def test_frozen_speed_still_trains_the_heading(self):
        agent = _agent(freeze_speed=1.0)
        before = agent.polar_head.log_kappa.detach().clone()
        _run_ppo(agent, epochs=4)
        assert not torch.allclose(agent.polar_head.log_kappa.detach(), before)


# ---------------------------------------------------------------------------
# Initialization and calibration against the Cartesian arms
# ---------------------------------------------------------------------------

class TestInit:

    def test_defaults_sit_on_the_designed_operating_point(self):
        # A DEFINITE direction: kappa carries the dir_soft shrink, and a random
        # 2-D Gaussian direction has real mass near the origin, so averaging
        # over one would report a number about the sampling distribution rather
        # than about the head.
        head = _head()
        d = head(torch.zeros(8, 16), torch.tensor([[1.0, 0.0]]).expand(8, 2))
        assert float(d.speed_mean.mean()) == pytest.approx(1.25, abs=1e-4)
        assert float(d.speed_std.mean()) == pytest.approx(0.375, abs=1e-3)
        assert float(d.kappa.mean()) == pytest.approx(6.36, abs=0.05)

    def test_init_kappa_matches_the_cartesian_init_angular_noise(self):
        """sigma = exp(-0.7) at mid-speed 1.25 is 0.397 rad; kappa = 6.34 gives
        23.8 deg. So a polar run starts with the SAME directional exploration
        as the Cartesian arm it is compared against."""
        cart = math.degrees(math.exp(-0.7) / 1.25)
        got = math.degrees(float(circular_sd(torch.tensor(6.34))))
        assert got == pytest.approx(cart, rel=0.05)

    def test_circular_sd_is_calibrated_to_the_section_9_3_column(self):
        """Section 9.3 measured 10.56 deg of sigma/||mu|| at n_dist=0. That is
        kappa = 29.4, which must report back here as ~10.66 deg -- the ~1%
        agreement is what lets both parameterizations share one axis."""
        got = math.degrees(float(circular_sd(torch.tensor(29.4))))
        assert got == pytest.approx(10.66, abs=0.1)

    def test_state_dependent_heads_start_at_the_global_values(self):
        """Zero weights, init in the bias -- so step one reproduces the
        global-parameter policy and state-dependence has to be learned."""
        feat = torch.randn(32, 16) * 10.0
        # Same direction tensor for both: kappa now carries the dir_soft
        # shrink factor, so two different draws would differ for a reason that
        # has nothing to do with the heads.
        dirs = torch.randn(32, 2)
        g = _head()(feat, dirs)
        s = _head(state_dependent_std=True)(feat, dirs)
        assert torch.allclose(g.kappa, s.kappa, atol=1e-4)
        assert torch.allclose(g.speed_mean, s.speed_mean, atol=1e-4)
        assert torch.allclose(g.speed_std, s.speed_std, atol=1e-4)

    def test_mu_is_exactly_the_mean_speed(self):
        """The readability half of choosing (mu, nu): the logged column IS the
        parameter, so 'did the head act?' is answerable from one number."""
        head = _head()
        with torch.no_grad():
            head.speed_mu.fill_(math.log(0.8 / 0.2))
        d = head(torch.zeros(64, 16), torch.randn(64, 2))
        emp = torch.stack([d.sample().norm(dim=-1) for _ in range(400)]).mean()
        assert float(d.speed_mean.mean()) == pytest.approx(LO + 1.5 * 0.8, abs=1e-4)
        assert float(emp) == pytest.approx(float(d.speed_mean.mean()), abs=0.02)


# ---------------------------------------------------------------------------
# state_dict: the default path must not move
# ---------------------------------------------------------------------------

class TestStateDict:

    def test_keys_unchanged_when_polar_is_off(self):
        keys = set(NavAgent(_cfg(action_polar=False), input_dim=8,
                            action_bounds=BOUNDS).state_dict())
        assert "movement_log_std" in keys
        assert not any(k.startswith("polar_head") for k in keys)

    def test_polar_adds_its_own_keys_and_drops_the_gaussian_sigma(self):
        keys = set(_agent().state_dict())
        assert "movement_mean.weight" in keys           # direction head reused
        assert "polar_head.log_kappa" in keys
        assert "polar_head.speed_mu" in keys
        assert "polar_head.speed_nu" in keys
        assert "movement_log_std" not in keys

    def test_a_cartesian_checkpoint_cannot_load_into_a_polar_agent(self):
        """Loudly, not silently: the sigma key is absent rather than present
        and unused, so the shapes genuinely disagree."""
        cart = NavAgent(_cfg(action_polar=False), input_dim=8,
                        action_bounds=BOUNDS).state_dict()
        with pytest.raises(RuntimeError):
            _agent().load_state_dict(cart)

    def test_polar_needs_action_bounds(self):
        with pytest.raises(ValueError, match="action_polar needs"):
            NavAgent(_cfg(), input_dim=8)


# ---------------------------------------------------------------------------
# Numerical edges -- one NaN reaches clip_grad_norm_ and zeroes the batch
# ---------------------------------------------------------------------------

class TestNumericalEdges:

    @pytest.mark.parametrize("log_kappa", [-1.0, 0.0, 1.85, 3.4, 5.0])
    def test_entropy_and_circular_sd_are_finite_across_the_clamp(self, log_kappa):
        k = torch.tensor(math.exp(log_kappa))
        assert torch.isfinite(vm_entropy(k)).all()
        assert torch.isfinite(circular_sd(k)).all()

    def test_large_kappa_does_not_overflow(self):
        """I0(148) overflows float32; the scaled Bessels are why this holds."""
        assert torch.isfinite(vm_entropy(torch.tensor(148.0))).all()
        assert float(circular_sd(torch.tensor(148.0))) < 0.1

    def test_off_policy_actions_outside_the_speed_range_stay_finite(self):
        """epsilon-greedy and auto-nav overrides are re-scored under the
        current policy before policy_action_mask excludes them; an inf here
        would reach the ratio first."""
        d, _ = _dist()
        for mag in (0.0, 0.01, LO, HI, 5.0, 100.0):
            a = torch.full((4, 6, 2), mag / math.sqrt(2))
            assert torch.isfinite(d.log_prob(a)).all(), mag

    def test_gradients_are_finite_and_bounded_at_the_parameter_edges(self):
        for kw in ({}, dict(init_speed_mu=0.06), dict(init_speed_mu=0.94),
                   dict(init_log_kappa=5.0), dict(init_log_kappa=-1.0)):
            head = _head(**kw)
            feat = torch.randn(64, 16)
            d = head(feat, torch.randn(64, 2))
            d.log_prob(d.sample()).sum().backward()
            for n, p in head.named_parameters():
                assert torch.isfinite(p.grad).all(), (kw, n)
                assert float(p.grad.abs().max()) < 1e4, (kw, n)

    def test_temperature_scales_both_spreads_and_leaves_frozen_speed_alone(self):
        d, _ = _dist()
        warm = d.with_temperature(2.0)
        assert float(circular_sd(warm.kappa).mean()) > float(circular_sd(d.kappa).mean())
        assert float(warm.speed_std.mean()) > float(d.speed_std.mean())
        f, _ = _dist(freeze_speed=1.0)
        assert torch.equal(f.with_temperature(2.0).speed_std, f.speed_std)
