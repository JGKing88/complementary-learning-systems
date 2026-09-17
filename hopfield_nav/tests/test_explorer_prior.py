"""The explorer prior (training/prior.py) and the polar KL it is built on.

Two properties carry the whole thing. The KL has to be a KL -- zero when the
two distributions coincide, positive otherwise, and equal to what Monte Carlo
says -- because it is written by hand (torch registers none for VonMises).
And the prior has to be inert at the anchor and bite after a step: an EWC
penalty of 0 at theta* that grows once the weights move, a KL of 0 while the
teacher and student agree, and neither reaching the value head.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch
from torch.distributions import Beta, VonMises

from hopfield_nav.config import AgentConfig, PPOConfig
from hopfield_nav.policy.agent import NavAgent
from hopfield_nav.policy.polar_head import PolarMove, polar_kl, vonmises_kl
from hopfield_nav.rollout.types import RolloutBatch
from hopfield_nav.training.prior import ExplorerPrior
from hopfield_nav.updates.ppo import ppo_update

LO, HI = 0.5, 1.0
B, T, D = 6, 10, 8


# --------------------------------------------------------------------------
# vonmises_kl / polar_kl
# --------------------------------------------------------------------------

def _mc_vm_kl(mu1, k1, mu2, k2, n=200_000, seed=0):
    torch.manual_seed(seed)
    p = VonMises(torch.tensor(mu1), torch.tensor(k1))
    q = VonMises(torch.tensor(mu2), torch.tensor(k2))
    x = p.sample((n,))
    return float((p.log_prob(x) - q.log_prob(x)).mean())


class TestVonMisesKL:
    def test_zero_when_identical(self):
        mu = torch.tensor([0.3, -2.0, 3.0])
        k = torch.tensor([0.5, 6.0, 40.0])
        assert torch.allclose(vonmises_kl(mu, k, mu, k), torch.zeros(3), atol=1e-6)

    @pytest.mark.parametrize("mu1,k1,mu2,k2", [
        (0.0, 2.0, 0.5, 2.0),      # heading shift only
        (0.0, 2.0, 0.0, 8.0),      # concentration only
        (1.0, 12.0, -1.0, 3.0),    # both
        (0.0, 0.3, 3.0, 0.3),      # near-uniform, opposite headings
    ])
    def test_matches_monte_carlo(self, mu1, k1, mu2, k2):
        ana = float(vonmises_kl(torch.tensor(mu1), torch.tensor(k1),
                                torch.tensor(mu2), torch.tensor(k2)))
        mc = _mc_vm_kl(mu1, k1, mu2, k2)
        assert ana == pytest.approx(mc, abs=0.01, rel=0.02)

    def test_nonnegative_and_periodic(self):
        mu1 = torch.linspace(-3, 3, 13)
        k1 = torch.full_like(mu1, 4.0)
        mu2 = torch.zeros_like(mu1)
        k2 = torch.full_like(mu1, 2.0)
        kl = vonmises_kl(mu1, k1, mu2, k2)
        assert (kl >= -1e-6).all()
        shifted = vonmises_kl(mu1 + 2 * math.pi, k1, mu2, k2)
        assert torch.allclose(kl, shifted, atol=1e-5)

    def test_large_kappa_is_finite(self):
        # I0(148) overflows float32; the scaled Bessels keep this finite.
        k = torch.tensor([148.0])
        kl = vonmises_kl(torch.tensor([0.0]), k, torch.tensor([0.1]), k)
        assert torch.isfinite(kl).all() and float(kl) > 0


def _polar(theta, kappa, mu=None, nu=None, const=None):
    return PolarMove(theta, kappa, mu, nu, lo=LO, hi=HI, speed_const=const)


class TestPolarKL:
    def test_zero_when_identical(self):
        th = torch.tensor([0.1, 1.0]); ka = torch.tensor([3.0, 9.0])
        mu = torch.tensor([0.4, 0.6]); nu = torch.tensor([5.0, 12.0])
        kl = polar_kl(_polar(th, ka, mu, nu), _polar(th, ka, mu, nu))
        assert torch.allclose(kl, torch.zeros(2), atol=1e-6)

    def test_is_heading_plus_speed(self):
        th1, ka1 = torch.tensor([0.0]), torch.tensor([3.0])
        th2, ka2 = torch.tensor([0.7]), torch.tensor([5.0])
        mu1, nu1 = torch.tensor([0.3]), torch.tensor([6.0])
        mu2, nu2 = torch.tensor([0.6]), torch.tensor([4.0])
        kl = polar_kl(_polar(th1, ka1, mu1, nu1), _polar(th2, ka2, mu2, nu2))
        vm = vonmises_kl(th1, ka1, th2, ka2)
        beta = torch.distributions.kl_divergence(
            Beta(mu1 * nu1, (1 - mu1) * nu1), Beta(mu2 * nu2, (1 - mu2) * nu2))
        assert torch.allclose(kl, vm + beta, atol=1e-6)

    def test_constant_speed_is_heading_only(self):
        th1, ka1 = torch.tensor([0.0]), torch.tensor([3.0])
        th2, ka2 = torch.tensor([0.7]), torch.tensor([5.0])
        kl = polar_kl(_polar(th1, ka1, const=0.7), _polar(th2, ka2, const=0.7))
        assert torch.allclose(kl, vonmises_kl(th1, ka1, th2, ka2))

    def test_mismatched_speed_models_refuse(self):
        th, ka = torch.tensor([0.0]), torch.tensor([3.0])
        with pytest.raises(ValueError):
            polar_kl(_polar(th, ka, const=0.7),
                     _polar(th, ka, torch.tensor([0.5]), torch.tensor([4.0])))
        with pytest.raises(ValueError):
            polar_kl(_polar(th, ka, const=0.7), _polar(th, ka, const=0.9))


# --------------------------------------------------------------------------
# ExplorerPrior
# --------------------------------------------------------------------------

def _agent(seed=0):
    torch.manual_seed(seed)
    cfg = AgentConfig(movement_mode="continuous", hopfield_mode="continuous",
                      hidden_size=16, action_polar=True,
                      state_dependent_std=True, input_encoded_state=False,
                      input_hopfield_signal=False)
    return NavAgent(cfg, input_dim=D, action_bounds=(LO, HI))


def _rollout(seed=0, explore_rows=None):
    """A task-shaped batch: rows 0..explore_rows-1 spend their first half
    searching (explore_mask 1), the rest is post-store."""
    torch.manual_seed(seed)
    mask = torch.zeros(B, T)
    n = B if explore_rows is None else explore_rows
    mask[:n, : T // 2] = 1.0
    ang = torch.rand(B, T) * 2 * math.pi
    r = LO + (HI - LO) * torch.rand(B, T)
    act = torch.stack([r * ang.cos(), r * ang.sin()], -1)
    return RolloutBatch(
        obs=torch.randn(B, T, D),
        move_actions=act,
        store_actions=torch.zeros(B, T),
        move_log_probs=torch.zeros(B, T),
        store_log_probs=torch.zeros(B, T),
        values=torch.zeros(B, T),
        rewards=torch.randn(B, T),
        bootstrap_value=torch.zeros(B),
        goal_reached=torch.zeros(B, T),
        explore_mask=mask,
        alive_mask=torch.ones(B, T),
        policy_action_mask=torch.ones(B, T),
    )


def _perturb(agent, scale=0.05, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in agent.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))


class TestExplorerPriorEWC:
    def test_needs_fisher_then_penalty_is_zero_at_anchor(self):
        agent = _agent()
        prior = ExplorerPrior(agent, ewc_lambda=10.0)
        assert prior.needs_fisher and prior.penalty(agent) is None
        stats = prior.estimate_fisher(agent, [_rollout()])
        assert not prior.needs_fisher
        assert stats["fisher_rows"] == B and stats["fisher_max"] > 0
        pen = prior.penalty(agent)
        assert pen is not None and float(pen.detach()) == 0.0

    def test_penalty_grows_with_drift(self):
        agent = _agent()
        prior = ExplorerPrior(agent, ewc_lambda=10.0)
        prior.estimate_fisher(agent, [_rollout()])
        _perturb(agent, 0.01)
        small = float(prior.penalty(agent))
        _perturb(agent, 0.05, seed=2)
        large = float(prior.penalty(agent))
        assert 0 < small < large
        assert prior.drift(agent) > 0

    def test_fisher_uses_search_steps_only(self):
        # Rows with no search step contribute nothing; a batch with none
        # at all has nothing to estimate on and says so.
        agent = _agent()
        prior = ExplorerPrior(agent, ewc_lambda=1.0)
        stats = prior.estimate_fisher(agent, [_rollout(explore_rows=2)])
        assert stats["fisher_rows"] == 2
        prior2 = ExplorerPrior(agent, ewc_lambda=1.0)
        with pytest.raises(RuntimeError):
            prior2.estimate_fisher(agent, [_rollout(explore_rows=0)])

    def test_value_and_store_heads_are_free(self):
        # The Fisher is of the movement log-prob: the value and store heads
        # get no importance, so drifting them costs nothing.
        agent = _agent()
        prior = ExplorerPrior(agent, ewc_lambda=10.0)
        prior.estimate_fisher(agent, [_rollout()])
        with torch.no_grad():
            for p in agent.value_head.parameters():
                p.add_(1.0)
            for p in agent.store_head.parameters():
                p.add_(1.0)
        assert float(prior.penalty(agent).detach()) == 0.0

    def test_fisher_rows_cap(self):
        agent = _agent()
        prior = ExplorerPrior(agent, ewc_lambda=1.0, fisher_trajectories=3)
        stats = prior.estimate_fisher(agent, [_rollout(), _rollout(seed=5)])
        assert stats["fisher_rows"] == 3


class TestExplorerPriorKL:
    def _pool(self, agent, prior, r):
        prior.begin_update(r.obs)
        idx = torch.arange(B)
        with torch.no_grad():
            dist, _, _, _ = agent(r.obs)
        return idx, dist

    def test_zero_while_student_is_the_teacher(self):
        agent = _agent()
        prior = ExplorerPrior(agent, kl_coef=1.0)
        r = _rollout()
        idx, dist = self._pool(agent, prior, r)
        kl = prior.kl(idx, dist, r.explore_mask)
        assert float(kl) == pytest.approx(0.0, abs=1e-6)

    def test_positive_after_the_student_moves(self):
        agent = _agent()
        prior = ExplorerPrior(agent, kl_coef=1.0)
        r = _rollout()
        _perturb(agent, 0.2)
        idx, dist = self._pool(agent, prior, r)
        assert float(prior.kl(idx, dist, r.explore_mask)) > 1e-4

    def test_mask_selects_the_steps(self):
        agent = _agent()
        prior = ExplorerPrior(agent, kl_coef=1.0)
        r = _rollout()
        _perturb(agent, 0.2)
        idx, dist = self._pool(agent, prior, r)
        full = prior.kl(idx, dist, torch.ones(B, T))
        none = prior.kl(idx, dist, torch.zeros(B, T))
        half = prior.kl(idx, dist, r.explore_mask)
        assert float(none) == 0.0
        assert float(full) > 0 and float(half) > 0
        assert not torch.isclose(full, half)

    def test_teacher_does_not_learn(self):
        agent = _agent()
        prior = ExplorerPrior(agent, kl_coef=1.0)
        assert all(not p.requires_grad for p in prior._teacher.parameters())
        before = copy.deepcopy(prior._teacher.state_dict())
        _perturb(agent, 0.2)
        assert all(torch.equal(before[k], v)
                   for k, v in prior._teacher.state_dict().items())

    def test_indexing_matches_direct_forward(self):
        # The pooled teacher output indexed by a minibatch's rows must be
        # the teacher's output on those rows.
        agent = _agent()
        prior = ExplorerPrior(agent, kl_coef=1.0)
        r = _rollout()
        prior.begin_update(r.obs)
        idx = torch.tensor([4, 1])
        with torch.no_grad():
            direct, _, _, _ = prior._teacher(r.obs[idx])
        _perturb(agent, 0.2)
        with torch.no_grad():
            student, _, _, _ = agent(r.obs[idx])
        via_pool = prior.kl(idx, student, torch.ones(2, T))
        stored = PolarMove(direct.theta, direct.kappa, direct.speed_mu,
                           direct.speed_nu, lo=LO, hi=HI)
        by_hand = polar_kl(stored, student).mean()
        assert torch.allclose(via_pool, by_hand, atol=1e-6)


class TestInsidePPO:
    def _run(self, **prior_kw):
        agent = _agent()
        prior = ExplorerPrior(agent, **prior_kw)
        r = _rollout()
        if prior.needs_fisher:
            prior.estimate_fisher(agent, [r])
        opt = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad],
                               lr=1e-2)
        cfg = PPOConfig(ppo_epochs=2, n_minibatches=2)
        torch.manual_seed(3)
        return ppo_update(agent, [r], cfg, opt, prior=prior), prior, agent

    def test_terms_are_reported(self):
        stats, prior, agent = self._run(ewc_lambda=5.0, kl_coef=1.0)
        assert "prior_ewc" in stats and "prior_kl" in stats
        assert stats["prior_ewc"] >= 0 and stats["prior_kl"] >= 0
        assert prior.drift(agent) > 0

    def test_inert_prior_reports_nothing(self):
        agent = _agent()
        prior = ExplorerPrior(agent)
        assert not prior.active
        r = _rollout()
        opt = torch.optim.Adam(agent.parameters(), lr=1e-2)
        stats = ppo_update(agent, [r], PPOConfig(ppo_epochs=1, n_minibatches=1),
                           opt, prior=prior)
        assert "prior_ewc" not in stats and "prior_kl" not in stats

    def test_ewc_holds_the_weights_closer(self):
        # Same seeds, same data: a large lambda ends nearer the anchor than
        # no prior does. That is the whole point of the term.
        _, free_prior, free_agent = self._run(kl_coef=0.0, ewc_lambda=0.0)
        _, held_prior, held_agent = self._run(ewc_lambda=1e4)
        assert held_prior.drift(held_agent) < free_prior.drift(free_agent)

    def test_kl_holds_search_behaviour_closer(self):
        _, free_prior, free_agent = self._run()
        _, held_prior, held_agent = self._run(kl_coef=50.0)
        r = _rollout()
        teacher = ExplorerPrior(_agent(), kl_coef=1.0)   # same seed = anchor
        teacher.begin_update(r.obs)
        idx = torch.arange(B)
        with torch.no_grad():
            d_free, _, _, _ = free_agent(r.obs)
            d_held, _, _, _ = held_agent(r.obs)
        kl_free = float(teacher.kl(idx, d_free, r.explore_mask))
        kl_held = float(teacher.kl(idx, d_held, r.explore_mask))
        assert kl_held < kl_free
