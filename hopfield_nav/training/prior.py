"""The explorer as a prior for phase 2 (docs/EXPLORE_FIRST_PLAN.md §4).

A policy forked from a trained explorer and then trained on the water-maze
task alone (no novelty, no explore rollouts) has nothing in its objective
that pays for exploring, so whatever it keeps of the explorer it keeps by
accident. This module supplies the two ways to keep it on purpose, both
measured against the same anchor -- the explorer's weights at the fork --
and both free of explore rollouts during training:

  EWC   0.5 * lam * sum_i F_i (theta_i - theta*_i)^2         weight space
  KL    beta * KL(pi_explorer || pi) on before-the-store steps  behaviour space

``F`` is a diagonal Fisher estimated ONCE, on the first update's rollouts
before the first gradient step, and only on the steps the explorer spent
searching (``explore_mask``): the states where exploring is the behaviour
being protected. So the estimate is of the explorer, on the task, doing the
thing the penalty then holds it to. The KL is the same statement in
behaviour space: on the search steps of every later rollout, stay close to
what the explorer would have done there. After the store neither term says
anything -- that is where exploit is learned, and the explorer has no
opinion worth keeping (it has no ``||q||`` gate; DUAL_TRAINING §9).

Neither term touches the value head: the Fisher of the movement log-prob is
zero there, and the KL is over the movement distribution. The value must be
re-learned for the new objective and is free to.

``OnlineEWC`` in ``hopfield_nav/continual/`` is the same idea written for the
sequential RNN driver (two-output agent, BC label masks); this one speaks
``NavAgent``'s four-output forward and PPO's pooled buffer, and estimates
once rather than per block.
"""
from __future__ import annotations

import copy

import torch

from ..policy.polar_head import PolarMove, polar_kl


def _frozen_copy(agent):
    snap = copy.deepcopy(agent)
    snap.eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    return snap


class ExplorerPrior:
    """EWC penalty and/or search-step KL toward the policy passed at construction.

    Construct it from the agent *as loaded from the explorer checkpoint*,
    before any gradient step: the anchor and the teacher are snapshots taken
    here. ``ewc_lambda == 0`` disables the penalty, ``kl_coef == 0`` the KL;
    with both zero the object is inert and ``ppo_update`` can be given
    ``None`` instead.
    """

    def __init__(self, agent, *, ewc_lambda: float = 0.0,
                 kl_coef: float = 0.0, fisher_trajectories: int = 256,
                 normalize_fisher: bool = False) -> None:
        self.ewc_lambda = float(ewc_lambda)
        self.kl_coef = float(kl_coef)
        self.fisher_trajectories = int(fisher_trajectories)
        self.normalize_fisher = bool(normalize_fisher)
        if self.ewc_lambda < 0 or self.kl_coef < 0:
            raise ValueError("ewc_lambda and kl_coef must be >= 0")
        self._anchor = {n: p.detach().clone()
                        for n, p in agent.named_parameters() if p.requires_grad}
        self._fisher: dict[str, torch.Tensor] | None = None
        self._teacher = _frozen_copy(agent) if self.kl_coef > 0 else None
        # Teacher outputs over the current update's pooled buffer, indexed by
        # the minibatch's trajectory rows. Computed once per update: the
        # teacher does not change, and the loop visits every row
        # ``ppo_epochs`` times.
        self._pool_teacher: PolarMove | None = None
        self.fisher_stats: dict[str, float] = {}

    # -- what the trainer asks ------------------------------------------------

    @property
    def active(self) -> bool:
        return self.ewc_lambda > 0 or self.kl_coef > 0

    @property
    def needs_fisher(self) -> bool:
        return self.ewc_lambda > 0 and self._fisher is None

    # -- EWC ---------------------------------------------------------------

    @torch.enable_grad()
    def estimate_fisher(self, agent, rollouts) -> dict[str, float]:
        """Diagonal true Fisher of the movement log-prob on search steps.

        One forward/backward per trajectory (per-sample gradients squared;
        batching them would square the *sum* instead), over up to
        ``fisher_trajectories`` rows that have at least one search step,
        taken in order across the rollouts. Actions are sampled from the
        model, which is what makes it the Fisher rather than the empirical
        Fisher. Each row's log-prob is a masked mean over its search steps,
        matching how PPO's surrogate weights timesteps, so the importances
        sit on the loss's own scale and ``lam`` reads the same way as in the
        continual suite.
        """
        params = {n: p for n, p in agent.named_parameters() if p.requires_grad}
        acc = {n: torch.zeros_like(p) for n, p in params.items()}
        n_rows = 0
        n_steps = 0.0
        was_training = agent.training
        agent.eval()
        try:
            for r in rollouts:
                mask = r.explore_mask
                if r.alive_mask is not None:
                    mask = mask * r.alive_mask
                if r.policy_action_mask is not None:
                    mask = mask * r.policy_action_mask
                for b in range(r.obs.shape[0]):
                    if n_rows >= self.fisher_trajectories:
                        break
                    mk = mask[b:b + 1]
                    if float(mk.sum()) == 0:
                        continue
                    dist, _, _, _ = agent(r.obs[b:b + 1])
                    action = dist.sample().detach()
                    logp = dist.log_prob(action)
                    if logp.dim() > mk.dim():
                        logp = logp.sum(-1)
                    ll = (logp * mk).sum() / mk.sum().clamp_min(1.0)
                    agent.zero_grad(set_to_none=True)
                    ll.backward()
                    for n, p in params.items():
                        if p.grad is not None:
                            acc[n] += p.grad.detach().pow(2)
                    n_rows += 1
                    n_steps += float(mk.sum())
                if n_rows >= self.fisher_trajectories:
                    break
        finally:
            agent.zero_grad(set_to_none=True)
            if was_training:
                agent.train()
        if n_rows == 0:
            raise RuntimeError(
                "explorer prior: no search steps in the first update's "
                "rollouts, so there is nothing to estimate the Fisher on")
        fisher = {n: v / n_rows for n, v in acc.items()}
        fmax = max(float(v.max()) for v in fisher.values())
        if self.normalize_fisher and fmax > 0:
            fisher = {n: v / fmax for n, v in fisher.items()}
        self._fisher = fisher
        total = sum(float(v.sum()) for v in fisher.values())
        n_par = sum(int(v.numel()) for v in fisher.values())
        self.fisher_stats = {
            "fisher_rows": float(n_rows),
            "fisher_steps": n_steps,
            "fisher_max": fmax,
            "fisher_mean": total / max(n_par, 1),
        }
        return dict(self.fisher_stats)

    def penalty(self, agent) -> torch.Tensor | None:
        if self.ewc_lambda == 0 or self._fisher is None:
            return None
        total = None
        for n, p in agent.named_parameters():
            f = self._fisher.get(n)
            a = self._anchor.get(n)
            if f is None or a is None or not p.requires_grad:
                continue
            term = (f * (p - a).pow(2)).sum()
            total = term if total is None else total + term
        if total is None:
            return None
        return 0.5 * self.ewc_lambda * total

    def drift(self, agent) -> float:
        """RMS of ``theta - theta*`` over the anchored parameters (logging)."""
        sq = 0.0
        n = 0
        with torch.no_grad():
            for name, p in agent.named_parameters():
                a = self._anchor.get(name)
                if a is None:
                    continue
                sq += float((p - a).pow(2).sum())
                n += int(p.numel())
        return (sq / max(n, 1)) ** 0.5

    # -- KL ----------------------------------------------------------------

    @torch.no_grad()
    def begin_update(self, obs: torch.Tensor) -> None:
        """Run the teacher over the pooled ``(N, T, D)`` buffer, once."""
        if self._teacher is None:
            self._pool_teacher = None
            return
        dist, _, _, _ = self._teacher(obs)
        if not isinstance(dist, PolarMove):
            raise TypeError("the explorer prior's KL is written for the polar "
                            "head; got %s" % type(dist).__name__)
        self._pool_teacher = dist

    def kl(self, idx: torch.Tensor, move_dist, mask: torch.Tensor) -> torch.Tensor | None:
        """``KL(teacher || student)`` masked-mean over ``mask`` for the rows ``idx``."""
        if self._teacher is None:
            return None
        t = self._pool_teacher
        if t is None:
            raise RuntimeError("explorer prior: begin_update was not called")
        teacher = PolarMove(
            t.theta[idx], t.kappa[idx],
            None if t.speed_mu is None else t.speed_mu[idx],
            None if t.speed_nu is None else t.speed_nu[idx],
            lo=t.lo, hi=t.hi, speed_const=t.speed_const)
        kl = polar_kl(teacher, move_dist)
        return (kl * mask).sum() / mask.sum().clamp_min(1.0)

    # -- the one call ppo_update makes -----------------------------------------

    def loss(self, agent, idx: torch.Tensor, move_dist, search_mask: torch.Tensor,
             ) -> tuple[torch.Tensor | None, dict[str, float]]:
        logs: dict[str, float] = {}
        total = None
        pen = self.penalty(agent)
        if pen is not None:
            logs["prior_ewc"] = float(pen.detach())
            total = pen
        if self.kl_coef > 0:
            kl = self.kl(idx, move_dist, search_mask)
            if kl is not None:
                logs["prior_kl"] = float(kl.detach())
                term = self.kl_coef * kl
                total = term if total is None else total + term
        return total, logs

    def describe(self) -> str:
        parts = []
        if self.ewc_lambda > 0:
            parts.append(f"EWC lambda={self.ewc_lambda:g} "
                         f"(Fisher on the first update's search steps, "
                         f"<= {self.fisher_trajectories} rows"
                         + (", normalized" if self.normalize_fisher else "")
                         + ")")
        if self.kl_coef > 0:
            parts.append(f"KL(explorer || policy) x {self.kl_coef:g} on "
                         f"search steps")
        return "; ".join(parts) if parts else "inert"
