"""PPO: GAE computation and clipped policy update.

Handles both discrete and continuous movement actions + binary store action.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli, Normal

from ..config import PPOConfig
from ..rollout.types import RolloutBatch


# ---------------------------------------------------------------------------
# Rollout data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float,
    lam: float,
    alive: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    ``alive`` is (B, T) 1/0: whether row b's episode was still running at step
    t. ``None`` means every row runs the whole rollout, which is the historical
    case -- truncation at the end and no terminal state inside -- and takes the
    identical arithmetic.

    A row that ends inside the rollout is **terminal**, not truncated: there is
    no future return past a reached goal, so the bootstrap must be cut there.
    Carrying ``gamma * V(next)`` across that boundary is the silent version of
    this bug -- the value head learns that reward keeps flowing after the
    episode is over, and nothing in the loss curve says so.

    rewards, values: (B, T).  bootstrap_value: (B,).
    Returns (advantages, returns) both (B, T).
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B, device=rewards.device)
    last_value = bootstrap_value

    for t in reversed(range(T)):
        if alive is None:
            # 1 where the return continues past t: always, bar truncation,
            # which `bootstrap_value` already accounts for.
            cont = 1.0
        elif t + 1 < T:
            # The episode continues past t exactly when t+1 was also alive.
            cont = alive[:, t + 1]
        else:
            # At the horizon the bootstrap applies only to rows still running.
            #
            # No test distinguishes this from `cont = 1.0`, and none can: a row
            # dead at T-1 has its advantage masked to zero below, and cannot
            # propagate backwards either, because the step before it reads
            # `alive[:, T-1] == 0`. Kept because it states the intent where a
            # reader looks for it -- but it is belt to the braces, not the
            # thing holding this up.
            cont = alive[:, t]
        delta = rewards[:, t] + gamma * last_value * cont - values[:, t]
        last_adv = delta + gamma * lam * cont * last_adv
        advantages[:, t] = last_adv
        last_value = values[:, t]

    returns = advantages + values
    if alive is not None:
        # Steps after a row finished are not transitions; leaving their
        # advantages nonzero would let the loss mask be the only thing standing
        # between them and the gradient.
        advantages = advantages * alive
        returns = returns * alive
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def _pool_rollouts(
    rollouts: list[RolloutBatch],
    gamma: float,
    gae_lambda: float,
) -> dict[str, torch.Tensor]:
    """Concatenate rollouts along the batch (trajectory) axis.

    GAE is computed PER rollout (each has its own bootstrap_value and its
    trajectories live on their own timeline), then advantages/returns are
    concatenated. All other per-trajectory tensors concat along dim 0.

    Returns a dict of (N, T, ...) tensors where N = sum_k B_k.
    """
    advs, rets = [], []
    for r in rollouts:
        a, g = compute_gae(r.rewards, r.values, r.bootstrap_value, gamma,
                           gae_lambda, alive=r.alive_mask)
        advs.append(a)
        rets.append(g)

    # None unless some rollout actually ended an episode early. A pool with no
    # terminations takes the historical code path rather than multiplying by a
    # mask of ones, so runs that do not use the feature are untouched.
    any_alive = any(r.alive_mask is not None for r in rollouts)
    alive = (torch.cat([r.alive_mask if r.alive_mask is not None
                        else torch.ones_like(r.explore_mask)
                        for r in rollouts], dim=0) if any_alive else None)

    # Per-rollout policy_action_mask is optional; default all-ones (every
    # step counts toward move_loss). All rollouts in a pool should agree on
    # whether the mask is populated.
    pol_masks = []
    for r in rollouts:
        if r.policy_action_mask is not None:
            pol_masks.append(r.policy_action_mask)
        else:
            pol_masks.append(torch.ones_like(r.explore_mask))
    # Folding `alive` into the two existing masks is what keeps the store and
    # move surrogates from scoring steps that never happened.
    explore_mask = torch.cat([r.explore_mask for r in rollouts], dim=0)
    pol_mask = torch.cat(pol_masks, dim=0)
    if alive is not None:
        explore_mask = explore_mask * alive
        pol_mask = pol_mask * alive
    return {
        "alive_mask": alive,
        "obs": torch.cat([r.obs for r in rollouts], dim=0),
        "move_actions": torch.cat([r.move_actions for r in rollouts], dim=0),
        "store_actions": torch.cat([r.store_actions for r in rollouts], dim=0),
        "old_move_lp": torch.cat([r.move_log_probs for r in rollouts], dim=0),
        "old_store_lp": torch.cat([r.store_log_probs for r in rollouts], dim=0),
        "goal_reached": torch.cat([r.goal_reached for r in rollouts], dim=0),
        "explore_mask": explore_mask,
        "policy_action_mask": pol_mask,
        "advantages": torch.cat(advs, dim=0),
        "returns": torch.cat(rets, dim=0),
    }


def ppo_update(
    agent: nn.Module,
    rollouts: list[RolloutBatch],
    cfg: PPOConfig,
    optimizer: torch.optim.Optimizer,
    aux_scale: float = 1.0,
) -> dict[str, float]:
    """Run PPO epochs on a pooled rollout buffer with minibatching.

    `rollouts` is the full set of rollouts collected this update (across all
    worlds + envs). They are concatenated along the trajectory axis; advantages
    are computed per-rollout (GAE is trajectory-local) and normalized globally
    across the pool. Each epoch shuffles trajectory indices and splits them
    into `cfg.n_minibatches` minibatches; each minibatch is one full gradient
    step.

    Minibatching is done over full trajectories (rows of length T), not over
    individual timesteps — this preserves each trajectory's RNN temporal
    structure when the agent re-forwards it.

    Args:
        aux_scale: Multiplier applied to auxiliary losses (store_bc_weight) this update,
            for linear annealing from 1.0 → 0.0 over training.

    Returns dict of loss components (averaged over all gradient steps) for logging.
    """
    pool = _pool_rollouts(rollouts, cfg.gamma, cfg.gae_lambda)

    # Whether the store head's terms are allowed into the loss at all.
    #
    # `set_phase_freeze(freeze_store=True)` clears `requires_grad` on the store
    # head's own weights, which stops *those* weights updating but not backprop
    # from flowing through the frozen Linear into the shared RNN trunk. So a
    # "frozen" store head still steered the trunk, every update, via store_loss
    # and the store entropy bonus -- the whole of train_navigate ran that way,
    # as did train_phased phases 2 and 3 (both default freeze_store=True).
    # Freezing a head has to mean its objective is gone, not just that its own
    # weights are pinned.
    #
    # Read off the agent rather than taken as an argument, for the same reason
    # `freeze_log_std` is enforced against the agent's own config: a separately
    # passed flag is a second source of truth that can drift out of step with
    # the freeze it is supposed to describe.
    store_head = getattr(agent, "store_head", None)
    store_trainable = store_head is not None and any(
        p.requires_grad for p in store_head.parameters())

    obs = pool["obs"]                   # (N, T, D)
    move_actions = pool["move_actions"] # (N, T) or (N, T, 2)
    store_actions = pool["store_actions"]
    old_move_lp = pool["old_move_lp"]
    old_store_lp = pool["old_store_lp"]
    goal_reached = pool["goal_reached"]
    explore_mask = pool["explore_mask"]
    policy_action_mask = pool["policy_action_mask"]
    alive_mask = pool["alive_mask"]
    advantages = pool["advantages"]
    returns = pool["returns"]

    # Normalize advantages across the full pool (not per-minibatch — that would
    # inject minibatch-dependent bias into the policy gradient).
    adv_mean = advantages.mean()
    adv_std = advantages.std().clamp_min(1e-8)
    advantages = (advantages - adv_mean) / adv_std

    effective_bc_weight = cfg.store_bc_weight * aux_scale

    N = obs.shape[0]
    n_mb = max(1, min(cfg.n_minibatches, N))
    mb_size = max(1, N // n_mb)

    total_move_loss = 0.0
    total_store_loss = 0.0
    total_value_loss = 0.0
    total_move_ent = 0.0
    total_store_ent = 0.0
    total_mu_norm = total_sigma = total_ang = total_kappa = total_dir = 0.0
    n_diag = n_kappa = 0
    total_store_bc = 0.0
    n_steps = 0
    n_nonfinite = 0
    reported_nonfinite = False

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(N, device=obs.device)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]
            if idx.numel() == 0:
                continue

            mb_obs = obs[idx]
            mb_move_act = move_actions[idx]
            mb_store_act = store_actions[idx]
            mb_old_move_lp = old_move_lp[idx]
            mb_old_store_lp = old_store_lp[idx]
            mb_adv = advantages[idx]
            mb_ret = returns[idx]
            mb_goal = goal_reached[idx]
            mb_mask = explore_mask[idx]
            mb_pol_mask = policy_action_mask[idx]
            mb_alive = None if alive_mask is None else alive_mask[idx]

            # Return features so detached-trunk BCE has access (below). When
            # bce_detach_trunk=False the features tensor is unused — PyTorch
            # keeps its graph alive until the loss backward finishes either way,
            # so the cost of always returning it is negligible.
            move_dist, store_dist, new_values, _, features = agent(
                mb_obs, return_features=True
            )

            # Movement policy loss
            new_move_lp = move_dist.log_prob(mb_move_act)
            if new_move_lp.dim() > mb_old_move_lp.dim():
                # Continuous: sum over the action's factor axis. Duck-typed on
                # the shape rather than `isinstance(move_dist, Normal)`, so the
                # polar head -- whose log_prob returns [heading, speed] on that
                # same axis -- takes this path too. An isinstance check would
                # have silently left the polar ratio one factor short.
                new_move_lp = new_move_lp.sum(-1)
            # CLAMPED before exp. The comment below already notes that ε /
            # auto-nav steps explode the ratio; under a von Mises whose kappa
            # can reach its ceiling the log-prob gap between two policies can
            # reach ~2*kappa, and exp() of that is `inf` in float32. An `inf`
            # then meets `* mb_pol_mask` and `inf * 0` is NaN -- so the mask
            # that exists to REMOVE those steps is what converts them into a
            # NaN that poisons every parameter at once. exp(20) = 4.9e8, far
            # past anything clip_coef admits, so this changes no healthy step.
            log_ratio_move = (new_move_lp - mb_old_move_lp).clamp(-20.0, 20.0)
            ratio_move = torch.exp(log_ratio_move)
            surr1 = ratio_move * mb_adv
            surr2 = torch.clamp(ratio_move, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
            # Mask ε / auto-nav steps out of move_loss — those actions did
            # not come from the policy sample, so including them in the PPO
            # surrogate explodes the importance ratio under narrow std.
            #
            # `torch.where`, not `* mask`: multiplication propagates a
            # non-finite value through a zero, selection does not. The mask is
            # applied to exactly the steps whose surrogate is most likely to be
            # non-finite, which is the worst possible place for that
            # distinction to be wrong.
            pol_mask_sum = mb_pol_mask.sum().clamp_min(1.0)
            _surr = -torch.min(surr1, surr2)
            move_loss = torch.where(mb_pol_mask > 0, _surr,
                                    torch.zeros_like(_surr)).sum() / pol_mask_sum

            # Store policy loss — masked by explore_mask: during exploit the
            # store action is inert (rollout ignores it, no store_cost/
            # store_bonus apply, Hopfield is frozen), so those timesteps carry
            # zero causal signal for the store head. Including them just pumps
            # variance into the store logits and the shared RNN trunk.
            new_store_lp = store_dist.log_prob(mb_store_act)
            ratio_store = torch.exp(
                (new_store_lp - mb_old_store_lp).clamp(-20.0, 20.0))
            surr1_s = ratio_store * mb_adv
            surr2_s = torch.clamp(ratio_store, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
            mask_sum = mb_mask.sum().clamp_min(1.0)
            _surr_s = -torch.min(surr1_s, surr2_s)
            store_loss = torch.where(mb_mask > 0, _surr_s,
                                     torch.zeros_like(_surr_s)).sum() / mask_sum

            # Value loss. Steps after a row's episode ended are not states the
            # value head should be fit to -- the agent was frozen there and the
            # return is defined to be zero, so regressing onto them teaches it
            # that finishing is worth nothing.
            sq_err = (mb_ret - new_values) ** 2
            # Action-parameterization diagnostics. ||mu|| and sigma are what
            # the phase-2 section 8.2 pathology lives in -- the commanded
            # magnitude drifted to 8.18 against a cap of 2.0, collapsing the
            # effective angular noise sigma/||mu||, and it was found only by
            # probing saved checkpoints long afterwards. Logged per update so
            # the next one is visible while it happens.
            if hasattr(move_dist, "diag"):
                # Polar. The columns are deliberately the same three names:
                # mean speed <-> ||mu||, speed sd <-> radial noise, circular sd
                # <-> sigma/||mu||. Calibrated so section 9.3's 10.56 deg reads
                # as 10.66 deg here, i.e. the two parameterizations plot on one
                # axis rather than needing separate panels.
                _d = move_dist.diag()
                total_mu_norm += _d["mu_norm"]
                total_sigma += _d["sigma"]
                total_ang += _d["ang_noise"]
                total_kappa += _d["kappa"]
                total_dir += _d.get("dir_norm", float("nan"))
                n_kappa += 1
                n_diag += 1
            elif hasattr(move_dist, "mean") and move_dist.mean.dim() >= 2 \
                    and move_dist.mean.shape[-1] == 2:
                with torch.no_grad():
                    _mu = move_dist.mean.norm(dim=-1)
                    _sd = move_dist.stddev.mean(-1)
                    total_mu_norm += float(_mu.mean())
                    total_sigma += float(_sd.mean())
                    # The ratio is the quantity that actually governs
                    # directional exploration, so take its mean rather than
                    # the ratio of the means.
                    total_ang += float((_sd / _mu.clamp_min(1e-8)).mean())
                    n_diag += 1
            move_entropy = move_dist.entropy()
            if move_entropy.dim() > 2:
                move_entropy = move_entropy.sum(-1)
            if mb_alive is None:
                # No row ended early: the historical reduction, unchanged.
                value_loss = sq_err.mean()
                move_ent = move_entropy.mean()
            else:
                # `where` rather than `* mb_alive`, for the reason given at
                # move_loss: dead steps are exactly where a stale observation
                # could make these non-finite, and a zero multiplier does not
                # remove a non-finite value, it spreads it.
                alive_sum = mb_alive.sum().clamp_min(1.0)
                keep = mb_alive > 0
                value_loss = torch.where(
                    keep, sq_err, torch.zeros_like(sq_err)).sum() / alive_sum
                move_ent = torch.where(
                    keep, move_entropy,
                    torch.zeros_like(move_entropy)).sum() / alive_sum
            _sent = store_dist.entropy()
            store_ent = torch.where(mb_mask > 0, _sent,
                                    torch.zeros_like(_sent)).sum() / mask_sum

            # Auxiliary BCE loss on store head: directly teach "fire store at
            # goal". Only applied where the store action is eligible (explore
            # phase). pos_weight compensates for class imbalance — goal_reached
            # timesteps are a small fraction of the batch, so an unweighted BCE
            # drives the network to always predict 0 (never store).
            # pos_weight = n_neg / n_pos restores balance.
            if effective_bc_weight > 0:
                # Phase 2 enrichment: route BCE through detached features so
                # its gradient only updates the store_head Linear's own weights,
                # not the shared RNN trunk. Keeps PPO's store log_prob gradient
                # (through store_dist.logits) flowing normally through the trunk.
                if cfg.bce_detach_trunk:
                    store_logits = agent.store_logits_from(features.detach())
                else:
                    store_logits = store_dist.logits
                masked_goal = mb_goal * mb_mask
                n_pos = masked_goal.sum().clamp_min(1.0)
                n_neg = (mb_mask - masked_goal).sum().clamp_min(1.0)
                pos_weight = n_neg / n_pos
                if cfg.bce_pos_weight_cap > 0:
                    pos_weight = pos_weight.clamp_max(cfg.bce_pos_weight_cap)
                bce_full = F.binary_cross_entropy_with_logits(
                    store_logits, mb_goal, reduction="none",
                    pos_weight=pos_weight,
                )
                store_bc_loss = (bce_full * mb_mask).sum() / mask_sum
            else:
                store_bc_loss = torch.zeros((), device=obs.device)

            # The store terms are still computed above, because they are the
            # diagnostics the run logs -- but a frozen store head contributes
            # none of them to the gradient. All three go together: with the
            # head frozen, the BCE can only reach the trunk (detached, it
            # reaches nothing at all), which is the same trunk pollution.
            loss = (
                move_loss
                + cfg.vf_coef * value_loss
                - cfg.ent_coef * move_ent
            )
            if store_trainable:
                loss = (
                    loss
                    + store_loss
                    - cfg.store_ent_coef * store_ent
                    + effective_bc_weight * store_bc_loss
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Computed WITHOUT mutating, because `clip_grad_norm_` scales every
            # gradient in place by `max_norm / (total_norm + 1e-6)` -- and when
            # total_norm is NaN that factor is NaN, so it smears NaN across
            # every parameter and destroys the evidence of which one was
            # actually bad. The first parameter-level report came back with the
            # four RNN tensors named and NOTHING finite anywhere, which is that
            # smearing, not a finding. Third time in this debug that a
            # diagnostic ran after the step that erased what it was reading.
            _grads = [p.grad for p in agent.parameters() if p.grad is not None]
            total_norm = torch.norm(
                torch.stack([g.norm() for g in _grads])) if _grads else \
                torch.zeros((), device=obs.device)
            # SKIP the step on a non-finite gradient rather than taking it.
            #
            # `clip_grad_norm_` scales every gradient by
            # `max_norm / (total_norm + 1e-6)`; when total_norm is inf that
            # factor is 0, and `inf * 0` is NaN -- so one bad sample does not
            # merely dominate the update, it writes NaN into a parameter
            # permanently, and every subsequent forward is NaN. That is how
            # both P10 arms died in update 2 with EVERY entry of the heading
            # NaN at once. One skipped minibatch costs nothing; a poisoned
            # parameter costs the run.
            if torch.isfinite(total_norm):
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()
            else:
                n_nonfinite += 1
                # Report BEFORE clearing. `zero_grad(set_to_none=True)` sets
                # every .grad to None, so inspecting after it reported empty
                # lists for both the non-finite and the finite parameters --
                # which reads as "no gradients anywhere" rather than as "the
                # diagnostic looked too late".
                if not reported_nonfinite:
                    # Named once per update, not per minibatch. Dying with no
                    # information about WHICH term went bad is what made the
                    # first two P10 crashes cost a full diagnosis; this makes
                    # the next occurrence say so itself.
                    reported_nonfinite = True
                    with torch.no_grad():
                        # WHICH PARAMETER, not just which loss. The first
                        # report showed every loss term finite and small
                        # (ratio_max 11.5, all losses < 1), so the non-finite
                        # value is created inside the BACKWARD, not carried in
                        # from the forward -- which rules out an overflowing
                        # ratio and points at gradient amplification. Naming
                        # the parameter separates "the 200-step RNN backward
                        # exploded" from "the polar head emitted it".
                        # The count matters as much as the names: "4 of 4 RNN
                        # tensors" and "4 of 12 parameters" mean different
                        # things, and a truncated list cannot tell them apart.
                        n_par = sum(1 for _, p in agent.named_parameters()
                                    if p.grad is not None)
                        bad = [n for n, p in agent.named_parameters()
                               if p.grad is not None
                               and not torch.isfinite(p.grad).all()]
                        big = sorted(
                            ((float(p.grad.abs().max()), n)
                             for n, p in agent.named_parameters()
                             if p.grad is not None and torch.isfinite(p.grad).all()),
                            reverse=True)[:3]
                        print("  [ppo] non-finite gradient, step skipped: "
                              f"ratio_max={float(ratio_move.max()):.3e} "
                              f"logratio_max={float(log_ratio_move.abs().max()):.3f} "
                              f"adv_absmax={float(mb_adv.abs().max()):.3e} "
                              f"move_loss={float(move_loss):.4g} "
                              f"value_loss={float(value_loss):.4g} "
                              f"move_ent={float(move_ent):.4g} "
                              f"| nonfinite {len(bad)}/{n_par}: {bad[:4]} "
                              f"| largest_finite="
                              f"{[(n, f'{v:.3e}') for v, n in big]}",
                              flush=True)
                optimizer.zero_grad(set_to_none=True)

            total_move_loss += move_loss.item()
            total_store_loss += store_loss.item()
            total_value_loss += value_loss.item()
            total_move_ent += move_ent.item()
            total_store_ent += store_ent.item()
            total_store_bc += store_bc_loss.item()
            n_steps += 1

    denom = max(n_steps, 1)
    d_diag = max(n_diag, 1)
    stats = {
        "mu_norm": total_mu_norm / d_diag,
        "sigma": total_sigma / d_diag,
        "ang_noise": total_ang / d_diag,
        "move_loss": total_move_loss / denom,
        "store_loss": total_store_loss / denom,
        "value_loss": total_value_loss / denom,
        "move_entropy": total_move_ent / denom,
        "store_entropy": total_store_ent / denom,
        "store_bc_loss": total_store_bc / denom,
        # Minibatches whose gradient was non-finite and therefore skipped. A
        # persistent nonzero here is a real problem being survived, not solved
        # -- it belongs in the log where it can be seen, not swallowed.
        "nonfinite_steps": float(n_nonfinite),
    }
    # Emitted only under the polar head, where kappa exists. Not 0.0 (which
    # would plot as a real measurement) and not NaN either: every per-update
    # field is asserted finite by test_smoke_train, and that invariant is how
    # a genuinely broken run gets caught. Absent is the unambiguous option.
    if n_kappa:
        stats["kappa"] = total_kappa / n_kappa
        # The direction head's magnitude: a gauge freedom nothing in the
        # objective pressures. Logged because the softening only BOUNDS what
        # happens when it decays -- it does not stop it decaying, and a run
        # whose heading has gone near-uniform should say so while it happens
        # rather than in a post-mortem.
        stats["dir_norm"] = total_dir / n_kappa
    return stats
