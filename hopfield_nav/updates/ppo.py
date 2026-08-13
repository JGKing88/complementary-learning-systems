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


def advantage_scale_by_group(
    rollouts: list[RolloutBatch],
    groups: list[str],
    gamma: float,
    gae_lambda: float,
) -> dict[str, float]:
    """Per-group advantage scale, *before* the pooled normalization.

    Advantages are normalized once over the whole pooled buffer, so when a
    single update mixes regimes the divisor is common to all of them. Each
    group's share of the resulting gradient is therefore set by its own raw
    advantage magnitude relative to that common divisor -- not by how many
    rollouts it contributed.

    That is easy to get badly wrong without noticing. An exploit rollout's
    reward is dominated by a `goal_reward`-sized spike; an explore rollout's by
    a novelty bonus a factor of ten or more smaller. Pool them and the shared
    std is set almost entirely by the exploit rows, which divides the explore
    rows' gradient by the ratio -- a silent, schedule-dependent reweighting of
    two objectives that the flag list makes look independent.

    Returns `{group: raw std}` plus `pooled` and, per group, `share`: the
    factor its gradient is scaled by relative to what it would have received
    had that group been normalized on its own. `share` of 1.0 means the group
    is unaffected by the mixing; 0.2 means its objective is being run at a
    fifth strength.
    """
    if len(groups) != len(rollouts):
        raise ValueError(
            f"got {len(groups)} group labels for {len(rollouts)} rollouts")
    per_rollout = [
        compute_gae(r.rewards, r.values, r.bootstrap_value, gamma, gae_lambda,
                    alive=r.alive_mask)[0]
        for r in rollouts
    ]
    pooled = torch.cat(per_rollout, dim=0)
    pooled_std = float(pooled.std().clamp_min(1e-8).item())
    out: dict[str, float] = {"pooled": pooled_std}
    for name in sorted(set(groups)):
        sel = [a for a, g in zip(per_rollout, groups) if g == name]
        if not sel:
            continue
        std = float(torch.cat(sel, dim=0).std().clamp_min(1e-8).item())
        out[f"{name}_std"] = std
        out[f"{name}_share"] = std / pooled_std
    return out


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
    total_store_bc = 0.0
    n_steps = 0

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
            if isinstance(move_dist, Normal):  # continuous: sum over action dims
                new_move_lp = new_move_lp.sum(-1)
            ratio_move = torch.exp(new_move_lp - mb_old_move_lp)
            surr1 = ratio_move * mb_adv
            surr2 = torch.clamp(ratio_move, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
            # Mask ε / auto-nav steps out of move_loss — those actions did
            # not come from the policy sample, so including them in the PPO
            # surrogate explodes the importance ratio under narrow std.
            pol_mask_sum = mb_pol_mask.sum().clamp_min(1.0)
            move_loss = (-torch.min(surr1, surr2) * mb_pol_mask).sum() / pol_mask_sum

            # Store policy loss — masked by explore_mask: during exploit the
            # store action is inert (rollout ignores it, no store_cost/
            # store_bonus apply, Hopfield is frozen), so those timesteps carry
            # zero causal signal for the store head. Including them just pumps
            # variance into the store logits and the shared RNN trunk.
            new_store_lp = store_dist.log_prob(mb_store_act)
            ratio_store = torch.exp(new_store_lp - mb_old_store_lp)
            surr1_s = ratio_store * mb_adv
            surr2_s = torch.clamp(ratio_store, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
            mask_sum = mb_mask.sum().clamp_min(1.0)
            store_loss = (-torch.min(surr1_s, surr2_s) * mb_mask).sum() / mask_sum

            # Value loss. Steps after a row's episode ended are not states the
            # value head should be fit to -- the agent was frozen there and the
            # return is defined to be zero, so regressing onto them teaches it
            # that finishing is worth nothing.
            sq_err = (mb_ret - new_values) ** 2
            move_entropy = move_dist.entropy()
            if move_entropy.dim() > 2:
                move_entropy = move_entropy.sum(-1)
            if mb_alive is None:
                # No row ended early: the historical reduction, unchanged.
                value_loss = sq_err.mean()
                move_ent = move_entropy.mean()
            else:
                alive_sum = mb_alive.sum().clamp_min(1.0)
                value_loss = (sq_err * mb_alive).sum() / alive_sum
                move_ent = (move_entropy * mb_alive).sum() / alive_sum
            store_ent = (store_dist.entropy() * mb_mask).sum() / mask_sum

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
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_move_loss += move_loss.item()
            total_store_loss += store_loss.item()
            total_value_loss += value_loss.item()
            total_move_ent += move_ent.item()
            total_store_ent += store_ent.item()
            total_store_bc += store_bc_loss.item()
            n_steps += 1

    denom = max(n_steps, 1)
    return {
        "move_loss": total_move_loss / denom,
        "store_loss": total_store_loss / denom,
        "value_loss": total_value_loss / denom,
        "move_entropy": total_move_ent / denom,
        "store_entropy": total_store_ent / denom,
        "store_bc_loss": total_store_bc / denom,
    }
