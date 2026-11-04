"""Supervised update for idea 3 (GRID_MLP_NAV_PLAN §7.6).

The collector's teacher (``collect_rollout(teacher_beta=...)``) labels every
step with its move -- search before the goal is found, straight at it after --
and records the vector to the goal. This replays each sequence whole (a joined
visit sequence or a packed lifetime) from a zero state and fits:

- the policy to the teacher's move: negative log-likelihood under the policy's
  own distribution (the polar head's heading, plus its speed at
  ``speed_weight``), on labelled steps (not at the goal, alive);
- the displacement head (``AgentConfig.aux_disp_weight``) to the vector to the
  goal, in tens of cells, on steps where the goal was already known -- the
  dense signal that teaches the GRU to hold the goal and turn it, with the
  current grid code, into a direction;
- the value head to the rollout's own discounted returns, so a later PPO
  fine-tune starts from a critic of this policy.

Backpropagation runs through each whole sequence, so the error at every step
after a find reaches the step where the goal was seen.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..rollout.types import RolloutBatch
from .ppo import compute_gae


def sup_update(agent: nn.Module, rollouts: list[RolloutBatch], cfg,
               optimizer: torch.optim.Optimizer, *, epochs: int = 2,
               n_minibatches: int = 8, speed_weight: float = 0.2,
               value_weight: float = 0.5) -> dict[str, float]:
    for r in rollouts:
        if r.teacher_move_action is None or r.goal_disp is None:
            raise ValueError("sup_update needs the supervised teacher's labels "
                             "(collect_rollout(teacher_beta=...))")
    disp_w = float(getattr(agent, "aux_disp_weight", 0.0))

    def cat(name):
        return torch.cat([getattr(r, name) for r in rollouts], dim=0)

    obs = cat("obs")
    label = cat("teacher_move_action")
    lmask = cat("move_label_mask")
    gdisp = cat("goal_disp") / 10.0
    gknown = cat("goal_known")
    alive = torch.cat([r.alive_mask if r.alive_mask is not None
                       else torch.ones_like(r.rewards) for r in rollouts], dim=0)
    returns = torch.cat([compute_gae(r.rewards, r.values, r.bootstrap_value,
                                     cfg.gamma, 1.0, alive=r.alive_mask)[1]
                         for r in rollouts], dim=0)
    lmask = lmask * alive
    dmask = gknown * alive

    N = obs.shape[0]
    mb = max(1, N // max(1, min(n_minibatches, N)))
    tot = {"act_nll": 0.0, "disp_loss": 0.0, "value_loss": 0.0,
           "dir_err_known_deg": 0.0, "dir_err_search_deg": 0.0, "grad_norm": 0.0}
    n_steps = 0
    for _ in range(int(epochs)):
        perm = torch.randperm(N, device=obs.device)
        for s in range(0, N, mb):
            idx = perm[s:s + mb]
            dist, _, values, _, feats = agent(obs[idx], return_features=True)
            lp = dist.log_prob(label[idx])
            if lp.dim() == 3:
                # Polar head: [heading, speed]; a Cartesian head: [x, y].
                lp = (lp[..., 0] + speed_weight * lp[..., 1]
                      if hasattr(dist, "theta") else lp.sum(-1))
            m = lmask[idx]
            act = -(lp * m).sum() / m.sum().clamp_min(1.0)
            vloss = (((values - returns[idx]) ** 2) * alive[idx]).sum() / alive[idx].sum().clamp_min(1.0)
            loss = act + value_weight * vloss
            if disp_w > 0 and agent.goal_disp_head is not None:
                dm = dmask[idx]
                derr = ((agent.goal_disp_head(feats) - gdisp[idx]) ** 2).sum(-1)
                dloss = (derr * dm).sum() / dm.sum().clamp_min(1.0)
                loss = loss + disp_w * dloss
            else:
                dloss = torch.zeros((), device=obs.device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                # The policy's mean heading against the label, on steps where
                # the goal was known (returning) and where it was not (search).
                theta = getattr(dist, "theta", None)
                if theta is not None:
                    lab = torch.atan2(label[idx][..., 1], label[idx][..., 0])
                    err = torch.remainder(theta - lab + math.pi, 2 * math.pi) - math.pi
                    err = err.abs() * (180.0 / math.pi)
                    for key, w in (("dir_err_known_deg", m * gknown[idx]),
                                   ("dir_err_search_deg", m * (1 - gknown[idx]))):
                        tot[key] += float((err * w).sum() / w.sum().clamp_min(1.0))
            tot["act_nll"] += float(act.detach())
            tot["disp_loss"] += float(dloss.detach())
            tot["value_loss"] += float(vloss.detach())
            tot["grad_norm"] += float(gn)
            n_steps += 1
    out = {k: v / max(n_steps, 1) for k, v in tot.items()}
    out["sup_steps"] = float(n_steps)
    return out


__all__ = ["sup_update"]
