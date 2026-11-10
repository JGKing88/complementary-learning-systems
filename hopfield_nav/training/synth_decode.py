"""Synthetic sequences for idea 3's supervised training (GRID_MLP_NAV_PLAN §7.6).

Built on the GPU from table lookups, with the input layout the collector gives
idea 3 (current and previous reward, previous action and displacement, the grid
code at the agent's cell; views zero). ``batch`` is stage 1 (one arena, noisy
goal pursuit, a teleport at each touch, new visits every ``visit`` steps);
``lifetime`` is stage 2 (short continual protocols over several arenas,
packed). Used by ``analysis/idea3/pretrain_decode.py`` to pretrain, and by the
trainer to rehearse during the real-task stages (``--synth_rehearsal_steps``).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..memory.grid_code import SmoothedGridCode
from ..policy import channels


class Synth:
    """Batched synthetic lifetimes on the GPU."""

    def __init__(self, cfg, device, size=20, visit=200):
        lam = [int(l) for l in cfg.vectorhash.lambdas]
        self.npos = int(cfg.vectorhash.Npos or np.prod(lam))
        g = SmoothedGridCode(lam, float(cfg.fwhm_ratio), self.npos)
        self.lam = lam
        self.tables = [torch.from_numpy(t).float().to(device) for t in g._tables]
        self.size, self.visit, self.device = size, visit, device
        self.goal_reward = float(cfg.env.goal_reward)
        self.time_pen = float(cfg.env.time_penalty)
        self.radius = float(cfg.env.goal_radius)
        self.specs = channels.channel_specs(cfg.agent, 1, cfg.env.observation_size)

    def code(self, gx, gy):
        return torch.cat([t[gx % l, gy % l] for l, t in zip(self.lam, self.tables)], -1)

    def batch(self, B, T, noise_lo=30.0, noise_hi=120.0):
        d, S = self.device, self.size
        off = torch.randint(0, self.npos - S, (B, 2), device=d)
        goal = torch.randint(0, S, (B, 2), device=d).float()
        pos = torch.rand(B, 2, device=d) * (S - 1)
        sigma = (torch.rand(B, device=d) * (noise_hi - noise_lo) + noise_lo) * math.pi / 180
        prev_r = torch.zeros(B, device=d)
        prev_a = torch.zeros(B, 2, device=d)
        prev_d = torch.zeros(B, 2, device=d)
        known = torch.zeros(B, dtype=torch.bool, device=d)
        first = torch.zeros(B, dtype=torch.bool, device=d)
        D = sum(s.width for s in self.specs)
        obs = torch.zeros(B, T, D, device=d)
        tgt = torch.zeros(B, T, 2, device=d)
        msk = torch.zeros(B, T, device=d)
        fst = torch.zeros(B, T, device=d)
        for t in range(T):
            if t > 0 and t % self.visit == 0:
                # A new visit: fresh start, no previous step, state carried.
                pos = torch.rand(B, 2, device=d) * (S - 1)
                prev_r.zero_(); prev_a.zero_(); prev_d.zero_()
                first = known.clone()
            cell = torch.round(pos).long().clamp(0, S - 1)
            at = (pos - goal).norm(dim=-1) <= self.radius
            cur_r = torch.where(at, torch.full_like(prev_r, self.goal_reward),
                                torch.full_like(prev_r, -self.time_pen))
            vals = {"current_reward": cur_r[:, None], "prev_reward": prev_r[:, None],
                    "prev_action": prev_a, "prev_displacement": prev_d,
                    "grid_code": self.code(off[:, 0] + cell[:, 0], off[:, 1] + cell[:, 1])}
            o = 0
            for s in self.specs:
                if s.name in vals:
                    obs[:, t, o:o + s.width] = vals[s.name]
                o += s.width
            tgt[:, t] = (goal - pos) / 10.0
            msk[:, t] = known.float()
            fst[:, t] = (first & known).float()
            first = torch.zeros_like(first)
            # The move: toward the goal with heavy heading noise.
            to = goal - pos
            ang = torch.atan2(to[:, 1], to[:, 0]) + torch.randn(B, device=d) * sigma
            a = 0.9 * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
            new = (pos + a).clamp(0, S - 1)
            moved = torch.where(at[:, None], torch.zeros_like(a), new - pos)
            # At the goal the move is discarded and the agent is teleported.
            tele = torch.rand(B, 2, device=d) * (S - 1)
            pos = torch.where(at[:, None], tele, new)
            first = at & known | at          # the step after a teleport
            known = known | at
            prev_r, prev_a, prev_d = cur_r, a, moved
        return obs, tgt, msk, fst

    def lifetime(self, B, n_envs, iters, cap=50, noise_lo=30.0, noise_hi=120.0,
                 offsets=None, goals=None, return_arena=False, prefind="pursue"):
        """A short continual protocol per row, packed (stage 2, §7.6).

        ``n_envs`` arenas at random scaffold places, each with a goal; trials
        in the protocol's order (block i: ``iters`` passes over arenas 0..i).
        A trial starts at a random cell with no previous step and ends at the
        goal or after ``cap`` steps; a finished row waits. In an arena whose
        goal this row has not reached yet the walker first wanders (a random
        number of steps), then heads for the goal. That pursuit does give an
        unseen goal away: the moves and the grid-code path point at it before
        the find (GRID_MLP_NAV_PLAN s7.6). ``prefind="random"`` keeps the walker
        on its random walk until it touches the goal by chance, so only the
        find itself can tell the state where the goal is. Returns packed (obs,
        disp target, disp mask, known, trial-start-in-a-known-arena mask,
        alive), all (B, L, ...).
        """
        if prefind not in ("pursue", "random"):
            raise ValueError(f"prefind must be 'pursue' or 'random', got {prefind!r}")
        from ..rollout.lifetime import protocol_order
        d, S = self.device, self.size
        order = protocol_order(n_envs, iters)
        # Fixed arenas (``offsets``, ``goals``: (n_envs, 2)) for diagnostics,
        # e.g. a run's held-out arenas; otherwise fresh draws per row.
        off = (torch.as_tensor(offsets, device=d).long()[None].expand(B, -1, -1)
               if offsets is not None else
               torch.randint(0, self.npos - S, (B, n_envs, 2), device=d))
        goal = (torch.as_tensor(goals, device=d).float()[None].expand(B, -1, -1)
                if goals is not None else
                torch.randint(0, S, (B, n_envs, 2), device=d).float())
        known = torch.zeros(B, n_envs, dtype=torch.bool, device=d)
        sigma = (torch.rand(B, device=d) * (noise_hi - noise_lo) + noise_lo) * math.pi / 180
        D = sum(s.width for s in self.specs)
        rows = torch.arange(B, device=d)
        chunks = []
        for j in order:
            pos = torch.rand(B, 2, device=d) * (S - 1)
            prev_r = torch.zeros(B, device=d)
            prev_a = torch.zeros(B, 2, device=d)
            prev_d = torch.zeros(B, 2, device=d)
            heading = torch.rand(B, device=d) * 2 * math.pi
            unseen = (torch.full((B,), float(cap), device=d) if prefind == "random"
                      else torch.randint(0, 30, (B,), device=d).float())
            wander = torch.where(known[:, j], torch.zeros(B, device=d), unseen)
            done = torch.zeros(B, dtype=torch.bool, device=d)
            g = goal[:, j]
            o_j = off[:, j]
            k_j = known[:, j].clone()
            obs = torch.zeros(B, cap, D, device=d)
            tgt = torch.zeros(B, cap, 2, device=d)
            msk = torch.zeros(B, cap, device=d)
            kn = torch.zeros(B, cap, device=d)
            fst = torch.zeros(B, cap, device=d)
            alive = torch.zeros(B, cap, device=d)
            for t in range(cap):
                if bool(done.all()):
                    break
                cell = torch.round(pos).long().clamp(0, S - 1)
                at = (pos - g).norm(dim=-1) <= self.radius
                cur_r = torch.where(at, torch.full_like(prev_r, self.goal_reward),
                                    torch.full_like(prev_r, -self.time_pen))
                vals = {"current_reward": cur_r[:, None], "prev_reward": prev_r[:, None],
                        "prev_action": prev_a, "prev_displacement": prev_d,
                        "grid_code": self.code(o_j[:, 0] + cell[:, 0], o_j[:, 1] + cell[:, 1])}
                oo = 0
                for s in self.specs:
                    if s.name in vals:
                        obs[:, t, oo:oo + s.width] = vals[s.name]
                    oo += s.width
                live = (~done).float()
                tgt[:, t] = (g - pos) / 10.0
                msk[:, t] = k_j.float() * live
                kn[:, t] = k_j.float()
                fst[:, t] = (k_j & (t == 0)).float() * live
                alive[:, t] = live
                to = g - pos
                pursue = torch.atan2(to[:, 1], to[:, 0]) + torch.randn(B, device=d) * sigma
                heading = heading + torch.randn(B, device=d) * 0.5
                ang = torch.where(wander > t, heading, pursue)
                a = 0.9 * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
                new = (pos + a).clamp(0, S - 1)
                moved = torch.where(at[:, None], torch.zeros_like(a), new - pos)
                pos = torch.where(done[:, None] | at[:, None], pos, new)
                # The at-goal step is the trial's last real one.
                k_j = k_j | (at & ~done)
                done = done | at
                prev_r, prev_a, prev_d = cur_r, a, moved
            known[:, j] = k_j
            chunks.append((obs, tgt, msk, kn, fst, alive,
                           torch.full((B, cap), float(j), device=d)))
        cat = [torch.cat([c[i] for c in chunks], 1) for i in range(7)]
        alive = cat[5]
        Ltot = alive.shape[1]
        L = int(alive.sum(1).max().item())
        key = (1.0 - alive) * Ltot + torch.arange(Ltot, device=d).float()
        idx = torch.argsort(key, dim=1)[:, :L]
        keep = torch.gather(alive, 1, idx)
        out = []
        for x in cat:
            ix = idx.view(B, L, *([1] * (x.dim() - 2))).expand(B, L, *x.shape[2:])
            out.append(torch.gather(x, 1, ix) * keep.view(B, L, *([1] * (x.dim() - 2))))
        # The 7th, the arena index of each step, only on request.
        return tuple(out) if return_arena else tuple(out[:6])


def rehearse(agent, synth, optimizer, *, steps: int, batch: int,
             max_envs: int = 6, weight: float = 1.0, max_grad_norm: float = 1.0,
             iters_choices=(2,), tbptt: int = 0, balance: bool = False) -> float:
    """``steps`` optimizer steps on synthetic lifetimes (displacement to the
    current arena's goal when known), so the real-task stages do not wear down
    what pretraining taught. Returns the mean loss.

    Each lifetime has ``iters`` passes per block, drawn from ``iters_choices``.
    With ``tbptt`` > 0 the lifetime runs in windows of that many steps, the
    state carried detached between them and each window's gradient added to
    the last, so a long lifetime fits in memory and still costs one optimizer
    step.

    With ``balance`` each arena position carries the same total weight. In
    protocol order the first arena has many times the trials of the last, and
    recall falls with the position a goal arrives in (GRID_MLP_NAV_PLAN §7.6)."""
    choices = [int(k) for k in iters_choices] or [2]
    tot = 0.0
    for _ in range(int(steps)):
        E = int(torch.randint(1, max_envs + 1, ()).item())
        K = choices[int(torch.randint(len(choices), ()).item())]
        obs, tgt, msk, _, _, alive, arena = synth.lifetime(
            batch, E, K, return_arena=True)
        if balance:
            cnt = torch.stack([(msk * (arena == j)).sum() for j in range(E)])
            w = (cnt.sum() / E) / cnt.clamp_min(1.0)
            msk = msk * w[arena.long().clamp(0, E - 1)]
        L = obs.shape[1]
        W = int(tbptt) if int(tbptt) > 0 else L
        denom = msk.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        h, loss_sum = None, 0.0
        for s0 in range(0, L, W):
            sl = slice(s0, s0 + W)
            if alive[:, sl].sum() == 0:
                break
            _, _, _, h_next, feats = agent(obs[:, sl], h, return_features=True)
            err = ((agent.goal_disp_head(feats) - tgt[:, sl]) ** 2).sum(-1)
            loss = weight * (err * msk[:, sl]).sum() / denom
            loss.backward()
            loss_sum += float(loss.detach())
            h = h_next.detach()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()
        tot += loss_sum
    return tot / max(int(steps), 1)


__all__ = ["Synth", "rehearse"]
