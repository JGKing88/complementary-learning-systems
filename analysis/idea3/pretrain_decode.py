"""Stage 1 of GRID_MLP_NAV_PLAN §7.6 in its pure form: teach the GRU the goal.

    python -m analysis.idea3.pretrain_decode --like <navigate_uN.pt> --out <pt>

Synthetic sequences, built on the GPU from table lookups, with the inputs the
collector gives idea 3 (current and previous reward, previous action and
displacement, the grid code at the agent's cell; the view is zero here, and
its input weights are zeroed in the saved model so it starts out ignored). An
agent walks in a 20x20 arena placed anywhere on the scaffold toward a goal
cell with heavy heading noise. Touching the goal pays the goal reward, the
move is discarded and the agent is teleported to a random cell, as in the
task. Every ``visit`` steps a new visit starts: random cell, previous reward,
action and displacement zeroed, state carried. From the first touch on, the
displacement head must output the vector to the goal (in tens of cells) at
every step. Previous displacement is zero right after a teleport, as in the
collector, so the step after a teleport can only be answered from the grid
codes; it is scored on its own (``first_step``).

The model is the one ``--like``'s config builds (same layout, cell, size), so
the saved checkpoint forks straight into the supervised task stages
(``LOAD_CKPT``). Many more examples per second than driving the collector,
and goals and places are fresh in every sequence.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint, load_agent
from hopfield_nav.memory.grid_code import SmoothedGridCode
from hopfield_nav.policy import channels


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

    def lifetime(self, B, n_envs, iters, cap=50, noise_lo=30.0, noise_hi=120.0):
        """A short continual protocol per row, packed (stage 2, §7.6).

        ``n_envs`` arenas at random scaffold places, each with a goal; trials
        in the protocol's order (block i: ``iters`` passes over arenas 0..i).
        A trial starts at a random cell with no previous step and ends at the
        goal or after ``cap`` steps; a finished row waits. In an arena whose
        goal this row has not reached yet the walker first wanders (a random
        number of steps), then heads for the goal, so its moves do not give an
        unseen goal away. Returns packed (obs, disp target, disp mask, known,
        trial-start-in-a-known-arena mask, alive), all (B, L, ...).
        """
        from hopfield_nav.rollout.lifetime import protocol_order
        d, S = self.device, self.size
        order = protocol_order(n_envs, iters)
        off = torch.randint(0, self.npos - S, (B, n_envs, 2), device=d)
        goal = torch.randint(0, S, (B, n_envs, 2), device=d).float()
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
            wander = torch.where(known[:, j], torch.zeros(B, device=d),
                                 torch.randint(0, 30, (B,), device=d).float())
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
            chunks.append((obs, tgt, msk, kn, fst, alive))
        cat = [torch.cat([c[i] for c in chunks], 1) for i in range(6)]
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
        return tuple(out)


def evaluate_lifetime(agent, known_head, synth, B, n_envs, iters):
    """Packed synthetic lifetimes: displacement on known steps, and at the
    first step of a trial in an arena whose goal was found before (recall the
    arena's goal and decode it from a fresh start); whether the current
    arena's goal is known, from the state."""
    agent.eval()
    with torch.no_grad():
        obs, tgt, msk, kn, fst, alive = synth.lifetime(B, n_envs, iters)
        _, _, _, _, feats = agent(obs, return_features=True)
        pred = agent.goal_disp_head(feats)
        kp = known_head(feats).squeeze(-1) > 0
    agent.train()
    out = {}
    for name, m in (("known", msk > 0), ("trial_start", fst > 0)):
        y, p = tgt[m] * 10, pred[m] * 10
        r2 = 1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
        far = y.norm(dim=-1) >= 2
        ang = torch.atan2(p[far, 1], p[far, 0]) - torch.atan2(y[far, 1], y[far, 0])
        err = (torch.remainder(ang + math.pi, 2 * math.pi) - math.pi).abs() * 180 / math.pi
        out[name] = {"r2": float(r2), "dir_err_deg": float(err.mean()),
                     "dir_err_median_deg": float(err.median()), "n": int(m.sum())}
    a = alive > 0
    out["known_acc"] = float((kp[a] == (kn[a] > 0)).float().mean())
    return out


def evaluate(agent, synth, B, T):
    agent.eval()
    with torch.no_grad():
        obs, tgt, msk, fst = synth.batch(B, T)
        _, _, _, _, feats = agent(obs, return_features=True)
        pred = agent.goal_disp_head(feats)
    agent.train()
    out = {}
    for name, m in (("all", msk > 0), ("first_step", fst > 0)):
        y, p = tgt[m] * 10, pred[m] * 10
        r2 = 1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
        far = y.norm(dim=-1) >= 2
        ang = torch.atan2(p[far, 1], p[far, 0]) - torch.atan2(y[far, 1], y[far, 0])
        err = (torch.remainder(ang + math.pi, 2 * math.pi) - math.pi).abs() * 180 / math.pi
        out[name] = {"r2": float(r2), "dir_err_deg": float(err.mean()),
                     "dir_err_median_deg": float(err.median()), "n": int(m.sum())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--like", required=True, help="checkpoint whose config to build")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seq", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["single", "lifetime"], default="single",
                   help="single: one arena per sequence (stage 1); lifetime: "
                        "short continual protocols over several arenas (stage 2)")
    p.add_argument("--init", default=None,
                   help="checkpoint to start from (e.g. a stage-1 output)")
    p.add_argument("--max_envs", type=int, default=6)
    p.add_argument("--max_iters", type=int, default=2)
    p.add_argument("--known_weight", type=float, default=0.5)
    a = p.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    ck = torch.load(a.like, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    init_sd = (torch.load(a.init, map_location="cpu", weights_only=False)["agent_state_dict"]
               if a.init else None)
    agent = load_agent(cfg, init_sd, 1024, device, eval_mode=False)
    if agent.goal_disp_head is None:
        raise SystemExit("the config has no displacement head (aux_disp_weight 0)")
    synth = Synth(cfg, device)
    # Whether the current arena's goal is known: a pretraining-only head (not
    # saved); it shapes the state the policy will read.
    known_head = torch.nn.Linear(cfg.agent.hidden_size, 1).to(device)
    opt = torch.optim.Adam(list(agent.parameters()) + list(known_head.parameters()), lr=a.lr)
    t0 = time.time()
    hist = []
    for step in range(1, a.steps + 1):
        if a.mode == "lifetime":
            E = int(torch.randint(2, a.max_envs + 1, ()).item())
            K = int(torch.randint(1, a.max_iters + 1, ()).item())
            obs, tgt, msk, kn, _, alive = synth.lifetime(a.batch, E, K)
        else:
            obs, tgt, msk, _ = synth.batch(a.batch, a.seq)
        _, _, _, _, feats = agent(obs, return_features=True)
        err = ((agent.goal_disp_head(feats) - tgt) ** 2).sum(-1)
        loss = (err * msk).sum() / msk.sum().clamp_min(1.0)
        if a.mode == "lifetime":
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                known_head(feats).squeeze(-1), kn, reduction="none")
            loss = loss + a.known_weight * (bce * alive).sum() / alive.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        if (step % a.eval_every == 0 or step == a.steps) and a.mode == "lifetime":
            ev = {f"E{E}": evaluate_lifetime(agent, known_head, synth, 128, E, 2)
                  for E in (3, 6)}
            hist.append({"step": step, "loss": loss.item(), **ev})
            print(f"step {step} loss {loss.item():.4f} | " + " | ".join(
                f"{k}: known r2 {v['known']['r2']:.3f} dir {v['known']['dir_err_deg']:.1f}, "
                f"trial start dir {v['trial_start']['dir_err_deg']:.1f} "
                f"(median {v['trial_start']['dir_err_median_deg']:.1f}), "
                f"known acc {v['known_acc']:.3f}" for k, v in ev.items())
                + f" | {time.time() - t0:.0f}s", flush=True)
        elif step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(agent, synth, 256, a.seq)
            hist.append({"step": step, "loss": loss.item(), **ev})
            print(f"step {step} loss {loss.item():.4f} | all r2 {ev['all']['r2']:.3f} "
                  f"dir {ev['all']['dir_err_deg']:.1f} deg | first step after teleport "
                  f"r2 {ev['first_step']['r2']:.3f} dir {ev['first_step']['dir_err_deg']:.1f} "
                  f"(median {ev['first_step']['dir_err_median_deg']:.1f}) deg | "
                  f"{time.time() - t0:.0f}s", flush=True)
    # Views were zero here: zero their input weights so the forked model starts
    # out ignoring them rather than reading them through random weights.
    specs = channels.channel_specs(cfg.agent, 1024, cfg.env.observation_size)
    o = 0
    for s in specs:
        if s.name == "sensory":
            with torch.no_grad():
                agent.rnn.weight_ih_l0[:, o:o + s.width] = 0.0
        o += s.width
    out = dict(ck)
    out["agent_state_dict"] = agent.state_dict()
    out["update"] = 0
    out["pretrain_decode"] = {"args": vars(a), "history": hist}
    torch.save(out, a.out)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
