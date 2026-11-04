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
    a = p.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    ck = torch.load(a.like, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    agent = load_agent(cfg, None, 1024, device, eval_mode=False)
    if agent.goal_disp_head is None:
        raise SystemExit("the config has no displacement head (aux_disp_weight 0)")
    synth = Synth(cfg, device)
    opt = torch.optim.Adam(agent.parameters(), lr=a.lr)
    t0 = time.time()
    hist = []
    for step in range(1, a.steps + 1):
        obs, tgt, msk, _ = synth.batch(a.batch, a.seq)
        _, _, _, _, feats = agent(obs, return_features=True)
        err = ((agent.goal_disp_head(feats) - tgt) ** 2).sum(-1)
        loss = (err * msk).sum() / msk.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
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
