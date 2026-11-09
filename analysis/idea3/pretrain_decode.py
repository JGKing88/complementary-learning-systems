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
from hopfield_nav.policy import channels
from hopfield_nav.training.synth_decode import Synth


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
    p.add_argument("--iters_choices", type=int, nargs="*", default=None,
                   help="passes per block to draw from (uniformly), e.g. 2 5 10 20; "
                        "default 1..max_iters")
    p.add_argument("--tbptt", type=int, default=0,
                   help="truncated-BPTT window in steps (0 = whole sequence)")
    p.add_argument("--balance_positions", action="store_true",
                   help="lifetime mode: weight the displacement loss so each arena "
                        "position carries the same total weight. In protocol order the "
                        "first arena has 6x the trials of the sixth, and recall falls "
                        "with position (GRID_MLP_NAV_PLAN s7.6).")
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
    import random
    for step in range(1, a.steps + 1):
        if a.mode == "lifetime":
            E = int(torch.randint(2, a.max_envs + 1, ()).item())
            K = (random.choice(a.iters_choices) if a.iters_choices
                 else int(torch.randint(1, a.max_iters + 1, ()).item()))
            obs, tgt, msk, kn, _, alive, arena = synth.lifetime(
                a.batch, E, K, return_arena=True)
            if a.balance_positions:
                # Each position's known steps get total weight sum/E, so the
                # mean weight per known step stays 1.
                cnt = torch.stack([(msk * (arena == j)).sum() for j in range(E)])
                w = (cnt.sum() / E) / cnt.clamp_min(1.0)
                msk = msk * w[arena.long().clamp(0, E - 1)]
        else:
            obs, tgt, msk, _ = synth.batch(a.batch, a.seq)
            kn = alive = None
        # Truncated BPTT for long lifetimes (--tbptt W): windows of W steps,
        # the state carried detached from one to the next, one optimizer step
        # per window. 0 = the whole sequence at once.
        L = obs.shape[1]
        W = a.tbptt if a.tbptt > 0 else L
        h = None
        for s0 in range(0, L, W):
            sl = slice(s0, s0 + W)
            if alive is not None and alive[:, sl].sum() == 0:
                break
            _, _, _, h_next, feats = agent(obs[:, sl], h, return_features=True)
            err = ((agent.goal_disp_head(feats) - tgt[:, sl]) ** 2).sum(-1)
            loss = (err * msk[:, sl]).sum() / msk[:, sl].sum().clamp_min(1.0)
            if a.mode == "lifetime":
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    known_head(feats).squeeze(-1), kn[:, sl], reduction="none")
                loss = loss + a.known_weight * (bce * alive[:, sl]).sum() / \
                    alive[:, sl].sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()
            h = h_next.detach()
        if (step % a.eval_every == 0 or step == a.steps) and a.mode == "lifetime":
            ev = {f"E{E}k{k}": evaluate_lifetime(agent, known_head, synth, 64, E, k)
                  for E, k in ((3, 2), (6, 2), (6, 10))}
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
