"""Basic control for idea 4: in-context associative recall scored in grid space
(GRID_MLP_NAV_PLAN §7b).

    python -m analysis.idea4.recall_grid --cell gru --H 512 --pairs 6 --steps 100000

The standard associative-recall task (Graves et al. 2014, Ba et al. 2016),
with idea 4's goal format. Each sequence draws `pairs` fresh random keys (+-1
over the 180 panorama slots, full mask, as a whole-panorama view would be)
and for each key a goal: a random cell of a 20 x 20 arena placed at random on
the grid scaffold, given as its 12 grid phases. The network is written each
(key, goal) pair in turn (key, goal phases and a write flag, `rep` steps
each), then queried with every key from a copy of its state. Keys are
identical at write and read, so there are no partial views and no walking:
this tests the recurrent memory itself.

Scoring follows idea 4's retention test. The read-out phases are decoded to
the best-matching cell of the goal's arena; a query is correct within 1.5
cells of the goal (chance about 0.02), and the mean distance in cells is
reported too, both by pair position (oldest written first).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from analysis.idea2.lifetimes import NPOS, SIZE
from analysis.idea4.gpu_views import goal_phases
from analysis.idea4.recall_probe import Memory, _tile
from analysis.idea4.train_memory import angle_cos


def sample(P, B, dev, g=None):
    keys = torch.randint(0, 2, (P, B, 180), device=dev, generator=g).float() * 2 - 1
    x = torch.cat([keys, torch.ones_like(keys)], -1)                      # (P, B, 360)
    off = torch.randint(0, NPOS - SIZE, (P, B, 2), device=dev, generator=g)
    goal = torch.randint(0, SIZE, (P, B, 2), device=dev, generator=g)
    return x, off, goal, goal_phases(off + goal)


def run(m, x, tgt, rep):
    P, B = x.shape[:2]
    h = m.init_state(B, x.device)
    for i in range(P):
        _, h = m.step(x[i:i + 1].expand(rep, B, -1), h, tgt[i:i + 1].expand(rep, B, -1))
    ph, _ = m.step(x.reshape(1, P * B, -1), _tile(h, P))
    return ph.reshape(P, B, -1)


@torch.no_grad()
def decode(ph, off, goal, chunk=4096):
    """Phases (R, 12) -> distance in cells from the best-matching cell of each
    row's arena (offset off, goal cell goal; (R, 2)) to the goal."""
    gx, gy = torch.meshgrid(torch.arange(SIZE, device=ph.device),
                            torch.arange(SIZE, device=ph.device), indexing="ij")
    cells = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)              # (400, 2)
    out = []
    for i in range(0, len(ph), chunk):
        p = ph[i:i + chunk].reshape(-1, 6, 2)
        p = (p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)).reshape(-1, 12)
        ref = goal_phases(off[i:i + chunk, None] + cells[None])            # (r, 400, 12)
        best = torch.einsum("rk,rck->rc", p, ref).argmax(-1)
        out.append((cells[best] - goal[i:i + chunk]).float().norm(dim=-1))
    return torch.cat(out)


@torch.no_grad()
def evaluate(m, P, rep, n, dev, g):
    m.eval()
    x, off, goal, tgt = sample(P, n, dev, g)
    ph = run(m, x, tgt, rep)
    d = decode(ph.reshape(P * n, -1), off.reshape(P * n, 2), goal.reshape(P * n, 2)).reshape(P, n)
    cos = angle_cos(ph, tgt)
    m.train()
    return (d <= 1.5).float().mean(1), d.mean(1), cos.mean()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", default="gru", choices=["gru", "lstm", "migru", "fw"])
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--read", type=int, default=256)
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--rep", type=int, default=1, help="steps per write")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--eval_every", type=int, default=10000)
    p.add_argument("--eval_n", type=int, default=1024)
    p.add_argument("--pairs_min", type=int, default=0,
                   help="if set, each batch draws its pair count uniformly from "
                        "[pairs_min, pairs] (mixed sizes)")
    p.add_argument("--curriculum", action="store_true",
                   help="start at 2 pairs and add one whenever the training cosine at the "
                        "current maximum averages --curriculum_cos over 500 steps; each "
                        "batch draws its size from [2, current maximum]")
    p.add_argument("--curriculum_cos", type=float, default=0.95)
    p.add_argument("--eval_pairs", type=int, nargs="+", default=None,
                   help="pair counts to test at (default: the training size only)")
    p.add_argument("--out", default=None,
                   help="directory for a resumable checkpoint and the eval log")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    m = Memory(a.cell, a.H, 360, read=a.read, enc_kind="none").to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    P = a.pairs
    mode = "curriculum" if a.curriculum else (f"mixed {a.pairs_min}-{P}" if a.pairs_min else "fixed")
    tag = f"{a.cell} H{a.H} pairs {P} {mode} lr {a.lr:g}"
    eval_pairs = a.eval_pairs or [P]
    P_cur, start, log = (2 if a.curriculum else P), 1, []
    ck_path = os.path.join(a.out, "recall.pt") if a.out else None
    if ck_path and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, P_cur, log = ck["step"] + 1, ck["P_cur"], ck["log"]
        torch.manual_seed(a.seed + start)
        print(f"{tag} | resumed at step {ck['step']} (curriculum max {P_cur})", flush=True)
    t0 = time.time()
    window = []
    for step in range(start, a.steps + 1):
        lo = 2 if a.curriculum else (a.pairs_min or P)
        hi = P_cur if a.curriculum else P
        Pb = int(torch.randint(lo, hi + 1, (1,)))
        x, off, goal, tgt = sample(Pb, a.batch, dev)
        cos_b = angle_cos(run(m, x, tgt, a.rep), tgt)
        loss = (1 - cos_b).mean()
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if a.curriculum and Pb == P_cur:
            window.append(float(cos_b.mean().detach()))
            if len(window) >= 500:
                if sum(window[-500:]) / 500 >= a.curriculum_cos and P_cur < P:
                    P_cur += 1
                    print(f"{tag} | step {step}: curriculum max now {P_cur} pairs", flush=True)
                    window = []                  # judge the new size on its own batches
                else:
                    window = window[-500:]
        if step % a.eval_every == 0 or step == a.steps:
            for Pe in eval_pairs:
                g = torch.Generator(device=dev).manual_seed(a.seed + 1 + Pe)
                acc, dist, cos = evaluate(m, Pe, a.rep, a.eval_n, dev, g)
                log.append(dict(step=step, P=Pe, acc=float(acc.mean()), dist=float(dist.mean()),
                                cos=float(cos), P_cur=P_cur))
                print(f"{tag} | step {step} | test {Pe} pairs: acc {float(acc.mean()):.3f} (oldest "
                      f"{float(acc[0]):.2f}, newest {float(acc[-1]):.2f}) | mean distance "
                      f"{float(dist.mean()):.1f} cells | cos {float(cos):.3f} | goals held "
                      f"{float(acc.mean()) * Pe:.1f} | train max {P_cur} ({time.time() - t0:.0f}s)",
                      flush=True)
            if ck_path:
                os.makedirs(a.out, exist_ok=True)
                torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step,
                            "P_cur": P_cur, "log": log, "args": vars(a)}, ck_path)
                with open(os.path.join(a.out, "log.json"), "w") as f:
                    json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
