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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    m = Memory(a.cell, a.H, 360, read=a.read, enc_kind="none").to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    P = a.pairs
    tag = f"{a.cell} H{a.H} pairs {P} rep {a.rep}"
    t0 = time.time()
    for step in range(1, a.steps + 1):
        x, off, goal, tgt = sample(P, a.batch, dev)
        loss = (1 - angle_cos(run(m, x, tgt, a.rep), tgt)).mean()
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            g = torch.Generator(device=dev).manual_seed(a.seed + 1)
            acc, dist, cos = evaluate(m, P, a.rep, a.eval_n, dev, g)
            print(f"{tag} | step {step}: acc {float(acc.mean()):.3f} (oldest {float(acc[0]):.2f}, "
                  f"newest {float(acc[-1]):.2f}) | mean distance {float(dist.mean()):.1f} cells "
                  f"(oldest {float(dist[0]):.1f}, newest {float(dist[-1]):.1f}) | cos "
                  f"{float(cos):.3f} | goals held {float(acc.mean()) * P:.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
