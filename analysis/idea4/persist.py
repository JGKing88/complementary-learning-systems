"""Does an idea 4 memory keep its goals through the continual protocol's
revisits? (GRID_MLP_NAV_PLAN §7b.14, stage 1)

    python -m analysis.idea4.persist --ckpt <run>/memory.pt [--arenas 6] [--iters 40]

The memory alone, no controller, on the continual protocol's schedule
(analysis/continual/agenthash.py, protocols.run_sequential_protocol): one
state for the whole lifetime, never reset. Arenas are added one per block.
In block i, each of ``--iters`` iterations runs one mini-episode in every
arena j <= i, in order. Arena i's first mini-episode is the search: a visit
walk through the goal (``train_experience.visit_path``) with the store flag
on the first arrival, ending there as the protocol's mini-episode does (the
whole walk with ``--tail``). Every other mini-episode is a revisit: a path
from a random cell straight to the goal (moves toward it, random axis on
ties), ending on the goal cell, flag off. That is the path a working agent
takes; the observed agent needs about 13 steps per revisit.

The memory is read at every step of every revisit, in sequence (the state
carries on), with the cell's own grid phases as input. A read is scored only
more than 1.5 cells from the goal, where repeating the cell's own phases
cannot pass for recall (``train_experience.far_from``): correct when the
decoded cell is within 1.5 cells of that arena's goal. Reported per block
and arena, for the first read of each revisit and for all far reads, plus
the mean c (sigmoid) on revisits.
"""
from __future__ import annotations

import argparse
import json
import time
import types

import numpy as np
import torch

from analysis.idea2.lifetimes import SIZE, EnvPool
from analysis.idea4 import train_experience as te
from analysis.idea4.gpu_views import GpuPool, headings, place


def toward(start, goal, gen):
    """Straight paths from ``start`` to ``goal`` (B, 2): positions (L, B, 2)
    with position 0 the start, one move per step toward the goal along a
    random axis that still differs, then standing on the goal. Also returns
    each row's number of moves (B,)."""
    pos, out = start.clone(), [start.clone()]
    n = (goal - start).abs().sum(-1)
    for _ in range(int(n.max())):
        d = goal - pos
        nz = d != 0
        pick_x = torch.rand(len(pos), device=pos.device, generator=gen) < 0.5
        use_x = nz[:, 0] & (pick_x | ~nz[:, 1])
        use_y = nz[:, 1] & ~use_x
        pos = pos.clone()
        pos[:, 0] += torch.where(use_x, d[:, 0].sign(), torch.zeros_like(d[:, 0]))
        pos[:, 1] += torch.where(use_y, d[:, 1].sign(), torch.zeros_like(d[:, 1]))
        out.append(pos.clone())
    return torch.stack(out), n


@torch.no_grad()
def run(model, gp, a, A, iters, B, seed, device, tail=False, search_steps=None):
    g = torch.Generator(device=device).manual_seed(seed)
    sa = a                                      # the search walk's settings
    if search_steps:
        sa = types.SimpleNamespace(**vars(a))
        sa.visit_steps = int(search_steps)
    env = torch.rand(B, len(gp), device=device, generator=g).argsort(1)[:, :A]
    off, goal, _ = place(B, A, device, g)
    h = model.init_state(B, device)
    rows = []                       # one per (block, iteration, arena) revisit
    steps = 0

    def step(h, j, pos, psi, flag, active):
        x = gp.canvas(env[:, j][None], pos[None, :, 0], pos[None, :, 1], psi[None])
        ex = te.extra_of(off[:, j][None], pos[None], flag[None])
        ph, c, h2 = model(x, h, extra=ex)
        keep = active[None, :, None]
        return ph[0], c[0], torch.where(keep, h2, h)

    for i in range(A):
        for it in range(iters):
            for j in range(i + 1):
                if j == i and it == 0:                       # the search, with the store
                    path, psi, first, found = te.visit_path(goal[:, j], sa, device, g)
                    act = torch.ones_like(first) if tail else (~found | first)
                    cpre = torch.zeros(B, device=device)
                    npre = torch.zeros(B, device=device)
                    for t in range(len(path)):
                        _, c, h = step(h, j, path[t], psi[t], first[t], act[t])
                        pre = act[t] & ~found[t]             # before the arrival: goal unknown
                        cpre += torch.sigmoid(c) * pre
                        npre += pre.float()
                    steps += int(act.sum(0).float().mean())
                    rows.append(dict(kind="search", block=i, iter=it, arena=j, step=steps,
                                     c_pre=float((cpre.sum() / npre.sum().clamp(min=1)).cpu()),
                                     len_pre=float(npre.mean().cpu())))
                    continue
                start = torch.randint(0, SIZE, (B, 2),
                                      device=device, generator=g)
                path, n = toward(start, goal[:, j], g)
                psi = headings(path, device, g)
                ok = torch.zeros(B, device=device)
                cnt = torch.zeros(B, device=device)
                nok = torch.zeros(B, device=device)
                ncnt = torch.zeros(B, device=device)
                first_ok = torch.full((B,), float("nan"), device=device)
                csum = torch.zeros(B, device=device)
                for t in range(len(path)):
                    active = t <= n
                    ph, c, h = step(h, j, path[t], psi[t],
                                    torch.zeros(B, device=device), active)
                    far = te.far_from(path[t], goal[:, j]) & active
                    hit = te._hits(ph[:, None], off[:, j], goal[:, j])
                    ok += hit * far
                    cnt += far.float()
                    near = active & ~far
                    nok += hit * near
                    ncnt += near.float()
                    csum += torch.sigmoid(c) * active
                    if t == 0:
                        first_ok = torch.where(far, hit, first_ok)
                steps += int((n + 1).float().mean())
                rows.append(dict(kind="revisit", block=i, iter=it, arena=j, step=steps,
                                 acc=float((ok.sum() / cnt.sum().clamp(min=1)).cpu()),
                                 acc_near=float((nok.sum() / ncnt.sum().clamp(min=1)).cpu()),
                                 acc_first=float(np.nanmean(first_ok.cpu().numpy())),
                                 c=float((csum / (n + 1)).mean().cpu()),
                                 len=float((n + 1).float().mean().cpu())))
    return rows, steps


def summary(rows, A, iters):
    out = {}
    search = [r for r in rows if r.get("kind") == "search"]
    rows = [r for r in rows if r.get("kind", "revisit") == "revisit"]
    for i in range(A):
        blk = [r for r in rows if r["block"] == i]
        per = {j: float(np.mean([r["acc"] for r in blk if r["arena"] == j])) for j in range(i + 1)
               if any(r["arena"] == j for r in blk)}
        last = [r for r in blk if r["iter"] == iters - 1]
        out[i] = dict(acc_by_arena=per,
                      acc_last_iter=float(np.mean([r["acc"] for r in last])) if last else None,
                      acc_first_last_iter=float(np.nanmean([r["acc_first"] for r in last]))
                      if last else None,
                      c_mean=float(np.mean([r["c"] for r in blk])) if blk else None,
                      acc_near=float(np.mean([r["acc_near"] for r in blk])) if blk else None,
                      c_search=float(np.mean([r["c_pre"] for r in search if r["block"] == i]))
                      if any(r["block"] == i for r in search) else None,
                      step_end=blk[-1]["step"] if blk else None)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--arenas", type=int, default=6)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--lifetimes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tail", action="store_true",
                   help="run the whole search walk, not only up to the first arrival")
    p.add_argument("--search_steps", type=int, default=None,
                   help="length of the search walk (default: the checkpoint's visit_steps); "
                        "the agent searches for up to 200 steps")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    c = p.parse_args()
    device = torch.device(c.device if torch.cuda.is_available() else "cpu")
    res = []
    gp = None
    for path in c.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = types.SimpleNamespace(**ck["args"])
        a.chrono = 0
        a.enc_kind = getattr(a, "enc_kind", "relu")
        if gp is None:
            gp = GpuPool(EnvPool(a.n_eval_envs, seed=a.seed + 2), device)   # held-out arenas
        model = te.build(a, device)
        model.load_state_dict(ck["model"])
        model.eval()
        t0 = time.time()
        rows, steps = run(model, gp, a, c.arenas, c.iters, c.lifetimes, c.seed, device, c.tail,
                          c.search_steps)
        s = summary(rows, c.arenas, c.iters)
        print(f"\n{path}  step {ck['step']}  {c.arenas} arenas x {c.iters} iterations, "
              f"{steps} memory steps per lifetime ({time.time() - t0:.0f}s)")
        for i, r in s.items():
            arenas = " ".join(f"{j}:{v:.2f}" for j, v in r["acc_by_arena"].items())
            print(f"block {i} (to step {r['step_end']}): far reads by arena {arenas} | last "
                  f"iteration {r['acc_last_iter']:.2f}, first read {r['acc_first_last_iter']:.2f} "
                  f"| c {r['c_mean']:.2f} | near reads {r['acc_near']:.2f} | c in search "
                  f"before arrival {r['c_search']:.2f}", flush=True)
        res.append(dict(ckpt=path, step=ck["step"], arenas=c.arenas, iters=c.iters,
                        tail=c.tail, memory_steps=steps, summary=s, rows=rows))
    if c.out:
        with open(c.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
