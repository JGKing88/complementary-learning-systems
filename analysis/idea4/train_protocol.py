"""Idea 4's memory trained on the continual protocol's own schedule
(GRID_MLP_NAV_PLAN §7b.14, stage 2).

    python -m analysis.idea4.train_protocol --out <dir> --hidden 2048 --cell lstm \
        [--init_from <train_experience run>/memory.pt] [--steps 50000]

The memories trained by ``train_experience.py`` forget within about 1,000
steps of reading (§7b.14 stage 1): they never saw more than about 40 steps
of reading between writes. Here a lifetime follows the protocol that
``persist.py`` tests. Arenas are added one per block; in block i each of
``--iters`` iterations runs one mini-episode in every arena j <= i. Arena
i's first mini-episode is the search, a visit walk with the store flag on
the first arrival, ending there. Every other mini-episode is a revisit, a
straight path from a random cell to the goal, flag off.

Loss: on every revisit step more than 1.5 cells from the goal, the cosine
error of the decoded goal phases; on every step, c (1 on revisits and from
the arrival on, 0 during a search before the arrival). Trained as a stream:
the loss is summed over ``--window`` steps, the optimizer steps, and the
state carries on detached, so one lifetime of thousands of steps gives many
updates. ``--steps`` counts updates.

Curriculum on the number of arenas (``--curriculum``): lifetimes have
A_cur arenas, from 2; A_cur rises when the cosine over the far revisit
reads of the last block, averaged over the last ``--curriculum_window``
lifetimes, reaches ``--curriculum_cos``.

Test every ``--eval_every`` updates: ``persist.run`` on held-out arenas at
the protocol's 6 arenas and 40 iterations.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from analysis.idea2.lifetimes import SIZE, EnvPool
from analysis.idea4 import motion as mo
from analysis.idea4 import train_experience as te
from analysis.idea4.gpu_views import GpuPool, headings, place
from analysis.idea4.persist import run as persist_run, summary as persist_summary, toward
from analysis.idea4.train_memory import angle_cos


def schedule(a, goal, A, iters, B, device, gen):
    """The protocol's env steps for B lifetimes of A arenas, one dict per
    step: arena index j, cell and heading (B, ...), store flag, active rows,
    phase-loss weight pw, c target ct and c weight cw, and whether the step
    is in the last block."""
    zeros = torch.zeros(B, device=device)
    ones = torch.ones(B, device=device)
    for i in range(A):
        last = i == A - 1
        for it in range(iters):
            for j in range(i + 1):
                if j == i and it == 0:                       # the search, storing at the arrival
                    if a.motion == "continuous":             # agent-like (motion.py)
                        path, psi, ns = mo.search(goal[:, j], SIZE, gen,
                                                  wander_max=a.search_wander_max)
                        tt = torch.arange(len(path), device=device)[:, None]
                        first, found = tt == ns[None], tt >= ns[None]
                    else:
                        path, psi, first, found = te.visit_path(goal[:, j], a, device, gen)
                    act = ~found | first
                    for t in range(len(path)):
                        yield dict(j=j, pos=path[t], psi=psi[t], flag=first[t].float(),
                                   active=act[t], pw=zeros, ct=found[t].float(),
                                   cw=act[t].float(), last=last)
                    continue
                if a.motion == "continuous":
                    path, psi, n = mo.revisit(goal[:, j], SIZE, gen, wander_max=a.wander_max,
                                              p_wander=a.p_wander)
                else:
                    start = torch.randint(0, SIZE, (B, 2), device=device, generator=gen)
                    path, n = toward(start, goal[:, j], gen)
                    psi = headings(path, device, gen)
                for t in range(len(path)):
                    act = t <= n
                    far = te.far_from(path[t], goal[:, j]) & act
                    yield dict(j=j, pos=path[t], psi=psi[t], flag=zeros, active=act,
                               pw=far.float(), ct=ones, cw=act.float(), last=last)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50000, help="optimizer updates")
    p.add_argument("--batch", type=int, default=32, help="lifetimes in parallel")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--enc", type=int, default=256)
    p.add_argument("--read", type=int, default=256)
    p.add_argument("--enc_kind", default="none", choices=["relu", "linear", "none"])
    p.add_argument("--cell", default="lstm", choices=["gru", "lstm"])
    p.add_argument("--visit_steps", type=int, default=40)
    p.add_argument("--tau_min", type=int, default=5)
    p.add_argument("--tau_tail", type=int, default=5)
    p.add_argument("--iters", type=int, default=10, help="iterations per block in training")
    p.add_argument("--motion", default="grid", choices=["grid", "continuous"],
                   help="grid: one-cell moves along an axis; continuous: the agent's unit steps "
                        "in any direction snapped to cells, searches that wander first, revisits "
                        "that sometimes detour (motion.py)")
    p.add_argument("--search_wander_max", type=int, default=60)
    p.add_argument("--wander_max", type=int, default=40, help="continuous: longest revisit detour")
    p.add_argument("--p_wander", type=float, default=0.3,
                   help="continuous: chance a revisit detours before approaching")
    p.add_argument("--A_min", type=int, default=2)
    p.add_argument("--A_max", type=int, default=6)
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--curriculum_cos", type=float, default=0.95)
    p.add_argument("--curriculum_window", type=int, default=20, help="lifetimes")
    p.add_argument("--window", type=int, default=256, help="env steps per update")
    p.add_argument("--n_train_envs", type=int, default=4000)
    p.add_argument("--n_eval_envs", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_arenas", type=int, default=6)
    p.add_argument("--eval_iters", type=int, default=40)
    p.add_argument("--eval_lifetimes", type=int, default=32)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--init_from", default=None,
                   help="start from these weights (a train_experience memory.pt), new optimizer")
    p.add_argument("--init_ckpt", default=None, help="resume this run's memory.pt")
    a = p.parse_args()
    a.chrono = 0
    os.makedirs(a.out, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    t0 = time.time()
    gp = GpuPool(EnvPool(a.n_train_envs, seed=a.seed + 1), device)
    gp_eval = GpuPool(EnvPool(a.n_eval_envs, seed=a.seed + 2), device)
    print(f"pools built: {len(gp)} train, {len(gp_eval)} eval envs ({time.time() - t0:.0f}s)",
          flush=True)
    model = te.build(a, device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    log = {"args": vars(a), "train": [], "eval": []}
    upd, A_cur, cwin = 0, (a.A_min if a.curriculum else a.A_max), []
    if a.init_ckpt:
        ck = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        upd, A_cur, cwin = ck["step"], ck["A_cur"], list(ck["cwin"])
        old_log = os.path.join(os.path.dirname(a.init_ckpt), "log.json")
        if os.path.exists(old_log):
            prev = json.load(open(old_log))
            log["train"], log["eval"] = prev["train"], prev["eval"]
        print(f"resumed from {a.init_ckpt} (update {upd}, {A_cur} arenas)", flush=True)
    elif a.init_from:
        ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
        for k in ("hidden", "layers", "enc", "read", "enc_kind", "cell"):
            if ck["args"].get(k, getattr(a, k)) != getattr(a, k):
                raise SystemExit(f"--init_from has {k}={ck['args'][k]}, this run {getattr(a, k)}")
        model.load_state_dict(ck["model"])
        print(f"weights from {a.init_from} (its step {ck['step']})", flush=True)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"model: {a.cell.upper()} {a.layers} x {a.hidden}, {n_par / 1e6:.2f}M parameters",
          flush=True)
    gen = torch.Generator(device=device).manual_seed(a.seed + 4 + 7919 * upd)

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "args": vars(a),
                    "step": upd, "A_cur": A_cur, "cwin": cwin},
                   os.path.join(a.out, "memory.pt"))
        with open(os.path.join(a.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)

    def evaluate():
        model.eval()
        rows, steps = persist_run(model, gp_eval, a, a.eval_arenas, a.eval_iters,
                                  a.eval_lifetimes, 10_000, device,
                                  motion=getattr(a, "motion", "grid"))
        s = persist_summary(rows, a.eval_arenas, a.eval_iters)
        model.train()
        lastb = s[a.eval_arenas - 1]
        by_block = " ".join(f"{float(np.mean(list(r['acc_by_arena'].values()))):.2f}"
                            for r in s.values())
        arenas = " ".join(f"{v:.2f}" for v in lastb["acc_by_arena"].values())
        print(f"eval update {upd}: far-read accuracy by block {by_block} | last block by "
              f"arena {arenas} | c {lastb['c_mean']:.2f} ({steps} memory steps)", flush=True)
        log["eval"].append(dict(update=upd, summary=s, memory_steps=steps))
        tested[0] = upd

    B, W = a.batch, a.window
    tested = [-1]                            # update of the last test
    stat = dict(loss=[], cos=0.0, n=0.0, csig=0.0, cn=0.0)
    n_life, done = 0, upd >= a.steps
    model.train()
    while not done:
        A = A_cur
        env = torch.rand(B, len(gp), device=device, generator=gen).argsort(1)[:, :A]
        off, goal, target = place(B, A, device, gen)
        h = model.init_state(B, device)
        pl = cl = 0.0
        npw = ncw = torch.zeros((), device=device)
        nstep = 0
        last_cos = last_n = torch.zeros((), device=device)
        for s in schedule(a, goal, A, a.iters, B, device, gen):
            j = s["j"]
            x = gp.canvas(env[:, j][None], s["pos"][None, :, 0], s["pos"][None, :, 1],
                          s["psi"][None])
            ex = te.extra_of(off[:, j][None], s["pos"][None], s["flag"][None])
            ph, c, h2 = model(x, h, extra=ex)
            h = torch.where(s["active"][None, :, None], h2, h)
            cos = angle_cos(ph[0], target[:, j])
            pl = pl + ((1 - cos) * s["pw"]).sum()
            cl = cl + (F.binary_cross_entropy_with_logits(c[0], s["ct"], reduction="none")
                       * s["cw"]).sum()
            npw, ncw = npw + s["pw"].sum(), ncw + s["cw"].sum()
            cd = cos.detach()
            stat["cos"] += (cd * s["pw"]).sum()
            stat["n"] += s["pw"].sum()
            rev = s["pw"] > 0
            stat["csig"] += (torch.sigmoid(c[0].detach()) * rev).sum()
            stat["cn"] += rev.sum()
            if s["last"]:
                last_cos, last_n = last_cos + (cd * s["pw"]).sum(), last_n + s["pw"].sum()
            nstep += 1
            if nstep == W:
                loss = pl / npw.clamp(min=1) + cl / ncw.clamp(min=1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                upd += 1
                stat["loss"].append(float(loss.detach()))
                h = h.detach()
                pl = cl = 0.0
                npw = ncw = torch.zeros((), device=device)
                nstep = 0
                if upd % a.log_every == 0:
                    m_cos = float(stat["cos"] / max(float(stat["n"]), 1.0))
                    m_c = float(stat["csig"] / max(float(stat["cn"]), 1.0))
                    rec = dict(update=upd, loss=float(np.mean(stat["loss"])), far_cos=m_cos,
                               c_revisit=m_c, A_cur=A_cur, lifetimes=n_life,
                               seconds=time.time() - t0)
                    log["train"].append(rec)
                    print(f"update {upd} loss {rec['loss']:.4f} far_cos {m_cos:.3f} c {m_c:.2f} "
                          f"arenas {A_cur} lifetimes {n_life} ({rec['seconds']:.0f}s)", flush=True)
                    stat = dict(loss=[], cos=0.0, n=0.0, csig=0.0, cn=0.0)
                if upd % a.eval_every == 0:
                    evaluate()
                if upd % a.save_every == 0:
                    save()
                if upd >= a.steps:
                    done = True
                    break
        if done:
            break
        if nstep > 0 and float(npw + ncw) > 0:                # the rest of the lifetime
            loss = pl / npw.clamp(min=1) + cl / ncw.clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        n_life += 1
        if a.curriculum and A_cur < a.A_max and float(last_n) > 0:
            cwin.append(float(last_cos / last_n))
            K = a.curriculum_window
            if len(cwin) >= K:
                if sum(cwin[-K:]) / K >= a.curriculum_cos:
                    A_cur += 1
                    print(f"curriculum: arenas now {A_cur} at update {upd}", flush=True)
                    cwin = []
                else:
                    cwin = cwin[-K:]
    if tested[0] != upd:
        evaluate()
    save()


if __name__ == "__main__":
    main()
