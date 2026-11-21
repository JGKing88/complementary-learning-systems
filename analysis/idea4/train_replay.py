"""Idea 4's memory trained on the agent's own movement (GRID_MLP_NAV_PLAN
§7b.15, DAgger for the memory).

    python -m analysis.idea4.train_replay --init_from <memory.pt> \
        --records '<dir>/rec_*.npz' --out <dir> [--steps 5000] [--lr 1e-4]

Along the agent's own path in the continual protocol, the memories trained on
scripted walks (``train_protocol.py``, grid or ``motion.py`` movement) decode
the goal on almost no reads, though they hold 3 or more goals offline. Here
the training paths are the agent's: ``agenthash --record_memory`` writes, per
read, the arena slot, cell, heading, store flag and whether that arena's goal
is stored, plus each arena's goal. This replays those paths through the
training pool's arenas at fresh scaffold offsets, so the views and phases are
new on every pass and none of the protocol's test arenas is seen.

Each of ``--batch`` rows replays one recorded lifetime; a row that reaches the
end of its recording starts another from a zero state, with new arenas and
offsets. Loss as ``train_protocol.py``: on every read more than 1.5 cells from
a stored goal, the cosine error of the decoded goal phases; on every read, c
(target 1 once the current arena's goal is stored, including the store read).
Streamed in windows of ``--window`` reads, the state carried on detached.

Test every ``--eval_every`` updates: ``persist.run`` on held-out arenas under
``motion.py`` movement (an offline check that the old skill survives); the
real test is the agent on the continual protocol.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from analysis.idea2.lifetimes import NPOS, SIZE, EnvPool
from analysis.idea4 import train_experience as te
from analysis.idea4.gpu_views import GpuPool, goal_phases
from analysis.idea4.persist import run as persist_run, summary as persist_summary
from analysis.idea4.train_memory import angle_cos


def load_records(pattern: str, device):
    """All recordings concatenated: per-read tensors (N,) and, per recording,
    its start, end and goals (R, A, 2)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no recordings match {pattern}")
    cols = {k: [] for k in ("arena", "x", "y", "psi", "flag", "known")}
    starts, ends, goals, n = [], [], [], 0
    A = 0
    for f in files:
        z = np.load(f)
        if int(z["arena_size"]) != SIZE:
            raise SystemExit(f"{f}: arena size {int(z['arena_size'])}, the pool's is {SIZE}")
        L = len(z["arena"])
        for k in cols:
            cols[k].append(z[k])
        starts.append(n)
        ends.append(n + L)
        goals.append(z["goals"])
        A = max(A, len(z["goals"]))
        n += L
    g = np.full((len(files), A, 2), -1, dtype=np.int64)
    for r, gr in enumerate(goals):
        g[r, :len(gr)] = gr
    t = lambda v, dt: torch.as_tensor(np.concatenate(v), dtype=dt, device=device)
    rec = dict(arena=t(cols["arena"], torch.long), x=t(cols["x"], torch.long),
               y=t(cols["y"], torch.long), psi=t(cols["psi"], torch.float32),
               flag=t(cols["flag"], torch.bool), known=t(cols["known"], torch.bool),
               start=torch.tensor(starts, device=device), end=torch.tensor(ends, device=device),
               goals=torch.as_tensor(g, device=device), A=A, files=files)
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init_from", required=True, help="the memory.pt to start from")
    p.add_argument("--records", required=True, help="glob of agenthash --record_memory files")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000, help="optimizer updates")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n_train_envs", type=int, default=4000)
    p.add_argument("--n_eval_envs", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_lifetimes", type=int, default=32)
    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--log_every", type=int, default=50)
    c = p.parse_args()
    os.makedirs(c.out, exist_ok=True)
    device = torch.device(c.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(c.seed)
    t0 = time.time()

    ck = torch.load(c.init_from, map_location="cpu", weights_only=False)
    a = types.SimpleNamespace(**ck["args"])
    a.chrono = 0
    a.enc_kind = getattr(a, "enc_kind", "none")
    model = te.build(a, device)
    model.load_state_dict(ck["model"])
    opt = torch.optim.Adam(model.parameters(), lr=c.lr)
    rec = load_records(c.records, device)
    R, A, B, W = len(rec["files"]), rec["A"], c.batch, c.window
    print(f"{R} recordings, {int(rec['end'][-1])} reads, {A} arenas; weights from "
          f"{c.init_from} (its step {ck['step']})", flush=True)
    gp = GpuPool(EnvPool(c.n_train_envs, seed=c.seed + 1), device)
    gp_eval = GpuPool(EnvPool(c.n_eval_envs, seed=getattr(a, "seed", 0) + 2), device)
    print(f"pools built ({time.time() - t0:.0f}s)", flush=True)
    gen = torch.Generator(device=device).manual_seed(c.seed + 4)
    log = {"args": vars(c), "memory_args": vars(a), "train": [], "eval": []}
    upd = 0

    def save():
        args = dict(vars(a))
        args.update(replay_records=c.records, replay_init=c.init_from)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "args": args,
                    "step": upd}, os.path.join(c.out, "memory.pt"))
        with open(os.path.join(c.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)

    def evaluate():
        model.eval()
        for motion in ("grid", "continuous"):
            rows, _ = persist_run(model, gp_eval, a, 6, 40, c.eval_lifetimes, 10_000, device,
                                  motion=motion)
            s = persist_summary(rows, 6, 40)
            held = float(sum(s[5]["acc_by_arena"].values()))
            print(f"eval update {upd} ({motion}): goals held in the last block {held:.2f} | "
                  f"c in search before arrival {s[5]['c_search']:.2f}", flush=True)
            log["eval"].append(dict(update=upd, motion=motion, held=held, summary=s))
        model.train()

    # Row state: which recording, where in it, its arenas and offsets.
    rid = torch.randint(0, R, (B,), device=device, generator=gen)
    ptr = rec["start"][rid].clone()
    env = torch.rand(B, len(gp), device=device, generator=gen).argsort(1)[:, :A]
    off = torch.randint(0, NPOS - SIZE, (B, A, 2), device=device, generator=gen)
    h = model.init_state(B, device)
    rows = torch.arange(B, device=device)
    stat = dict(loss=[], cos=0.0, n=0.0, csig=0.0, cn=0.0, cuk=0.0, nuk=0.0)
    pl = cl = 0.0
    npw = ncw = torch.zeros((), device=device)
    nstep, n_life = 0, 0
    evaluate()
    model.train()
    while upd < c.steps:
        i = ptr
        j = rec["arena"][i]
        pos = torch.stack([rec["x"][i], rec["y"][i]], -1)
        flag, known = rec["flag"][i], rec["known"][i]
        goal = rec["goals"][rid, j]
        o = off[rows, j]
        x = gp.canvas(env[rows, j][None], pos[None, :, 0], pos[None, :, 1], rec["psi"][i][None])
        ex = te.extra_of(o[None], pos[None], flag[None])
        ph, cl_logit, h = model(x, h, extra=ex)
        target = goal_phases(o + goal)
        cos = angle_cos(ph[0], target)
        pw = (known & te.far_from(pos, goal)).float()
        ct = (known | flag).float()
        pl = pl + ((1 - cos) * pw).sum()
        cl = cl + F.binary_cross_entropy_with_logits(cl_logit[0], ct, reduction="sum")
        npw, ncw = npw + pw.sum(), ncw + B
        cd, sg = cos.detach(), torch.sigmoid(cl_logit[0].detach())
        stat["cos"] += (cd * pw).sum()
        stat["n"] += pw.sum()
        stat["csig"] += (sg * known).sum()
        stat["cn"] += known.sum()
        stat["cuk"] += (sg * ~(known | flag)).sum()
        stat["nuk"] += (~(known | flag)).sum()
        nstep += 1
        # Advance; a finished row starts another recording from a zero state.
        ptr = ptr + 1
        fin = ptr >= rec["end"][rid]
        if fin.any():
            k = int(fin.sum())
            rid = torch.where(fin, torch.randint(0, R, (B,), device=device, generator=gen), rid)
            ptr = torch.where(fin, rec["start"][rid], ptr)
            new_env = torch.rand(k, len(gp), device=device, generator=gen).argsort(1)[:, :A]
            env[fin] = new_env
            off[fin] = torch.randint(0, NPOS - SIZE, (k, A, 2), device=device, generator=gen)
            h = torch.where(fin[None, :, None], torch.zeros_like(h), h)
            n_life += k
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
            if upd % c.log_every == 0:
                f = lambda a_, b_: float(stat[a_] / max(float(stat[b_]), 1.0))
                r = dict(update=upd, loss=float(np.mean(stat["loss"])), far_cos=f("cos", "n"),
                         c_known=f("csig", "cn"), c_unknown=f("cuk", "nuk"),
                         lifetimes=n_life, seconds=time.time() - t0)
                log["train"].append(r)
                print(f"update {upd} loss {r['loss']:.4f} far_cos {r['far_cos']:.3f} "
                      f"c known {r['c_known']:.2f} unknown {r['c_unknown']:.2f} "
                      f"lifetimes done {n_life} ({r['seconds']:.0f}s)", flush=True)
                stat = dict(loss=[], cos=0.0, n=0.0, csig=0.0, cn=0.0, cuk=0.0, nuk=0.0)
            if upd % c.eval_every == 0:
                evaluate()
            if upd % c.save_every == 0:
                save()
    save()
    print(f"[train_replay] done at update {upd} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
