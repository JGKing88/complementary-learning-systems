"""Idea 4 trained from experience, with no write input (GRID_MLP_NAV_PLAN §7b).

    python -m analysis.idea4.train_experience --out <dir> --hidden 512 [--steps 20000]
    python -m analysis.idea4.train_experience --out <dir> --eval_ckpt <dir>/memory.pt

Every step the memory gets the view, the agent's own current grid phases (12
numbers, as path integration gives them) and a goal flag. The flag is 1 on the
first arrival at an arena's goal cell and 0 otherwise, which is idea 1's
oracle store signal. Nothing hands the memory the goal. To recall a goal
later it has to keep the grid phases it had when the flag fired.

A lifetime brings in T arenas in turn (T from [T_min, T_max]). Each arena gets
one visit, a 40-step random walk that stands on the goal at a random step.
From the first arrival on, every step is scored on the goal's phases with
c -> 1; before it, c -> 0, since the goal is not known yet. After each visit:
- copies of the state read 16 views at random cells and headings of every
  arena visited so far (goal phases, c -> 1) and of 2 unseen arenas (c -> 0),
  each view with that cell's grid phases;
- the state walks 0 to 40 steps in one arena, carried forward and scored at
  every step: a visited arena (a revisit), or with probability p_unseen an
  unseen arena or the next one to be visited (c -> 0). The flag stays 0, as
  the store is locked after the first arrival.
The loss backpropagates through the whole lifetime.

Test: N held-out arenas visited once each in sequence, then 64 views at
random cells and headings per arena, each with its grid phases. A view is
correct if the decoded goal cell is within 1.5 cells of the goal (chance
about 0.02). Reads are scored from copies of the state after the last visit
("copy") and in sequence, arena by arena ("seq"). c is read from copies on
visited and unseen arenas.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from hopfield_nav.memory.plastic import N_PHASE
from hopfield_nav.memory.recurrent import RecurrentMemory
from analysis.idea2.lifetimes import SIZE, EnvPool
from analysis.idea4.gpu_views import (GpuPool, goal_phases, headings, place, random_cells,
                                      rw_from, walk_through)
from analysis.idea4.train_memory import angle_cos


def extra_of(off, pos, flag):
    """Offsets and cells (..., 2) and the flag (...) -> (..., 13), the agent's
    grid phases at that cell and the at-goal flag."""
    return torch.cat([goal_phases(off + pos), flag[..., None].float()], -1)


def visit_path(goal, a, device, gen):
    """A visit for each row: path (S, B, 2), headings (S, B), first-arrival
    flag (S, B) and found-so-far mask (S, B)."""
    S = a.visit_steps
    tau = torch.randint(a.tau_min, S - a.tau_tail, (len(goal),), device=device, generator=gen)
    path = walk_through(goal, S, tau, device, gen)
    on = (path == goal[None]).all(-1)
    first = on & (on.long().cumsum(0) == 1)
    return path, headings(path, device, gen), first, first.long().cumsum(0) > 0


def lifetime_loss(model, gp, a, rng, gen, device, T_max=None):
    B = a.batch
    T = rng.randint(a.T_min, (T_max or a.T_max) + 1)
    K = T + a.n_unseen
    env = torch.rand(B, len(gp), device=device, generator=gen).argsort(1)[:, :K]
    off, goal, target = place(B, K, device, gen)
    ar = torch.arange(B, device=device)
    h = model.init_state(B, device)
    total, weight = 0.0, 0.0
    st = {"visit_cos": [], "copy_cos": [], "old_cos": [], "seq_cos": []}
    for t in range(T):
        S = a.visit_steps
        path, psi, first, found = visit_path(goal[:, t], a, device, gen)
        x = gp.canvas(env[:, t].expand(S, B), path[..., 0], path[..., 1], psi)
        ph, c, h = model(x, h, extra=extra_of(off[:, t].expand(S, B, 2), path, first))
        f = found.float()
        cos = angle_cos(ph, target[:, t][None])
        total = total + ((1 - cos) * f).sum() / f.sum().clamp(min=1.0) \
            + F.binary_cross_entropy_with_logits(c, f)
        weight += 1.0
        st["visit_cos"].append(float(((cos * f).sum() / f.sum().clamp(min=1.0)).detach()))

        # Copy reads of every visited arena and the unseen ones.
        cols = torch.as_tensor(np.concatenate([np.arange(t + 1), np.arange(T, K)]), device=device)
        is_w = cols < T
        J, Q = len(cols), a.test_views
        cx, cy, cpsi = random_cells((B, J, Q), device, gen)
        x = gp.canvas(env[:, cols][:, :, None].expand(B, J, Q), cx, cy, cpsi)
        ex = extra_of(off[:, cols][:, :, None].expand(B, J, Q, 2), torch.stack([cx, cy], -1),
                      torch.zeros(B, J, Q, device=device))
        ph, c, _ = model(x.reshape(1, B * J * Q, -1), h.repeat_interleave(J * Q, dim=1),
                         extra=ex.reshape(1, B * J * Q, -1))
        ph, c = ph.reshape(B, J, Q, N_PHASE), c.reshape(B, J, Q)
        cos = angle_cos(ph, target[:, cols][:, :, None])
        total = total + (1 - cos[:, is_w]).mean() + F.binary_cross_entropy_with_logits(
            c, is_w[None, :, None].expand_as(c).float())
        weight += 1.0
        st["copy_cos"].append(float(cos[:, is_w].mean().detach()))
        old = is_w & (cols != t)
        if old.any():
            st["old_cos"].append(float(cos[:, old].mean().detach()))

        # A carried walk: a revisit, or an unseen / not-yet-visited arena.
        L = rng.randint(0, a.R_max + 1)
        if L >= 2:
            not_yet = np.concatenate([[t + 1] if t + 1 < T else [], np.arange(T, K)])
            unseen = torch.rand(B, device=device, generator=gen) < a.p_unseen
            pick_w = torch.randint(0, t + 1, (B,), device=device, generator=gen)
            pick_u = torch.as_tensor(not_yet, device=device).long()[
                torch.randint(0, len(not_yet), (B,), device=device, generator=gen)]
            which = torch.where(unseen, pick_u, pick_w)
            start = torch.randint(0, SIZE, (B, 2), device=device, generator=gen)
            rpath = rw_from(start, L, device, gen)
            rpsi = headings(rpath, device, gen)
            x = gp.canvas(env[ar, which].expand(L, B), rpath[..., 0], rpath[..., 1], rpsi)
            ex = extra_of(off[ar, which][None].expand(L, B, 2), rpath, torch.zeros(L, B, device=device))
            ph, c, h = model(x, h, extra=ex)
            cos = angle_cos(ph, target[ar, which][None])
            wr = (~unseen).float()[None].expand(L, B)
            total = total + a.seq_weight * (((1 - cos) * wr).sum() / wr.sum().clamp(min=1.0)
                                            + F.binary_cross_entropy_with_logits(c, wr))
            weight += a.seq_weight
            if wr.sum() > 0:
                st["seq_cos"].append(float(((cos * wr).sum() / wr.sum()).detach()))
        if a.tbptt_envs and (t + 1) % a.tbptt_envs == 0:
            h = h.detach()
    return total / weight, T, st


def _hits(ph, off, goal):
    """ph (N, V, 12) predicted phases for arenas at offsets off (N, 2) with
    goals goal (N, 2) -> fraction of views decoded within 1.5 cells (N,)."""
    gx, gy = torch.meshgrid(torch.arange(SIZE, device=ph.device),
                            torch.arange(SIZE, device=ph.device), indexing="ij")
    cells = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)              # (400, 2)
    ref = goal_phases(off[:, None] + cells[None])                          # (N, 400, 12)
    p = ph.reshape(*ph.shape[:2], 6, 2)
    p = (p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)).reshape(*ph.shape)
    best = torch.einsum("nvk,nck->nvc", p, ref).argmax(-1)                 # (N, V)
    d = (cells[best] - goal[:, None]).float().norm(dim=-1)
    return (d <= 1.5).float().mean(1)


@torch.no_grad()
def retention(model, gp, N, seed, a, device):
    g = torch.Generator(device=device).manual_seed(seed)
    K = N + 10
    env = torch.randperm(len(gp), device=device, generator=g)[:K]
    off, goal, _ = place(1, K, device, g)
    off, goal = off[0], goal[0]
    h = model.init_state(1, device)
    S = a.visit_steps
    for t in range(N):
        path, psi, first, _ = visit_path(goal[t:t + 1], a, device, g)
        x = gp.canvas(env[t:t + 1].expand(S, 1), path[..., 0], path[..., 1], psi)
        _, _, h = model(x, h, extra=extra_of(off[t:t + 1].expand(S, 1, 2), path, first))
    V = a.eval_views
    cx, cy, cpsi = random_cells((K, V), device, g)
    X = gp.canvas(env[:, None].expand(K, V), cx, cy, cpsi)
    EX = extra_of(off[:, None].expand(K, V, 2), torch.stack([cx, cy], -1),
                  torch.zeros(K, V, device=device))
    Xw, EXw = X[:N].reshape(1, N * V, -1), EX[:N].reshape(1, N * V, -1)
    ph_c, c_w, _ = model(Xw, h.repeat_interleave(N * V, dim=1), extra=EXw)
    _, c_u, _ = model(X[N:].reshape(1, 10 * V, -1), h.repeat_interleave(10 * V, dim=1),
                      extra=EX[N:].reshape(1, 10 * V, -1))
    ph_s, c_s, _ = model(Xw.reshape(N * V, 1, -1), h, extra=EXw.reshape(N * V, 1, -1))
    acc_c = _hits(ph_c.reshape(N, V, N_PHASE), off[:N], goal[:N]).cpu().numpy()
    acc_s = _hits(ph_s.reshape(N, V, N_PHASE), off[:N], goal[:N]).cpu().numpy()
    cw, cu = torch.sigmoid(c_w[0]).cpu().numpy(), torch.sigmoid(c_u[0]).cpu().numpy()
    auc = float((cw[:, None] > cu[None, :]).mean() + 0.5 * (cw[:, None] == cu[None, :]).mean())
    q = max(1, N // 4)
    return dict(N=N, seed=seed,
                acc_copy=float(acc_c.mean()), acc_seq=float(acc_s.mean()),
                copy_oldest_quarter=float(acc_c[:q].mean()),
                copy_newest_quarter=float(acc_c[-q:].mean()),
                seq_oldest_quarter=float(acc_s[:q].mean()),
                seq_newest_quarter=float(acc_s[-q:].mean()),
                c_written=float(cw.mean()), c_unseen=float(cu.mean()),
                c_written_seq=float(torch.sigmoid(c_s).mean()), c_auc=auc,
                acc_copy_per_env=acc_c.round(4).tolist(),
                acc_seq_per_env=acc_s.round(4).tolist())


def _line(r):
    return (f"N={r['N']:3d} seed {r['seed']}: copy {r['acc_copy']:.3f} (oldest "
            f"{r['copy_oldest_quarter']:.2f}, newest {r['copy_newest_quarter']:.2f}) | seq "
            f"{r['acc_seq']:.3f} (oldest {r['seq_oldest_quarter']:.2f}, newest "
            f"{r['seq_newest_quarter']:.2f}) | c visited {r['c_written']:.2f} unseen "
            f"{r['c_unseen']:.2f} auc {r['c_auc']:.3f}")


def build(a, device):
    return RecurrentMemory(hidden=a.hidden, layers=a.layers, enc=a.enc, read=a.read,
                           chrono=a.chrono).to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32, help="lifetimes per step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--enc", type=int, default=256)
    p.add_argument("--read", type=int, default=256)
    p.add_argument("--chrono", type=int, default=0)
    p.add_argument("--T_min", type=int, default=2)
    p.add_argument("--T_max", type=int, default=20)
    p.add_argument("--n_unseen", type=int, default=2)
    p.add_argument("--visit_steps", type=int, default=40)
    p.add_argument("--tau_min", type=int, default=5, help="earliest step of the goal visit")
    p.add_argument("--tau_tail", type=int, default=5,
                   help="the goal visit is at least this many steps before the visit ends")
    p.add_argument("--test_views", type=int, default=16)
    p.add_argument("--R_max", type=int, default=40)
    p.add_argument("--p_unseen", type=float, default=0.2)
    p.add_argument("--seq_weight", type=float, default=1.0)
    p.add_argument("--tbptt_envs", type=int, default=0)
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--curriculum_cos", type=float, default=0.8)
    p.add_argument("--n_train_envs", type=int, default=4000)
    p.add_argument("--n_eval_envs", type=int, default=300)
    p.add_argument("--eval_views", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_N", type=int, nargs="+", default=[2, 6, 30])
    p.add_argument("--final_N", type=int, nargs="+", default=[1, 2, 3, 6, 10, 20, 30, 50, 100])
    p.add_argument("--final_seeds", type=int, nargs="+", default=[10_000, 20_000, 30_000])
    p.add_argument("--init_ckpt", default=None)
    p.add_argument("--eval_ckpt", default=None)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    t0 = time.time()
    gp_eval = GpuPool(EnvPool(a.n_eval_envs, seed=a.seed + 2), device)

    def final_eval(model, tag):
        res = []
        for base in a.final_seeds:
            for N in a.final_N:
                r = retention(model, gp_eval, N, base + N, a, device)
                r["tag"] = tag
                res.append(r)
                print("final " + _line(r), flush=True)
        with open(os.path.join(a.out, f"eval_{tag}.json"), "w") as f:
            json.dump(res, f, indent=1)

    if a.eval_ckpt:
        ck = torch.load(a.eval_ckpt, map_location="cpu", weights_only=False)
        for k in ("hidden", "layers", "enc", "read"):
            setattr(a, k, ck["args"][k])
        a.chrono = 0
        model = build(a, device)
        model.load_state_dict(ck["model"])
        model.eval()
        final_eval(model, f"step{ck['step']}")
        return

    gp = GpuPool(EnvPool(a.n_train_envs, seed=a.seed + 1), device)
    print(f"pools built: {len(gp)} train, {len(gp_eval)} eval envs ({time.time() - t0:.0f}s)",
          flush=True)
    model = build(a, device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    log = {"args": vars(a), "train": [], "eval": []}
    start = 1
    if a.init_ckpt:
        ck = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start = ck["step"] + 1
        old_log = os.path.join(os.path.dirname(a.init_ckpt), "log.json")
        if os.path.exists(old_log):
            prev = json.load(open(old_log))
            log["train"], log["eval"] = prev["train"], prev["eval"]
        print(f"resumed from {a.init_ckpt} (step {ck['step']})", flush=True)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"model: GRU {a.layers} x {a.hidden}, {n_par / 1e6:.2f}M parameters", flush=True)
    rng = np.random.RandomState(a.seed + 3 + 7919 * (start - 1))
    gen = torch.Generator(device=device).manual_seed(a.seed + 4 + 7919 * (start - 1))

    def save(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "args": vars(a), "step": step}, os.path.join(a.out, "memory.pt"))
        with open(os.path.join(a.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)

    T_cur = a.T_max
    if a.curriculum:
        T_cur = max(a.T_min, 2)
        if log["train"] and "T_cur" in log["train"][-1]:
            T_cur = log["train"][-1]["T_cur"]
    keys = ("visit_cos", "copy_cos", "old_cos", "seq_cos")
    window = []
    for step in range(start, a.steps + 1):
        model.train()
        loss, T, st = lifetime_loss(model, gp, a, rng, gen, device, T_max=T_cur)
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        skipped = not np.isfinite(gn)        # inf * 0 = NaN would poison Adam
        if skipped:
            opt.zero_grad()
        else:
            opt.step()
        window.append([float(loss.detach()), np.nan if skipped else gn, skipped]
                      + [np.mean(st[k]) if st[k] else np.nan for k in keys])
        if step % 50 == 0:
            w = np.array(window, dtype=np.float64)
            m = {k: float(np.nanmean(w[:, 3 + i])) for i, k in enumerate(keys)}
            n_skip = int(w[:, 2].sum())
            log["train"].append(dict(step=step, loss=float(np.nanmean(w[:, 0])),
                                     grad_norm=float(np.nanmean(w[:, 1])), skipped=n_skip,
                                     T_cur=T_cur, seconds=time.time() - t0, **m))
            print(f"step {step} loss {np.nanmean(w[:, 0]):.4f} visit_cos {m['visit_cos']:.3f} "
                  f"copy_cos {m['copy_cos']:.3f} old_cos {m['old_cos']:.3f} seq_cos "
                  f"{m['seq_cos']:.3f} T_cur {T_cur} grad {np.nanmean(w[:, 1]):.2f}"
                  f"{f' SKIPPED {n_skip}' if n_skip else ''} ({time.time() - t0:.0f}s)", flush=True)
            window = []
            if a.curriculum and T_cur < a.T_max and m["old_cos"] >= a.curriculum_cos:
                T_cur += 1
                print(f"curriculum: T_max now {T_cur} at step {step}", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            for N in a.eval_N:
                r = retention(model, gp_eval, N, 10_000 + N, a, device)
                r["step"] = step
                log["eval"].append(r)
                print(f"eval step {step} " + _line(r), flush=True)
            save(step)
    final_eval(model, f"step{a.steps}")


if __name__ == "__main__":
    main()
