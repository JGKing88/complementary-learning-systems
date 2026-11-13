"""Train idea 4's recurrent memory on random-walk lifetimes, then run idea 2's
retention test (GRID_MLP_NAV_PLAN §7b.2).

    python -m analysis.idea4.train_memory --out <dir> --hidden 512 [--steps 20000]
    python -m analysis.idea4.train_memory --out <dir> --eval_ckpt <dir>/memory.pt

Training. A batch of lifetimes runs in parallel, each bringing in T envs (T
from [T_min, T_max], one T per batch). Each env is written once, by a 40-step
random walk with the goal phases and the write flag fed in. After each write
the module is scored from a copy of its state on fresh views of every env
written so far (phase loss, c -> 1) and of unseen envs (c -> 0). Then it reads
a walk of 0 to R_max steps in one env, carrying its state forward and scored
at every step, as navigation reads would be. The walk's env is a written one,
or with probability p_unseen an unseen one or the env to be written next (as
exploring before the find would be). The loss backpropagates through the whole
lifetime unless --tbptt_envs cuts it.

Test ("once", the go/no-go of §6.4). It draws the same envs, goals, write
walks and test views as idea 2's ``retention`` in analysis/idea2/meta_train.py
at the same seed. Write N held-out envs once each in sequence, then decode each
env's goal from 64 fresh views at uniform cells and headings: correct if the
decoded cell is within 1.5 cells of the goal (chance about 0.02; one goal for
every env scores 1/N). Reads change a recurrent memory, so the test scores
them two ways. "copy" reads each view from a copy of the state right after the
last write, which measures storage alone. "seq" reads all views in sequence,
env by env in write order, carrying the state through every read, which adds
the erosion reading causes. c is read from copies on written and unseen envs.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from hopfield_nav.memory.plastic import N_PHASE, decode_cells
from hopfield_nav.memory.recurrent import RecurrentMemory
from analysis.idea2.lifetimes import LAMBDAS, EnvPool, Placed
from analysis.idea4.gpu_views import GpuPool, place, random_cells, walk


def T_(x, device):
    return torch.from_numpy(np.asarray(x, np.float32)).to(device)


def angle_cos(ph, tgt):
    """Mean cosine over the 6 phase angles: (..., 12) x (..., 12) -> (...)."""
    return (ph * tgt).reshape(*ph.shape[:-1], 6, 2).sum(-1).mean(-1)


def lifetime_loss(model, gp, a, rng, gen, device, T_max=None):
    B = a.batch
    T = rng.randint(a.T_min, (T_max or a.T_max) + 1)
    K = T + a.n_unseen
    env = torch.rand(B, len(gp), device=device, generator=gen).argsort(1)[:, :K]
    _, _, target = place(B, K, device, gen)                    # (B, K, 12)
    ar = torch.arange(B, device=device)
    h = model.init_state(B, device)
    total, weight = 0.0, 0.0
    stats = {"copy_cos": [], "seq_cos": [], "old_cos": []}
    for t in range(T):
        S = a.visit_steps
        xs, ys, psi = walk(B, S, device, gen)
        x = gp.canvas(env[:, t].expand(S, B), xs, ys, psi)
        _, _, h = model(x, h, target[:, t].expand(S, B, N_PHASE))

        # Copy reads: one step from the post-write state on Q fresh views of
        # each scored written env and each unseen env.
        written = np.arange(t + 1)
        if a.score_envs and t + 1 > a.score_envs:
            written = np.sort(rng.choice(t + 1, size=a.score_envs, replace=False))
        cols = torch.as_tensor(np.concatenate([written, np.arange(T, K)]), device=device)
        is_w = cols < T
        J, Q = len(cols), a.test_views
        xs, ys, psi = random_cells((B, J, Q), device, gen)
        x = gp.canvas(env[:, cols][:, :, None].expand(B, J, Q), xs, ys, psi)
        ph, c, _ = model(x.reshape(1, B * J * Q, -1), h.repeat_interleave(J * Q, dim=1))
        ph, c = ph.reshape(B, J, Q, N_PHASE), c.reshape(B, J, Q)
        cos = angle_cos(ph, target[:, cols][:, :, None])
        total = total + (1 - cos[:, is_w]).mean() + F.binary_cross_entropy_with_logits(
            c, is_w[None, :, None].expand_as(c).float())
        weight += 1.0
        stats["copy_cos"].append(float(cos[:, is_w].mean().detach()))
        old = is_w & (cols != t)
        if old.any():                      # written before the newest write
            stats["old_cos"].append(float(cos[:, old].mean().detach()))

        # Navigation-like reads, carried forward.
        L = rng.randint(0, a.R_max + 1)
        if L > 0:
            not_yet = np.concatenate([[t + 1] if t + 1 < T else [], np.arange(T, K)])
            unseen = torch.rand(B, device=device, generator=gen) < a.p_unseen
            pick_w = torch.randint(0, t + 1, (B,), device=device, generator=gen)
            pick_u = torch.as_tensor(not_yet, device=device).long()[
                torch.randint(0, len(not_yet), (B,), device=device, generator=gen)]
            which = torch.where(unseen, pick_u, pick_w)
            xs, ys, psi = walk(B, L, device, gen)
            x = gp.canvas(env[ar, which].expand(L, B), xs, ys, psi)
            ph, c, h = model(x, h)
            cos = angle_cos(ph, target[ar, which][None])                 # (L, B)
            wr = (~unseen).float()[None].expand(L, B)
            ph_loss = ((1 - cos) * wr).sum() / wr.sum().clamp(min=1.0)
            total = total + a.seq_weight * (
                ph_loss + F.binary_cross_entropy_with_logits(c, wr))
            weight += a.seq_weight
            if wr.sum() > 0:
                stats["seq_cos"].append(float(((cos * wr).sum() / wr.sum()).detach()))
        if a.tbptt_envs and (t + 1) % a.tbptt_envs == 0:
            h = h.detach()
    return total / weight, T, stats


def _hits(ph, env):
    cell = decode_cells(ph, env.cells_global, LAMBDAS)
    return float((np.linalg.norm(env.cells_local[cell] - env.goal, axis=1) <= 1.5).mean())


@torch.no_grad()
def retention(model, pool, N, seed, a, device):
    """Idea 2's "once" test at ``seed`` (the same random calls in the same
    order), scored from state copies and in sequence."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(pool), size=N + 10, replace=False)
    envs = [Placed(pool, i, rng) for i in idx[:N]]
    unseen = [Placed(pool, i, rng) for i in idx[N:]]
    h = model.init_state(1, device)
    S = a.visit_steps
    for env in envs:
        x = T_(env.visit(rng, S), device)[:, None]
        _, _, h = model(x, h, T_(env.target, device).expand(S, 1, N_PHASE))
    V = a.eval_views
    X = torch.cat([T_(env.test_views(rng, V)[0], device) for env in envs])
    Xu = torch.cat([T_(u.test_views(rng, V)[0], device) for u in unseen])
    ph_c, c_w, _ = model(X[None], h.repeat_interleave(len(X), dim=1))
    _, c_u, _ = model(Xu[None], h.repeat_interleave(len(Xu), dim=1))
    ph_s, c_s, _ = model(X[:, None], h)
    ph_c, ph_s = ph_c[0].cpu().numpy(), ph_s[:, 0].cpu().numpy()
    acc_c = np.array([_hits(ph_c[k * V:(k + 1) * V], e) for k, e in enumerate(envs)])
    acc_s = np.array([_hits(ph_s[k * V:(k + 1) * V], e) for k, e in enumerate(envs)])
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
            f"{r['seq_newest_quarter']:.2f}) | c written {r['c_written']:.2f} unseen "
            f"{r['c_unseen']:.2f} auc {r['c_auc']:.3f}")


def build(a, device):
    return RecurrentMemory(hidden=a.hidden, layers=a.layers, enc=a.enc,
                           read=a.read).to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32, help="lifetimes per step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512, help="GRU width")
    p.add_argument("--layers", type=int, default=1, help="GRU layers")
    p.add_argument("--enc", type=int, default=256, help="view encoder width (idea 2's M)")
    p.add_argument("--read", type=int, default=256, help="read MLP width")
    p.add_argument("--T_min", type=int, default=2)
    p.add_argument("--T_max", type=int, default=20)
    p.add_argument("--n_unseen", type=int, default=2)
    p.add_argument("--visit_steps", type=int, default=40)
    p.add_argument("--test_views", type=int, default=16)
    p.add_argument("--score_envs", type=int, default=None,
                   help="after each write, score a random this many written envs, not all")
    p.add_argument("--R_max", type=int, default=40,
                   help="navigation-like reads after each write: 0 to this many steps")
    p.add_argument("--p_unseen", type=float, default=0.2)
    p.add_argument("--seq_weight", type=float, default=1.0)
    p.add_argument("--tbptt_envs", type=int, default=0,
                   help="detach the state every this many envs (0 = full lifetime)")
    p.add_argument("--curriculum", action="store_true",
                   help="start lifetimes at T <= 2 and lengthen them as older goals are held")
    p.add_argument("--curriculum_cos", type=float, default=0.8)
    p.add_argument("--n_train_envs", type=int, default=4000)
    p.add_argument("--n_eval_envs", type=int, default=300)
    p.add_argument("--eval_views", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_N", type=int, nargs="+", default=[6, 30, 100])
    p.add_argument("--final_N", type=int, nargs="+", default=[1, 2, 3, 6, 10, 20, 30, 50, 100])
    p.add_argument("--final_seeds", type=int, nargs="+", default=[10_000, 20_000, 30_000],
                   help="test seed bases (seed = base + N); 10_000 is idea 2's")
    p.add_argument("--init_ckpt", default=None)
    p.add_argument("--eval_ckpt", default=None, help="evaluate a saved memory.pt and exit")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    t0 = time.time()
    eval_pool = EnvPool(a.n_eval_envs, seed=a.seed + 2)      # idea 2's eval pool

    def final_eval(model, tag):
        res = []
        for base in a.final_seeds:
            for N in a.final_N:
                r = retention(model, eval_pool, N, base + N, a, device)
                r["tag"] = tag
                res.append(r)
                print("final " + _line(r), flush=True)
        with open(os.path.join(a.out, f"eval_{tag}.json"), "w") as f:
            json.dump(res, f, indent=1)

    if a.eval_ckpt:
        ck = torch.load(a.eval_ckpt, map_location="cpu", weights_only=False)
        for k in ("hidden", "layers", "enc", "read"):
            setattr(a, k, ck["args"][k])
        model = build(a, device)
        model.load_state_dict(ck["model"])
        model.eval()
        final_eval(model, f"step{ck['step']}")
        return

    train_pool = EnvPool(a.n_train_envs, seed=a.seed + 1)
    gp = GpuPool(train_pool, device)
    print(f"pools built: {len(train_pool)} train, {len(eval_pool)} eval envs "
          f"({time.time() - t0:.0f}s)", flush=True)
    model = build(a, device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    log = {"args": vars(a), "train": [], "eval": []}
    start = 1
    if a.init_ckpt:
        # Resume: model, optimiser and step count. Seeds the random streams
        # from the step, so a resumed run does not replay its first lifetimes.
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
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: GRU {a.layers} x {a.hidden}, {n_par / 1e6:.2f}M parameters", flush=True)
    rng = np.random.RandomState(a.seed + 3 + 7919 * (start - 1))
    gen = torch.Generator(device=device).manual_seed(a.seed + 4 + 7919 * (start - 1))

    def save(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "args": vars(a), "step": step}, os.path.join(a.out, "memory.pt"))
        with open(os.path.join(a.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)

    # Curriculum on lifetime length: T is drawn from [T_min, T_cur], and T_cur
    # grows by one whenever the copy-read cosine on goals written before the
    # newest one averages --curriculum_cos over 50 steps. Holding only the
    # newest goal cannot pass the gate.
    T_cur = a.T_max
    if a.curriculum:
        T_cur = max(a.T_min, 2)
        if log["train"] and "T_cur" in log["train"][-1]:
            T_cur = log["train"][-1]["T_cur"]
    window = []
    for step in range(start, a.steps + 1):
        model.train()
        loss, T, st = lifetime_loss(model, gp, a, rng, gen, device, T_max=T_cur)
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        window.append((float(loss.detach()), T, np.mean(st["copy_cos"]),
                       np.mean(st["seq_cos"]) if st["seq_cos"] else np.nan, gn,
                       np.mean(st["old_cos"]) if st["old_cos"] else np.nan))
        if step % 50 == 0:
            w = np.array(window)
            old_cos = float(np.nanmean(w[:, 5]))
            log["train"].append(dict(step=step, loss=float(w[:, 0].mean()),
                                     copy_cos=float(w[:, 2].mean()),
                                     seq_cos=float(np.nanmean(w[:, 3])),
                                     old_cos=old_cos, T_cur=T_cur,
                                     grad_norm=float(w[:, 4].mean()),
                                     seconds=time.time() - t0))
            print(f"step {step} loss {w[:, 0].mean():.4f} copy_cos {w[:, 2].mean():.3f} "
                  f"old_cos {old_cos:.3f} seq_cos {np.nanmean(w[:, 3]):.3f} T_cur {T_cur} "
                  f"grad {w[:, 4].mean():.2f} ({time.time() - t0:.0f}s)", flush=True)
            window = []
            if a.curriculum and T_cur < a.T_max and old_cos >= a.curriculum_cos:
                T_cur += 1
                print(f"curriculum: T_max now {T_cur} at step {step}", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            for N in a.eval_N:
                r = retention(model, eval_pool, N, 10_000 + N, a, device)
                r["step"] = step
                log["eval"].append(r)
                print(f"eval step {step} " + _line(r), flush=True)
            save(step)
    final_eval(model, f"step{a.steps}")


if __name__ == "__main__":
    main()
