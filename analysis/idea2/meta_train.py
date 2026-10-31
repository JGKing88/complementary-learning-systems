"""Meta-train idea 2's memory offline, then test retention against N (plan §6.4).

    python -m analysis.idea2.meta_train --out <dir> [--steps 20000] [--device cuda]
    python -m analysis.idea2.meta_train --out <dir> --baseline_only

Meta-learned: the initial weights θ₀ of ``M`` and a learning rate per weight.
A training lifetime brings in T envs (T drawn from [T_min, T_max]). For each
new env it writes one visit, scores ``M`` on fresh views of every env written
so far (phase loss + ``c`` -> 1) and of unseen envs (``c`` -> 0), then
revisits every earlier env with another write. The outer loss averages those
scores over the lifetime and backpropagates through every write (exact second
order).

The retention test runs the same lifetime structure, with no gradients into
θ₀, on held-out envs, for N = 6, 30, 100 new envs. After the last env, each
env is tested from fresh views: correct if the decoded goal cell is within 1.5
cells of the true goal. ``c`` is reported on written and unseen envs. The
baseline is the same network from a random start with a fixed learning rate,
for several rates, which shows the forgetting meta-learning has to prevent.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from hopfield_nav.memory.plastic import (
    decode_cells, forward, init_params, phase_loss, write)
from analysis.idea2.lifetimes import LAMBDAS, EnvPool, Placed


def T_(x, device):
    return torch.from_numpy(np.asarray(x, np.float32)).to(device)


def _wkw(a):
    return dict(n_plastic=a.plastic_tensors, key=a.key,
                loss_kind=a.write_loss)


def lifetime_loss(theta, log_alpha, pool, rng, a, device):
    T = rng.randint(a.T_min, a.T_max + 1)
    idx = rng.choice(len(pool), size=T + a.n_unseen, replace=False)
    envs = [Placed(pool, i, rng) for i in idx[:T]]
    unseen = [Placed(pool, i, rng) for i in idx[T:]]
    params, total, n = list(theta), 0.0, 0
    for t, env in enumerate(envs):
        params = write(params, log_alpha, T_(env.visit(rng, a.visit_steps), device),
                       T_(env.target, device), a.k, create_graph=True, **_wkw(a))
        scored = range(t + 1)
        if a.score_envs and t + 1 > a.score_envs:
            scored = rng.choice(t + 1, size=a.score_envs, replace=False)
        for j in scored:
            x, _ = envs[j].test_views(rng, a.test_views)
            ph, c = forward(params, T_(x, device), a.key)
            total = total + phase_loss(ph, T_(envs[j].target, device).expand_as(ph)) \
                + F.binary_cross_entropy_with_logits(c, torch.ones_like(c))
            n += 1
        for u in unseen:
            x, _ = u.test_views(rng, a.test_views)
            _, c = forward(params, T_(x, device), a.key)
            total = total + F.binary_cross_entropy_with_logits(c, torch.zeros_like(c))
            n += 1
        if a.train_revisit:
            for j in range(t):
                params = write(params, log_alpha, T_(envs[j].visit(rng, a.visit_steps), device),
                               T_(envs[j].target, device), a.k, create_graph=True, **_wkw(a))
    return total / n


@torch.no_grad()
def _test(params, envs, rng, n_views, device, key="relu"):
    acc, conf = [], []
    for env in envs:
        x, _ = env.test_views(rng, n_views)
        ph, c = forward(params, T_(x, device), key)
        cell = decode_cells(ph.cpu().numpy(), env.cells_global, LAMBDAS)
        d = np.linalg.norm(env.cells_local[cell] - env.goal, axis=1)
        acc.append(float((d <= 1.5).mean()))
        conf.append(torch.sigmoid(c).cpu().numpy())
    return np.array(acc), conf


def retention(theta, log_alpha, pool, N, seed, a, device, revisit=True):
    """Write N held-out envs in sequence (revisiting all earlier ones after each
    new env), then test every env. No gradient reaches θ₀."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(pool), size=N + 10, replace=False)
    envs = [Placed(pool, i, rng) for i in idx[:N]]
    unseen = [Placed(pool, i, rng) for i in idx[N:]]
    params = [p.detach().clone().requires_grad_(True) for p in theta]

    def do_write(env):
        nonlocal params
        with torch.enable_grad():
            new = write(params, log_alpha, T_(env.visit(rng, a.visit_steps), device),
                        T_(env.target, device), a.k, create_graph=False, **_wkw(a))
        params = [p.detach().requires_grad_(True) for p in new]

    for t, env in enumerate(envs):
        do_write(env)
        if revisit:
            for j in range(t):
                do_write(envs[j])
    acc, conf_w = _test(params, envs, rng, a.eval_views, device, a.key)
    _, conf_u = _test(params, unseen, rng, a.eval_views, device, a.key)
    cw, cu = np.concatenate(conf_w), np.concatenate(conf_u)
    auc = float((cw[:, None] > cu[None, :]).mean() + 0.5 * (cw[:, None] == cu[None, :]).mean())
    q = max(1, N // 4)
    return dict(N=N, acc=float(acc.mean()), acc_oldest_quarter=float(acc[:q].mean()),
                acc_newest_quarter=float(acc[-q:].mean()), c_written=float(cw.mean()),
                c_unseen=float(cu.mean()), c_auc=auc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--meta_batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--alpha0", type=float, default=0.03)
    p.add_argument("--T_min", type=int, default=2)
    p.add_argument("--T_max", type=int, default=6)
    p.add_argument("--n_unseen", type=int, default=2)
    p.add_argument("--plastic_tensors", type=int, default=None,
                   help="writes change only the last this many tensors; 2 = head only "
                        "(plan §6.5 R + H, the rest is meta-learned and fixed in a lifetime)")
    p.add_argument("--write_loss", choices=["cos", "mse"], default="cos",
                   help="loss a write minimises; mse on raw phases avoids the cosine "
                        "loss's shrinking gradient after a large write")
    p.add_argument("--normalize_hidden", action="store_true",
                   help="unit-length last hidden layer, so a head write acts like a delta rule")
    p.add_argument("--linear_key", action="store_true",
                   help="with --normalize_hidden, no ReLU on the last hidden layer, so keys "
                        "can be signed and different envs' keys can be orthogonal")
    p.add_argument("--train_revisit", action=argparse.BooleanOptionalAction, default=True,
                   help="training lifetimes revisit every earlier env after each new one (plan §6.3)")
    p.add_argument("--visit_steps", type=int, default=40)
    p.add_argument("--test_views", type=int, default=16)
    p.add_argument("--eval_views", type=int, default=64)
    p.add_argument("--n_train_envs", type=int, default=4000)
    p.add_argument("--n_eval_envs", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_N", type=int, nargs="+", default=[6, 30, 100])
    p.add_argument("--score_envs", type=int, default=None,
                   help="after each write, score a random this many written envs instead of "
                        "all of them, so long lifetimes cost O(T) rather than O(T^2)")
    p.add_argument("--init_ckpt", default=None,
                   help="start meta-training from a saved meta_memory.pt (same architecture)")
    p.add_argument("--baseline_only", action="store_true")
    p.add_argument("--eval_ckpt", default=None,
                   help="evaluate a saved meta_memory.pt (and the baselines) and exit")
    p.add_argument("--eval_modes", nargs="+", default=["once", "revisit"],
                   choices=["once", "revisit"])
    p.add_argument("--baseline_alphas", type=float, nargs="+",
                   default=[0.003, 0.01, 0.03, 0.1, 0.3])
    a = p.parse_args()
    if a.linear_key and not a.normalize_hidden:
        p.error("--linear_key needs --normalize_hidden")
    a.key = "linear_unit" if a.linear_key else ("unit" if a.normalize_hidden else "relu")
    os.makedirs(a.out, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    t0 = time.time()
    train_pool = EnvPool(a.n_train_envs, seed=a.seed + 1)
    eval_pool = EnvPool(a.n_eval_envs, seed=a.seed + 2)
    print(f"pools built: {len(train_pool)} train, {len(eval_pool)} eval envs "
          f"({time.time() - t0:.0f}s)", flush=True)
    in_dim = 2 * 180
    log = {"args": vars(a), "train": [], "eval": [], "baseline": []}

    def save():
        with open(os.path.join(a.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)

    if a.eval_ckpt:
        # Evaluation only. "once" is the go/no-go of plan §6.4: N envs written
        # once each, in sequence, then all tested, which is what idea 1's
        # 0.96-1.00 measures. "revisit" is the training protocol; with many
        # revisits even a plain network rehearses, so it is not a memory test.
        ck = torch.load(a.eval_ckpt, map_location="cpu", weights_only=False)
        models = [("meta", [t.to(device) for t in ck["theta"]],
                   [t.to(device) for t in ck["log_alpha"]])]
        for alpha in a.baseline_alphas:
            th = [t.to(device) for t in init_params(in_dim, a.hidden, a.layers, gen)]
            models.append((f"baseline_{alpha:g}", th,
                           [torch.full_like(t, math.log(alpha)) for t in th]))
        res = []
        for name, th, la in models:
            for mode in a.eval_modes:
                for N in a.eval_N:
                    r = retention(th, la, eval_pool, N, 10_000 + N, a, device,
                                  revisit=(mode == "revisit"))
                    r.update(model=name, mode=mode, ckpt_step=ck.get("step"))
                    res.append(r)
                    print(f"{name:14} {mode:7} N={N:3d} acc={r['acc']:.3f} "
                          f"(oldest {r['acc_oldest_quarter']:.3f}, newest "
                          f"{r['acc_newest_quarter']:.3f}) c_auc={r['c_auc']:.3f}", flush=True)
        with open(os.path.join(a.out, f"eval_step{ck.get('step')}.json"), "w") as f:
            json.dump(res, f, indent=1)
        return

    # Baseline: random start, fixed rate, no meta-learning.
    for alpha in a.baseline_alphas:
        theta_b = [t.to(device) for t in init_params(in_dim, a.hidden, a.layers, gen)]
        la_b = [torch.full_like(t, math.log(alpha)) for t in theta_b]
        for mode, N in [(m, N) for m in a.eval_modes for N in a.eval_N]:
            r = retention(theta_b, la_b, eval_pool, N, 10_000 + N, a, device,
                          revisit=(mode == "revisit"))
            r.update(alpha=alpha, mode=mode)
            log["baseline"].append(r)
            print(f"baseline alpha={alpha:g} {mode} N={N:3d} acc={r['acc']:.3f} "
                  f"(oldest {r['acc_oldest_quarter']:.3f}, newest {r['acc_newest_quarter']:.3f}) "
                  f"c_auc={r['c_auc']:.3f}", flush=True)
    save()
    if a.baseline_only:
        return

    theta = [torch.nn.Parameter(t.to(device)) for t in init_params(in_dim, a.hidden, a.layers, gen)]
    log_alpha = [torch.nn.Parameter(torch.full_like(t, math.log(a.alpha0))) for t in theta]
    if a.init_ckpt:
        ck = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for p, t in zip(theta + log_alpha, ck["theta"] + ck["log_alpha"]):
                p.copy_(t.to(device))
        print(f"started from {a.init_ckpt} (step {ck.get('step')})", flush=True)
    opt = torch.optim.Adam(theta + log_alpha, lr=a.lr)
    rng = np.random.RandomState(a.seed + 3)
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        loss = sum(lifetime_loss(theta, log_alpha, train_pool, rng, a, device)
                   for _ in range(a.meta_batch)) / a.meta_batch
        loss.backward()
        torch.nn.utils.clip_grad_norm_(theta + log_alpha, 1.0)
        opt.step()
        if step % 50 == 0:
            plastic = log_alpha[-(a.plastic_tensors or len(log_alpha)):]
            lr_med = float(torch.cat([la.detach().exp().flatten() for la in plastic]).median())
            log["train"].append(dict(step=step, loss=float(loss.detach()), alpha_median=lr_med,
                                     seconds=time.time() - t0))
            print(f"step {step} loss {float(loss.detach()):.4f} alpha_med {lr_med:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            Ns = a.eval_N if step == a.steps else a.eval_N[:1]
            for mode, N in [(m, N) for m in a.eval_modes for N in Ns]:
                r = retention(theta, log_alpha, eval_pool, N, 10_000 + N, a, device,
                              revisit=(mode == "revisit"))
                r.update(step=step, mode=mode)
                log["eval"].append(r)
                print(f"eval step {step} {mode} N={N:3d} acc={r['acc']:.3f} "
                      f"(oldest {r['acc_oldest_quarter']:.3f}, newest {r['acc_newest_quarter']:.3f}) "
                      f"c_written={r['c_written']:.2f} c_unseen={r['c_unseen']:.2f} "
                      f"c_auc={r['c_auc']:.3f}", flush=True)
            torch.save({"theta": [t.detach().cpu() for t in theta],
                        "log_alpha": [t.detach().cpu() for t in log_alpha],
                        "args": vars(a), "step": step}, os.path.join(a.out, "meta_memory.pt"))
            save()


if __name__ == "__main__":
    main()
