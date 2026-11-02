"""Idea 2 with a continual-learning write: XdG plasticity gating plus SI (plan §6.4).

    python -m analysis.idea2.xdg_si [--hidden 1024] [--steps 50] [--lr 0.05]

No meta-learning. ``M`` is an MLP (canvas -> hidden -> hidden -> 12 goal phases)
from a random start. Each new env is written once: ``steps`` SGD steps on one
40-step walk's views, labelled with its goal phases, squared error on the raw
phases.

XdG as used here gates plasticity, not activity. On entering a new env a random
fraction ``p`` of each hidden layer's units is drawn, and during that env's
write only the weights into those units, and out of the last layer's chosen
units, may change. Every unit is active when reading, so no env identity is
needed at read time. ``p = 1`` is no gating.

SI (Zenke, Poole and Ganguli 2017) adds c * sum_i Omega_i (theta_i - theta*_i)^2
to each write, where theta* is the weights after the previous write and Omega_i
accumulates, over writes, each weight's share of the loss decrease divided by
its squared total change.

Reported per setting: "once" retention at N = 6, 30, 100 (decoded cell within
1.5 cells, 64 fresh views per env, same envs and views as meta_train's eval),
with the oldest and newest quarter, and the mean accuracy of each env right
after its own write (does a write store at all).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from hopfield_nav.memory.plastic import canvas, decode_cells
from hopfield_nav.world.env import PANORAMA_BINS, cone_offsets, panorama_bins
from analysis.idea2.lifetimes import LAMBDAS, OBS, EnvPool, Placed


def T_(x):
    return torch.from_numpy(np.asarray(x, np.float32))


def omni_key(env, n_dirs=6):
    """Idea 1's store for one env: a 360-degree snapshot from the goal cell,
    as one value per absolute-direction slot (views at n_dirs headings merged)."""
    psi = np.arange(n_dirs) * 2 * np.pi / n_dirs
    g = np.full(n_dirs, 1.0)
    c = canvas(env.pool.views(env.idx, g * env.goal[0], g * env.goal[1], psi), psi)
    filled = c[:, PANORAMA_BINS:].sum(0)
    assert (filled > 0).all(), "the snapshot should cover every direction"
    return ((c[:, :PANORAMA_BINS] * c[:, PANORAMA_BINS:]).sum(0) / filled).astype(np.float32)


def view_from_key(keys, psi):
    """The 120-degree view at heading psi cut out of each 360-degree snapshot."""
    out = np.zeros((len(keys), 2 * PANORAMA_BINS), np.float32)
    for i, (k, p) in enumerate(zip(keys, psi)):
        b = panorama_bins(p + cone_offsets(OBS))
        out[i, b] = k[b]
        out[i, PANORAMA_BINS + b] = 1.0
    return out


def init(hidden, seed):
    gen = torch.Generator().manual_seed(seed)
    dims, ps = [360, hidden, hidden, 12], []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        bound = 1.0 / np.sqrt(d_in)
        ps += [(torch.rand(d_in, d_out, generator=gen) * 2 - 1) * bound,
               (torch.rand(d_out, generator=gen) * 2 - 1) * bound]
    return [p.requires_grad_(True) for p in ps]


def fwd(ps, x, units=None):
    """``units`` = (u1, u2) gates the hidden activity (original XdG)."""
    h = torch.relu(x @ ps[0] + ps[1])
    if units is not None:
        h = h * units[0]
    h = torch.relu(h @ ps[2] + ps[3])
    if units is not None:
        h = h * units[1]
    return h @ ps[4] + ps[5]


def draw_units(hidden, p, rng):
    """The units chosen for one env: a random fraction p of each hidden layer."""
    u1 = torch.zeros(hidden)
    u2 = torch.zeros(hidden)
    u1[torch.from_numpy(rng.choice(hidden, int(hidden * p), replace=False))] = 1
    u2[torch.from_numpy(rng.choice(hidden, int(hidden * p), replace=False))] = 1
    return u1, u2


def plastic_masks(hidden, units):
    """Which weights this env's write may change: those into the chosen units of
    each hidden layer, and those out of the last layer's chosen units."""
    u1, u2 = units
    return [u1[None, :].expand(360, -1), u1, u2[None, :].expand(hidden, -1), u2,
            u2[:, None].expand(-1, 12), torch.zeros(12)]


def accuracy(ps, envs, rng, units=None):
    acc = []
    with torch.no_grad():
        for j, e in enumerate(envs):
            x, _ = e.test_views(rng, 64)
            u = None if units is None else units[j]
            cell = decode_cells(fwd(ps, T_(x), u).numpy(), e.cells_global, LAMBDAS)
            acc.append(float((np.linalg.norm(e.cells_local[cell] - e.goal, axis=1) <= 1.5).mean()))
    return np.array(acc)


def run(pool, N, a, p, c, replay=0, per_env=40):
    rng = np.random.RandomState(10_000 + N)                 # same envs as meta_train's eval
    envs = [Placed(pool, i, rng) for i in rng.choice(len(pool), size=N, replace=False)]
    mrng, crng = np.random.RandomState(1), np.random.RandomState(2)
    ps = init(a.hidden, a.seed)
    Omega = [torch.zeros_like(q) for q in ps]
    star = [q.detach().clone() for q in ps]
    right_after, env_units = [], []
    buf_x, buf_y, rrng = [], [], np.random.RandomState(3)
    for e in envs:
        x, y = T_(e.visit(rng, a.visit_steps)), T_(e.target)
        units = draw_units(a.hidden, p, mrng) if p < 1 else None
        env_units.append(units)
        m = None if units is None else plastic_masks(a.hidden, units)
        u_act = units if a.gate_activity else None
        start = [q.detach().clone() for q in ps]
        omega = [torch.zeros_like(q) for q in ps]
        if replay and buf_x:
            bx, by = (np.stack(buf_x), torch.cat(buf_y)) if a.buffer_mode != "walk" \
                else (torch.cat(buf_x), torch.cat(buf_y))
        for _ in range(a.steps):
            task = ((fwd(ps, x, u_act) - y) ** 2).sum(1).mean()
            if replay and buf_x:
                # Experience replay: stored views of earlier envs, each with its
                # env's goal, weighted equally with the current env's views.
                i = rrng.randint(len(bx), size=replay)
                if a.buffer_mode == "omni":     # a random heading cut from each snapshot
                    xr = T_(view_from_key(bx[i], rrng.uniform(-np.pi, np.pi, replay)))
                elif a.buffer_mode == "goalcell":
                    # Diagnostic: exact views ray-cast from the goal cell at random
                    # headings, the snapshot's information without its 2-degree slots.
                    psi = rrng.uniform(-np.pi, np.pi, replay)
                    xr = T_(np.concatenate([canvas(envs[j].pool.views(
                        envs[j].idx, np.array([float(envs[j].goal[0])]),
                        np.array([float(envs[j].goal[1])]), psi[n:n + 1]), psi[n:n + 1])
                        for n, j in enumerate(i)]))
                else:
                    xr = bx[torch.from_numpy(i)]
                task = task + ((fwd(ps, xr) - by[torch.from_numpy(i)]) ** 2).sum(1).mean()
            grads = torch.autograd.grad(task, ps)
            with torch.no_grad():
                for i, (q, g) in enumerate(zip(ps, grads)):
                    # Gradient step on the task loss, then the exact minimiser of the
                    # SI quadratic around it (a proximal step). An explicit step on
                    # the penalty diverges once 2 lr c Omega > 2, and Omega is huge
                    # for weights that barely moved during an earlier write.
                    k = 2 * a.lr * c * Omega[i]
                    step = (q - a.lr * g + k * star[i]) / (1 + k) - q
                    if m is not None:
                        step = step * m[i]
                    omega[i] -= g * step                   # this weight's share of the loss decrease
                    q += step
        with torch.no_grad():
            for i, q in enumerate(ps):
                Omega[i] += (omega[i] / ((q - start[i]) ** 2 + a.xi)).clamp(min=0)
                star[i] = q.detach().clone()
        if replay and a.buffer_mode != "walk":
            buf_x.append(omni_key(e))       # goalcell mode only uses its length
            buf_y.append(y)
        elif replay:
            keep = torch.from_numpy(rrng.choice(len(x), size=min(per_env, len(x)), replace=False))
            buf_x.append(x[keep])
            buf_y.append(y.expand(len(keep), -1))
        right_after.append(accuracy(ps, [e], crng, [u_act] if a.gate_activity else None)[0])
    acc = accuracy(ps, envs, rng, env_units if a.gate_activity else None)
    q = max(1, N // 4)
    return acc.mean(), acc[:q].mean(), acc[-q:].mean(), float(np.mean(right_after))


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--hidden", type=int, default=1024)
    pa.add_argument("--steps", type=int, default=50)
    pa.add_argument("--lr", type=float, default=0.05)
    pa.add_argument("--xi", type=float, default=1e-3)
    pa.add_argument("--visit_steps", type=int, default=40)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--p", type=float, nargs="+", default=[1.0, 0.2, 0.05])
    pa.add_argument("--c", type=float, nargs="+", default=[0.0, 1.0, 10.0])
    pa.add_argument("--N", type=int, nargs="+", default=[6, 30, 100])
    pa.add_argument("--gate_activity", action="store_true",
                    help="original XdG control: also gate hidden activity with the env's units, "
                         "in the write and in the read, so the read must know the env (an oracle)")
    pa.add_argument("--replay", type=int, nargs="+", default=[0],
                    help="stored views of earlier envs replayed per write step (0 = none)")
    pa.add_argument("--buffer_per_env", type=int, nargs="+", default=[40],
                    help="walk views kept per env for replay")
    pa.add_argument("--buffer_mode", choices=["walk", "omni", "goalcell"], default="walk",
                    help="walk = keep walk views; omni = keep only idea 1's store, one "
                         "360-degree snapshot from the goal cell, and replay random "
                         "120-degree views cut from it (--buffer_per_env is ignored); "
                         "goalcell = diagnostic, exact views from the goal cell")
    pa.add_argument("--threads", type=int, default=4)
    a = pa.parse_args()
    if a.gate_activity and min(a.p) >= 1:
        pa.error("--gate_activity needs some p < 1")
    if a.gate_activity and max(a.replay) > 0:
        pa.error("--replay with --gate_activity is not supported")
    torch.set_num_threads(a.threads)
    pool = EnvPool(300, seed=2)                            # meta_train's eval pool at seed 0
    for p in a.p:
        for c in a.c:
            for R in a.replay:
                Bs = a.buffer_per_env if R and a.buffer_mode == "walk" else [0]
                for B in Bs:
                    for N in a.N:
                        acc, old, new, after = run(pool, N, a, p, c, R, B)
                        tag = "activity-gated (oracle) " if a.gate_activity else ""
                        store = f"buffer/env={B}" if a.buffer_mode == "walk" else f"buffer={a.buffer_mode}"
                        rep = f"replay={R} {store} steps={a.steps} " if R else ""
                        print(f"{tag}{rep}p={p:<5g} c={c:<5g} N={N:3d}: once {acc:.3f} (oldest "
                              f"{old:.2f}, newest {new:.2f}), right after own write {after:.2f}",
                              flush=True)


if __name__ == "__main__":
    main()
