"""Is idea 4's limit storage or retrieval? Probes on the frozen state
(GRID_MLP_NAV_PLAN §7b.6).

    python -m analysis.idea4.probe_storage --ckpt <run>/memory.pt [--N 6 10 20]

After N visits (as in eval_many), the final hidden state h is frozen and fresh
read-outs are trained on it to decode each visited arena's goal:
  ordinal  [h, one-hot visit position]      is goal k in h at all?
  key      [h, the arena's whole panorama]  retrievable with a perfect key?
  view     [h, one live view + its cell's grid phases]
           the model's own task, with a read-out trained to convergence
They are compared with the model's own read (one GRU step on the view, then
its MLP). Train and test use separate lifetimes on the held-out arenas; goals
and offsets are drawn fresh per lifetime, so an arena's identity says nothing
about where its goal is. A view counts as correct within 1.5 cells.
"""
from __future__ import annotations

import argparse
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.idea2.lifetimes import EnvPool
from analysis.idea4 import train_experience as te
from analysis.idea4.gpu_views import GpuPool, goal_phases, place, random_cells
from analysis.idea4.train_memory import angle_cos

N_MAX = 20


@torch.no_grad()
def lifetimes(model, gp, a, N, B, V, g, device, chunk=1024):
    """B lifetimes of N visits; final state, arenas, offsets, goals, goal
    phases, and V random test views per arena (canvas + cell grid phases)."""
    out = {k: [] for k in ("h", "env", "off", "goal", "tgt", "x", "ex")}
    S = a.visit_steps
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        env = torch.rand(b, len(gp), device=device, generator=g).argsort(1)[:, :N]
        off, goal, tgt = place(b, N, device, g)
        h = model.init_state(b, device)
        for t in range(N):
            path, psi, first, _ = te.visit_path(goal[:, t], a, device, g)
            x = gp.canvas(env[:, t].expand(S, b), path[..., 0], path[..., 1], psi)
            _, _, h = model(x, h, extra=te.extra_of(off[:, t].expand(S, b, 2), path, first))
        cx, cy, cpsi = random_cells((b, N, V), device, g)
        x = gp.canvas(env[:, :, None].expand(b, N, V), cx, cy, cpsi)
        ex = te.extra_of(off[:, :, None].expand(b, N, V, 2), torch.stack([cx, cy], -1),
                         torch.zeros(b, N, V, device=device))
        for k, v in (("h", h), ("env", env), ("off", off), ("goal", goal), ("tgt", tgt),
                     ("x", x), ("ex", ex)):
            out[k].append(v)
    out["h"] = torch.cat(out["h"], 1)                    # (layers, B, H)
    for k in ("env", "off", "goal", "tgt", "x", "ex"):
        out[k] = torch.cat(out[k], 0)
    return out


def probe_input(kind, d, gp, bi, ni, vi):
    """Rows (bi, ni, vi) of lifetime data -> probe inputs."""
    h = d["h"][-1][bi]
    if kind == "ordinal":
        return torch.cat([h, F.one_hot(ni, N_MAX).float()], -1)
    if kind == "key":
        return torch.cat([h, gp.panos[d["env"][bi, ni]]], -1)
    return torch.cat([h, d["x"][bi, ni, vi], d["ex"][bi, ni, vi, :12]], -1)


def hits_rows(ph, off, goal, chunk=4096):
    """Per-row accuracy for predicted phases (R, 12), offsets and goals (R, 2)."""
    return torch.cat([te._hits(ph[i:i + chunk, None], off[i:i + chunk], goal[i:i + chunk])
                      for i in range(0, len(ph), chunk)])


def train_probe(kind, dtr, gp, N, V, in_dim, a_cli, device):
    net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                        nn.Linear(512, 12)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    B = dtr["h"].shape[1]
    for step in range(a_cli.probe_steps):
        bi = torch.randint(0, B, (a_cli.probe_batch,), device=device)
        ni = torch.randint(0, N, (a_cli.probe_batch,), device=device)
        vi = torch.randint(0, V, (a_cli.probe_batch,), device=device)
        ph = net(probe_input(kind, dtr, gp, bi, ni, vi)).reshape(-1, 6, 2)
        ph = (ph / (ph.norm(dim=-1, keepdim=True) + 1e-6)).reshape(-1, 12)
        loss = (1 - angle_cos(ph, dtr["tgt"][bi, ni])).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def score(net_or_model, kind, dte, gp, N, V, device):
    """Accuracy and cosine by visit position on the test lifetimes."""
    B = dte["h"].shape[1]
    bi, ni, vi = torch.meshgrid(torch.arange(B, device=device), torch.arange(N, device=device),
                                torch.arange(V, device=device), indexing="ij")
    bi, ni, vi = bi.reshape(-1), ni.reshape(-1), vi.reshape(-1)
    if kind == "model":
        x = dte["x"].reshape(1, B * N * V, -1)
        ex = dte["ex"].reshape(1, B * N * V, -1)
        ph, _, _ = net_or_model(x, dte["h"].repeat_interleave(N * V, dim=1), extra=ex)
        ph = ph[0]
    else:
        ph = torch.cat([net_or_model(probe_input(kind, dte, gp, bi[i:i + 8192], ni[i:i + 8192],
                                                 vi[i:i + 8192]))
                        for i in range(0, len(bi), 8192)]).reshape(-1, 6, 2)
        ph = (ph / (ph.norm(dim=-1, keepdim=True) + 1e-6)).reshape(-1, 12)
    tgt = dte["tgt"][bi, ni]
    cos = angle_cos(ph, tgt).reshape(B, N, V).mean((0, 2))
    acc = hits_rows(ph, dte["off"][bi, ni], dte["goal"][bi, ni]).reshape(B, N, V).mean((0, 2))
    return acc.cpu(), cos.cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--N", type=int, nargs="+", default=[6, 10, 20])
    p.add_argument("--train_lifetimes", type=int, default=8192)
    p.add_argument("--test_lifetimes", type=int, default=1024)
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--probe_steps", type=int, default=4000)
    p.add_argument("--probe_batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a_cli = p.parse_args()
    device = torch.device(a_cli.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a_cli.seed)
    gp = None
    for path in a_cli.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = types.SimpleNamespace(**ck["args"])
        a.chrono = 0
        a.enc_kind = getattr(a, "enc_kind", "relu")
        if gp is None:
            gp = GpuPool(EnvPool(a.n_eval_envs, seed=a.seed + 2), device)
        model = te.build(a, device)
        model.load_state_dict(ck["model"])
        model.eval()
        H = a.hidden
        print(f"\n{path}  step {ck['step']}  hidden {H}", flush=True)
        for N in a_cli.N:
            g = torch.Generator(device=device).manual_seed(a_cli.seed + 100 * N)
            V = a_cli.views
            dtr = lifetimes(model, gp, a, N, a_cli.train_lifetimes, V, g, device)
            dte = lifetimes(model, gp, a, N, a_cli.test_lifetimes, V, g, device)
            rows = [("model's own read", score(model, "model", dte, gp, N, V, device))]
            for kind, in_dim in (("ordinal", H + N_MAX), ("key", H + 180), ("view", H + 372)):
                net = train_probe(kind, dtr, gp, N, V, in_dim, a_cli, device)
                rows.append((f"probe: {kind}", score(net, kind, dte, gp, N, V, device)))
            print(f"N = {N}", flush=True)
            for name, (acc, cos) in rows:
                print(f"  {name:18} acc {float(acc.mean()):.3f} (oldest {float(acc[0]):.2f}, "
                      f"newest {float(acc[-1]):.2f}) | cos {float(cos.mean()):.3f} | acc by "
                      f"position " + " ".join(f"{float(v):.2f}" for v in acc), flush=True)
            del dtr, dte


if __name__ == "__main__":
    main()
