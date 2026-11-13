"""Retention of an experience-trained idea 4 memory, averaged over many
lifetimes (GRID_MLP_NAV_PLAN §7b.6).

    python -m analysis.idea4.eval_many --ckpt <run>/memory.pt [--N 1 2 3 6 10 20] [--lifetimes 64]

The in-training test (``train_experience.retention``) writes one lifetime per
N, so at N = 2 it rests on two arenas. Here each N runs ``--lifetimes``
lifetimes in parallel on held-out arenas: N arenas visited once each (a walk
through the goal, flag on the first arrival), then ``--views`` views per arena
at random cells and headings, each with its grid phases. Reads are scored from
copies of the state after the last visit ("copy") and carried in sequence,
arena by arena in visit order ("seq"). A view is correct if the decoded goal
cell is within 1.5 cells of the goal. Also reported: the mean cosine between
decoded and true phases, and accuracy by visit position (oldest first).
"""
from __future__ import annotations

import argparse
import json
import types

import numpy as np
import torch

from analysis.idea2.lifetimes import EnvPool
from analysis.idea4 import train_experience as te
from analysis.idea4.gpu_views import GpuPool, place, random_cells
from analysis.idea4.train_memory import angle_cos


@torch.no_grad()
def evaluate(model, gp, a, N, lifetimes, views, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    B, V, S = lifetimes, views, a.visit_steps
    env = torch.rand(B, len(gp), device=device, generator=g).argsort(1)[:, :N]
    off, goal, target = place(B, N, device, g)
    h = model.init_state(B, device)
    for t in range(N):
        path, psi, first, _ = te.visit_path(goal[:, t], a, device, g)
        x = gp.canvas(env[:, t].expand(S, B), path[..., 0], path[..., 1], psi)
        _, _, h = model(x, h, extra=te.extra_of(off[:, t].expand(S, B, 2), path, first))
    cx, cy, cpsi = random_cells((B, N, V), device, g)
    x = gp.canvas(env[:, :, None].expand(B, N, V), cx, cy, cpsi)
    ex = te.extra_of(off[:, :, None].expand(B, N, V, 2), torch.stack([cx, cy], -1),
                     torch.zeros(B, N, V, device=device))
    ph_c, _, _ = model(x.reshape(1, B * N * V, -1), h.repeat_interleave(N * V, dim=1),
                       extra=ex.reshape(1, B * N * V, -1))
    ph_s, _, _ = model(x.reshape(B, N * V, -1).transpose(0, 1), h,
                       extra=ex.reshape(B, N * V, -1).transpose(0, 1))
    ph_c = ph_c.reshape(B, N, V, 12)
    ph_s = ph_s.transpose(0, 1).reshape(B, N, V, 12)
    out = {"N": N}
    for name, ph in (("copy", ph_c), ("seq", ph_s)):
        acc = torch.stack([te._hits(ph[:, j], off[:, j], goal[:, j]) for j in range(N)], 1)
        cos = angle_cos(ph, target[:, :, None]).mean(2)
        out[f"acc_{name}"] = float(acc.mean())
        out[f"cos_{name}"] = float(cos.mean())
        out[f"acc_{name}_by_position"] = acc.mean(0).cpu().numpy().round(3).tolist()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--N", type=int, nargs="+", default=[1, 2, 3, 6, 10, 20])
    p.add_argument("--lifetimes", type=int, default=64)
    p.add_argument("--views", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="write all results as JSON here")
    p.add_argument("--pool", type=int, default=0,
                   help="test on the first this many training arenas instead of held-out ones")
    a_cli = p.parse_args()
    device = torch.device(a_cli.device if torch.cuda.is_available() else "cpu")
    res = []
    gp = None
    for path in a_cli.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = types.SimpleNamespace(**ck["args"])
        a.chrono = 0
        a.enc_kind = getattr(a, "enc_kind", "relu")
        if gp is None:
            gp = GpuPool(EnvPool(a_cli.pool, seed=a.seed + 1) if a_cli.pool
                         else EnvPool(a.n_eval_envs, seed=a.seed + 2), device)   # held-out arenas
        model = te.build(a, device)
        model.load_state_dict(ck["model"])
        model.eval()
        print(f"\n{path}  step {ck['step']}  hidden {a.hidden}")
        for N in [n for n in a_cli.N if n <= len(gp)]:
            r = evaluate(model, gp, a, N, a_cli.lifetimes, a_cli.views, a_cli.seed + N, device)
            r.update(ckpt=path, step=ck["step"], hidden=a.hidden)
            res.append(r)
            pos = r["acc_copy_by_position"]
            print(f"N={N:3d}: copy acc {r['acc_copy']:.3f} (cos {r['cos_copy']:.3f}) | seq acc "
                  f"{r['acc_seq']:.3f} (cos {r['cos_seq']:.3f}) | copy acc oldest {pos[0]:.2f}, "
                  f"newest {pos[-1]:.2f}", flush=True)
    if a_cli.out:
        with open(a_cli.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
