"""Update-gate statistics of a trained idea 4 GRU (GRID_MLP_NAV_PLAN §7b.7).

    python -m analysis.idea4.gate_stats <run>/memory.pt

Runs 64 lifetimes of 6 visits on held-out arenas and records the GRU's update
gate z (h_new = z * h_old + (1 - z) * n) at every step: how much of each
unit's value survives a step and a whole later visit, and whether units that
hold their value elsewhere open their gate at the goal flag (a gated write).
"""
import sys
import types

import torch

from analysis.idea2.lifetimes import EnvPool
from analysis.idea4 import train_experience as te
from analysis.idea4.gpu_views import GpuPool, place

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
a = types.SimpleNamespace(**ck["args"])
a.chrono = 0
dev = torch.device("cpu")
model = te.build(a, dev)
model.load_state_dict(ck["model"])
model.eval()
H, B, N, S = a.hidden, 64, 6, a.visit_steps
gp = GpuPool(EnvPool(100, seed=a.seed + 2), dev)
g = torch.Generator().manual_seed(0)
W_i, W_h = model.gru.weight_ih_l0[H:2 * H], model.gru.weight_hh_l0[H:2 * H]
b = model.gru.bias_ih_l0[H:2 * H] + model.gru.bias_hh_l0[H:2 * H]
env = torch.rand(B, len(gp), generator=g).argsort(1)[:, :N]
off, goal, _ = place(B, N, dev, g)
h = model.init_state(B, dev)
zs, flags = [], []
with torch.no_grad():
    for t in range(N):
        path, psi, first, _ = te.visit_path(goal[:, t], a, dev, g)
        x = gp.canvas(env[:, t].expand(S, B), path[..., 0], path[..., 1], psi)
        ex = te.extra_of(off[:, t].expand(S, B, 2), path, first)
        for s in range(S):
            inp = torch.cat([model.enc(x[s]), ex[s]], -1)
            z = torch.sigmoid(inp @ W_i.T + h[-1] @ W_h.T + b)        # (B, H)
            zs.append(z)
            flags.append(first[s])
            _, _, h = model(x[s:s + 1], h, extra=ex[s:s + 1])
z = torch.stack(zs)                       # (steps, B, H)
fl = torch.stack(flags)                   # (steps, B)
print(f"{sys.argv[1].split('/')[-2]}  step {ck['step']}")
print(f"mean z: all steps {z.mean():.3f} | steps at the goal flag {z[fl].mean():.3f} | "
      f"other steps {z[~fl].mean():.3f}")
for q in (0.5, 0.9, 0.99, 0.999):
    print(f"  fraction of (unit, step) pairs with z > {q}: {(z > q).float().mean():.3f}")
# Survival of a unit's value across a whole later visit: product of z over its 40 steps.
vis = z[S:].reshape(N - 1, S, B, H).prod(1)          # visits 2..N
per_unit = vis.mean((0, 1))                          # average over visits and lifetimes
for q in (0.01, 0.1, 0.5, 0.9):
    print(f"  units keeping > {q} of their value through one 40-step visit: "
          f"{(per_unit > q).float().mean():.3f}")
print(f"  median survival through one visit {per_unit.median():.2e}; "
      f"through five visits (product) {(vis.prod(0).mean(0) > 0.1).float().mean():.3f} "
      f"of units keep > 0.1")
# Do any units open their gate specifically at the goal flag (a gated write)?
zf = z[fl].mean(0)                         # (H,) mean z at flag steps
zn = z[~fl].mean(0)                        # (H,) mean z elsewhere
holder = zn > 0.99
print(f"  holder units (z > 0.99 away from the flag): {int(holder.sum())} of {H}")
for thr in (0.9, 0.5, 0.1):
    print(f"    of these, z at the flag below {thr}: {int((holder & (zf < thr)).sum())}")
drop = zn - zf
print(f"  largest drops in z at the flag (z elsewhere -> z at flag): " + ", ".join(
    f"{float(zn[i]):.3f}->{float(zf[i]):.3f}" for i in drop.argsort(descending=True)[:8]))
# Gated-write units: shut away from the flag, open at it.
for hold, open_ in ((0.95, 0.3), (0.97, 0.3), (0.98, 0.5)):
    w = (zn > hold) & (zf < open_)
    keep = (zn[w] ** 200).mean() if w.any() else torch.tensor(0.)
    print(f"  write units (z > {hold} elsewhere, < {open_} at the flag): {int(w.sum())}; "
          f"their mean z elsewhere {float(zn[w].mean()) if w.any() else 0:.3f}, "
          f"fraction kept over 200 steps {float(keep):.2f}")
