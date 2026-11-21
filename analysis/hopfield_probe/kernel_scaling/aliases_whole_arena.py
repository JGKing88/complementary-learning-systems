"""Aliases: similarity to the goal over the whole arena (clean codes), via one 2D FFT.

k(delta) = sum_u w_u^2 cos(2 pi n_u . delta / L) / sum w^2, for all L x L displacements.
Reports the largest k beyond 2r and 3r, and far local maxima (4-neighbour, wrapped)
above k(2r) = e^-2 = 0.135 and above 0.25.

Docs: EXPERIMENTS_IDEAL_ENCODER.md §25.3. Run on a compute node, from the repo root:
    python analysis/hopfield_probe/kernel_scaling/aliases_whole_arena.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
L, r = 1716, 16.0
d = np.arange(L); d = np.where(d > L // 2, d - L, d)
RAD = np.hypot(d[:, None], d[None, :])


def kernel_all(n, w2):
    A = np.zeros((L, L), complex)
    np.add.at(A, (n[:, 0] % L, n[:, 1] % L), w2)
    return np.fft.ifft2(A).real * L * L / w2.sum()


def report(tag, n, w2, rows):
    k = kernel_all(n, w2)
    nb = np.max(np.stack([np.roll(k, s, a) for a in (0, 1) for s in (1, -1)]), 0)
    peak = (k > nb)
    far3 = RAD > 3 * r
    row = (k[RAD > 2 * r].max(), k[far3].max(), int((peak & far3 & (k > np.exp(-2))).sum()),
           int((peak & far3 & (k > 0.25)).sum()))
    rows.append(row)
    return row


def units_ideal(m, g):
    return np.rint(g.normal(0, L / (2 * np.pi * r), (m, 2))).astype(int), np.ones(m)


def units_uniform(pool, g):
    n = g.integers(-858, 859, (pool, 2))
    o = 2 * np.pi * n / L
    w2 = np.exp(-256 * (2 - np.cos(o[:, 0]) - np.cos(o[:, 1])))
    order = np.argsort(-w2)
    keep = order[: np.searchsorted(np.cumsum(w2[order]) / w2.sum(), 0.995) + 1]
    return n[keep], w2[keep]


print(f"{'network':40s} {'units':>6s}   max k beyond 2r | beyond 3r | far peaks > 0.135 | far peaks > 0.25")
from analysis.hopfield_probe.gwn_encoder import load_gwn
enc = load_gwn("/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/gwn/online/encoders/conc_clean_eta0.2_late.pt")
Wc, Ws, mod = enc.Wc.numpy(), enc.Ws.numpy(), enc.module.numpy()
lams = enc.lambdas; offs = np.cumsum([0] + [l * l for l in lams]); cf = np.array([L // l for l in lams])
J = np.zeros((Wc.shape[1], 2), int)
for p in range(Wc.shape[1]):
    m = mod[p]; l = lams[m]
    w = (Wc[offs[m]:offs[m + 1], p] + 1j * Ws[offs[m]:offs[m + 1], p]).reshape(l, l)
    J[p] = np.unravel_index((np.abs(np.fft.ifft2(w)) ** 2).argmax(), (l, l))
nt = (J[enc.triples.numpy()] * cf[None, :, None]).sum(1) % L
rows = []
a = report("trained network", nt, enc.w.numpy() ** 2, rows)
print(f"{'trained network':40s} {len(nt):6d}   {a[0]:.3f}           | {a[1]:.3f}     | {a[2]:5d}             | {a[3]:5d}", flush=True)
del enc, Wc, Ws

for kind, sizes in (("ideal", (512, 128, 32)), ("uniform + depression", (400_000, 100_000, 25_000))):
    for size in sizes:
        rows, nu = [], []
        for seed in range(3):
            g = np.random.default_rng(1000 * seed + size % 997)
            n, w2 = units_ideal(size, g) if kind == "ideal" else units_uniform(size, g)
            report(kind, n, w2, rows); nu.append(len(n))
        R = np.array(rows)
        label = kind if kind == "ideal" else f"{kind}, pool {size // 1000}k"
        print(f"{label:40s} {int(np.mean(nu)):6d}   {R[:,0].min():.3f}-{R[:,0].max():.3f}     | "
              f"{R[:,1].min():.3f}-{R[:,1].max():.3f} | {R[:,2].astype(int).tolist()!s:17s} | {R[:,3].astype(int).tolist()}",
              flush=True)
