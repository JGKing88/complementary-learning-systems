"""Local maxima of the similarity k(delta) on the integer grid within 2r and 3r (r = 16).

A cell other than delta = 0 with no 4-neighbour of higher similarity is a trap for
gradient following. Also: strict decrease along 16 integer rays.

Docs: EXPERIMENTS_IDEAL_ENCODER.md §25.2. Run on a compute node, from the repo root:
    python analysis/hopfield_probe/kernel_scaling/local_maxima.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
L, r = 1716, 16.0
R3 = int(3 * r)
gx, gy = np.meshgrid(np.arange(-R3 - 1, R3 + 2), np.arange(-R3 - 1, R3 + 2), indexing="ij")
G = np.stack([gx.ravel(), gy.ravel()], 1)
RAD = np.hypot(gx, gy)
DIRS = [(a, b) for a in range(0, 4) for b in range(-3, 4) if (a, b) != (0, 0) and np.gcd(a, abs(b)) == 1
        and (a > 0 or b > 0)]


def kernel_grid(n, w2):
    k = np.zeros(len(G))
    for i in range(0, len(n), 2000):
        k += (w2[None, i:i + 2000] * np.cos(2 * np.pi * (G @ n[i:i + 2000].T) / L)).sum(1)
    return (k / w2.sum()).reshape(gx.shape)


def check(tag, n, w2):
    K = kernel_grid(n, w2)
    c = K[1:-1, 1:-1]
    nb = np.max(np.stack([K[2:, 1:-1], K[:-2, 1:-1], K[1:-1, 2:], K[1:-1, :-2]]), 0)
    rad = RAD[1:-1, 1:-1]
    trap = (c >= nb) & (rad > 0)
    t2, t3 = int((trap & (rad <= 2 * r)).sum()), int((trap & (rad <= 3 * r)).sum())
    first3 = rad[trap & (rad <= 3 * r)].min() if t3 else None
    rise = []
    for a, b in DIRS:
        s = np.arange(0, int(3 * r / np.hypot(a, b)) + 1)
        D = np.stack([s * a, s * b], 1)
        k = (w2[None, :] * np.cos(2 * np.pi * (D @ n.T) / L)).sum(1) / w2.sum()
        up = np.nonzero(np.diff(k) >= 0)[0]
        if len(up):
            rise.append((up[0] + 1) * np.hypot(a, b))
    print(f"{tag:50s} traps within 2r: {t2}, within 3r: {t3}"
          f"{'' if first3 is None else f' (nearest at {first3:.1f} cells)'}; "
          f"rays rising before 3r: {len(rise)}/{len(DIRS)}"
          f"{'' if not rise else f' (first at {min(rise):.1f} cells)'}", flush=True)


# the trained network: frequencies from its L1 weights, weights from its L3
from analysis.hopfield_probe.gwn_encoder import load_gwn
enc = load_gwn("/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/gwn/online/encoders/conc_clean_eta0.2_late.pt")
Wc, Ws, mod = enc.Wc.numpy(), enc.Ws.numpy(), enc.module.numpy()
lams = enc.lambdas; offs = np.cumsum([0] + [l * l for l in lams]); cf = np.array([L // l for l in lams])
J = np.zeros((Wc.shape[1], 2), int)
for p in range(Wc.shape[1]):
    m = mod[p]; l = lams[m]
    w = (Wc[offs[m]:offs[m + 1], p] + 1j * Ws[offs[m]:offs[m + 1], p]).reshape(l, l)
    J[p] = np.unravel_index((np.abs(np.fft.ifft2(w)) ** 2).argmax(), (l, l))
tr = enc.triples.numpy()
nt = (J[tr] * cf[None, :, None]).sum(1) % L
nt = np.where(nt > L // 2, nt - L, nt)
check("trained network (1,306 units)", nt, enc.w.numpy() ** 2)
del enc, Wc, Ws

for seed in range(10):
    g = np.random.default_rng(seed)
    n = np.rint(g.normal(0, L / (2 * np.pi * r), (512, 2))).astype(int)
    check(f"ideal, 512 drawn from P, seed {seed}", n, np.ones(512))
for seed in range(10):
    g = np.random.default_rng(100 + seed)
    n = g.integers(-858, 859, (400_000, 2))
    o = 2 * np.pi * n / L
    w2 = np.exp(-256 * (2 - np.cos(o[:, 0]) - np.cos(o[:, 1])))
    order = np.argsort(-w2)
    keep = order[: np.searchsorted(np.cumsum(w2[order]) / w2.sum(), 0.995) + 1]
    check(f"uniform 400k + depression weights, seed {seed}", n[keep], w2[keep])
