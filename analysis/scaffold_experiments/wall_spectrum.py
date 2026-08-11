"""Does a 1/f wall code beat the single-scale Pareto frontier?

Single-scale smoothing picks ONE stripe width, and the two properties we want
sit at opposite ends of that choice: coarse stripes interpolate but alias, fine
stripes identify but cannot interpolate. A 1/f (power-law) code has structure at
every scale at once, so in principle it can carry coarse bands for interpolation
AND sub-cell detail for disambiguation.

Generated in the Fourier domain: amplitude ~ f^(-beta/2), random phase, inverse
FFT, take the sign. beta=0 is white (== sigma=0); beta=2 is heavily red (== a
large sigma); beta~1 is pink, equal power per octave.

The question is not whether pink is good in absolute terms -- it is whether at
MATCHED interpolability it has fewer twins than any single-scale code, i.e.
whether it dominates rather than just moving along the same frontier.
"""
import sys
import numpy as np

from analysis.scaffold_experiments.wall_resolution import raycast_hires

SIZE, OBS, RES = 20, 60, 8
NS = [256, 1024, 4096]


def wall_gauss(size, res, sigma, seed):
    rng = np.random.RandomState(seed)
    raw = rng.randn(4, size * res)
    if sigma > 0:
        half = max(1, int(np.ceil(3 * sigma)))
        t = np.arange(-half, half + 1)
        k = np.exp(-0.5 * (t / sigma) ** 2); k /= k.sum()
        raw = np.stack([np.convolve(np.tile(r, 3), k, "same")
                        [size * res:2 * size * res] for r in raw])
    return np.sign(raw).astype(np.float32)


def wall_pink(size, res, beta, seed):
    """Power-law spectrum: amplitude ~ f^(-beta/2), random phase, sign."""
    L = size * res
    rng = np.random.RandomState(seed)
    n_f = L // 2 + 1
    f = np.arange(n_f, dtype=float)
    amp = np.zeros(n_f)
    amp[1:] = f[1:] ** (-beta / 2.0)
    out = []
    for _ in range(4):
        spec = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, n_f))
        spec[0] = 0.0                       # no DC, keeps +/-1 balanced
        out.append(np.sign(np.fft.irfft(spec, n=L)))
    return np.array(out, dtype=np.float32)


def views(wc, res, xy):
    V = raycast_hires(wc, SIZE, res, xy[:, 0], xy[:, 1], np.zeros(len(xy)), OBS)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def stripe(wc, res):
    w = wc[0]
    return len(w) / max(1, np.sum(w[1:] != w[:-1]) + 1) / res


def twins(make, res, n_seeds=4):
    gx, gy = np.meshgrid(np.arange(SIZE, dtype=float),
                         np.arange(SIZE, dtype=float), indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)
    out = []
    for s in range(n_seeds):
        C = views(make(s), res, cells)
        C = C @ C.T
        np.fill_diagonal(C, -1.0)
        out.append((C > 0.9999).any(axis=1).mean())
    return float(np.mean(out))


rng = np.random.RandomState(0)
test = rng.uniform(0, SIZE - 1, size=(600, 2))
floors = []
for n in NS:
    tr = np.random.RandomState(100).uniform(0, SIZE - 1, size=(n, 2))
    floors.append(np.median(np.linalg.norm(
        test[:, None, :] - tr[None, :, :], axis=2).min(axis=1)))


def interp(make, res, n_seeds=3):
    ratios = []
    for i, n in enumerate(NS):
        per = []
        for s in range(n_seeds):
            wc = make(s)
            tr = np.random.RandomState(100 + s).uniform(0, SIZE - 1, size=(n, 2))
            A, B = views(wc, res, tr), views(wc, res, test)
            nn = (B @ A.T).argmax(axis=1)
            per.append(np.median(np.linalg.norm(tr[nn] - test, axis=1)))
        ratios.append(np.mean(per) / floors[i])
    return ratios


ROWS = [("res=1  (current)", 1, lambda s: wall_gauss(SIZE, 1, 0.0, s))]
for sig in (0.0, 1.0, 2.0, 4.0, 8.0):
    ROWS.append((f"gauss sigma={sig:<4}", RES,
                 lambda s, g=sig: wall_gauss(SIZE, RES, g, s)))
for b in (0.0, 0.5, 1.0, 1.5, 2.0):
    ROWS.append((f"pink  beta={b:<5}", RES,
                 lambda s, bb=b: wall_pink(SIZE, RES, bb, s)))

print(f"{SIZE}x{SIZE}, {OBS} rays, res={RES} for all but the baseline")
print("interpolability = 1-NN error / floor (1.00 is the sampling limit)")
print()
print(f"  {'code':>18} {'stripe':>7} {'twins':>7}   " +
      "  ".join(f"n={n:<5}" for n in NS))
print("  " + "-" * 62)
for label, res, make in ROWS:
    r = interp(make, res)
    print(f"  {label:>18} {stripe(make(0), res):6.2f}c {twins(make, res):6.1%}   " +
          "  ".join(f"{x:6.2f} " for x in r))

print()
print("Pareto check: for each interpolability level, who has the fewest twins?")
pts = []
for label, res, make in ROWS:
    pts.append((label, twins(make, res), interp(make, res)[1]))   # n=1024
pts.sort(key=lambda p: p[2])
print(f"  {'code':>18} {'interp@1024':>12} {'twins':>8}")
best = 1e9
for label, tw, it in pts:
    mark = ""
    if tw < best:
        best, mark = tw, "  <- frontier"
    print(f"  {label:>18} {it:12.2f} {tw:8.1%}{mark}")
