"""Rewire instead of prune: each neuron redraws its frequency when eliminated; a frequency that
survives T steps since it was drawn becomes permanent. L = 1716, r = 16, 512 neurons.

Each walk step is along axis 0 or 1 (random); a live neuron with frequency n changes by
d = 2 - 2 cos(2 pi n_axis / L) and is eliminated with probability kappa' d. kappa' T = r^2.
Reports steps until 50 / 90 / 100% of neurons are permanent, tries per neuron, and the
kernel of the permanent frequencies (equal weights): half-width and largest alias beyond 3r.

Docs: EXPERIMENTS_IDEAL_ENCODER.md §25.7. Run on a compute node, from the repo root:
    python analysis/hopfield_probe/kernel_scaling/rewire.py
"""
import numpy as np
L, r, N = 1716, 16.0, 512
d = np.arange(L); d = np.where(d > L // 2, d - L, d)
RAD = np.hypot(d[:, None], d[None, :])
print(f"target half-width {r*np.sqrt(2*np.log(2)):.2f}; ideal with 512 frequencies has alias about 0.12")
for kap in (0.01, 0.05, 0.2):
    T = int(round(r * r / kap))
    g = np.random.default_rng(0)
    n = g.integers(-858, 859, (N, 2))
    age = np.zeros(N, int); frozen = np.zeros(N, bool); tries = np.ones(N, int)
    t, marks = 0, {}
    while not frozen.all() and t < 5_000_000:
        ax = g.integers(2)
        live = ~frozen
        dd = 2 - 2 * np.cos(2 * np.pi * n[:, ax] / L)
        kill = live & (g.random(N) < kap * dd)
        k = np.nonzero(kill)[0]
        if len(k):
            n[k] = g.integers(-858, 859, (len(k), 2)); age[k] = 0; tries[k] += 1
        surv = live & ~kill
        age[surv] += 1
        frozen |= surv & (age >= T)
        t += 1
        f = frozen.mean()
        for q in (0.5, 0.9):
            if f >= q and q not in marks:
                marks[q] = t
    A = np.zeros((L, L), complex)
    np.add.at(A, (n[frozen, 0] % L, n[frozen, 1] % L), 1.0)
    kk = np.fft.ifft2(A).real * L * L / frozen.sum()
    row = kk[0, :64]
    half = np.interp(0.5, row[::-1], np.arange(64)[::-1])
    print(f"kappa' {kap:4.2f} T {T:6d} | steps to 50% {marks.get(0.5)}, 90% {marks.get(0.9)}, 100% {t} | "
          f"tries per neuron: median {int(np.median(tries))}, mean {tries.mean():.0f} | "
          f"half-width {half:5.2f}, alias {kk[RAD > 3*r].max():.3f} ({frozen.sum()} permanent)", flush=True)
