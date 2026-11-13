"""Depression rule: trading kappa against steps at a fixed width (r = 16, kappa T = 128). L = 1716.

Exact weights for one axis: every frequency is present, and a walk of T steps puts T_x ~
Binomial(T, 1/2) steps on this axis, so w^2 = clip(1 - kappa (2 - 2 cos omega), 0)^(2 T_x).
The kernel along the axis is sum w^2 cos(omega d) / sum w^2. Over 300 walks per kappa:
half-width (target r sqrt(2 ln 2) = 18.84), its walk-to-walk spread, and the largest gap from
the Gaussian over 0-120 cells; then the same with exactly T/2 steps on the axis, which isolates
the error of log(1 - u) ~ -u from the error of the random step counts.

Docs: EXPERIMENTS_IDEAL_ENCODER.md §25.8. Run on a compute node, from the repo root:
    python analysis/hopfield_probe/kernel_scaling/kappa_steps.py
"""
import numpy as np

L, r = 1716, 16.0
n = np.arange(L); n = np.where(n > L // 2, n - L, n); w = 2 * np.pi * n / L
d = 2 - 2 * np.cos(w)
D = np.arange(0, 120, 0.25)
gauss = np.exp(-D ** 2 / (2 * r * r)); hw_t = r * np.sqrt(2 * np.log(2))
rng = np.random.default_rng(0)
print(f"target half-width {hw_t:.2f}")
for kap in (0.005, 0.05, 0.5, 1.0, 2.0, 5.0):
    T = round(r * r / (2 * kap))               # kappa T = 128, r = sqrt(2 kappa T) = 16
    hws, errs = [], []
    for s in range(300):
        Tx = rng.binomial(T, 0.5)
        w2 = np.clip(1 - kap * d, 0, None) ** (2 * Tx)
        k = (w2[None, :] * np.cos(np.outer(D, w))).sum(1) / w2.sum()
        hws.append(np.interp(0.5, k[::-1], D[::-1])); errs.append(np.abs(k - gauss).max())
    hws = np.array(hws)
    w2a = np.clip(1 - kap * d, 0, None) ** T
    ka = (w2a[None, :] * np.cos(np.outer(D, w))).sum(1) / w2a.sum()
    print(f"kappa {kap:5.3f}  T {T:6d}  half-width {hws.mean():.2f} +- {hws.std():.2f} "
          f"({100 * hws.std() / hw_t:.1f}%)  mean max|k-G| {np.mean(errs):.3f}   "
          f"with exactly T/2 steps per axis: max|k-G| {np.abs(ka - gauss).max():.4f}, "
          f"half-width {np.interp(0.5, ka[::-1], D[::-1]):.2f}", flush=True)
