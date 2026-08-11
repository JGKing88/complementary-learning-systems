"""The lag is a readable displacement signal: check its law.

Theory: facing North from (x, y), ray i reads W(x + D*tan(theta_i)) with
D = size - 0.5 - y. A translation dx slides the read by dx, which near the cone
centre is a shift of dx / (D * dtheta) ray indices, dtheta = 2*half/n_rays.

    predicted lag = dx / (D * dtheta)

So lag should be LINEAR in dx and INVERSE in D. If both hold, cross-correlating
two views recovers the parallel displacement, and the dilation recovers D.
"""
import sys
import numpy as np

from analysis.scaffold_experiments.wall_resolution import (
    raycast_hires, wall_pattern,
)

SIZE = 8


def view(wc, res, x, y, n_rays):
    return raycast_hires(wc, SIZE, res, np.array([x]), np.array([y]),
                         np.array([0.0]), n_rays)[0]


def best_lag(a, b, max_lag):
    """Lag maximising normalised overlap correlation, and that correlation."""
    best, bl = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            u, v = a[lag:], b[:len(b) - lag] if lag else b
        else:
            u, v = a[:lag], b[-lag:]
        if len(u) < 40:
            continue
        c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        if c > best:
            best, bl = c, lag
    return bl, best


def sweep(res, sigma, n_rays, n_seeds=12):
    dtheta = 2 * np.deg2rad(60.0) / n_rays

    print(f"\n  res={res} sigma={sigma} n_rays={n_rays}   dtheta={dtheta:.5f} rad/ray")
    print(f"\n  LAG vs dx      (y=2.0, so D={SIZE - 0.5 - 2.0})")
    print(f"    {'dx':>6} {'measured':>9} {'predicted':>10} {'peak corr':>10}")
    D = SIZE - 0.5 - 2.0
    for dx in (0.25, 0.5, 1.0, 1.5, 2.0):
        lags, cors = [], []
        for s in range(n_seeds):
            wc = wall_pattern(np.random.RandomState(s), res, sigma)
            a = view(wc, res, 3.0, 2.0, n_rays)
            b = view(wc, res, 3.0 + dx, 2.0, n_rays)
            L, c = best_lag(a, b, n_rays // 2)
            lags.append(L); cors.append(c)
        print(f"    {dx:6.2f} {np.mean(lags):9.1f} {dx / (D * dtheta):10.1f} "
              f"{np.mean(cors):10.3f}")

    print(f"\n  LAG vs distance-to-wall D   (dx = 1.0)")
    print(f"    {'D':>6} {'measured':>9} {'predicted':>10} {'peak corr':>10}")
    for y in (1.0, 2.0, 3.0, 4.0, 5.0):
        D = SIZE - 0.5 - y
        lags, cors = [], []
        for s in range(n_seeds):
            wc = wall_pattern(np.random.RandomState(s), res, sigma)
            a = view(wc, res, 3.0, y, n_rays)
            b = view(wc, res, 4.0, y, n_rays)
            L, c = best_lag(a, b, n_rays // 2)
            lags.append(L); cors.append(c)
        print(f"    {D:6.1f} {np.mean(lags):9.1f} {1.0 / (D * dtheta):10.1f} "
              f"{np.mean(cors):10.3f}")


print("Does the correlation lag follow the predicted law?")
sweep(res=1, sigma=0.0, n_rays=240)
sweep(res=1, sigma=0.0, n_rays=60)

print("\n\nShape of the correlation curve (res=1, n_rays=240, dx=1, y=2):")
wc = wall_pattern(np.random.RandomState(0), 1, 0.0)
a = view(wc, 1, 3.0, 2.0, 240)
b = view(wc, 1, 4.0, 2.0, 240)
row = []
for lag in range(0, 61, 5):
    u, v = a[lag:], b[:len(b) - lag] if lag else b
    row.append((lag, float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))))
print("    lag  " + " ".join(f"{l:6d}" for l, _ in row))
print("    cor  " + " ".join(f"{c:6.2f}" for _, c in row))
