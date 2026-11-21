"""Place fields from a rotation of the ideal code (checks for docs/place_fields_rotation.tex).

2D torus L = 1716, r = 16, M = 512 waves with distinct frequencies (no n = 0, no n_k = +-n_l), so the
2M coordinates are orthogonal functions on the torus and every unit of every rotation has energy
sum_p y(p)^2 = L^2 / (2M).

1. Matched row q = z(c): tuning y(p) = K_M(p - c). Peak, half-max width, energy inside 3r, ripple rms
   outside 3r against sqrt((1 - 2M/D) / (2M)), D = L^2 / (pi r^2).
2. N matched rows on a sqrt(N) x sqrt(N) lattice, made orthonormal by the symmetric (Loewdin) map
   Q = G^{-1/2} Z, G = Z Z^T. Gram eigenvalue range against 1 +- 2 sqrt(N / 2M); per unit: peak at its
   own centre, half-max width, energy fraction inside 3r, largest |y| outside 3r.
"""
import numpy as np

L, r, M = 1716, 16.0, 512
rng = np.random.default_rng(3)
sig_n = L / (2 * np.pi * r)
D = L * L / (np.pi * r * r)

seen, n = set(), []
while len(n) < M:
    v = tuple(np.rint(rng.normal(0, sig_n, 2)).astype(int))
    if v == (0, 0) or v in seen or (-v[0], -v[1]) in seen:
        continue
    seen.add(v)
    n.append(v)
n = np.array(n)
print(f"L={L} r={r} M={M} distinct frequencies, D = L^2/(pi r^2) = {D:.0f}, 2M/D = {2 * M / D:.3f}")

dx = np.minimum(np.arange(L), L - np.arange(L))
R2 = dx[:, None] ** 2 + dx[None, :] ** 2                      # squared torus distance to (0, 0)
inner = R2 <= (3 * r) ** 2


def zcode(c):
    th = 2 * np.pi * (n @ np.asarray(c, float)) / L
    return np.concatenate([np.cos(th), np.sin(th)]) / np.sqrt(M)


def tuning(q):
    """y(p) = M^{-1/2} sum_k a_k cos(w_k.p) + b_k sin(w_k.p), q = (a, b) / sqrt(M)-free coefficients."""
    a, b = q[:M], q[M:]
    C = np.zeros((L, L), complex)
    np.add.at(C, (n[:, 0] % L, n[:, 1] % L), (a - 1j * b) / np.sqrt(M))
    return np.real(np.fft.ifft2(C)) * L * L


def stats(y, c):
    y = np.roll(np.roll(y, -int(c[0]), 0), -int(c[1]), 1)         # centre at (0, 0)
    h = y[0, 0]
    e = (y ** 2).sum()
    half = ((y > h / 2) & inner).sum()
    sig = np.sqrt(half / (2 * np.pi * np.log(2)))
    out = y[~inner]
    return h, sig, (y[inner] ** 2).sum() / e, np.abs(out).max(), np.sqrt((out ** 2).mean()), e


print(f"energy per unit predicted L^2/(2M) = {L * L / (2 * M):.1f}")
print("== matched row q = z(c)")
h, sig, fin, mx, rms, e = stats(tuning(zcode((0, 0))), (0, 0))
print(f"peak {h:.3f}  half-max sd {sig:.2f} (r = {r})  energy {e:.1f}  fraction inside 3r {fin:.3f} "
      f"(pi r^2 / (L^2/2M) = {np.pi * r * r * 2 * M / L ** 2:.3f})  ripple rms {rms:.4f} "
      f"(pred {np.sqrt((1 - 2 * M / D) / (2 * M)):.4f})  largest outside 3r {mx:.3f}")

print("== N matched rows on a lattice, symmetric orthonormalisation")
for N in (16, 64, 256, 1024):
    g = int(round(np.sqrt(N)))
    sp = L / g
    cs = np.array([(i * sp, j * sp) for i in range(g) for j in range(g)])
    cs = np.rint(cs).astype(int)
    Z = np.array([zcode(c) for c in cs])                         # (N, 2M)
    G = Z @ Z.T
    ev, V = np.linalg.eigh(G)
    Q = (V / np.sqrt(np.clip(ev, 1e-12, None))) @ V.T @ Z         # rows orthonormal
    orth = np.abs(Q @ Q.T - np.eye(N)).max()
    pick = rng.choice(N, size=min(N, 48), replace=False)
    S = np.array([stats(tuning(Q[i]), cs[i]) for i in pick])
    lo, hi = (1 - np.sqrt(N / (2 * M))) ** 2, (1 + np.sqrt(N / (2 * M))) ** 2
    print(f"N={N:5d} spacing {sp:6.1f} ({sp / r:.2f} r)  Gram eig [{ev.min():.3f}, {ev.max():.3f}] "
          f"(MP [{lo:.3f}, {hi:.3f}])  max|QQ^T - I| {orth:.1e}")
    print(f"        over {len(pick)} units: peak {S[:, 0].mean():.3f} (min {S[:, 0].min():.3f})  "
          f"half-max sd {S[:, 1].mean():.2f}  inside 3r {S[:, 2].mean():.3f}  "
          f"largest outside 3r {S[:, 3].mean():.3f} (max {S[:, 3].max():.3f})  ripple rms {S[:, 4].mean():.4f}  "
          f"energy {S[:, 5].mean():.1f}")

print("== N = 2M centres on a jittered lattice (generic positions, so G is full rank and Q is exactly orthogonal)")
for jit in (0.25, 0.5):
    N = 2 * M
    g = int(round(np.sqrt(N)))
    sp = L / g
    base = np.array([(i * sp, j * sp) for i in range(g) for j in range(g)])
    cs = np.rint(base + rng.uniform(-jit * sp, jit * sp, base.shape)).astype(int) % L
    Z = np.array([zcode(c) for c in cs])
    ev, V = np.linalg.eigh(Z @ Z.T)
    Q = (V / np.sqrt(ev)) @ V.T @ Z
    orth = np.abs(Q @ Q.T - np.eye(N)).max()
    pick = rng.choice(N, size=96, replace=False)
    S = np.array([stats(tuning(Q[i]), cs[i]) for i in pick])
    q = lambda a: np.percentile(a, [5, 50, 95])
    print(f"jitter +-{jit:.2f} spacing  Gram eig [{ev.min():.2e}, {ev.max():.3f}]  max|QQ^T - I| {orth:.1e}")
    for k, nm in enumerate(["peak", "half-max sd", "inside 3r", "largest outside 3r", "ripple rms"]):
        print(f"   {nm:20s} 5/50/95%: " + " ".join(f"{x:.4f}" for x in q(S[:, k])))
