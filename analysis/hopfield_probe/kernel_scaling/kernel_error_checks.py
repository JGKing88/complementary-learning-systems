"""Numerical checks for docs/kernel_error.tex (kernel error from finite M, independent noise, shared
grid-phase noise). Frequency-level, CPU only; run on a compute node.

Code: M waves, integer frequencies n ~ round(N(0, sigma_n^2)) per axis, sigma_n = L / (2 pi r),
z(p) = M^{-1/2} [cos, sin](2 pi n . p / L). K_M(D) = (1/M) sum_k cos(2 pi n_k . D / L).

Part 1  finite M: sd of K_M(D) - k(D) against (1 - k(D)^2) / sqrt(2M); relative sd of the
        curvature against sqrt(2/M); largest far |K| (beyond 3r) against the Rice (1D) and
        Euler-characteristic (2D) prediction u* / sqrt(2M).
Part 2  independent noise: greedy similarity climber (reads K at p-1, p, p+1 with fresh noise of sd
        s / sqrt(M), moves to the largest); rms distance from the goal against r^2 s / sqrt(M).
Part 3  shared grid-phase noise: module phases read as p + e_l (cells); peak similarity to a clean
        goal against sinc(e11 - e13) sinc(e12 - e13) for the representative choice with
        sum_l j_l / lam_l = n / L, and the same consistent error under other choices.
Part 4  edge of the monotone region: where the true slope equals one sd of the frozen error's
        slope, against the first non-monotone point in 1D and the nearest trap in 2D.
"""
import numpy as np
from scipy.optimize import brentq

L, r0 = 1716, 16.0
rng = np.random.default_rng(1)


def freqs(M, r, d=1):
    s = L / (2 * np.pi * r)
    return np.rint(rng.normal(0, s, (M, d))).astype(int)


def K1d(n):
    c = np.bincount(n[:, 0] % L, minlength=L).astype(float)
    return np.real(np.fft.fft(c)) / len(n)


def K2d(n):
    c = np.zeros((L, L))
    np.add.at(c, (n[:, 0] % L, n[:, 1] % L), 1.0)
    return np.real(np.fft.fft2(c)) / len(n)


def kg(d, r):
    return np.exp(-np.asarray(d, float) ** 2 / (2 * r * r))


print("== Part 1a: sd of K_M(D) - k(D), 1D, r = 16, 400 seeds")
D = np.array([2, 4, 8, 16, 32, 64, 400])
for M in (32, 512):
    errs = np.array([K1d(freqs(M, r0))[D] - kg(D, r0) for _ in range(400)])
    pred = (1 - kg(D, r0) ** 2) / np.sqrt(2 * M)
    print(f"M={M:4d}  D      " + " ".join(f"{x:7d}" for x in D))
    print(f"        measured " + " ".join(f"{x:7.4f}" for x in errs.std(0)))
    print(f"        formula  " + " ".join(f"{x:7.4f}" for x in pred))
    print(f"        mean err " + " ".join(f"{x:7.4f}" for x in errs.mean(0)))

print("== Part 1b: relative sd of the curvature (1/M) sum w^2 r^2, 1D")
for M in (8, 32, 512):
    c = np.array([((2 * np.pi * freqs(M, r0)[:, 0] / L) ** 2).mean() * r0 ** 2 for _ in range(2000)])
    print(f"M={M:4d}  mean {c.mean():.3f}  rel sd {c.std() / c.mean():.3f}  sqrt(2/M) {np.sqrt(2 / M):.3f}")

print("== Part 1c: largest far |K| (|D| > 3r)")
Tfar = 2 * (L / 2 - 3 * r0)                                    # K is even; both signs counted
u1 = np.sqrt(2 * np.log(Tfar / (2 * np.pi * r0)))
V2 = L * L - np.pi * (3 * r0) ** 2
u2 = brentq(lambda u: V2 / ((2 * np.pi) ** 1.5 * r0 ** 2) * u * np.exp(-u * u / 2) - 1, 1, 10)
print(f"u* 1D {u1:.3f}   u* 2D {u2:.3f}")
d1 = np.minimum(np.arange(L), L - np.arange(L))
for M in (32, 128, 512, 2048):
    m = [np.abs(K1d(freqs(M, r0))[d1 > 3 * r0]).max() for _ in range(200)]
    print(f"1D M={M:5d}  mean {np.mean(m):.4f}  median {np.median(m):.4f}  pred {u1 / np.sqrt(2 * M):.4f}")
dx = np.minimum(np.arange(L), L - np.arange(L))
far2 = (dx[:, None] ** 2 + dx[None, :] ** 2) > (3 * r0) ** 2
for M in (32, 512):
    m = [np.abs(K2d(freqs(M, r0, 2))[far2]).max() for _ in range(12)]
    print(f"2D M={M:5d}  mean {np.mean(m):.4f}  median {np.median(m):.4f}  pred {u2 / np.sqrt(2 * M):.4f}")

print("== Part 2: greedy similarity climber with fresh read noise, 1D")


def climb(Kf, s, M, T=4000, agents=400):
    p = rng.integers(-3, 4, agents)
    off = np.array([-1, 0, 1])
    acc = []
    for t in range(T):
        cand = p[:, None] + off[None, :]
        val = Kf[cand % L] + rng.normal(0, s / np.sqrt(M), cand.shape)
        p = cand[np.arange(agents), val.argmax(1)]
        if t >= T // 2:
            acc.append(((p + L // 2) % L) - L // 2)
    return np.sqrt((np.array(acc) ** 2).mean())


for r, M, s in [(16, 512, 0.2), (16, 512, 0.4), (16, 128, 0.2), (32, 512, 0.1), (8, 32, 0.2),
                (16, 2048, 0.4), (32, 2048, 0.2), (16, 512, 0.8)]:
    scale = r * r * s / np.sqrt(M)
    rms = np.mean([climb(K1d(freqs(M, r)), s, M) for _ in range(3)])
    print(f"r={r:3.0f} M={M:5d} s={s:.2f}  r^2 s/sqrt(M) {scale:5.2f}  rms dist {rms:5.2f}  ratio {rms / scale:.3f}")

print("== Part 3: shared grid-phase noise, 1D per axis, r = 16, M = 512")
lam = np.array([11, 12, 13])
cmul = np.array([6, -1, 7])
Lo = L // lam                                                   # 156, 143, 132
n = freqs(512, r0)[:, 0]
res = (np.outer(n, cmul) % lam)                                 # residues in [0, lam - 1]
sym = np.where(res > lam // 2, res - lam, res)                  # |j| <= lam / 2
m0 = sym.copy()
m0[:, 2] = (n - Lo[0] * sym[:, 0] - Lo[1] * sym[:, 1]) // Lo[2]
assert np.all(m0 @ Lo == n)
for name, J in (("[0, lam-1]", res), ("|j| <= lam/2", sym), ("sum j/lam = n/L", m0)):
    u = J / lam
    print(f"{name:16s} mean u {np.round(u.mean(0), 3)}  cov u*12:\n{np.round(np.cov(u.T) * 12, 3)}")
    print(f"   integer part m = sum u - n/L: values {np.unique(np.rint(u.sum(1) - n / L))}")
pgrid = np.arange(-6, 7)


def peak(J, e):
    phi = 2 * np.pi * (J / lam) @ e                             # (M,)
    K = np.cos(2 * np.pi * np.outer(pgrid, n) / L + phi[None, :]).mean(1)
    return K.max()


print("consistent error e = (x, x, x): peak similarity (prediction 1 for the m = 0 choice)")
for x in (0.1, 0.25, 0.5):
    e = np.full(3, x)
    print(f"  x={x:.2f}  " + "  ".join(f"{nm} {peak(J, e):.3f}" for nm, J in
                                       (("res", res), ("sym", sym), ("m0", m0))))
print("random errors, m = 0 choice: peak against sinc(e1-e3) sinc(e2-e3) and the Gaussian form")
for se in (0.1, 0.2, 0.3, 0.5):
    pk, sn, ga = [], [], []
    for _ in range(300):
        e = rng.normal(0, se, 3)
        x, y = e[0] - e[2], e[1] - e[2]
        pk.append(peak(m0, e))
        sn.append(np.sinc(x) * np.sinc(y))
        ga.append(np.exp(-np.pi ** 2 / 6 * (x * x + y * y)))
    pk, sn, ga = map(np.array, (pk, sn, ga))
    print(f"  sd e {se:.2f}  mean peak {pk.mean():.3f}  mean sinc {sn.mean():.3f}  mean gauss {ga.mean():.3f}"
          f"  rms(peak - sinc) {np.sqrt(((pk - sn) ** 2).mean()):.3f}")
print("same, |j| <= lam/2 choice (no m = 0 constraint)")
for se in (0.1, 0.2, 0.3):
    pk = [peak(sym, rng.normal(0, se, 3)) for _ in range(300)]
    print(f"  sd e {se:.2f}  mean peak {np.mean(pk):.3f}")

print("== Part 4: edge of the monotone region (frozen error), r = 16")
print("prediction: rho e^{-rho^2/2} = 1/sqrt(2M), edge = rho r")
for M in (32, 128, 512, 2048):
    rho = brentq(lambda x: x * np.exp(-x * x / 2) - 1 / np.sqrt(2 * M), 1.0, 6.0)
    first1 = []
    for _ in range(300):
        K = K1d(freqs(M, r0))
        dd = np.arange(1, L // 2)
        bad = np.nonzero(K[dd + 1] >= K[dd])[0]
        first1.append(dd[bad[0]] if len(bad) else L // 2)
    near2 = []
    for _ in range(8):
        K = np.fft.fftshift(K2d(freqs(M, r0, 2)))
        c = L // 2
        W = 6 * int(r0)
        sub = K[c - W:c + W + 1, c - W:c + W + 1]
        core = sub[1:-1, 1:-1]
        ismax = np.ones_like(core, bool)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                if a or b:
                    ismax &= core > sub[1 + a:sub.shape[0] - 1 + a, 1 + b:sub.shape[1] - 1 + b]
        ii, jj = np.nonzero(ismax)
        dist = np.hypot(ii + 1 - W, jj + 1 - W)
        dist = dist[dist > 0]
        near2.append(dist.min() if len(dist) else np.inf)
    print(f"M={M:5d}  pred {rho * r0:5.1f}  1D first non-monotone: median {np.median(first1):5.1f} "
          f"q10 {np.percentile(first1, 10):5.1f}   2D nearest trap: median {np.median(near2):5.1f} "
          f"min {np.min(near2):5.1f}")
