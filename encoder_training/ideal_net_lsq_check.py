"""Non-negative least-squares amplitudes for the ideal draw's fixed frequencies.

min_{a >= 0} sum_pairs ( sum_i a_i cos(omega_i . Delta) - exp(-|Delta|^2 / 2 r^2) )^2
over within-patch pairs (two points uniform in one 50x50 patch, distinct), the
pair distribution Stage A trains on. a_i plays the role of a_i^2 in the code;
equal weights are a_i = 1/512. Solved through the Gram matrix: with
G = F^T F = R^T R (Cholesky), the objective is ||R a - R^-T F^T t||^2 + const.
"""
import math

import numpy as np
import torch
from scipy.optimize import nnls

from analysis.hopfield_probe.ideal_net import IdealNet, load_ideal_net
from encoder_training.ideal_net_diagnostics import decode_frequencies

R_ = "/orcd/pool/003/jackking/cls_runs/results/ideal_net/stageA"
NPOS, r = 1716, 16.0
net0 = IdealNet(r=r)
n = net0.freqs.numpy().astype(np.float64)               # (512, 2), the ideal draw
F_ = n.shape[0]
absn = np.sqrt((n ** 2).sum(1))

# the 23 rows that drifted in integer_s0
tr, _ = load_ideal_net(f"{R_}/integer_s0/ideal_net_ep500.pt")
W = tr.harmonics.weight.detach().numpy()
drift = np.abs(W - np.rint(W)).max(1) >= 0.05

rng = np.random.RandomState(0)
G = np.zeros((F_, F_)); b = np.zeros(F_); tt = 0.0; npairs = 0
for _ in range(10):
    m = 100_000
    a = rng.randint(0, 50, size=(m, 2)); c = rng.randint(0, 50, size=(m, 2))
    d = (a - c).astype(np.float64)
    keep = (d != 0).any(1); d = d[keep]
    X = np.cos(2 * math.pi * d @ n.T / NPOS)             # (m, 512)
    t = np.exp(-(d ** 2).sum(1) / (2 * r * r))
    G += X.T @ X; b += X.T @ t; tt += t @ t; npairs += len(d)

def loss(w):
    return float((w @ G @ w - 2 * w @ b + tt) / npairs)

eq = np.full(F_, 1.0 / F_)
# G is singular (repeated frequencies; n and -n are the same wave), so factor it
# by eigendecomposition and drop the null directions: G = R^T R, y = pinv(R^T) b.
lam, V = np.linalg.eigh(G)
ok = lam > lam.max() * 1e-10
Rm = (np.sqrt(lam[ok])[:, None] * V[:, ok].T)
y = (V[:, ok].T @ b) / np.sqrt(lam[ok])
w_opt, _ = nnls(Rm, y, maxiter=20 * F_)

print(f"pairs {npairs}")
print(f"loss  equal weights 1/512: {loss(eq):.6f}   NNLS optimum: {loss(w_opt):.6f}"
      f"   sum of weights {w_opt.sum():.3f} (equal: 1.000)")
print(f"rows set to zero by NNLS: {(w_opt < 1e-8).sum()}")
rel = w_opt * F_                                          # 1 = the equal weight
print(f"\nrelative weight (1 = equal weight): drifted rows median {np.median(rel[drift]):.2f}"
      f" mean {rel[drift].mean():.2f};  other rows median {np.median(rel[~drift]):.2f}"
      f" mean {rel[~drift].mean():.2f}")
print(f"  (trained network: drifted rows behave at ~0.80-0.85 of full strength)")
print("\nrelative weight by |n| band (median / mean / frac zero, rows)")
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 80)]:
    m = (absn >= lo) & (absn < hi)
    if m.any():
        print(f"  |n| {lo:>2d}–{hi:<2d}: {np.median(rel[m]):5.2f} / {rel[m].mean():5.2f}"
              f" / {np.mean(rel[m] < 1e-6):.2f}   ({m.sum()} rows, {drift[m].sum()} drifted)")

# does the in-patch optimum change the far field?
a = rng.randint(0, NPOS, size=(40_000, 2)); c = rng.randint(0, NPOS, size=(40_000, 2))
X = np.cos(2 * math.pi * (a - c).astype(np.float64) @ n.T / NPOS)
print(f"\nfar-field sd over random scaffold pairs: equal {np.std(X @ eq):.4f}"
      f"   NNLS {np.std(X @ w_opt):.4f}")

# local test: at the equal-weight start, which way does the loss push each weight?
g = 2 * (G @ eq - b) / npairs                 # dL/da_i; positive = wants a_i lower
z = (g - g.mean()) / g.std()
print(f"\nlocal gradient at equal weights (z-scored; positive = loss wants that wave turned DOWN)")
print(f"  drifted rows: median z {np.median(z[drift]):+.2f}, {np.mean(z[drift] > 0):.0%} positive"
      f";   other rows: median z {np.median(z[~drift]):+.2f}, {np.mean(z[~drift] > 0):.0%} positive")
top = np.argsort(-z)[:23]
print(f"  of the 23 rows the gradient most wants turned down, {drift[top].sum()} are the drifted rows"
      f" (chance ~{23 * 23 / 512:.1f})")
print(f"  corr(z, |n|) = {np.corrcoef(z, absn)[0, 1]:+.2f}")
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 80)]:
    m = (absn >= lo) & (absn < hi)
    if m.any():
        print(f"  |n| {lo:>2d}–{hi:<2d}: median z {np.median(z[m]):+.2f}  ({m.sum()} rows)")
