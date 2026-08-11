"""Adjacent views are a WARP of each other, not a perturbation.

Facing North from (x, y), ray i reads W(x + D*tan(theta_i)) with D the distance
to the wall. Translating by dx shifts every ray's read by dx; approaching or
receding dilates the sampling. So o(p) and o(p+delta) sample the SAME wall
function through a reparametrised grid.

Zero-lag cosine is therefore the wrong statistic -- it is the pixelwise
correlation of two frames of a panning video. Test 1 asks whether the structure
shows up under the right one (cross-correlation over lag).

Test 2 asks what smoothness is actually FOR. Identifiability says a decoder
exists; it says nothing about how many samples finding one takes. So: decode
position from a single view by nearest neighbour, sweeping how densely the room
was sampled during training, and evaluate at fresh continuous positions.
"""
import sys
import numpy as np

from analysis.scaffold_experiments.wall_resolution import (
    raycast_hires, wall_pattern,
)

SIZE, OBS = 8, 240


def views(wc, res, xs, ys, psi=0.0):
    return raycast_hires(wc, SIZE, res, xs, ys, np.full(len(xs), psi), OBS)


def unit(V):
    return V / np.linalg.norm(V, axis=-1, keepdims=True)


print("TEST 1 -- is the structure there, under the right statistic?")
print("Two cells one apart in x, facing North. Zero-lag cosine vs the best")
print("cosine after sliding one ray-vector against the other.")
print()
print(f"  {'code':>18} {'zero-lag cos':>13} {'best-lag cos':>13} {'lag':>6}")
print("  " + "-" * 54)
for label, res, sigma in (("res=1 (current)", 1, 0.0), ("res=16 sigma=0", 16, 0.0),
                          ("res=16 sigma=8", 16, 8.0)):
    zs, bs, ls = [], [], []
    for seed in range(8):
        wc = wall_pattern(np.random.RandomState(seed), res, sigma)
        for x in (2.0, 3.0, 4.0):
            for y in (1.0, 2.0, 3.0):
                a, b = views(wc, res, np.array([x]), np.array([y]))[0], \
                       views(wc, res, np.array([x + 1]), np.array([y]))[0]
                a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
                zs.append(float(a @ b))
                # slide b against a over integer ray lags
                best, bl = -1.0, 0
                for lag in range(-120, 121):
                    bb = np.roll(b, lag)
                    n = min(len(a), len(bb))
                    sl = slice(max(0, lag), len(a) + min(0, lag))
                    u, v = a[sl], bb[sl]
                    if len(u) < 40:
                        continue
                    c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
                    if c > best:
                        best, bl = c, lag
                bs.append(best)
                ls.append(bl)
    print(f"  {label:>18} {np.mean(zs):13.3f} {np.mean(bs):13.3f} {np.mean(ls):6.1f}")

print()
print()
print("TEST 2 -- what smoothness buys: samples, not possibility.")
print("1-NN decode of position from a single view, trained on n random")
print("continuous positions, tested on 400 fresh ones. Median error, cells.")
print()
rng = np.random.RandomState(0)
test_xy = rng.uniform(0, SIZE - 1, size=(400, 2))
print(f"  {'code':>18} " + " ".join(f"n={n:<5}" for n in (64, 256, 1024, 4096)))
print("  " + "-" * 58)
for label, res, sigma in (("res=1 (current)", 1, 0.0), ("res=16 sigma=8", 16, 8.0),
                          ("res=16 sigma=0", 16, 0.0)):
    errs = []
    for n in (64, 256, 1024, 4096):
        per_seed = []
        for seed in range(4):
            wc = wall_pattern(np.random.RandomState(seed), res, sigma)
            tr = np.random.RandomState(100 + seed).uniform(0, SIZE - 1, size=(n, 2))
            A = unit(views(wc, res, tr[:, 0], tr[:, 1]))
            B = unit(views(wc, res, test_xy[:, 0], test_xy[:, 1]))
            nn = (B @ A.T).argmax(axis=1)
            per_seed.append(np.median(np.linalg.norm(tr[nn] - test_xy, axis=1)))
        errs.append(np.mean(per_seed))
    print(f"  {label:>18} " + " ".join(f"{e:7.3f}" for e in errs))

print()
print("  For reference, the median distance to the nearest training point")
print("  (the floor any decoder could reach):")
floor = []
for n in (64, 256, 1024, 4096):
    tr = np.random.RandomState(100).uniform(0, SIZE - 1, size=(n, 2))
    d = np.linalg.norm(test_xy[:, None, :] - tr[None, :, :], axis=2).min(axis=1)
    floor.append(np.median(d))
print(f"  {'floor':>18} " + " ".join(f"{f:7.3f}" for f in floor))
