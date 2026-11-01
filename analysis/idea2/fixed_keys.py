"""Reference points for idea 2 with no meta-learning: fixed canvas keys (plan §6.4).

    python -m analysis.idea2.fixed_keys

Keys are the canvas values, centred by the mean value per absolute-direction
slot (from other envs' views), masked to the slots a view fills, and scaled to
unit length. An env's stored key is the mean over one 40-step walk.

1. A linear head, written once per env by an exact delta rule on the stored
   key, then read from fresh views ("once" retention at N = 6, 30, 100).
2. The same linear head stored all at once by least squares, so every stored
   key maps exactly to its goal and nothing is forgotten. If it matches 1, the
   limit is the read, not forgetting.
3. Nearest neighbour on the same keys: return the goal of the stored key with
   the highest cosine (idea 1's argmax read).
4. Why 1 falls short of 3. A linear head reads sum_j (k . k_j) y_j, so a view's
   cosine with its own env's key is the signal and the cosines with the other
   N-1 envs' keys add up as noise, growing as their root sum of squares.
"""
from __future__ import annotations

import numpy as np

from hopfield_nav.memory.plastic import decode_cells
from analysis.idea2.lifetimes import LAMBDAS, EnvPool, Placed


def slot_means(pool, rng, views=32):
    s, n = np.zeros(180), np.zeros(180)
    for i in range(len(pool)):
        c, _ = Placed(pool, i, rng).test_views(rng, views)
        s += (c[:, :180] * c[:, 180:]).sum(0)
        n += c[:, 180:].sum(0)
    return s / np.maximum(n, 1)


def keys_of(c, mu):
    k = (c[:, :180] - mu) * c[:, 180:]
    return k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-8)


def unit_pairs(y):
    p = y.reshape(len(y), 6, 2)
    return (p / (np.linalg.norm(p, axis=2, keepdims=True) + 1e-8)).reshape(len(y), 12)


def hits(pred, env):
    cell = decode_cells(pred, env.cells_global, LAMBDAS)
    return float((np.linalg.norm(env.cells_local[cell] - env.goal, axis=1) <= 1.5).mean())


def main():
    pool = EnvPool(300, seed=2)                    # meta_train's eval pool at seed 0
    mu = slot_means(EnvPool(100, seed=1), np.random.RandomState(0))
    for N in (6, 30, 100):
        rng = np.random.RandomState(10_000 + N)
        envs = [Placed(pool, i, rng) for i in rng.choice(len(pool), size=N, replace=False)]
        H, stored = np.zeros((12, 180)), []
        for e in envs:
            k = keys_of(e.visit(rng, 40), mu).mean(0)
            k /= np.linalg.norm(k) + 1e-8
            stored.append(k)
            H += np.outer(e.target[0] - H @ k, k)          # one exact delta-rule write
        stored = np.array(stored)
        # Least squares: H_all k_j = y_j exactly for every stored key.
        H_all = np.array([e.target[0] for e in envs]).T @ np.linalg.pinv(stored.T)
        lin, lsq, nn, own, other = [], [], [], [], []
        for j, e in enumerate(envs):
            c, _ = e.test_views(rng, 64)
            k = keys_of(c, mu)
            cos = k @ stored.T
            lin.append(hits(unit_pairs(k @ H.T), e))
            lsq.append(hits(unit_pairs(k @ H_all.T), e))
            nn.append(hits(np.array([envs[m].target[0] for m in cos.argmax(1)]), e))
            own.append(cos[:, j])
            other.append(np.delete(cos, j, axis=1))
        own, other = np.concatenate(own), np.concatenate(other)
        q = max(1, N // 4)
        print(f"N={N:3d}: delta rule {np.mean(lin):.3f} (oldest {np.mean(lin[:q]):.3f}, newest "
              f"{np.mean(lin[-q:]):.3f}) | all at once {np.mean(lsq):.3f} | "
              f"nearest neighbour {np.mean(nn):.3f} | own-key cos "
              f"{own.mean():.3f}, other-key rms cos {np.sqrt((other ** 2).mean()):.3f}, "
              f"crosstalk noise {np.sqrt(((other ** 2).sum(1)).mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
