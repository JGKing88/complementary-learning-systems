"""Per-fold metrics: parallelism (cosine of group task vectors) and decodability
(L2 logistic regression balanced accuracy).

Both metrics operate on a *pre-pooled* `(h, phase, arena_id)` triple rather than
a TrialsDataset. The caller pools once, then runs every fold against masks of
the same arrays. This avoids a fresh O(N×D) `np.concatenate` per fold, which
blew up memory on the 100-arena × 100-start runs (LOO with ~1.8M rows × 512
hidden dims).
"""
from __future__ import annotations

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def parallelism_score(
    h: np.ndarray,
    phase: np.ndarray,
    arena_id: np.ndarray,
    train_arenas: set[int] | frozenset[int],
    test_arenas: set[int] | frozenset[int],
) -> float:
    """Group-level cosine similarity of (centroid_exploit - centroid_explore)
    between train and test arena groups."""
    train_mask = np.isin(arena_id, np.asarray(sorted(train_arenas)))
    test_mask = np.isin(arena_id, np.asarray(sorted(test_arenas)))
    p0 = phase == 0
    p1 = phase == 1

    def _centroid(mask):
        if not mask.any():
            return np.full(h.shape[1], np.nan, dtype=np.float32)
        # mean computed in float64 for stability, returned as float32.
        return h[mask].mean(axis=0, dtype=np.float64).astype(np.float32)

    c0_tr = _centroid(train_mask & p0)
    c1_tr = _centroid(train_mask & p1)
    c0_te = _centroid(test_mask & p0)
    c1_te = _centroid(test_mask & p1)
    if (np.isnan(c0_tr).any() or np.isnan(c1_tr).any()
            or np.isnan(c0_te).any() or np.isnan(c1_te).any()):
        return float("nan")
    return _cosine(c1_tr - c0_tr, c1_te - c0_te)


def decodability(
    h: np.ndarray,
    phase: np.ndarray,
    arena_id: np.ndarray,
    train_arenas: set[int] | frozenset[int],
    test_arenas: set[int] | frozenset[int],
    *, C: float = 1.0, max_iter: int = 1000,
    subsample_train: int | None = None,
    seed: int = 0,
) -> float:
    """Balanced accuracy of an L2 LR fit on pooled train and evaluated on
    pooled test.

    `subsample_train` optionally caps the number of training rows (after
    masking) to keep memory bounded on huge LOO pools. Sklearn's LR copies
    the input, so for 1.8M × 512 float32 we briefly hold ~3.6 GB twice.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    train_mask = np.isin(arena_id, np.asarray(sorted(train_arenas)))
    test_mask = np.isin(arena_id, np.asarray(sorted(test_arenas)))
    if not train_mask.any() or not test_mask.any():
        return float("nan")

    h_tr = h[train_mask]
    y_tr = phase[train_mask]
    h_te = h[test_mask]
    y_te = phase[test_mask]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan")

    if subsample_train is not None and h_tr.shape[0] > subsample_train:
        rng = np.random.RandomState(seed)
        idx = rng.choice(h_tr.shape[0], size=subsample_train, replace=False)
        h_tr = h_tr[idx]
        y_tr = y_tr[idx]

    scaler = StandardScaler().fit(h_tr)
    Xtr = scaler.transform(h_tr)
    clf = LogisticRegression(
        penalty="l2", C=C, max_iter=max_iter, solver="lbfgs",
        n_jobs=1, random_state=0,
    ).fit(Xtr, y_tr)
    # Free training-side copies before transforming test (sklearn keeps a ref
    # to the fitted X internally only via the solver, not on `clf`).
    del Xtr, h_tr, y_tr
    Xte = scaler.transform(h_te)
    pred = clf.predict(Xte)
    return float(balanced_accuracy_score(y_te, pred))


def within_arena_baseline(
    h: np.ndarray,
    phase: np.ndarray,
    arena_id: np.ndarray,
    *, test_frac: float = 0.2, seed: int = 0,
    C: float = 1.0, max_iter: int = 1000,
    min_per_phase: int = 5,
) -> list[dict]:
    """Per-arena timestep-level 80/20 split.

    For each arena, randomly partition its timesteps into a train (1-test_frac)
    and test (test_frac) half, then compute parallelism + decodability on
    that split. Returns one row per arena with both scalars; ``exp1`` averages
    across arenas to produce the "Within-arena" bar.

    The within-arena bar is the sanity-check ceiling: if it's near 1.0, the
    decoder reliably picks up phase within an arena, and any cross-arena
    falloff measures generalization. If it's not near 1.0, the per-arena data
    is too noisy or one of the conditions has too few rows in this arena.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)
    rows: list[dict] = []
    for a in np.unique(arena_id):
        m = arena_id == a
        h_a, y_a = h[m], phase[m]
        n0 = int((y_a == 0).sum())
        n1 = int((y_a == 1).sum())
        if min(n0, n1) < min_per_phase:
            rows.append({"arena_id": int(a), "parallelism": float("nan"),
                         "decodability": float("nan"),
                         "n": int(h_a.shape[0]), "skipped": True})
            continue
        n = h_a.shape[0]
        idx = rng.permutation(n)
        n_test = max(1, int(round(test_frac * n)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        h_tr, y_tr = h_a[train_idx], y_a[train_idx]
        h_te, y_te = h_a[test_idx], y_a[test_idx]
        if (len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2):
            rows.append({"arena_id": int(a), "parallelism": float("nan"),
                         "decodability": float("nan"),
                         "n": int(n), "skipped": True})
            continue

        # Parallelism: cosine sim of (centroid_exploit - centroid_explore)
        # between the two within-arena halves.
        c0_tr = h_tr[y_tr == 0].mean(axis=0, dtype=np.float64).astype(np.float32)
        c1_tr = h_tr[y_tr == 1].mean(axis=0, dtype=np.float64).astype(np.float32)
        c0_te = h_te[y_te == 0].mean(axis=0, dtype=np.float64).astype(np.float32)
        c1_te = h_te[y_te == 1].mean(axis=0, dtype=np.float64).astype(np.float32)
        par = _cosine(c1_tr - c0_tr, c1_te - c0_te)

        scaler = StandardScaler().fit(h_tr)
        clf = LogisticRegression(
            penalty="l2", C=C, max_iter=max_iter, solver="lbfgs",
            n_jobs=1, random_state=0,
        ).fit(scaler.transform(h_tr), y_tr)
        dec = float(balanced_accuracy_score(
            y_te, clf.predict(scaler.transform(h_te))))

        rows.append({"arena_id": int(a), "parallelism": par,
                     "decodability": dec, "n": int(n), "skipped": False})
    return rows
