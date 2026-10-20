"""Abstraction geometry of a binary task variable across contexts.

Conditions are (arena, phase). An arena is a context; phase is the variable
whose abstraction is tested. Definitions follow Bernardi et al. 2020 (Cell,
"The geometry of abstraction in the hippocampus and prefrontal cortex"):

- Parallelism score (PS). One coding vector per arena,
  d_a = mean(z | a, phase 1) - mean(z | a, phase 0), on z-scored features.
  PS is the mean cosine over all pairs of arenas. Arenas fix the pairing of
  conditions, so there is no max over pairings. Null: phase labels permuted
  across whole trials within each arena.
- Cross-condition generalisation (CCGP). A linear decoder is trained on k
  arenas and tested on held-out arenas. Bernardi's null is the geometric
  random model: every condition centroid is moved to a random isotropic
  Gaussian position, rescaled to keep the total centroid variance, and the
  within-condition residuals are kept.

Every split is by arena or by trial, never by timestep: consecutive steps of
one trial are nearly identical and would leak across a timestep split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Rows:
    """Row-aligned arrays for one feature space (h, policy input, ...)."""
    z: np.ndarray        # (n, F) features
    phase: np.ndarray    # (n,) int, 0 or 1
    arena: np.ndarray    # (n,) int
    trial: np.ndarray    # (n,) int, unique across the whole dataset

    def __post_init__(self):
        n = self.z.shape[0]
        assert self.z.ndim == 2
        for name in ("phase", "arena", "trial"):
            a = getattr(self, name)
            assert a.shape == (n,), (name, a.shape, n)
        assert set(np.unique(self.phase)) <= {0, 1}
        # A trial belongs to one arena and one phase.
        for col in ("arena", "phase"):
            pairs = np.unique(np.stack([self.trial, getattr(self, col)], 1), axis=0)
            assert len(pairs) == len(np.unique(self.trial)), f"trial spans several {col}s"

    def subset(self, mask: np.ndarray) -> "Rows":
        return Rows(self.z[mask], self.phase[mask], self.arena[mask], self.trial[mask])


def zscore_features(z: np.ndarray, *, min_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Z-score each feature; drop features whose std is below min_std.

    Returns the z-scored array and the boolean mask of kept features. Dead units
    (constant over the dataset) carry no information and would divide by zero.
    """
    mu = z.mean(0, dtype=np.float64)
    sd = z.std(0, dtype=np.float64)
    keep = sd >= min_std
    assert keep.any(), "every feature is constant"
    return ((z[:, keep] - mu[keep]) / sd[keep]).astype(np.float32), keep


# ---------------------------------------------------------------- parallelism

def _trial_table(r: Rows) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-trial feature sums, row counts, arena and phase."""
    trials, inv = np.unique(r.trial, return_inverse=True)
    sums = np.zeros((len(trials), r.z.shape[1]), np.float64)
    np.add.at(sums, inv, r.z.astype(np.float64))
    counts = np.bincount(inv).astype(np.float64)
    first = np.zeros(len(trials), int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    return sums, counts, r.arena[first], r.phase[first]


def _coding_vectors(sums, counts, t_arena, t_phase, arenas) -> np.ndarray:
    d = np.empty((len(arenas), sums.shape[1]))
    for i, a in enumerate(arenas):
        m1 = (t_arena == a) & (t_phase == 1)
        m0 = (t_arena == a) & (t_phase == 0)
        assert m1.any() and m0.any(), f"arena {a} lacks a phase"
        d[i] = sums[m1].sum(0) / counts[m1].sum() - sums[m0].sum(0) / counts[m0].sum()
    return d


def _mean_pairwise_cos(d: np.ndarray) -> float:
    norms = np.linalg.norm(d, axis=1)
    assert (norms > 0).all()
    u = d / norms[:, None]
    c = u @ u.T
    iu = np.triu_indices(len(d), 1)
    return float(c[iu].mean())


def parallelism(r: Rows, *, n_shuffle: int, rng: np.random.Generator) -> dict:
    """Mean pairwise cosine of per-arena coding vectors, with a trial-shuffle null."""
    sums, counts, t_arena, t_phase = _trial_table(r)
    arenas = np.unique(t_arena)
    assert len(arenas) >= 2
    d = _coding_vectors(sums, counts, t_arena, t_phase, arenas)
    ps = _mean_pairwise_cos(d)
    null = np.empty(n_shuffle)
    for s in range(n_shuffle):
        perm_phase = t_phase.copy()
        for a in arenas:
            m = np.flatnonzero(t_arena == a)
            perm_phase[m] = t_phase[rng.permutation(m)]
        null[s] = _mean_pairwise_cos(
            _coding_vectors(sums, counts, t_arena, perm_phase, arenas))
    # Fraction of coding-vector energy in the direction shared by all arenas.
    shared = float(np.linalg.norm(d.mean(0)) ** 2 / np.mean(np.sum(d ** 2, 1)))
    return {
        "ps": ps,
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + (null >= ps).sum()) / (1 + n_shuffle)),
        "shared_fraction": shared,
        "n_arenas": int(len(arenas)),
        "n_shuffle": int(n_shuffle),
    }


# ------------------------------------------------------------------- decoding

def _fit_score(ztr, ytr, zte, yte, *, C: float) -> float:
    sc = StandardScaler().fit(ztr)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(ztr), ytr)
    return float(balanced_accuracy_score(yte, clf.predict(sc.transform(zte))))


def geometric_null_rows(r: Rows, rng: np.random.Generator) -> Rows:
    """Bernardi's geometric random model for one draw.

    Each (arena, phase) centroid is replaced by an isotropic Gaussian vector;
    the set is rescaled so its total variance about the grand mean equals the
    data's. Each row keeps its residual from its own condition centroid.
    """
    cond = r.arena.astype(np.int64) * 2 + r.phase
    conds, inv = np.unique(cond, return_inverse=True)
    F = r.z.shape[1]
    mu = np.zeros((len(conds), F))
    np.add.at(mu, inv, r.z.astype(np.float64))
    mu /= np.bincount(inv)[:, None]
    resid = r.z - mu[inv]
    spread = np.sum((mu - mu.mean(0)) ** 2) / len(conds)
    g = rng.standard_normal((len(conds), F))
    g -= g.mean(0)
    g *= np.sqrt(spread / (np.sum(g ** 2) / len(conds)))
    return Rows((g[inv] + resid).astype(np.float32), r.phase, r.arena, r.trial)


def _ccgp_job(r: Rows, tr, te, null_seed, C: float) -> float:
    rows = r if null_seed is None else geometric_null_rows(r, np.random.default_rng(null_seed))
    mtr, mte = np.isin(rows.arena, tr), np.isin(rows.arena, te)
    return _fit_score(rows.z[mtr], rows.phase[mtr], rows.z[mte], rows.phase[mte], C=C)


def ccgp(r: Rows, *, k_train: list[int], n_rep: int, n_test_arenas: int,
         n_null: int, C: float, rng: np.random.Generator, n_jobs: int) -> dict:
    """Train on k arenas, test on held-out arenas, for each k.

    For each k: ``n_rep`` random draws of k training arenas, each tested on
    ``n_test_arenas`` other arenas (all others if fewer). The geometric-null
    distribution uses ``n_null`` fresh null datasets, each with its own split.
    All splits and null seeds are drawn up front from ``rng``, so the result
    does not depend on ``n_jobs``.
    """
    from joblib import Parallel, delayed

    arenas = np.unique(r.arena)
    jobs, keys = [], []
    for k in k_train:
        assert 1 <= k < len(arenas), (k, len(arenas))
        n_te = min(n_test_arenas, len(arenas) - k)
        for kind, n in (("data", n_rep), ("null", n_null)):
            for _ in range(n):
                p = rng.permutation(arenas)
                seed = None if kind == "data" else int(rng.integers(2**31))
                jobs.append(delayed(_ccgp_job)(r, p[:k], p[k:k + n_te], seed, C))
                keys.append((k, kind))
    scores = Parallel(n_jobs=n_jobs)(jobs)
    out = {}
    for k in k_train:
        vals = np.array([s for s, kk in zip(scores, keys) if kk == (k, "data")])
        null = np.array([s for s, kk in zip(scores, keys) if kk == (k, "null")])
        n_te = min(n_test_arenas, len(arenas) - k)
        out[str(k)] = {
            "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if n_rep > 1 else 0.0,
            "values": vals.tolist(),
            "null_mean": float(null.mean()) if n_null else None,
            "null_q95": float(np.quantile(null, 0.95)) if n_null else None,
            "n_test_arenas": int(n_te),
        }
    return out


def within_arena_decoding(r: Rows, *, n_folds: int, C: float,
                          rng: np.random.Generator) -> dict:
    """Ceiling: decode phase inside one arena, holding out whole trials.

    Arenas with fewer than ``n_folds`` trials of either phase cannot be split
    this way; they are skipped and counted in ``n_skipped``.
    """
    accs, skipped = [], 0
    for a in np.unique(r.arena):
        ra = r.subset(r.arena == a)
        trials = np.unique(ra.trial)
        t_phase = np.array([ra.phase[ra.trial == t][0] for t in trials])
        if min((t_phase == 0).sum(), (t_phase == 1).sum()) < n_folds:
            skipped += 1
            continue
        folds = np.empty(len(trials), int)
        # Stratified by phase so every test fold holds both phases.
        for ph in (0, 1):
            idx = rng.permutation(np.flatnonzero(t_phase == ph))
            folds[idx] = np.arange(len(idx)) % n_folds
        fa = []
        for f in range(n_folds):
            te = np.isin(ra.trial, trials[folds == f])
            fa.append(_fit_score(ra.z[~te], ra.phase[~te], ra.z[te], ra.phase[te], C=C))
        accs.append(np.mean(fa))
    assert accs, "no arena has enough trials of both phases"
    accs = np.asarray(accs)
    return {"mean": float(accs.mean()),
            "sd": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "n_arenas": int(len(accs)), "n_skipped": int(skipped)}


# ------------------------------------------------------------------- matching

def match_rows(phase: np.ndarray, arena: np.ndarray, nuisance: np.ndarray,
               edges: list[np.ndarray], *, rng: np.random.Generator) -> np.ndarray:
    """Indices of a subsample in which each arena has the same joint histogram
    of the binned nuisance variables in both phases.

    ``nuisance`` is (n, V); ``edges[v]`` are the bin edges for column v. Rows
    outside the edges are dropped. Within every (arena, bin) cell, the larger
    phase is subsampled down to the smaller one.
    """
    n, V = nuisance.shape
    assert len(edges) == V
    codes = np.zeros(n, np.int64)
    inside = np.ones(n, bool)
    for v in range(V):
        b = np.digitize(nuisance[:, v], edges[v]) - 1
        inside &= (b >= 0) & (b < len(edges[v]) - 1)
        codes = codes * len(edges[v]) + np.clip(b, 0, len(edges[v]) - 2)
    keep = []
    key = arena.astype(np.int64) * (codes.max() + 1) + codes
    for c in np.unique(key[inside]):
        m = inside & (key == c)
        i0 = np.flatnonzero(m & (phase == 0))
        i1 = np.flatnonzero(m & (phase == 1))
        n_take = min(len(i0), len(i1))
        if n_take:
            keep.append(rng.choice(i0, n_take, replace=False))
            keep.append(rng.choice(i1, n_take, replace=False))
    assert keep, "no (arena, bin) cell holds both phases"
    return np.sort(np.concatenate(keep))


def balance_rows(phase: np.ndarray, arena: np.ndarray, *, cap: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Equal row counts per phase in each arena, at most ``cap`` per phase."""
    keep = []
    for a in np.unique(arena):
        i0 = np.flatnonzero((arena == a) & (phase == 0))
        i1 = np.flatnonzero((arena == a) & (phase == 1))
        n_take = min(len(i0), len(i1), cap)
        assert n_take > 0, f"arena {a} lacks a phase"
        keep += [rng.choice(i0, n_take, replace=False), rng.choice(i1, n_take, replace=False)]
    return np.sort(np.concatenate(keep))
