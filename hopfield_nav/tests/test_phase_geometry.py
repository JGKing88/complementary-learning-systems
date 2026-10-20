"""The abstraction metrics on synthetic geometries with known answers."""
from __future__ import annotations

import numpy as np
import pytest

from analysis.phase_decoding import geometry as g


def _synthetic(kind: str, *, n_arenas=12, n_trials=10, steps=8, F=40, seed=0):
    """Rows with large arena offsets and a phase coding vector that is

    - "abstract": shared across arenas, offsets orthogonal to it (factorised);
    - "parallel_offset": shared, but offsets also move along it, so PS stays
      high while a decoder trained on one arena gets the wrong threshold;
    - "random": drawn independently per arena.
    """
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(F)
    shared *= 3 / np.linalg.norm(shared)
    u = shared / np.linalg.norm(shared)
    z, phase, arena, trial = [], [], [], []
    tid = 0
    for a in range(n_arenas):
        offset = rng.standard_normal(F)
        offset -= (offset @ u) * u
        if kind == "parallel_offset":
            offset += 3 * rng.standard_normal() * u
        if kind in ("abstract", "parallel_offset"):
            d = shared
        else:
            d = rng.standard_normal(F)
            d *= 3 / np.linalg.norm(d)
        for ph in (0, 1):
            for _ in range(n_trials):
                trial_mean = offset + ph * d + 0.3 * rng.standard_normal(F)
                z.append(trial_mean + 0.5 * rng.standard_normal((steps, F)))
                phase += [ph] * steps
                arena += [a] * steps
                trial += [tid] * steps
                tid += 1
    return g.Rows(np.concatenate(z).astype(np.float32), np.array(phase),
                  np.array(arena), np.array(trial))


def test_parallelism_abstract_vs_random():
    rng = np.random.default_rng(1)
    pa = g.parallelism(_synthetic("abstract"), n_shuffle=50, rng=rng)
    pr = g.parallelism(_synthetic("random"), n_shuffle=50, rng=rng)
    assert pa["ps"] > 0.9 and pa["p_value"] < 0.05
    assert abs(pr["ps"]) < 0.15
    assert abs(pa["null_mean"]) < 0.1


def test_ccgp_separates_abstract_from_random_geometry():
    rng = np.random.default_rng(2)
    kw = dict(k_train=[1], n_rep=8, n_test_arenas=11, n_null=8, C=1.0, rng=rng, n_jobs=2)
    ca = g.ccgp(_synthetic("abstract"), **kw)["1"]
    cr = g.ccgp(_synthetic("random"), **kw)["1"]
    cp = g.ccgp(_synthetic("parallel_offset"), **kw)["1"]
    assert ca["mean"] > 0.9
    assert cr["mean"] < 0.7
    # The geometric null destroys the shared direction.
    assert ca["null_mean"] < 0.7
    # Parallel coding vectors do not guarantee one-arena generalisation: arena
    # offsets along the coding axis move the decision threshold.
    pp = g.parallelism(_synthetic("parallel_offset"), n_shuffle=20, rng=rng)
    assert pp["ps"] > 0.9 and cp["mean"] < ca["mean"] - 0.15


def test_within_arena_decoding_is_high_for_both():
    rng = np.random.default_rng(3)
    for kind in ("abstract", "random"):
        w = g.within_arena_decoding(_synthetic(kind), n_folds=5, C=1.0, rng=rng)
        assert w["mean"] > 0.95, kind


def test_geometric_null_keeps_total_centroid_variance_and_residuals():
    r = _synthetic("abstract")
    rn = g.geometric_null_rows(r, np.random.default_rng(4))

    def centroids(rows):
        cond = rows.arena * 2 + rows.phase
        return np.stack([rows.z[cond == c].mean(0) for c in np.unique(cond)]), cond

    mu, cond = centroids(r)
    mun, _ = centroids(rn)
    spread = lambda m: np.sum((m - m.mean(0)) ** 2) / len(m)
    assert spread(mun) == pytest.approx(spread(mu), rel=1e-4)
    idx = {c: i for i, c in enumerate(np.unique(cond))}
    res = r.z - mu[[idx[c] for c in cond]]
    resn = rn.z - mun[[idx[c] for c in cond]]
    np.testing.assert_allclose(res, resn, atol=1e-3)


def test_match_rows_equalises_histograms_per_arena():
    rng = np.random.default_rng(5)
    n = 4000
    arena = rng.integers(0, 4, n)
    phase = rng.integers(0, 2, n)
    # Phase 1 is shifted toward early times: a confound before matching.
    t = np.where(phase == 1, rng.integers(0, 10, n), rng.integers(0, 40, n)).astype(float)
    edges = [np.arange(0, 41, 5.0)]
    keep = g.match_rows(phase, arena, t[:, None], edges, rng=rng)
    for a in range(4):
        m = arena[keep] == a
        h0 = np.histogram(t[keep][m & (phase[keep] == 0)], edges[0])[0]
        h1 = np.histogram(t[keep][m & (phase[keep] == 1)], edges[0])[0]
        np.testing.assert_array_equal(h0, h1)


def test_rows_rejects_trial_spanning_two_phases():
    with pytest.raises(AssertionError):
        g.Rows(np.zeros((2, 3), np.float32), np.array([0, 1]), np.array([0, 0]),
               np.array([7, 7]))
