"""Synthetic checks for analysis/coding_comparison, run before trusting any result.

Each test builds data whose answer is known: a Poisson field process, a
gamma-Poisson process with Rich 2014's shape r = 0.57, Gaussian bumps at known
places, and a code remixed by a random rotation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from analysis.coding_comparison import geometry as geo
from analysis.coding_comparison import measures as M

DEF_A = M.FieldDef("A", "peak_fraction", frac=0.2, min_len=3, min_area=9)


def bump_track(centers_per_unit, L, sigma, peak=1.0):
    """Rate maps ``(units, L)``: a sum of Gaussian bumps per unit."""
    x = np.arange(L, dtype=np.float64)
    R = np.zeros((len(centers_per_unit), L))
    for u, cs in enumerate(centers_per_unit):
        for c in cs:
            R[u] += peak * np.exp(-0.5 * ((x - c) / sigma) ** 2)
    return R


# --------------------------------------------------------------- count models

def test_poisson_counts_fit_poisson():
    x = np.random.default_rng(0).poisson(3.5, size=4000)
    fit = M.fit_count_models(x)
    assert abs(fit["var_over_mean"] - 1) < 0.1
    # The negative binomial adds nothing: r runs off to the Poisson limit or
    # stays huge, and the likelihood ratio is not significant.
    assert fit["negbin"]["poisson_limit"] or fit["negbin"]["r"] > 20
    assert fit["lr_p"] > 0.01


def test_gamma_poisson_counts_recover_rich_r():
    # Rich 2014: rate ~ Gamma(0.57, rate 7.75 per m) on a 48 m track.
    x = M.sample_gamma_poisson(6000, 0.57, 7.75, 48.0, seed=1)
    fit = M.fit_count_models(x)
    nb = fit["negbin"]
    assert abs(nb["r"] - 0.57) < 0.05
    assert abs(nb["p"] - 7.75 / (7.75 + 48)) < 0.02
    assert fit["lr_p"] < 1e-10 and fit["delta_aic"] > 100
    # P(0) of the fit matches the empirical zero fraction.
    assert abs(nb["p"] ** nb["r"] - fit["frac_zero"]) < 0.02


def _field_process(counts, L, sigma, rng):
    """Units whose fields sit at uniform random places, ``counts[u]`` each."""
    return [np.sort(rng.uniform(5 * sigma, L - 5 * sigma, size=int(c))) for c in counts]


def test_poisson_field_process_through_the_detector():
    rng = np.random.default_rng(2)
    L, sigma = 6000, 2.0
    counts = rng.poisson(3.5, size=1500)
    R = bump_track(_field_process(counts, L, sigma, rng), L, sigma)
    F = M.detect_fields_1d(R, DEF_A)
    # Overlapping bumps can merge, so detected counts are at most the truth
    # and almost always equal to it at this density.
    assert np.mean(F.counts == counts) > 0.97
    fit = M.fit_count_models(F.counts)
    assert fit["lr_p"] > 0.01
    assert abs(fit["var_over_mean"] - 1) < 0.12


def test_gamma_poisson_field_process_through_the_detector():
    rng = np.random.default_rng(3)
    L, sigma = 20000, 2.0
    counts = M.sample_gamma_poisson(2500, 0.57, 7.75, 48.0, seed=4)
    R = bump_track(_field_process(counts, L, sigma, rng), L, sigma)
    F = M.detect_fields_1d(R, DEF_A)
    fit = M.fit_count_models(F.counts)
    assert abs(fit["negbin"]["r"] - 0.57) < 0.08
    lay = M.within_unit_layout(F, min_fields=6)
    # Uniform placement: exponential-like gaps (CV near 1, a little below
    # because fields cannot overlap) and few uniformity rejections.
    assert 0.75 < lay["gap_cv_median"] < 1.1
    assert lay["ks_frac_reject_05"] < 0.12


def test_periodic_fields_have_low_gap_cv():
    L, sigma = 3000, 2.0
    R = bump_track([np.arange(20, L - 20, 50) + o for o in range(10)], L, sigma)
    lay = M.within_unit_layout(M.detect_fields_1d(R, DEF_A))
    assert lay["gap_cv_median"] < 0.05


# --------------------------------------------------------------- field detector

def test_detector_finds_gaussian_bumps_1d():
    L, sigma = 400, 4.0
    centers = [[50.0, 200.0, 330.0], [120.0], []]
    R = bump_track(centers, L, sigma)
    F = M.detect_fields_1d(R, DEF_A)
    assert F.counts.tolist() == [3, 1, 0]
    got = sorted(zip(F.unit.tolist(), F.center.tolist()))
    want = sorted((u, c) for u, cs in enumerate(centers) for c in cs)
    for (gu, gc), (wu, wc) in zip(got, want):
        assert gu == wu and abs(gc - wc) < 0.5
    # Edge at 20% of peak: half-width sigma * sqrt(2 ln 5), so the run is
    # that width rounded to whole bins.
    width = 2 * sigma * math.sqrt(2 * math.log(5))
    assert np.all(np.abs(F.length - width) <= 1.0)


def test_detector_absolute_threshold_and_min_length():
    L, sigma = 300, 3.0
    R = bump_track([[100.0], [200.0]], L, sigma)
    R[1] *= 0.5                                  # peaks of 1.0 and 0.5
    rich = M.FieldDef("B", "absolute", thresh_hz=2.0, rate_scale=3.0)   # peaks 3 Hz and 1.5 Hz
    assert M.detect_fields_1d(R, rich).counts.tolist() == [1, 0]
    long_min = rich.with_(min_len=100)
    assert M.detect_fields_1d(R, long_min).counts.tolist() == [0, 0]


def test_field_peak_floor():
    # Eliav's rule: 20% of peak, and each field's own peak must reach 1 Hz.
    L, sigma = 400, 3.0
    R = bump_track([[100.0, 300.0]], L, sigma)
    R[0, 200:] *= 0.4                            # second field peaks at 0.4
    ap = DEF_A.with_(rate_scale=2.0, min_peak_hz=1.0)   # peaks 2.0 Hz and 0.8 Hz
    assert M.detect_fields_1d(R, DEF_A).counts.tolist() == [2]
    F = M.detect_fields_1d(R, ap)
    assert F.counts.tolist() == [1] and abs(F.center[0] - 100) < 0.5
    S = 60
    yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    m = (np.exp(-0.5 * ((xx - 15) ** 2 + (yy - 15) ** 2) / 9)
         + 0.4 * np.exp(-0.5 * ((xx - 45) ** 2 + (yy - 45) ** 2) / 9))[None]
    assert M.detect_fields_2d(m, DEF_A).counts.tolist() == [2]
    assert M.detect_fields_2d(m, ap).counts.tolist() == [1]


def test_detector_finds_gaussian_bumps_2d():
    S, sigma = 80, 3.0
    yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    def blob(cx, cy):
        return np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma ** 2)
    R = np.stack([blob(20, 20) + blob(60, 50), blob(40, 40), np.zeros((S, S))])
    F = M.detect_fields_2d(R, DEF_A)
    assert F.counts.tolist() == [2, 1, 0]
    area = math.pi * 2 * math.log(5) * sigma ** 2      # disc inside 20% of peak
    assert np.all(np.abs(F.area - area) / area < 0.15)
    cents = sorted(zip(F.unit.tolist(), np.round(F.cx).tolist(), np.round(F.cy).tolist()))
    assert cents == [(0, 20, 20), (0, 60, 50), (1, 40, 40)]
    # Coverage of a location is the fraction of units with a field there.
    assert F.coverage_frac[40, 40] == pytest.approx(1 / 3)


def test_onoff_convention_splits_sign():
    z = np.array([[1.0, -2.0, 0.0]])
    assert M.apply_convention(z, "onoff").tolist() == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    assert M.apply_convention(z, "rectify").tolist() == [[1.0, 0.0, 0.0]]


# --------------------------------------------------------------- similarity

def test_cosine_similarity_is_rotation_invariant():
    rng = np.random.default_rng(5)
    U, P = 64, 300
    V = np.cumsum(rng.standard_normal((U, P)), axis=1)      # smooth-ish code
    Q = M.random_orthogonal(U, seed=9)
    assert np.allclose(Q @ Q.T, np.eye(U), atol=1e-10)
    W = Q @ V
    for circ in (True, False):
        a, _ = M.lag_profile_1d(V, "cosine", circular=circ)
        b, _ = M.lag_profile_1d(W, "cosine", circular=circ)
        assert np.allclose(a, b, atol=1e-10)
    i, j = rng.integers(0, P, 200), rng.integers(0, P, 200)
    assert np.allclose(M.pair_similarity(V[:, i], V[:, j], "cosine"),
                       M.pair_similarity(W[:, i], W[:, j], "cosine"), atol=1e-10)
    # Pearson centres each PV across units, which a rotation does not commute with.
    assert not np.allclose(M.pair_similarity(V[:, i], V[:, j], "pearson"),
                           M.pair_similarity(W[:, i], W[:, j], "pearson"), atol=1e-3)


def test_lag_profile_matches_direct_computation():
    rng = np.random.default_rng(6)
    V = rng.random((20, 50))
    for method in M.SIMILARITIES:
        S = M.similarity_matrix(V, method)
        s, n = M.lag_profile_1d(V, method, circular=False)
        for d in (0, 1, 7, 30):
            assert s[d] == pytest.approx(np.mean(np.diagonal(S, d)), abs=1e-10)
            assert n[d] == 50 - d
        sc, _ = M.lag_profile_1d(V, method, circular=True)
        Sc = [np.mean([S[p, (p + d) % 50] for p in range(50)]) for d in (0, 3, 25)]
        assert np.allclose(sc[[0, 3, 25]], Sc, atol=1e-10)


def test_kernel_summary_on_a_gaussian():
    r = 10.0
    d = np.arange(0, 400, dtype=np.float64)
    s = np.exp(-d ** 2 / (2 * r ** 2)) - 0.02
    k = M.kernel_summary(d, s, width=r, far=(5, 15))
    assert k["half_width_05"] == pytest.approx(r * math.sqrt(2 * math.log(1 / 0.52)), abs=0.05)
    assert k["full_width_05"] == pytest.approx(2 * k["half_width_05"])
    assert k["zero_crossing"] == pytest.approx(r * math.sqrt(2 * math.log(50)), abs=0.05)
    assert k["floor"] == pytest.approx(-0.02, abs=1e-6)


def test_overlap_of_independent_sets_is_at_chance():
    rng = np.random.default_rng(7)
    a, b = rng.random(20000) < 0.3, rng.random(20000) < 0.3
    o = M.active_set_overlap(a, b)
    assert o["both_ratio"] == pytest.approx(1.0, abs=0.05)
    assert o["jaccard_ratio"] == pytest.approx(1.0, abs=0.05)
    assert M.active_set_overlap(a, a)["both_ratio"] == pytest.approx(1 / a.mean(), rel=1e-9)


def test_sparsity_of_binary_maps_is_active_fraction():
    R = np.zeros((2, 100))
    R[0, :25] = 3.0
    R[1] = 1.0
    assert M.sparsity(R).tolist() == pytest.approx([0.25, 1.0])


# --------------------------------------------------------------- geometry

def test_serpentine_is_continuous_and_unique():
    xs, ys = geo.serpentine(5, 9, 5000, run=1716, gap=200, axis=0)
    steps = np.abs(np.diff(xs)) + np.abs(np.diff(ys))
    assert xs.size == 5000 and np.all(steps == 1)
    cells = set(zip((xs % 1716).tolist(), (ys % 1716).tolist()))
    assert len(cells) == 5000
    assert geo.serpentine_capacity(1716, 1716, 200) == 9 * 1716 + 8 * 200


# --------------------------------------------------------------- populations

def test_rotated_ideal_has_the_ideals_similarity():
    pytest.importorskip("torch")
    from analysis.coding_comparison.populations import make_population
    ideal, rot = make_population("ideal"), make_population("ideal_rot")
    rng = np.random.default_rng(8)
    xs, ys = rng.integers(0, 1716, (2, 300))
    a = M.pair_similarity(ideal.code(xs, ys).T, ideal.code(xs + 7, ys - 3).T, "cosine")
    b = M.pair_similarity(rot.code(xs, ys).T, rot.code(xs + 7, ys - 3).T, "cosine")
    assert np.allclose(a, b, atol=1e-5)
    assert ideal.signed and rot.signed and ideal.n_units == 1024
