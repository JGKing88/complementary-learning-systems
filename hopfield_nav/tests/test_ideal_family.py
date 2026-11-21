"""Synthetic checks for analysis/ideal_family/theory.py.

Every closed form is tested against Monte Carlo samples drawn directly:
standard normals, bivariate normals, Gaussian random fields (GRFs) made by
spectral synthesis, and random-frequency kernels on a small torus. No encoder
is involved until the last two tests, which pin the family's codes to the
tuning viewer's populations.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage

from analysis.coding_comparison import measures as CM
from analysis.ideal_family import grf
from analysis.ideal_family import measure as IM
from analysis.ideal_family import theory as T

TS = [0.0, 0.5, T.t_of_p(0.1), T.t_of_p(0.02)]


def upcrossings(y: np.ndarray, u: float) -> np.ndarray:
    """Upcrossings of ``u`` per row of a periodic field, seam included."""
    a = y >= u
    return (a & ~np.roll(a, 1, axis=1)).sum(axis=1)


# --------------------------------------------------------------- 2. relu moments

def test_relu_moments_match_monte_carlo():
    x = np.random.default_rng(0).standard_normal(4_000_000)
    for t in TS:
        y = np.maximum(x - t, 0)
        assert abs((x > t).mean() - T.Phibar(t)) < 4 * math.sqrt(T.Phibar(t) / x.size)
        assert y.mean() == pytest.approx(T.relu_m1(t), rel=0.01)
        assert (y * y).mean() == pytest.approx(T.relu_m2(t), rel=0.01)
        assert y.mean() ** 2 / (y * y).mean() == pytest.approx(T.treves_rolls(t), rel=0.02)
    assert T.treves_rolls(0.0) == pytest.approx(1 / math.pi, abs=1e-12)
    assert T.treves_rolls(T.t_of_p(0.1)) == pytest.approx(0.0570, abs=2e-4)


# --------------------------------------------------------------- 5. thresholded kernel

def test_relu_cross_closed_form_matches_series_and_monte_carlo():
    rho = np.array([-0.8, -0.3, 0.0, 0.25, 0.5, 0.75, 0.9])
    rng = np.random.default_rng(1)
    n = 1_000_000
    for t in TS + [-0.7]:
        closed = T.relu_cross(rho, t)
        assert np.allclose(closed, T.relu_cross_hermite(rho, t, kmax=120), atol=1e-7)
        for r_, c in zip(rho, closed):
            x = rng.standard_normal(n)
            y = r_ * x + math.sqrt(1 - r_ * r_) * rng.standard_normal(n)
            v = np.maximum(x - t, 0) * np.maximum(y - t, 0)
            assert abs(v.mean() - c) < 5 * v.std() / math.sqrt(n) + 1e-6
    # Endpoints: C(1) = m2, C(0) = m1^2.
    for t in TS:
        assert T.relu_cross(np.array([1.0]), t)[0] == pytest.approx(T.relu_m2(t))
        assert T.relu_cross(np.array([0.0]), t)[0] == pytest.approx(T.relu_m1(t) ** 2, abs=1e-12)


def test_rectified_kernel_is_the_arc_cosine_kernel():
    rho = np.linspace(-0.99, 0.99, 41)
    th = np.arccos(rho)
    assert np.allclose(T.relu_cross(rho, 0.0), (np.sin(th) + (math.pi - th) * rho) / (2 * math.pi),
                       atol=1e-12)


def test_population_kernel_of_thresholded_grf_units():
    """Cosine and Pearson across 1024 thresholded GRF units, against K(rho(d); t)."""
    rng = np.random.default_rng(2)
    r, L = 8.0, 2048
    y = grf.grf_1d(1024, L, r, rng)                      # (units, positions)
    lags = np.arange(0, 33, 4)
    rho = np.exp(-0.5 * (lags / r) ** 2)
    for t in [0.0, T.t_of_p(0.1)]:
        R = np.maximum(y - t, 0)
        for form, f in (("cosine", T.kernel_cos), ("pearson", T.kernel_pearson)):
            s, _ = CM.lag_profile_1d(R, form, circular=True)
            assert np.allclose(s[lags], f(rho, t), atol=0.02), (t, form)
        # The far cosine floor is the Treves-Rolls sparsity a(t).
        s, _ = CM.lag_profile_1d(R, "cosine", circular=True)
        assert s[200:600].mean() == pytest.approx(T.treves_rolls(t), abs=0.01)


def test_kernel_widths_and_width_matching():
    assert T.gaussian_width(0.5, 1.0) == pytest.approx(2.35482, abs=1e-5)
    assert T.gaussian_width(0.2, 1.0) == pytest.approx(3.58825, abs=1e-5)
    t = T.t_of_p(0.1)
    for form in ("cos", "pearson"):
        r = T.r_for_width(37.68, 0.5, t, form)
        assert T.kernel_width(0.5, r, t, form) == pytest.approx(37.68, rel=1e-9)
    # Thresholding narrows the kernel in units of r, so r must grow as p falls.
    rs = [T.r_for_width(1.0, 0.5, T.t_of_p(p), "pearson") for p in (0.5, 0.2, 0.1, 0.02)]
    assert all(a < b for a, b in zip(rs, rs[1:]))


# --------------------------------------------------------------- 2-3. GRF single units, 1D

def test_grf_coverage_and_treves_rolls():
    y = grf.grf_1d(256, 8192, 8.0, np.random.default_rng(3))
    for t in TS:
        R = np.maximum(y - t, 0)
        cov = (R > 0).mean(axis=1)
        assert cov.mean() == pytest.approx(T.Phibar(t), rel=0.04)
        tr = IM.tr_summary(R)
        assert tr["pooled"] == pytest.approx(T.treves_rolls(t), rel=0.04)


def test_rice_rate_and_mean_length():
    r = 8.0
    y = grf.grf_1d(128, 8192, r, np.random.default_rng(4))
    for u in (0.0, 1.0, 2.0):
        n = upcrossings(y, u)
        rate = n.sum() / y.size
        assert rate == pytest.approx(T.nu_1d(u, r), rel=0.04), u
        mean_len = (y >= u).sum() / n.sum()
        assert mean_len == pytest.approx(T.mean_len_1d(u, r), rel=0.04), u
        # Fields per field width equals the coverage, p.
        assert rate * T.mean_len_1d(u, r) == pytest.approx(T.Phibar(u), rel=0.04)


def test_expected_max_on_loops():
    r, L = 16.0, 1716
    y = grf.grf_1d(2048, L, r, np.random.default_rng(5))
    assert y.max(axis=1).mean() == pytest.approx(T.expected_max_1d(L, r), rel=0.03)
    # Gumbel's shortcut is close but not the same thing.
    assert abs(T.gumbel_max_1d(L, r) - T.expected_max_1d(L, r)) < 0.15


@pytest.mark.parametrize("p", [0.5, 0.1])
def test_definition_A_width_on_grf_loops(p):
    """W_A on 1716-cell loops at r = 16, through the suite's field detector."""
    t = T.t_of_p(p)
    y = grf.grf_1d(1024, 1716, 16.0, np.random.default_rng(6))
    R = np.maximum(y - t, 0)
    st, F = IM.loop_stats(R)
    pred = T.def_A_1d(t, 1716, 16.0)
    assert st["mean_field_size"] == pytest.approx(pred["W_A"], rel=0.03)
    assert st["fields_per_unit"] == pytest.approx(pred["fields_per_unit"], rel=0.03)
    assert st["field_coverage_all"] == pytest.approx(pred["field_coverage"] * pred["active_frac"],
                                                     rel=0.03)


def test_fano_of_upcrossings_matches_simulation():
    r, L = 8.0, 2048
    y = grf.grf_1d(4000, L, r, np.random.default_rng(7))
    for u in (0.0, 1.0, 2.0):
        n = upcrossings(y, u)
        fano = n.var() / n.mean()
        assert fano == pytest.approx(T.fano_upcross(u, r, L, periodic=True), abs=0.06), u


def test_fano_never_exceeds_one():
    us = np.linspace(-4, 6, 41)
    f = np.array([T.fano_upcross(u, 10.0) for u in us])
    # Rises from 0.29 at u = 0 to Poisson; any overshoot is below 1e-4 (2e-5 at u ~ 3.7).
    assert (f < 1 + 1e-4).all()
    assert f[np.argmin(np.abs(us))] < 0.3
    assert np.all(np.diff(f[(us >= 0) & (us <= 3)]) > 0)


# --------------------------------------------------------------- 4. 2D Euler characteristic

def test_ec_counts_fields_at_high_level_and_undercounts_at_low():
    r, B = 4.0, 512
    z = grf.grf_2d(6, B, r, np.random.default_rng(8))
    for u, ok in ((2.0, True), (1.0, True), (0.75, True), (0.25, False), (0.0, False)):
        n_comp, above = 0, 0
        for f in z:
            lab, n = ndimage.label(f >= u)
            n_comp += n
            above += int((f >= u).sum())
        ec = T.ec_square(u, B, r) * len(z)
        if ok:
            assert n_comp == pytest.approx(ec, rel=0.05), u
            assert above / n_comp == pytest.approx(float(T.mean_area_2d(u, r)), rel=0.06), u
        else:
            # Below u ~ 0.5 fields have holes and merge: components outnumber the EC.
            assert n_comp > 1.15 * ec, (u, n_comp, ec)


def test_expected_max_over_squares():
    r = 6.0
    for side, tol in ((48, 0.04), (96, 0.04), (24, 0.12)):
        B = grf.arena_box(side, r)
        z = grf.grf_2d(300, B, r, np.random.default_rng(9))[:, :side, :side]
        assert z.reshape(len(z), -1).max(axis=1).mean() == pytest.approx(
            T.expected_max_2d(side, r), rel=tol), side


# --------------------------------------------------------------- 6. finite M

def _rff_kernel_on_torus(M, r, N, seed):
    sigma = N / (2 * math.pi * r)
    rng = np.random.RandomState(seed)
    n = np.rint(rng.normal(0, sigma, size=(M, 2))).astype(np.int64) % N
    H = np.zeros((N, N))
    np.add.at(H, (n[:, 0], n[:, 1]), 1.0)
    return np.fft.ifft2(H).real * N * N / M


def test_far_field_sd_and_alias_maximum():
    N, r = 384, 6.0
    excl = 3 * T.gaussian_width(0.5, r)
    d = np.minimum(np.arange(N), N - np.arange(N))
    far = (d[:, None] ** 2 + d[None, :] ** 2) > excl ** 2
    for M in (32, 256):
        sds, maxes = [], []
        for seed in range(6):
            K = _rff_kernel_on_torus(M, r, N, seed)
            sds.append(K[far].std())
            maxes.append(K[far].max())
        assert np.mean(sds) == pytest.approx(T.far_kernel_sd(M), rel=0.06), M
        pred = T.alias_prediction(M, r, npos=N, excl_radius=excl)["alias"]
        # Sums of few cosines have lighter tails than a Gaussian, so the
        # prediction is an upper estimate at small M.
        lo = 0.80 if M < 64 else 0.90
        assert lo * pred < np.mean(maxes) < 1.08 * pred, (M, np.mean(maxes), pred)


def test_spectral_moment_spread_across_rotated_units():
    """Units of one draw share frequencies; Dirichlet weights give CV^2 = (s_a^2 / abar^2) / (M + 1)."""
    rng = np.random.default_rng(10)
    for M in (16, 64):
        a = rng.standard_normal(M) ** 2                       # one draw, shared by every unit
        w = rng.exponential(1.0, size=(20000, M))
        w /= w.sum(1, keepdims=True)
        lam = w @ a
        a_cv2 = a.var() / a.mean() ** 2
        assert (lam.var() / lam.mean() ** 2) == pytest.approx(T.lambda2_cv2(M, a_cv2), rel=0.05)


def test_finite_M_units_are_overdispersed_on_long_tracks():
    """Units of one draw (shared frequencies, own weights and phases) on a long track.

    Count Fano = F(u) + L nu a_cv2 / (4 (M + 1)); above 1 for few waves.
    """
    rng = np.random.default_rng(11)
    r, L, M, u = 8.0, 20000, 16, 1.0
    x = np.arange(L)
    k = rng.standard_normal(M) / r
    a_cv2 = (k * k).var() / (k * k).mean() ** 2
    counts = []
    for _ in range(400):
        A = np.sqrt(rng.exponential(1.0, M))
        ph = rng.uniform(0, 2 * math.pi, M)
        y = (A[:, None] * np.cos(k[:, None] * x[None, :] + ph[:, None])).sum(0)
        y /= math.sqrt((A * A).sum() / 2)
        a = y >= u
        counts.append(int((a[1:] & ~a[:-1]).sum()))
    counts = np.asarray(counts)
    nu = (k * k).mean() ** 0.5 * r * T.nu_1d(u, r)          # this draw's mean rate
    pred = T.fano_upcross(u, r) + T.heterogeneity_fano_excess(M, L * nu, a_cv2)
    assert counts.var() / counts.mean() == pytest.approx(pred, rel=0.25)
    assert counts.var() / counts.mean() > 1


def test_definition_A_level_spread_adds_count_variance():
    """Identically distributed GRF units: definition A's per-unit level overdisperses long environments."""
    r, L = 8.0, 4096
    y = grf.grf_1d(3000, L, r, np.random.default_rng(16))
    st, F = IM.loop_stats(np.maximum(y, 0))
    pred = T.fano_def_A(0.0, L, r)
    assert st["var_over_mean"] == pytest.approx(pred["total"], abs=0.08)
    assert pred["level_term"] > 0.05


def test_edgeworth_coverage_of_few_sinusoids():
    rng = np.random.default_rng(12)
    M, n, t = 8, 2_000_000, 2.0
    A = np.sqrt(rng.exponential(1.0, (n, M)))
    y = (A * np.cos(rng.uniform(0, 2 * math.pi, (n, M)))).sum(1)
    y /= np.sqrt((A * A).sum(1) / 2)
    p = (y > t).mean()
    assert abs(p - T.edgeworth_coverage(t, M)) < abs(p - T.Phibar(t))


# --------------------------------------------------------------- 7. native plane waves

def test_native_plane_waves_on_a_loop():
    N, r = 1716, 16.0
    sigma = N / (2 * math.pi * r)
    n = np.rint(np.random.RandomState(13).normal(0, sigma, size=4096)).astype(np.int64)
    x = np.arange(N)
    th = 2 * math.pi * np.outer(n, x) / N + np.random.default_rng(13).uniform(0, 2 * math.pi, (4096, 1))
    R = np.maximum(np.cos(th), 0)
    st, F = IM.loop_stats(R)
    assert st["mean_field_size"] == pytest.approx(T.native_mean_width_1d(r), rel=0.03)
    # Waves with n_x = 0 are constant along the loop (TR 1 when positive); the rest give 4/pi^2.
    assert IM.tr_summary(R[n != 0])["active"] == pytest.approx(T.NATIVE_TR, rel=0.005)
    assert st["coverage_all"] == pytest.approx(T.NATIVE_COVERAGE, abs=0.01)
    assert st["field_coverage_all"] == pytest.approx(T.native_field_fraction(), abs=0.01)
    assert st["var_over_mean"] == pytest.approx(T.native_fano_1d(N, r), rel=0.1)
    lags = np.arange(0, 60, 6)
    for form in ("cos", "pearson"):
        s, _ = CM.lag_profile_1d(R, {"cos": "cosine", "pearson": "pearson"}[form], circular=True)
        assert np.allclose(s[lags], T.native_kernel(lags, r, form), atol=0.02), form


def test_native_tr_on_short_track_is_inflated():
    out = T.native_tr_track(252, 16.0)
    assert out["pooled"] == pytest.approx(T.NATIVE_TR, rel=0.01)
    assert out["active"] > T.NATIVE_TR + 0.015


# --------------------------------------------------------------- helpers

def test_loop_fields_join_the_seam():
    x = np.arange(100)
    R = np.exp(-0.5 * (np.minimum(x, 100 - x) / 4.0) ** 2)[None, :]   # one bump across the seam
    F = IM.loop_fields(R)
    assert F.counts[0] == 1
    assert CM.detect_fields_1d(R, IM.DEF_A).counts[0] == 2


def test_area95_of_a_box_and_of_a_spike():
    R = np.zeros((2, 100))
    R[0, :40] = 1.0
    R[1, 7] = 1.0
    a = IM.area95(R)
    assert a[0] == pytest.approx(0.38) and a[1] == pytest.approx(0.01)


# --------------------------------------------------------------- the encoder family

def test_fast_rotation_is_the_viewers_hadamard_matrix():
    pytest.importorskip("torch")
    from analysis.ideal_family.codes import apply_rotation, rotation_signs
    from analysis.tuning_viewer.export import hadamard_rotation
    for D in (64, 1024):
        signs = rotation_signs(D, 0)
        Q = hadamard_rotation(signs)
        z = np.random.default_rng(14).standard_normal((5, D))
        assert np.allclose(apply_rotation(z, signs), z @ Q.T, atol=1e-10)
        assert np.allclose(Q @ Q.T, np.eye(D), atol=1e-10)


def test_family_reproduces_the_viewer_populations():
    pytest.importorskip("torch")
    from analysis.coding_comparison.populations import make_population
    from analysis.ideal_family.codes import Family, rates
    rng = np.random.default_rng(15)
    xs, ys = rng.integers(0, 1716, (2, 300))
    f16 = Family(16.0, 512, 0)
    z = f16.native(xs, ys)
    assert np.allclose(z, make_population("ideal").code(xs, ys), atol=1e-6)
    assert np.allclose(f16.rotated(z), make_population("ideal_rot").code(xs, ys), atol=1e-6)
    f23 = Family(23.2, 512, 0)
    y23 = f23.rotated(f23.native(xs, ys))
    got = rates("sparse_p0.1", None, y23, f23.sd)
    assert np.allclose(got, make_population("ideal_sparse").code(xs, ys), atol=1e-6)
    # The torus kernel is the draw's mean cosine.
    K = f16.raw_kernel_torus()
    d = np.array([[5, 0], [40, 17], [900, 3]])
    direct = np.cos(2 * math.pi * (d @ f16.freqs.T) / 1716).mean(axis=1)
    assert np.allclose(K[d[:, 0], d[:, 1]], direct, atol=1e-10)
