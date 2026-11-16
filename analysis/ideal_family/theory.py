"""Closed forms for the ideal-encoder family in its large-M limit.

The ideal code (``hopfield_probe/ideal_encoder.py``) draws ``M = n_freq``
integer wave vectors ``n_i`` with each component ``round(Normal(0, sigma_n))``,
``sigma_n = Npos / (2 pi r)``, so ``omega_i = 2 pi n_i / Npos`` has SD ``1/r``
per component. Unit ``i`` is ``cos(omega_i . p) / sqrt(M)`` and unit ``M + i``
is ``sin(omega_i . p) / sqrt(M)``, so there are ``D = 2M`` units and every
place has norm 1. ``r`` is therefore the SD, in cells, of the Gaussian kernel

    z(p) . z(p + d) = (1/M) sum_i cos(omega_i . d)  ->  exp(-|d|^2 / 2 r^2).

Rotated units ``y = Q z`` (``Q`` orthogonal) have the same kernel. Each one is
a sum of M sinusoids with random amplitudes and phases, so as M grows it tends
to a stationary isotropic Gaussian random field (GRF) with variance ``1/D``
and correlation ``rho(d) = exp(-d^2 / 2 r^2)``. Every function below takes
levels in units of that SD (``u = level * sqrt(D)``) and lengths in cells.

Contents, in the order of the write-up (``docs/EXPERIMENTS_IDEAL_FAMILY.md``):

1. raw kernel widths;
2. moments of ``relu(X - t)``: coverage, Treves-Rolls sparsity;
3. Rice formulas in 1D, the GRF maximum, and definition A's level ``u_A``;
4. 2D Euler-characteristic densities, field areas and counts;
5. the thresholded population kernel ``K(rho; t)``, and the ``r`` that holds
   a kernel width as ``p`` falls;
6. finite M: the far-field alias and unit-to-unit heterogeneity;
7. the native (plane-wave) code under rectification;
8. the Fano factor of upcrossing counts (Cramer-Leadbetter second moment).

Nothing here touches an encoder. ``hopfield_nav/tests/test_ideal_family.py``
checks every formula against Monte Carlo samples drawn directly.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize, special

SQ2PI = math.sqrt(2.0 * math.pi)
NPOS = 1716                     # scaffold side for lambdas 11, 12, 13
DEF_A_FRAC = 0.2                # definition A: 20% of the unit's peak


# ---------------------------------------------------------------------------
# Normal helpers
# ---------------------------------------------------------------------------

def phi(x):
    """Standard normal density."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x * x) / SQ2PI


def Phibar(x):
    """Upper tail ``1 - Phi(x)``, accurate in the far tail."""
    return special.ndtr(-np.asarray(x, dtype=np.float64))


def t_of_p(p: float) -> float:
    """Threshold (in SDs) that leaves a fraction ``p`` above it."""
    return float(-special.ndtri(p))


# ---------------------------------------------------------------------------
# 1. Raw kernel
# ---------------------------------------------------------------------------

def gaussian_width(level: float, r: float) -> float:
    """Full width where ``exp(-d^2 / 2 r^2)`` falls to ``level``: ``2 r sqrt(2 ln(1/level))``.

    At 0.5 this is 2.355 r; at 0.2 it is 3.588 r.
    """
    return 2.0 * r * math.sqrt(2.0 * math.log(1.0 / level))


# ---------------------------------------------------------------------------
# 2. Moments of relu(X - t), X ~ N(0, 1)
# ---------------------------------------------------------------------------

def relu_m1(t):
    """``E relu(X - t) = phi(t) - t (1 - Phi(t))``."""
    return phi(t) - t * Phibar(t)


def relu_m2(t):
    """``E relu(X - t)^2 = (1 + t^2)(1 - Phi(t)) - t phi(t)``."""
    return (1.0 + t * t) * Phibar(t) - t * phi(t)


def treves_rolls(t):
    """Treves-Rolls sparsity ``m1^2 / m2`` of one unit over a long environment.

    ``a(0) = 1/pi``; ``a`` falls with ``t`` (0.057 at p = 0.1).
    """
    return relu_m1(t) ** 2 / relu_m2(t)


# ---------------------------------------------------------------------------
# 5. Thresholded bivariate moment and population kernels
# ---------------------------------------------------------------------------

def relu_cross(rho, t):
    """``C(rho; t) = E[relu(X - t) relu(Y - t)]``, ``(X, Y)`` standard bivariate normal.

    Closed form (truncated bivariate normal moments, Rosenbaum 1961):

        C = (rho + t^2) L - 2 t phi(t) Phibar(q) + (s / sqrt(2 pi)) phi(t sqrt(2 / (1 + rho)))

    with ``s = sqrt(1 - rho^2)``, ``q = t sqrt((1 - rho)/(1 + rho))`` and
    ``L = P(X > t, Y > t) = Phibar(t) - 2 T(t, sqrt((1 - rho)/(1 + rho)))``
    (Owen's T). ``C(1) = m2``; ``C(0) = m1^2``; at ``t = 0`` it is the
    arc-cosine kernel ``(sin th + (pi - th) cos th) / 2 pi``, ``cos th = rho``.
    """
    rho = np.clip(np.asarray(rho, dtype=np.float64), -1.0, 1.0)
    t = float(t)
    out = np.empty_like(rho)
    hi = rho > 1 - 1e-12
    lo = rho < -1 + 1e-12
    mid = ~(hi | lo)
    rr = rho[mid]
    a = np.sqrt((1 - rr) / (1 + rr))
    s = np.sqrt(1 - rr * rr)
    L = Phibar(t) - 2.0 * special.owens_t(t, a)
    out[mid] = ((rr + t * t) * L - 2.0 * t * phi(t) * Phibar(t * a)
                + s / SQ2PI * phi(t * np.sqrt(2.0 / (1 + rr))))
    out[hi] = relu_m2(t)
    # rho = -1: Y = -X, so both exceed t >= 0 never; for t < 0 they overlap on (t, -t).
    if lo.any():
        if t >= 0:
            out[lo] = 0.0
        else:
            f = lambda x: (x - t) * (-x - t) * phi(x)
            out[lo] = integrate.quad(f, t, -t)[0]
    return out


def relu_cross_hermite(rho, t, kmax: int = 60):
    """Mehler series ``sum_k c_k^2 rho^k / k!`` with ``c_0 = m1``, ``c_1 = Phibar(t)``,
    ``c_k = phi(t) He_{k-2}(t)``. Converges for ``|rho| < 1``; used as a check."""
    rho = np.asarray(rho, dtype=np.float64)
    c = np.empty(kmax + 1)
    c[0], c[1] = relu_m1(t), Phibar(t)
    he = [1.0, t]                                      # He_0, He_1
    for k in range(2, kmax + 1):
        if k - 2 >= len(he):
            n = len(he)
            he.append(t * he[n - 1] - (n - 1) * he[n - 2])
        c[k] = phi(t) * he[k - 2]
    k = np.arange(kmax + 1)
    w = c * c / np.exp(special.gammaln(k + 1))
    return np.polynomial.polynomial.polyval(rho, w)


def kernel_cos(rho, t):
    """Cosine kernel of the thresholded code, large D: ``C(rho) / m2``. Floor ``a(t)`` at ``rho = 0``."""
    return relu_cross(rho, t) / relu_m2(t)


def kernel_pearson(rho, t):
    """Pearson kernel of the thresholded code, large D: ``(C - m1^2) / (m2 - m1^2)``. Zero at ``rho = 0``."""
    m1, m2 = relu_m1(t), relu_m2(t)
    return (relu_cross(rho, t) - m1 * m1) / (m2 - m1 * m1)


def rho_at_level(level: float, t: float | None, form: str) -> float:
    """The input correlation ``rho*`` at which a kernel reaches ``level``.

    ``form`` is ``raw`` (the kernel is ``rho``), ``cos`` or ``pearson`` (the
    thresholded kernels at threshold ``t``; ``t = 0`` is rectification).
    """
    if form == "raw":
        return float(level)
    f = {"cos": kernel_cos, "pearson": kernel_pearson}[form]
    lo = 0.0
    if f(np.array([lo]), t)[0] >= level:
        raise ValueError(f"{form} kernel at t={t} never falls to {level} for rho >= 0")
    return float(optimize.brentq(lambda x: f(np.array([x]), t)[0] - level, lo, 1.0, xtol=1e-13))


def kernel_width(level: float, r: float, t: float | None = None, form: str = "raw") -> float:
    """Full width (cells) where the kernel falls to ``level``: ``2 r sqrt(2 ln(1/rho*))``."""
    return gaussian_width(rho_at_level(level, t, form), r)


def r_for_width(width: float, level: float, t: float | None, form: str) -> float:
    """The ``r`` that gives a full width ``width`` at ``level`` for this threshold and form."""
    return width / kernel_width(level, 1.0, t, form)


def far_slope(t: float, form: str) -> float:
    """``dK/drho`` at ``rho = 0``: how a far-field raw alias maps into the rate kernel.

    ``c_1^2 / m2`` for cosine, ``c_1^2 / (m2 - m1^2)`` for Pearson, ``c_1 = Phibar(t)``.
    """
    if form == "raw":
        return 1.0
    c1 = Phibar(t)
    if form == "cos":
        return float(c1 * c1 / relu_m2(t))
    return float(c1 * c1 / (relu_m2(t) - relu_m1(t) ** 2))


# ---------------------------------------------------------------------------
# 3. Rice formulas in 1D, the maximum, definition A
# ---------------------------------------------------------------------------

def nu_1d(u, r):
    """Upcrossings of level ``u`` per cell, ``exp(-u^2/2) / (2 pi r)`` (lambda_2 = 1/r^2)."""
    return np.exp(-0.5 * np.asarray(u, dtype=np.float64) ** 2) / (2.0 * math.pi * r)


def mean_len_1d(u, r):
    """Mean excursion length above ``u``: ``Phibar(u) / nu = 2 pi r Phibar(u) exp(u^2/2)``."""
    u = np.asarray(u, dtype=np.float64)
    return 2.0 * math.pi * r * Phibar(u) * np.exp(0.5 * u * u)


def max_cdf_1d(u, length, r):
    """``P(max over a stretch of `length` cells <= u) ~ Phi(u) exp(-length nu(u))``.

    The Poisson-clumping (Rice) approximation; good once the stretch spans a
    few ``2 pi r``. The ``Phi(u)`` factor (start below ``u``) keeps it a CDF at
    negative ``u``.
    """
    u = np.asarray(u, dtype=np.float64)
    return special.ndtr(u) * np.exp(-length * nu_1d(u, r))


_UGRID = np.linspace(-6.0, 9.0, 6001)


def max_density_1d(length, r, grid=_UGRID):
    F = max_cdf_1d(grid, length, r)
    f = np.gradient(F, grid)
    return grid, np.clip(f, 0, None)


def expected_max_1d(length, r) -> float:
    g, f = max_density_1d(length, r)
    return float(np.trapezoid(g * f, g) / np.trapezoid(f, g))


def var_max_1d(length, r) -> float:
    """Variance of the maximum over a stretch, from the same Rice distribution."""
    g, f = max_density_1d(length, r)
    Z = np.trapezoid(f, g)
    m = np.trapezoid(g * f, g) / Z
    return float(np.trapezoid((g - m) ** 2 * f, g) / Z)


def gumbel_max_1d(length, r) -> float:
    """``m* + 0.5772 / m*`` with ``m* = sqrt(2 ln(length / 2 pi r))``: the textbook shortcut."""
    m = math.sqrt(2.0 * math.log(length / (2.0 * math.pi * r)))
    return m + 0.5772156649 / m


def level_A(t: float, peak, frac: float = DEF_A_FRAC):
    """Definition A's level in SDs: ``u_A = t + frac (peak - t)`` (the rate is ``relu(y - t)``)."""
    return t + frac * (np.asarray(peak, dtype=np.float64) - t)


def min_len_cut(mean_len, min_len: float):
    """Fields shorter than ``min_len`` cells dropped: (kept fraction, mean length of kept fields).

    Excursion lengths are taken as Rayleigh with the Rice mean. Rayleigh is the
    high-level limit (a high excursion is a parabolic cap, its length squared
    exponential); matching its mean to the Rice mean extends it to moderate
    levels. With ``s = mean_len / sqrt(pi/2)``: kept ``= exp(-m^2 / 2 s^2)`` and
    ``E[l | l > m] = m + sqrt(2 pi) s exp(m^2 / 2 s^2) Phibar(m / s)``.
    """
    L = np.asarray(mean_len, dtype=np.float64)
    if min_len <= 0:
        return np.ones_like(L), L
    sR = L / math.sqrt(math.pi / 2)
    x = min_len / sR
    keep = np.exp(-0.5 * x * x)
    kept_mean = min_len + SQ2PI * sR * np.exp(0.5 * x * x) * Phibar(x)
    return keep, kept_mean


def def_A_1d(t: float, length, r, frac: float = DEF_A_FRAC, min_len: float = 3.0) -> dict:
    """Definition A on a 1D loop of ``length`` cells, from the GRF maximum's distribution.

    Each unit's level is ``u_A(M) = t + frac (M - t)`` with ``M`` its maximum;
    units with ``M <= t`` are silent. Pooling fields over units,

        W_A = E[Phibar(u_A)] / E[nu(u_A)]          (total length over total count),
        fields per unit = length E[nu(u_A)],   field coverage = E[Phibar(u_A)].

    ``crude`` plugs ``E[M]`` in instead of averaging. ``*_cont`` are the
    continuum values; the headline values also drop fields shorter than
    ``min_len`` cells (definition A's 3-cell minimum, ``min_len_cut``), which
    matters once W_A is a few cells (r below about 8).
    """
    g, f = max_density_1d(length, r)
    act = g > t
    w = f * act
    Z = np.trapezoid(w, g)
    uA = level_A(t, g, frac)
    nu = nu_1d(uA, r)
    keep, kmean = min_len_cut(mean_len_1d(uA, r), min_len)
    e_bar = np.trapezoid(w * Phibar(uA), g) / Z
    e_nu = np.trapezoid(w * nu, g) / Z
    e_nu_cut = np.trapezoid(w * nu * keep, g) / Z
    e_len_cut = np.trapezoid(w * nu * keep * kmean, g) / Z
    EM = float(np.trapezoid(g * f, g) / np.trapezoid(f, g))
    uc = float(level_A(t, EM, frac))
    return {"E_max": EM, "u_A_crude": uc, "W_A_crude": float(mean_len_1d(uc, r)),
            "W_A_cont": float(e_bar / e_nu), "fields_per_active_unit_cont": float(length * e_nu),
            "W_A": float(e_len_cut / e_nu_cut),
            "fields_per_unit": float(length * e_nu_cut * Z),
            "fields_per_active_unit": float(length * e_nu_cut),
            "active_frac": float(Z), "field_coverage": float(e_len_cut),
            "field_coverage_cont": float(e_bar),
            "u_A_mean": float(np.trapezoid(w * uA, g) / Z)}


# ---------------------------------------------------------------------------
# 4. Two dimensions: Euler characteristic
# ---------------------------------------------------------------------------

def ec_density_2d(u, r):
    """Expected Euler characteristic of the excursion set per cell^2: ``u exp(-u^2/2) / ((2 pi)^1.5 r^2)``."""
    u = np.asarray(u, dtype=np.float64)
    return u * np.exp(-0.5 * u * u) / ((2.0 * math.pi) ** 1.5 * r * r)


def mean_area_2d(u, r):
    """Mean field area ``Phibar(u) / EC density = (2 pi)^1.5 r^2 Phibar(u) exp(u^2/2) / u``.

    Counts fields as Euler characteristic (components minus holes), so it
    holds only where fields are simply connected and do not percolate, about
    ``u > 1`` (``p < 0.16``). At ``u <= 0`` it is undefined.
    """
    u = np.asarray(u, dtype=np.float64)
    return Phibar(u) / ec_density_2d(u, r)


def ec_square(u, side, r):
    """Expected EC of the excursion set in a ``side x side`` square (Adler and Taylor 2007).

    ``Phibar(u) + (perimeter / 2) exp(-u^2/2) / (2 pi r) + area * EC density``.
    At high ``u`` this is the expected number of fields, boundary fields included.
    """
    u = np.asarray(u, dtype=np.float64)
    return (Phibar(u) + 2.0 * side * np.exp(-0.5 * u * u) / (2.0 * math.pi * r)
            + side * side * ec_density_2d(u, r))


def square_fields(u, side, r, min_area: float = 9.0) -> dict:
    """Fields in a ``side x side`` arena at level ``u``: EC count, wall-cut mean area, and a minimum-area cut.

    The mean area of all fields in the box, wall-cut ones included, is
    ``(1 - Phi(u)) side^2 / EC``. A high excursion is a paraboloid cap whose
    area is proportional to its overshoot, so areas are taken as exponential:
    dropping fields under ``min_area`` keeps ``exp(-min_area / A)`` of them,
    with mean area ``A + min_area``. Valid where the EC counts fields, ``u >= 0.5``.
    """
    ec = float(ec_square(u, side, r))
    A = float(Phibar(u)) * side * side / ec
    keep = math.exp(-min_area / A) if min_area > 0 else 1.0
    return {"ec": ec, "area_box": A, "count": ec * keep, "area": A + min_area if min_area > 0 else A}


def expected_max_2d(side, r, grid=None) -> float:
    """``E[max]`` over a square, with ``P(max > u) = 1 - exp(-E chi(A_u))`` (Poisson clumping).

    Checked against GRF samples: within 3% for sides of 8 r and more, about
    10% low at 4 r and 17% low at 2 r, where the maximum is not a rare event.
    """
    grid = np.linspace(0.0, 9.0, 4501) if grid is None else grid
    return float(np.trapezoid(1.0 - np.exp(-ec_square(grid, side, r)), grid))


def expected_max_area(area, r, grid=None) -> float:
    """``E[max]`` of a unit-variance field with Gaussian covariance over ``area`` cells^2,
    interior only (no boundary terms), by the same Poisson-clumping form."""
    grid = np.linspace(0.0, 12.0, 6001) if grid is None else grid
    ec = area * ec_density_2d(grid, r) + Phibar(grid)
    return float(np.trapezoid(1.0 - np.exp(-ec), grid))


# ---------------------------------------------------------------------------
# 6. Finite M
# ---------------------------------------------------------------------------

def far_kernel_sd(M: int) -> float:
    """SD of ``(1/M) sum_i cos(omega_i . d)`` far from the origin: ``1/sqrt(2M) = 1/sqrt(D)``."""
    return 1.0 / math.sqrt(2.0 * M)


def alias_prediction(M: int, r: float, npos: int = NPOS, excl_radius: float = 0.0) -> dict:
    """Expected largest far-field value of the raw kernel over the whole scaffold.

    The far field of ``K_M(d)`` is a field in ``d`` with SD ``1/sqrt(2M)`` and,
    to leading order, Gaussian covariance of scale ``r`` (so ``lambda_2 = 1/r^2``).
    ``K(d) = K(-d)``, so half the torus is independent: area ``npos^2 / 2``
    minus the excluded disc. Then ``alias ~ u*(A / r^2) / sqrt(2M)`` with
    ``u*`` the EC-heuristic expected maximum of a unit field over that area.
    Writing ``u* = sqrt(2 ln N_eff)`` gives ``alias = sqrt(ln N_eff / M)``.
    """
    area = npos * npos / 2.0 - math.pi * excl_radius ** 2 / 2.0
    u = expected_max_area(area, r)
    return {"area": area, "u_star": u, "alias": u * far_kernel_sd(M),
            "N_eff": math.exp(u * u / 2.0), "area_over_r2": area / r ** 2}


def lambda2_cv2(M: int, a_cv2: float = 2.0) -> float:
    """Squared CV, across rotated units of one draw, of a unit's second spectral moment along an axis.

    A rotated unit is ``sum_i A_i cos(omega_i . p + phase_i)``; every unit
    shares the draw's M frequencies and differs only in its weights
    ``w_i = A_i^2 / sum A^2``, a row of an orthogonal matrix squared and paired.
    For a Haar-like rotation the weights are Dirichlet(1, ..., 1), and for fixed
    ``a_i = r^2 omega_ix^2``,

        Var_u(sum_i w_i a_i) = s_a^2 / (M + 1),    s_a^2 = mean_i a_i^2 - (mean_i a_i)^2,

    so ``CV^2(lambda_2) = (s_a^2 / abar^2) / (M + 1)``. ``a_cv2 = s_a^2 / abar^2``
    is 2 in expectation (``omega_ix^2 r^2`` is chi-square with 1 dof) but varies
    a lot between draws at small M; pass the draw's own value when known.
    """
    return a_cv2 / (M + 1.0)


def heterogeneity_fano_excess(M: int, mean_count: float, a_cv2: float = 2.0) -> float:
    """Extra variance over mean in 1D from unit-to-unit spread in field rate.

    Field rate ``nu_u ~ sqrt(lambda_2,u)``, so ``Var ln nu = CV^2(lambda_2) / 4``
    and the excess is ``mean_count * a_cv2 / (4 (M + 1))``, about ``mean_count / (2 (M + 1))``.
    """
    return mean_count * lambda2_cv2(M, a_cv2) / 4.0


def edgeworth_coverage(t: float, M: int) -> float:
    """Coverage of a unit made of M random-phase sinusoids with Rayleigh amplitudes.

    The unit's excess kurtosis over space is ``-(3/2) sum A^4 / (sum A^2)^2``,
    about ``-3/M``. The first Edgeworth term gives
    ``Phibar(t) - (t^3 - 3t) phi(t) / (8 M)``: lighter tails, so slightly less
    coverage than ``p`` at high thresholds.
    """
    return float(Phibar(t) - (t ** 3 - 3 * t) * phi(t) / (8.0 * M))


# ---------------------------------------------------------------------------
# 7. Native plane-wave code, rectified
# ---------------------------------------------------------------------------

# relu(cos th) = 1/pi + cos(th)/2 + sum_{k even >= 2} a_k cos(k th),
# a_k = (2/pi) (-1)^(k/2 + 1) / (k^2 - 1).
def _relu_cos_coeffs(kmax: int = 400) -> np.ndarray:
    a = np.zeros(kmax + 1)
    a[0], a[1] = 1.0 / math.pi, 0.5
    for k in range(2, kmax + 1, 2):
        a[k] = (2.0 / math.pi) * (-1) ** (k // 2 + 1) / (k * k - 1)
    return a


_A_RELU_COS = _relu_cos_coeffs()


def native_kernel(d, r, form: str = "cos"):
    """Population kernel of the rectified native code against distance ``d`` (cells).

    Averaging a unit's phase, ``E relu(cos th) relu(cos(th + delta)) =
    a_0^2 + sum_k (a_k^2 / 2) cos(k delta)``, and ``E cos(k omega . d) =
    rho^(k^2)`` with ``rho = exp(-d^2 / 2 r^2)``. Cosine: divide by 1/4.
    Pearson removes the ``a_0^2`` term. Floor 4/pi^2 (cosine) or 0 (Pearson).
    """
    rho = np.exp(-0.5 * (np.asarray(d, dtype=np.float64) / r) ** 2)
    k = np.arange(1, _A_RELU_COS.size)
    w = _A_RELU_COS[1:] ** 2 / 2.0
    s = (w[None, :] * rho.reshape(-1, 1) ** (k[None, :] ** 2)).sum(axis=1).reshape(rho.shape)
    if form == "cos":
        return (_A_RELU_COS[0] ** 2 + s) / 0.25
    if form == "pearson":
        return s / w.sum()
    raise ValueError(form)


NATIVE_TR = 4.0 / math.pi ** 2          # Treves-Rolls of relu(cos) over whole periods
NATIVE_COVERAGE = 0.5


def native_field_fraction(frac: float = DEF_A_FRAC) -> float:
    """Share of each period inside a definition-A field: ``arccos(frac) / pi`` (0.436 at 0.2)."""
    return math.acos(frac) / math.pi


def native_mean_width_1d(r: float, frac: float = DEF_A_FRAC) -> float:
    """Field-weighted mean field width along a line, ``2 arccos(frac) r sqrt(pi/2)`` (3.43 r).

    A wave of projected frequency ``k`` has fields ``2 arccos(frac) / |k|`` wide,
    ``|k| / 2 pi`` per cell; with ``k ~ N(0, 1/r^2)`` the pooled mean is
    ``2 arccos(frac) / E|k|``. The unit-weighted width ``~ 1/|k|`` has a Pareto
    tail of index 1 (infinite mean), cut off by the scaffold at ``|n| = 1``.
    """
    return 2.0 * math.acos(frac) * r * math.sqrt(math.pi / 2.0)


def native_fano_1d(length: float, r: float) -> float:
    """Variance over mean of native field counts on a line: ``(L / 2 pi r)(1 - 2/pi) / sqrt(2/pi)``.

    Counts are ``L |k| / 2 pi`` (deterministic given the wave), so all the
    dispersion is the spread of ``|k|`` across units and grows with ``L``.
    """
    return length / (2 * math.pi * r) * (1 - 2 / math.pi) / math.sqrt(2 / math.pi)


def native_tr_track(length: float, r: float, n_k: int = 400, n_ph: int = 64) -> dict:
    """Treves-Rolls of rectified plane waves on a track of ``length`` cells.

    Averages over ``k ~ N(0, 1/r^2)`` (a quantile grid) and a uniform phase.
    ``pooled`` is ``(E mean)^2 / E mean^2`` over all units (4/pi^2 for any
    length); ``active`` is the mean of each unit's own ratio over units that
    fire somewhere on the track, the suite's convention.
    """
    x = np.arange(int(round(length)), dtype=np.float64)
    q = (np.arange(n_k) + 0.5) / n_k
    k = -special.ndtri(q) / r
    ph = np.linspace(0, 2 * math.pi, n_ph, endpoint=False)
    m1 = np.empty((n_k, n_ph))
    m2 = np.empty((n_k, n_ph))
    for i, kk in enumerate(k):
        R = np.maximum(np.cos(kk * x[None, :] + ph[:, None]), 0.0)
        m1[i], m2[i] = R.mean(-1), (R * R).mean(-1)
    act = m2 > 1e-12
    tr = np.where(act, m1 * m1 / np.where(act, m2, 1), 0.0)
    return {"active": float(tr[act].mean()), "active_frac": float(act.mean()),
            "pooled": float(m1.mean() ** 2 / m2.mean())}


# ---------------------------------------------------------------------------
# 8. Fano factor of upcrossing counts
# ---------------------------------------------------------------------------

def _gauss_cov(tau, r):
    x = np.asarray(tau, dtype=np.float64) / r
    rho = np.exp(-0.5 * x * x)
    d1 = -x / r * rho                              # rho'
    d2 = (x * x - 1.0) / (r * r) * rho             # rho''
    return rho, d1, d2


def upcross_pair_density(tau, u: float, r: float, n_quad: int = 200):
    """Second-moment Rice density ``f2(tau)``: upcrossings of ``u`` at 0 and at ``tau``.

    ``f2 = p(u, u) E[Y'(0)+ Y'(tau)+ | Y(0) = Y(tau) = u]`` for a unit-variance
    stationary Gaussian process with ``rho(tau) = exp(-tau^2 / 2 r^2)``. Given
    the two values, the derivatives have means ``-+ rho' u / (1 + rho)`` and
    covariance ``[[l2 - rho'^2/(1-rho^2), -rho'' - rho rho'^2/(1-rho^2)], [., .]]``,
    ``l2 = 1/r^2``. The positive-part product moment is a 1D quadrature.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    rho, d1, d2 = _gauss_cov(tau, r)
    l2 = 1.0 / (r * r)
    om = 1.0 - rho * rho
    var = l2 - d1 * d1 / om
    cov = -d2 - rho * d1 * d1 / om
    var = np.maximum(var, 1e-300)
    mu0 = -d1 * u / (1 + rho)
    mu1 = -mu0
    sd = np.sqrt(var)
    # E[D0+ D1+] = int_0^inf x N(x; mu0, var) E[(D1)+ | D0 = x] dx
    xq, wq = np.polynomial.legendre.leggauss(n_quad)
    hi = np.maximum(mu0, 0) + 12 * sd
    x = 0.5 * (xq[None, :] + 1.0) * hi[:, None]
    w = 0.5 * wq[None, :] * hi[:, None]
    dens = np.exp(-0.5 * ((x - mu0[:, None]) / sd[:, None]) ** 2) / (SQ2PI * sd[:, None])
    beta = (cov / var)[:, None]
    cm = mu1[:, None] + beta * (x - mu0[:, None])
    cv = np.maximum(var[:, None] - cov[:, None] ** 2 / var[:, None], 0.0)
    cs = np.sqrt(cv)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(cs > 0, cm / np.where(cs > 0, cs, 1), np.sign(cm) * np.inf)
    pos = np.where(cs > 0, cs * phi(z) + cm * special.ndtr(z), np.maximum(cm, 0))
    e = (w * x * dens * pos).sum(axis=1)
    p_uu = np.exp(-u * u / (1 + rho)) / (2 * math.pi * np.sqrt(om))
    return p_uu * e


def fano_upcross(u: float, r: float, length: float | None = None, *, periodic: bool = True,
                 tau_max: float = 12.0, n_tau: int = 1500) -> float:
    """Variance over mean of the number of upcrossings of ``u``.

    ``length=None`` gives the long-environment limit
    ``1 + (2 / nu) int_0^inf (f2(tau) - nu^2) dtau``. On a loop of ``length``
    cells (``periodic``) the same integral runs to ``length / 2``; on a segment
    it is weighted by ``(length - tau) / length``.
    """
    nu = float(nu_1d(u, r))
    tmax = tau_max * r if length is None else (min(length / 2, tau_max * r) if periodic
                                                else min(length, tau_max * r))
    tau = np.linspace(1e-3 * r, tmax, n_tau)
    g = upcross_pair_density(tau, u, r) - nu * nu
    if length is not None and not periodic:
        g = g * (1 - tau / length)
    return float(1.0 + 2.0 / nu * np.trapezoid(g, tau))


def fano_def_A(t: float, length: float, r: float, *, M: int | None = None,
               a_cv2: float = 2.0, frac: float = DEF_A_FRAC, periodic: bool = True) -> dict:
    """Variance over mean of definition-A field counts on a 1D environment of ``length`` cells.

    Low levels (``u_A`` below about 1, e.g. rectification, ``t = 0``): three
    parts, added as independent contributions to ``Var ln E[N | unit]``:

    ``F(u_A)``
        the upcrossing Fano factor at the mean level (``fano_upcross``);
    ``N frac^2 u_A^2 Var(M_env)``
        the level is ``t + frac (M_env - t)`` with ``M_env`` the unit's own
        maximum, and ``d ln nu / du = -u``; this grows with the number of fields
        per unit ``N``, i.e. with the environment's length;
    ``N a_cv2 / (4 (M + 1))``
        finite M: units of one draw differ in spectral width (absent for GRFs).

    High levels (sparse codes, ``u_A`` above about 1.2): ``total_peaks``,
    ``F (1 - 1/N) + (N frac u_A - E[M])^2 Var(M) / N`` plus the same finite-M
    term. Checked against GRF loops: the low form within 0.02 at ``t = 0``, the
    high form within 0.04 at ``p = 0.1`` and ``0.02``.
    """
    d = def_A_1d(t, length, r, frac)
    u = d["u_A_mean"]
    N = d["fields_per_active_unit"]
    F = fano_upcross(u, r, length, periodic=periodic)
    vm = var_max_1d(length, r)
    level = N * frac ** 2 * u * u * vm
    # Peak-process version, for high u_A where fields are rare peaks. The
    # field holding the maximum is always counted, and a unit with more fields
    # also has a higher maximum: E[N | M] ~ 1 + L (nu(u_A(M)) - nu(M)), whose
    # slope in M is -N frac u_A + M L nu(M) ~ -N frac u_A + E[M], and the
    # other N - 1 fields carry the Fano factor F.
    EM = d["E_max"]
    level_peaks = (N * frac * u - EM) ** 2 * vm / N
    het = 0.0 if M is None else heterogeneity_fano_excess(M, N, a_cv2)
    return {"F": F, "level_term": level, "level_term_peaks": level_peaks,
            "heterogeneity_term": het, "total": F + level + het,
            "total_peaks": F * (1 - 1 / N) + level_peaks + het, "u_A": u, "fields_per_active_unit": N}


__all__ = [n for n in dir() if not n.startswith("_") and n not in (
    "annotations", "math", "np", "integrate", "optimize", "special")]
