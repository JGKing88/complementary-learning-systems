"""The measurements the place-cell literature makes, as pure functions on rate maps.

Every function here takes plain arrays and parameters and knows nothing about
encoders, the scaffold or the literature targets, so the same code can run on
real recordings. The conventions are:

* a 1D rate map is ``(units, positions)``; a 2D map is ``(units, S, S)``;
* a population-vector (PV) set is ``(units, positions)``, one column per place;
* lengths are in bins (for the models, scaffold cells). Callers convert to
  units of the mean field width.

Field definitions are parameters (``FieldDef``). Two families are used:

``peak_fraction``
    a field is a contiguous run of bins at or above ``frac`` times the unit's
    peak in this environment, at least ``min_len`` bins long (``min_area`` in
    2D). This is the rule of Kjelstrup 2008, Henriksen 2010 and Eliav 2021
    (20% of peak).
``absolute``
    rates are multiplied by ``rate_scale`` (Hz per model unit), smoothed with a
    Gaussian of ``smooth_sd`` bins, and a field is a contiguous run at or above
    ``thresh_hz``, at least ``min_len`` bins long. This is Rich 2014's rule
    (2 Hz, 15 cm minimum, 10 cm smoothing).

Population similarity has two constructions. ``cosine`` is the normalized dot
product of the raw vectors and is unchanged by any orthogonal remixing of the
units. ``pearson`` is the correlation across units of two PVs, as the papers
compute it; centering each PV across units makes it basis dependent.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage, optimize, special, stats


# ---------------------------------------------------------------------------
# Rate conventions for signed units
# ---------------------------------------------------------------------------

CONVENTIONS = ("identity", "rectify", "onoff")


def apply_convention(z: np.ndarray, convention: str, *, axis: int = 0) -> np.ndarray:
    """Turn activity into non-negative rates.

    ``identity`` leaves it alone (for codes that are already non-negative),
    ``rectify`` keeps the positive part, and ``onoff`` splits each unit into an
    "on" cell ``max(z, 0)`` and an "off" cell ``max(-z, 0)``, stacked along
    ``axis`` (on cells first), so a signed code of U units becomes 2U cells.
    """
    if convention == "identity":
        return z
    if convention == "rectify":
        return np.maximum(z, 0)
    if convention == "onoff":
        return np.concatenate([np.maximum(z, 0), np.maximum(-z, 0)], axis=axis)
    raise ValueError(f"unknown convention {convention!r}")


def random_orthogonal(n: int, seed: int) -> np.ndarray:
    """A Haar-distributed ``n x n`` orthogonal matrix (QR of a Gaussian, signs fixed)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))[None, :]


# ---------------------------------------------------------------------------
# Field definitions and detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldDef:
    """How a field is found. Lengths and areas are in bins."""

    name: str
    kind: str                 # "peak_fraction" or "absolute"
    frac: float = 0.2         # peak_fraction: edge at frac * peak
    thresh_hz: float = 2.0    # absolute: edge at this rate after scaling
    rate_scale: float = 1.0   # absolute: Hz per unit of model activity
    smooth_sd: float = 0.0    # Gaussian SD in bins, applied before thresholding
    min_len: float = 1.0      # 1D minimum field length
    min_area: float = 1.0     # 2D minimum field area
    min_peak_hz: float = 0.0  # a field's own peak, after scaling, must reach this

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "FieldDef":
        return replace(self, **kw)


# Above this SD (bins) the Gaussian is applied by FFT: the direct filter costs
# about 8 SD operations per bin, the FFT a few log(n).
FFT_SD = 6.0


def _gauss_axis(x: np.ndarray, sd: float, axis: int) -> np.ndarray:
    """``ndimage.gaussian_filter1d(x, sd, axis, mode="constant")``, by FFT for wide kernels.

    The kernel is the one ndimage builds (truncated at 4 SD, summed to 1), so
    the two paths agree to rounding; ``test_fft_smoothing_matches_direct``
    checks it.
    """
    if sd <= FFT_SD:
        return ndimage.gaussian_filter1d(x, sd, axis=axis, mode="constant", cval=0.0)
    from scipy.signal import fftconvolve
    radius = int(4.0 * sd + 0.5)
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (t / sd) ** 2)
    w /= w.sum()
    shape = [1] * x.ndim
    shape[axis] = w.size
    return fftconvolve(x, w.reshape(shape), mode="same", axes=axis)


def smooth_1d(rates: np.ndarray, sd: float) -> np.ndarray:
    """Gaussian smoothing along the last axis, renormalized at the ends.

    Dividing by the smoothed ones keeps a constant map constant up to the
    track ends, which is what smoothing spike counts and occupancy separately
    does in the papers.
    """
    if sd <= 0:
        return rates
    num = _gauss_axis(rates.astype(np.float64), sd, -1)
    den = _gauss_axis(np.ones(rates.shape[-1]), sd, -1)
    return num / den


def smooth_2d(rates: np.ndarray, sd: float) -> np.ndarray:
    """Gaussian smoothing over the last two axes, renormalized at the walls."""
    if sd <= 0:
        return rates
    num = _gauss_axis(_gauss_axis(rates.astype(np.float64), sd, -1), sd, -2)
    one = np.ones(rates.shape[-2:])
    den = _gauss_axis(_gauss_axis(one, sd, -1), sd, -2)
    return num / den


def _threshold_mask(r: np.ndarray, fdef: FieldDef, n_spatial: int) -> np.ndarray:
    """Above-threshold bins. ``r`` is already scaled and smoothed."""
    if fdef.kind == "peak_fraction":
        flat = r.reshape(r.shape[0], -1)
        peak = flat.max(axis=1)
        thr = fdef.frac * peak
        shape = (r.shape[0],) + (1,) * n_spatial
        # A unit whose peak is not positive has no field.
        return (r >= thr.reshape(shape)) & (peak.reshape(shape) > 0) & (r > 0)
    if fdef.kind == "absolute":
        return r >= fdef.thresh_hz
    raise ValueError(f"unknown field kind {fdef.kind!r}")


@dataclass
class Fields1D:
    """Fields found on one track. ``unit``, ``start``, ``stop`` (exclusive),
    ``center`` (rate-weighted, in bins) and ``length`` are one entry per field."""

    n_units: int
    length_bins: int
    unit: np.ndarray
    start: np.ndarray
    stop: np.ndarray
    center: np.ndarray

    @property
    def length(self) -> np.ndarray:
        return self.stop - self.start

    @property
    def counts(self) -> np.ndarray:
        return np.bincount(self.unit, minlength=self.n_units)

    def coverage(self) -> np.ndarray:
        """``(units, positions)`` bool: is this bin inside one of the unit's fields."""
        delta = np.zeros((self.n_units, self.length_bins + 1), dtype=np.int32)
        np.add.at(delta, (self.unit, self.start), 1)
        np.add.at(delta, (self.unit, self.stop), -1)
        return np.cumsum(delta[:, :-1], axis=1) > 0


def detect_fields_1d(rates: np.ndarray, fdef: FieldDef) -> Fields1D:
    """Fields of every unit on one linear track. ``rates`` is ``(units, positions)``."""
    rates = np.asarray(rates, dtype=np.float64)
    U, L = rates.shape
    r = smooth_1d(rates * fdef.rate_scale, fdef.smooth_sd)
    mask = _threshold_mask(r, fdef, 1)
    pad = np.zeros((U, 1), dtype=np.int8)
    d = np.diff(np.concatenate([pad, mask.astype(np.int8), pad], axis=1), axis=1)
    su, ss = np.nonzero(d == 1)
    eu, ee = np.nonzero(d == -1)
    # Starts and ends alternate within a row and nonzero() is row-major, so
    # the k-th start pairs with the k-th end.
    assert np.array_equal(su, eu)
    keep = (ee - ss) >= fdef.min_len
    if fdef.min_peak_hz > 0 and su.size:
        flat = np.concatenate([r.reshape(-1), [0.0]])
        bounds = np.stack([su * L + ss, su * L + ee], axis=1).reshape(-1)
        run_max = np.maximum.reduceat(flat, bounds)[0::2]
        keep &= run_max >= fdef.min_peak_hz
    su, ss, ee = su[keep], ss[keep], ee[keep]
    # Rate-weighted centre of each run, from cumulative sums over the flat map.
    w = np.clip(r, 0, None).reshape(-1)
    pos = np.tile(np.arange(L, dtype=np.float64), U)
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cwx = np.concatenate([[0.0], np.cumsum(w * pos)])
    a, b = su * L + ss, su * L + ee
    mass = cw[b] - cw[a]
    center = np.where(mass > 0, (cwx[b] - cwx[a]) / np.where(mass > 0, mass, 1),
                      (ss + ee - 1) / 2.0)
    return Fields1D(U, L, su.astype(np.int64), ss.astype(np.int64), ee.astype(np.int64),
                    center)


@dataclass
class Fields2D:
    """Fields found in one square arena, one entry per field."""

    n_units: int
    side: int
    unit: np.ndarray
    area: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    coverage_frac: np.ndarray      # per position: fraction of units in a field
    unit_coverage: np.ndarray      # per unit: fraction of the arena in its fields

    @property
    def counts(self) -> np.ndarray:
        return np.bincount(self.unit, minlength=self.n_units)


_STRUCT_4 = np.zeros((3, 3, 3), dtype=bool)
_STRUCT_4[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]


def detect_fields_2d(rates: np.ndarray, fdef: FieldDef, *, block: int = 128) -> Fields2D:
    """Fields of every unit in one square arena. ``rates`` is ``(units, S, S)``.

    Fields are 4-connected components (edge neighbours), so two blobs that
    touch only at a corner are separate fields. Units are processed in blocks
    so the label array stays small.
    """
    U, S, S2 = rates.shape
    assert S == S2, "square arenas only"
    units, areas, cxs, cys = [], [], [], []
    cover = np.zeros((S, S), dtype=np.float64)
    unit_cov = np.zeros(U, dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(S, dtype=np.float64), np.arange(S, dtype=np.float64),
                         indexing="ij")
    for u0 in range(0, U, block):
        r = smooth_2d(np.asarray(rates[u0:u0 + block], dtype=np.float64) * fdef.rate_scale,
                      fdef.smooth_sd)
        mask = _threshold_mask(r, fdef, 2)
        lab, n = ndimage.label(mask, structure=_STRUCT_4)
        if n == 0:
            continue
        flat = lab.reshape(-1)
        size = np.bincount(flat, minlength=n + 1).astype(np.float64)
        keep = size >= fdef.min_area
        if fdef.min_peak_hz > 0:
            peak = np.concatenate([[0.0], ndimage.maximum(r, labels=lab, index=np.arange(1, n + 1))])
            keep &= peak >= fdef.min_peak_hz
        keep[0] = False
        uidx = np.repeat(np.arange(r.shape[0], dtype=np.float64), S * S)
        sx = np.bincount(flat, weights=np.tile(xx.reshape(-1), r.shape[0]), minlength=n + 1)
        sy = np.bincount(flat, weights=np.tile(yy.reshape(-1), r.shape[0]), minlength=n + 1)
        su = np.bincount(flat, weights=uidx, minlength=n + 1)
        ids = np.nonzero(keep)[0]
        units.append(u0 + np.rint(su[ids] / size[ids]).astype(np.int64))
        areas.append(size[ids])
        cxs.append(sx[ids] / size[ids])
        cys.append(sy[ids] / size[ids])
        inside = keep[lab]                                  # (block, S, S)
        cover += inside.sum(axis=0)
        unit_cov[u0:u0 + r.shape[0]] = inside.reshape(r.shape[0], -1).mean(axis=1)
    cat = (lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt))
    return Fields2D(U, S, cat(units, np.int64), cat(areas, np.float64),
                    cat(cxs, np.float64), cat(cys, np.float64), cover / U, unit_cov)


# ---------------------------------------------------------------------------
# Field-count distributions
# ---------------------------------------------------------------------------

def _nb_loglik(x: np.ndarray, r: float, mu: float) -> float:
    if mu <= 0:
        return 0.0 if np.all(x == 0) else -np.inf
    return float(np.sum(special.gammaln(x + r) - special.gammaln(r) - special.gammaln(x + 1)
                        + r * np.log(r / (r + mu)) + x * np.log(mu / (r + mu))))


def _pois_loglik(x: np.ndarray, lam: float) -> float:
    if lam <= 0:
        return 0.0 if np.all(x == 0) else -np.inf
    return float(np.sum(x * np.log(lam) - lam - special.gammaln(x + 1)))


def fit_count_models(counts) -> dict:
    """Poisson and negative binomial fits by maximum likelihood, and their comparison.

    The negative binomial is parameterized as Rich 2014 reports it: shape
    ``r`` and ``p = r / (r + mean)``, so ``P(0) = p ** r`` and the variance is
    ``mean / p``. Its MLE mean is the sample mean, so only ``r`` is searched.
    When the counts are not overdispersed the MLE runs to ``r = inf`` (the
    Poisson limit) and ``r`` is reported as ``inf``.

    The likelihood-ratio statistic is compared with the boundary null
    ``0.5 chi2(0) + 0.5 chi2(1)``, because ``1/r = 0`` sits on the edge of the
    parameter space.
    """
    x = np.asarray(counts, dtype=np.float64).reshape(-1)
    n = x.size
    mean = float(x.mean()) if n else float("nan")
    var = float(x.var()) if n else float("nan")      # MLE (1/n) variance
    out = {"n": int(n), "mean": mean, "var": var,
           "var_over_mean": var / mean if mean > 0 else float("nan"),
           "frac_zero": float(np.mean(x == 0)) if n else float("nan")}
    if n == 0 or mean == 0:
        out.update(poisson=None, negbin=None, lr_stat=None, lr_p=None, delta_aic=None)
        return out
    llp = _pois_loglik(x, mean)
    pois = {"lambda": mean, "loglik": llp, "aic": 2 * 1 - 2 * llp}
    if var <= mean:
        r_hat, lln = math.inf, llp
    else:
        res = optimize.minimize_scalar(lambda lr: -_nb_loglik(x, math.exp(lr), mean),
                                       bounds=(-12.0, 14.0), method="bounded",
                                       options={"xatol": 1e-7})
        r_hat, lln = math.exp(res.x), -res.fun
        if res.x > 13.9:
            r_hat, lln = math.inf, llp
    p_hat = (r_hat / (r_hat + mean)) if math.isfinite(r_hat) else 1.0
    negbin = {"r": r_hat, "p": p_hat, "mean": mean, "loglik": lln, "aic": 2 * 2 - 2 * lln,
              "poisson_limit": not math.isfinite(r_hat)}
    lr = max(0.0, 2 * (lln - llp))
    out.update(poisson=pois, negbin=negbin, lr_stat=lr,
               lr_p=float(0.5 * stats.chi2.sf(lr, 1)) if lr > 0 else 1.0,
               delta_aic=pois["aic"] - negbin["aic"])
    return out


def sample_gamma_poisson(n: int, r: float, beta: float, length: float, seed: int) -> np.ndarray:
    """Counts from Rich's generative model: rate ~ Gamma(shape r, rate beta), count ~ Poisson(rate * length)."""
    rng = np.random.default_rng(seed)
    lam = rng.gamma(r, 1.0 / beta, size=n)
    return rng.poisson(lam * length)


# ---------------------------------------------------------------------------
# Within-unit field layout
# ---------------------------------------------------------------------------

def unit_layouts(fields: Fields1D, *, min_fields: int = 6) -> dict:
    """Per-unit gap CV, normalized gaps and uniformity p-values (see ``within_unit_layout``)."""
    cvs, ks_p, pooled = [], [], []
    L = fields.length_bins
    order = np.lexsort((fields.center, fields.unit))
    u, c = fields.unit[order], fields.center[order]
    bounds = np.flatnonzero(np.diff(u)) + 1
    for cu in (np.split(c, bounds) if c.size else []):
        if cu.size < min_fields:
            continue
        g = np.diff(cu)
        if g.mean() <= 0:
            continue
        cvs.append(float(g.std() / g.mean()))
        pooled.append(g / g.mean())
        ks_p.append(float(stats.kstest(cu / L, "uniform").pvalue))
    return {"cv": cvs, "ks_p": ks_p, "gaps": pooled}


def summarize_layouts(parts: list[dict], *, min_fields: int = 6) -> dict:
    """Pool ``unit_layouts`` results from several tracks."""
    cvs = [v for p in parts for v in p["cv"]]
    ks_p = [v for p in parts for v in p["ks_p"]]
    pooled = [g for p in parts for g in p["gaps"]]
    if not cvs:
        return {"n_units": 0, "min_fields": min_fields}
    pg = np.concatenate(pooled)
    return {
        "n_units": len(cvs), "min_fields": min_fields,
        "gap_cv_median": float(np.median(cvs)), "gap_cv_mean": float(np.mean(cvs)),
        "gap_cv_pooled": float(pg.std() / pg.mean()),
        "ks_frac_reject_05": float(np.mean(np.asarray(ks_p) < 0.05)),
        "ks_p_median": float(np.median(ks_p)),
    }


def within_unit_layout(fields: Fields1D, *, min_fields: int = 6) -> dict:
    """Gap regularity and positional uniformity for units with many fields.

    For each unit with at least ``min_fields`` fields (Rich 2014 used six):
    the coefficient of variation of the gaps between consecutive field
    centres (exponential gaps give 1, a periodic layout well below 1), and a
    Kolmogorov-Smirnov test of the centres against a uniform distribution on
    the track. ``gap_cv_pooled`` pools every unit's gaps after dividing by
    that unit's mean gap.
    """
    return summarize_layouts([unit_layouts(fields, min_fields=min_fields)],
                             min_fields=min_fields)


def field_size_stats(sizes) -> dict:
    s = np.asarray(sizes, dtype=np.float64)
    if s.size == 0:
        return {"n": 0}
    return {"n": int(s.size), "mean": float(s.mean()), "median": float(np.median(s)),
            "cv": float(s.std() / s.mean()), "p95": float(np.percentile(s, 95)),
            "max": float(s.max())}


def unit_size_ratios(unit: np.ndarray, sizes: np.ndarray) -> list[float]:
    """Largest-to-smallest field size of each unit with two or more fields."""
    if unit.size == 0:
        return []
    order = np.argsort(unit, kind="stable")
    u, s = unit[order], sizes[order]
    bounds = np.flatnonzero(np.diff(u)) + 1
    return [float(g.max() / g.min()) for g in np.split(s, bounds) if g.size >= 2 and g.min() > 0]


def summarize_ratios(ratios) -> dict:
    if not len(ratios):
        return {"n_units": 0}
    return {"n_units": len(ratios), "mean": float(np.mean(ratios)),
            "median": float(np.median(ratios)), "max": float(np.max(ratios))}


def within_unit_size_ratio(unit: np.ndarray, sizes: np.ndarray) -> dict:
    """Largest-to-smallest field size within each unit that has two or more fields (Eliav 2021)."""
    return summarize_ratios(unit_size_ratios(unit, sizes))


def sparsity(rates: np.ndarray) -> np.ndarray:
    """Treves-Rolls (Skaggs) sparsity of each unit, ``mean(r)^2 / mean(r^2)``, uniform occupancy.

    1 for a unit firing equally everywhere, the active fraction for a binary
    map. ``rates`` is ``(units, positions...)`` and non-negative; units that
    never fire give NaN.
    """
    r = np.asarray(rates, dtype=np.float64).reshape(rates.shape[0], -1)
    m1 = r.mean(axis=1)
    m2 = (r * r).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m2 > 0, m1 * m1 / m2, np.nan)


# ---------------------------------------------------------------------------
# Population similarity
# ---------------------------------------------------------------------------

SIMILARITIES = ("cosine", "pearson")


def prepare_columns(V: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Unit-normalize each column (one PV per column) for a dot-product similarity.

    ``cosine`` divides each PV by its norm. ``pearson`` first subtracts the
    PV's mean across units, so that the dot product of two prepared columns is
    their Pearson correlation. Returns the prepared array and a validity mask
    (a PV with zero norm, for example all units silent, has no similarity).
    """
    X = np.asarray(V, dtype=np.float64)
    if method == "pearson":
        X = X - X.mean(axis=0, keepdims=True)
    elif method != "cosine":
        raise ValueError(f"unknown similarity {method!r}")
    norm = np.linalg.norm(X, axis=0)
    valid = norm > 1e-12
    X = X / np.where(valid, norm, 1.0)[None, :]
    X[:, ~valid] = 0.0
    return X, valid


def pair_similarity(A: np.ndarray, B: np.ndarray, method: str) -> np.ndarray:
    """Similarity of column ``i`` of ``A`` with column ``i`` of ``B`` (``(units, n)`` each); NaN if undefined."""
    XA, va = prepare_columns(A, method)
    XB, vb = prepare_columns(B, method)
    s = np.einsum("ij,ij->j", XA, XB)
    return np.where(va & vb, s, np.nan)


def similarity_matrix(V: np.ndarray, method: str) -> np.ndarray:
    """All-pairs PV similarity of the columns of ``V``, as in a PV correlation matrix."""
    X, valid = prepare_columns(V, method)
    S = X.T @ X
    S[~valid, :] = np.nan
    S[:, ~valid] = np.nan
    return S


def lag_profile_1d(V: np.ndarray, method: str, *, circular: bool,
                   max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean PV similarity against separation along one track.

    ``V`` is ``(units, positions)``. ``circular=True`` treats the track as a
    loop (exact for a scaffold row of the full period); otherwise lags only
    pair positions inside the track. Returns ``(similarity, n_pairs)`` for lags
    ``0..max_lag``. Computed with FFTs, so a whole scaffold row is cheap.
    """
    X, valid = prepare_columns(V, method)
    U, L = X.shape
    nfft = L if circular else 2 * L
    num = np.zeros(nfft)
    for u0 in range(0, U, 256):
        F = np.fft.rfft(X[u0:u0 + 256], n=nfft, axis=1)
        num += np.fft.irfft((F * F.conj()).sum(axis=0), n=nfft)
    Fv = np.fft.rfft(valid.astype(np.float64), n=nfft)
    cnt = np.rint(np.fft.irfft(Fv * Fv.conj(), n=nfft))
    max_lag = (L // 2 if circular else L - 1) if max_lag is None else max_lag
    num, cnt = num[:max_lag + 1], cnt[:max_lag + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan), cnt


def bin_curve(dist: np.ndarray, sim: np.ndarray, edges: np.ndarray) -> dict:
    """Mean similarity in distance bins (NaNs ignored)."""
    dist = np.asarray(dist).reshape(-1)
    sim = np.asarray(sim).reshape(-1)
    ok = np.isfinite(sim)
    idx = np.digitize(dist[ok], edges) - 1
    nb = len(edges) - 1
    inb = (idx >= 0) & (idx < nb)
    s = np.bincount(idx[inb], weights=sim[ok][inb], minlength=nb)
    n = np.bincount(idx[inb], minlength=nb)
    d = np.bincount(idx[inb], weights=dist[ok][inb], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"d": (d / n).tolist(), "s": (s / n).tolist(), "n": n.tolist()}


def _first_crossing(d: np.ndarray, s: np.ndarray, level: float) -> float | None:
    """First distance where ``s`` drops to ``level`` or below, linearly interpolated."""
    ok = np.isfinite(s)
    d, s = d[ok], s[ok]
    below = np.nonzero(s <= level)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:
        return float(d[0])
    d0, d1, s0, s1 = d[i - 1], d[i], s[i - 1], s[i]
    return float(d0 + (s0 - level) * (d1 - d0) / (s0 - s1)) if s0 != s1 else float(d1)


def kernel_summary(d, s, *, width: float, far: tuple[float, float] = (10.0, 20.0)) -> dict:
    """Shape of a similarity-against-distance curve, in bins and in units of ``width``.

    ``half_width_05`` is the distance where the curve first falls to 0.5;
    ``full_width_05`` doubles it, because Kjelstrup 2008 reports the band width
    across the diagonal. ``zero_crossing`` is the first distance at or below
    0, or None if the curve never gets there. ``floor`` is the mean of the
    curve between ``far[0]`` and ``far[1]`` field widths (None if the curve
    does not reach that far).
    """
    d = np.asarray(d, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    hw = _first_crossing(d, s, 0.5)
    zc = _first_crossing(d, s, 0.0)
    lo, hi = far[0] * width, far[1] * width
    sel = (d >= lo) & (d <= hi) & np.isfinite(s)
    floor = float(s[sel].mean()) if sel.any() and d.max() >= hi * 0.999 else None
    tail = (d >= lo) & np.isfinite(s)
    def w(x):
        return None if x is None or not (width > 0) else float(x / width)
    return {
        "width_bins": float(width),
        "half_width_05": hw, "full_width_05": None if hw is None else 2 * hw,
        "zero_crossing": zc,
        "half_width_05_W": w(hw), "full_width_05_W": None if hw is None else w(2 * hw),
        "zero_crossing_W": w(zc),
        "floor": floor, "floor_range_W": list(far),
        "tail_min": float(s[tail].min()) if tail.any() else None,
        "tail_max": float(s[tail].max()) if tail.any() else None,
        "max_distance_W": w(float(d[np.isfinite(s)].max())),
    }


# ---------------------------------------------------------------------------
# Active fractions and remapping
# ---------------------------------------------------------------------------

def active_set_overlap(active_a: np.ndarray, active_b: np.ndarray) -> dict:
    """Overlap of two active sets and its value expected if units activate independently.

    ``both_frac`` is the fraction of all units active in both; ``jaccard`` is
    active-in-both over active-in-either. The expectations hold each set's
    size and draw membership independently.
    """
    a = np.asarray(active_a, dtype=bool)
    b = np.asarray(active_b, dtype=bool)
    U = a.size
    pa, pb = a.mean(), b.mean()
    both = float((a & b).mean())
    either = float((a | b).mean())
    e_both = float(pa * pb)
    e_either = float(pa + pb - pa * pb)
    jac = both / either if either > 0 else float("nan")
    e_jac = e_both / e_either if e_either > 0 else float("nan")
    return {"n_units": int(U), "frac_a": float(pa), "frac_b": float(pb),
            "both_frac": both, "both_expected": e_both,
            "both_ratio": both / e_both if e_both > 0 else float("nan"),
            "jaccard": jac, "jaccard_expected": e_jac,
            "jaccard_ratio": jac / e_jac if e_jac > 0 else float("nan")}


def mean_rate_correlation(rates_a: np.ndarray, rates_b: np.ndarray) -> dict:
    """Correlation across units of each unit's mean rate in two environments.

    ``pearson`` uses raw mean rates (Kim 2012). ``log_pearson`` uses log mean
    rates of units active in both (Mizuseki & Buzsaki 2013).
    """
    ma = np.asarray(rates_a, dtype=np.float64).reshape(rates_a.shape[0], -1).mean(axis=1)
    mb = np.asarray(rates_b, dtype=np.float64).reshape(rates_b.shape[0], -1).mean(axis=1)
    out = {"pearson": _corr(ma, mb)}
    pos = (ma > 0) & (mb > 0)
    out["log_pearson"] = _corr(np.log(ma[pos]), np.log(mb[pos])) if pos.sum() > 2 else None
    out["n_log"] = int(pos.sum())
    return out


def _corr(a, b) -> float | None:
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def ratemap_correlations(maps_a: np.ndarray, maps_b: np.ndarray, *, seed: int = 0,
                         n_shuffle: int = 20) -> dict:
    """Per-unit rate-map correlation at matched positions, against a unit-shuffle null.

    ``maps_*`` are ``(units, positions)``. The null correlates each unit's map
    in ``a`` with a different, randomly chosen unit's map in ``b``. The result
    is the share of units whose correlation beats the null's 95th percentile
    (about 5% if maps are unrelated).
    """
    A = np.asarray(maps_a, dtype=np.float64)
    B = np.asarray(maps_b, dtype=np.float64)
    def rowcorr(X, Y):
        Xc = X - X.mean(axis=1, keepdims=True)
        Yc = Y - Y.mean(axis=1, keepdims=True)
        den = np.linalg.norm(Xc, axis=1) * np.linalg.norm(Yc, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, (Xc * Yc).sum(axis=1) / den, np.nan)
    obs = rowcorr(A, B)
    rng = np.random.default_rng(seed)
    U = A.shape[0]
    null = []
    for _ in range(n_shuffle):
        perm = rng.permutation(U)
        same = perm == np.arange(U)
        perm[same] = (perm[same] + 1) % U
        null.append(rowcorr(A, B[perm]))
    null = np.concatenate(null)
    null = null[np.isfinite(null)]
    thr = float(np.percentile(null, 95)) if null.size else float("nan")
    ok = np.isfinite(obs)
    return {"median": float(np.median(obs[ok])) if ok.any() else None,
            "mean": float(np.mean(obs[ok])) if ok.any() else None,
            "null_p95": thr, "frac_above_null_p95": float(np.mean(obs[ok] > thr)) if ok.any() else None,
            "n_defined": int(ok.sum())}


__all__ = [
    "CONVENTIONS", "SIMILARITIES", "FieldDef", "Fields1D", "Fields2D",
    "active_set_overlap", "apply_convention", "bin_curve", "detect_fields_1d",
    "detect_fields_2d", "field_size_stats", "fit_count_models", "kernel_summary",
    "lag_profile_1d", "mean_rate_correlation", "pair_similarity", "prepare_columns",
    "random_orthogonal", "ratemap_correlations", "sample_gamma_poisson",
    "similarity_matrix", "smooth_1d", "smooth_2d", "within_unit_layout",
    "within_unit_size_ratio",
]
