"""Tests for the unique-radius metric in ``encoder_training.viz``.

The metric answers: *how far can you walk away from a reference position
before the cosine-similarity map stops telling you where you came from?* A
sample at distance t counts only if its similarity strictly exceeds every
similarity farther out along the same ray, so the accepted run is both
strictly decreasing and dominant over its own tail — a distant near-duplicate
of the reference (perceptual aliasing) truncates the radius no matter how
clean the local neighbourhood is.

``compute_unique_radius`` computes that with suffix/prefix maxima rather than
an ``np.max`` over a slice per step. ``brute_unique_radius`` below is a direct
transcription of the definition; the randomized test pins the fast version to
it. The rest of the file pins the properties a sweep depends on: index order,
distance units, the grid-boundary cap, and the deliberate asymmetry of
``unique_radius_per_theta`` relative to the archived 1D original.
"""
from __future__ import annotations

import numpy as np
import pytest

from encoder_training.viz import compute_unique_radius, unique_radius_per_theta


def brute_unique_radius(xs, ref_ix, sim):
    """Definition, transcribed literally: walk out while sim[j] beats its tail."""
    sim = np.asarray(sim, dtype=float)
    n = len(sim)

    right_r = 0.0
    for j in range(ref_ix + 1, n):
        beyond = sim[j + 1:]
        max_beyond = beyond.max() if len(beyond) else -np.inf
        if sim[j] > max_beyond:
            right_r = xs[j] - xs[ref_ix]
        else:
            break

    left_r = 0.0
    for j in range(ref_ix - 1, -1, -1):
        beyond = sim[:j]
        max_beyond = beyond.max() if len(beyond) else -np.inf
        if sim[j] > max_beyond:
            left_r = xs[ref_ix] - xs[j]
        else:
            break

    if ref_ix == 0:
        return float(right_r)
    if ref_ix == n - 1:
        return float(left_r)
    return float(min(left_r, right_r))


# ---------------------------------------------------------------------------
# compute_unique_radius
# ---------------------------------------------------------------------------

def test_matches_brute_force_over_random_profiles():
    """The suffix/prefix-maxima shortcut must not change the answer.

    Three profile families, because they stress different branches: integer
    draws produce the exact ties that the strict ``>`` has to reject, sorted
    draws produce runs that never terminate early, and uniform draws produce
    the ordinary early break.
    """
    rng = np.random.default_rng(0)
    for trial in range(3000):
        n = int(rng.integers(1, 14))
        ref_ix = int(rng.integers(0, n))
        xs = np.sort(rng.uniform(0, 20, n))
        if trial % 3 == 0:
            sim = rng.integers(0, 3, n).astype(float)   # heavy ties
        elif trial % 3 == 1:
            sim = np.sort(rng.uniform(-1, 1, n))[::-1]  # strictly decreasing
        else:
            sim = rng.uniform(-1, 1, n)
        assert compute_unique_radius(xs, ref_ix, sim) == pytest.approx(
            brute_unique_radius(xs, ref_ix, sim)
        ), f"n={n} ref_ix={ref_ix} sim={sim}"


def test_monotone_decrease_runs_to_the_end_of_the_samples():
    xs = np.arange(6.0)
    sim = np.array([1.0, .9, .8, .7, .6, .5])
    assert compute_unique_radius(xs, 0, sim) == pytest.approx(5.0)


def test_a_distant_alias_collapses_the_radius():
    """The tail-dominance clause is what makes this a *unique* radius.

    Local decrease is perfect out to t=4, but the sample at t=5 is nearly as
    similar as the reference, so no distance is uniquely coded.
    """
    xs = np.arange(6.0)
    sim = np.array([1.0, .9, .8, .7, .6, .95])
    assert compute_unique_radius(xs, 0, sim) == pytest.approx(0.0)


def test_a_plateau_terminates_the_walk():
    """Strict ``>``: tied neighbours are not distinguishable, so the walk stops.

    Quantised encoders make this the common case rather than a corner case.
    """
    xs = np.arange(6.0)
    sim = np.array([1.0, .9, .9, .5, .4, .3])
    assert compute_unique_radius(xs, 0, sim) == pytest.approx(0.0)


def test_edge_references_use_only_the_flank_they_have():
    """A reference at either end must not be zeroed by ``min(left, right)``.

    ``unique_radius_per_theta`` always passes ``ref_ix=0``, so without this
    guard every ray would report 0. The archived 1D original returned
    ``min(left_r, right_r)`` unconditionally and had exactly that failure.
    """
    xs = np.arange(6.0)
    assert compute_unique_radius(
        xs, 0, np.array([1.0, .9, .8, .7, .6, .5])) == pytest.approx(5.0)
    assert compute_unique_radius(
        xs, 5, np.array([.1, .2, .3, .4, .5, 1.0])) == pytest.approx(5.0)


def test_interior_reference_takes_the_smaller_flank():
    xs = np.arange(6.0)
    sim = np.array([.1, .2, .5, 1.0, .9, .8])   # left reaches 3.0, right 2.0
    assert compute_unique_radius(xs, 3, sim) == pytest.approx(2.0)


def test_single_sample_is_zero_not_an_index_error():
    assert compute_unique_radius(
        np.array([0.0]), 0, np.array([1.0])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# unique_radius_per_theta
# ---------------------------------------------------------------------------

def _cone(size=201, centre=None, scale=300.0):
    """Similarity that falls off linearly with Euclidean distance from centre."""
    centre = (size // 2, size // 2) if centre is None else centre
    yy, xx = np.mgrid[0:size, 0:size]
    return 1.0 - np.sqrt((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / scale


def test_cos_map_is_indexed_y_then_x():
    """Rays must read ``cos_map[y, x]``; a transpose here mirrors every angle.

    Blocking alias placed due +x of the reference: only the +x ray may shorten.
    """
    cos_map = _cone()
    cos_map[98:103, 138:143] = 0.95        # rows are y, columns are x
    thetas, widths = unique_radius_per_theta(cos_map, 100.0, 100.0, n_rays=72)

    r_px = widths[int(np.argmin(np.abs(thetas - 0.0)))]
    r_py = widths[int(np.argmin(np.abs(thetas - np.pi / 2)))]
    r_mx = widths[int(np.argmin(np.abs(thetas - np.pi)))]
    assert r_px < 40.0
    assert r_py > 90.0
    assert r_mx > 90.0


def test_radius_is_euclidean_cells_and_capped_by_the_grid_edge():
    """Units are cells of true Euclidean distance, not steps in x or y.

    On a clean cone centred in the grid nothing ever breaks monotonicity, so
    every ray runs until it leaves the array: axis rays stop at 100 cells and
    diagonals at ~141. Every value here is censored by the boundary — a sweep
    that records this number is recording a lower bound.
    """
    thetas, widths = unique_radius_per_theta(_cone(), 100.0, 100.0, n_rays=72)
    assert widths.min() == pytest.approx(100.0)
    assert widths.max() == pytest.approx(141.0, abs=1.0)

    diag = widths[int(np.argmin(np.abs(thetas - np.pi / 4)))]
    assert diag == pytest.approx(100.0 * np.sqrt(2.0), abs=1.5)


def test_rays_are_directed_and_not_forced_symmetric():
    """Each θ uses only its own outward half-line.

    The archived 2D original sampled ±t about the reference and took
    ``min(left, right)``, which forces ``r(θ) == r(θ+π)``. This version passes
    ``ref_ix=0`` so an obstruction on one side leaves the opposite ray intact.
    """
    cos_map = _cone()
    cos_map[98:103, 138:143] = 0.95
    thetas, widths = unique_radius_per_theta(cos_map, 100.0, 100.0, n_rays=72)

    i0 = int(np.argmin(np.abs(thetas - 0.0)))
    i180 = int(np.argmin(np.abs(thetas - np.pi)))
    assert widths[i0] != pytest.approx(widths[i180])


def test_reference_in_a_corner_still_returns_finite_radii():
    """Rays that leave the grid immediately yield 0, not NaN or an exception."""
    thetas, widths = unique_radius_per_theta(_cone(), 0.0, 0.0, n_rays=36)
    assert np.all(np.isfinite(widths))
    assert np.all(widths >= 0.0)
