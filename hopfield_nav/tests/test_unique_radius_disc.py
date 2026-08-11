"""Tests for the disc-based unique coding radius used by the sweep.

The ray-based metric in ``encoder_training.viz`` (pinned by
``test_unique_radius``) estimates a worst case from 72 directions, which is the
wrong tool for a minimum: at 5 degree spacing neighbouring rays are 4.4 cells
apart at r=50, so a narrow alias between two rays is invisible and the reported
minimum is optimistic. ``encoder_training.unique_radius`` computes the same
worst case directly over every cell instead.

The strongest tests here are the ones with an analytically known answer: a cone
``1 - d/300`` plus an alias plateau at level A separates at exactly
``r* = 300 * (1 - A)``, so the metric can be checked against arithmetic rather
than against itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from encoder_training.unique_radius import unique_radius, unique_radius_report

SIZE = 600
CENTRE = 300.0


def _cone(scale=300.0):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    d = np.sqrt((xx - CENTRE) ** 2 + (yy - CENTRE) ** 2)
    return 1.0 - d / scale


@pytest.mark.parametrize("alias_level", [0.99, 0.95, 0.90, 0.80])
def test_matches_the_analytic_radius(alias_level):
    """r* = 300*(1-A); reported value is the last passing integer shell."""
    m = _cone()
    m[40:100, 40:100] = alias_level            # 3600 cells, far beyond any trim
    expected = 300.0 * (1.0 - alias_level)
    r, saturated = unique_radius(m, CENTRE, CENTRE, trim=16, max_r=250)
    assert not saturated
    assert r == pytest.approx(expected - 1.0, abs=1.0)


@pytest.mark.parametrize("trim", [0, 4, 16, 64])
def test_trim_is_neutral_against_genuine_aliasing(trim):
    """Trim must not be able to explain away a region larger than itself.

    This is what makes a fixed count the right choice over a percentile: at
    Npos=1716 a 99.9th percentile would discard 2944 cells and erase exactly
    this kind of region.
    """
    m = _cone()
    m[40:100, 40:100] = 0.90
    r, _ = unique_radius(m, CENTRE, CENTRE, trim=trim, max_r=250)
    assert r == pytest.approx(29.0, abs=1.0)


def test_trim_absorbs_isolated_hot_cells():
    """A handful of anomalous cells must not decide a worst-case statistic."""
    m = _cone()
    rng = np.random.default_rng(0)
    m[rng.integers(0, SIZE, 8), rng.integers(0, SIZE, 8)] = 0.999

    assert unique_radius(m, CENTRE, CENTRE, trim=0, max_r=200)[0] == 0.0
    r16, sat16 = unique_radius(m, CENTRE, CENTRE, trim=16, max_r=200)
    assert r16 == 200.0 and sat16


def test_clean_map_saturates_rather_than_inventing_a_failure():
    r, saturated = unique_radius(_cone(), CENTRE, CENTRE, trim=16, max_r=150)
    assert r == 150.0
    assert saturated


def test_first_failure_not_last_success():
    """A near-duplicate close in must cap the radius, not be stepped over.

    The condition is not monotone in r — it holds again once the offending
    cell falls inside the disc — and it is vacuously true at the outermost
    shell, where nothing lies beyond. Taking the last success returns the
    grid's corner distance for every map, however badly aliased.
    """
    m = _cone()
    m[295:306, 340:351] = 0.999                # ~40 cells at r ~ 45
    r, _ = unique_radius(m, CENTRE, CENTRE, trim=16, max_r=250)
    assert r < 10.0


def test_is_invariant_to_the_axis_convention():
    """Distance is symmetric under transpose, so [x, y] vs [y, x] cannot matter.

    Unlike the ray version, which mirrors its angles if the caller transposes.
    """
    m = _cone()
    m[40:100, 200:260] = 0.93
    a = unique_radius(m, CENTRE, CENTRE, trim=16, max_r=250)[0]
    b = unique_radius(np.ascontiguousarray(m.T), CENTRE, CENTRE,
                      trim=16, max_r=250)[0]
    assert a == b


def test_report_diagnostics_track_the_alias_that_caused_the_cap():
    m = _cone()
    m[40:100, 40:100] = 0.90
    rep = unique_radius_report(m, CENTRE, CENTRE, max_r=250)

    assert rep["alias_ceiling"] == pytest.approx(0.90, abs=1e-6)
    # margin is the signed quantity whose zero crossing is the radius
    assert rep["margin_r25"] > 0.0
    assert rep["margin_r50"] < 0.0
    assert rep["r_headline"] == pytest.approx(29.0, abs=1.0)
    assert rep["headline_trim"] == 16


def test_report_covers_every_requested_trim_and_radius():
    rep = unique_radius_report(_cone(), CENTRE, CENTRE, max_r=100,
                               trims=(0, 16), margin_radii=(5, 10),
                               profile_levels=(0.5,))
    for key in ("r_trim0", "r_trim16", "saturated_trim0", "saturated_trim16",
                "margin_r5", "margin_r10", "r_at_cos0.5", "cos_floor",
                "alias_ceiling", "n_cells"):
        assert key in rep, key
    assert rep["n_cells"] == SIZE * SIZE


def test_headline_trim_is_added_when_absent_from_trims():
    rep = unique_radius_report(_cone(), CENTRE, CENTRE, max_r=50,
                               trims=(0,), headline_trim=16)
    assert rep["r_trim16"] == rep["r_headline"]


# ---------------------------------------------------------------------------
# The evaluator's pure-numpy pieces (no torch, no checkpoints)
# ---------------------------------------------------------------------------

def _reference_codes(lam, fwhm_ratio):
    """What training actually feeds the encoder, built the expensive way."""
    from gridcode.codebook import gen_gbook_2d
    from gridcode.smoothing import smooth_gbook

    Npos, Ng = int(np.prod(lam)), sum(l * l for l in lam)
    gbook = gen_gbook_2d(lam, Ng, Npos)
    if fwhm_ratio > 0:
        gbook = smooth_gbook(gbook, lam, fwhm_ratio)
    return gbook.reshape(Ng, -1).T


def test_lazy_grid_codes_equal_gen_gbook_2d_unsmoothed():
    """The evaluator regenerates codes per batch instead of materialising them.

    At lambdas (11,12,13) the codebook is 434 x 1716 x 1716 float64 = 10.2 GB,
    so the sweep cannot afford it — but the lazy version has to agree exactly
    or every similarity is computed on the wrong code.
    """
    from encoder_training.eval_unique_radius import grid_code_batch, npos_for

    lam = [3, 4, 5]
    Npos = npos_for(lam)
    xs, ys = np.meshgrid(np.arange(Npos), np.arange(Npos), indexing="ij")

    lazy = grid_code_batch(lam, xs.ravel(), ys.ravel(), fwhm_ratio=0.0)
    assert np.array_equal(lazy.astype(np.float64), _reference_codes(lam, 0.0))
    assert np.all(lazy.sum(axis=1) == len(lam))     # one active unit per module


@pytest.mark.parametrize("fwhm_ratio", [0.25, 0.4])
def test_lazy_grid_codes_match_smooth_gbook(fwhm_ratio):
    """Smoothing is what makes neighbouring positions overlap at all.

    Raw one-hot codes of adjacent positions are disjoint — x % 3, x % 4 and
    x % 5 all change when x moves by one — so an encoder fed them sees no
    neighbourhood structure and every similarity map collapses inside a single
    cell. Training and ``evaluate_nav`` both smooth; scoring must match, or the
    encoder is evaluated off-distribution and the radius means nothing.
    """
    from encoder_training.eval_unique_radius import grid_code_batch, npos_for

    lam = [3, 4, 5]
    Npos = npos_for(lam)
    xs, ys = np.meshgrid(np.arange(Npos), np.arange(Npos), indexing="ij")

    lazy = grid_code_batch(lam, xs.ravel(), ys.ravel(), fwhm_ratio=fwhm_ratio)
    assert np.allclose(lazy, _reference_codes(lam, fwhm_ratio), atol=1e-6)


def test_smoothing_makes_adjacent_positions_overlap():
    from encoder_training.eval_unique_radius import grid_code_batch

    lam = [11, 12, 13]
    xs, ys = np.array([500, 501]), np.array([500, 500])

    raw = grid_code_batch(lam, xs, ys, fwhm_ratio=0.0)
    assert float(raw[0] @ raw[1]) == 0.0             # disjoint: no signal

    smooth = grid_code_batch(lam, xs, ys, fwhm_ratio=0.25)
    cos = float(smooth[0] @ smooth[1] /
                (np.linalg.norm(smooth[0]) * np.linalg.norm(smooth[1])))
    assert cos > 0.5


def test_references_respect_the_border_and_are_reproducible():
    from encoder_training.eval_unique_radius import sample_references, npos_for

    lam = [11, 12, 13]
    Npos = npos_for(lam)
    refs = sample_references(lam, n_refs=20, border=100, seed=0)

    assert refs.shape == (20, 2)
    assert refs.min() >= 100
    assert refs.max() <= Npos - 1 - 100
    # every encoder in a sweep must be scored at the same positions
    assert np.array_equal(refs, sample_references(lam, 20, 100, 0))
    assert not np.array_equal(refs, sample_references(lam, 20, 100, 1))


def test_border_too_wide_for_the_arena_is_an_error_not_a_silent_clamp():
    from encoder_training.eval_unique_radius import sample_references

    with pytest.raises(ValueError, match="border"):
        sample_references([3, 4], n_refs=5, border=100, seed=0)


def test_summarize_reports_the_worst_reference_as_the_headline():
    from encoder_training.eval_unique_radius import summarize

    records = []
    for j, r in enumerate([12.0, 3.0, 40.0, 7.0]):
        records.append({
            "r_trim0": r, "r_trim4": r, "r_trim16": r, "r_trim64": r,
            "saturated_trim16": False, "alias_ceiling": 0.5 + 0.01 * j,
            "margin_r5": 0.1, "margin_r10": 0.05, "margin_r25": -0.01,
            "margin_r50": -0.2, "r_at_cos0.9": 5.0, "r_at_cos0.5": 20.0,
            "r_at_cos0.1": 60.0, "cos_floor": -0.1,
        })
    out = summarize(records)
    assert out["r_min"] == 3.0
    assert out["r_max"] == 40.0
    assert out["r_median"] == pytest.approx(9.5)
    assert out["n_refs"] == 4
    assert out["n_saturated"] == 0
    assert out["alias_ceiling_max"] == pytest.approx(0.53)
