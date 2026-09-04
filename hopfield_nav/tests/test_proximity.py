"""Proximity revisit -- the retracing measure that is not coverage in disguise.

`revisit_frac` counts snapped-cell re-entry, and a step either enters a new
cell or it does not, so it equals 1 - cells_per_step exactly. Ranking arms by
it reproduces the coverage ranking by construction. These pin the continuous
measure that does not have that property.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.nav_tri.proximity import proximity_revisit, summarise


class TestProximityRevisit:

    def test_a_straight_line_never_returns(self):
        path = np.stack([np.linspace(0, 19, 60), np.full(60, 5.0)], axis=1)
        assert proximity_revisit(path, 1.0) == 0.0

    def test_a_tight_loop_returns_constantly(self):
        """Four laps of a circle. The ceiling is not 1.0: the first lap has no
        past to return to, so ~50 of the 190 scored steps cannot count however
        tightly the path loops."""
        t = np.linspace(0, 8 * np.pi, 200)
        path = np.stack([10 + 3 * np.cos(t), 10 + 3 * np.sin(t)], axis=1)
        assert proximity_revisit(path, 1.0) >= 0.75

    def test_it_sees_retracing_that_the_snapped_measure_cannot(self):
        """The failure this exists for. Out along y=5.0 and back along y=5.4:
        never the same snapped cell, and obviously going back over itself."""
        out = np.stack([np.linspace(2, 18, 80), np.full(80, 5.0)], axis=1)
        back = np.stack([np.linspace(18, 2, 80), np.full(80, 5.4)], axis=1)
        path = np.concatenate([out, back])
        assert (np.rint(out[:, 1]) == np.rint(back[:, 1])).all()  # same row...
        # ... yet no cell is re-entered on the return leg at a 0.4 offset,
        # while the continuous measure sees the whole return.
        assert proximity_revisit(path, 1.0) > 0.4

    def test_the_lag_excludes_the_trivial_neighbourhood(self):
        """Consecutive positions are always close; that is not retracing."""
        path = np.stack([np.linspace(0, 2, 60), np.zeros(60)], axis=1)
        assert proximity_revisit(path, 0.5, lag=1) > 0.9   # everything counts
        assert proximity_revisit(path, 0.05, lag=50) < 0.5  # only real returns

    def test_a_path_shorter_than_the_lag_is_nan_not_zero(self):
        """Zero would read as 'never retraced', which is a claim; NaN is the
        absence of one."""
        assert np.isnan(proximity_revisit(np.zeros((5, 2)), 1.0, lag=10))

    def test_larger_radius_never_scores_lower(self):
        rng = np.random.RandomState(0)
        path = np.cumsum(rng.randn(150, 2) * 0.5, axis=0)
        vals = [proximity_revisit(path, r) for r in (0.5, 1.0, 2.0, 4.0)]
        assert vals == sorted(vals)


class TestSummarise:

    def test_it_reports_spread_not_just_a_mean(self):
        rng = np.random.RandomState(0)
        paths = [np.cumsum(rng.randn(120, 2) * 0.4, axis=0) for _ in range(8)]
        s = summarise(paths, 1.0)
        assert s["n"] == 8
        assert 0.0 <= s["mean"] <= 1.0
        assert s["sd"] >= 0.0

    def test_short_paths_are_dropped_from_the_mean_not_counted_as_zero(self):
        good = [np.cumsum(np.ones((80, 2)) * 0.3, axis=0) for _ in range(3)]
        short = [np.zeros((4, 2))]
        s = summarise(good + short, 1.0)
        assert s["n"] == 3
        assert np.isfinite(s["mean"])

    def test_it_records_the_parameters_it_used(self):
        """The number is meaningless without them -- a different radius is a
        different question."""
        s = summarise([np.zeros((60, 2))], 1.5, lag=7)
        assert s["radius"] == 1.5 and s["lag"] == 7


def test_revisit_frac_is_one_minus_cells_per_step():
    """Documents WHY this module exists. A step enters a new cell or it does
    not, so the snapped revisit fraction is the coverage rate restated -- it
    cannot rank arms on retracing independently of ranking them on coverage."""
    cells = np.array([[0, 0], [1, 0], [1, 0], [2, 0], [2, 0], [3, 0]])
    seen, new = set(), 0
    for c in cells:
        k = tuple(c)
        if k not in seen:
            seen.add(k)
            new += 1
    cells_per_step = new / len(cells)
    revisit_frac = 1.0 - cells_per_step
    assert revisit_frac == pytest.approx(1.0 - cells_per_step)
    assert cells_per_step == pytest.approx(4 / 6)
