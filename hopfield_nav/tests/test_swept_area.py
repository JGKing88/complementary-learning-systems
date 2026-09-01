"""The swept-area headline metric (evaluation/swept.py, P2 doc §19).

The point of this metric is that it measures what `at_goal` actually tests --
an L2 ball of `goal_radius` on the CONTINUOUS position -- rather than unique
snapped cells. These pin the properties that distinction buys, so a future
"simplification" back to cell counting fails here rather than silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from hopfield_nav.evaluation.swept import SweptArea


def _sweep(path, *, size=20, radius=1.0, grid=8):
    sw = SweptArea(size, radius, 1, grid=grid)
    for p in path:
        sw.add(np.asarray(p, dtype=float).reshape(1, 2))
    return float(sw.fraction()[0])


class TestGeometry:

    def test_a_single_point_sweeps_its_disc(self):
        """Area of one disc, well inside the arena, is pi r^2 / size^2."""
        got = _sweep([(10.0, 10.0)], radius=2.0)
        want = np.pi * 4.0 / 400.0
        assert got == pytest.approx(want, rel=0.05), (got, want)

    def test_a_straight_run_sweeps_a_capsule(self):
        """A segment of length L sweeps 2rL + pi r^2."""
        L, r = 10.0, 1.0
        path = [(5.0, 10.0)] + [(5.0 + t, 10.0) for t in
                                np.linspace(0, L, 11)[1:]]
        got = _sweep(path, radius=r)
        want = (2 * r * L + np.pi * r * r) / 400.0
        assert got == pytest.approx(want, rel=0.06), (got, want)

    def test_revisiting_adds_nothing(self):
        """It is a UNION. Walking the same line twice sweeps the same area."""
        fwd = [(5.0 + t, 10.0) for t in np.linspace(0, 8, 9)]
        once = _sweep(fwd)
        twice = _sweep(fwd + fwd[::-1])
        assert twice == pytest.approx(once, rel=1e-6)


class TestTheReasonItExists:

    def test_a_long_stride_is_not_penalised_for_cells_it_swept(self):
        """The whole point. Two agents covering the SAME LINE, one in strides
        of 1.0 and one in strides of 2.0, sweep the same corridor -- but the
        fast one lands on half as many cell centres. Cell counting charges it
        for that; swept area does not."""
        slow = [(2.0 + t, 10.0) for t in np.arange(0, 16.1, 1.0)]
        fast = [(2.0 + t, 10.0) for t in np.arange(0, 16.1, 2.0)]
        s_slow, s_fast = _sweep(slow), _sweep(fast)
        assert s_fast == pytest.approx(s_slow, rel=0.02), (s_slow, s_fast)

        # and the cell count really does disagree, so the test is not vacuous
        cells = lambda p: len({(int(round(x)), int(round(y))) for x, y in p})
        assert cells(fast) < cells(slow)

    def test_subsampling_closes_gaps_a_long_stride_would_leave(self):
        """Without segment sub-sampling a stride of 3 with r=1 leaves holes
        between discs, which would re-introduce the stride penalty by the back
        door."""
        far = [(2.0, 10.0), (5.0, 10.0), (8.0, 10.0), (11.0, 10.0)]
        got = _sweep(far, radius=1.0)
        want = (2 * 1.0 * 9.0 + np.pi) / 400.0        # capsule over length 9
        assert got == pytest.approx(want, rel=0.08), (got, want)


class TestUnion:

    def test_union_is_the_or_across_trials(self):
        """Three trials parked far apart: the union is their sum, and each
        trial on its own is a third of it."""
        sw = SweptArea(20, 1.0, 3)
        pos = np.array([[3.0, 3.0], [10.0, 10.0], [16.0, 16.0]])
        for _ in range(3):
            sw.add(pos)
        res = sw.result()
        assert res.union == pytest.approx(res.per_trial.sum(), rel=1e-6)
        assert res.union == pytest.approx(3 * np.pi / 400.0, rel=0.06)

    def test_a_collapsed_policy_gains_nothing_from_more_trials(self):
        """The diagnostic this metric is for: if every trial walks the SAME
        route, the union equals a single trial and extra attempts buy nothing.
        """
        sw = SweptArea(20, 1.0, 4)
        for t in np.linspace(0, 10, 11):
            sw.add(np.tile([[3.0 + t, 10.0]], (4, 1)))
        res = sw.result()
        assert res.union == pytest.approx(float(res.per_trial[0]), rel=1e-6)
        assert np.allclose(res.per_trial, res.per_trial[0])

    def test_union_is_at_least_the_best_single_trial(self):
        sw = SweptArea(20, 1.0, 3)
        rng = np.random.RandomState(0)
        pos = rng.uniform(2, 17, size=(3, 2))
        for _ in range(8):
            pos = np.clip(pos + rng.uniform(-1, 1, size=(3, 2)), 0, 19)
            sw.add(pos)
        res = sw.result()
        assert res.union >= res.per_trial.max() - 1e-9
        assert res.union <= res.per_trial.sum() + 1e-9


class TestBatching:

    def test_trials_do_not_leak_into_each_other(self):
        sw = SweptArea(20, 1.0, 3)
        pos = np.array([[3.0, 3.0], [10.0, 10.0], [16.0, 16.0]])
        for _ in range(4):
            sw.add(pos)
        f = sw.fraction()
        assert f.shape == (3,)
        # three stationary agents each sweep one disc, so all equal and small
        assert np.allclose(f, f[0], rtol=0.05)
        assert f[0] == pytest.approx(np.pi / 400.0, rel=0.06)

    def test_radius_must_be_positive(self):
        with pytest.raises(ValueError):
            SweptArea(20, 0.0, 1)
