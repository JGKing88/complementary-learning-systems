"""The collapsed-tail report — the headline measurement for the explore half.

§5.3.2 established that a MEAN hides the failure this project is looking for:
on the phase-1 combined model the median swept barely moved from d=0 to d=10
(0.688 → 0.668) while 14.6% of episodes collapsed outright, and the p95 rose.
So the reduction that gets reported is the distribution and the tail.

§5.3.4 added that the tail itself is two populations: collapse WITHOUT chasing
is the motor wall pin, collapse WITH chasing is the corner trap, and they want
opposite fixes. `chase_q` inside the tail is what separates them.

These tests use synthetic trajectories with known answers, because the point of
the module is that it must not smooth a mixture into a mean.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from analysis.nav_tri.tail_report import main


def _serpentine(size, T, rows=9):
    """A lawnmower — the behaviour that actually sweeps the arena.

    A single straight line only sweeps ~0.09 at r=1 in a 20×20 box, which is
    below any sensible collapse threshold; the first draft of this fixture used
    one and every trial read as collapsed.
    """
    per = max(2, T // rows)
    pts = []
    for r in range(rows):
        y = 1.0 + r * (size - 2) / max(rows - 1, 1)
        xs = np.linspace(1, size - 2, per)
        if r % 2:
            xs = xs[::-1]
        pts.extend([[x, y] for x in xs])
    pts = pts[:T]
    while len(pts) < T:
        pts.append(pts[-1])
    return np.asarray(pts, dtype=float)


def _traj(n, *, sweeping, chase=0.0, size=20, T=200):
    """A dump in explore_traj's shape. `sweeping` rows cover the arena; the
    rest sit in one corner, which is what a collapsed episode looks like."""
    trials = []
    for i in range(n):
        if i < sweeping:
            path = _serpentine(size, T)
        else:
            path = np.tile(np.array([0.5, 0.5]), (T, 1))
        trials.append({"by_ckpt": {"m": {
            "path": path.tolist(),
            "chase_q": 0.0 if i < sweeping else chase,
            "edge_frac": 0.1 if i < sweeping else 0.95,
        }}})
    return {"size": size, "n_distractors": 10, "labels": ["m"],
            "trials": trials}


def _write(tmp_path, d):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(d))
    return str(p)


class TestItReportsTheTail:

    def test_a_clean_run_has_no_tail(self, tmp_path, capsys):
        main(_write(tmp_path, _traj(20, sweeping=20)))
        out = capsys.readouterr().out
        assert "0.000" in out

    def test_a_mixture_is_reported_as_a_fraction(self, tmp_path, capsys):
        """4 of 20 collapsed = 0.200."""
        main(_write(tmp_path, _traj(20, sweeping=16, chase=0.5)))
        out = capsys.readouterr().out
        assert "0.200" in out

    def test_the_median_can_look_healthy_while_the_tail_is_not(
            self, tmp_path, capsys):
        """The §5.3.2 shape: most episodes fine, a sixth destroyed. The p50
        must stay high while frac<t is clearly non-zero -- if the module
        reported only a mean this case would be indistinguishable from a
        uniform shift."""
        main(_write(tmp_path, _traj(24, sweeping=20, chase=0.4)))
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.strip().startswith("m ")][0]
        parts = line.split()
        mean, p5, p50 = float(parts[2]), float(parts[3]), float(parts[4])
        frac = float(parts[7])
        assert frac == pytest.approx(4 / 24, abs=1e-3)
        assert p50 > mean          # the mean is dragged down by the tail
        assert p5 < 0.35 < p50     # and the tail is separated from the body


class TestItSplitsTheTailByChase:

    def test_a_chasing_tail_reports_elevated_chase(self, tmp_path, capsys):
        main(_write(tmp_path, _traj(20, sweeping=16, chase=0.6)))
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.strip().startswith("m ")][0]
        chase_tail = float(line.split()[8])
        chase_rest = float(line.split()[9])
        assert chase_tail == pytest.approx(0.6, abs=1e-6)
        assert chase_rest == pytest.approx(0.0, abs=1e-6)

    def test_a_wall_pin_tail_reports_chase_zero(self, tmp_path, capsys):
        """The dissociation §5.3.4 found: a collapsed episode with chase ~0 is
        a motor wall pin, not the corner trap, and --persistence_realized is
        the fix for one and does nothing for the other."""
        main(_write(tmp_path, _traj(20, sweeping=16, chase=0.0)))
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.strip().startswith("m ")][0]
        assert float(line.split()[8]) == pytest.approx(0.0, abs=1e-6)

    def test_no_tail_prints_a_dash_rather_than_a_number(self, tmp_path,
                                                       capsys):
        """An empty tail has no chase value; inventing a 0.0 there would read
        as 'the tail does not chase' rather than 'there is no tail'."""
        main(_write(tmp_path, _traj(12, sweeping=12)))
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.strip().startswith("m ")][0]
        assert line.split()[8] == "--"


class TestThreshold:

    def test_the_threshold_is_settable(self, tmp_path, capsys):
        """Every threshold in this project is swept or stated, because one
        applied once and invisibly is how a 'collapse count' previously
        flattered an arm (EXPLOIT_DIAGNOSTIC §4)."""
        d = _traj(20, sweeping=16)
        main(_write(tmp_path, d), 0.0)
        assert "0.000" in capsys.readouterr().out
        main(_write(tmp_path, d), 1.0)
        assert "1.000" in capsys.readouterr().out
