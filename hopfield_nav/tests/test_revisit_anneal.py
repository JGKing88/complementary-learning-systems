"""Annealing revisit_penalty from 0 to its configured value.

§34.3 measured why a CONSTANT penalty is self-defeating. Positive reward needs
coverage rate `c > rp/(0.3 + rp)`, and the agent starts pinned at c ~ 0.10, so
every increment raises the bar it must clear BEFORE reward turns positive while
making the pin itself more punishing. 0.15 escaped slowly, 0.25 stalled at u200,
0.40 never moved at all.

Annealing separates the two jobs: escape the pin under no penalty, then apply
the pressure. The pressure is worth applying because the credit for avoiding a
revisit is otherwise diffuse -- it pays as slightly more novelty over later
steps -- while persistence pays immediately and certainly.
"""
from __future__ import annotations

import pytest

from hopfield_nav.config import HopfieldConfig


def effective(target, anneal_updates, update):
    """The ramp as train_navigate computes it."""
    if anneal_updates <= 0:
        return target
    return target * min(1.0, float(update) / float(anneal_updates))


class TestConfig:

    def test_off_by_default(self):
        assert HopfieldConfig().revisit_anneal_updates == 0

    def test_wired_through_every_layer(self):
        from hopfield_nav import train_navigate as tn
        assert tn.CFG_FIELDS["revisit_anneal_updates"] == (
            "hopfield.revisit_anneal_updates",)
        src = open("hopfield_nav/train_navigate.py").read()
        assert "--revisit_anneal_updates" in src
        assert "REVISIT_ANNEAL_UPDATES" in open(
            "hopfield_nav/navigate_job.sh").read()


class TestRamp:

    def test_zero_is_constant(self):
        assert effective(0.4, 0, 1) == 0.4
        assert effective(0.4, 0, 999) == 0.4

    def test_it_starts_at_nothing_and_reaches_the_target(self):
        """The whole point: no penalty while the agent is still pinned."""
        assert effective(0.4, 200, 0) == 0.0
        assert effective(0.4, 200, 100) == pytest.approx(0.2)
        assert effective(0.4, 200, 200) == pytest.approx(0.4)

    def test_it_never_overshoots(self):
        assert effective(0.4, 200, 700) == pytest.approx(0.4)

    def test_the_ramp_clears_the_break_even_before_the_penalty_bites(self):
        """§34.3's arithmetic: positive reward needs c > rp/(0.3+rp), and the
        agent starts pinned at c ~ 0.10, so the tolerable penalty there is
        exactly 0.0333.

        That number sets the ramp length. With a 300-update ramp to 0.4 the
        penalty reaches 0.0333 at u25 -- the agent is AT the margin that early,
        which is too soon given the control needs ~u75 to break the pin. A
        400-update ramp buys until u33, and the run uses 400 for that reason."""
        def break_even(rp):
            return rp / (0.3 + rp)
        assert break_even(effective(0.4, 400, 10)) < 0.05   # comfortably out
        assert break_even(effective(0.4, 300, 25)) == pytest.approx(0.10)
        assert break_even(effective(0.4, 400, 400)) > 0.50  # full pressure


class TestTargetCapture:

    def test_the_target_is_read_before_the_loop(self):
        """cfg.hopfield.revisit_penalty is OVERWRITTEN every update, so a ramp
        that re-read it inside the loop would scale the already-scaled value
        and decay geometrically to zero."""
        src = open("hopfield_nav/train_navigate.py").read()
        cap = src.index("_rp_target = float(")
        loop = src.index("for update in range(start_update + 1")
        assert cap < loop
        assert "cfg.hopfield.revisit_penalty = _rp_target * _rp_frac" in src

    def test_a_naive_reread_would_decay_to_zero(self):
        """Documents the bug the capture avoids, so nobody 'simplifies' it."""
        rp = 0.4
        for u in (1, 2, 3, 4, 5):
            rp = rp * min(1.0, u / 300.0)      # the wrong version
        assert rp < 1e-9
