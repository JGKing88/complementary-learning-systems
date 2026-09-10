"""The shared visitation probe, and the diagnostic input channel (§27.5).

One probe feeds two consumers -- the auxiliary head's target (§24.2 lever B)
and the diagnostic input channel -- deliberately, so the diagnostic tests
exactly the vector the head was trained on. These pin that, and pin the
read-before-mark ordering that makes the vector a memory rather than a
prediction of the future.
"""
from __future__ import annotations

import numpy as np
import pytest

from hopfield_nav.config import AgentConfig
from hopfield_nav.policy import channels
from hopfield_nav.rollout.visited import N_DIR, VisitedProbe


class TestProbe:

    def test_nothing_visited_at_the_first_read(self):
        p = VisitedProbe(20, 3.0, 2)
        out = p.read(np.array([[10, 10], [5, 5]]))
        assert out.shape == (2, N_DIR)
        assert out.sum() == 0.0

    def test_it_marks_where_you_stand_after_reading(self):
        """Read-then-mark. If it marked first, standing still would report
        yourself as visited and the vector would describe the present rather
        than the past."""
        p = VisitedProbe(20, 1.0, 1)
        first = p.read(np.array([[10, 10]]))
        assert first.sum() == 0.0
        # step one cell east; the probe pointing back west now sees the old cell
        second = p.read(np.array([[11, 10]]))
        assert second.sum() > 0.0

    def test_trials_are_independent(self):
        p = VisitedProbe(20, 1.0, 2)
        for _ in range(4):
            p.read(np.array([[3, 3], [15, 15]]))
        # trial 0 has never been near trial 1's cells
        assert not p.seen[0, 15, 15]
        assert not p.seen[1, 3, 3]

    def test_offsets_are_eight_compass_directions_at_the_radius(self):
        p = VisitedProbe(20, 3.0, 1)
        assert p.offsets.shape == (N_DIR, 2)
        # cardinal ones land exactly on the radius
        assert [3, 0] in p.offsets.tolist()
        assert [-3, 0] in p.offsets.tolist()
        assert [0, 3] in p.offsets.tolist()
        assert [0, -3] in p.offsets.tolist()

    def test_probes_clamp_at_the_arena_edge(self):
        """A corner agent must not index outside the grid."""
        p = VisitedProbe(6, 3.0, 1)
        out = p.read(np.array([[0, 0]]))
        assert out.shape == (1, N_DIR)
        out2 = p.read(np.array([[5, 5]]))
        assert np.isfinite(out2).all()


class TestChannel:

    def test_absent_by_default(self):
        cfg = AgentConfig(movement_mode="continuous",
                          hopfield_mode="continuous")
        names = [s.name for s in channels.channel_specs(cfg, embed_dim=8)]
        assert "visited" not in names

    def test_present_and_eight_wide_when_enabled(self):
        cfg = AgentConfig(movement_mode="continuous",
                          hopfield_mode="continuous", input_visited=True)
        specs = channels.channel_specs(cfg, embed_dim=8)
        w = {s.name: s.width for s in specs}
        assert w["visited"] == N_DIR

    def test_it_widens_the_policy_input_by_exactly_eight(self):
        base = AgentConfig(movement_mode="continuous",
                           hopfield_mode="continuous")
        on = AgentConfig(movement_mode="continuous",
                         hopfield_mode="continuous", input_visited=True)
        d0 = channels.input_dim(base, embed_dim=8, sensory_dim=60)
        d1 = channels.input_dim(on, embed_dim=8, sensory_dim=60)
        assert d1 - d0 == N_DIR

    def test_a_missing_channel_still_raises(self):
        """The diagnostic relies on build_policy_input being strict: if a site
        forgets to supply `visited`, it must fail loudly rather than train on
        zeros."""
        cfg = AgentConfig(movement_mode="continuous",
                          hopfield_mode="continuous", input_visited=True)
        specs = channels.channel_specs(cfg, embed_dim=8)
        import torch
        values = {s.name: torch.zeros(2, s.width)
                  for s in specs if s.name != "visited"}
        with pytest.raises(KeyError):
            channels.build_policy_input(specs, values, batch_size=2)


class TestAbsPositionChannel:
    """§29.4's diagnostic. Also an oracle, also not shippable."""

    def test_absent_by_default(self):
        cfg = AgentConfig(movement_mode="continuous",
                          hopfield_mode="continuous")
        names = [s.name for s in channels.channel_specs(cfg, embed_dim=8)]
        assert "abs_position" not in names

    def test_two_dims_when_enabled(self):
        cfg = AgentConfig(movement_mode="continuous",
                          hopfield_mode="continuous", input_abs_position=True)
        w = {s.name: s.width
             for s in channels.channel_specs(cfg, embed_dim=8)}
        assert w["abs_position"] == 2

    def test_normalisation_maps_the_arena_to_pm1(self):
        from hopfield_nav.rollout.visited import abs_position_channel

        class _Vec:
            def __init__(self, p):
                self._p = np.asarray(p, float)

            def positions_continuous(self):
                return self._p

        got = abs_position_channel(_Vec([[0.0, 0.0], [19.0, 19.0],
                                         [9.5, 9.5]]), 20)
        assert got[0] == pytest.approx([-1.0, -1.0])
        assert got[1] == pytest.approx([1.0, 1.0])
        assert got[2] == pytest.approx([0.0, 0.0], abs=1e-6)

    def test_falls_back_to_snapped_positions(self):
        """Discrete envs have no sub-cell position; the snap IS the position."""
        from hopfield_nav.rollout.visited import abs_position_channel

        class _Vec:
            def positions(self):
                return np.array([[0, 19]])

        got = abs_position_channel(_Vec(), 20)
        assert got[0] == pytest.approx([-1.0, 1.0])
