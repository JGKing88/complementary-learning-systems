"""The wall-directedness cue: geometry and the AUC that scores it.

Following `test_state_probe`, the heavy part is not exercised here -- the field
evaluation is `readout_field.field_over_cells`, covered by its own module. What
is tested is the arithmetic that turns a field into a claim, because the claim
is a comparison against ||q||'s 0.881 and it is worth exactly as much as the
AUC is correct.

The geometry matters more than it looks. Outwardness is cos(q, outward normal),
so a normal that points the wrong way, or corner normals that are not unit,
would silently flip or shrink the very quantity the section reports.
"""
from __future__ import annotations

import numpy as np

from analysis.nav_tri.wall_cue import auc, perimeter_normals


class TestPerimeterNormals:

    def test_only_perimeter_cells_are_returned(self):
        size = 6
        pos, _ = perimeter_normals(size)
        assert len(pos) == size * size - (size - 2) ** 2
        interior = ((pos[:, 0] > 0) & (pos[:, 0] < size - 1)
                    & (pos[:, 1] > 0) & (pos[:, 1] < size - 1))
        assert not interior.any()

    def test_every_normal_is_unit_length(self):
        """Corners sum two axes; without normalising they would be sqrt(2) and
        would over-weight four cells in the mean."""
        _, nrm = perimeter_normals(7)
        assert np.allclose(np.linalg.norm(nrm, axis=-1), 1.0)

    def test_edge_normals_point_out_not_in(self):
        size = 5
        pos, nrm = perimeter_normals(size)
        d = dict(zip(map(tuple, pos), nrm))
        assert np.allclose(d[(0, 2)], [-1, 0])          # west edge -> -x
        assert np.allclose(d[(size - 1, 2)], [1, 0])    # east edge -> +x
        assert np.allclose(d[(2, 0)], [0, -1])          # south edge -> -y
        assert np.allclose(d[(2, size - 1)], [0, 1])    # north edge -> +y

    def test_corner_normal_is_the_diagonal(self):
        size = 5
        pos, nrm = perimeter_normals(size)
        d = dict(zip(map(tuple, pos), nrm))
        r = 1.0 / np.sqrt(2.0)
        assert np.allclose(d[(0, 0)], [-r, -r])
        assert np.allclose(d[(size - 1, size - 1)], [r, r])

    def test_a_field_pointing_at_the_centre_scores_negative(self):
        """The sign convention the whole section rests on: a stored goal pulls
        INWARD, which must come out as a negative cosine."""
        size = 9
        pos, nrm = perimeter_normals(size)
        centre = np.array([(size - 1) / 2.0, (size - 1) / 2.0])
        v = centre[None, :] - pos.astype(float)
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        cos = (v * nrm).sum(-1)
        assert cos.max() < 0.0, "inward field must score negative everywhere"

    def test_a_field_pointing_away_scores_positive(self):
        size = 9
        pos, nrm = perimeter_normals(size)
        centre = np.array([(size - 1) / 2.0, (size - 1) / 2.0])
        v = pos.astype(float) - centre[None, :]
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        assert ((v * nrm).sum(-1).min() > 0.0)


class TestAUC:

    def test_perfect_separation_is_one(self):
        assert auc([3.0, 4.0, 5.0], [0.0, 1.0, 2.0]) == 1.0

    def test_perfect_inversion_is_zero(self):
        assert auc([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]) == 0.0

    def test_identical_distributions_are_one_half(self):
        x = [1.0, 2.0, 3.0, 4.0]
        assert np.isclose(auc(x, list(x)), 0.5)

    def test_ties_are_handled_at_one_half(self):
        """All-tied inputs must give 0.5, not 0 or 1 -- a rank implementation
        that breaks ties by order would report a spurious perfect score."""
        assert np.isclose(auc([1.0] * 5, [1.0] * 5), 0.5)

    def test_matches_the_brute_force_definition(self):
        rng = np.random.RandomState(0)
        p, n = rng.rand(40), rng.rand(55) + 0.2
        brute = np.mean([[(a > b) + 0.5 * (a == b) for b in n] for a in p])
        assert np.isclose(auc(p, n), brute)

    def test_empty_input_is_nan_not_a_number(self):
        """A level with no draws must not silently report 0.5, which would
        read as 'measured, no signal'."""
        assert np.isnan(auc([], [1.0, 2.0]))
        assert np.isnan(auc([1.0, 2.0], []))

    def test_is_invariant_to_a_monotone_rescale(self):
        """Cosines could be reported on any monotone scale; the AUC must not
        move if they are."""
        rng = np.random.RandomState(1)
        p, n = rng.rand(30), rng.rand(30) - 0.3
        assert np.isclose(auc(p, n), auc(np.exp(3 * p), np.exp(3 * n)))
