"""`explore_traj` must score a CARTESIAN checkpoint, not only a polar one.

Found by running it, not by reading it: scoring the phase-1 combined model
`w6_pers` died with

    IndexError: too many indices for array: array is 1-dimensional,
                but 2 were indexed
    ... in collect:  for v in rec["circ_sd"][:, b]

`behavior_probe` initialises `rec` with the keys "sigma", "mu_norm" and
"circ_sd" unconditionally and only FILLS the polar ones under a von Mises
head. So for a Cartesian model `rec["circ_sd"]` is an empty list, `"circ_sd"
in rec` is nevertheless True, and the 2-D index blows up.

**Scope of the bug: `explore_traj` could not score ANY non-polar checkpoint,
which is every phase-1 model** -- so no phase-1 model could be put on the
swept metric at all. `behavior_probe` itself gets this right (it guards on
`.size`); only this consumer did not.
"""
from __future__ import annotations

import numpy as np
import pytest


def _select(rec, b):
    """The guarded extraction, mirroring explore_traj.collect."""
    st = {}
    for key, prec in (("circ_sd", 5), ("mu_norm", 4)):
        arr = np.asarray(rec.get(key, []))
        if arr.ndim == 2 and arr.size:
            st[key] = [round(float(v), prec) for v in arr[:, b]]
    return st


class TestGuard:

    def test_a_cartesian_record_yields_no_polar_keys(self):
        """The failing case: keys present, arrays empty."""
        rec = {"circ_sd": [], "mu_norm": []}
        assert _select(rec, 0) == {}

    def test_a_polar_record_still_yields_them(self):
        rec = {"circ_sd": np.arange(6.0).reshape(3, 2),
               "mu_norm": np.ones((3, 2))}
        got = _select(rec, 1)
        assert got["circ_sd"] == [1.0, 3.0, 5.0]
        assert got["mu_norm"] == [1.0, 1.0, 1.0]

    def test_a_missing_key_is_fine(self):
        assert _select({}, 0) == {}

    def test_membership_is_NOT_a_sufficient_test(self):
        """Pins the exact mistake: the key IS present on a Cartesian record,
        so `in rec` cannot be the guard."""
        rec = {"circ_sd": []}
        assert "circ_sd" in rec
        with pytest.raises(IndexError):
            np.asarray(rec["circ_sd"])[:, 0]


class TestWiring:

    def test_explore_traj_guards_on_ndim_and_size(self):
        src = open("analysis/nav_tri/explore_traj.py").read()
        assert 'if arr.ndim == 2 and arr.size:' in src
        assert 'rec["circ_sd"][:, b]' not in src

    def test_behavior_probe_already_guarded_on_size(self):
        """The consumer that got it right, kept as the reference."""
        src = open("analysis/nav_tri/behavior_probe.py").read()
        assert ".size else None" in src
