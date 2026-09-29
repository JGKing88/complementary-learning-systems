"""The task-faithful training path reproduces its golden fingerprint bit-for-bit.

See ``golden_task_run.py`` for what is recorded and when to regenerate.
"""
from __future__ import annotations

import numpy as np
import pytest

from hopfield_nav.tests.golden_task_run import GOLDEN, compare, fingerprint


@pytest.mark.slow
def test_task_run_matches_golden():
    assert GOLDEN.exists(), "run `python -m hopfield_nav.tests.golden_task_run`"
    golden = dict(np.load(GOLDEN, allow_pickle=False))
    diffs = compare(fingerprint(), golden)
    assert not diffs, "\n".join(diffs[:20])
