"""World-spec writes must survive many writers sharing one directory.

A 272-run sweep wrote every run's `world.json` into the same output directory.
`WorldSpec.write` staged through a fixed `world.json.tmp`, so the writers raced:
the first `os.replace` consumed the shared temp and every later one raised

    FileNotFoundError: '.../world.json.tmp' -> '.../world.json'

246 of the 272 runs died that way, after the environments were built but before
any training happened -- so the failure cost a full node-hour and produced
nothing. Staging through `tempfile.mkstemp` makes the temp name unique across
threads and processes alike while keeping the rename atomic.
"""
from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from hopfield_nav.world import domains as dom
from hopfield_nav.world.spec import (
    EnvSpec, GeneratedSplit, TraitDomains, WorldSpec, WORLD_SPEC_NAME)

SIZE = 8


def _spec(seed: int) -> WorldSpec:
    """A minimal but real WorldSpec. The concurrency being tested is in
    `write`, which does not care what the spec contains -- only that it
    serialises -- so this avoids building a scaffold and an encoder."""
    domains = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                           goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    train = [EnvSpec(wall_seed=seed, size=SIZE, offset=(0, 0), goal=(1, 1))]
    split = GeneratedSplit(
        domains=domains, train=train, base_val=[],
        goal_cells_train=frozenset({(1, 1)}), goal_cells_val=frozenset(),
        margin=1, period=SIZE, Npos=SIZE * SIZE)
    return WorldSpec(scaffold={"lambdas": [3, 4], "Npos": SIZE * SIZE},
                     generator="test", split=split)


def test_write_is_atomic_and_leaves_no_temp(tmp_path):
    p = _spec(0).write(tmp_path)
    assert os.path.basename(p) == WORLD_SPEC_NAME
    with open(p) as f:
        json.load(f)
    leftovers = glob.glob(os.path.join(tmp_path, "*.tmp"))
    assert leftovers == [], f"staging files left behind: {leftovers}"


def test_many_concurrent_writers_to_one_directory_all_succeed(tmp_path):
    """The regression. With a shared temp name this raises FileNotFoundError
    in most of the workers."""
    n = 48
    errors: list[BaseException] = []

    def go(i):
        try:
            _spec(i).write(tmp_path)
        except BaseException as e:      # noqa: BLE001 - recorded, then asserted
            errors.append(e)

    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(go, range(n)))

    assert not errors, f"{len(errors)}/{n} writers failed; first: {errors[0]!r}"
    # Exactly one world.json, valid, and nothing staged left over.
    with open(os.path.join(tmp_path, WORLD_SPEC_NAME)) as f:
        json.load(f)
    assert glob.glob(os.path.join(tmp_path, "*.tmp")) == []


def test_a_reader_never_sees_a_partial_file(tmp_path):
    """The property the staging exists for in the first place: `world.json` is
    either absent or complete, never half-written."""
    for i in range(20):
        _spec(i).write(tmp_path)
        with open(os.path.join(tmp_path, WORLD_SPEC_NAME)) as f:
            d = json.load(f)
        assert d["spec_version"] and "split" in d and "scaffold" in d
