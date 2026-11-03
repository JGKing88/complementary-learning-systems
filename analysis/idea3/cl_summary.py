"""Continual-protocol summary on one definition for every model (§7.5, §10.8).

    python -m analysis.idea3.cl_summary name=<agenthash json> [...]

``analysis.continual.retention`` scores revisits only in "live" arenas (those
learned in their own block), which leaves idea 3, which learns none, with no
row at all. This reads the agenthash history directly and scores every arena:

- ``first``: the first trial in each arena (a search; nothing is known).
- ``own_block``: all trials in the arena's own block (store allowed).
- ``revisit``: all trials in later blocks; ``revisit_steps`` over the
  successful ones.
- ``after_find``: trials after the agent first reached this arena's goal.
  With a memory this is near 1; without one it is the search rate.
- ``worst_delta``: min over arenas of (last block - own block) success.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def summarise(path):
    d = json.load(open(path))
    n = d["metadata"]["n_envs"]
    own = {i: [] for i in range(n)}
    last = {i: [] for i in range(n)}
    first, rev, rev_steps, after = [], [], [], []
    found = [False] * n
    seen = [False] * n
    for _it, block, inner in d["trace"]:
        for js, rec in inner.items():
            j = int(js)
            r = int(rec["reached"])
            if not seen[j]:
                first.append(r)
                seen[j] = True
            elif found[j]:
                after.append(r)
            if j == block:
                own[j].append(r)
            else:
                rev.append(r)
                if r:
                    rev_steps.append(rec["steps_to_goal"])
            if block == n - 1:
                last[j].append(r)
            found[j] |= bool(r)
    delta = min(np.mean(last[j]) - np.mean(own[j]) for j in range(n))
    return {"first": np.mean(first), "own_block": np.mean([np.mean(own[j]) for j in range(n)]),
            "revisit": np.mean(rev), "revisit_steps": np.mean(rev_steps) if rev_steps else np.nan,
            "after_find": np.mean(after) if after else np.nan, "worst_delta": delta,
            "n_revisits": len(rev)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("histories", nargs="+", help="name=path")
    a = p.parse_args()
    keys = ["first", "own_block", "revisit", "revisit_steps", "after_find",
            "worst_delta", "n_revisits"]
    print(f"{'run':<28s}" + "".join(f"{k:>14s}" for k in keys))
    for item in a.histories:
        name, path = item.split("=", 1)
        s = summarise(path)
        print(f"{name:<28s}" + "".join(f"{s[k]:14.3f}" for k in keys))


if __name__ == "__main__":
    main()
