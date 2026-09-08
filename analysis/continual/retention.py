"""Forgetting matrix and retention summary from a continual-learning history.

`agenthash.py` / `baseline.py` write a per-episode trace; `plotting.py` turns it
into figures. This turns it into the numbers, because the claim a continual
result actually makes is a comparison and not a picture:

    primary   an episode in the env's OWN block, where the store is allowed.
              It starts with nothing in memory for that env, so reaching the
              goal requires EXPLORING to find it first.
    revisit   an episode in an env introduced earlier, with the store LOCKED
              (`run_sequential_protocol` passes allow_store=(j == i)). The goal
              is already stored, so this is a pure exploit episode and a pure
              retention measurement.

Splitting on that is the whole point: a pooled success rate averages the two
and reports neither. On d0_base u725 the pooled number is 0.833 while the
revisit number over live envs is 1.0000 -- the pooled version would have
understated retention by seventeen points.

DEAD ENVS. An env whose goal is never found during its own block stores
nothing, and under autostore nothing else may ever write it, so it is
permanently unsolvable and its column is zero for the rest of the protocol.
Those columns say something about the explore half and the protocol's
seriality; they say nothing about retention, and averaging them into a
retention number is how "zero forgetting" would get reported as 0.83. Hence
`live_only`, which is the default for the retention figures and never the
default for the headline success rate.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def load(path):
    with open(path) as fh:
        h = json.load(fh)
    return h, h["metadata"]


def cells(history, n):
    """{(block, env): [reached, ...]} and the same for steps on successes."""
    reach, steps = {}, {}
    for _it, blk, per in history["trace"]:
        for k, rec in per.items():
            j = int(k)
            reach.setdefault((blk, j), []).append(int(rec["reached"]))
            if rec["reached"] and rec.get("steps_to_goal") is not None:
                steps.setdefault((blk, j), []).append(float(rec["steps_to_goal"]))
    return reach, steps


def matrix(reach, n):
    """(n, n) success rate; NaN where the pair never ran (env after block)."""
    m = np.full((n, n), np.nan)
    for (b, j), v in reach.items():
        m[b, j] = float(np.mean(v))
    return m


def live_envs(reach, n, thresh=0.5):
    """Envs whose OWN block cleared `thresh` -- i.e. that were ever learned."""
    return [j for j in range(n)
            if reach.get((j, j)) and np.mean(reach[(j, j)]) >= thresh]


def summarise(history, md, *, thresh=0.5):
    n = int(md["n_envs"])
    reach, steps = cells(history, n)
    live = live_envs(reach, n, thresh)
    dead = [j for j in range(n) if j not in live]

    def pool(d, pred, envs):
        out = []
        for (b, j), v in d.items():
            if j in envs and pred(b, j):
                out.extend(v)
        return out

    prim = lambda b, j: b == j          # noqa: E731
    revi = lambda b, j: j < b           # noqa: E731
    allv = list(range(n))
    r = {
        "n_envs": n, "iters_per_block": int(md["iters_per_block"]),
        "live": live, "dead": dead,
        "episodes": int(sum(len(v) for v in reach.values())),
        "primary_success": float(np.mean(pool(reach, prim, live))),
        "revisit_success": float(np.mean(pool(reach, revi, live))),
        "revisit_success_all": float(np.mean(pool(reach, revi, allv))),
        "primary_steps": float(np.mean(pool(steps, prim, live))),
        "revisit_steps": float(np.mean(pool(steps, revi, live))),
        "revisit_episodes": len(pool(reach, revi, live)),
        "primary_episodes": len(pool(reach, prim, live)),
    }
    m = matrix(reach, n)
    r["final_block_success_live"] = float(
        np.nanmean([m[n - 1, j] for j in live])) if live else float("nan")
    # Retention delta per env: own block -> final block. Negative is forgetting.
    r["retention_delta"] = {
        int(j): float(m[n - 1, j] - m[j, j]) for j in live}
    r["worst_retention_delta"] = (min(r["retention_delta"].values())
                                  if live else float("nan"))
    return r, m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--live_thresh", type=float, default=0.5)
    a = p.parse_args()
    h, md = load(a.history)
    r, m = summarise(h, md, thresh=a.live_thresh)
    n = r["n_envs"]

    print("run=%s  n_envs=%d  iters_per_block=%d  episodes=%d"
          % (md.get("run_name"), n, r["iters_per_block"], r["episodes"]))
    print("live envs %d   dead (never learned in own block) %s"
          % (len(r["live"]), r["dead"] or "none"))
    print()
    print("  SUCCESS -- rows block, cols env. Diagonal = own block (store")
    print("  allowed); left of it = locked-store revisit.")
    print("  %-7s" % "block" + "".join("%6s" % ("e%d" % j) for j in range(n)))
    for i in range(n):
        row = "  %-7s" % ("i=%d" % i)
        for j in range(n):
            row += "     ." if np.isnan(m[i, j]) else "%6.2f" % m[i, j]
        print(row)
    print()
    print("  %-34s %8s %8s %8s" % ("", "episodes", "success", "steps"))
    print("  %-34s %8d %8.4f %8.2f"
          % ("primary (own block, can store)", r["primary_episodes"],
             r["primary_success"], r["primary_steps"]))
    print("  %-34s %8d %8.4f %8.2f"
          % ("revisit (store locked, live envs)", r["revisit_episodes"],
             r["revisit_success"], r["revisit_steps"]))
    print("  %-34s %8s %8.4f"
          % ("revisit incl. dead envs", "-", r["revisit_success_all"]))
    print()
    print("  worst per-env retention delta (own block -> final): %+.4f"
          % r["worst_retention_delta"])
    print("  a NEGATIVE value is forgetting; 0 or positive is not.")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"summary": r, "matrix": m.tolist()}, fh, indent=1)
        print("  wrote", a.json)


if __name__ == "__main__":
    main()
