"""The ceiling, the floor, and where the in-context policy sits between them.

A null is not a measurement until it has both ends. The withdrawn version of
section 5.2 had neither: no floor (a random walker scores 0.208 on these
environments, and the policy scored 0.100) and no ceiling (nothing established
what a policy that was simply *told* the answer would reach). This reports all
three numbers together, because any one of them alone is misleading.

    floor     a random walker on the same environments and step budget.
    ceiling   `goal_channel=abs` -- the agent is handed the goal's coordinates
              and must still localise itself from the barcode ray-cast. No
              amount of in-context memory can beat this, because remembering
              where the goal is does not tell you where you are.
    actual    the real in-context arm, and its episodic control.

The headline is then a *fraction of the available signal*:

    captured = (in-context - episodic) / (ceiling - episodic)

which is bounded, interpretable in either direction, and cannot be produced by
a broken policy -- if the in-context arm sits at the floor, the numerator is
zero and the denominator is not, and that is a real answer rather than an
artefact.

`sanity` is `goal_channel=rel`, the follow-the-arrow arm. If that is not near
the top, the network cannot act on the goal even when handed it directly and
nothing else in the wave is interpretable.

    python -m analysis.continual.incontext_upper_bound \\
        --logs <dir of training logs> --incontext_dir <dir of eval JSONs>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from .incontext_generalization import chance_rate

#: Filename prefix -> what the arm is for.
ARMS = {
    "ceilabs": "ceiling (told the goal, must localise)",
    "ceilrel": "sanity (told the displacement)",
    "carry": "carry control (goal shown in episode 1 only)",
    "ic": "in-context (state carried)",
    "ep": "episodic control (state reset)",
}


def final_holdout(path: str) -> tuple[float | None, float | None, float | None]:
    """(final holdout nav_det, final pool nav_det, end slope) from a train log.

    The slope is over the last third of the recorded holdout points, and it is
    not decoration: a run still climbing when its budget ran out has not
    measured a ceiling, it has measured where it got to -- which is the error
    the joint-ceiling sweep made earlier in this project.
    """
    txt = open(path, errors="ignore").read()
    vals = [float(v) for v in
            re.findall(r"HOLDOUT nav_det=([0-9.]+)", txt)]
    pool = [float(v) for v in
            re.findall(r"HOLDOUT nav_det=[0-9.]+\s+\(pool ([0-9.]+)", txt)]
    if not vals:
        return None, None, None
    tail = vals[max(0, len(vals) - max(3, len(vals) // 3)):]
    slope = (float(np.polyfit(range(len(tail)), tail, 1)[0])
             if len(tail) >= 2 else 0.0)
    return vals[-1], (pool[-1] if pool else None), slope


def collect(logs: str) -> dict[str, dict[int, list[dict]]]:
    """-> {arm: {hidden: [per-seed dicts]}}"""
    out: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(logs, "*.log"))):
        m = re.match(r"(ceilabs|ceilrel|carry|ic|ep)_h(\d+)_s(\d+)\.log$",
                     os.path.basename(p))
        if not m:
            continue
        arm, hid, seed = m.group(1), int(m.group(2)), int(m.group(3))
        ho, pool, slope = final_holdout(p)
        if ho is None:
            continue
        out[arm][hid].append({"seed": seed, "holdout": ho, "pool": pool,
                              "slope": slope})
    return {a: dict(v) for a, v in out.items()}


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", required=True)
    p.add_argument("--incontext_dir", default=None)
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[9001, 9002, 9003])
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=60)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=64)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    floor = chance_rate(args.eval_seeds, args.n_envs, args.size,
                        args.observation_size, args.max_steps, args.n_trials)
    data = collect(args.logs)

    print("=" * 96)
    print("SECTION 5.2, WITH A CEILING   --  held-out single-episode success")
    print("=" * 96)
    print(f"\nfloor (random walker, same envs, {args.max_steps}-step episodes): "
          f"{floor:.4f}\n")
    print(f"{'arm':>10}  {'hidden':>7}  {'n':>2}  {'holdout':>9}  {'pool':>7}"
          f"  {'gap':>6}  {'end slope':>10}  {'vs floor':>9}")
    print("-" * 82)
    report: dict = {"floor": floor, "arms": {}}
    for arm in ("ceilrel", "ceilabs", "carry", "ic", "ep"):
        for hid in sorted(data.get(arm, {})):
            rows = data[arm][hid]
            ho, pool = _mean(rows, "holdout"), _mean(rows, "pool")
            slope = _mean(rows, "slope")
            gap = pool / ho if ho else float("nan")
            print(f"{arm:>10}  {hid:>7}  {len(rows):>2}  {ho:>9.4f}  "
                  f"{pool:>7.4f}  {gap:>6.1f}x  {slope:>10.5f}  "
                  f"{ho / floor:>8.2f}x")
            report["arms"].setdefault(arm, {})[str(hid)] = {
                "n": len(rows), "holdout": ho, "pool": pool, "gap": gap,
                "end_slope": slope, "vs_floor": ho / floor if floor else None,
                "per_seed": rows,
            }

    # --- the bound -----------------------------------------------------------
    def best(arm):
        rows = report["arms"].get(arm, {})
        if not rows:
            return None, None
        h = max(rows, key=lambda k: rows[k]["holdout"])
        return h, rows[h]

    h_c, ceil = best("ceilabs")
    h_i, ic = best("ic")
    h_e, ep = best("ep")
    print("\n" + "-" * 96)
    print("READING")
    if ceil and ic and ep:
        span = ceil["holdout"] - ep["holdout"]
        captured = ((ic["holdout"] - ep["holdout"]) / span
                    if span > 1e-9 else float("nan"))
        report.update({"ceiling": ceil["holdout"], "ceiling_hidden": h_c,
                       "in_context": ic["holdout"], "in_context_hidden": h_i,
                       "episodic": ep["holdout"], "captured_fraction": captured})
        print(f"  ceiling (told the goal, hidden={h_c}): {ceil['holdout']:.4f}")
        print(f"  in-context (best, hidden={h_i}):       {ic['holdout']:.4f}")
        print(f"  episodic control:                      {ep['holdout']:.4f}")
        print(f"  floor (random walker):                 {floor:.4f}")
        print(f"\n  captured fraction of the available signal: {captured:+.3f}")
        if ceil["holdout"] < 2 * floor:
            print("\n  WARNING: the ceiling itself is close to the floor. A "
                  "policy that cannot navigate even when told the goal makes "
                  "every other number here uninterpretable -- fix that before "
                  "reading anything about memory.")
        still = [f"{a} h{h}" for a, hs in report["arms"].items()
                 for h, r in hs.items() if (r["end_slope"] or 0) > 0.005]
        if still:
            print(f"\n  STILL CLIMBING at the budget limit: {', '.join(still)}. "
                  "These have not measured a ceiling, only where they got to.")
    else:
        print("  (incomplete: need ceilabs, ic and ep arms)")
    print("-" * 96)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
