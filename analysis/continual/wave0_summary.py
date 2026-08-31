"""Read Wave 0's outputs and answer the one question that gates everything.

Wave 0 measures the axes of the continual figure, not a method. Its job is to
say whether the recorded ~0.19 retention is a *forgetting* result or partly a
*capacity* result, and it does that by putting three numbers side by side:

    T0.3  oracle ceiling  -- what a perfect policy scores under the step cap
    T0.1  joint ceiling   -- what one network scores on all N envs at once
    T0.4  sequential floor-- what naive streaming SGD retains

If T0.1 is near T0.3, every point of the T0.1 - T0.4 gap is genuinely
forgetting and Tier 2 is interpretable. If T0.1 is well below T0.3 at every
capacity, the network cannot hold N envs at all, no continual method can exceed
that, and the honest headline changes. This module prints which of those it is.

Reads whatever exists and says what is missing rather than failing, so it can
be run against a partially-finished job.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _sem(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))


# ---------------------------------------------------------------------------
# T0.3
# ---------------------------------------------------------------------------

def load_oracle(d: str) -> float | None:
    p = os.path.join(d, "T0.3_oracle.json")
    if not os.path.exists(p):
        return None
    return float(json.load(open(p))["overall_oracle_reached"])


# ---------------------------------------------------------------------------
# T0.1 -- from train_rnn's final.pt history (list of {env_idx: metrics})
# ---------------------------------------------------------------------------

_T01 = re.compile(r"wave0_T0\.1_h(\d+)_l(\d+)_s(\d+)$")


def load_joint(runs_root: str) -> dict[tuple[int, int], dict]:
    """-> {(hidden, layers): {"final": [...], "at200": [...], "seeds": n}}"""
    import torch

    out: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"final": [], "at200": [], "seeds": 0})
    for path in sorted(glob.glob(os.path.join(runs_root, "rnn", "wave0_T0.1_*"))):
        m = _T01.search(path)
        ckpt = os.path.join(path, "final.pt")
        if m is None or not os.path.exists(ckpt):
            continue
        hid, lay = int(m.group(1)), int(m.group(2))
        try:
            d = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception as e:                       # a run that died mid-write
            print(f"  ! could not read {ckpt}: {e}")
            continue
        history = d.get("history") or []
        if not history:
            continue
        eval_every = int((d.get("cfg") or {}).get("eval_every") or 25)
        key = (hid, lay)
        out[key]["final"].append(_mean([mm["nav_det"] for mm in history[-1].values()]))
        # The budget-matched point: joint training given the same number of
        # rollouts per env the sequential protocol gets (200 updates).
        idx = max(0, min(len(history) - 1, 200 // eval_every - 1))
        out[key]["at200"].append(_mean([mm["nav_det"] for mm in history[idx].values()]))
        out[key]["seeds"] += 1
    return dict(out)


# ---------------------------------------------------------------------------
# T0.4 -- from the standard history JSON
# ---------------------------------------------------------------------------

def _final_block_per_env(hist: dict, tail_frac: float = 0.2) -> dict[int, float]:
    """Mean `reached` per env over the last `tail_frac` of the final block."""
    blocks, trace = hist.get("blocks") or [], hist.get("trace") or []
    if not blocks or not trace:
        return {}
    lo_all, hi, _ = blocks[-1]
    lo = lo_all + int((1 - tail_frac) * (hi - lo_all))
    acc: dict[int, list] = defaultdict(list)
    for step, _train_env, inner in trace:
        if step < lo or step > hi:
            continue
        for k, v in inner.items():
            r = v.get("reached")
            if r is None:
                continue
            acc[int(k)].extend(r if isinstance(r, list) else [r])
    return {k: _mean(v) for k, v in sorted(acc.items())}


def load_sequential(d: str, arm: str) -> dict:
    """-> {"per_env": {i: [across seeds]}, "seeds": n}"""
    per_env: dict[int, list] = defaultdict(list)
    n = 0
    for p in sorted(glob.glob(os.path.join(d, f"T0.4_{arm}_s*.json"))):
        try:
            hist = json.load(open(p))
        except Exception as e:
            print(f"  ! could not read {p}: {e}")
            continue
        fin = _final_block_per_env(hist)
        if not fin:
            continue
        for k, v in fin.items():
            per_env[k].append(v)
        n += 1
    return {"per_env": dict(per_env), "seeds": n}


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True, help="Wave-0 history directory.")
    p.add_argument("--runs_root", default=os.environ.get("CLS_RUNS"),
                   help="Root holding rnn/wave0_T0.1_* (default: $CLS_RUNS).")
    args = p.parse_args()

    print("=" * 72)
    print("WAVE 0 SUMMARY   docs/CONTINUAL_CONTROLS_PLAN.md section 2")
    print("=" * 72)

    # -- T0.3 ---------------------------------------------------------------
    oracle = load_oracle(args.dir)
    print("\nT0.3  ORACLE CEILING (what a perfect policy scores)")
    if oracle is None:
        print("  missing.")
    else:
        print(f"  {oracle:.4f}"
              + ("" if oracle > 0.99 else
                 "   <- `reached` is capped here, NOT at 1.0"))

    # -- T0.1 ---------------------------------------------------------------
    print("\nT0.1  JOINT CEILING (one net, all envs at once)")
    joint = load_joint(args.runs_root) if args.runs_root else {}
    best = None
    if not joint:
        print("  missing.")
    else:
        print(f"  {'hidden':>7} {'layers':>7} {'seeds':>6} "
              f"{'final':>16} {'@200upd':>10}")
        for (hid, lay) in sorted(joint):
            v = joint[(hid, lay)]
            fm, fs = _mean(v["final"]), _sem(v["final"])
            print(f"  {hid:>7} {lay:>7} {v['seeds']:>6} "
                  f"{fm:>9.4f} +/-{fs:<5.4f} {_mean(v['at200']):>10.4f}")
            if best is None or fm > best[1]:
                best = ((hid, lay), fm)

    # -- T0.4 ---------------------------------------------------------------
    print("\nT0.4  SEQUENTIAL FLOOR (naive streaming SGD, from scratch)")
    floors = {}
    for arm, label in [("noprev", "no prev_action (legacy surface)"),
                       ("prev", "with prev_action (settled surface)")]:
        r = load_sequential(args.dir, arm)
        if not r["seeds"]:
            print(f"  {label}: missing.")
            continue
        pe = r["per_env"]
        n_env = max(pe) + 1 if pe else 0
        old = [_mean(pe[i]) for i in range(n_env - 1) if i in pe]
        cur = _mean(pe.get(n_env - 1, []))
        floors[arm] = _mean(old)
        per = "  ".join(f"e{i}={_mean(pe[i]):.3f}" for i in sorted(pe))
        print(f"  {label}  ({r['seeds']} seeds)")
        print(f"    {per}")
        print(f"    retained (envs 0..N-2): {_mean(old):.4f}    "
              f"current env: {cur:.4f}")

    # -- the verdict --------------------------------------------------------
    print("\n" + "-" * 72)
    print("VERDICT")
    if best is None or oracle is None or not floors:
        print("  Incomplete -- rerun once the missing pieces above have landed.")
    else:
        (hid, lay), ceil = best
        ref = oracle
        floor = min(floors.values())
        print(f"  oracle {ref:.3f}  |  joint ceiling {ceil:.3f} "
              f"(best: hidden={hid}, layers={lay})  |  floor {floor:.3f}")
        if ceil >= 0.9 * ref:
            print("  -> The network CAN hold all envs at once. The retention gap")
            print("     is genuinely forgetting, and Tier 2 is interpretable.")
            print(f"     Headroom for continual methods: {ceil - floor:.3f}.")
            print("     T0.2 (per-env experts) is not needed; skip it.")
        else:
            print("  -> The joint ceiling is well below the oracle at every")
            print("     capacity tested. Part of the recorded 'forgetting' is a")
            print("     CAPACITY result and must be reported as such.")
            print("     Run T0.2 (per-env experts) to size the interference,")
            print("     and extend the capacity sweep before starting Tier 2.")
    print("-" * 72)


if __name__ == "__main__":
    main()
