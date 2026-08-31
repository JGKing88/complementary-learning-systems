"""Did the in-context pretraining learn a strategy, or memorise its pool?

Section 5.2 asks whether a frozen recurrent policy can adapt to a *new*
environment through recurrent activity alone. That question is only meaningful
if the policy can operate in a new environment at all. If pretraining across a
pool of N environments produced a network that solves those N and nothing else,
then the held-out evaluation is measuring a broken policy, a flat
success-vs-episode curve is the only possible outcome, and the resulting null
says nothing about activation memory.

This is the precondition the original design lacked. It compares the
pretrained policy's success on the environments it trained on against its
success on the held-out environments 5.2 evaluates, and states plainly whether
the gap is a memorisation signature.

The two numbers come from different places on purpose. The training-pool figure
is parsed out of the pretraining run's own final evaluation, so it is what that
run actually achieved rather than a re-derivation; the held-out figure is
episode-1 success from the 5.2 evaluation itself, which is a fresh episode in an
unseen environment and therefore the right comparison.

    python -m analysis.continual.incontext_generalization \\
        --logs <dir with pre_{arm}_s{seed}.log> \\
        --incontext_dir <dir with incontext_s*.json> --out gen.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

#: Above this, the pool was learned rather than generalised from, and section
#: 5.2's held-out evaluation is measuring a policy that cannot navigate.
MEMORISATION_RATIO = 3.0


def training_pool_scores(logs: str, arm: str) -> dict[int, float]:
    """Final per-env nav_det on the training pool, per seed, from the log."""
    out: dict[int, float] = {}
    for p in sorted(glob.glob(os.path.join(logs, f"pre_{arm}_s*.log"))):
        m = re.search(r"_s(\d+)\.log$", p)
        if not m:
            continue
        txt = open(p, errors="ignore").read()
        vals = [float(v) for v in
                re.findall(r"eval env_\d+: nav_det=([0-9.]+)", txt)]
        if not vals:
            continue
        # The final evaluation block: one line per env, so the pool size is the
        # highest env index seen plus one.
        idx = [int(i) for i in re.findall(r"eval env_(\d+): nav_det=", txt)]
        pool = max(idx) + 1
        last = vals[-pool:]
        out[int(m.group(1))] = sum(last) / len(last)
    return out


def held_out_scores(incontext_dir: str, arm: str) -> dict[int, float]:
    """Episode-1 success on held-out envs, per seed, from the 5.2 evaluation."""
    out: dict[int, float] = {}
    for p in sorted(glob.glob(os.path.join(incontext_dir, "incontext_s*.json"))):
        m = re.search(r"_s(\d+)\.json$", p)
        if not m:
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        curve = (j.get("arms", {}).get(arm, {}) or {}).get("mean_curve")
        if curve:
            out[int(m.group(1))] = curve[0]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", required=True)
    p.add_argument("--incontext_dir", required=True)
    p.add_argument("--arms", nargs="+", default=["lifetime", "episodic"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    report: dict = {"arms": {}, "ratio_threshold": MEMORISATION_RATIO}
    print(f"{'arm':>10}  {'train pool':>12}  {'held out':>10}  {'ratio':>7}")
    print("-" * 46)
    for arm in args.arms:
        tr = training_pool_scores(args.logs, arm)
        ho = held_out_scores(args.incontext_dir, arm)
        seeds = sorted(set(tr) & set(ho))
        if not seeds:
            print(f"{arm:>10}  (no paired seeds)")
            continue
        tr_m = sum(tr[s] for s in seeds) / len(seeds)
        ho_m = sum(ho[s] for s in seeds) / len(seeds)
        ratio = tr_m / ho_m if ho_m > 0 else float("inf")
        report["arms"][arm] = {
            "seeds": seeds, "train_pool": tr_m, "held_out": ho_m,
            "ratio": ratio,
            "per_seed_train": {str(s): tr[s] for s in seeds},
            "per_seed_held_out": {str(s): ho[s] for s in seeds},
        }
        print(f"{arm:>10}  {tr_m:>12.3f}  {ho_m:>10.3f}  {ratio:>7.1f}x")

    worst = max((a["ratio"] for a in report["arms"].values()), default=0.0)
    report["memorised"] = worst >= MEMORISATION_RATIO
    report["worst_ratio"] = worst
    print()
    if report["memorised"]:
        report["verdict"] = (
            "The pretrained policy solves its training pool and fails on "
            "held-out environments by a wide margin. It learned those "
            "environments rather than a strategy for navigating an unseen one, "
            "so the section 5.2 evaluation is measuring a policy that cannot "
            "navigate its test environments at all. A flat success-vs-episode "
            "curve is the only outcome such a policy can produce, and the null "
            "result therefore does not distinguish 'activation memory cannot "
            "do this job' from 'the pretraining pool was too small to force "
            "learning an adaptation strategy'.")
    else:
        report["verdict"] = (
            "The pretrained policy transfers to held-out environments, so the "
            "section 5.2 evaluation is measuring a working policy and its "
            "success-vs-episode curve is interpretable.")
    print(report["verdict"])

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
