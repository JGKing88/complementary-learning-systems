"""Is section 5.2's measurement interpretable at all? Two preconditions.

5.2 asks whether a frozen recurrent policy adapts to a *new* environment
through recurrent activity alone. The measurement is behavioural: success on
episode k against success on episode 1. That only means something if the policy
can act in a new environment in the first place -- because **memory is only
observable here through behaviour.** A hidden state that perfectly encoded the
goal would raise no success rate at all in a policy that cannot navigate to a
known location. The measurement channel would be closed, and a flat curve would
be guaranteed regardless of what the state contains.

So two things have to hold before a null means anything, and neither was
checked when 5.2 was first reported.

**1. The policy must generalise.** If pretraining across a pool of N
environments produced a network that solves those N and nothing else, the
held-out evaluation is scoring a policy that never learned the task. The
train-pool-to-held-out ratio is the memorisation signature.

**2. The policy must beat chance on a held-out environment.** This is the
stronger and more basic test, and it is the one that actually disqualified the
original run. "Chance" is not a guess: it is measured, by running a random
walker on the same environments under the same per-episode step budget. A
trained policy scoring *below* that has learned something actively
counterproductive on new environments -- confidently heading somewhere wrong is
worse than exploring -- and cannot express memory through behaviour no matter
how much of it there is.

The numbers come from three places on purpose. The training-pool figure is
parsed from the pretraining run's own final evaluation, so it is what that run
achieved rather than a re-derivation. The held-out figure is episode-1 success
from the 5.2 evaluation itself. The chance rate is simulated here, on
environments built from the same seeds the evaluation used.

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

import numpy as np

from hopfield_nav.world.env import GridEnv, at_goal
from hopfield_nav.world.vec_env import make_vec

#: Above this, the pool was learned rather than generalised from, and section
#: 5.2's held-out evaluation is measuring a policy that cannot navigate.
MEMORISATION_RATIO = 3.0


def chance_rate(seeds, n_envs: int, size: int, observation_size: int,
                max_steps: int, n_trials: int) -> float:
    """Episode success for a random walker, on the evaluation's own environments.

    The floor a policy has to clear before "did it adapt?" is a question with an
    answer. Actions are drawn from a unit Gaussian, which is exactly what the
    policy itself emits at `init_log_std = 0` before any learning -- so this is
    the same agent with its weights untrained, not an arbitrary baseline.
    """
    rates = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        envs = [GridEnv(size=size, observation_size=observation_size,
                        seed=int(rng.randint(0, 10_000_000)))
                for _ in range(n_envs)]
        for env in envs:
            vec = make_vec(env, n_trials, "continuous", 1.0)
            vec.reset_all()
            hit = np.zeros(n_trials, dtype=bool)
            for _ in range(max_steps):
                hit |= at_goal(vec)
                vec.step_batch(rng.randn(n_trials, 2).astype(np.float32))
            hit |= at_goal(vec)
            rates.append(float(hit.mean()))
    return float(np.mean(rates)) if rates else float("nan")


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
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[9001, 9002, 9003],
                   help="The seeds the 5.2 evaluation built its held-out envs "
                        "from. The chance rate is simulated on the same ones.")
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=60)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=64)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    chance = chance_rate(args.eval_seeds, args.n_envs, args.size,
                         args.observation_size, args.max_steps, args.n_trials)
    print(f"chance (random walker, same envs, {args.max_steps}-step episodes): "
          f"{chance:.4f}\n")

    report: dict = {"arms": {}, "ratio_threshold": MEMORISATION_RATIO,
                    "chance": chance}
    print(f"{'arm':>10}  {'train pool':>12}  {'held out':>10}  {'ratio':>7}"
          f"  {'vs chance':>10}")
    print("-" * 58)
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
            "vs_chance": ho_m / chance if chance else float("nan"),
            "beats_chance": bool(ho_m > chance),
            "per_seed_train": {str(s): tr[s] for s in seeds},
            "per_seed_held_out": {str(s): ho[s] for s in seeds},
        }
        vs = f"{ho_m / chance:>9.2f}x" if chance else f"{'n/a':>10}"
        print(f"{arm:>10}  {tr_m:>12.3f}  {ho_m:>10.3f}  {ratio:>7.1f}x  {vs}")

    worst = max((a["ratio"] for a in report["arms"].values()), default=0.0)
    report["memorised"] = worst >= MEMORISATION_RATIO
    report["worst_ratio"] = worst
    # The gate. Not "is the policy good" -- "is it above the floor set by having
    # learned nothing at all", which is the weakest possible bar and the one
    # that has to clear before a behavioural null can mean anything.
    report["gate_passed"] = all(a["beats_chance"]
                                for a in report["arms"].values())
    print()
    if not report["gate_passed"]:
        report["verdict"] = (
            f"The policy scores BELOW chance on held-out environments "
            f"({report['arms'].get('lifetime', {}).get('held_out', float('nan')):.3f} "
            f"against a measured random-walk rate of {chance:.3f}). Section "
            "5.2's statistic is behavioural, so memory can only show up as a "
            "higher success rate -- and a policy that cannot reach a goal in an "
            "unseen arena cannot express memory of where that goal is, however "
            "much of it the hidden state holds. The measurement channel is "
            "closed and a flat curve is guaranteed independently of the "
            "question being asked. The null is not evidence about activation "
            "memory.")
    elif report["memorised"]:
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
            "The pretrained policy transfers to held-out environments and "
            "clears the chance floor, so the section 5.2 evaluation is "
            "measuring a working policy and its success-vs-episode curve is "
            "interpretable.")
    print(report["verdict"])

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
