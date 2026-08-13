"""What does a given `mean_coverage` actually mean?

`evaluate_exploration` reports a fraction of cells visited, and a number like
0.52 is only interpretable against something. This prints the coverage that
each *memoryless* movement statistic earns under the env's exact step
semantics -- no policy, no network, no scaffold. The walkers themselves are in
`probes/walks.py`; this is the table.

    python -m hopfield_nav.probes.coverage_reference --steps 400 --size 20

The result that matters: nothing memoryless beats ~0.56 on a 20x20 arena in 400
steps. Every explore policy in this project's history scores 0.50-0.55, so the
whole lineage has been at the ceiling of the family its reward knobs can reach,
and the gap to 1.0 needs the policy to remember where it has been.
"""
from __future__ import annotations

import argparse

import numpy as np

from ..world.walks import (
    bounce_walk, correlated_walk, diffusive_walk, lawnmower_coverage,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--trials", type=int, default=512)
    p.add_argument("--strides", type=float, nargs="+",
                   default=[1.0, 2.0, 3.0, 4.0, 6.0])
    p.add_argument("--turn_sigmas", type=float, nargs="+",
                   default=[0.1, 0.4, 0.8, 1.6, 3.0])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    print(f"{args.size}x{args.size} arena, {args.steps} steps, "
          f"{args.trials} trials, walls CLIP (a blocked step is lost)\n")
    print(f"{'family':<24} " + " ".join(f"stride={s:<5g}" for s in args.strides))
    print("-" * (24 + 13 * len(args.strides)))

    def row(label, fn):
        cells = [f"{fn(s).mean():.3f}" for s in args.strides]
        print(f"{label:<24} " + " ".join(f"{c:<12}" for c in cells))

    row("diffusive",
        lambda s: diffusive_walk(args.trials, args.size, args.steps, s, rng))
    for ts in args.turn_sigmas:
        row(f"correlated turn={ts:g}",
            lambda s, ts=ts: correlated_walk(args.trials, args.size,
                                             args.steps, s, ts, rng))
    row("bounce (straight)",
        lambda s: bounce_walk(args.trials, args.size, args.steps, s, rng))
    print(f"\n{'lawnmower (ideal)':<24} "
          f"{lawnmower_coverage(args.size, args.steps):.3f}"
          f"   <- upper bound at this step budget")


if __name__ == "__main__":
    main()
