"""What does a given `mean_coverage` actually mean?

`evaluate_exploration` reports a fraction of cells visited, and a number like
0.52 is only interpretable against something. This simulates the movement
statistics an agent could have -- with no policy, no network and no scaffold --
under the env's exact step semantics, and reports the coverage each one earns.

The step semantics matter and are copied from `ContinuousVecEnv.step_batch`:
position is continuous, the action is added to it, and the result is
**clipped** to the arena rather than reflected. A step into a wall is therefore
absorbed: the agent stays where it is and loses the step. Coverage counts
distinct *snapped* cells, exactly as `batched_exploration_trials` does.

The families, in order of how much structure they assume:

    diffusive     direction redrawn uniformly every step. A random walk.
    correlated    heading does a wrapped random walk of width `turn_sigma`;
                  turn_sigma -> 0 is ballistic, -> large is diffusive. This is
                  the family `persistence_bonus` moves the policy along, since
                  that bonus is exactly `cos` of successive headings.
    bounce        straight until the step is clipped, then a new direction.
                  Memoryless, and the cheapest thing that is not diffusive.
    lawnmower     the ideal boustrophedon sweep. An upper bound, and not
                  reachable without knowing where you have been.

    python -m hopfield_nav.probes.coverage_reference --steps 400 --size 20
"""
from __future__ import annotations

import argparse

import numpy as np


def _coverage(pos_f: np.ndarray, size: int, steps: int,
              direction_fn, rng: np.random.RandomState) -> np.ndarray:
    """Run B walkers for `steps` and return each one's covered-cell fraction.

    `direction_fn(t, blocked)` returns the (B, 2) displacement for step t;
    `blocked` says which walkers had their previous step absorbed by the clip.
    """
    B = pos_f.shape[0]
    visited = np.zeros((B, size, size), dtype=bool)
    idx = np.arange(B)
    snapped = np.clip(np.rint(pos_f).astype(int), 0, size - 1)
    visited[idx, snapped[:, 0], snapped[:, 1]] = True
    blocked = np.zeros(B, dtype=bool)
    for t in range(steps):
        step = direction_fn(t, blocked)
        before = pos_f.copy()
        pos_f = np.clip(pos_f + step, 0.0, float(size - 1))
        moved = np.linalg.norm(pos_f - before, axis=1)
        blocked = moved < 1e-9
        snapped = np.clip(np.rint(pos_f).astype(int), 0, size - 1)
        visited[idx, snapped[:, 0], snapped[:, 1]] = True
    return visited.reshape(B, -1).sum(1) / float(size * size)


def _start(B: int, size: int, rng: np.random.RandomState) -> np.ndarray:
    """Uniform integer cells, matching `random_start` / `reset_all`."""
    return rng.randint(0, size, size=(B, 2)).astype(np.float64)


def _unit(theta: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def run_family(name: str, B: int, size: int, steps: int, stride: float,
               turn_sigma: float, rng: np.random.RandomState) -> np.ndarray:
    pos = _start(B, size, rng)
    if name == "diffusive":
        def fn(t, blocked):
            return _unit(rng.uniform(0, 2 * np.pi, B)) * stride
    elif name == "correlated":
        theta = rng.uniform(0, 2 * np.pi, B)

        def fn(t, blocked):
            nonlocal theta
            theta = theta + rng.normal(0.0, turn_sigma, B)
            # A clipped step means the wall is ahead; without this the walker
            # pushes into it forever, which is exactly the dead fixed point a
            # collapsed policy falls into.
            if blocked.any():
                theta[blocked] = rng.uniform(0, 2 * np.pi, int(blocked.sum()))
            return _unit(theta) * stride
    elif name == "bounce":
        theta = rng.uniform(0, 2 * np.pi, B)

        def fn(t, blocked):
            nonlocal theta
            if blocked.any():
                theta[blocked] = rng.uniform(0, 2 * np.pi, int(blocked.sum()))
            return _unit(theta) * stride
    elif name == "lawnmower":
        # Column-major serpentine at one cell per step, from the walker's own
        # start. Coverage is then purely a function of the step budget.
        def fn(t, blocked):
            return np.zeros((B, 2))
        order = []
        for x in range(size):
            ys = range(size) if x % 2 == 0 else range(size - 1, -1, -1)
            order.extend((x, y) for y in ys)
        order = np.array(order)
        visited = np.zeros((B, size, size), dtype=bool)
        idx = np.arange(B)
        for t in range(min(steps + 1, len(order))):
            visited[idx, order[t, 0], order[t, 1]] = True
        return visited.reshape(B, -1).sum(1) / float(size * size)
    else:
        raise ValueError(f"unknown family {name!r}")
    return _coverage(pos, size, steps, fn, rng)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--trials", type=int, default=512)
    p.add_argument("--strides", type=float, nargs="+",
                   default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--turn_sigmas", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    print(f"{args.size}x{args.size} arena, {args.steps} steps, "
          f"{args.trials} trials, walls CLIP (a blocked step is lost)\n")

    print(f"{'family':<24} " + " ".join(f"stride={s:<5g}" for s in args.strides))
    print("-" * (24 + 13 * len(args.strides)))

    def row(label, name, turn_sigma=0.0):
        cells = []
        for stride in args.strides:
            cov = run_family(name, args.trials, args.size, args.steps, stride,
                             turn_sigma, rng)
            cells.append(f"{cov.mean():.3f}")
        print(f"{label:<24} " + " ".join(f"{c:<12}" for c in cells))

    row("diffusive", "diffusive")
    for ts in args.turn_sigmas:
        row(f"correlated turn={ts:g}", "correlated", ts)
    row("bounce (straight)", "bounce")
    lm = run_family("lawnmower", args.trials, args.size, args.steps, 1.0, 0.0,
                    rng)
    print(f"\n{'lawnmower (ideal)':<24} {lm.mean():.3f}"
          f"   <- upper bound at this step budget")


if __name__ == "__main__":
    main()
