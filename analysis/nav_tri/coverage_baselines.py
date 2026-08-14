"""What coverage is worth chasing, and what each source of action noise costs.

The arena `train_navigate` trains on has **no interior obstacles**. Movement is
`pos_f = clip(pos_f + action * scale, 0, size-1)` (`world/vec_env.py:410`); the
four "walls" are boundary planes at x,y = -0.5 and size-0.5 whose only role is
to carry the +/-1 stripe codes the sensory cone ray-casts
(`world/env.py:141-166`). Nothing blocks a step. So the geometry of both target
behaviours is trivially known in advance:

  explore  -- a lawnmower sweep visits one NEW cell per step, so
              `mean_coverage` at T steps is (T+1)/size**2 and `cells_per_step`
              is 1.0. At size=20, T=200 that is 0.5025 -- a HARD ceiling that
              no policy can pass, because a step can add at most one cell.
  exploit  -- a beeline. `mean_steps` is the Euclidean start-goal distance
              divided by the step magnitude, minus whatever `goal_radius`
              forgives.

That makes the interesting question empirical rather than architectural: how
far up that range does each *behaviour class* get, and how much does each
source of action noise cost? Both are pure geometry -- no encoder, no scaffold,
no GPU -- so they are answerable in seconds instead of GPU-hours, and the
answers set the reference lines every trained policy in
`docs/EXPERIMENTS_NAV_TRI.md` is scored against.

The policies below are deliberately *scripted*, not learned. They bracket the
space a learned policy could land in:

  uniform     -- direction resampled uniformly every step (a random walk)
  persistent  -- heading diffuses by N(0, sigma) per step (a correlated walk);
                 sigma=0 is ballistic, sigma=pi is uniform
  billiard    -- straight until the boundary, then a specular reflection
  serpentine  -- the lawnmower, i.e. the ceiling

Two noise sources are modelled on top of any of them, because both are live
knobs in the trainer and both act on the *behaviour* policy while evaluation
scores the mean:

  eps    -- `epsilon_explore`: with probability eps the step is replaced by a
            uniform random direction (`rollout/collector.py`).
  sigma_a -- `exp(init_log_std)`: Gaussian noise added to each action
            component. At `init_log_std=-1.8` that is 0.165 of a cell.

Usage:
    python -m analysis.nav_tri.coverage_baselines
    python -m analysis.nav_tri.coverage_baselines --steps 400 --size 20
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# --------------------------------------------------------------------------
# The environment, reduced to the two lines that matter
# --------------------------------------------------------------------------


def _snap(pos_f: np.ndarray, size: int) -> np.ndarray:
    """`world/vec_env.py:275` -- round, then clip into the grid."""
    return np.clip(np.round(pos_f), 0, size - 1).astype(np.int64)


def _rollout(headings_fn, *, n: int, steps: int, size: int, mag: float,
             eps: float, sigma_a: float, rng: np.random.Generator,
             reflect: bool = False) -> np.ndarray:
    """Run `n` episodes and return the (n, steps+1, 2) snapped cell track.

    `headings_fn(theta, t, hit)` returns the next heading given the current
    one, the step index, and which boundary (if any) the previous step hit.
    """
    pos_f = rng.uniform(0, size - 1, size=(n, 2))
    theta = rng.uniform(-np.pi, np.pi, size=n)
    track = np.empty((n, steps + 1, 2), dtype=np.int64)
    track[:, 0] = _snap(pos_f, size)

    for t in range(steps):
        theta = headings_fn(theta, t)
        step = np.stack([np.cos(theta), np.sin(theta)], axis=1) * mag
        if sigma_a > 0:
            step = step + rng.normal(0.0, sigma_a, size=step.shape)
        if eps > 0:
            # epsilon_explore replaces the action outright with a uniform
            # random direction, at the same magnitude the policy would use.
            swap = rng.random(n) < eps
            if swap.any():
                phi = rng.uniform(-np.pi, np.pi, size=int(swap.sum()))
                step[swap] = np.stack([np.cos(phi), np.sin(phi)], 1) * mag
        new = np.clip(pos_f + step, 0.0, float(size - 1))
        if reflect:
            # A specular bounce off whichever boundary absorbed the step. The
            # env itself does not reflect -- it clips -- so this models a
            # POLICY that turns at the wall, which is what a good sweep does.
            hit_x = (new[:, 0] <= 0.0) | (new[:, 0] >= size - 1)
            hit_y = (new[:, 1] <= 0.0) | (new[:, 1] >= size - 1)
            theta = np.where(hit_x, np.pi - theta, theta)
            theta = np.where(hit_y, -theta, theta)
        pos_f = new
        track[:, t + 1] = _snap(pos_f, size)
    return track


def _coverage(track: np.ndarray, size: int) -> tuple[float, float]:
    """(mean per-episode coverage, union coverage over all episodes)."""
    flat = track[..., 0] * size + track[..., 1]
    per = np.array([len(np.unique(row)) for row in flat]) / float(size * size)
    union = len(np.unique(flat)) / float(size * size)
    return float(per.mean()), float(union)


# --------------------------------------------------------------------------
# The behaviour classes
# --------------------------------------------------------------------------


def _uniform(rng, n):
    return lambda theta, t: rng.uniform(-np.pi, np.pi, size=n)


def _persistent(rng, n, sigma):
    return lambda theta, t: theta + rng.normal(0.0, sigma, size=n)


def _serpentine_track(*, n, steps, size, rng):
    """The lawnmower: the ceiling, run as a track so it scores identically.

    Starts at a random cell, walks to the nearest corner, then sweeps columns.
    Any step that would leave the grid is spent in place, which is exactly the
    penalty a real sweep pays for its turns.
    """
    track = np.empty((n, steps + 1, 2), dtype=np.int64)
    for i in range(n):
        start = rng.integers(0, size, size=2)
        path = [tuple(start)]
        x, y = int(start[0]), int(start[1])
        while y > 0:                       # drop to the bottom row
            y -= 1
            path.append((x, y))
        while x > 0:                       # then to the left edge
            x -= 1
            path.append((x, y))
        up = True
        while len(path) <= steps:
            for _ in range(size - 1):      # sweep a column
                y = y + 1 if up else y - 1
                path.append((x, y))
            x += 1
            if x >= size:
                break
            path.append((x, y))
            up = not up
        while len(path) <= steps:          # ran out of grid: stand still
            path.append(path[-1])
        track[i] = np.array(path[:steps + 1])
    return track


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--trials", type=int, default=32,
                   help="episodes per condition; union coverage is over these")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    size, steps, n = args.size, args.steps, args.trials
    ceiling = min(1.0, (steps + 1) / float(size * size))
    print(f"grid {size}x{size} = {size*size} cells, {steps} steps, "
          f"{n} trials/condition")
    print(f"coverage ceiling (one new cell per step) = {ceiling:.4f}\n")

    rows = []

    def run(name, track):
        cov, union = _coverage(track, size)
        rows.append({"policy": name, "mean_coverage": cov,
                     "cells_per_step": cov * size * size / steps,
                     "union_coverage": union})
        print(f"  {name:<44s} cov={cov:.4f}  cells/step={cov*size*size/steps:.3f}"
              f"  union={union:.4f}")

    rng = np.random.default_rng(args.seed)
    print("--- behaviour class, noiseless, magnitude 1.0 ---")
    run("serpentine (lawnmower ceiling)",
        _serpentine_track(n=n, steps=steps, size=size, rng=rng))
    run("billiard (straight, specular bounce)",
        _rollout(_persistent(rng, n, 0.0), n=n, steps=steps, size=size,
                 mag=1.0, eps=0.0, sigma_a=0.0, rng=rng, reflect=True))
    for sigma in (0.05, 0.2, 0.5, 1.0):
        run(f"persistent walk sigma={sigma}",
            _rollout(_persistent(rng, n, sigma), n=n, steps=steps, size=size,
                     mag=1.0, eps=0.0, sigma_a=0.0, rng=rng))
    run("uniform random walk",
        _rollout(_uniform(rng, n), n=n, steps=steps, size=size,
                 mag=1.0, eps=0.0, sigma_a=0.0, rng=rng))

    print("\n--- step magnitude (billiard) ---")
    for mag in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        run(f"billiard |a|={mag}",
            _rollout(_persistent(rng, n, 0.0), n=n, steps=steps, size=size,
                     mag=mag, eps=0.0, sigma_a=0.0, rng=rng, reflect=True))

    print("\n--- epsilon_explore, on top of a billiard ---")
    for eps in (0.0, 0.05, 0.1, 0.2, 0.4):
        run(f"billiard + eps={eps}",
            _rollout(_persistent(rng, n, 0.0), n=n, steps=steps, size=size,
                     mag=1.0, eps=eps, sigma_a=0.0, rng=rng, reflect=True))

    print("\n--- action Gaussian sigma = exp(init_log_std), billiard ---")
    for ils in (-1.8, -1.2, -0.8, -0.5, 0.0):
        run(f"billiard + init_log_std={ils} (sigma={np.exp(ils):.3f})",
            _rollout(_persistent(rng, n, 0.0), n=n, steps=steps, size=size,
                     mag=1.0, eps=0.0, sigma_a=float(np.exp(ils)), rng=rng,
                     reflect=True))

    print("\n--- exploit reference: straight-line steps to goal ---")
    g = np.random.default_rng(args.seed + 1)
    a = g.integers(0, size, size=(20000, 2)).astype(float)
    b = g.integers(0, size, size=(20000, 2)).astype(float)
    d = np.linalg.norm(a - b, axis=1)
    d = d[d > 0]
    for radius in (0.5, 1.0):
        for mag in (1.0, 1.5, 2.0):
            steps_ideal = np.maximum(0.0, d - radius) / mag
            print(f"  goal_radius={radius} |a|={mag}: "
                  f"mean_steps_ideal={steps_ideal.mean():.2f}  "
                  f"median={np.median(steps_ideal):.2f}")
    print(f"  (mean start-goal Euclidean distance = {d.mean():.2f})")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"size": size, "steps": steps, "trials": n,
                       "ceiling": ceiling, "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
