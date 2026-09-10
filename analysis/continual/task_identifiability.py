"""How much is the oracle task id actually worth in this setting?

Every Wave 3 arm is told which environment it is in, at training time and at
evaluation time. Waves 1 and 2 are not, and neither is the Hopfield store. That
is recorded as a cost axis, but "needs a task id" is only a *large* cost if the
task id is hard to come by -- and in this task it might not be, because the
environments differ by a random barcode painted on their walls and the
observation is a ray-cast of exactly that barcode.

The plan (section 4.3) asks for multi-head in two conditions, oracle and
inferred, on the grounds that the gap between them measures how much of the
problem is task inference rather than forgetting. Plumbing a learned classifier
through the protocol is a second training loop with its own failure modes. This
measures the thing that gap would report, directly and once: fit a classifier
from observations to environment index and see how well it does.

  - Near-perfect => oracle and inferred coincide, the task id is nearly free,
    and Wave 3's advantage over Waves 1 and 2 is much smaller than the word
    "oracle" suggests.
  - Near chance => the oracle task id is a real and large advantage, and every
    Wave 3 number is an upper bound no boundary-free method could reach.

Two things this gets right that the obvious version does not.

**The split is by trajectory, not by sample.** Consecutive observations from one
random walk are strongly correlated; a shuffled split puts neighbouring
timesteps of the same walk on both sides and reports memorisation as accuracy.

**The classifier is swept from weak to strong.** A single linear readout on one
observation answers "is identity sitting in the observation already", which is
the wrong question if the answer is no -- a sequence carries more than a frame,
and identity may be nonlinear in the code. So it runs linear and MLP readouts
over windows of 1 to 64 observations, and the headline is the best of them. A
low number from the *best* classifier tried is worth something; a low number
from the weakest is not.

    python -m analysis.continual.task_identifiability --n_envs 5 --size 20
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hopfield_nav.world.env import GridEnv
from hopfield_nav.world.vec_env import make_vec


def collect_trajectories(envs, steps: int, batch: int, movement_mode: str,
                         scale: float) -> tuple[np.ndarray, np.ndarray]:
    """-> obs (n_traj, steps, D), env index (n_traj,).

    A random walk rather than the trained policy on purpose: the question is
    whether the *environment* is identifiable, and a policy that goes to
    different places in different envs would let the answer come from where the
    agent went rather than from what it saw.
    """
    X, y = [], []
    for i, env in enumerate(envs):
        vec = make_vec(env, batch, movement_mode, scale)
        vec.reset_all()
        frames = []
        for _ in range(steps):
            frames.append(vec.obs_batch().astype(np.float32))
            if movement_mode == "discrete":
                a = np.random.randint(0, 4, size=batch)
            else:
                a = np.random.randn(batch, 2).astype(np.float32)
            vec.step_batch(a)
        X.append(np.stack(frames, axis=1))            # (batch, steps, D)
        y.append(np.full(batch, i, dtype=np.int64))
    return np.concatenate(X, 0), np.concatenate(y, 0)


def windows(X: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_traj, T, D) -> (n_traj * n_windows, k*D), labels repeated."""
    n, T, D = X.shape
    n_win = T // k
    Xw = X[:, : n_win * k].reshape(n, n_win, k * D).reshape(n * n_win, k * D)
    yw = np.repeat(y, n_win)
    return Xw, yw


def fit(Xtr, ytr, Xte, yte, n_classes: int, hidden: int = 0,
        epochs: int = 300, lr: float = 0.01) -> float:
    """Linear (hidden=0) or one-hidden-layer readout. Held-out accuracy."""
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xte_t, yte_t = torch.from_numpy(Xte), torch.from_numpy(yte)
    # Standardised, so a wide window is not simply harder to optimise than a
    # narrow one at the same learning rate.
    mu, sd = Xtr_t.mean(0, keepdim=True), Xtr_t.std(0, keepdim=True) + 1e-6
    Xtr_t, Xte_t = (Xtr_t - mu) / sd, (Xte_t - mu) / sd

    if hidden:
        model = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_classes))
    else:
        model = torch.nn.Linear(Xtr.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(Xtr_t), ytr_t).backward()
        opt.step()
    with torch.no_grad():
        return float((model(Xte_t).argmax(-1) == yte_t).float().mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_envs", type=int, default=5)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=60)
    p.add_argument("--movement_mode", default="continuous")
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--windows", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    envs = [GridEnv(size=args.size, observation_size=args.observation_size,
                    seed=int(rng.randint(0, 10_000_000)))
            for _ in range(args.n_envs)]

    X, y = collect_trajectories(envs, args.steps, args.batch,
                                args.movement_mode, 1.0)
    # Split by trajectory. Consecutive frames of one random walk are strongly
    # correlated, so a shuffled per-frame split would put neighbours on both
    # sides and report memorisation as identifiability.
    order = np.random.permutation(len(X))
    X, y = X[order], y[order]
    cut = int(0.7 * len(X))
    Xtr_raw, ytr_raw, Xte_raw, yte_raw = X[:cut], y[:cut], X[cut:], y[cut:]

    chance = 1.0 / args.n_envs
    print(f"{len(X)} trajectories x {args.steps} steps x {X.shape[2]} dims, "
          f"{args.n_envs} envs; split {cut}/{len(X) - cut} by trajectory")
    print(f"\n{'window':>7}  {'linear':>8}  {'mlp':>8}")
    print("-" * 27)
    results = []
    for k in args.windows:
        if k > args.steps:
            continue
        Xtr, ytr = windows(Xtr_raw, ytr_raw, k)
        Xte, yte = windows(Xte_raw, yte_raw, k)
        lin = fit(Xtr, ytr, Xte, yte, args.n_envs, hidden=0)
        mlp = fit(Xtr, ytr, Xte, yte, args.n_envs, hidden=args.hidden)
        results.append({"window": k, "linear": lin, "mlp": mlp})
        print(f"{k:>7}  {lin:>8.4f}  {mlp:>8.4f}")

    best = max(max(r["linear"], r["mlp"]) for r in results)
    print(f"\nbest over every window and readout: {best:.4f}  "
          f"(chance {chance:.4f})")
    verdict = (
        "The environment is essentially readable off the observation stream, so "
        "an inferred task id would cost little and the oracle advantage in "
        "Wave 3 is much smaller than the label suggests."
        if best > 0.95 else
        "Even the strongest readout tried cannot identify the environment from "
        "its observations, so the oracle task id handed to every Wave 3 arm is "
        "a real and large advantage. Those arms are upper bounds on their "
        "family, not peers of the boundary-free methods."
    )
    print(f"\n{verdict}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results, "best": best, "chance": chance,
                       "n_envs": args.n_envs, "n_trajectories": int(len(X)),
                       "steps": args.steps, "obs_dim": int(X.shape[2]),
                       "seed": args.seed, "verdict": verdict}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
