"""What can the agent actually know from its 60-ray sensory cone?

The two target behaviours need different information, and the cone does not
obviously supply either:

  billiard   needs "am I about to hit a wall" -- a DISTANCE.
  lawnmower  needs "where have I been" -- which, without a `prev_action`
             channel to path-integrate from, means absolute POSITION.

But a ray does not report a distance. `raycast_codes` (`world/env.py:141-166`)
returns the +/-1 code of the wall *segment* the ray lands on, and nothing else.
Distance is present only implicitly, through the spatial frequency of the code
pattern across the cone: standing close to a wall, adjacent rays hit adjacent
segments and the pattern varies slowly; standing far, adjacent rays hit distant
segments and it varies fast. Position is present only through *which* segments
of *which* walls are visible.

So both are in principle decodable and neither is explicit, and the question is
quantitative: how much is there, and does `wall_resolution` change it? A ridge
regression is a weak decoder, so a high score is a lower bound on what the
policy could learn; an MLP is closer to what the RNN can do. If neither can
recover position, **the lawmower line (coverage 0.478) is unreachable by
construction** and the billiard line (0.387) is the real ceiling -- which is a
finding about the observation, not about any hyper-parameter.

Needs no encoder, no scaffold, no GPU: this is a property of the sensor.

Usage:
    python -m analysis.nav_tri.sensory_decodability
    python -m analysis.nav_tri.sensory_decodability --resolutions 1 4 8 16
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from hopfield_nav.world.env import raycast_codes


def _sample(*, n, size, obs_size, resolution, rng):
    """(obs, targets) for n random (position, heading) pairs in one env."""
    wall_code = rng.choice([-1.0, 1.0], size=(4, size * resolution))
    x = rng.uniform(-0.5, size - 0.5, size=n)
    y = rng.uniform(-0.5, size - 0.5, size=n)
    psi = rng.uniform(-np.pi, np.pi, size=n)
    obs = raycast_codes(wall_code, size, x, y, psi, obs_size, resolution)

    hi = size - 0.5
    d_wall = np.minimum.reduce([x + 0.5, hi - x, y + 0.5, hi - y])
    return obs.astype(np.float64), {
        "pos_x": x, "pos_y": y,
        "dist_to_nearest_wall": d_wall,
        "heading_sin": np.sin(psi), "heading_cos": np.cos(psi),
    }


def _autocorr_features(obs):
    """Lag-k agreement along the cone, plus sign-change rate.

    The mechanism by which distance could be present at all. The wall code is
    random per segment, so `obs` is a hash of position and no smooth decoder
    can invert it -- but how fast the hash *varies across the cone* is pure
    geometry and is the same in every env: standing close to a wall, adjacent
    rays land on the same or neighbouring segments and agree; standing far,
    they land on distant segments and are independent. These features are
    codebook-independent by construction, which is what makes them usable in a
    held-out env, unlike anything read off the raw pattern.
    """
    f = [(obs[:, k:] * obs[:, :-k]).mean(1) for k in (1, 2, 3, 5, 8, 13, 21)]
    f.append((obs[:, 1:] != obs[:, :-1]).mean(1))     # sign-change rate
    f.append(obs.mean(1))
    return np.stack(f, axis=1)


def _score(X, y, seed, mlp):
    cut = int(0.7 * len(y))
    sc = StandardScaler().fit(X[:cut])
    Xtr, Xte = sc.transform(X[:cut]), sc.transform(X[cut:])
    out = {}
    out["ridge"] = float(r2_score(
        y[cut:], Ridge(alpha=1.0).fit(Xtr, y[:cut]).predict(Xte)))
    if mlp:
        m = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=400,
                         random_state=seed, early_stopping=True)
        out["mlp"] = float(r2_score(y[cut:], m.fit(Xtr, y[:cut]).predict(Xte)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--obs_size", type=int, default=60)
    p.add_argument("--resolutions", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--n", type=int, default=12000)
    p.add_argument("--envs", type=int, default=3,
                   help="distinct wall codes; scores are averaged over them")
    p.add_argument("--no-mlp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    print(f"{args.size}x{args.size}, {args.obs_size} rays, {args.n} samples, "
          f"{args.envs} wall codes, decoder = ridge"
          f"{'' if args.no_mlp else ' + MLP(256,128)'}")
    print("R^2 on held-out 30%. Each env has its OWN wall code, so a decoder "
          "is fit per env:\nthis is what a policy could learn about the env it "
          "is standing in, not across envs.\n")

    targets = ["dist_to_nearest_wall", "pos_x", "pos_y",
               "heading_sin", "heading_cos"]
    hdr = f"{'wall_res':>9s}" + "".join(f"{t:>22s}" for t in targets)
    print(hdr)

    out: dict = {"size": args.size, "obs_size": args.obs_size, "rows": []}
    for res in args.resolutions:
        acc = {t: [] for t in targets}
        for e in range(args.envs):
            rng = np.random.default_rng(args.seed + 1000 * res + e)
            X, ys = _sample(n=args.n, size=args.size, obs_size=args.obs_size,
                            resolution=res, rng=rng)
            A = _autocorr_features(X)
            for t in targets:
                s = _score(X, ys[t], args.seed, not args.no_mlp)
                s["autocorr"] = _score(A, ys[t], args.seed, False)["ridge"]
                acc[t].append(s)
        row = {"wall_resolution": res}
        cells = []
        for t in targets:
            r = float(np.mean([a["ridge"] for a in acc[t]]))
            a_ = float(np.mean([a["autocorr"] for a in acc[t]]))
            row[f"{t}_ridge"], row[f"{t}_autocorr"] = r, a_
            if not args.no_mlp:
                m = float(np.mean([a["mlp"] for a in acc[t]]))
                row[f"{t}_mlp"] = m
                cells.append(f"{r:>6.2f}/{m:>5.2f}/{a_:<8.2f}")
            else:
                cells.append(f"{r:>13.2f}/{a_:<7.2f}")
        out["rows"].append(row)
        print(f"{res:>9d}" + "".join(cells))

    print("\n(cells are ridge / MLP / ridge-on-autocorr-features)")
    print("The autocorr column is the codebook-INDEPENDENT one: it is the only\n"
          "route to a quantity an agent could use in a held-out env.")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
