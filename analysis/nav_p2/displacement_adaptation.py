"""How much does a little experience in the NEW env buy?

The second half of Jack's question. `displacement_decodability.py` asks what a
decoder trained on many envs transfers to an unseen one with zero experience
there; this asks what `k` steps inside the new env add, for
`k in {0, 4, 16, 64, 256}` -- 256 being an entire explore episode plus a
quarter.

The spec asked for two adaptation regimes:

  * **supervised anchors** -- `k` labelled `(view, position)` pairs. The agent
    has no such labels, so this was meant as an *upper bound* on what
    adaptation could ever buy, not a proposal.
  * **self-motion self-supervision** -- a `k`-step trajectory of
    `(view_t, displacement_t)` with the absolute position unknown. The
    displacements are exact (`prev_displacement`, `rollout/collector.py:659`,
    is read straight off the env), so every pair of timesteps in the
    trajectory carries a known relative label and `k` steps give `O(k^2)`
    training pairs for free. **This is the realistic one** -- it is what an
    explore rollout already provides.

For decoding a *displacement* the two collapse into one, which is worth
saying rather than hiding: an anchor table keyed by absolute position and the
same table keyed by position-relative-to-the-start differ by a constant, and a
difference of two matched entries cancels it. The unrealistic upper bound is
reachable with no labels at all. So the arms are:

  * `NN table (label-free)` -- match each view to the most similar stored view
    and inherit its position. This is the decoder that fits the structure:
    within an env the map cell -> cone is a random hash, which is injective
    but not smooth, so a table beats any regression. It is also exactly what
    an associative memory over stored views does, which is why it is the
    relevant bound for this project rather than a curiosity.
  * `selfsup ridge` -- the same codebook-independent features the transfer
    study uses, refit on the trajectory's own pairs inside this env.
  * `cross-env + selfsup` -- the many-env solution as a prior mean, corrected
    here.

And one measurement that explains all of them, run first: **how does cone
similarity decay with grid distance?** A table only localizes if similarity
carries locality. If two cells one step apart already have near-orthogonal
cones, then `k` anchors localize the agent only when it is standing on an
anchor, and "experience in a new env" means "cells you have literally
visited", not "an area you have explored". That is a different claim about
what memory buys, and it is measurable in three lines.

And one measurement that runs before any of it, because it decides whether
the sensory arms are even the right question: **integrate `prev_displacement`
and check it against the truth.** If self-motion is already exact in the
observation then `dpos` needs no decoding at all, and P2's premise -- that
sensory decodability is what gates the lawnmower ceiling -- does not hold
under phase 2's own input set.

Everything is run in the `fixed` framing (both views facing North) *and* in
the `free` framing (views at the pose actually occupied, table keyed by pose).
Fixed is the favourable case and free is the real one.

    python -m analysis.nav_p2.displacement_adaptation --test_envs 24
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from hopfield_nav.world.env import raycast_codes
from analysis.nav_p2.displacement_decodability import (
    build_env, fit_ridge, framing_views, metrics, predict,
    raycast_range, simulate, spread,
)

ALPHAS = np.logspace(-1, 6, 8)


def _obs(wall_code, size, resolution, sensor, obs_size, cells, psi):
    if sensor == "dist":
        return raycast_range(size, cells[:, 0], cells[:, 1], psi, obs_size)
    return raycast_codes(wall_code, size, cells[:, 0], cells[:, 1], psi,
                         obs_size, resolution)


# ---------------------------------------------------------------------------
# Does cone similarity carry locality at all?
# ---------------------------------------------------------------------------


def locality_profile(*, size, obs_size, resolutions, sensors, envs, seed):
    """cos(cone(ref), cone(c)) binned by grid distance, at psi=0.

    The precondition for every table-based decoder below. Reported next to the
    same quantity for the *encoder* (EXPERIMENTS_NAV_P2 5.8: xi.xi stays 0.99
    corner to corner) -- the two are different objects and the contrast is the
    point.
    """
    gx, gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    D = np.linalg.norm(cells[:, None, :] - cells[None, :, :], axis=-1)
    # The leading bin is the same cell: it must read exactly 1.000, which is
    # the guard that the similarity is being computed on what it claims to be.
    edges = [-0.5, 0.5, 1.01, 2.01, 3.01, 5.01, 8.01, 13.01, 30.0]
    rows = []
    for sensor in sensors:
        for res in resolutions:
            acc = {i: [] for i in range(len(edges) - 1)}
            far = []
            for e in range(envs):
                rng = np.random.default_rng(seed + 977 * res + e)
                wc = rng.choice([-1.0, 1.0], size=(4, size * res))
                V = np.asarray(_obs(wc, size, res, sensor, obs_size, cells,
                                    np.zeros(len(cells))), dtype=np.float64)
                V = V - V.mean() if sensor == "dist" else V
                Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
                C = Vn @ Vn.T
                for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                    m = (D > lo) & (D <= hi)
                    acc[i].append(float(np.median(C[m])))
                # What the true match has to beat: the best of ~400 candidates
                # is the p99.75 of the far-pair similarity, not its median.
                far.append(float(np.percentile(C[D > 8.0], 99.75)))
            rows.append({"sensor": sensor, "resolution": res,
                         "bins": ([f"{lo:g}-{hi:g}" for lo, hi
                                   in zip(edges[:-1], edges[1:])]
                                  + ["p99.75 far"]),
                         "median_cos": ([float(np.median(acc[i]))
                                         for i in range(len(edges) - 1)]
                                        + [float(np.median(far))])})
    return rows


# ---------------------------------------------------------------------------
# The motor route to the same quantity
# ---------------------------------------------------------------------------


def path_integration_check(*, size, steps, episodes, min_norm, max_norm, seed):
    """Integrate the realized displacement and compare to the truth.

    The sensory arms above ask whether `dpos` can be *decoded*. Phase 2 also
    hands it over directly: `input_prev_displacement=1` feeds
    `ContinuousVecEnv._last_displacement`, the realized post-clip move, read
    off the env (`rollout/collector.py:659`). If that is exact then the
    information a lawnmower needs is already in the observation and no
    decoding is required -- so the claim has to be checked, not assumed.

    Two things are checked, because position-relative-to-start is not the same
    as position-in-the-arena:

      * integration error -- ``|cumsum(realized) - (pos_t - pos_0)|``.
      * when the arena frame gets pinned. The agent starts without its
        absolute position, but a step the arena clips is a wall contact, and
        it is *visible*: the committed action and the realized displacement
        differ exactly then, and both are fed. Two non-parallel walls fix the
        origin; all four fix the origin and the extent.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, size - 1.0, size=(episodes, 2))
    start = pos.copy()
    integ = np.zeros_like(pos)
    walls = np.zeros((episodes, 4), dtype=bool)      # x-lo, x-hi, y-lo, y-hi
    first2 = np.full(episodes, -1)
    first4 = np.full(episodes, -1)
    n_clip = np.zeros(episodes)
    err = 0.0
    for t in range(steps):
        th = rng.uniform(-np.pi, np.pi, size=episodes)
        nrm = rng.uniform(min_norm, max_norm, size=episodes)
        act = nrm[:, None] * np.stack([np.sin(th), np.cos(th)], axis=1)
        new = np.clip(pos + act, 0.0, float(size - 1))
        moved = new - pos
        clipped = np.abs(moved - act) > 1e-12
        n_clip += clipped.any(axis=1)
        walls[:, 0] |= clipped[:, 0] & (new[:, 0] <= 1e-12)
        walls[:, 1] |= clipped[:, 0] & (new[:, 0] >= size - 1 - 1e-12)
        walls[:, 2] |= clipped[:, 1] & (new[:, 1] <= 1e-12)
        walls[:, 3] |= clipped[:, 1] & (new[:, 1] >= size - 1 - 1e-12)
        pinned2 = walls[:, :2].any(1) & walls[:, 2:].any(1)
        first2[(first2 < 0) & pinned2] = t + 1
        first4[(first4 < 0) & walls.all(1)] = t + 1
        pos = new
        integ += moved
        err = max(err, float(np.abs(integ - (pos - start)).max()))
    got2, got4 = first2 >= 0, first4 >= 0
    return {
        "max_integration_error": err,
        "steps": steps, "episodes": episodes,
        "clip_rate": float(np.median(n_clip / steps)),
        "frac_pinned_origin": float(got2.mean()),
        "frac_pinned_extent": float(got4.mean()),
        "steps_to_origin": [float(np.percentile(first2[got2], q))
                            for q in (10, 50, 90)] if got2.any() else None,
        "steps_to_extent": [float(np.percentile(first4[got4], q))
                            for q in (10, 50, 90)] if got4.any() else None,
    }


# ---------------------------------------------------------------------------
# The two adaptation regimes
# ---------------------------------------------------------------------------


def experience(rng, *, size, steps, min_norm, max_norm):
    """One trajectory in the new env: the `k` steps of experience."""
    P, S = simulate(rng, size=size, n_walks=1, steps=steps,
                    min_norm=min_norm, max_norm=max_norm)
    return P[:, 0, :], S[:, 0]


def nn_table_decode(query_views, table_views, table_pos):
    """Nearest stored view wins; inherit its position."""
    q = np.asarray(query_views, dtype=np.float64)
    t = np.asarray(table_views, dtype=np.float64)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    tn = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    return table_pos[(qn @ tn.T).argmax(axis=1)]


def _with_residual(sub, framing, W_cross):
    """A copy of `sub` whose target is the cross-env decoder's residual."""
    pr, y, _ = predict(sub, framing, "xcorr", W_cross)
    new = dict(sub)
    new["_target"] = y - pr
    return new


def run_env(d, *, k_list, framing, sensor, size, obs_size, resolution,
            min_norm, max_norm, W_cross, seed):
    """All adaptation arms for one held-out env, evaluated on its pair set.

    `d` is the *evaluation* set: pairs drawn anywhere in the arena,
    independent of the experience trajectory. That is the honest test -- can
    the agent decode displacement between two observations it has not seen,
    after k steps somewhere else in this env.
    """
    rng = np.random.default_rng(seed)
    traj_p, traj_psi = experience(rng, size=size, steps=max(k_list),
                                  min_norm=min_norm, max_norm=max_norm)
    traj_cell = np.clip(np.round(traj_p), 0, size - 1)
    wc = d["wall_code"]
    s1, s2, _, y, _ = framing_views(d, framing)

    out = {}
    for k in k_list:
        row = {}
        if k == 0:
            # zero experience: the many-env decoder, unchanged
            pr, tv, _ = predict(d, framing, "xcorr", W_cross)
            row["cross-env only"] = metrics(pr, tv)
            out[k] = row
            continue

        nT = k + 1
        tp = traj_psi[:nT] if framing != "fixed" else np.zeros(nT)
        views = _obs(wc, size, resolution, sensor, obs_size,
                     traj_cell[:nT], tp)

        # ---- NN table over the k visited poses ----------------------------
        # Listed once, because the supervised-anchor and self-supervised
        # versions are the SAME computation. Anchors key the table by absolute
        # position; self-supervision keys it by position relative to where the
        # trajectory started, which the agent knows exactly by integrating
        # `prev_displacement`. Decoding a *displacement* takes a difference of
        # two matched entries, and the unknown origin cancels. So the
        # "unrealistic upper bound" and the "realistic" regime coincide here,
        # and no position labels are needed at all.
        row["NN table (label-free)"] = metrics(
            nn_table_decode(s2, views, traj_p[:nT])
            - nn_table_decode(s1, views, traj_p[:nT]), y)

        # A regression fit on the trajectory's own pairs: every pair of
        # timesteps carries a known relative displacement, so k steps give
        # O(k^2) labelled training pairs for free.
        ii, jj = np.meshgrid(np.arange(nT), np.arange(nT), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
        # Validate on a spatially DISJOINT slice of the trajectory -- pairs
        # whose endpoints both fall in its last quarter. A random split would
        # validate in the same few cells the fit saw and would always prefer
        # the least-regularized solution; this is the only honest alpha
        # selection available without labels outside the trajectory, and the
        # fact that it is still not enough is itself part of the answer.
        tsplit = int(0.75 * nT)
        is_fit = (ii < tsplit) & (jj < tsplit)
        is_val = (ii >= tsplit) & (jj >= tsplit)
        ii, jj = ii[is_fit | is_val], jj[is_fit | is_val]
        is_val = (ii >= tsplit) & (jj >= tsplit)
        if len(ii) > 20000:
            sel = rng.permutation(len(ii))[:20000]
            ii, jj, is_val = ii[sel], jj[sel], is_val[sel]
        order = np.argsort(~is_val)          # validation pairs first
        ii, jj = ii[order], jj[order]
        cut_val = int(is_val.sum())
        if len(ii) >= 8 and cut_val >= 4 and len(ii) - cut_val >= 4:
            sub = {
                "s1_fix": views[ii], "s2_fix": views[jj],
                "s1_free": views[ii], "s2_free": views[jj],
                "psi1": tp[ii], "psi2": tp[jj],
                "p1": traj_p[:nT][ii], "p2": traj_p[:nT][jj],
                "c1": traj_cell[:nT][ii], "c2": traj_cell[:nT][jj],
                "lag": np.abs(jj - ii), "wall_code": wc,
            }
            cut = cut_val

            def part(sl, sub=sub, n=len(ii)):
                return {kk: (v[sl] if isinstance(v, np.ndarray) and len(v) == n
                             else v) for kk, v in sub.items()}
            # fit_ridge validates on the FIRST env in the list
            Wi, _, _ = fit_ridge([part(slice(cut)), part(slice(cut, None))],
                                 framing, "xcorr", ALPHAS, val_frac=0.5)
            pr, tvv, _ = predict(d, framing, "xcorr", Wi)
            row["selfsup ridge"] = metrics(pr, tvv)

            # the many-env solution as a prior mean, corrected here
            Wr, _, _ = fit_ridge(
                [_with_residual(part(slice(cut)), framing, W_cross),
                 _with_residual(part(slice(cut, None)), framing, W_cross)],
                framing, "xcorr", ALPHAS, val_frac=0.5)
            prb, tvb, _ = predict(d, framing, "xcorr", W_cross + Wr)
            row["cross-env + selfsup"] = metrics(prb, tvb)
        out[k] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--obs_size", type=int, default=60)
    p.add_argument("--resolution", type=int, default=4)
    p.add_argument("--min_norm", type=float, default=0.5)
    p.add_argument("--max_norm", type=float, default=2.0)
    p.add_argument("--train_envs", type=int, default=48)
    p.add_argument("--test_envs", type=int, default=24)
    p.add_argument("--test_pairs", type=int, default=1500)
    p.add_argument("--lags", type=int, nargs="+", default=[1, 4])
    p.add_argument("--k", type=int, nargs="+", default=[0, 4, 16, 64, 256])
    p.add_argument("--framings", nargs="+", default=["fixed", "free"])
    p.add_argument("--sensors", nargs="+", default=["code"])
    p.add_argument("--locality_res", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    results = {"args": vars(args)}

    print("=== the motor route: is dpos already exact in the observation? ===")
    pi = path_integration_check(size=args.size, steps=200, episodes=4000,
                                min_norm=args.min_norm,
                                max_norm=args.max_norm, seed=args.seed)
    results["path_integration"] = pi
    print(f"  max |integrated - true| over 200 steps x 4000 episodes: "
          f"{pi['max_integration_error']:.3e}")
    print(f"  wall-clip rate (random actions): {pi['clip_rate']:.1%} of steps")
    print(f"  origin pinned (2 non-parallel walls touched) in "
          f"{pi['frac_pinned_origin']:.1%} of episodes; steps to it "
          f"[p10,p50,p90] = {pi['steps_to_origin']}")
    print(f"  extent pinned (all 4 walls touched) in "
          f"{pi['frac_pinned_extent']:.1%} of episodes; steps to it "
          f"[p10,p50,p90] = {pi['steps_to_extent']}\n")

    print("=== cone similarity vs grid distance (psi=0, median over cell "
          "pairs and envs) ===")
    print("The precondition for any table/memory decoder: does similarity "
          "carry locality?\n")
    loc = locality_profile(size=args.size, obs_size=args.obs_size,
                           resolutions=args.locality_res,
                           sensors=args.sensors, envs=6, seed=args.seed)
    results["locality"] = loc
    print(f"  {'sensor':>7s} {'wall_res':>9s}" +
          "".join(f"{b:>10s}" for b in loc[0]["bins"]))
    for r in loc:
        print(f"  {r['sensor']:>7s} {r['resolution']:>9d}" +
              "".join(f"{v:>10.3f}" for v in r["median_cos"]))

    for sensor in args.sensors:
        tr = [build_env(args.seed * 100000 + i, size=args.size,
                        obs_size=args.obs_size, resolution=args.resolution,
                        sensor=sensor, lags=args.lags, pairs_per_lag=1200,
                        min_norm=args.min_norm, max_norm=args.max_norm)
              for i in range(args.train_envs)]
        te = [build_env(args.seed * 100000 + 70000 + i, size=args.size,
                        obs_size=args.obs_size, resolution=args.resolution,
                        sensor=sensor, lags=args.lags,
                        pairs_per_lag=args.test_pairs,
                        min_norm=args.min_norm, max_norm=args.max_norm)
              for i in range(args.test_envs)]

        for framing in args.framings:
            t0 = time.time()
            W_cross, _, _ = fit_ridge(tr, framing, "xcorr", ALPHAS,
                                      seed=args.seed)
            per_env = []
            for i, d in enumerate(te):
                per_env.append(run_env(
                    d, k_list=args.k, framing=framing, sensor=sensor,
                    size=args.size, obs_size=args.obs_size,
                    resolution=args.resolution, min_norm=args.min_norm,
                    max_norm=args.max_norm, W_cross=W_cross,
                    seed=args.seed * 7919 + i))
            print(f"\n=== sensor={sensor} framing={framing} "
                  f"({time.time() - t0:.0f}s) ===")
            names = []
            for k in args.k:
                for n in per_env[0][k]:
                    if n not in names:
                        names.append(n)
            print(f"  {'decoder':<22s}" +
                  "".join(f"{f'k={k}':>18s}" for k in args.k))
            rows = {}
            for n in names:
                cells = []
                for k in args.k:
                    got = [e[k][n] for e in per_env if n in e[k]]
                    if not got:
                        cells.append(f"{'-':>18s}")
                        continue
                    r2 = spread(got, "r2")[0]
                    am = spread(got, "ang_med")[0]
                    cells.append(f"{r2:>10.3f} /{am:>6.1f}")
                    rows.setdefault(n, {})[k] = {
                        "r2_med": r2, "ang_med": am,
                        "r2_p10": spread(got, "r2")[1],
                        "r2_p90": spread(got, "r2")[2],
                        "frac_lt45": spread(got, "frac_lt45")[0]}
                print(f"  {n:<22s}" + "".join(cells))
            results.setdefault("adaptation", []).append(
                {"sensor": sensor, "framing": framing, "rows": rows})
            print("  (cells are R2 / median angular error in degrees)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
