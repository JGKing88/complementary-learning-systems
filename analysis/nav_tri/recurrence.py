"""Recurrence curve: does the trajectory come back to where it was?

**The primary orbit diagnostic.** ``mean |p(t) - p(t+tau)|`` over t, as a
function of tau. A path that orbits with period T shows a clear MINIMUM at
tau = T; one that does not just wanders and shows no post-rise dip.

Why this and not the statistics it replaces
-------------------------------------------
Three earlier attempts each had a blind spot, and each cost a wrong call:

``signed_turn_mean``   an EPISODE MEAN, so a path that circles one way then the
                       other cancels to ~0. It hid the looping, then hid the
                       bidirectional circling, then hid a 6.9 deg/step curl
                       that reads as "straight". Three corrections (P2 §18.6).
``straightness``       an UNSIGNED cosine, so a steady curl and an unbiased
                       walk score the same. `p20_e_kcap` is locally STRAIGHTER
                       than the control while orbiting.
windowed |sum dtheta|  contaminated by wall bounces -- a billiard null scores
                       59% on it (§18.6).
revisit LAG            blind to a PRECESSING orbit. The agent returns to the
                       same REGION (within 4-5 cells) but a different snapped
                       cell, so cell-based revisit counts never see the period.
                       This produced a false negative that retracted a correct
                       claim; §27's "mechanism is open" was wrong because of it.

The recurrence curve has none of those: no null to compare against, no window
size to choose, no signed/unsigned ambiguity, and it works on the continuous
position so precession does not hide the period.

Measured on the P20 arms, 100 trajectories each, 200 steps (2026-09-02):

    tau              20     30     40     50     54     60     70
    p20_e_kcap    13.67  14.47  10.98   5.30   4.06   6.18  11.84
    p20_e         11.82  12.77  11.93  10.22     --    8.12   9.33

`p20_e_kcap` is 14.5 cells from where it was 30 steps ago and back within 4
cells 54 steps later -- **dip depth 10.66**, in 98 of 100 trajectories, median
lag 57, against a curl period of 2*pi/0.121 = 51.9. `p20_e` has no post-rise
dip in the mean curve.

  python -m analysis.nav_tri.recurrence --json traj.json
"""

import argparse
import json
import warnings

import numpy as np

MAX_TAU = 140
MIN_OVERLAP = 10          # need this many (t, t+tau) pairs to trust a tau
# A dip smaller than this is noise, not an orbit. 1.0 was far too lenient:
# it labelled 91/100 `p20_e` trajectories as orbiting while the AGGREGATE
# curve correctly showed none, because individual noisy minima sit at
# scattered lags and cancel in the mean. 3.0 cells is 15% of the arena.
DIP_CELLS = 3.0


def recurrence_curve(path, max_tau: int = MAX_TAU) -> np.ndarray:
    """(max_tau+1,) mean |p(t) - p(t+tau)|; NaN where there is no overlap."""
    p = np.asarray(path, dtype=float)
    T = len(p)
    out = np.full(max_tau + 1, np.nan)
    for tau in range(1, max_tau + 1):
        if T - tau < MIN_OVERLAP:
            break
        out[tau] = float(np.linalg.norm(p[tau:] - p[:-tau], axis=1).mean())
    return out


def orbit_stats(curve: np.ndarray, min_tau: int = 10) -> dict:
    """Period and depth of the first real dip after the initial rise.

    ``min_tau`` skips the rise out of the agent's own footprint, which every
    trajectory has and which is not an orbit.

    **The minimum must be INTERIOR, not global.** An orbit is a *post-rise*
    dip -- the curve climbs as the agent leaves, falls as it comes back, and
    climbs again as it leaves a second time. A global minimum over the window
    is a different thing, and taking one made the verdict turn on whether the
    initial rise happened to finish before ``min_tau``:

        p20_e, ten distractors, 144 trajectories, 200 steps
          deterministic   tau=10: 7.91   tau=60: 8.04   -> depth 0.00, "no orbit"
          sampled         tau=10: 7.84   tau=60: 7.62   -> depth 4.85, "ORBITS"

    Those two curves are the same curve to within 0.3 cells at every lag, and
    the old rule gave them opposite verdicts, because in the deterministic one
    the window edge sits 0.13 cells below the real dip and captured the argmin.
    Requiring the minimum to be a genuine interior trough removes that.
    """
    tail = curve[min_tau:]
    ok = np.isfinite(tail)
    if ok.sum() < 5:
        return {"period": float("nan"), "depth": 0.0, "orbits": False}
    idx = np.nonzero(ok)[0]
    vals = tail[idx]

    # Interior troughs: strictly below the running max before them AND
    # followed by a rise. `rise_after` is what makes it an orbit rather than
    # a curve that simply stops going up.
    best = None
    for j in range(1, len(idx) - 1):
        before = vals[:j + 1]
        after = vals[j:]
        peak_before = float(before.max())
        rise_after = float(after.max()) - float(vals[j])
        depth = peak_before - float(vals[j])
        if rise_after < DIP_CELLS / 2.0:
            continue                      # it never left again
        if best is None or depth > best[1]:
            best = (int(idx[j]), depth)

    if best is None:
        # No interior trough with a rise after it. Report the shallowest
        # possible reading rather than a global-minimum artefact.
        return {"period": float("nan"), "depth": 0.0, "orbits": False}

    k, depth = best
    return {"period": float(min_tau + k),
            "depth": float(depth),
            "orbits": bool(depth > DIP_CELLS)}


def summarise(paths, label: str = "") -> dict:
    curves = np.stack([recurrence_curve(p) for p in paths])
    with warnings.catch_warnings():
        # Long tau have no overlap for short paths; an all-NaN column is
        # expected and is reported as NaN rather than warned about.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_curve = np.nanmean(curves, axis=0)
    agg = orbit_stats(mean_curve)
    per = [orbit_stats(c) for c in curves]
    orb = [s for s in per if s["orbits"]]
    out = {
        "label": label,
        "n": len(paths),
        "mean_curve": mean_curve.tolist(),
        "aggregate": agg,
        "n_orbiting": len(orb),
        "median_period": (float(np.median([s["period"] for s in orb]))
                          if orb else float("nan")),
        "median_depth": (float(np.median([s["depth"] for s in orb]))
                         if orb else 0.0),
        # The discriminator between a real orbit and noise: a real one has the
        # SAME period across trajectories, so the per-trajectory dips reinforce
        # in the mean curve. Scattered periods cancel, which is why the
        # aggregate is the number to trust.
        "period_iqr": ([float(np.percentile([s["period"] for s in orb], 25)),
                        float(np.percentile([s["period"] for s in orb], 75))]
                       if orb else [float("nan")] * 2),
    }
    return out


def _report(s):
    print("\n=== %s ===  %d trajectories" % (s["label"], s["n"]))
    c = s["mean_curve"]
    print("  mean |p(t)-p(t+tau)|:")
    row = [t for t in (10, 20, 30, 40, 50, 60, 70, 80, 100, 120)
           if t < len(c) and np.isfinite(c[t])]
    print("    tau  " + "".join("%7d" % t for t in row))
    print("    dist " + "".join("%7.2f" % c[t] for t in row))
    a = s["aggregate"]
    print("  aggregate dip: depth %.2f at tau=%.0f  ->  %s"
          % (a["depth"], a["period"],
             "ORBITS" if a["orbits"] else "no orbit"))
    print("  per-trajectory: %d/%d orbit, median period %.0f (IQR %.0f-%.0f), "
          "median depth %.2f"
          % (s["n_orbiting"], s["n"], s["median_period"],
             s["period_iqr"][0], s["period_iqr"][1], s["median_depth"]))
    if not a["orbits"] and s["n_orbiting"] > 0.5 * s["n"]:
        print("  NOTE per-trajectory dips are at scattered lags -- they cancel "
              "in the mean, so this is NOT an orbit. Trust the aggregate.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True,
                   help="an explore_traj dump")
    p.add_argument("--labels", nargs="*", default=None,
                   help="which checkpoints to score (default: all in the file)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    d = json.load(open(a.json))
    labels = a.labels or d["labels"]
    results = []
    for lab in labels:
        paths = [t["by_ckpt"][lab]["path"] for t in d["trials"]]
        s = summarise(paths, lab)
        results.append(s)
        _report(s)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
