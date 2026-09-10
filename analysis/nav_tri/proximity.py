"""How much does the path go back over ground it already covered?

**`revisit_frac` cannot answer this**, and the reason is worth stating plainly
because it was used as if it could. It counts re-entry into a SNAPPED CELL, and
a step either enters a new cell or it does not -- so

    revisit_frac == 1 - cells_per_step

exactly. Verified to four decimals on all five P27-P30 arms. It is coverage
restated, not an independent measurement, and ranking arms by it reproduces the
coverage ranking by construction.

The continuous version is independent of coverage: the share of steps whose
position passes within ``r`` of a point the path held at least ``lag`` steps
earlier. It sees the failure the snapped measure is blind to -- a path that
precesses, running a half-cell alongside its own earlier track, never re-enters
a cell and scores zero while looking exactly like retracing.

``lag`` excludes the trivial neighbourhood: consecutive positions are always
within a step of each other and that is not going back over anything.

``r`` should be the goal radius, because that is the scale at which re-covering
ground actually wastes detection area -- which is what exploration is for
(§0.0). Smaller r asks a stricter question than the task cares about.

    python -m analysis.nav_tri.proximity --json traj.json --radius 1.0
"""
from __future__ import annotations

import argparse
import json

import numpy as np

LAG = 10


def proximity_revisit(path, radius: float, lag: int = LAG) -> float:
    """Share of steps passing within `radius` of ground held `lag`+ steps ago.

    Each step is compared against the whole prefix up to `t - lag`, so a return
    to ground covered at any earlier time counts, not just a recent one.
    """
    p = np.asarray(path, dtype=np.float64)
    T = len(p)
    if T <= lag:
        return float("nan")
    hits = 0
    for t in range(lag, T):
        past = p[: t - lag + 1]
        if np.min(np.linalg.norm(past - p[t], axis=1)) < radius:
            hits += 1
    return hits / float(T - lag)


def summarise(paths, radius: float, lag: int = LAG) -> dict:
    v = np.array([proximity_revisit(p, radius, lag) for p in paths],
                 dtype=np.float64)
    ok = np.isfinite(v)
    return {"n": int(ok.sum()),
            "mean": float(v[ok].mean()) if ok.any() else float("nan"),
            "sd": float(v[ok].std(ddof=1)) if ok.sum() > 1 else float("nan"),
            "radius": float(radius), "lag": int(lag)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True, help="an explore_traj dump")
    p.add_argument("--radius", type=float, required=True,
                   help="goal_radius; the scale at which re-covering ground "
                        "actually wastes detection area.")
    p.add_argument("--extra_radii", type=float, nargs="*",
                   default=[0.5, 2.0],
                   help="reported alongside, to show whether a ranking is an "
                        "artefact of one threshold.")
    p.add_argument("--lag", type=int, default=LAG)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    d = json.load(open(a.json))
    labels = a.labels or d["labels"]
    radii = [a.radius] + [r for r in a.extra_radii if r != a.radius]
    rows: dict = {}

    hdr = "  %-16s %12s %10s" % ("arm", "revisit_cell", "cells/step")
    hdr += "".join("  prox<%-5.1f" % r for r in radii)
    print(hdr)
    for lab in labels:
        tr = [t["by_ckpt"][lab] for t in d["trials"]]
        paths = [x["path"] for x in tr]
        rv = float(np.mean([x["revisit_frac"] for x in tr]))
        rows[lab] = {"revisit_cell": rv, "cells_per_step": 1.0 - rv}
        line = "  %-16s %12.3f %10.3f" % (lab, rv, 1.0 - rv)
        for r in radii:
            s = summarise(paths, r, a.lag)
            rows[lab][f"prox_{r}"] = s
            line += "  %9.3f" % s["mean"]
        print(line)
    print("\n  revisit_cell IS 1 - cells/step, exactly -- the two columns are "
          "one number.")
    print("  prox<r is the independent measure: share of steps within r of "
          "ground the")
    print("  path held at least %d steps earlier." % a.lag)

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()
