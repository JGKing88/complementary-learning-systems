"""Why a partial recall step lands on the manifold for one code and not the other.

With ``alpha < 1`` the state is a blend of the cue and the recall term, so it
sits on the CHORD from ``z(here)`` toward ``z(goal)``. ``alpha_walk_check.py``
found production walks that chord through real intermediate positions (cos to
its nearest cell never below 0.967) while arm B stalls and snaps, dipping to
cos 0.906 mid-transition. This is why, and the number is predictable.

For a BINARY code, blend ``(1-t)*b_here + t*b_goal``. On coordinates where the
two codes agree the blend keeps magnitude 1; on the fraction ``f = H/D`` where
they disagree it takes the graded value ``|1 - 2t|``. Two consequences:

  * the blend's SIGN pattern is that of whichever endpoint dominates, so its
    nearest corner flips discontinuously from ``b_here`` to ``b_goal`` at
    ``t = 1/2``. There is no intermediate sign pattern, hence no intermediate
    position to decode to;
  * at ``t = 1/2`` the disagreeing coordinates are exactly zero, and

        cos(blend, nearest corner) = sqrt(1 - H/D)

    which for the measured 10.46-cell separation and m = 18.4 flips per cell is
    sqrt(1 - 0.188) = 0.901, against the 0.906 dip observed.

A CONTINUOUS ballistic code has no such obstruction: its manifold is flat over
the operating range, so a chord is a near-geodesic and every point on it is
close to a real code.

Measured here directly -- walk the chord in ``t`` for both encoders and report,
at each ``t``, the decoded position and the cosine to that cell's code.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.hopfield_probe.attractor import retrieve
from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (ProbeConfig, build_cell_bank,
                                             build_memory, load_probe_encoder,
                                             local_cells, sample_worlds,
                                             scored_envs)

from cls_paths import results_dir

DEFAULT_DIR = os.path.join(
    str(results_dir()),
    "hopfield_probe/20260827/probe_ladder7")
TS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0)


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-30)


def headers(d: str, want: list[str]):
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    return [(e["label"], e["header"]) for e in man["encoders"]
            if any(e["label"].startswith(w) for w in want)]


def chord(header: dict, k: int, sep: float):
    """Walk the chord from z(here) to z(goal) and decode every point on it."""
    enc, ecfg, gain, fwhm, _h = load_probe_encoder(
        header["path"], fwhm_fallback=header.get("fwhm_ratio", 0.25))
    gain = float(header.get("gain", gain))
    enc.gain = gain
    beta = header.get("beta")
    cfg = ProbeConfig(n_worlds=1, n_envs_per_world=20, env_size=20, Npos=1716,
                      k_values=(k,), steps=(1,), seed=0, basin_radius=0,
                      beta_override=float(beta) if beta else None)
    field = Field(enc, list(ecfg.lambdas), fwhm, gain, cfg.Npos)

    w = sample_worlds(cfg)[0]
    rng = np.random.RandomState(w.seed * 31 + k)
    build_memory(field, w, k, cfg, rng)
    bank = build_cell_bank(field, w, k, cfg, rng)
    env = scored_envs(cfg, k)[0]
    off, goal = w.specs[env].offset, np.array(w.specs[env].goal)

    # Every cell of the env at ~`sep` cells from the goal, as "here".
    cells = local_cells(cfg.env_size)
    d = np.sqrt(((cells - goal) ** 2).sum(1))
    here = cells[np.abs(d - sep) < 0.75]
    z_here = unit(field.encode(here[:, 0] + off[0], here[:, 1] + off[1]))
    z_goal = unit(field.encode(np.full(here.shape[0], goal[0] + off[0]),
                               np.full(here.shape[0], goal[1] + off[1])))

    # Hamming fraction between the two, for the sqrt(1 - H/D) prediction.
    f = float(np.mean(np.sign(z_here) != np.sign(z_goal)))

    rows = []
    for t in TS:
        x = unit((1.0 - t) * z_here + t * z_goal)
        idx, val = retrieve(x, bank, cfg)
        r_env, r_x, r_y = bank.decode(idx)
        same = r_env == env
        dist = np.sqrt((r_x - goal[0]) ** 2 + (r_y - goal[1]) ** 2)
        rows.append((t, float(np.mean(dist[same])) if same.any() else np.nan,
                     float(val.mean()), float(same.mean())))
    return f, here.shape[0], rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--labels", nargs="+", default=["10% · ", "10% gain=1e6"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sep", type=float, default=10.0)
    args = ap.parse_args()

    seen: set[str] = set()
    for label, header in headers(args.dir, args.labels):
        arm = label.split(" · ")[0]
        if arm in seen:
            continue
        seen.add(arm)
        f, n, rows = chord(header, args.k, args.sep)
        print(f"\n=== {label} ===  {n} start cells ~{args.sep:g} from the goal")
        print(f"  sign disagreement H/D = {f:.4f}   "
              f"sqrt(1 - H/D) = {np.sqrt(1 - f):.4f}  <- predicted worst cos "
              f"(binary codes only)")
        print(f"  {'t':>6s}{'decoded dist':>14s}{'cos to that cell':>18s}"
              f"{'in env':>9s}")
        for t, dist, cos, ie in rows:
            print(f"  {t:>6.2f}{dist:>14.2f}{cos:>18.4f}{ie:>9.2f}")


if __name__ == "__main__":
    main()
