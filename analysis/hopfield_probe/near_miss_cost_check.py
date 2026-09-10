"""Does a *near* retrieval cost the policy the goal, or does the walk fix it?

Sec 3.2 of THEORY_ENCODER_HOPFIELD.md split basin failures into `other_goal`
(cross-talk) and `near` (the memory returned a cell within 2 of the goal), and
argued the second is cheap because `q = basis @ (z_goal - z_here)` makes a
1-cell target error a ~2 degree bearing error. The objection is that arrival is
an exact-cell criterion -- ARRIVAL_RADIUS is 0.5 -- so a target one cell off is
a policy that stops one cell off, every time, and never scores.

Both can be true at different ranges, because `q` is recomputed at every cell:
a cue 25 cells out that retrieves 1 cell off is corrected as the walk carries
it inward, provided the cells NEAR the goal retrieve exactly. So the question
is not "is a near miss benign" but "at what range".

Measured here, on one memory per (world, K) so that the retrieval and the flow
refer to the same object:

  * per-cell retrieval outcome over the env (Test A's classification)
  * per-cell arrival under `continuous_flow` (Test D's criterion)
  * arrival rate cross-tabulated by outcome, and by outcome x start distance

`build_memory`'s rng defaults to Test D's convention (`w.seed*13 + k`), so the
reach column reproduces Sec 10.14. Test A draws a different memory (`*31`), and
cross-tabbing across the two would compare a retrieval against a flow through a
different bank -- so both halves here use the one memory `--rng_mul` selects.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.hopfield_probe.attractor import classify_outcomes, retrieve
from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.flow import continuous_flow
from analysis.hopfield_probe.harness import (OUTCOMES, ProbeConfig,
                                             build_cell_bank, build_memory,
                                             load_probe_encoder, local_cells,
                                             recall_trajectory,
                                             sample_worlds, scored_envs)
from analysis.hopfield_probe.qfield import cell_q_field

from cls_paths import results_dir

DEFAULT_DIR = os.path.join(
    str(results_dir()),
    "hopfield_probe/20260827/probe_ladder7")
BANDS = [(0, 4), (4, 8), (8, 12), (12, 18), (18, 30)]


def headers(d: str, want: list[str]) -> list[tuple[str, dict]]:
    """(label, header) for the manifest entries whose label starts with `want`."""
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    out = []
    for e in man["encoders"]:
        if any(e["label"].startswith(w) for w in want):
            out.append((e["label"], e["header"]))
    return out


def one_encoder(header: dict, cfg: ProbeConfig, k: int, rng_mul: int = 13):
    """(outcome, reached, start_dist) concatenated over every scored env."""
    enc, ecfg, gain, fwhm, _h = load_probe_encoder(
        header["path"], fwhm_fallback=header.get("fwhm_ratio", 0.25))
    gain = float(header.get("gain", gain))
    enc.gain = gain
    field = Field(enc, list(ecfg.lambdas), fwhm, gain, cfg.Npos)

    size = cfg.env_size
    cells = local_cells(size)
    n_cell = size * size
    O, R, D = [], [], []

    for w in sample_worlds(cfg):
        rng = np.random.RandomState(w.seed * rng_mul + k)
        mem = build_memory(field, w, k, cfg, rng)
        bank = build_cell_bank(field, w, k, cfg, rng)
        test_envs = scored_envs(cfg, k)

        cues = np.concatenate([
            field.encode(cells[:, 0] + w.specs[e].offset[0],
                         cells[:, 1] + w.specs[e].offset[1])
            for e in test_envs], axis=0)
        traj = recall_trajectory(mem, cues, cfg.steps, cfg)
        s = cfg.steps[0]
        idx, _ = retrieve(traj[s], bank, cfg)
        ret_env, ret_x, ret_y = bank.decode(idx)

        for j, e in enumerate(test_envs):
            sl = slice(j * n_cell, (j + 1) * n_cell)
            goal = w.specs[e].goal
            outcome, _rd = classify_outcomes(
                ret_env[sl], ret_x[sl], ret_y[sl], e, goal)
            qf, _c, _b = cell_q_field(field, w, e, mem, cfg)
            cnt = continuous_flow(qf[s], size, goal, cfg)
            O.append(outcome)
            R.append(cnt["reached"])
            D.append(cnt["start_dist"])
    return (np.concatenate(O), np.concatenate(R), np.concatenate(D))


def report(label: str, outcome, reached, dist) -> None:
    print(f"\n=== {label} ===   reach {reached.mean():.3f} over "
          f"{reached.size} starts")
    print(f"  {'retrieval':<14s}{'cells':>8s}{'share':>8s}{'arrival':>9s}")
    for i, name in enumerate(OUTCOMES):
        m = outcome == i
        if not m.any():
            continue
        print(f"  {name:<14s}{int(m.sum()):>8d}{m.mean():>8.3f}"
              f"{reached[m].mean():>9.3f}")

    print(f"\n  arrival by (retrieval, start distance)")
    print(f"  {'band':<10s}" + "".join(f"{n:>16s}" for n in OUTCOMES[:4]))
    for lo, hi in BANDS:
        row = f"  {f'{lo}-{hi}':<10s}"
        for i in range(4):
            m = (dist >= lo) & (dist < hi) & (outcome == i)
            row += (f"{reached[m].mean():>10.3f}{int(m.sum()):>6d}"
                    if m.any() else f"{'-':>16s}")
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--labels", nargs="+",
                    default=["10% · ", "1.25%", "0.75%"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rng_mul", type=int, default=13,
                    help="13 reproduces Test D's memory draw, 31 Test A's")
    args = ap.parse_args()

    cfg = ProbeConfig(n_worlds=8, n_envs_per_world=20, env_size=20, Npos=1716,
                      k_values=(args.k,), steps=(1,), seed=0, basin_radius=0)
    for label, header in headers(args.dir, args.labels):
        report(label, *one_encoder(header, cfg, args.k, args.rng_mul))


if __name__ == "__main__":
    main()
