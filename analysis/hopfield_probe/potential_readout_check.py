"""Stage 0 of escape (iii-c): read the direction off a scalar, not a projection.

Sec 7.1 of THEORY_ENCODER_HOPFIELD.md. Both readouts are finite differences of
the same field ``s(p) = <zhat, z(p)>``; the current one is a FORWARD difference
plus a constant ``(1 - C(1))``, and that constant is a correction calibrated for
a QUADRATIC similarity profile. Binarising the encoder makes the profile LINEAR,
the constant no longer cancels, and it survives as the same additive term in both
components -- so the readout returns (1 + cos t, 1 + sin t) instead of
(cos t, sin t), which is a translation of the 2-vector and destroys the bearing.
``readout_offset_check.py`` predicts acc45 0.423 for that, against arm B's
measured 0.392.

A CENTRAL difference never picks the constant up:

    q_i = [ <zhat, z(p + e_i)> - <zhat, z(p - e_i)> ] / 2

so the prediction is that arm B's acc45 goes 0.392 -> ~1.

Measured here on the real encoders, with the memory in the loop -- ``zhat`` is
the recalled vector, not the true goal code -- alongside the existing readout on
the identical memory, so the control has to reproduce the published numbers
(arm B 0.392, production 0.995) or the comparison means nothing.

Also censuses the failure mode the mechanism predicts is binding. A descent stops
at a LOCAL MAXIMUM of ``s``, so a cell is a sink when none of its four
neighbours has a larger ``s`` than it does. That is what a greedy walk would
actually stall on, it uses the true per-cell ``zhat``, and nothing in the campaign
has ever looked at it.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.flow import continuous_flow, discrete_flow
from analysis.hopfield_probe.harness import (ProbeConfig, build_memory,
                                             load_probe_encoder, local_cells,
                                             recall_trajectory, sample_worlds,
                                             scored_envs)
from analysis.hopfield_probe.qfield import project_q
from analysis.hopfield_probe.stats import wrap_to_pi

from cls_paths import results_dir

DEFAULT_DIR = os.path.join(
    str(results_dir()),
    "hopfield_probe/20260827/probe_ladder7")
# (East, North), matching project_q's output order.
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BANDS = ((0, 4), (4, 8), (8, 12), (12, 18), (18, 30))


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-30)


def headers(d: str, want: list[str]):
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    return [(e["label"], e["header"]) for e in man["encoders"]
            if any(e["label"].startswith(w) for w in want)]


def one_encoder(header: dict, cfg: ProbeConfig, k: int, rng_mul: int = 17,
                flows: bool = False):
    """Per-cell errors for both readouts, the sink census, and optionally
    Stage 1: the same `q` fields pushed through the unmodified flows.

    ``rng_mul`` selects which test's memory draw to reproduce -- 17 is Test B's,
    so the `acc45` control matches the published 0.392 / 0.995; 13 is Test D's,
    so the *reach* control matches the published 0.103 / 0.987. They are
    different memories, so one run cannot validate against both.
    """
    enc, ecfg, gain, fwhm, _h = load_probe_encoder(
        header["path"], fwhm_fallback=header.get("fwhm_ratio", 0.25))
    gain = float(header.get("gain", gain))
    enc.gain = gain
    field = Field(enc, list(ecfg.lambdas), fwhm, gain, cfg.Npos)

    size = cfg.env_size
    cells = local_cells(size)
    s = cfg.steps[0]
    err_now, err_new, dists, sinks, closer = [], [], [], [], []
    flow: dict[str, list] = {"cont_now": [], "cont_new": [], "disc_now": [],
                             "disc_new": [], "fsink_now": [], "fsink_new": [],
                             "cyc_now": [], "cyc_new": []}

    for w in sample_worlds(cfg):
        rng = np.random.RandomState(w.seed * rng_mul + k)
        mem = build_memory(field, w, k, cfg, rng)

        for e in scored_envs(cfg, k):
            off = w.specs[e].offset
            goal = w.specs[e].goal
            cues = _unit(field.encoded_state(cells, off))
            zhat = _unit(recall_trajectory(mem, cues, cfg.steps, cfg)[s])

            # s_n at the four neighbours, and s_0 at the cell itself.
            sn = {n: np.einsum("id,id->i", zhat,
                               _unit(field.encoded_state(cells + np.array(n),
                                                         off)))
                  for n in NEIGHBOURS}
            s0 = np.einsum("id,id->i", zhat, cues)

            q_new = 0.5 * np.stack(
                [sn[(1, 0)] - sn[(-1, 0)], sn[(0, 1)] - sn[(0, -1)]], axis=1)
            q_now = project_q(field.local_basis(cells, off), cues, zhat)

            dx = goal[0] - cells[:, 0]
            dy = goal[1] - cells[:, 1]
            d = np.hypot(dx, dy)
            keep = d > 0                       # bearing undefined at the goal
            theta = np.arctan2(dy, dx)

            for q, acc in ((q_now, err_now), (q_new, err_new)):
                acc.append(np.abs(wrap_to_pi(
                    np.arctan2(q[keep, 1], q[keep, 0]) - theta[keep])))
            dists.append(d[keep])

            # A greedy descent stalls where no neighbour is better.
            best = np.stack([sn[n] for n in NEIGHBOURS], axis=1)
            sinks.append((best.max(1) <= s0)[keep])
            # ...and when it does move, does it move closer in real space?
            step = np.array(NEIGHBOURS)[best.argmax(1)]
            nd = np.hypot(dx - step[:, 0], dy - step[:, 1])
            closer.append((nd < d)[keep])

            if flows:
                for tag, q in (("now", q_now), ("new", q_new)):
                    c = continuous_flow(q, size, goal, cfg)
                    dsc = discrete_flow(q, size, goal)
                    flow[f"cont_{tag}"].append(c["reach_rate"])
                    flow[f"disc_{tag}"].append(dsc["reach_rate"])
                    flow[f"fsink_{tag}"].append(len(dsc["sinks"]))
                    flow[f"cyc_{tag}"].append(len(dsc["limit_cycles"]))

    return (np.concatenate(err_now), np.concatenate(err_new),
            np.concatenate(dists), np.concatenate(sinks),
            np.concatenate(closer),
            {kk: np.asarray(v, dtype=float) for kk, v in flow.items()})


def report(label: str, err_now, err_new, d, sinks, closer, flow, expect,
           expect_reach) -> None:
    q45 = np.pi / 4
    print(f"\n=== {label} ===  {err_now.size} cells")
    print(f"  {'readout':<12s}{'acc45':>9s}{'|err| deg':>11s}   published")
    print(f"  {'current':<12s}{np.mean(err_now < q45):>9.3f}"
          f"{np.degrees(err_now.mean()):>11.1f}   {expect}")
    print(f"  {'(iii-c)':<12s}{np.mean(err_new < q45):>9.3f}"
          f"{np.degrees(err_new.mean()):>11.1f}   <- prediction ~1.0")

    print(f"\n  {'band':<10s}{'acc45 now':>11s}{'acc45 new':>11s}"
          f"{'sinks':>9s}{'step closer':>13s}{'n':>8s}")
    for lo, hi in BANDS:
        m = (d >= lo) & (d < hi)
        if not m.any():
            continue
        print(f"  {f'{lo}-{hi}':<10s}{np.mean(err_now[m] < q45):>11.3f}"
              f"{np.mean(err_new[m] < q45):>11.3f}"
              f"{np.mean(sinks[m]):>9.3f}{np.mean(closer[m]):>13.3f}"
              f"{int(m.sum()):>8d}")
    print(f"  {'ALL':<10s}{np.mean(err_now < q45):>11.3f}"
          f"{np.mean(err_new < q45):>11.3f}{np.mean(sinks):>9.3f}"
          f"{np.mean(closer):>13.3f}{err_now.size:>8d}")

    if flow["cont_now"].size:
        n_env = flow["cont_now"].size
        print(f"\n  STAGE 1 -- the same q fields through the unmodified flows,"
              f" {n_env} envs")
        print(f"  {'readout':<12s}{'reach cont':>12s}{'reach disc':>12s}"
              f"{'sinks/env':>11s}{'cycles/env':>12s}   published")
        for tag, name, note in (("now", "current", expect_reach),
                                ("new", "(iii-c)", "")):
            print(f"  {name:<12s}{flow[f'cont_{tag}'].mean():>12.3f}"
                  f"{flow[f'disc_{tag}'].mean():>12.3f}"
                  f"{flow[f'fsink_{tag}'].mean():>11.2f}"
                  f"{flow[f'cyc_{tag}'].mean():>12.2f}   {note}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--labels", nargs="+",
                    default=["10% gain=1e6", "10% · "])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=2,
                    help="how many training seeds per arm")
    ap.add_argument("--rng_mul", type=int, default=17,
                    help="17 = Test B's memory draw (acc45 control), "
                         "13 = Test D's (reach control)")
    ap.add_argument("--flows", action="store_true",
                    help="Stage 1: push the same q fields through the flows")
    args = ap.parse_args()

    seen: dict[str, int] = {}
    for label, header in headers(args.dir, args.labels):
        arm = label.split(" · ")[0]
        seen[arm] = seen.get(arm, 0) + 1
        if seen[arm] > args.seeds:
            continue
        beta = header.get("beta")
        cfg = ProbeConfig(n_worlds=8, n_envs_per_world=20, env_size=20,
                          Npos=1716, k_values=(args.k,), steps=(1,), seed=0,
                          basin_radius=0,
                          beta_override=float(beta) if beta else None)
        armb = "1e6," in arm
        report(label,
               *one_encoder(header, cfg, args.k, args.rng_mul, args.flows),
               expect="0.392 (arm B)" if armb else "0.995 (production)",
               expect_reach="0.103 (arm B)" if armb else "0.987 (production)")


if __name__ == "__main__":
    main()
