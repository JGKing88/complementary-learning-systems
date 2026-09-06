"""The collapsed-tail report — the headline measurement for the explore half.

§5.3.2 established that a MEAN hides the failure this project is looking for:
on the phase-1 combined model the median swept barely moved from d=0 to d=10
(0.688 -> 0.668) and the p95 ROSE, while 14.6% of episodes collapsed outright.
So the reduction that gets reported is the distribution and the tail.

§5.3.4 added that the tail is two populations: collapse WITHOUT chasing is the
motor wall pin, collapse WITH chasing is the corner trap, and they want
opposite fixes. `chase_q` inside the tail is what separates them, so it is
printed for the tail and the body side by side.

**The threshold is RELATIVE TO THE BILLIARD AT THE EPISODE'S OWN SPEED**, and
that is not a detail. An absolute cut mislabels a slow policy: measured on the
wave-1 arms at u125, `d0_base` runs at realized speed 0.702, where a billiard
sweeps only 0.542 -- so an absolute 0.35 is 65% of *chance* and flagged 37.5%
of its episodes as collapsed, every one of them with `chase_q` exactly 0.000.
Those are ordinary undertrained episodes, not a pathology. The same error class
as reading `follow_q` without `align_true`, or `edge_frac` against zero when
uniform occupancy is 0.19: a statistic compared against a constant when its
baseline moves.

A collapsed episode is therefore one that swept far less than a random walker
would have at the speed it actually travelled.

    python -m analysis.nav_tri.tail_report <explore_traj json> [frac_of_chance]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from analysis.nav_tri.coverage_baselines import swept_billiard
from hopfield_nav.evaluation.swept import SweptArea

SIZE, R = 20, 1.0
# Half of what a billiard achieves at the same speed. Stated rather than swept
# because unlike EXPLOIT_DIAGNOSTIC §4's straddle rule this constant has no
# prediction riding on it -- but the raw per-episode values are printed, so any
# reader can re-cut it.
FRAC_OF_CHANCE = 0.5


def main(path, frac=FRAC_OF_CHANCE, abs_thr=None):
    d = json.load(open(path))
    size = int(d.get("size", SIZE))
    print(f"  --- collapsed-tail report  ({path.split('/')[-1]}, "
          f"n_dist={d.get('n_distractors')}, "
          + (f"threshold {abs_thr} ABSOLUTE" if abs_thr is not None
             else f"threshold = {frac:g} x billiard at each model's own speed")
          + ") ---")
    print("  %-22s %5s %6s %6s %6s %6s | %6s %6s %7s | %7s %7s"
          % ("label", "n", "mean", "p5", "p50", "sd",
             "speed", "thresh", "frac<t", "chase_t", "chase_r"))
    for lab in d["labels"]:
        sw, ch, sp = [], [], []
        for t in d["trials"]:
            st = t["by_ckpt"][lab]
            p = np.asarray(st["path"], float)
            T = len(p)
            sa = SweptArea(size, R, 1)
            for k in range(T):
                sa.add(p[k][None, :])
            sw.append(float(sa.result().per_trial.mean()))
            ch.append(st.get("chase_q", float("nan")))
            sp.append(float(np.linalg.norm(np.diff(p, axis=0),
                                           axis=-1).mean()))
        sw, ch, sp = map(np.asarray, (sw, ch, sp))
        speed = float(sp.mean())
        if abs_thr is not None:
            thr = float(abs_thr)
        else:
            thr = frac * swept_billiard(speed, size, T, R)
        bad = sw < thr
        q = np.percentile(sw, [5, 50])
        print("  %-22s %5d %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f %7.3f | "
              "%7s %7.3f"
              % (lab[-22:], len(sw), sw.mean(), q[0], q[1], sw.std(),
                 speed, thr, bad.mean(),
                 f"{ch[bad].mean():.3f}" if bad.any() else "--",
                 ch[~bad].mean() if (~bad).any() else float("nan")))
    print("  frac<t is the headline; the specialist's value is 0.000 at both")
    print("  distractor levels. chase_t vs chase_r separates the corner trap")
    print("  (chase elevated in the tail) from a plain wall pin (chase ~0).")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) > 1 and a[1].startswith("abs="):
        main(a[0], abs_thr=float(a[1][4:]))
    else:
        main(a[0], float(a[1]) if len(a) > 1 else FRAC_OF_CHANCE)
