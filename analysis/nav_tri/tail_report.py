"""The collapsed-tail report — the measurement §5.3.2/§5.3.4 say is the headline.

A mean over the explore half hides the failure: on the phase-1 combined model
the MEDIAN swept barely moved from d=0 to d=10 (0.688 -> 0.668) while 14.6% of
episodes collapsed outright. So report the distribution and the tail, and split
the tail by chase_q, because two different failures both read as "a collapsed
episode":

    collapse WITHOUT chasing = the motor wall pin (§18.6 mode 1)
    collapse WITH chasing    = the corner trap (D2)

They want opposite fixes. The specialist's target is frac = 0.000.

    python -m analysis.nav_tri.tail_report <explore_traj json> [threshold]
"""
import json
import sys

import numpy as np

from hopfield_nav.evaluation.swept import SweptArea

SIZE, R = 20, 1.0


def main(path, thr=0.35):
    d = json.load(open(path))
    size = int(d.get("size", SIZE))
    print(f"  --- collapsed-tail report  ({path.split('/')[-1]}, "
          f"n_dist={d.get('n_distractors')}, threshold {thr}) ---")
    print("  %-22s %5s %6s %6s %6s %6s | %7s %7s %7s"
          % ("label", "n", "mean", "p5", "p50", "sd",
             "frac<t", "chase_t", "chase_r"))
    for lab in d["labels"]:
        sw, ch, ed = [], [], []
        for t in d["trials"]:
            st = t["by_ckpt"][lab]
            p = np.asarray(st["path"], float)
            sa = SweptArea(size, R, 1)
            for k in range(len(p)):
                sa.add(p[k][None, :])
            sw.append(float(sa.result().per_trial.mean()))
            ch.append(st.get("chase_q", float("nan")))
            ed.append(st.get("edge_frac", float("nan")))
        sw, ch, ed = map(np.asarray, (sw, ch, ed))
        bad = sw < thr
        q = np.percentile(sw, [5, 50])
        print("  %-22s %5d %6.3f %6.3f %6.3f %6.3f | %7.3f %7s %7.3f"
              % (lab[-22:], len(sw), sw.mean(), q[0], q[1], sw.std(),
                 bad.mean(),
                 f"{ch[bad].mean():.3f}" if bad.any() else "--",
                 ch[~bad].mean() if (~bad).any() else float("nan")))
    print("  frac<t is the headline: the specialist's value is 0.000 at both")
    print("  distractor levels. chase_t vs chase_r separates the corner trap")
    print("  (chase elevated in the tail) from a plain wall pin (chase ~0).")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.35)
