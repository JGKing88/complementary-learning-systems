"""What field size would a recording report for a two-scale code?

    python -m analysis.field_kernel.recorded_fields

For each fast share: true rate maps of 1024 units in a square arena, then a simulated recording:
uniform occupancy of ``dwell`` seconds per cell, Poisson spike counts at a peak rate of ``peak_hz``
(each unit scaled so its own arena peak is that rate), rate map = counts / dwell smoothed with a
Gaussian of ``sd`` cells. Fields by definition A (connected patch at or above 20% of the unit's
peak, at least 9 cells) on the true map and on the recorded map. Units with a true peak under 20%
of the population's median peak are treated as silent. Also the lag-16 Pearson similarity of
population vectors, which should not depend on any of this.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from analysis.coding_comparison import measures as CM
from analysis.field_kernel.measure import build
from analysis.ideal_family import measure as FM

T = 1.2816


def drive(spec, gx, gy):
    e = build(spec)
    return e.drive(e._waves_np(gx, gy)) * math.sqrt(e.out_dim)


def fields(R3):
    F = CM.detect_fields_2d(R3, FM.DEF_A)
    c = F.counts
    act = c > 0
    d = 2 * np.sqrt(F.area / math.pi)
    return (float(d.mean()) if d.size else float("nan"), float(np.median(d)) if d.size else float("nan"),
            float(c[act].mean()) if act.any() else 0.0, float(act.mean()))


def main(side=40, peak_hz=10.0, sd=1.0):
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(side) + 700, np.arange(side) + 300, indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    ys = drive("ideal:r=23.2,n_freq=512,seed=0,relu_mix=1,active=0.1", gx, gy)
    yf = drive("ideal:r=3,n_freq=512,seed=1,relu_mix=1,active=0.1", gx, gy)
    y4 = drive("ideal:r=4.2,n_freq=512,seed=2,relu_mix=1,active=0.1", gx, gy)
    print(f"arena {side} cells, peak {peak_hz} Hz, smoothing SD {sd} cell; diameters in cells")
    print(f"{'code':<22s}{'dwell':>6s} | true: mean  med  n/unit active | recorded: mean  med  n/unit active")
    rows = [("one scale r 23.2", ys), ("5% fast", None), ("30% fast", None), ("50% fast", None),
            ("one scale r 4.2", y4)]
    for name, y in rows:
        if y is None:
            a = float(name.split("%")[0]) / 100
            y = math.sqrt(1 - a) * ys + math.sqrt(a) * yf
        R = np.maximum(y - T, 0).T.reshape(-1, side, side)
        pk = R.reshape(len(R), -1).max(1)
        R = R[pk > 0.2 * np.median(pk[pk > 0])]
        pk = R.reshape(len(R), -1).max(1)
        true = fields(R)
        for dwell in (1.0, 4.0):
            lam = R / pk[:, None, None] * peak_hz * dwell
            rec = ndimage.gaussian_filter(rng.poisson(lam) / dwell, sigma=(0, sd, sd), mode="nearest")
            r = fields(rec)
            print(f"{name:<22s}{dwell:6.0f} |      {true[0]:5.1f} {true[1]:4.1f}  {true[2]:5.2f}  {true[3]:5.2f} |"
                  f"          {r[0]:5.1f} {r[1]:4.1f}  {r[2]:5.2f}  {r[3]:5.2f}")


if __name__ == "__main__":
    main()
