"""Which module phases does a corner-trained encoder still read outside its
corner? The 1-D similarity profile along each axis says.

A displacement along x changes only the three x-phases (periods 11, 12, 13);
along y only the y-phases. So the profile ``cos(z(p), z(p + (d, 0)))`` for
``d = 1..D`` is a function of the x-phase triple alone, and its peaks say which
modules the code is reading on that axis: a code that reads all three revives
only at 1716; one that has lost module 13 revives at the 11&12 period 132 (and
its multiples, 792 and 924 among them); one that reads nothing on that axis is
flat at ~1.

For each encoder this takes one reference inside the corner and one outside
with an unseen x-triple (x >= 500) and a seen y-triple (y < 500), and reports
the top peaks of each axis profile with the displacement's residues mod
11, 12, 13, plus the mean and the fraction of the profile above 0.9.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder
from analysis.hopfield_probe.corner_scan import default_encoders, unit

NPOS = 1716
LAMBDAS = (11, 12, 13)


def profile(field: Field, ref: tuple[int, int], axis: int, dmax: int):
    gx, gy = ref
    z0 = unit(field.encode(np.array([gx]), np.array([gy])))[0]
    d = np.arange(1, dmax + 1)
    if axis == 0:
        zz = unit(field.encode(gx + d, np.full(d.size, gy)))
    else:
        zz = unit(field.encode(np.full(d.size, gx), gy + d))
    return d, zz @ z0


def peaks(d, c, n=6, exclude=30):
    """Top-n local maxima beyond ``exclude`` cells, separated by >= 8 cells."""
    out = []
    order = np.argsort(-c)
    for i in order:
        if d[i] <= exclude or c[i] < 0.3:
            continue
        if all(abs(d[i] - d[j]) >= 8 for j in out):
            out.append(i)
        if len(out) == n:
            break
    return sorted(out, key=lambda i: -c[i])


def report(label, field, ref, dmax):
    print(f"\n  ref {ref}  ({'inside' if max(ref) < 500 else 'outside'}; "
          f"x-triple {'seen' if ref[0] < 500 else 'UNSEEN'}, "
          f"y-triple {'seen' if ref[1] < 500 else 'UNSEEN'})")
    for axis, name in ((0, "along x"), (1, "along y")):
        d, c = profile(field, ref, axis, dmax)
        pk = peaks(d, c)
        desc = "  ".join(f"{int(d[i])}:{c[i]:.2f}"
                         f"({','.join(str(int(d[i]) % l) for l in LAMBDAS)})"
                         for i in pk)
        print(f"    {name}: C(1) {c[0]:.3f}  mean {c.mean():.3f}  "
              f">0.9 {np.mean(c > 0.9):.2f}  >0.5 {np.mean(c > 0.5):.2f}")
        print(f"      peaks d:cos(res 11,12,13): {desc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="corner500")
    ap.add_argument("--dmax", type=int, default=1100)
    ap.add_argument("--refs", default="200,300;700,300;1100,300")
    args = ap.parse_args()
    refs = [tuple(int(v) for v in r.split(",")) for r in args.refs.split(";")]
    torch.set_num_threads(4)
    for label, path in default_encoders():
        if args.only not in label:
            continue
        enc, ecfg, gain, fwhm, _h = load_probe_encoder(path, fwhm_fallback=0.25)
        enc.gain = gain
        field = Field(enc, list(ecfg.lambdas), fwhm, gain, NPOS)
        print(f"\n=== {label} ===")
        for ref in refs:
            report(label, field, ref, args.dmax)


if __name__ == "__main__":
    main()
