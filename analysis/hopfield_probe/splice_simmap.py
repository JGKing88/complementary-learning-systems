"""Compute the code's 2D similarity kernel per encoder and splice it in.

Everything in THEORY_ENCODER_HOPFIELD.md reduces to one function -- the
similarity kernel ``C(a) = <phi(p), phi(p + a)>`` -- and the report shows every
downstream summary of it (res90, the alias rate, the basin, `q`'s angular error)
without ever showing the thing itself. This adds it.

Two panels, because one colour scale cannot carry both regimes and the whole
point of Sec 5.2 is that they are different regimes:

  near   a in [-16, 16]^2, full scale 0 to 1. The chart the readout differences
         across: how fast similarity falls, and whether it falls isotropically.
  far    a in [-48, 48]^2, diverging and CLIPPED, so the +-1/sqrt(d_eff) tail is
         legible instead of being a uniform grey field beside a saturated core.
         Lattice revivals -- the grid code's aliases at multiples of lambda,
         11 / 12 / 13 -- appear here if the encoder failed to suppress them.

Averaged over reference positions rather than taken at one. The code is
approximately stationary (Sec 2.0), so the average IS the kernel and a single
reference is a noisy sample of it.

Written into each result JSON as a top-level ``sim_map`` so the report can be
rebuilt without re-running any probe: this needs one encoder forward pass per
reference position and nothing else. Reach, basin and direction are untouched.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

NEAR_R, FAR_R = 16, 48
N_REFS = 48


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-30)


def kernel(field: Field, radius: int, n_refs: int, npos: int,
           rng: np.random.RandomState) -> np.ndarray:
    """Mean ``cos(z(ref), z(ref + a))`` over refs, as a (2R+1, 2R+1) array.

    Indexed ``[dx + R][dy + R]`` so it matches ``figures.heatmap``'s convention
    of column = x rightward, row = y upward.
    """
    off = np.arange(-radius, radius + 1)
    dx, dy = np.meshgrid(off, off, indexing="ij")
    dx, dy = dx.ravel(), dy.ravel()

    # Keep every reference far enough inside the scaffold that no offset clips.
    lo, hi = radius + 1, npos - radius - 2
    gx = rng.randint(lo, hi, size=n_refs)
    gy = rng.randint(lo, hi, size=n_refs)

    acc = np.zeros(dx.size, dtype=np.float64)
    for i in range(n_refs):
        z0 = _unit(field.encode(gx[i:i + 1], gy[i:i + 1]))[0]
        zz = _unit(field.encode(gx[i] + dx, gy[i] + dy))
        acc += zz @ z0
    return (acc / n_refs).reshape(2 * radius + 1, 2 * radius + 1)


def sim_map(header: dict, npos: int, seed: int = 0) -> dict:
    enc, ecfg, gain, fwhm, _h = load_probe_encoder(
        header["path"], fwhm_fallback=header.get("fwhm_ratio", 0.25))
    gain = float(header.get("gain", gain))
    enc.gain = gain
    field = Field(enc, list(ecfg.lambdas), fwhm, gain, npos)

    rng = np.random.RandomState(seed)
    near = kernel(field, NEAR_R, N_REFS, npos, rng)
    far = kernel(field, FAR_R, N_REFS, npos,
                 np.random.RandomState(seed + 1))

    # The far-field spread, which is what sets the clip and is also Sec 2.0's
    # 1/sqrt(d_eff): taken outside the core so the core does not inflate it.
    o = np.arange(-FAR_R, FAR_R + 1)
    r = np.hypot(*np.meshgrid(o, o, indexing="ij"))
    tail = far[r > 32]
    sd = float(tail.std())

    # res90 read straight off the near panel, along the axes, as a check that
    # this map and the reported res90 are the same object.
    mid = NEAR_R
    axis = np.concatenate([near[mid, mid:], near[mid:, mid]])
    hit = np.flatnonzero(axis < 0.9)
    return {
        "near_radius": NEAR_R, "far_radius": FAR_R, "n_refs": N_REFS,
        "near": [[float(v) for v in row] for row in near],
        "far": [[float(v) for v in row] for row in far],
        "far_sd": sd,
        "far_absmax": float(np.abs(tail).max()),
        "res90_axis": int(hit[0]) if hit.size else None,
        "clip": float(3.0 * sd),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dir", help="a probe output dir with a manifest")
    p.add_argument("--out", default=None,
                   help="write here instead of patching in place")
    p.add_argument("--npos", type=int, default=1716)
    args = p.parse_args(argv)

    src = pathlib.Path(args.result_dir)
    dst = pathlib.Path(args.out) if args.out else src
    if dst != src:
        dst.mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(src / "*.json")):
            shutil.copy2(f, dst / os.path.basename(f))

    files = sorted(f for f in glob.glob(str(dst / "*.json"))
                   if os.path.basename(f) != "manifest.json")
    for f in files:
        with open(f) as fh:
            res = json.load(fh)
        if "sim_map" in res:
            print(f"  skip (has one) {os.path.basename(f)}")
            continue
        npos = res.get("config", {}).get("Npos", args.npos)
        res["sim_map"] = sim_map(res["header"], npos)
        with open(f, "w") as fh:
            json.dump(res, fh)
        sm = res["sim_map"]
        print(f"  {res['header']['label']:<34s} res90 {sm['res90_axis']}"
              f"  far sd {sm['far_sd']:.4f}  |far| max {sm['far_absmax']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
