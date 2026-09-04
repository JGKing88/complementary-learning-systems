"""Why does the direction field die when the encoder is saturated?

Sec 10.20 asserts a mechanism -- `q` is a finite difference of neighbouring cell
embeddings, and `sign(z)` has no usable local derivative -- and then does not
test it. This does.

`q` is built in two stages (`Field.local_basis`, `qfield.project_q`):

    d_fwd = z(x, y+1) - z(x, y)          North
    d_rgt = z(x+1, y) - z(x, y)          East
    basis = gram_schmidt(d_fwd, d_rgt)
    q     = basis @ (z_goal - z_here)

so there are two separate ways it can fail, and they have different fixes:

  (H1) the BASIS is degenerate -- `d_fwd` and `d_rgt` are near-zero or
       near-parallel, so the 2D frame is noise;
  (H2) the DISPLACEMENT is uninformative -- `z_goal - z_here` stops pointing
       anywhere in particular once the two codes decorrelate, which for a
       binary code happens as soon as the Hamming distance saturates.

Measured here for both, at the same positions, at encoder gain 100 and 1e6:

  ||d_fwd||, ||d_rgt||, and how often either is exactly zero  -> H1
  cos(d_fwd, d_rgt), the frame's conditioning                 -> H1
  cos(z(x + k*North) - z(x), d_fwd) against k                 -> H2
  cos(z(x), z(x + k*North)) against k, i.e. does the code
      decorrelate and plateau                                 -> H2

The linear arm is the control: whatever the binary arm does, the same numbers at
gain 100 say what a working direction field looks like.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
CK = f"{S}/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt"
NPOS, N_POS_SAMPLE = 1716, 1500
KS = (1, 2, 4, 8, 16, 32)


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-30)


def main() -> None:
    rng = np.random.RandomState(0)
    gx = rng.randint(64, NPOS - 64, size=N_POS_SAMPLE)
    gy = rng.randint(64, NPOS - 64, size=N_POS_SAMPLE)

    for gain in (100.0, 1e6):
        enc, ecfg, _g, fwhm, _h = load_probe_encoder(CK, fwhm_fallback=0.25)
        enc.gain = gain
        field = Field(enc, list(ecfg.lambdas), fwhm, gain, NPOS)

        z0 = unit(field.encode(gx, gy))
        d_fwd = unit(field.encode(gx, gy + 1)) - z0
        d_rgt = unit(field.encode(gx + 1, gy)) - z0
        n_f = np.linalg.norm(d_fwd, axis=1)
        n_r = np.linalg.norm(d_rgt, axis=1)
        # Angle between the two axes the frame is built from. Near 0 or 180
        # degrees means the frame is one direction, not two.
        cos_fr = (unit(d_fwd) * unit(d_rgt)).sum(1)

        # How much of the code even changes over one cell.
        flip = np.mean(np.sign(z0) != np.sign(unit(field.encode(gx, gy + 1))),
                       axis=1)

        print(f"\n=== encoder gain {gain:g} ===  {N_POS_SAMPLE} scaffold "
              f"positions, D={z0.shape[1]}")
        print(f"  ||d_fwd||                 mean {n_f.mean():.4f}   "
              f"min {n_f.min():.4f}   zero at {np.mean(n_f < 1e-9):.1%} of "
              f"positions")
        print(f"  ||d_rgt||                 mean {n_r.mean():.4f}   "
              f"min {n_r.min():.4f}   zero at {np.mean(n_r < 1e-9):.1%} of "
              f"positions")
        print(f"  |cos(d_fwd, d_rgt)|       mean {np.abs(cos_fr).mean():.4f}   "
              f"max {np.abs(cos_fr).max():.4f}    (0 = orthogonal axes)")
        print(f"  coords changed per cell   {flip.mean():.1%}")

        print(f"  {'k cells north':<22s}" + "".join(f"{k:>9d}" for k in KS))
        align, corr, nrm, proj = [], [], [], []
        nf = unit(d_fwd)
        for k in KS:
            zk = unit(field.encode(gx, gy + k))
            dk = zk - z0
            align.append(float(np.mean((unit(dk) * nf).sum(1))))
            corr.append(float(np.mean((zk * z0).sum(1))))
            nrm.append(float(np.mean(np.linalg.norm(dk, axis=1))))
            # THE readout. `project_q` is basis @ (z_goal - z_here), so for a
            # goal k cells due north this projection IS q_north. If it does not
            # grow with k, `q` cannot encode distance and its ratio to q_east
            # cannot encode bearing.
            proj.append(float(np.mean((dk * nf).sum(1))))
        print(f"  {'cos(z(x+k)-z(x), d_fwd)':<22s}"
              + "".join(f"{v:>9.3f}" for v in align))
        print(f"  {'cos(z(x), z(x+k))':<22s}"
              + "".join(f"{v:>9.3f}" for v in corr))
        print(f"  {'||z(x+k) - z(x)||':<22s}"
              + "".join(f"{v:>9.3f}" for v in nrm))
        print(f"  {'q_north = <dk, d_fwd>':<22s}"
              + "".join(f"{v:>9.3f}" for v in proj)
              + "   <- the readout itself")


if __name__ == "__main__":
    main()
