"""Power spectrum and position-averaged kernel of an encoder over the whole 1716 x 1716 torus.

    python -m analysis.hopfield_probe.trained_spectrum --out DIR [--only ideal att42 ...]

Every grid-code encoder is a function on the torus, because the grid code repeats every
L = 1716 cells along each axis. So each output unit's map z_i(x) is a sum of the plane waves
exp(2 pi i n.x / L), n = (n1, n2) whole numbers mod L. For each encoder this script encodes
all L^2 positions, takes the 2D FFT F_i of every unit's map and saves

    S(n) = sum_i |F_i(n)|^2 / L^4

the share of the code's power on wave n. For a unit-norm code S sums to 1. The position-averaged
kernel is its inverse FFT:

    k(D) = (1/L^2) sum_x z(x).z(x+D) = sum_n S(n) cos(2 pi n.D / L) = L^2 ifft2(S)

It also saves z(x).z(x+D) at 20,000 random positions x for a list of displacements D, which
measures how much the similarity depends on where you are (zero spread for the ideal code), and
each unit's purity: the share of its power (constant removed) on its strongest wave pair +-n.

Encoders (label: what):
  ideal       ideal:r=16,n_freq=512,seed=0,gain=100, the probe's reference ideal code
  att42/43    w52 att0.5, seeds 42 and 43, encoder_final.pt (gradient descent, production)
  att42u/43u  the same before the tanh, u / |u| with u = net(c)
  gwn         the grid wave network learned with local rules (conc_clean_eta0.2_late.pt)
  input       the smoothed grid code itself, c / |c|
Writes DIR/S_<label>.npy (float64, L x L, row n1 mod L, column n2 mod L), DIR/ti_<label>.npz and
DIR/meta_<label>.json. Encoders whose S file exists are skipped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .encode import grid_codes
from .harness import load_probe_encoder

L = 1716
LAMBDAS = [11, 12, 13]
SW = "/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm"
GWN = ("/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/gwn/online/encoders/"
       "conc_clean_eta0.2_late.pt")
ENCODERS = {
    "ideal": ("ideal:r=16,n_freq=512,seed=0,gain=100", "final"),
    "att42": (f"{SW}/000_att0.5_seed=42/encoder_final.pt", "final"),
    "att43": (f"{SW}/001_att0.5_seed=43/encoder_final.pt", "final"),
    "att42u": (f"{SW}/000_att0.5_seed=42/encoder_final.pt", "pretanh"),
    "att43u": (f"{SW}/001_att0.5_seed=43/encoder_final.pt", "pretanh"),
    "gwn": ("gwn:" + GWN, "final"),
    "input": (None, "input"),
}
# displacements (along x unless the second entry is set) for the translation test
SHIFTS = [(1, 0), (2, 0), (5, 0), (10, 0), (11, 0), (13, 0), (19, 0), (24, 0), (32, 0),
          (50, 0), (100, 0), (300, 0), (780, 0), (792, 0), (924, 0), (936, 0), (0, 792),
          (5, 5), (19, 19)]
N_TI = 20_000
BLOCK = 1024          # output units held in memory at once: L * L * BLOCK * 4 bytes = 12 GB
FFT_CHUNK = 32        # units per batched FFT on the device


def make_fn(spec, kind, device):
    """Map a batch of grid codes to the code being analysed, and the smoothing it expects."""
    if kind == "input":
        return (lambda c: F.normalize(c, dim=-1)), 0.25, {"what": "grid code / |grid code|"}
    enc, _cfg, gain, fwhm, header = load_probe_encoder(spec, device=device, fwhm_fallback=0.25)
    enc.eval()
    info = {"spec": spec, "kind": kind, "gain": float(gain), "fwhm_ratio": float(fwhm)}
    if kind == "pretanh":
        return (lambda c: F.normalize(enc.net(c), dim=-1)), fwhm, info
    return (lambda c: enc(c)), fwhm, info


def encode_block(fn, fwhm, u0, u1, device, rows):
    """All L^2 positions, output units u0..u1-1: (L, L, u1 - u0) float32, x then y."""
    Z = np.empty((L, L, u1 - u0), np.float32)
    ys = np.arange(L)
    for x0 in range(0, L, rows):
        xr = np.arange(x0, min(L, x0 + rows))
        gx, gy = np.repeat(xr, L), np.tile(ys, len(xr))
        c = torch.from_numpy(grid_codes(LAMBDAS, gx, gy, fwhm)).to(device)
        with torch.no_grad():
            z = fn(c)[:, u0:u1]
        Z[x0:x0 + len(xr)] = z.float().cpu().numpy().reshape(len(xr), L, u1 - u0)
    return Z


def run(label, out: Path, device, rows):
    spec, kind = ENCODERS[label]
    t0 = time.time()
    fn, fwhm, info = make_fn(spec, kind, device)
    with torch.no_grad():
        D = int(fn(torch.from_numpy(grid_codes(LAMBDAS, np.zeros(1), np.zeros(1), fwhm))
                   .to(device)).shape[-1])
    rng = np.random.default_rng(0)
    px, py = rng.integers(0, L, N_TI), rng.integers(0, L, N_TI)
    S = np.zeros((L, L))
    ti = np.zeros((len(SHIFTS), N_TI))
    norm2 = np.zeros(N_TI)
    purity = []
    for u0 in range(0, D, BLOCK):
        u1 = min(D, u0 + BLOCK)
        Z = encode_block(fn, fwhm, u0, u1, device, rows)
        print(f"  {label}: units {u0}-{u1} encoded at {time.time() - t0:.0f} s", flush=True)
        a = Z[px, py].astype(np.float64)
        norm2 += (a * a).sum(1)
        for k, (dx, dy) in enumerate(SHIFTS):
            ti[k] += (a * Z[(px + dx) % L, (py + dy) % L]).sum(1)
        for i0 in range(0, u1 - u0, FFT_CHUNK):
            blk = np.ascontiguousarray(Z[:, :, i0:i0 + FFT_CHUNK].transpose(2, 0, 1))
            P = torch.fft.fft2(torch.from_numpy(blk).to(device, torch.float64)).abs() ** 2
            S += P.sum(0).cpu().numpy() / float(L) ** 4
            P[:, 0, 0] = 0.0                                   # constant removed for purity
            tot = P.flatten(1).sum(1)
            top = P.flatten(1).max(1).values
            purity.append((2 * top / tot.clamp_min(1e-300)).cpu().numpy())
        del Z
    np.save(out / f"S_{label}.npy", S)
    np.savez(out / f"ti_{label}.npz", shifts=np.array(SHIFTS), dots=ti, norm2=norm2,
             px=px, py=py, purity=np.concatenate(purity))
    meta = dict(label=label, D=D, seconds=round(time.time() - t0, 1), S_total=float(S.sum()),
                S_const=float(S[0, 0]), norm2_min=float(norm2.min()), norm2_max=float(norm2.max()),
                **info)
    (out / f"meta_{label}.json").write_text(json.dumps(meta, indent=1))
    print(f"{label}: D {D}, S total {S.sum():.6f}, constant share {S[0, 0]:.4f}, "
          f"{time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=list(ENCODERS))
    ap.add_argument("--rows", type=int, default=32, help="x rows of positions per forward pass")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
    for label in a.only:
        if (out / f"S_{label}.npy").exists():
            print(f"{label}: exists, skipped", flush=True)
            continue
        run(label, out, device, a.rows)


if __name__ == "__main__":
    main()
