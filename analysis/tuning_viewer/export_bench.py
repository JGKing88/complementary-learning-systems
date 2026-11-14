"""Export the trained encoder and reference maps for the browser speed test.

    python -m analysis.tuning_viewer.export_bench [--encoder PATH] [--out DIR]

Writes three kinds of file to ``DIR`` (default ``bench_data/`` beside this
module, which git ignores). Artifacts serve no raw binary type, so the arrays
travel as base64 text of their float32 little-endian bytes:

``weights.b64.txt``
    the arrays named in ``manifest.json``, in order.
``ref_<name>.b64.txt``
    float32 ``(S*S, out_dim)``: the encoder's output over one arena, from the
    repo's own ``encode_positions``. Cell ``n = row*S + col`` sits at scaffold
    ``x = (x0 + col) % Npos``, ``y = (y0 + row) % Npos``. The page compares
    its own output against these.
``manifest.json``
    shapes and offsets, gain, lambdas, fwhm, Npos, the reference arenas, and
    the result of the fold check below.

The first Linear layer is folded into one table per grid module. The smoothed
grid code is a concatenation of per-module blocks, and module ``m``'s block
depends only on ``(x mod l_m, y mod l_m)`` (``hopfield_probe/encode.py``), so

    W1 g(x, y) = sum_m T_m[(x mod l_m) * l_m + (y mod l_m)],
    T_m = G_m W1[:, block_m]^T,   G_m the l_m^2 smoothed codes of module m.

The tables are stacked into one ``T`` of ``sum l^2 = 434`` rows; module ``m``'s
rows start at ``row_offsets[m]``. The page then needs three row lookups per
cell instead of a 434-wide matrix product.

The fold check runs that computation in numpy, with the polynomial erf the
page uses for GELU (Abramowitz and Stegun 7.1.26, |error| <= 1.5e-7), and
reports its largest difference from the real encoder over random scaffold
cells. A wrong row layout would show up here as an O(1) error, not a rounding
one.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf as erf_exact

from analysis.hopfield_probe.encode import encode_positions, grid_codes
from analysis.hopfield_probe.harness import load_probe_encoder

HERE = Path(__file__).resolve().parent
ENCODER = ("/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/"
           "001_att0.5_seed=43/encoder_final.pt")
# Reference arenas. "cosyne" is the first arena of cosyne fig 3, seed 1
# (agenthash_heldout/agenthash_s1.json, metadata.extra.env_offsets_per_iter[0]);
# "wrap" crosses both scaffold edges, to test the page's modular indexing.
REF_ARENAS = [
    {"name": "cosyne", "x0": 280, "y0": 239, "S": 20,
     "label": "cosyne fig 3, seed 1, first arena"},
    {"name": "wrap", "x0": 1700, "y0": 1705, "S": 24,
     "label": "crosses both scaffold edges"},
]
N_CHECK = 4096


def arena_xy(x0: int, y0: int, S: int, npos: int) -> tuple[np.ndarray, np.ndarray]:
    """Scaffold coordinates of an arena's cells, in the page's cell order."""
    n = np.arange(S * S)
    row, col = n // S, n % S
    return (x0 + col) % npos, (y0 + row) % npos


def erf_poly(x: np.ndarray) -> np.ndarray:
    """Abramowitz and Stegun 7.1.26, the erf the page computes."""
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
          - 0.284496736) * t + 0.254829592) * t
    return np.sign(x) * (1.0 - p * np.exp(-a * a))


def folded_forward(xs, ys, tables, layers, lambdas, row_offsets, gain, erf):
    """The page's computation, in float64."""
    T, b1 = tables
    h = b1[None, :].copy()
    for l, off in zip(lambdas, row_offsets):
        h = h + T[off + (xs % l) * l + (ys % l)]
    gelu = lambda v: 0.5 * v * (1.0 + erf(v / math.sqrt(2.0)))
    h = gelu(h)
    for W, b in layers[:-1]:
        h = gelu(h @ W + b)
    W, b = layers[-1]
    z = np.tanh(np.clip(gain * (h @ W + b), -15.0, 15.0))
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def write_b64(path: Path, raw: bytes) -> None:
    path.write_text(base64.b64encode(raw).decode("ascii"))


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=HERE,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--encoder", default=ENCODER)
    p.add_argument("--out", type=Path, default=HERE / "bench_data")
    args = p.parse_args()

    encoder, cfg, gain, fwhm, header = load_probe_encoder(args.encoder)
    encoder.eval()
    lambdas = [int(l) for l in cfg.lambdas]
    npos = int(np.prod(lambdas))
    if float(encoder.gain) != float(gain):
        raise ValueError(f"encoder.gain {encoder.gain} != resolved gain {gain}; "
                         "encode_positions uses encoder.gain")
    linears = [m for m in encoder.net if isinstance(m, torch.nn.Linear)]
    acts = [m for m in encoder.net if isinstance(m, torch.nn.GELU)]
    if len(acts) != len(linears) - 1 or any(a.approximate != "none" for a in acts):
        raise ValueError("expected exact GELU between every pair of Linear layers")
    if encoder.output_nonlinearity != "tanh":
        raise ValueError(f"expected tanh output, got {encoder.output_nonlinearity}")
    Wb = [(m.weight.detach().double().numpy(), m.bias.detach().double().numpy())
          for m in linears]
    hidden, out_dim = Wb[0][0].shape[0], Wb[-1][0].shape[0]

    # Fold layer 1: T_m[px * l + py] = W1[:, block_m] @ code_m(px, py).
    W1, b1 = Wb[0]
    parts, row_offsets, col = [], [], 0
    for l in lambdas:
        px, py = np.meshgrid(np.arange(l), np.arange(l), indexing="ij")
        codes = grid_codes(lambdas, px.ravel(), py.ravel(), fwhm).astype(np.float64)
        block = codes[:, col:col + l * l]                    # (l^2 phases, l^2)
        row_offsets.append(sum(t.shape[0] for t in parts))
        parts.append(block @ W1[:, col:col + l * l].T)      # (l^2, hidden)
        col += l * l
    T = np.concatenate(parts, axis=0)
    layers = [(W.T.copy(), b) for W, b in Wb[1:]]           # stored [in][out]

    # Fold check against the real encoder, on random scaffold cells.
    rng = np.random.default_rng(0)
    xs, ys = rng.integers(0, npos, N_CHECK), rng.integers(0, npos, N_CHECK)
    truth = encode_positions(encoder, lambdas, fwhm, xs, ys).astype(np.float64)
    check = {}
    for name, erf in (("exact_erf", erf_exact), ("poly_erf", erf_poly)):
        got = folded_forward(xs, ys, (T, b1), layers, lambdas, row_offsets,
                             float(gain), erf)
        check[name] = float(np.abs(got - truth).max())
    print(f"fold check over {N_CHECK} cells, max |diff|: "
          + ", ".join(f"{k} {v:.2e}" for k, v in check.items()))
    if check["poly_erf"] > 1e-3:
        raise RuntimeError("folded forward disagrees with the encoder; not exporting")

    # Weights, in the order the page reads them.
    arrays = [("T", T), ("b1", b1)]
    for i, (W, b) in enumerate(layers, start=2):
        arrays += [(f"W{i}", W), (f"b{i}", b)]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_arrays, offset, chunks = [], 0, []
    for name, a in arrays:
        a32 = np.ascontiguousarray(a, dtype="<f4")
        chunks.append(a32.tobytes())
        manifest_arrays.append({"name": name, "shape": list(a32.shape),
                                "offset": offset})
        offset += a32.size
    write_b64(args.out / "weights.b64.txt", b"".join(chunks))

    refs = []
    for arena in REF_ARENAS:
        xs_a, ys_a = arena_xy(arena["x0"], arena["y0"], arena["S"], npos)
        ref = encode_positions(encoder, lambdas, fwhm, xs_a, ys_a)
        fname = f"ref_{arena['name']}.b64.txt"
        write_b64(args.out / fname, np.ascontiguousarray(ref, dtype="<f4").tobytes())
        refs.append({**arena, "file": fname, "shape": list(ref.shape)})

    manifest = {
        "encoder": args.encoder,
        "encoder_header": header,
        "git_sha": git_sha(),
        "lambdas": lambdas, "npos": npos, "fwhm_ratio": float(fwhm),
        "gain": float(gain), "hidden": int(hidden), "out_dim": int(out_dim),
        "row_offsets": row_offsets,
        "weights_file": "weights.b64.txt",
        "arrays": manifest_arrays,
        "refs": refs,
        "fold_check": {"n_cells": N_CHECK, "max_abs_diff": check},
    }
    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    total = sum((args.out / n).stat().st_size
                for n in ["weights.b64.txt", "manifest.json"] + [r["file"] for r in refs])
    print(f"wrote {args.out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
