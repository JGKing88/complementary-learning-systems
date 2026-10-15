"""Encoder figure numbers: similarity against distance.

Run: ~/.conda/envs/cls/bin/python -m cosyne2027.encoder_similarity
Writes cosyne2027/data/encoder_similarity.json for fig_encoder.py. CPU only;
grid codes are computed per position (eval_unique_radius.grid_code_batch),
so the 12 GB scaffold is never loaded.

Three representations of a scaffold position: the smoothed grid code (434-d,
lambdas 11/12/13, fwhm 0.25), d0_base's trained encoder applied to it, and an
untrained encoder of the same architecture (create_encoder, torch seed 0,
same gain). For each distance d = 0..D_MAX, N_REF reference positions at
least BORDER cells from the scaffold edge are paired with a position d cells
away along x or y (an axis-aligned offset, uniform over the four); the
curve is the mean and SD over those pairs of the raw cosine similarity.
Axis-aligned, as in encoder_training/notebooks/explore.ipynb's slice: the
grid code is a 2-D lattice code, so its similarity depends on direction,
and averaging over all directions at a Euclidean distance mixes a finite,
ring-dependent set of lattice offsets into a jagged curve that dilutes the
lambda bump (0.25 against 0.91 along an axis). The grid code is
translation-invariant (each module's bump shifts with position), so its
curve is exact and its SD is zero; the encoders' SD is spread across
positions. Raw, not
mean-centred: it is what the attractor compares. (An untrained encoder maps
every position to nearly one vector, so its raw cosine sits near 1; centred,
it would retrace the grid-code curve.)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from encoder_training.eval_unique_radius import grid_code_batch
from encoder_training.models import create_encoder
from hopfield_nav.encoder_io import load_encoder

HERE = Path(__file__).resolve().parent
ENCODER = ("/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/"
           "001_att0.5_seed=43/encoder_final.pt")
LAMBDAS = [11, 12, 13]
FWHM = 0.25
NPOS = 11 * 12 * 13
D_MAX = 100
N_REF = 2000
BORDER = 150
# Example units: explore.ipynb cell 5's view (window, unit draw, processing)
FIELD_WINDOW = 200
FIELD_SEED = 11
N_UNITS = 6
HIGHPASS_SIGMA = 30.0
BLUR_SIGMA = 2.0
FIELD_THRESHOLD = 0.6     # final ReLU at this fraction of the unit's max


@torch.no_grad()
def embed(encoder, xs, ys, batch=8192):
    out = []
    for i in range(0, len(xs), batch):
        g = torch.from_numpy(grid_code_batch(LAMBDAS, xs[i:i + batch],
                                             ys[i:i + batch], FWHM))
        out.append(encoder(g).numpy())
    return np.concatenate(out)


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def offsets_at(d):
    """The axis-aligned offsets of length d: (+-d, 0) and (0, +-d)."""
    if d == 0:
        return np.zeros((1, 2), dtype=int)
    return np.array([[d, 0], [-d, 0], [0, d], [0, -d]])


def similarity_curves(encoders, rng):
    lo, hi = BORDER, NPOS - BORDER - 1
    stats = {name: {"mean": [], "sd": []} for name in ["grid code", *encoders]}
    for d in range(D_MAX + 1):
        ref = rng.integers(lo, hi + 1, size=(N_REF, 2))
        offs = offsets_at(d)
        other = ref + offs[rng.integers(0, len(offs), N_REF)]
        g1 = unit(grid_code_batch(LAMBDAS, ref[:, 0], ref[:, 1], FWHM))
        g2 = unit(grid_code_batch(LAMBDAS, other[:, 0], other[:, 1], FWHM))
        reps = {"grid code": (g1, g2)}
        for name, enc in encoders.items():
            reps[name] = (embed(enc, ref[:, 0], ref[:, 1]),
                          embed(enc, other[:, 0], other[:, 1]))
        for name, (a, b) in reps.items():
            c = np.sum(a * b, axis=1)
            stats[name]["mean"].append(float(c.mean()))
            stats[name]["sd"].append(float(c.std()))
    return stats


def unit_maps(encoder):
    """Six trained units over a FIELD_WINDOW-cell window at the scaffold centre,
    processed as in explore.ipynb cell 5: subtract the unit's mean, ReLU,
    subtract a sigma-30 Gaussian blur (high-pass), blur at sigma 2; then a
    final ReLU at FIELD_THRESHOLD of the map's maximum, which keeps the
    peaks (16% of the window active instead of 43%). A stronger high-pass
    (sigma 10-15) gives ridges along region borders, not fields. The raw
    units are near-binary +-0.034 sign patterns; the processing turns their
    positive regions into graded fields. Units are drawn with
    RandomState(FIELD_SEED), the notebook's draw, not selected for looks.
    Returns (n_units, W, W) maps indexed [unit, x, y] and the unit ids."""
    s = FIELD_WINDOW
    c0 = NPOS // 2 - s // 2
    xs, ys = np.meshgrid(np.arange(c0, c0 + s), np.arange(c0, c0 + s),
                         indexing="ij")
    z = embed(encoder, xs.ravel(), ys.ravel()).reshape(s, s, -1)
    units = np.random.RandomState(FIELD_SEED).choice(z.shape[2], N_UNITS,
                                                     replace=False)
    maps = []
    for u in units:
        x = z[:, :, u].astype(np.float64)
        x = np.maximum(x - x.mean(), 0.0)
        x = x - gaussian_filter(x, HIGHPASS_SIGMA)
        x = gaussian_filter(x, BLUR_SIGMA)
        maps.append(np.maximum(x - FIELD_THRESHOLD * x.max(), 0.0))
    return np.array(maps, dtype=np.float32), units


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    trained, cfg, _gain = load_encoder(ENCODER, "cpu")
    trained.eval()
    torch.manual_seed(0)
    untrained = create_encoder(cfg, "cpu").eval()
    rng = np.random.default_rng(args.seed)
    curves = similarity_curves({"trained encoder": trained,
                                "untrained encoder": untrained}, rng)
    out = {"distances": list(range(D_MAX + 1)), "n_ref": N_REF,
           "lambdas": LAMBDAS, "encoder": ENCODER, "curves": curves}
    (HERE / "data").mkdir(exist_ok=True)
    with open(HERE / "data" / "encoder_similarity.json", "w") as f:
        json.dump(out, f)
    maps, units = unit_maps(trained)
    np.savez_compressed(HERE / "data" / "encoder_unit_maps.npz", maps=maps,
                        units=units, window=FIELD_WINDOW)
    print("unit maps:", units.tolist(), maps.shape)
    for name, s in curves.items():
        m = np.array(s["mean"])
        print(f"{name:18s} d=1 {m[1]:.3f}  d=12 {m[12]:.3f}  d=20 {m[20]:.3f}"
              f"  d=60 {m[60]:.3f}  d=100 {m[100]:.3f}")


if __name__ == "__main__":
    main()
