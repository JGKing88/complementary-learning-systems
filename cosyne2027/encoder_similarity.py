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
from scipy import ndimage
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
# Example units: explore.ipynb cell 5's processing, sharpened into peaks
FIELD_WINDOW = 200        # window processed and shown
DOMINANT = 0.7            # largest field's share of a unit's map mass
PEAK_BLUR = 3.0
PEAK_POWER = 2
N_ROWS, N_COLS = 3, 2     # panel layout; the fields tile the same partition
HIGHPASS_SIGMA = 30.0
BLUR_SIGMA = 2.0
FIELD_THRESHOLD = 0.7     # ReLU at this fraction of the unit's max


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
    """Six trained units with one dominant, peaked place field, tiling the
    FIELD_WINDOW-cell window at the scaffold centre.

    Each unit is processed as in explore.ipynb cell 5 (subtract the unit's
    mean, ReLU, subtract a sigma-30 Gaussian blur, blur at sigma 2), then
    sharpened into peaks: ReLU at FIELD_THRESHOLD of the map's maximum,
    a sigma-PEAK_BLUR blur and a power PEAK_POWER. The raw units are
    near-binary +-0.034 sign patterns over large regions, so after the
    notebook's steps a field is a plateau; the last three steps keep its
    core and round it into a peak. A unit qualifies if its largest field
    holds at least DOMINANT of the map's mass and does not touch the window
    edge; the count is returned for the caption. From the qualifying units,
    the one whose field centroid is nearest each cell centre of an
    N_ROWS x N_COLS partition of the window is taken, row-major from the
    top-left, so the six fields tile the window and map k sits where panel
    k sits.

    Returns maps (6, W, W) indexed [unit, x, y], the unit ids and the number
    of qualifying units.
    """
    s = FIELD_WINDOW
    c0 = NPOS // 2 - s // 2
    xs, ys = np.meshgrid(np.arange(c0, c0 + s), np.arange(c0, c0 + s),
                         indexing="ij")
    z = embed(encoder, xs.ravel(), ys.ravel()).reshape(s, s, -1)
    cands = []
    for u in range(z.shape[2]):
        x = np.maximum(z[:, :, u].astype(np.float64) - z[:, :, u].mean(), 0.0)
        x = x - gaussian_filter(x, HIGHPASS_SIGMA)
        x = gaussian_filter(x, BLUR_SIGMA)
        x = np.maximum(x - FIELD_THRESHOLD * x.max(), 0.0)
        m = gaussian_filter(x, PEAK_BLUR) ** PEAK_POWER
        if m.max() <= 0:
            continue
        lab, n = ndimage.label(m > 1e-3 * m.max())
        mass = ndimage.sum(m, lab, range(1, n + 1))
        big = lab == int(np.argmax(mass)) + 1
        on_edge = (big[0].any() or big[-1].any() or big[:, 0].any()
                   or big[:, -1].any())
        if mass.max() / mass.sum() >= DOMINANT and not on_edge:
            cands.append((u, np.array(ndimage.center_of_mass(m * big)), m))
    cents = np.array([c for _, c, _ in cands])
    chosen = []
    for row in range(N_ROWS):
        for col in range(N_COLS):
            target = np.array([(col + 0.5) / N_COLS,
                               1 - (row + 0.5) / N_ROWS]) * s
            d = np.linalg.norm(cents - target, axis=1)
            d[chosen] = np.inf
            chosen.append(int(np.argmin(d)))
    maps = np.array([cands[i][2] for i in chosen], dtype=np.float32)
    return maps, np.array([cands[i][0] for i in chosen]), len(cands)


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
    maps, units, n_qual = unit_maps(trained)
    np.savez_compressed(HERE / "data" / "encoder_unit_maps.npz", maps=maps,
                        units=units, window=FIELD_WINDOW,
                        n_qualifying=n_qual, rows=N_ROWS, cols=N_COLS)
    print(f"unit maps: {units.tolist()}  {n_qual} of 1024 units qualify "
          f"({n_qual / 1024:.0%})")
    for name, s in curves.items():
        m = np.array(s["mean"])
        print(f"{name:18s} d=1 {m[1]:.3f}  d=12 {m[12]:.3f}  d=20 {m[20]:.3f}"
              f"  d=60 {m[60]:.3f}  d=100 {m[100]:.3f}")


if __name__ == "__main__":
    main()
