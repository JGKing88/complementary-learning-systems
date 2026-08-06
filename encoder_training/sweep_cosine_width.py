#!/usr/bin/env python3
"""
Sweep (lambdas, Np, thresh, fwhm_ratio) and record cosine-similarity peak width
in the central place-code window (same metric as vector_hash_playground).

Run from the repository root (directory that contains the `cls` package), e.g.:
    python sweep_cosine_width.py -o cosine_width_sweep.csv

Each configuration recomputes gbook, Wpg, and pbook (no retained caches).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import randint, randn
from tqdm import tqdm

# Repo root = parent of this file
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gridcode.smoothing import smooth_gbook
from gridcode.assoc import nonlin, train_pbook
from gridcode.codebook import gen_gbook_2d


# Default sweep grids — edit here or extend CLI later
DEFAULT_LAMBDA_TUPLES = [(11, 12), (11, 12, 13)]
DEFAULT_NP_LIST = [500, 1000, 2000]
DEFAULT_THRESH_LIST = [0.5, 1.0, 1.5, 2.0, 2.5]
DEFAULT_FWHM_RATIO_LIST = [0.3, 0.35, 0.4, 0.45, 0.5]


def find_peak_bounds(slc: np.ndarray, peak_idx: int) -> tuple[int, int]:
    left = peak_idx
    while left > 0 and slc[left - 1] <= slc[left]:
        left -= 1
    right = peak_idx
    while right < len(slc) - 1 and slc[right + 1] <= slc[right]:
        right += 1
    outside = np.concatenate([slc[:left], slc[right + 1 :]])
    if len(outside) == 0:
        return 0, len(slc) - 1
    thresh_val = outside.max()
    left_bound = peak_idx
    while left_bound > 0 and slc[left_bound - 1] > thresh_val:
        left_bound -= 1
    right_bound = peak_idx
    while right_bound < len(slc) - 1 and slc[right_bound + 1] > thresh_val:
        right_bound += 1
    return left_bound, right_bound


def make_Wpg(Np: int, Ng: int, c: float = 0.5) -> np.ndarray:
    Wpg = randn(Np, Ng)
    prune = int((1 - c) * Np * Ng)
    mask = np.ones((Np, Ng))
    mask[
        randint(low=0, high=Np, size=prune),
        randint(low=0, high=Ng, size=prune),
    ] = 0
    return np.multiply(mask, Wpg)


def make_gbook_eff(lambdas: tuple[int, ...], fwhm_ratio: float) -> np.ndarray:
    """Fresh grid codebook; smooth when fwhm_ratio > 0."""
    Ng = int(np.sum(np.square(lambdas)))
    Npos = int(np.prod(lambdas))
    gbook = gen_gbook_2d(list(lambdas), Ng, Npos)
    if fwhm_ratio > 0:
        return smooth_gbook(gbook, lambdas, fwhm_ratio)
    return gbook


def cosine_slice_width(
    lambdas: tuple[int, ...],
    Np: int,
    thresh: float,
    fwhm_ratio: float,
    pbook: np.ndarray,
    *,
    env_size: int | None = None,
    rng_seed: int = 0,
) -> dict:
    Npos = int(np.prod(lambdas))
    if pbook.shape != (Np, Npos, Npos):
        raise ValueError(
            f"pbook shape {pbook.shape} != expected ({Np}, {Npos}, {Npos})"
        )
    if env_size is None:
        env_size = Npos

    cx, cy = Npos // 2, Npos // 2
    x0, x1 = cx - env_size // 2, cx + env_size // 2
    y0, y1 = cy - env_size // 2, cy + env_size // 2
    pbook_env = pbook[:, x0:x1, y0:y1]

    pop_vectors = pbook_env.reshape(Np, env_size * env_size).T
    norms = np.linalg.norm(pop_vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    pop_normed = pop_vectors / norms

    ref_x = env_size // 2
    ref_y = env_size // 2
    ref_idx = ref_y * env_size + ref_x
    cosine_map = (pop_normed @ pop_normed[ref_idx]).reshape(env_size, env_size)

    slc = cosine_map[ref_y, :]
    lb, rb = find_peak_bounds(slc, ref_x)
    width = int(rb - lb)

    return {
        "lambdas": str(lambdas),
        "Np": Np,
        "thresh": thresh,
        "fwhm_ratio": fwhm_ratio,
        "Npos": Npos,
        "env_size": env_size,
        "rng_seed": rng_seed,
        "cosine_width": width,
        "peak_x_lo": lb,
        "peak_x_hi": rb,
    }


def run_sweep(
    *,
    lambda_tuples: list[tuple[int, ...]],
    Np_list: list[int],
    thresh_list: list[float],
    fwhm_ratio_list: list[float],
    rng_seed: int,
    env_size: int | None,
    wpg_c: float,
) -> pd.DataFrame:
    n_total = len(lambda_tuples) * len(Np_list) * len(thresh_list) * len(fwhm_ratio_list)
    sweep_iter = itertools.product(
        lambda_tuples, Np_list, thresh_list, fwhm_ratio_list
    )

    rows: list[dict] = []
    for lambdas, Np, thresh, fwhm in tqdm(
        sweep_iter, total=n_total, desc="Cosine width sweep", unit="cfg"
    ):
        lt = tuple(int(x) for x in lambdas)
        Ng = int(np.sum(np.square(lt)))
        np.random.seed(rng_seed)
        g_eff = make_gbook_eff(lt, fwhm)
        Wpg = make_Wpg(Np, Ng, c=wpg_c)
        p_raw = train_pbook(Wpg, g_eff)
        pbook = nonlin(p_raw, thresh=thresh)
        rows.append(
            cosine_slice_width(
                lt, Np, thresh, fwhm, pbook, env_size=env_size, rng_seed=rng_seed
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("cosine_width_sweep.csv"),
        help="Output CSV path",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--env-size",
        type=int,
        default=None,
        help="Central crop size (default: full Npos)",
    )
    p.add_argument(
        "--wpg-c",
        type=float,
        default=0.5,
        help="Sparse mask strength for Wpg (match VectorHash default c)",
    )
    args = p.parse_args()

    df = run_sweep(
        lambda_tuples=list(DEFAULT_LAMBDA_TUPLES),
        Np_list=list(DEFAULT_NP_LIST),
        thresh_list=list(DEFAULT_THRESH_LIST),
        fwhm_ratio_list=list(DEFAULT_FWHM_RATIO_LIST),
        rng_seed=args.seed,
        env_size=args.env_size,
        wpg_c=args.wpg_c,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nWrote {args.output.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
