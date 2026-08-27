"""Evaluate an encoder's unique coding radius.

This is the evaluation mechanism, not the sweep. It takes a live encoder
module, so it can be called from inside a training loop as easily as from the
offline driver in ``sweep_unique_radius`` -- nothing here knows about
checkpoint files or CSVs.

Why the grid codes are generated lazily
---------------------------------------
``gridcode.gen_gbook_2d`` materialises ``(Ng, Npos, Npos)``. At the default
lambdas (11, 12, 13) that is 434 x 1716 x 1716 float64 = 10.2 GB for a codebook
whose every column is one-hot per module and trivially recomputable. This
module builds the codes a batch at a time instead, so peak memory is set by the
cosine maps -- ``(n_refs, Npos, Npos)`` float32, 235 MB at 20 references --
rather than by the codebook or the embeddings.

Why the reference embeddings go to the GPU and the embeddings never come back
----------------------------------------------------------------------------
Only the 20 cosine values per position are needed, not the embedding. Reducing
on the device turns a 12 GB device-to-host transfer (2.94M x embed_dim) into
235 MB.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from encoder_training.unique_radius import (
    DEFAULT_MARGIN_RADII, DEFAULT_PROFILE_LEVELS, DEFAULT_TRIMS, HEADLINE_TRIM,
    unique_radius_report,
)

DEFAULT_N_REFS = 20
DEFAULT_BORDER = 100
DEFAULT_FWHM_RATIO = 0.25       # the default in data.py and evaluate_nav.py


def npos_for(lambdas: Sequence[int]) -> int:
    """Positions per axis: the period of the combined code."""
    return int(np.prod(np.asarray(lambdas, dtype=int)))


def sample_references(
    lambdas: Sequence[int],
    n_refs: int = DEFAULT_N_REFS,
    border: int = DEFAULT_BORDER,
    seed: int = 0,
) -> np.ndarray:
    """``(n_refs, 2)`` integer positions at least ``border`` cells from any edge.

    Seeded independently of the encoder so that every encoder in a sweep is
    scored at the *same* positions: the comparison is paired, which removes
    between-encoder variance that has nothing to do with the encoders.
    """
    Npos = npos_for(lambdas)
    lo, hi = border, Npos - 1 - border
    if hi <= lo:
        raise ValueError(
            f"border={border} leaves no interior for Npos={Npos} "
            f"(need Npos > 2*border + 1)")
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(lo, hi + 1, n_refs),
                     rng.integers(lo, hi + 1, n_refs)], axis=1).astype(np.int64)


def grid_code_batch(lambdas: Sequence[int], xs: np.ndarray, ys: np.ndarray,
                    fwhm_ratio: float = DEFAULT_FWHM_RATIO,
                    dtype=np.float32) -> np.ndarray:
    """``(len(xs), in_dim)`` grid codes: ``gen_gbook_2d`` then ``smooth_gbook``.

    The smoothing is not optional. Raw one-hot codes of adjacent positions are
    *disjoint* -- x % 11, x % 12 and x % 13 all change when x moves by one --
    so an encoder fed them sees no neighbourhood structure at all and every
    similarity map collapses within a single cell. Both ``data.build_full_grid``
    (training) and ``evaluate_nav`` smooth with the checkpoint's
    ``train_config['fwhm_ratio']``, and this must match or the encoder is being
    evaluated off-distribution.

    Reproduced lazily rather than by calling ``smooth_gbook``, which needs the
    whole 10.2 GB codebook. Within a module the smoothed block is one canonical
    wrapped-Gaussian bump shifted to the active phase, and the bump is separable
    in the two phase axes, so each position's block is an outer product of two
    length-lambda factors.
    """
    lam = np.asarray(lambdas, dtype=int)
    in_dim = int((lam ** 2).sum())
    offsets = np.zeros(len(lam), dtype=int)
    offsets[1:] = np.cumsum(lam[:-1] ** 2)

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n = len(xs)
    out = np.zeros((n, in_dim), dtype=dtype)

    for m, per in enumerate(lam):
        off, l = int(offsets[m]), int(per)
        cy, cx = xs % l, ys % l
        if fwhm_ratio <= 0:
            out[np.arange(n), off + cy * l + cx] = 1.0
            continue
        sigma = (l * fwhm_ratio) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        axis = np.arange(l)
        dy = np.abs(axis[None, :] - cy[:, None])
        dy = np.minimum(dy, l - dy)                      # phases wrap
        dx = np.abs(axis[None, :] - cx[:, None])
        dx = np.minimum(dx, l - dx)
        fy = np.exp(-0.5 * (dy / sigma) ** 2)            # (n, l)
        fx = np.exp(-0.5 * (dx / sigma) ** 2)            # (n, l)
        out[:, off:off + l * l] = (fy[:, :, None] * fx[:, None, :]).reshape(n, l * l)
    return out


def embed(encoder, lambdas, xs, ys, gain, device, fwhm_ratio) -> torch.Tensor:
    codes = torch.from_numpy(
        grid_code_batch(lambdas, xs, ys, fwhm_ratio)).to(device)
    with torch.no_grad():
        z = encoder(codes, gain=gain)
    return torch.nn.functional.normalize(z.float(), dim=1, eps=1e-12)


# Back-compat: this was `_embed` until the layering test caught
# alias_structure importing it across module boundaries. Kept so any
# out-of-tree caller of the private name keeps working; new code should use
# `embed`, which is exported.
_embed = embed


def cosine_maps(
    encoder,
    lambdas: Sequence[int],
    gain: float,
    refs: np.ndarray,
    device: torch.device | str = "cuda",
    batch_size: int = 16384,
    fwhm_ratio: float = DEFAULT_FWHM_RATIO,
) -> np.ndarray:
    """``(n_refs, Npos, Npos)`` float32 cosine maps, indexed ``[ref, gx, gy]``."""
    device = torch.device(device)
    Npos = npos_for(lambdas)
    total = Npos * Npos

    was_training = encoder.training
    encoder.eval()
    try:
        z_ref = embed(encoder, lambdas, refs[:, 0], refs[:, 1], gain, device,
                       fwhm_ratio)
        cos = np.empty((len(refs), total), dtype=np.float32)
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            idx = np.arange(start, stop)
            z = embed(encoder, lambdas, idx // Npos, idx % Npos, gain, device,
                       fwhm_ratio)
            cos[:, start:stop] = (z @ z_ref.T).T.cpu().numpy()
    finally:
        if was_training:
            encoder.train()
    return cos.reshape(len(refs), Npos, Npos)


def evaluate_unique_radius(
    encoder,
    *,
    lambdas: Sequence[int],
    gain: float,
    n_refs: int = DEFAULT_N_REFS,
    border: int = DEFAULT_BORDER,
    seed: int = 0,
    trims: Sequence[int] = DEFAULT_TRIMS,
    headline_trim: int = HEADLINE_TRIM,
    margin_radii: Sequence[int] = DEFAULT_MARGIN_RADII,
    profile_levels: Sequence[float] = DEFAULT_PROFILE_LEVELS,
    device: torch.device | str = "cuda",
    batch_size: int = 16384,
    fwhm_ratio: float = DEFAULT_FWHM_RATIO,
) -> tuple[list[dict], dict]:
    """Score one encoder. Returns (per-reference records, summary).

    ``fwhm_ratio`` must be the one the encoder was trained with -- take it from
    ``ckpt['train_config']``, as ``evaluate_nav`` does. Evaluating with the
    wrong value feeds the encoder inputs it has never seen.

    Every reference is capped at the same ``max_r = border`` rather than at its
    own distance to the edge. References land at different depths in the arena,
    so per-reference ceilings would make the summary's min depend on which
    references happened to fall near the middle; a common cap keeps the 20
    numbers commensurable. Any radius below ``border`` is exact regardless.
    """
    refs = sample_references(lambdas, n_refs, border, seed)
    maps = cosine_maps(encoder, lambdas, gain, refs, device, batch_size,
                       fwhm_ratio)

    records = []
    for j, (rx, ry) in enumerate(refs):
        rep = unique_radius_report(
            maps[j], float(rx), float(ry),
            trims=trims, headline_trim=headline_trim,
            margin_radii=margin_radii, profile_levels=profile_levels,
            max_r=float(border),
        )
        rep["ref_index"] = j
        rep["ref_x"], rep["ref_y"] = int(rx), int(ry)
        records.append(rep)

    return records, summarize(records, trims=trims, headline_trim=headline_trim,
                              margin_radii=margin_radii,
                              profile_levels=profile_levels)


def _agg(fn, values) -> float:
    """NaN-tolerant aggregate that stays quiet when *everything* is NaN.

    ``r_at_cos{level}`` is NaN whenever the profile never falls to that level,
    which is a legitimate outcome for a broad code, not a numerical problem --
    numpy's nan-aggregates warn on an all-NaN slice regardless.
    """
    values = np.asarray(values, dtype=float)
    return float(fn(values)) if np.any(np.isfinite(values)) else float("nan")


def summarize(
    records: list[dict],
    *,
    trims: Sequence[int] = DEFAULT_TRIMS,
    headline_trim: int = HEADLINE_TRIM,
    margin_radii: Sequence[int] = DEFAULT_MARGIN_RADII,
    profile_levels: Sequence[float] = DEFAULT_PROFILE_LEVELS,
) -> dict:
    """Collapse per-reference records to one row.

    ``r_min`` -- the worst of the sampled locations -- is the headline: the
    arena's guarantee is set by its weakest position, not its typical one.

    The headline is the **per-direction** radius (``r_monotone_min``), not the
    disc radius. The disc condition compares the worst cell inside r against
    the best cell just outside it, and those lie in different directions, so it
    silently demands circular level sets: a 1.25:1 ellipse drives it to 1 while
    every ray stays monotone for hundreds of cells. Its columns are kept as
    ``disc_*`` for reference but should not be ranked on.
    """
    key = "r_monotone_min" if "r_monotone_min" in records[0] \
        else f"r_trim{headline_trim}"
    head = np.array([r[key] for r in records], dtype=float)
    out = {
        "n_refs": len(records),
        "headline": key,
        "headline_trim": int(headline_trim),
        "r_min": float(head.min()),
        "r_p25": float(np.percentile(head, 25)),
        "r_median": float(np.median(head)),
        "r_mean": float(head.mean()),
        "r_max": float(head.max()),
        "r_std": float(head.std()),
        "n_saturated": int(sum(r[f"saturated_trim{headline_trim}"]
                               for r in records)),
    }
    disc = np.array([r[f"r_trim{headline_trim}"] for r in records], dtype=float)
    out["disc_min"] = float(disc.min())
    out["disc_median"] = float(np.median(disc))
    for t in trims:
        vals = np.array([r[f"r_trim{t}"] for r in records], dtype=float)
        out[f"r_min_trim{t}"] = float(vals.min())
        out[f"r_median_trim{t}"] = float(np.median(vals))

    ceil = np.array([r["alias_ceiling"] for r in records], dtype=float)
    out["alias_ceiling_max"] = float(ceil.max())
    out["alias_ceiling_mean"] = float(ceil.mean())

    # The anisotropy-tolerant radii. ``r_min`` above is the disc statistic and
    # is destroyed by any departure from circular level sets, so these are the
    # ones to rank on; see unique_radius.unique_radius_report.
    for key, label in (("r_alias", "alias"), ("r_monotone_min", "mono"),
                       ("r_monotone_median", "mono_med")):
        if key not in records[0]:
            continue
        vals = np.array([r[key] for r in records], dtype=float)
        out[f"{label}_min"] = float(vals.min())
        out[f"{label}_median"] = float(np.median(vals))
        out[f"{label}_max"] = float(vals.max())
    if "far_ceiling" in records[0]:
        far = np.array([r["far_ceiling"] for r in records], dtype=float)
        out["far_ceiling_max"] = float(far.max())
        out["far_ceiling_mean"] = float(far.mean())
        out["n_saturated_alias"] = int(sum(r["saturated_alias"] for r in records))
        out["n_rays"] = int(records[0].get("n_rays", 0))

    for R in margin_radii:
        vals = [r[f"margin_r{R}"] for r in records]
        out[f"margin_r{R}_min"] = _agg(np.nanmin, vals)
        out[f"margin_r{R}_mean"] = _agg(np.nanmean, vals)
    for lvl in profile_levels:
        out[f"r_at_cos{lvl}_median"] = _agg(
            np.nanmedian, [r[f"r_at_cos{lvl}"] for r in records])
    out["cos_floor_mean"] = _agg(np.nanmean, [r["cos_floor"] for r in records])
    return out


__all__ = [
    "DEFAULT_N_REFS",
    "DEFAULT_BORDER",
    "npos_for",
    "sample_references",
    "grid_code_batch",
    "embed",
    "cosine_maps",
    "evaluate_unique_radius",
    "summarize",
]
