#!/usr/bin/env python3
"""Where are an encoder's far-field aliases, and what periodicity do they have?

``eval_unique_radius`` reports how *high* the alias ceiling is. This reports
*where* the offending positions are, which is the diagnostic that says which
degenerate solution a run fell into.

The reason this is worth asking: with lambdas (11, 12, 13) the code is unique
only over 11*12*13 = 1716 cells, but any *subset* of the modules is unique over
a much shorter period -- 132, 143 or 156 cells for a pair, 11/12/13 for one. An
encoder trained only on within-patch pairs is never told to separate positions
farther apart than a patch, so it is free to discard a module and keep a code
that repeats. If it does, the alias peaks land on a lattice whose spacing is one
of those numbers, and the residues below say exactly which modules survived.

Usage::

    python -m encoder_training.alias_structure <ckpt.pt> [more.pt ...]
    python -m encoder_training.alias_structure --untrained --seed 42
    python -m encoder_training.alias_structure --raw          # grid code itself
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from encoder_training.eval_unique_radius import (
    cosine_maps, grid_code_batch, npos_for, sample_references,
)


class _RawCode(torch.nn.Module):
    """Identity 'encoder': the smoothed grid code, L2-normalised.

    The ceiling any trained encoder is working against. It cannot be beaten by
    a deterministic function of the code, so it says how much of the radius a
    run threw away rather than failed to find.
    """

    def forward(self, x: torch.Tensor, gain: float | None = None) -> torch.Tensor:
        return torch.nn.functional.normalize(x, dim=-1)


def peak_offsets(
    cos_map: np.ndarray,
    ref: tuple[int, int],
    *,
    exclusion: int = 50,
    n_peaks: int = 12,
    suppress: int = 25,
) -> list[dict]:
    """The ``n_peaks`` highest far-field cells, greedily de-duplicated.

    A single alias is a broad blob, so the top-K raw cells all come from one of
    them; ``suppress`` blanks a square around each pick so the list reports
    distinct aliases rather than one alias K times.
    """
    Npos = cos_map.shape[0]
    m = cos_map.copy()
    gy, gx = np.ogrid[:Npos, :Npos]
    d2 = (gy - ref[0]) ** 2 + (gx - ref[1]) ** 2
    m[d2 <= exclusion ** 2] = -np.inf

    out = []
    for _ in range(n_peaks):
        flat = int(np.argmax(m))
        y, x = divmod(flat, Npos)
        val = float(m[y, x])
        if not np.isfinite(val):
            break
        dy, dx = int(y - ref[0]), int(x - ref[1])
        out.append({
            "cos": val, "dy": dy, "dx": dx,
            "dist": float(np.hypot(dy, dx)),
            # A perfect alias of a discarded module has residue 0 for every
            # module it kept, so a row of zeros names the surviving subset.
            "res": tuple(int(v) for v in
                         (min(abs(dy) % l, l - abs(dy) % l) for l in (11, 12, 13))),
            "res_x": tuple(int(v) for v in
                           (min(abs(dx) % l, l - abs(dx) % l) for l in (11, 12, 13))),
        })
        y0, y1 = max(0, y - suppress), min(Npos, y + suppress + 1)
        x0, x1 = max(0, x - suppress), min(Npos, x + suppress + 1)
        m[y0:y1, x0:x1] = -np.inf
    return out


def code_stats(encoder, lambdas, gain, *, n: int, device: str,
               fwhm_ratio: float, seed: int = 0) -> dict:
    """How many dimensions the code actually uses, over ``n`` random positions.

    The alias offsets say the two most *dissimilar* inputs in the arena land on
    nearly the same output. With a saturated tanh, cosine 0.98 over 1024 units
    means barely ten of them disagree — so the natural reading is that almost
    every unit has gone constant and the code lives in a handful of directions.
    That is a claim about the code, not about any pair, and this measures it:

    ``live_frac``
        fraction of coordinates whose std across positions clears 10% of the
        largest coordinate's std -- the ones still carrying position.
    ``pr``
        participation ratio ``(sum s_i)^2 / sum s_i^2`` of the covariance
        eigenvalues: the effective number of dimensions, 1 at full collapse and
        ``out_dim`` for an isotropic code.
    """
    from encoder_training.eval_unique_radius import _embed
    rng = np.random.default_rng(seed)
    Npos = npos_for(lambdas)
    xs = rng.integers(0, Npos, n)
    ys = rng.integers(0, Npos, n)
    z = _embed(encoder, lambdas, xs, ys, gain, device, fwhm_ratio).float()

    std = z.std(dim=0)
    live = float((std > 0.1 * std.max()).float().mean())
    zc = (z - z.mean(0, keepdim=True)).cpu().numpy()
    ev = np.linalg.svd(zc, compute_uv=False) ** 2
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    return {
        "out_dim": int(z.shape[1]),
        "live_frac": live,
        "pr": pr,
        "std_max": float(std.max()),
        "std_median": float(std.median()),
        # A saturated tanh makes the code binary; how binary says whether the
        # gain schedule, rather than the loss, is what flattened it.
        "frac_saturated": float((z.abs() * (z.shape[1] ** 0.5) > 0.95)
                                .float().mean()),
    }


def radial_profile(cos_map: np.ndarray, ref: tuple[int, int],
                   radii=(1, 2, 3, 5, 10, 20, 40, 80, 160, 320, 640)) -> dict:
    """Shell-mean cosine at a few radii: the decay curve in one row."""
    Npos = cos_map.shape[0]
    gy, gx = np.ogrid[:Npos, :Npos]
    d = np.sqrt((gy - ref[0]) ** 2 + (gx - ref[1]) ** 2)
    out = {}
    for r in radii:
        sel = (d >= r - 0.5) & (d < r + 0.5)
        out[r] = float(cos_map[sel].mean()) if sel.any() else float("nan")
    return out


def lattice_score(peaks: list[dict], periods=(11, 12, 13, 132, 143, 156)) -> dict:
    """Fraction of the top peaks that sit (within 2 cells) on each period's lattice.

    A score near 1 for period p means the aliases repeat every p cells, i.e. the
    encoder is behaving like a code with period p rather than 1716.
    """
    out = {}
    for p in periods:
        hits = 0
        for pk in peaks:
            ry = min(abs(pk["dy"]) % p, p - abs(pk["dy"]) % p)
            rx = min(abs(pk["dx"]) % p, p - abs(pk["dx"]) % p)
            if ry <= 2 and rx <= 2:
                hits += 1
        out[p] = hits / max(len(peaks), 1)
    return out


def report(name: str, encoder, lambdas, gain, *, n_refs: int, seed: int,
           device: str, fwhm_ratio: float, batch_size: int,
           n_code: int = 20000) -> None:
    refs = sample_references(lambdas, n_refs, border=100, seed=seed)
    maps = cosine_maps(encoder, lambdas, gain, refs, device, batch_size,
                       fwhm_ratio)
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"lambdas={lambdas} gain={gain:.2f} fwhm={fwhm_ratio} "
          f"Npos={npos_for(lambdas)} refs={n_refs} (seed {seed})")

    cs = code_stats(encoder, lambdas, gain, n=n_code, device=device,
                    fwhm_ratio=fwhm_ratio)
    print(f"\ncode: out_dim={cs['out_dim']}  live_frac={cs['live_frac']:.3f}  "
          f"participation_ratio={cs['pr']:.1f}  "
          f"std max/median={cs['std_max']:.4f}/{cs['std_median']:.4f}  "
          f"saturated={cs['frac_saturated']:.3f}")

    prof = radial_profile(maps[0], tuple(refs[0]))
    print("\nshell-mean cosine vs radius (ref 0):")
    print("  " + "  ".join(f"r{r}={v:+.3f}" for r, v in prof.items()))

    all_peaks: list[dict] = []
    for j in range(len(refs)):
        pk = peak_offsets(maps[j], tuple(refs[j]))
        all_peaks += pk
        if j < 2:
            print(f"\ntop far-field peaks, ref {j} at {tuple(refs[j])}:")
            for p in pk[:8]:
                print(f"  cos={p['cos']:.4f}  d=({p['dy']:+5d},{p['dx']:+5d})  "
                      f"|d|={p['dist']:7.1f}  res_y={p['res']} res_x={p['res_x']}")

    ceil = np.array([p["cos"] for p in all_peaks])
    dist = np.array([p["dist"] for p in all_peaks])
    print(f"\nover all {len(refs)} refs, top-12 peaks each:")
    print(f"  cos      max={ceil.max():.4f}  median={np.median(ceil):.4f}")
    print(f"  |offset| min={dist.min():7.1f}  median={np.median(dist):7.1f}  "
          f"max={dist.max():7.1f}")
    ls = lattice_score(all_peaks)
    print("  on-lattice fraction: " +
          "  ".join(f"p{p}={f:.2f}" for p, f in ls.items()))

    # How much of the arena is "confusable" at a few thresholds -- a scalar that
    # keeps meaning when the radius bottoms out at 0.
    print("  arena fraction above cos:", end="")
    for t in (0.5, 0.8, 0.9, 0.95):
        frac = float((maps > t).mean())
        print(f"  >{t}: {frac:.2e}", end="")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="*", type=Path)
    p.add_argument("--untrained", action="store_true",
                   help="also score a random-init encoder with the BASE arch")
    p.add_argument("--raw", action="store_true",
                   help="also score the smoothed grid code itself")
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--out_dim", type=int, default=1024)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--gain", type=float, default=5.0)
    p.add_argument("--fwhm_ratio", type=float, default=0.25)
    p.add_argument("--n_refs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init_seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16384)
    args = p.parse_args()

    if args.raw:
        report("RAW SMOOTHED GRID CODE (identity encoder)", _RawCode().to(args.device),
               args.lambdas, 1.0, n_refs=args.n_refs, seed=args.seed,
               device=args.device, fwhm_ratio=args.fwhm_ratio,
               batch_size=args.batch_size)

    if args.untrained:
        from encoder_training.config import EncoderModelConfig
        from encoder_training.models import create_encoder
        torch.manual_seed(args.init_seed)
        cfg = EncoderModelConfig(
            out_dim=args.out_dim, hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers, lambdas=args.lambdas,
            gain=args.gain)
        report(f"UNTRAINED (random init, seed {args.init_seed})",
               create_encoder(cfg, args.device), args.lambdas, args.gain,
               n_refs=args.n_refs, seed=args.seed, device=args.device,
               fwhm_ratio=args.fwhm_ratio, batch_size=args.batch_size)

    from encoder_training.train import load_encoder
    for path in args.ckpts:
        encoder, ckpt = load_encoder(str(path), device=args.device)
        lam = list(ckpt["model_config"]["lambdas"])
        fwhm = float((ckpt.get("train_config") or {}).get(
            "fwhm_ratio", args.fwhm_ratio))
        report(str(path), encoder, lam, float(ckpt["gain"]),
               n_refs=args.n_refs, seed=args.seed, device=args.device,
               fwhm_ratio=fwhm, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
