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
    nearly the same output, so the natural reading is that most units have gone
    constant and the code lives in a handful of directions. That is a claim
    about the code, not about any pair, and this measures it.

    NOTE, measured: the tanh is **not** saturated in any encoder here. The
    median ``|gain * net(x)|`` is 0.053 and no coordinate exceeds 0.8, so
    ``tanh`` is acting as an identity map and the code is continuous, not
    binary. An earlier version of this docstring assumed saturation and an
    earlier ``frac_saturated`` measured it *after* normalisation, where the
    rescaling by ``1/||z||`` made an unsaturated code look 34% saturated. Both
    are corrected. The consequence is that the ``gain`` ramp is close to inert:
    ``normalize(tanh(g*z))`` at g=1 and g=5 differ by cosine 0.9996 on the §6
    best encoder, since normalisation cancels a scalar and tanh is linear here.

    ``live_frac``
        fraction of coordinates whose std across positions clears 10% of the
        largest coordinate's std -- the ones still carrying position.
    ``pr``
        participation ratio ``(sum s_i)^2 / sum s_i^2`` of the covariance
        eigenvalues: the effective number of dimensions, 1 at full collapse and
        ``out_dim`` for an isotropic code.
    """
    from encoder_training.eval_unique_radius import embed
    rng = np.random.default_rng(seed)
    Npos = npos_for(lambdas)
    xs = rng.integers(0, Npos, n)
    ys = rng.integers(0, Npos, n)
    z = embed(encoder, lambdas, xs, ys, gain, device, fwhm_ratio).float()

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
        # Concentration relative to an isotropic code, which is what the
        # post-normalisation quantity actually measures: 1.0 for a code with
        # every coordinate at 1/sqrt(D), lower as the mass concentrates.
        # It is NOT tanh saturation -- see the note in the docstring.
        "frac_above_isotropic": float((z.abs() * (z.shape[1] ** 0.5) > 0.95)
                                      .float().mean()),
        # Tanh saturation, measured where it exists: before normalisation.
        # Reads ~0.000 on every encoder in this campaign.
        "frac_saturated_pre": _tanh_saturation(encoder, lambdas, gain, xs, ys,
                                               device, fwhm_ratio),
    }


def _tanh_saturation(encoder, lambdas, gain, xs, ys, device, fwhm_ratio,
                     thresh: float = 0.95) -> float:
    """Fraction of coordinates the output tanh has actually saturated.

    Has to be read before ``F.normalize``, which divides by ``||z||`` and so
    rescales an unsaturated code up to look saturated.
    """
    from encoder_training.eval_unique_radius import grid_code_batch
    net = getattr(encoder, "net", None)
    if net is None or getattr(encoder, "output_nonlinearity", "") != "tanh":
        return float("nan")
    phi = torch.as_tensor(grid_code_batch(lambdas, ys, xs, fwhm_ratio),
                          device=device)
    with torch.no_grad():
        pre = torch.tanh(gain * net(phi))
    return float((pre.abs() > thresh).float().mean())


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


def ref_vs_patch(encoder, ckpt, lambdas, gain, *, n_refs: int, seed: int,
                 device: str, fwhm_ratio: float) -> None:
    """Is the reference that sets ``r_min`` the one farthest from a training patch?

    ``r_min`` is the worst of 20 references, so it moves only when the *worst*
    one improves — and across the encoders measured so far the mean alias
    ceiling falls steadily with rank while the max barely does. That makes the
    identity of the bad reference the thing to know. If bad references are the
    ones the training patches never got near, coverage is the lever; if they are
    scattered, it is not, and the coverage wave is not worth its GPU hours.
    """
    from encoder_training.eval_unique_radius import evaluate_unique_radius

    y0s, x0s, sizes = ckpt.get("y0s"), ckpt.get("x0s"), ckpt.get("sizes")
    if not y0s:
        print("\n[no patch layout stored in checkpoint; skipping ref-vs-patch]")
        return
    records, _ = evaluate_unique_radius(
        encoder, lambdas=lambdas, gain=gain, n_refs=n_refs, border=100,
        seed=seed, device=device, fwhm_ratio=fwhm_ratio)

    rows = []
    for rec in records:
        rx, ry = rec["ref_x"], rec["ref_y"]
        # 0 inside a patch, else the L-infinity gap to the nearest one; the
        # patches are axis-aligned squares, so that is the natural distance.
        best = min(max(y0 - rx, 0, rx - (y0 + s - 1)) +
                   max(x0 - ry, 0, ry - (x0 + s - 1))
                   for y0, x0, s in zip(y0s, x0s, sizes))
        rows.append((rec["r_monotone_min"], rec["alias_ceiling"], best))

    rows.sort()
    print("\nper-reference radius vs distance to the nearest training patch:")
    print("   r_mono  alias   patch_dist")
    for r, a, d in rows:
        print(f"   {r:6.1f}  {a:.3f}  {d:>6}")
    r_arr = np.array([r for r, _, _ in rows], dtype=float)
    d_arr = np.array([d for _, _, d in rows], dtype=float)
    a_arr = np.array([a for _, a, _ in rows], dtype=float)
    if d_arr.std() > 0:
        print(f"   corr(patch_dist, r_mono) = "
              f"{np.corrcoef(d_arr, r_arr)[0, 1]:+.3f}   "
              f"corr(patch_dist, alias) = {np.corrcoef(d_arr, a_arr)[0, 1]:+.3f}")
    else:
        print("   every reference is inside a patch; distance says nothing here")


def alias_partner(encoder, ckpt, lambdas, gain, *, n_refs: int, seed: int,
                  device: str, fwhm_ratio: float, batch_size: int,
                  n_peaks: int = 5) -> None:
    """Is the position that aliases to a reference trained on, or is it not?

    §5.6e inferred, from the fact that being inside a training patch barely
    helps a reference, that what kills ``r_min`` is an aliasing *partner* drawn
    from the untrained part of the arena. That was an inference from a null
    result; this measures the partner directly.

    It decides something concrete. Every spread term available here —
    ``uniformity``, ``vicreg``, ``coding_rate`` — acts on the encodings of the
    batch, and the batch holds training points only. If the partners are
    untrained positions then no term in the loss has ever seen them, and no
    amount of tuning the spread strength can address them (which is what §5.6h
    found happening). If instead the partners are inside training patches, the
    loss *can* reach them and the failure is one of optimisation, not coverage.
    """
    y0s, x0s, sizes = ckpt.get("y0s"), ckpt.get("x0s"), ckpt.get("sizes")
    if not y0s:
        print("\n[no patch layout stored in checkpoint; skipping alias-partner]")
        return

    # env_id per cell, -1 outside every patch. The *pair* question needs the
    # identity, not just membership: under exclude_cross_env_pairs the repel
    # term is `~near & same_env`, so a (reference, alias-partner) pair enters
    # the loss only if both sit in the SAME patch.
    env_id = np.full((npos_for(lambdas),) * 2, -1, dtype=np.int32)
    for i, (y0, x0, s) in enumerate(zip(y0s, x0s, sizes)):
        env_id[y0:y0 + s, x0:x0 + s] = i
    inside_mask = env_id >= 0
    covered = inside_mask.mean()

    refs = sample_references(lambdas, n_refs, border=100, seed=seed)
    maps = cosine_maps(encoder, lambdas, gain, refs, device, batch_size,
                       fwhm_ratio)

    n_in = n_tot = n_ref_in = n_same_env = 0
    cos_in, cos_out = [], []
    for cos_map, ref in zip(maps, refs):
        ref_env = int(env_id[int(ref[0]), int(ref[1])])
        n_ref_in += ref_env >= 0
        for pk in peak_offsets(cos_map, tuple(ref), n_peaks=n_peaks):
            y, x = int(ref[0] + pk["dy"]), int(ref[1] + pk["dx"])
            pk_env = int(env_id[y, x])
            if pk_env >= 0:
                n_in += 1
                cos_in.append(pk["cos"])
            else:
                cos_out.append(pk["cos"])
            if ref_env >= 0 and pk_env == ref_env:
                n_same_env += 1
            n_tot += 1

    print(f"\nalias partners: where do the top-{n_peaks} far-field peaks live?")
    print(f"   arena covered by training patches : {covered:6.1%}")
    print(f"   references inside a patch         : {n_ref_in}/{len(refs)}")
    print(f"   peaks landing inside a patch      : {n_in}/{n_tot} "
          f"({n_in / max(n_tot, 1):6.1%})")
    print(f"   enrichment over chance            : "
          f"{(n_in / max(n_tot, 1)) / max(covered, 1e-9):6.2f}x")
    # The one that says whether the repel term could ever have acted on this
    # pair. Under exclude_cross_env_pairs, only same-patch pairs are repelled.
    print(f"   pairs in the SAME patch (repelled): {n_same_env}/{n_tot} "
          f"({n_same_env / max(n_tot, 1):6.2%})")
    if cos_in:
        print(f"   mean cos, partner inside patch    : {np.mean(cos_in):.3f}")
    if cos_out:
        print(f"   mean cos, partner outside         : {np.mean(cos_out):.3f}")


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
          f"above_isotropic={cs['frac_above_isotropic']:.3f}  "
          f"tanh_saturated={cs['frac_saturated_pre']:.3f}")

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
    p.add_argument("--ref_vs_patch", action="store_true",
                   help="per-reference radius against distance to the nearest "
                        "training patch -- says whether coverage is the lever "
                        "for the worst reference, which is what r_min reports")
    p.add_argument("--ur_refs", type=int, default=20,
                   help="references for --ref_vs_patch; 20 matches the metric")
    p.add_argument("--alias_partner", action="store_true",
                   help="do the far-field peaks that alias to a reference sit "
                        "inside a training patch or outside it -- says whether "
                        "the loss can reach them at all")
    p.add_argument("--partner_refs", type=int, default=40,
                   help="references for --alias_partner")
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
        if args.ref_vs_patch:
            ref_vs_patch(encoder, ckpt, lam, float(ckpt["gain"]),
                         n_refs=args.ur_refs, seed=args.seed,
                         device=args.device, fwhm_ratio=fwhm)
        if args.alias_partner:
            alias_partner(encoder, ckpt, lam, float(ckpt["gain"]),
                          n_refs=args.partner_refs, seed=args.seed,
                          device=args.device, fwhm_ratio=fwhm,
                          batch_size=args.batch_size)


if __name__ == "__main__":
    main()
