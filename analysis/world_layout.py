"""Draw a generated world: where its envs sit on the whole scaffold.

The env generator (``hopfield_nav/world/generate.py``) enforces separation
numerically -- ``verify_split`` asserts it and ``split_diagnostics`` reports the
minimum achieved gap -- but a number cannot say whether a *region* is being used
the way it was declared, whether a packing is comfortable or on the edge of what
the margin allows, or what a place-OOD complement would actually have left to
sample from. This renders the layout so those are visible.

Two things the picture has to get right, because both are easy to draw wrongly:

**The forbidden zone is a square, and it is a square around the offset.**
Separation is ``max`` over axes -- two footprints are apart as soon as *one* axis
separates them -- so env B is illegal exactly when ``|dx| < size + margin`` **and**
``|dy| < size + margin``. That region is a square of side ``2*(size + margin)``
centred on A's offset, not a disc and not the footprint dilated by the margin.

**The scaffold is a torus** when ``Npos == prod(lambdas)``, which is the working
config. A halo crossing an edge reappears on the far side; drawn flat it would
show clearance that is not there. The measured worst case -- ``(1715, 987)`` and
``(4, 989)``, 1711 cells apart flat and 5 apart in truth -- is exactly this.

Usage:
    python -m analysis.world_layout --seeds 0 1 2
    python -m analysis.world_layout --configs working rect --refresh 20
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from cls_paths import figures_dir
from hopfield_nav.config import EnvConfig, VectorHashConfig
from hopfield_nav.training.refresh import Cadence, Refresher
from hopfield_nav.world import domains as dom
from hopfield_nav.world import generate as gen
from hopfield_nav.world.scaffold import VectorHash
from hopfield_nav.world.spec import TraitDomains
from hopfield_nav.world.world import build_world

LAMBDAS = [11, 12, 13]

TRAIN_C = "#2166ac"
VAL_C = "#d6604d"
REGION_C = "#1a9850"

# Each entry is one row of the figure. `margin` is given rather than derived:
# `derive_margin` reads the scaffold's cosine-vs-distance curve and so needs the
# real encoder, while placement itself is a pure function of coordinates. 80 is
# the value it returns at lambdas=11,12,13, fwhm_ratio=0.25 (q99 < 0.15).
CONFIGS = {
    "working": dict(
        label="the working config — place=anywhere, size 8, margin 80, "
              "80 train + 10 val",
        place=dom.Anywhere(), size=8, margin=80, n_train=80, n_val=10),
    "rect": dict(
        label="a declared region — place=rect:200,200,700,700, 40 train + 8 val "
              "(everything outside the dashed box is the place-OOD set)",
        place=dom.Rect(200, 200, 700, 700), size=8, margin=80,
        n_train=40, n_val=8),
    "dense": dict(
        label="halved margin, many more envs — margin 40, 200 train + 20 val",
        place=dom.Anywhere(), size=8, margin=40, n_train=200, n_val=20),
    "big": dict(
        label="larger arenas — size 20, margin 80, 60 train + 10 val",
        place=dom.Anywhere(), size=20, margin=80, n_train=60, n_val=10),
}


def _field():
    """The scaffold, without ``encoded_Phi``.

    Placement reads only ``Npos`` and ``lambdas``; the 12 GB embedding table is
    needed for ``derive_margin`` and the cosine diagnostic, neither of which a
    layout figure uses. Skipping it is what makes this run in a second.
    """
    vh = VectorHash(VectorHashConfig(lambdas=LAMBDAS, Np=400,
                                     static_vectorhash=True))
    vh.build_scaffold()
    return vh


def _wrapped(x: float, y: float, w: float, h: float, period: int):
    """``(x, y)`` boxes covering a footprint that may run off the torus edge."""
    out = []
    for dx in (-period, 0, period):
        for dy in (-period, 0, period):
            if x + dx < period and x + dx + w > 0 and \
               y + dy < period and y + dy + h > 0:
                out.append((x + dx, y + dy))
    return out


def _draw_envs(ax, specs, period, margin, color, *, halo=True, lw=1.4,
               grow=0.0, zorder=3, alpha=1.0):
    """Footprints, plus the square each one forbids to every other offset.

    ``grow`` inflates the drawn footprint only. At size 8 on a 1716 scaffold a
    true-scale footprint is a third of a pixel, so the validation envs -- the
    thing the figure exists to locate -- would be invisible at their real size.
    The halo is always drawn to scale, because its size is the claim being made.
    """
    for s in specs:
        if halo:
            # The forbidden square: no other offset may land within
            # (size + margin) of this one on *both* axes.
            side = 2 * (s.size + margin)
            for bx, by in _wrapped(s.offset[0] - s.size - margin,
                                   s.offset[1] - s.size - margin,
                                   side, side, period):
                ax.add_patch(Rectangle((bx, by), side, side, facecolor=color,
                                       alpha=0.055, edgecolor="none", zorder=1))
        w = s.size + grow
        for bx, by in _wrapped(s.offset[0] - grow / 2, s.offset[1] - grow / 2,
                               w, w, period):
            ax.add_patch(Rectangle((bx, by), w, w, facecolor=color, alpha=alpha,
                                   edgecolor=color, linewidth=lw, zorder=zorder))


def _min_gap(specs, period):
    return min((gen.toroidal_gap(a.offset, a.size, b.offset, b.size, period)
                for i, a in enumerate(specs) for b in specs[i + 1:]),
               default=None)


def _panel(ax, field, cfg, seed, refresh_ticks):
    period = int(np.prod(field.lambdas))
    domains = TraitDomains(place=cfg["place"], wall=dom.SeedRange(0, 10_000_000),
                           goal=dom.AnyCells(), size=dom.Sizes((cfg["size"],)))
    env_cfg = EnvConfig(size=cfg["size"], observation_size=12)

    # Which placement regime a config lands in is itself the interesting fact:
    # rejection sampling gives an irregular scatter, the lattice fallback a
    # visible grid. Reported rather than left to be inferred from the picture.
    hits = [0]
    orig = gen._lattice_places
    gen._lattice_places = lambda *a, **k: (hits.__setitem__(0, hits[0] + 1),
                                           orig(*a, **k))[1]
    try:
        split = gen.generate_split(field, env_cfg, domains, cfg["n_train"],
                                   cfg["n_val"], seed=seed, margin=cfg["margin"],
                                   refresh_goal=refresh_ticks > 0,
                                   diagnostics=False)
    finally:
        gen._lattice_places = orig

    history = []
    if refresh_ticks:
        envs = gen.build_envs(split.train, env_cfg, "discrete")
        worlds = [build_world(field, envs,
                              offsets=[s.offset for s in split.train])]
        r = Refresher(Cadence(place=1), split, worlds, env_cfg, "discrete", seed)
        for tick in range(1, refresh_ticks + 1):
            r.maybe_refresh(tick)
            history.extend(split.train)

    if isinstance(cfg["place"], dom.Rect):
        rr = cfg["place"]
        ax.add_patch(Rectangle((rr.x0, rr.y0), rr.w, rr.h, facecolor="none",
                               edgecolor=REGION_C, linewidth=1.6,
                               linestyle="--", zorder=2))

    grow = max(0.0, 14.0 - cfg["size"])       # see _draw_envs on why
    if history:
        # Every placement the run ever used, faint, with the final draw solid on
        # top. This is `split.used["place"]` -- the set a later held-out val env
        # has to clear, which is the whole reason refresh records anything.
        _draw_envs(ax, history, period, cfg["margin"], TRAIN_C, halo=False,
                   lw=0.0, grow=grow, zorder=2, alpha=0.18)
    _draw_envs(ax, split.train, period, cfg["margin"], TRAIN_C, grow=grow)
    _draw_envs(ax, split.base_val, period, cfg["margin"], VAL_C,
               grow=grow + 10, lw=2.0, zorder=4)

    ax.set_xlim(0, period)
    ax.set_ylim(0, period)
    ax.set_aspect("equal")
    ax.set_xticks([0, period // 2, period])
    ax.set_yticks([0, period // 2, period])
    ax.tick_params(labelsize=7)
    gap = _min_gap(split.train + split.base_val, period)
    cap = cfg["place"].capacity(cfg["size"], cfg["margin"], field.Npos)
    how = "lattice" if hits[0] else "rejection"
    used = f"\n{len(split.used_offsets())} distinct offsets used" if history else ""
    ax.set_title(f"seed {seed}  ·  min gap {gap}  ·  ~{cap} slots  ·  {how}"
                 f"{used}", fontsize=8)
    return dict(seed=seed, min_gap=gap, capacity=cap, how=how,
                n_used=len(split.used_offsets()))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS),
                   choices=list(CONFIGS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--refresh", type=int, default=0,
                   help="Also run this many --refresh_place ticks and draw the "
                        "union of every placement, faint, under the final draw.")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    field = _field()
    rows, cols = len(args.configs), len(args.seeds)
    # One subfigure per config, so a row's caption belongs to the row instead of
    # being crammed into the leftmost panel's y-label.
    # A blank strip at the top and bottom holds the overall title and the
    # legend; `subplots_adjust` has no effect once subfigures are in play.
    fig = plt.figure(figsize=(3.6 * cols, 4.0 * rows + 1.3))
    strips = fig.subfigures(rows + 2, 1, hspace=0.05,
                            height_ratios=[0.22] + [1] * rows + [0.11])
    subfigs = strips[1:-1]

    for r, name in enumerate(args.configs):
        cfg = CONFIGS[name]
        axes = subfigs[r].subplots(1, cols, squeeze=False)[0]
        for c, seed in enumerate(args.seeds):
            stats = _panel(axes[c], field, cfg, seed, args.refresh)
            print(f"  {name:8s} seed={stats['seed']}  min_gap={stats['min_gap']}"
                  f"  slots~{stats['capacity']}  {stats['how']}"
                  f"  n_used={stats['n_used']}", flush=True)
        subfigs[r].suptitle(f"{chr(65 + r)}.  {cfg['label']}", fontsize=10,
                            x=0.02, ha="left")

    handles = [Rectangle((0, 0), 1, 1, facecolor=TRAIN_C, edgecolor=TRAIN_C),
               Rectangle((0, 0), 1, 1, facecolor=VAL_C, edgecolor=VAL_C),
               Rectangle((0, 0), 1, 1, facecolor=TRAIN_C, alpha=0.15,
                         edgecolor="none"),
               Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=REGION_C,
                         linestyle="--")]
    labels = ["train env", "validation env",
              "offsets this env forbids (side 2·(size+margin))",
              "declared place region"]
    strips[-1].legend(handles, labels, loc="center", ncol=4, fontsize=9,
                      frameon=False)

    sub = ("Footprints are drawn oversized to be visible — at size 8 on a 1716 "
           "scaffold one is a third of a pixel.\nThe halos are to scale, and "
           "wrap at the torus seam.")
    if args.refresh:
        sub += (f"  Faint blue: every placement across {args.refresh} "
                "--refresh_place ticks.")
    strips[0].suptitle(f"Generated worlds on the lambdas={LAMBDAS} scaffold "
                       f"(Npos={field.Npos})\n{sub}", fontsize=10.5)

    out = args.out or str(figures_dir(ensure=True) / "world_layout.png")
    fig.savefig(out, dpi=140)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out} and .pdf", flush=True)


if __name__ == "__main__":
    main()
