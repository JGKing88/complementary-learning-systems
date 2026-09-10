"""What kills the basin edge -- cross-talk, or local precision?

Sec 10.18 reports the basin as ``r_exact_all``: the radius within which every
cue retrieves the goal **cell** exactly, out of a bank of every cell in a
radius-64 disc plus the other stored goals. That is one number for what turn
out to be two structurally different failures:

  other_goal   the recalled state is nearest a DIFFERENT STORED GOAL. Cross-talk
               -- the far-field mechanism, governed by `d_eff`, and the same
               thing that sets the alias rate and therefore continuous reach.

  near         the recalled state is nearest another CELL of the disc, within
               NEAR_CELLS of the goal. No other memory is involved: the memory
               returned about the right thing and the argmax picked a
               neighbour. `d_eff` has no obvious claim on this.

They matter differently to navigation. The retrieved code is consumed as
``q = basis @ (z_goal - z_here)``, so retrieving the cell one north of the goal
is not a failed retrieval -- it is a target one cell off, which at r ~ 29 is a
~2 degree bearing error. An `other_goal` retrieval points at a different
environment entirely.

So this reports the same discs three ways:

  r_exact    first radius at which some cue misses the goal cell    (published)
  r_tol2/4   first radius at which some cue lands more than 2 / 4 cells away
  r_goal     first radius at which some cue lands on another stored goal

Skips any map whose offset arrays do not match its outcome arrays. `dx`/`dy`
were not filtered by the scaffold-edge clip until 2026-09-08, so 60 of the 140
maps recorded before that pair outcomes with the wrong offsets. They are
skipped, not repaired -- rerun the probe to replace them.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from cls_paths import results_dir

CATS = ("exact", "near", "far", "other_goal")
DEFAULT_DIR = os.path.join(
    str(results_dir()),
    "hopfield_probe/20260827/probe_ladder7")


def group_of(label: str) -> str:
    return label.split(" · ")[0]


def first_bad(r: np.ndarray, bad: np.ndarray) -> float:
    """Largest R with no `bad` cue at radius <= R. -1 if the goal cue is bad."""
    if not bad.any():
        return float(np.floor(r.max()))
    return float(np.floor(r[bad].min())) - 1.0


def load_maps(d: str, k: str):
    """(group, map, cat, cue radius, landing radius) for every usable map."""
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    order: list[str] = []
    rows: dict[str, list] = {}
    skipped = 0
    for e in man["encoders"]:
        g = group_of(e["label"])
        if g not in rows:
            order.append(g)
            rows[g] = []
        with open(os.path.join(d, e["file"])) as f:
            js = json.load(f)
        for bm in (js["test_a"]["k"][k].get("basin_maps") or []):
            cat = np.asarray(bm["cat"], dtype=int)
            dx = np.asarray(bm["dx"], dtype=float)
            if dx.size != cat.size:
                skipped += 1
                continue
            dy = np.asarray(bm["dy"], dtype=float)
            rdx = np.array([np.nan if v is None else v for v in bm["rdx"]])
            rdy = np.array([np.nan if v is None else v for v in bm["rdy"]])
            rows[g].append((bm, cat, np.hypot(dx, dy), np.hypot(rdx, rdy)))
    return order, rows, skipped


def composition(order, rows) -> None:
    hdr = (f"{'encoder':<24s}{'n':>3s}{'r_all':>7s}"
           f"{'| EDGE  near':>13s}{'far':>7s}{'other':>7s}{'n':>7s}"
           f"{'| ALL   near':>13s}{'far':>7s}{'other':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for g in order:
        ms = rows.get(g)
        if not ms:
            continue
        edge, allf, ralls = np.zeros(4), np.zeros(4), []
        for bm, cat, r, _land in ms:
            ra = bm["r_exact_all"]
            ralls.append(ra)
            fail = cat != 0
            for i in range(1, 4):
                allf[i] += np.sum(cat[fail] == i)
            if ra is not None and ra >= 0:
                m = (r >= ra) & (r < ra + 4) & fail
                for i in range(1, 4):
                    edge[i] += np.sum(cat[m] == i)
        e, a = edge / max(edge.sum(), 1), allf / max(allf.sum(), 1)
        print(f"{g:<24s}{len(ms):>3d}{np.median(ralls):>7.1f}"
              f"{e[1]:>13.3f}{e[2]:>7.3f}{e[3]:>7.3f}{int(edge.sum()):>7d}"
              f"{a[1]:>13.3f}{a[2]:>7.3f}{a[3]:>7.3f}")


def tolerance(order, rows) -> None:
    hdr = (f"{'encoder':<24s}{'n':>3s}{'r_exact':>9s}{'r_tol2':>9s}"
           f"{'r_tol4':>9s}{'r_goal':>9s}{'tol2/exact':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for g in order:
        ms = rows.get(g)
        if not ms:
            continue
        cols: list[list[float]] = [[], [], [], []]
        for bm, _cat, r, land in ms:
            other = np.isnan(land)          # landed on another stored goal
            cols[0].append(bm["r_exact_all"])
            cols[1].append(first_bad(r, other | (land > 2)))
            cols[2].append(first_bad(r, other | (land > 4)))
            cols[3].append(first_bad(r, other))
        m = [float(np.median(c)) for c in cols]
        print(f"{g:<24s}{len(ms):>3d}" + "".join(f"{v:>9.1f}" for v in m)
              + f"{m[1] / max(m[0], 1e-9):>12.2f}")


def paired(order, rows) -> None:
    """`r_exact` against `r_goal` map by map, not median against median.

    The two columns can have the same median while no single map has the same
    pair, so the gap that matters -- how much exact-cell basin a given goal
    loses to misses that are not memory failures -- has to be read paired.
    """
    print(f"{'encoder':<24s}{'r_exact -> r_goal, per map':<42s}"
          f"{'median gap':>11s}")
    print("-" * 77)
    for g in order:
        ms = rows.get(g)
        if not ms:
            continue
        pairs, gaps = [], []
        for bm, _cat, r, land in ms:
            ex = bm["r_exact_all"]
            rg = first_bad(r, np.isnan(land))
            pairs.append(f"{ex:.0f}->{rg:.0f}")
            gaps.append(rg - ex)
        print(f"{g:<24s}{'  '.join(pairs):<42s}{np.median(gaps):>11.1f}")


def by_radius(order, rows, group: str) -> None:
    ms = rows.get(group) or []
    if not ms:
        return
    print(f"\n{group} -- composition against radius (failures only)")
    print(f"  {'band':<10s}{'near':>9s}{'far':>9s}{'other_goal':>12s}"
          f"{'fail rate':>11s}{'n':>8s}")
    for lo, hi in [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48),
                   (48, 56), (56, 64.5)]:
        num, tot = np.zeros(4), 0
        for _bm, cat, r, _land in ms:
            m = (r >= lo) & (r < hi)
            tot += int(m.sum())
            for i in range(4):
                num[i] += np.sum(cat[m] == i)
        f = num[1:].sum()
        print(f"  {f'{lo}-{hi}':<10s}"
              + "".join(f"{num[i] / max(f, 1):>9.3f}" for i in (1, 2))
              + f"{num[3] / max(f, 1):>12.3f}"
              + f"{f / max(tot, 1):>11.3f}{tot:>8d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=DEFAULT_DIR,
                    help="a probe output directory carrying a manifest.json")
    ap.add_argument("--k", default="5", help="K to read the maps at")
    args = ap.parse_args()

    order, rows, skipped = load_maps(args.dir, args.k)
    n = sum(len(v) for v in rows.values())
    print(f"{args.dir}\nK={args.k}   maps used {n}, "
          f"skipped for the offset bug {skipped}\n")
    composition(order, rows)
    print()
    tolerance(order, rows)
    print()
    paired(order, rows)
    if order:
        by_radius(order, rows, order[0])


if __name__ == "__main__":
    main()
