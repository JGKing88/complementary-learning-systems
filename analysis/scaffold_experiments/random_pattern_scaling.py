"""How the sensory->place->grid association scales with completely random patterns.

Read-only probe: builds its own scaffold, touches nothing in hopfield_nav.

WHAT IS MEASURED
    Exact grid-state recovery, the same criterion ``scaffold._test_assoc`` gates
    on: run obs -> Wps -> p -> Wgp -> module-wise winner-take-all -> g, and count
    a pattern correct only when g matches the stored grid state in *every*
    module. With lambdas=[11,12] the modules are one-hot of size 121 and 144, so
    chance is 1/(121*144) ~ 0.006%. There is no partial credit.

WHY RANDOM PATTERNS
    ``Wps`` is fitted by pseudoinverse, so it is a linear map from an Ns-
    dimensional sensory space. Random +/-1 patterns in high dimension are close
    to orthogonal, which is the *best* case such a map can be handed: while the
    stored patterns stay linearly independent -- i.e. while Npatts <= Ns -- the
    pseudoinverse is an exact left inverse and recall should be perfect. Past
    that the system is over-determined and least squares starts trading error
    across patterns.

    So the random curve is the ceiling. Anything structured -- like the real
    ray-cast wall codes, where neighbouring cells see overlapping walls and the
    patterns are strongly correlated -- can only do worse at the same load, and
    the gap between the two curves is the price of that correlation.

    The x-axis is therefore *load* = Npatts / Ns, which is where the interesting
    boundary sits, rather than Npatts alone.

Usage:
    python -m analysis.scaffold_experiments.random_pattern_scaling
    python -m analysis.scaffold_experiments.random_pattern_scaling --quick
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from gridcode.assoc import pseudotrain_Wps, pseudotrain_Wsp
from hopfield_nav.config import EnvConfig, VectorHashConfig
from hopfield_nav.world.env import make_env
from hopfield_nav.world.scaffold import EnvAssoc, VectorHash

LAMBDAS = [11, 12]
NP = 1600


def build_field(seed: int) -> VectorHash:
    # build_scaffold draws pbook from the *global* numpy stream (scaffold.py
    # binds `_randn = np.random.randn`), so the seed has to be set globally or
    # arms silently get different scaffolds and the comparison is meaningless.
    np.random.seed(seed)
    field = VectorHash(VectorHashConfig(lambdas=list(LAMBDAS), Np=NP))
    field.build_scaffold()
    return field


def targets(field, npatts, rng):
    """``npatts`` distinct scaffold positions, as (pbook, gbook) columns."""
    flat = rng.choice(field.Npos * field.Npos, size=npatts, replace=False)
    xs, ys = np.divmod(flat, field.Npos)
    pb = np.stack([field.pbook[:, x, y] for x, y in zip(xs, ys)], axis=1)
    gb = np.stack([field.gbook[:, x, y] for x, y in zip(xs, ys)], axis=1)
    return pb, gb


def real_sbook(npatts, observation_size, rng, size=8):
    """Real ray-cast North-facing views, drawn from as many envs as needed."""
    need = -(-npatts // (size * size))
    cols = []
    for i in range(need):
        env = make_env(EnvConfig(size=size, observation_size=observation_size),
                       "discrete", seed=1000 + i)
        for x in range(size):
            for y in range(size):
                cols.append(env._codebook[x, y, 0])
    cols = np.array(cols)
    return cols[rng.choice(len(cols), npatts, replace=False)].T


def real_views_sbook(n_loc, k, observation_size, rng, size=8):
    """``k`` real cardinal views for each of ``n_loc`` cells, grouped by cell.

    Column order is cell-major -- cell 0's k views, then cell 1's -- matching
    ``np.repeat`` on the target books.
    """
    need = -(-n_loc // (size * size))
    cells = []
    for i in range(need):
        env = make_env(EnvConfig(size=size, observation_size=observation_size),
                       "discrete", seed=1000 + i)
        for x in range(size):
            for y in range(size):
                cells.append(env._codebook[x, y, :k])       # (k, obs)
    cells = np.array(cells)
    pick = rng.choice(len(cells), n_loc, replace=False)
    return cells[pick].reshape(-1, observation_size).T      # (obs, n_loc*k)


def recovery(field, sbook, pb, gb):
    """Fraction of patterns whose recall lands on the exact grid state."""
    npatts = sbook.shape[1]
    assoc = EnvAssoc(field, pseudotrain_Wsp(sbook, pb, npatts),
                     pseudotrain_Wps(pb, sbook, npatts), Ns=sbook.shape[0])
    _, _, g = assoc.recall_batch(sbook.T)
    return float((g.T == gb).all(axis=0).mean())


def many_to_one(field, ns, k, load, seed, real=False):
    """``k`` distinct patterns per location, all targeting one grid state.

    ``load`` counts *total* stored patterns against Ns, so a given load means
    the same number of columns however they are grouped -- which is exactly the
    comparison that says whether many-to-one costs anything beyond its pattern
    count.
    """
    n_total = int(round(load * ns))
    n_loc = max(1, n_total // k)
    n_total = n_loc * k
    rng = np.random.RandomState(seed + n_total + k)
    pb, gb = targets(field, n_loc, rng)
    pb = np.repeat(pb, k, axis=1)                           # cell-major
    gb = np.repeat(gb, k, axis=1)
    sbook = (real_views_sbook(n_loc, k, ns, rng) if real
             else rng.choice([-1.0, 1.0], size=(ns, n_total)))
    return n_loc, n_total, recovery(field, sbook, pb, gb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer loads and one Ns")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    loads = ([0.5, 1.0, 2.0] if args.quick
             else [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
    ns_values = [60] if args.quick else [60, 240]

    field = build_field(args.seed)
    chance = 1.0 / np.prod([l ** 2 for l in LAMBDAS])

    print("Exact grid-state recovery vs load, random +/-1 sensory patterns")
    print(f"lambdas={LAMBDAS}  Np={NP}  Npos={field.Npos}  "
          f"modules={[l**2 for l in LAMBDAS]}  chance={chance:.2%}")
    print("Recovery is all-or-nothing per pattern: every module's winner-take-all")
    print("must be correct. 'real' = ray-cast wall codes at the same load.")

    for ns in ns_values:
        print(f"\nNs = {ns}")
        print(f"  {'load':>6} {'Npatts':>7}   {'random':>8}   {'real':>8}")
        print("  " + "-" * 36)
        for load in loads:
            npatts = int(round(load * ns))
            if npatts < 1 or npatts > field.Npos * field.Npos:
                continue
            rng = np.random.RandomState(args.seed + npatts)
            pb, gb = targets(field, npatts, rng)

            rnd = rng.choice([-1.0, 1.0], size=(ns, npatts))
            acc_rand = recovery(field, rnd, pb, gb)
            acc_real = recovery(field, real_sbook(npatts, ns, rng), pb, gb)

            print(f"  {load:6.2f} {npatts:7d}   {acc_rand:7.1%}   {acc_real:7.1%}")

    print("\nThe boundary to look at is load = 1.00 (Npatts = Ns): below it the")
    print("pseudoinverse is an exact left inverse for independent patterns.")

    # Why the two curves diverge: capacity is set by how many dimensions the
    # patterns actually span, which for random patterns is all Ns of them and
    # for ray-cast codes is far fewer. Participation ratio of the singular
    # values, (sum s^2)^2 / sum s^4, is the usual soft rank -- it counts
    # directions carrying real variance rather than ones merely nonzero.
    print("\nEffective dimensionality of the sensory patterns (soft rank)")
    print(f"  {'Ns':>5} {'random':>9} {'real':>9}   {'real/Ns':>8}")
    print("  " + "-" * 36)
    for ns in ns_values:
        rng = np.random.RandomState(args.seed)
        npatts = min(4 * ns, field.Npos * field.Npos)
        books = {"random": rng.choice([-1.0, 1.0], size=(ns, npatts)),
                 "real": real_sbook(npatts, ns, rng)}
        soft = {}
        for name, S in books.items():
            s = np.linalg.svd(S, compute_uv=False) ** 2
            soft[name] = float(s.sum() ** 2 / (s ** 2).sum())
        print(f"  {ns:5d} {soft['random']:9.1f} {soft['real']:9.1f}   "
              f"{soft['real'] / ns:8.2f}")
    print("\nCapacity tracks these, not Ns. Random patterns spend every dimension")
    print("they are given; ray-cast views spend a small fraction, so adding")
    print("sensory bits buys far less capacity than the random curve suggests.")

    # Many-to-one: k distinct patterns per location, all pointing at the same
    # grid state. The question is whether collapsing k views onto one target
    # costs anything BEYOND the k-fold increase in stored patterns. Plotted
    # against total load it should not: a pseudoinverse is an exact left inverse
    # whenever the columns are independent, and it does not care that some of
    # them share a target.
    ns = ns_values[-1]
    ks = [1, 2, 4, 8]
    print(f"\nMany-to-one: k patterns per location, one grid state (Ns={ns})")
    print("  random patterns, x-axis is TOTAL load = k * n_loc / Ns")
    header = "  ".join(f"{'k=' + str(k):>6}" for k in ks)
    # ~patts is the nominal total; each k rounds n_loc down to a whole number of
    # locations, so its actual count can sit a few columns below this.
    print(f"  {'load':>6} {'~patts':>6}   {header}")
    print("  " + "-" * (16 + 8 * len(ks)))
    for load in loads:
        cells = [f"{many_to_one(field, ns, k, load, args.seed)[2]:6.1%}"
                 for k in ks]
        print(f"  {load:6.2f} {int(round(load * ns)):6d}   " + "  ".join(cells))

    print("\n  Same, but real ray-cast views (k = 4 cardinal headings):")
    print(f"  {'load':>6} {'n_loc':>6}   {'random k=4':>11}   {'real k=4':>9}")
    print("  " + "-" * 40)
    for load in loads:
        n_loc, n_total, acc_r = many_to_one(field, ns, 4, load, args.seed)
        _, _, acc_real = many_to_one(field, ns, 4, load, args.seed, real=True)
        print(f"  {load:6.2f} {n_loc:6d}   {acc_r:10.1%}   {acc_real:8.1%}")

    print("\nIf the k columns collapse onto one curve, many-to-one costs exactly")
    print("its pattern count and nothing more -- the locations you can store")
    print("drop by k, but the association itself is unbothered by the sharing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
