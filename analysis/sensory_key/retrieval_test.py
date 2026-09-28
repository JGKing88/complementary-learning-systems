"""Offline test of a sensory-keyed goal memory (grid-MLP competitor, idea 1).

Memory: one entry per env. Key = the four-heading observation at the env's
goal (``GridEnv.omni_obs_at``, heading-invariant); value = the goal's grid
state (not needed here -- only addressing is tested). Query = the same
observation at the agent's current cell. Retrieval = argmax cosine.

Two questions, answered exhaustively over every cell of every env:

1. With every env's goal stored, does argmax return the querying env's own
   goal?  -> accuracy overall, per env, and by distance to the goal.
2. Can the top similarity score tell "my env's goal is stored" from "my env's
   goal is NOT stored (so argmax returns some other env's goal)"?
   -> the two top-score distributions and their AUC, overall and by distance.

No agent, no scaffold, no training. Envs are the d0_base recorded ``base_val``
set (6) plus fresh envs minted with the same env config for larger N.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from hopfield_nav.config import EnvConfig
from hopfield_nav.world.env import make_env

# d0_base (navigate_navp2_d0_base_s42_22133273) env config, read from the ckpt.
D0_ENV = dict(size=20, observation_size=60, wall_resolution=4, goal_radius=1.0,
              time_penalty=0.05, goal_reward=2.0, egocentric_heading=True)
D0_WORLD = ("/orcd/pool/003/jackking/cls_runs/agent_ckpts/"
            "navigate_navp2_d0_base_s42_22133273/world.json")
MULTI_ELLS = (8.0, 4.0, 2.0)
DIST_BINS = [("0 (goal)", 0.0, 0.5), ("1-2", 0.5, 2.5), ("3-5", 2.5, 5.5),
             ("6-10", 5.5, 10.5), (">10", 10.5, 1e9)]


def smooth_walls(env, ell_cells: float, mode: str) -> None:
    """Replace the env's i.i.d. ±1 barcode with a spatially correlated one.

    Gaussian-filters the env's own white code along each wall (sigma =
    ``ell_cells`` cells), then either rescales it by one env-wide std ("cont") or re-binarises it
    to ±1 with long runs ("sign"). ``ell_cells`` 0 leaves the walls untouched.
    The seed's code is the noise source, so each env stays distinct.
    """
    if mode.startswith("multi:"):
        # Sum of independent components at several spatial frequencies, each
        # scaled to unit std then weighted: per-wall constant, Gaussian-smoothed
        # noise at each length in MULTI_ELLS (cells), and the env's own fine ±1
        # barcode. Weights are "multi:w_const,w_ell1,...,w_fine".
        w = [float(x) for x in mode.split(":")[1].split(",")]
        assert len(w) == len(MULTI_ELLS) + 2, (w, MULTI_ELLS)
        from scipy.ndimage import gaussian_filter1d
        rng = np.random.RandomState(env.seed + 1)
        shape = env._wall_code.shape
        code = w[0] * rng.randn(4, 1) * np.ones(shape)
        for wk, ell in zip(w[1:-1], MULTI_ELLS):
            sm = gaussian_filter1d(rng.randn(*shape), sigma=ell * env.wall_resolution,
                                   axis=1, mode="nearest")
            code = code + wk * sm / sm.std()
        code = code + w[-1] * env._wall_code
        env._wall_code = code.astype(np.float32)
        env._codebook = env._build_sensory_codebook(env._codebook.shape[-1])
        return
    if mode.startswith("wallconst"):
        # Each wall one random N(0,1) value; "wallconst+A" adds the original
        # fine ±1 barcode on top, scaled so the constant has weight A.
        rng = np.random.RandomState(env.seed + 1)
        const = rng.randn(4, 1) * np.ones_like(env._wall_code)
        amp = float(mode.split("+")[1]) if "+" in mode else None
        code = const if amp is None else amp * const + env._wall_code
        env._wall_code = code.astype(np.float32)
        env._codebook = env._build_sensory_codebook(env._codebook.shape[-1])
        return
    if ell_cells <= 0:
        return
    from scipy.ndimage import gaussian_filter1d
    sm = gaussian_filter1d(env._wall_code.astype(np.float64),
                           sigma=ell_cells * env.wall_resolution, axis=1,
                           mode="nearest")
    if mode == "sign":
        code = np.where(sm >= 0, 1.0, -1.0)
    else:
        # One scale for the whole env: per-wall centring would delete each
        # wall's level, which is most of the signal once smoothing is long.
        code = sm / sm.std()
    env._wall_code = code.astype(np.float32)
    env._codebook = env._build_sensory_codebook(env._codebook.shape[-1])


def load_envs(n_fresh: int, seed: int, ell_cells: float = 0.0, mode: str = "cont"):
    cfg = EnvConfig(**D0_ENV)
    with open(D0_WORLD) as f:
        recorded = json.load(f)["split"]["base_val"]
    specs = [(int(s["wall_seed"]), tuple(s["goal"])) for s in recorded]
    rng = np.random.RandomState(seed)
    used = {w for w, _ in specs}
    while len(specs) < len(recorded) + n_fresh:
        w = int(rng.randint(0, 10_000_000))
        if w not in used:
            used.add(w)
            specs.append((w, None))
    obs, goals = [], []
    for w, g in specs:
        env = make_env(cfg, "continuous", seed=w)
        if g is not None:
            env.set_goal(g)
        smooth_walls(env, ell_cells, mode)
        goals.append(tuple(int(x) for x in env._goal))
        o = env.omni_obs_all()
        if mode.startswith("distal:"):
            # Distal panorama: a per-env ±1 pattern indexed by ABSOLUTE ray
            # angle, so it reads the same from every cell (walls at infinity).
            # The omni view's 4x60 rays sit at fixed absolute angles, so the
            # panorama is one fixed 240-vector per env added to every cell.
            amp = float(mode.split(":")[1])
            pano = np.random.RandomState(env.seed + 7).choice([-1.0, 1.0], o.shape[-1])
            o = o + amp * pano.astype(np.float32)
        obs.append(o)
    return np.stack(obs), np.array(goals), len(recorded)  # (E,S,S,D), (E,2)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg), ties count half (Mann-Whitney)."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    # average ranks for ties
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run_subset(obs, goals, idx):
    """All cells of the envs in ``idx`` against a memory of those envs' goals."""
    S = obs.shape[1]
    keys = unit(np.stack([obs[e, goals[e][0], goals[e][1]] for e in idx]))  # (N,D)
    yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    rows = []
    for k, e in enumerate(idx):
        q = unit(obs[e].reshape(S * S, -1))
        sims = q @ keys.T                                  # (S*S, N)
        own = sims[:, k]
        others = np.delete(sims, k, axis=1)
        other_max = others.max(axis=1)
        top = np.maximum(own, other_max)
        am = sims.argmax(axis=1)
        tie = (own == other_max)
        dist = np.hypot(yy.ravel() - goals[e][0], xx.ravel() - goals[e][1])
        rows.append(dict(env=e, correct=(am == k) & ~tie, tie=tie,
                         own=own, other_max=other_max, top_stored=top,
                         top_unstored=other_max, dist=dist))
    return rows


def summarise(rows):
    cat = {k: np.concatenate([r[k] for r in rows]) for k in
           ("correct", "tie", "own", "top_stored", "top_unstored", "dist")}
    d = cat["dist"]
    out = dict(
        n_cells=int(len(d)),
        acc=float(cat["correct"].mean()),
        tie_frac=float(cat["tie"].mean()),
        acc_per_env=[float(r["correct"].mean()) for r in rows],
        auc_top=auc(cat["top_stored"], cat["top_unstored"]),
        auc_own_vs_unstored=auc(cat["own"], cat["top_unstored"]),
        by_dist=[],
    )
    for name, lo, hi in DIST_BINS:
        m = (d >= lo) & (d < hi)
        if m.sum() == 0:
            continue
        out["by_dist"].append(dict(
            bin=name, n=int(m.sum()),
            acc=float(cat["correct"][m].mean()),
            auc_top=auc(cat["top_stored"][m], cat["top_unstored"][m]),
            own_mean=float(cat["own"][m].mean()),
            unstored_top_mean=float(cat["top_unstored"][m].mean()),
        ))
    return out, cat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_fresh", type=int, default=94)
    p.add_argument("--Ns", type=str, default="6,10,30,100")
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    obs, goals, n_rec = load_envs(a.n_fresh, a.seed)
    E = len(goals)
    print(f"envs: {E} ({n_rec} recorded d0_base base_val + {E - n_rec} fresh); "
          f"obs dim {obs.shape[-1]}", flush=True)

    # Sanity: how distinct are the goal keys from each other at all?
    K = unit(np.stack([obs[e, goals[e][0], goals[e][1]] for e in range(E)]))
    G = K @ K.T
    off = G[~np.eye(E, dtype=bool)]
    print(f"goal-key cross-sim: mean {off.mean():.3f} max {off.max():.3f}", flush=True)

    rng = np.random.RandomState(a.seed + 1)
    results, pooled = {}, {}
    for N in [int(x) for x in a.Ns.split(",")]:
        if N == n_rec:
            subsets = [list(range(n_rec))]
        else:
            subsets = [sorted(rng.choice(E, N, replace=False).tolist())
                       for _ in range(1 if N == E else a.draws)]
        allrows = []
        per_draw = []
        for idx in subsets:
            rows = run_subset(obs, goals, idx)
            s, _ = summarise(rows)
            per_draw.append(dict(acc=s["acc"], auc_top=s["auc_top"]))
            allrows += rows
        s, cat = summarise(allrows)
        s["per_draw"] = per_draw
        results[N] = s
        pooled[N] = cat
        print(f"\nN={N} ({len(subsets)} draw(s), {s['n_cells']} cells)", flush=True)
        print(f"  Q1 argmax own-goal acc {s['acc']:.3f}  (ties {s['tie_frac']:.3f})"
              f"  per-draw {[round(d['acc'], 3) for d in per_draw]}")
        print(f"  Q2 AUC top-score stored vs unstored {s['auc_top']:.3f}"
              f"  per-draw {[round(d['auc_top'], 3) for d in per_draw]}")
        print(f"     AUC own-sim vs unstored-top {s['auc_own_vs_unstored']:.3f}")
        print("  by distance to goal:  bin      n     acc   AUC   own_sim  unstored_top")
        for b in s["by_dist"]:
            print(f"                        {b['bin']:>6} {b['n']:6d}  {b['acc']:.3f} "
                  f"{b['auc_top']:.3f}  {b['own_mean']:.3f}    {b['unstored_top_mean']:.3f}")

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(dict(envs=E, n_recorded=n_rec, goal_key_crosssim=dict(
            mean=float(off.mean()), max=float(off.max())),
            results={str(k): v for k, v in results.items()}), f, indent=1)
    np.savez_compressed(os.path.join(a.out, "pooled.npz"), **{
        f"N{N}_{k}": v for N, c in pooled.items() for k, v in c.items()})
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
