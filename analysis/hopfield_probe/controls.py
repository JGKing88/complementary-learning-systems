"""Sec 6 controls, plus the Sec 3.1a rescue sweep.

Without these, none of Tests A-D is attributable.

The oracle (6.1) is the one most easily misread, so it is worth saying here
too: it is a **ceiling under the same projection**, not a ground truth.
``z_goal - z_c`` is a large displacement in embedding space and ``W`` is a
*local* tangent frame built from two one-cell neighbour displacements, so
projecting a far displacement onto it is a first-order approximation on a
curved manifold and degrades with distance for reasons that have nothing to do
with the encoder. It is still the right control precisely *because* the
Hopfield path has the identical pathology -- ``recall(z_c)`` is approximately
``z_goal``, so it is the same large displacement through the same frame.
``|err|_hopfield - |err|_oracle`` isolates recall error because the projection
error is common to both.

6.1b is the control that objection actually motivates: a **one-cell**
displacement, which exercises the frame at the scale it was built at.
"""
from __future__ import annotations

import numpy as np

from .encode import Field
from .harness import (
    ProbeConfig, World, build_memory, local_cells, scored_envs,
)
from .qfield import GridAcc, bearing, cell_q_field, project_q, q_error
from .stats import Scalars, wrap_to_pi


# ---------------------------------------------------------------------------
# 6.1 / 6.1b oracles
# ---------------------------------------------------------------------------

def oracle_q(field: Field, world: World, env: int, cfg: ProbeConfig,
             *, swap_gram_schmidt: bool = False) -> np.ndarray:
    """Sec 6.1: displacement to the exact goal embedding, no Hopfield.

    Mirrors ``rollout.signal.oracle_signal_at``: same Gram-Schmidt projection
    as the real path, with ``goal_embedding - current`` in place of a recalled
    pattern. No ``K`` axis and no ``steps`` axis.
    """
    cells = local_cells(cfg.env_size)
    offset = world.specs[env].offset
    cur = field.encoded_state(cells, offset)
    basis = field.local_basis(cells, offset,
                              swap_gram_schmidt=swap_gram_schmidt)
    goal_z = field.encoded_state(np.array([world.specs[env].goal]), offset)
    return project_q(basis, cur, np.broadcast_to(goal_z, cur.shape))


def local_oracle(
    field: Field, world: World, env: int, cfg: ProbeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sec 6.1b: one-cell displacement toward the goal.

    Returns ``(q_local, theta_true_local)``. The target is the neighbour cell
    one step along the straight line to the goal, and the error is scored
    against the bearing to *that neighbour*, not to the goal -- the whole point
    is to test the basis at its own scale.

    If this is accurate while the oracle is bad at range, the basis is fine and
    the embedding manifold is curved: the readout is intrinsically local. If
    this is already bad, the basis is broken at its own scale and every number
    in Tests B and C inherits that.
    """
    size = cfg.env_size
    cells = local_cells(size)
    offset = world.specs[env].offset
    goal = np.array(world.specs[env].goal)

    delta = (goal - cells).astype(float)
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    unit = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 1e-12)
    nbr = np.clip(np.round(cells + unit), 0, size - 1).astype(np.int64)

    cur = field.encoded_state(cells, offset)
    nxt = field.encoded_state(nbr, offset)
    basis = field.local_basis(cells, offset)
    q = project_q(basis, cur, nxt)

    d = (nbr - cells).astype(float)
    theta = bearing(d[:, 0], d[:, 1])
    # A cell whose neighbour is itself (the goal cell, or a clip against a
    # wall) has no target bearing; NaN keeps it out of the aggregates instead
    # of contributing a fake zero.
    theta[np.linalg.norm(d, axis=1) < 1e-12] = np.nan
    return q, theta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_controls(
    field: Field, worlds: list[World], cfg: ProbeConfig, *, progress=None,
) -> dict:
    """Every Sec 6 control on the same worlds the main tests used."""
    size = cfg.env_size
    cells = local_cells(size)
    k_ref = cfg.k_values[len(cfg.k_values) // 2]
    s_ref = cfg.steps[0]

    oracle_acc = GridAcc(cfg)
    local_acc = GridAcc(cfg)
    swap_acc = GridAcc(cfg)
    plain_acc = GridAcc(cfg)
    notanh_acc = GridAcc(cfg)
    empty = Scalars()

    for w in worlds:
        rng = np.random.RandomState(w.seed * 7 + 1)
        test_envs = scored_envs(cfg, k_ref)
        mem = build_memory(field, w, k_ref, cfg, rng)

        # use_tanh is read at recall, not at storage, so the same memory
        # serves both -- building a second would only consume the RNG.
        cfg_nt = _replace(cfg, use_tanh=False)

        for e in test_envs:
            goal = w.specs[e].goal

            # 6.1 oracle, and 6.2 the same oracle through a swapped basis, so
            # the comparison isolates the basis rather than the recall.
            oracle_acc.add_env(oracle_q(field, w, e, cfg), cells, goal)
            swap_acc.add_env(
                oracle_q(field, w, e, cfg, swap_gram_schmidt=True),
                cells, goal)

            # 6.1b local oracle, scored against its own one-cell target.
            ql, theta_l = local_oracle(field, w, e, cfg)
            adeg = np.abs(np.degrees(q_error(ql, theta_l)))
            d = np.sqrt(((cells - np.array(goal)) ** 2).sum(1).astype(float))
            local_acc.abs_err.add(d, adeg)
            local_acc.acc45.add(d, (adeg < 45.0).astype(float))
            local_acc.acc90.add(d, (adeg < 90.0).astype(float))
            local_acc.scalars.add("abs_err_mean", float(np.nanmean(adeg)))
            local_acc.scalars.add("acc45", float(np.nanmean(adeg < 45.0)))
            local_acc.scalars.add("acc90", float(np.nanmean(adeg < 90.0)))

            # The Hopfield path itself, at the reference (K, steps), so the
            # three-way gap on one chart is apples to apples.
            qf, _c, _b = cell_q_field(field, w, e, mem, cfg)
            plain_acc.add_env(qf[s_ref], cells, goal)

            # 6.5 linear control.
            qnt, _c2, _b2 = cell_q_field(field, w, e, mem, cfg_nt)
            notanh_acc.add_env(qnt[s_ref], cells, goal)

            # 6.3 empty memory. The production path short-circuits on
            # `num_memories == 0` and emits an all-zero signal
            # (rollout/signal.py:102) -- there is nothing recalled, so there is
            # no direction to report. Recall through an all-zero W does NOT
            # give that: F.normalize sends the zero vector to zero, so
            # `recalled - current` is `-current` and q comes back pointing
            # *away* from where the agent stands. Both are measured, because
            # the gap between them is exactly the leak this control exists to
            # catch, and a harness that silently took the second would report a
            # confident wrong direction as "chance".
            q_naive = _empty_memory_q(field, w, e, cfg)
            empty.add("naive_max_abs_q", float(np.abs(q_naive).max()))
            d_e = np.sqrt(((cells - np.array(goal)) ** 2).sum(1).astype(float))
            th = bearing((goal[0] - cells[:, 0]).astype(float),
                         (goal[1] - cells[:, 1]).astype(float))
            ad = np.abs(np.degrees(q_error(q_naive, th)))
            ad[d_e == 0] = np.nan
            empty.add("naive_acc45", float(np.nanmean(ad < 45.0)))
            empty.add("production_max_abs_q", 0.0)

        if progress:
            progress(f"C6 world={w.index}")

    return {
        "reference": {"k": int(k_ref), "steps": int(s_ref)},
        "oracle": oracle_acc.to_json(),
        "local_oracle": local_acc.to_json(),
        "gram_schmidt_swapped": swap_acc.to_json(),
        "hopfield_reference": plain_acc.to_json(),
        "no_tanh": notanh_acc.to_json(),
        "empty_memory": empty.to_json(),
    }


def _replace(cfg: ProbeConfig, **kw) -> ProbeConfig:
    import dataclasses
    return dataclasses.replace(cfg, **kw)


def _empty_memory_q(
    field: Field, world: World, env: int, cfg: ProbeConfig,
) -> np.ndarray:
    """``q`` from an all-zero ``W``, i.e. what recall gives with no short-circuit."""
    from hopfield import Hopfield
    from .harness import Memory, recall_trajectory

    cells = local_cells(cfg.env_size)
    offset = world.specs[env].offset
    cur = field.encoded_state(cells, offset)
    basis = field.local_basis(cells, offset)

    dim = cur.shape[1]
    hop = Hopfield(dim, beta=float(field.gain), zero_diag=cfg.zero_diag,
                   scale=cfg.hopfield_scale, device=cfg.device)
    mem = Memory(hopfield=hop, Z=np.zeros((0, dim), dtype=np.float32),
                 owner=np.zeros(0, dtype=int), diag_frac=0.0)
    traj = recall_trajectory(mem, cur, (cfg.steps[0],), cfg)
    return project_q(basis, cur, traj[cfg.steps[0]])


# ---------------------------------------------------------------------------
# 3.1a rescue
# ---------------------------------------------------------------------------


# Sec 5.7 Table B corrupts a cue to this cosine and asks whether recall pulls it
# back. 0.70 is far enough out that an identity map cannot fake a recovery.
CORRUPT_COS = 0.70


def _recovery(mem, cfg, rng: np.random.RandomState) -> float:
    """Median cos-to-original after recalling a cue corrupted to CORRUPT_COS.

    The corruption is a random orthogonal perturbation rather than noise on the
    raw vector, so the starting cosine is exactly CORRUPT_COS for every pattern
    and cells are comparable across the grid.
    """
    from .harness import recall_trajectory

    Z = mem.Z
    if Z.shape[0] == 0:
        return float("nan")
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-12)
    noise = rng.randn(*Zn.shape)
    noise -= (np.sum(noise * Zn, axis=1, keepdims=True)) * Zn      # orthogonal
    noise /= np.linalg.norm(noise, axis=1, keepdims=True).clip(1e-12)
    cues = CORRUPT_COS * Zn + np.sqrt(1 - CORRUPT_COS ** 2) * noise
    cues = (cues / np.linalg.norm(cues, axis=1, keepdims=True)
            ).astype(np.float32)

    s_last = max(cfg.steps)
    out = recall_trajectory(mem, cues, (s_last,), cfg)[s_last]
    out = out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-12)
    return float(np.median(np.sum(out * Zn, axis=1)))


def run_rescue(
    field: Field, worlds: list[World], cfg: ProbeConfig, *, progress=None,
) -> dict:
    """Sec 3.1a -- can *any* setting give attractor behaviour? Off by default.

    **This sweep is aimed at the compensable knob, and cannot succeed alone.**
    ``docs/EXPERIMENTS_NAV_P2.md`` Sec 5.6-5.7 (unmerged ``nav-tri-metric``)
    separates two conditions that are easy to conflate:

    * **loop gain** ``beta*S/D > 1`` -- set by ``beta`` *or* the storage norm,
      which are the same knob twice. Buys a nonzero fixed point instead of
      decay to zero. This is what the grid below moves.
    * **near-corner patterns** -- set by the *storage/encoder* gain ``g`` only,
      and not compensable by the first. Buys that the fixed point is *your*
      memory. Sharp turn-on at ``g ~ 100``.

    Sweeping ``beta`` alone was measured to 1e6 and fails, because it repairs
    the first and never touches the second. Pre-saturating the patterns does
    work inside this architecture -- Sec 5.7 gets cos-to-self 0.990 at M=5
    through 0.923 at M=50, with real basins -- but that is the **encoder's**
    gain, which this harness does not set.

    Measured on the live encoders, gain 100 gets most of the way and not all:
    L7-s42 sits at cos 0.895 to its own binarisation and 64% saturated against
    Sec 5.7's 0.954 / 83%, and its patterns are not fixed points at any
    ``(scale, beta)`` here. The encoder's final L2 normalisation is part of
    why -- it pins ``S = 1`` however saturated the pre-activation was.

    So read a positive result here as a reason to re-check ``nav_p2``, not as a
    new operating point. And read it beside a rank check: the untrained encoder
    scores a near-perfect 0.999 on the fixed-point test purely because it has
    **collapsed** -- pairwise cos 0.9986, effective rank 1.00 of 25, every
    pattern at cos 0.999 to the top eigenvector. One vector stored 25 times is
    trivially made of fixed points. That is why Sec 3.1's success criterion is
    ``frac_self_consistent`` **and** ``mean_pairwise_cos``: either alone is
    maximised by a degenerate memory.

    Only the **product** ``beta * scale`` reaches the argument of the ``tanh``
    -- ``beta * W`` is invariant under ``(p -> lambda p, beta -> beta/lambda^2)``
    -- so the sweep is reported against that product rather than either knob.
    """
    from .attractor import fixed_point_probe

    dim = field.embed_dim
    scales = [1.0 / dim, 1.0 / np.sqrt(dim), 1.0]
    # The encoder's own gain is in the grid because production sets
    # beta = encoder_gain: without it the sweep shows the alternatives without
    # showing what they are alternatives to. D^1.5 is the saturation threshold
    # (the tanh argument is beta * D^-1.5 at unit-norm storage), so it is the
    # one value the argument says should matter.
    betas = sorted({1.0, 10.0, 100.0, 1000.0, 10000.0,
                    float(field.gain), float(dim ** 1.5)})
    long_steps = tuple(sorted(set(cfg.steps) | {25, 50}))
    prod_scale = 1.0 / dim

    rows = []
    ks = [k for k in cfg.k_values if k <= 10] or [cfg.k_values[0]]
    use_worlds = worlds[:max(1, min(4, len(worlds)))]

    for zero_diag in (True, False):
        for alpha in (1.0, 0.5, 0.1):
            for scale in scales:
                for beta in betas:
                    sub = _replace(
                        cfg, zero_diag=zero_diag, alpha=alpha,
                        hopfield_scale=scale, beta_override=beta,
                        steps=long_steps,
                    )
                    for k in ks:
                        fsc, mpc, rec = [], [], []
                        for w in use_worlds:
                            rng = np.random.RandomState(w.seed * 5 + k)
                            mem = build_memory(field, w, k, sub, rng)
                            fp = fixed_point_probe(mem, sub)
                            last = fp[str(long_steps[-1])]
                            fsc.append(last["frac_self_consistent"])
                            mpc.append(last["mean_pairwise_cos"])
                            rec.append(_recovery(mem, sub, rng))
                        rows.append({
                            "zero_diag": zero_diag, "alpha": alpha,
                            "scale": float(scale), "beta": float(beta),
                            "beta_scale": float(beta * scale), "k": int(k),
                            "steps": int(long_steps[-1]),
                            "frac_self_consistent": float(np.mean(fsc)),
                            "mean_pairwise_cos": float(np.nanmean(mpc)),
                            # Sec 5.7 Table B: a cue corrupted to cos 0.70,
                            # recalled. Above 0.70 the dynamics pulled it back.
                            "recovery": float(np.mean(rec)),
                            # The anchor: what the live stack actually runs.
                            "is_production": bool(
                                zero_diag and alpha == 1.0
                                and abs(scale - prod_scale) < 1e-12
                                and abs(beta - float(field.gain)) < 1e-9),
                            # Tanh argument per coordinate. Below ~1 the recall
                            # is a linear matched filter whatever else is set.
                            "tanh_arg": float(beta * scale * dim
                                              / np.sqrt(dim)),
                        })
                    if progress:
                        progress(
                            f"R  zd={zero_diag} a={alpha} "
                            f"bs={beta * scale:.3g}")

    # Success is BOTH: patterns stay on themselves AND different cues do not
    # collapse onto one vector. Either alone is meaningless -- a single global
    # attractor scores a perfect frac_self_consistent at K=1.
    # K=1 is excluded, not counted as a pass. With one stored pattern
    # frac_self_consistent is trivially 1.0 (argmax over a single candidate)
    # and mean_pairwise_cos is undefined, so every K=1 cell would "succeed"
    # vacuously -- 166 of 378 in the first run of this sweep, which is exactly
    # the degenerate-metric trap Sec 3.1a warns about. A claim of attractor
    # behaviour needs at least two memories to be about anything.
    good = [r for r in rows
            if r["k"] > 1
            and r["frac_self_consistent"] > 0.95
            and np.isfinite(r["mean_pairwise_cos"])
            and r["mean_pairwise_cos"] < 0.5
            # Third condition, and the one that separates a basin from an
            # identity map: a corrupted cue has to come back.
            and r["recovery"] > CORRUPT_COS + 0.1]
    prod = [r for r in rows if r["is_production"]]
    return {"rows": rows, "n_success": len(good),
            "corrupt_cos": CORRUPT_COS,
            "production": prod,
            "betas": [float(b) for b in betas],
            "scales": [float(s_) for s_ in scales],
            "embed_dim": int(dim),
            "success": sorted(good, key=lambda r: -r["frac_self_consistent"])[:20],
            "note": "Not the production operating point. Not encoder-quality "
                    "numbers. A setting that works here still has to re-run "
                    "Sec 3.2 and Sec 4: a fixed point with a zero-radius basin "
                    "would be worse for navigation than what we have."}


__all__ = ["local_oracle", "oracle_q", "run_controls", "run_rescue"]
