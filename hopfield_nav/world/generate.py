"""The env generator: draw train and validation env sets that are provably apart.

One entry point serves training, in-training validation and post-hoc validation:

    generate_split(...)   train + base_val together, so separation is enforced
                          jointly rather than checked after the fact
    make_val_set(...)     its generalization, per-trait level: same/held_out/ood

Separation, per trait (``docs/EVAL_SPLITS_DESIGN.md`` §2.5 and
``docs/ENV_GENERATOR_STATUS.md`` Phase 2):

    place   A **spacing** requirement: `margin` cells of empty scaffold between
            every pair of env footprints -- train<->train, train<->val and
            val<->val alike. Measured edge to edge (so it is independent of env
            size), max-over-axes (two boxes are apart as soon as one axis
            separates them), and wrapped mod prod(lambdas) because the scaffold
            is a torus when Npos == prod(lambdas): a flat check calls
            (1715,987) and (4,989) 1711 cells apart when they are 5.
    wall    seed disjointness, verified on the codebook, reported as a Hamming
            margin over *live* bits only. Which bits are live depends on whether
            the agent's cone can turn: under a fixed North cone the South wall
            is never hit (dy = cos(theta) >= 0.5 > 0), leaving 3*size live bits;
            under egocentric heading all 4*size are reachable.
    goal    env-local cell-set disjointness.
    size    set disjointness.

The cosine diagnostic is computed and reported but **never gates** -- a split
stays derivable from coordinates alone, with no encoder in the loop. The cost is
accepted knowingly: worst-case cosine does not fall with distance, so at the
derived margin a val env can still sit near a train env in embedding space. That
shows up as a number rather than being silently repaired.
"""
from __future__ import annotations

import dataclasses
from functools import lru_cache

import numpy as np

from . import domains as dom
from .env import GridEnv, make_env
from .spec import EnvSpec, GeneratedSplit, TraitDomains

_MAX_ATTEMPTS = 20_000


# ---------------------------------------------------------------------------
# Wall codes and their live bits
# ---------------------------------------------------------------------------

def wall_code_for(seed: int, size: int, resolution: int = 1) -> np.ndarray:
    """The (4, size*resolution) +/-1 code ``GridEnv(seed=seed)`` builds.

    Reproduces ``env.py``'s first draw exactly: the wall code is the first thing
    the env's RNG is used for, so this needs no env instance. ``resolution``
    must match ``EnvConfig.wall_resolution`` or the draw will not agree.
    """
    return np.random.RandomState(seed).choice(
        [-1.0, 1.0], size=(4, size * resolution)).astype(np.float32)


@lru_cache(maxsize=32)
def live_wall_bits(size: int, observation_size: int,
                   egocentric: bool = True, resolution: int = 1) -> tuple:
    """Which of the 4*size*resolution wall bits any ray can actually see.

    Flips each bit and diffs the resulting codebook.

    Under a **fixed** North-facing cone, wall 2 (South) comes back entirely dead
    by construction -- every ray has ``dy = cos(theta) > 0`` -- so effective env
    identity is 3*size bits, not 4*size, and a Hamming distance over all 4*size
    would count a quarter silent no-ops.

    Under **egocentric** heading the cone turns with the agent, so all four
    walls are reachable and identity is the full 4*size bits. That only ever
    *raises* the distance between two envs, so a split separated under the old
    count stays separated under this one.

    At ``resolution`` > 1 a segment is a fraction of a cell, so more bits exist
    and finitely many rays cannot land on all of them -- expect some to come
    back dead simply for want of a ray, which is a true statement about what
    this env can distinguish and is exactly what the Hamming margin should
    count.

    Returned as a tuple of tuples so it is hashable/cacheable.
    """
    env = GridEnv(size=size, observation_size=observation_size, seed=0,
                  wall_resolution=resolution)
    # A fixed-heading agent only ever sees the North slab; an egocentric one can
    # reach all four, so identity is read off whichever the agent can observe.
    view = (lambda cb: cb) if egocentric else (lambda cb: cb[:, :, 0, :])
    base = view(env._codebook).copy()
    live = np.zeros((4, size * resolution), dtype=bool)
    for w in range(4):
        for k in range(size * resolution):
            env._wall_code[w, k] *= -1
            live[w, k] = not np.array_equal(
                view(env._build_sensory_codebook(observation_size)), base)
            env._wall_code[w, k] *= -1
    return tuple(tuple(bool(v) for v in row) for row in live)


def wall_hamming(seed_a: int, seed_b: int, size: int, observation_size: int,
                 egocentric: bool = True, resolution: int = 1) -> int:
    """Hamming distance between two wall codes, over live bits only."""
    live = np.array(live_wall_bits(size, observation_size, egocentric, resolution),
                    dtype=bool)
    a = wall_code_for(seed_a, size, resolution)
    b = wall_code_for(seed_b, size, resolution)
    return int(((a != b) & live).sum())


# ---------------------------------------------------------------------------
# Placement separation
# ---------------------------------------------------------------------------

def axis_separation(a: int, size_a: int, b: int, size_b: int, period: int) -> int:
    """Clearance between two intervals on a circle of circumference ``period``.

    Two arcs on a circle leave two gaps; they are disjoint only if both are
    non-negative, so the separation is the *smaller* of the two.
    """
    delta = (b - a) % period
    return min(delta - size_a, (period - delta) - size_b)


def toroidal_gap(off_a, size_a: int, off_b, size_b: int, period: int) -> int:
    """Edge-to-edge clearance between two square footprints, max over axes.

    Max, not min: two axis-aligned boxes are disjoint as soon as they separate
    on *one* axis. Negative means they overlap.
    """
    return max(
        axis_separation(off_a[0], size_a, off_b[0], size_b, period),
        axis_separation(off_a[1], size_a, off_b[1], size_b, period),
    )


def _lattice_pitch(domain: dom.PlaceDomain, size: int, Npos: int, margin: int,
                   n_needed: int, period: int):
    """The most generous lattice that still holds ``n_needed`` envs.

    Starts at the tightest legal pitch (``size + margin``) and widens while the
    lattice still fits what was asked for, so leftover room becomes spacing
    rather than being wasted. Returns ``(pitch, xs, ys)`` or ``None``.
    """
    x_lo, y_lo, x_hi, y_hi = domain.bounds(size, Npos)
    best = None
    pitch = size + margin
    while pitch <= max(x_hi - x_lo, y_hi - y_lo) + 1:
        xs = _axis_slots(x_lo, x_hi, pitch, size, margin, period)
        ys = _axis_slots(y_lo, y_hi, pitch, size, margin, period)
        if len(xs) * len(ys) < n_needed:
            break
        best = (pitch, xs, ys)
        pitch += 1
    return best


def _axis_slots(lo: int, hi: int, pitch: int, size: int, spacing: int,
                period: int) -> list[int]:
    """Lattice coordinates along one axis, with the wrap taken into account.

    ``range(lo, hi+1, pitch)`` is not a valid lattice on a torus: when the
    domain spans the period, the last slot is adjacent to the first. At
    ``Npos=132, pitch=41`` the slots are 0/41/82/123, and 123 is **9** cells from
    0, not 123 -- so the "lattice" silently violates its own spacing at exactly
    one pair, which is enough to starve the placement loop. Drop trailing slots
    until the seam clears.
    """
    vals = list(range(lo, hi + 1, pitch))
    while len(vals) > 1 and axis_separation(
            vals[-1], size, vals[0], size, period) < spacing:
        vals.pop()
    return vals


def _axis_slack(vals: list[int], size: int, spacing: int, period: int,
                pitch: int) -> int:
    """How far a slot on this axis may be jittered without closing a gap.

    Read off the slots the lattice actually got, not off the pitch. The two
    disagree at the seam: slots ``[0, 10]`` on a period of 15 are 7 apart going
    up and **2** apart going round, so pitch-based slack licenses a jitter that
    drops the wrap gap below the spacing. The cost is not a near-miss -- the
    jittered env then blocks a later lattice slot, and a packing the lattice
    provably had room for comes back as "no lattice of that pitch fits".

    Halved because both neighbours can jitter toward each other.
    """
    if len(vals) < 2:
        return max(0, (pitch - size - spacing) // 2)
    gaps = [b - a - size for a, b in zip(vals, vals[1:])]
    gaps.append(period - vals[-1] + vals[0] - size)      # the wrap
    return max(0, (min(gaps) - spacing) // 2)


def _lattice_places(
    domain: dom.PlaceDomain, rng, n: int, *, size: int, Npos: int, period: int,
    exclude, margin: int, self_margin: int,
) -> list[tuple[int, int]] | None:
    """Deterministic fallback for dense packings.

    Rejection sampling finds a free spot easily when the region is mostly empty
    and hardly ever when it is mostly full -- and "a packing exists" (which is
    what the capacity preflight checks) is not the same as "uniform probing will
    stumble onto one". At 15 envs in a 25-slot region the free area is a few
    percent of the search space and 20k probes routinely miss.

    So: lay down a lattice, shuffle the slots, and take the first ``n`` that
    clear ``exclude``. Jitter within whatever slack the chosen pitch left, which
    keeps offsets off a perfectly regular grid when there is room to spare and
    degrades to an exact lattice when there is not. Same shape as the historical
    ``spread_offsets``, which is what training placement has always looked like.
    """
    # The lattice has to satisfy whichever constraint binds: clearance from
    # `exclude` (margin) and mutual clearance among the envs drawn here
    # (self_margin) are separate parameters, and a lattice pitched for only one
    # of them fails the other on nearly every slot.
    spacing = max(int(margin), int(self_margin))
    found = _lattice_pitch(domain, size, Npos, spacing, n, period)
    if found is None:
        return None
    pitch, xs, ys = found
    slack_x = _axis_slack(xs, size, spacing, period, pitch)
    slack_y = _axis_slack(ys, size, spacing, period, pitch)
    slots = [(x, y) for x in xs for y in ys]
    order = rng.permutation(len(slots))
    placed: list[tuple[int, int]] = []
    x_lo, y_lo, x_hi, y_hi = domain.bounds(size, Npos)
    for idx in order:
        if len(placed) >= n:
            break
        bx, by = slots[idx]
        # A few jitter tries, then the bare slot. That last attempt is what
        # makes the fallback honour its own contract: the lattice is chosen
        # because its slots are known to clear each other, but a jittered
        # candidate has no such guarantee against `exclude` -- envs placed by an
        # earlier draw and therefore off this lattice. Without it, eight unlucky
        # jitters abandon a slot that was legal all along, and a tight packing
        # fails with "no lattice of that pitch fits" when one plainly did. Only
        # reachable once something places against a non-empty `exclude`, which
        # is every per-trait refresh (training/refresh.py).
        for k in range(9):
            jx = int(rng.randint(-slack_x, slack_x + 1)) if slack_x and k < 8 else 0
            jy = int(rng.randint(-slack_y, slack_y + 1)) if slack_y and k < 8 else 0
            cand = (min(max(bx + jx, x_lo), x_hi), min(max(by + jy, y_lo), y_hi))
            if not domain.contains(cand, size, Npos):
                continue
            if any(toroidal_gap(cand, size, o, s, period) < margin
                   for o, s in exclude):
                continue
            if any(toroidal_gap(cand, size, p, size, period) < self_margin
                   for p in placed):
                continue
            placed.append(cand)
            break
    return placed if len(placed) == n else None


def sample_places(
    domain: dom.PlaceDomain, rng, n: int, *, size: int, Npos: int, period: int,
    exclude: list[tuple[tuple[int, int], int]] = (), margin: int = 0,
    self_margin: int = 0, max_attempts: int = _MAX_ATTEMPTS,
) -> list[tuple[int, int]]:
    """``n`` offsets in ``domain``, clear of each other and of ``exclude``.

    ``exclude`` is ``[(offset, size), ...]`` already placed elsewhere -- the train
    set, when drawing validation. ``self_margin=0`` means "must not overlap";
    anything larger demands that much clearance between the envs drawn here.

    Callers generally want ``self_margin == margin`` even for the train set. Not
    because two train envs near each other leaks -- it does not -- but because
    the *joint* packing needs a common basis: train envs placed with no clearance
    can occupy every slot a margin-separated val env could use, and then this
    loop exhausts its attempts on a problem that looked feasible.
    """
    placed: list[tuple[int, int]] = []
    attempts = 0
    while len(placed) < n:
        attempts += 1
        if attempts > max_attempts:
            # Dense packing: probing is the wrong tool, lay a lattice instead.
            fallback = _lattice_places(
                domain, rng, n, size=size, Npos=Npos, period=period,
                exclude=exclude, margin=margin, self_margin=self_margin)
            if fallback is not None:
                return fallback
            raise RuntimeError(
                f"could not place {n} envs of size {size} in {domain!r} "
                f"(margin={margin}, self_margin={self_margin}, Npos={Npos}): "
                f"{len(placed)} by rejection sampling in {max_attempts} "
                f"attempts, and no lattice of that pitch fits either. Capacity "
                f"estimate is "
                f"{domain.capacity(size, max(margin, self_margin), Npos)}. Raise Npos, "
                f"enlarge the region, lower the margin, or ask for fewer envs."
            )
        cand = domain.candidate(rng, size, Npos)
        if not domain.contains(cand, size, Npos):
            continue
        if any(toroidal_gap(cand, size, o, s, period) < margin for o, s in exclude):
            continue
        if any(toroidal_gap(cand, size, p, size, period) < self_margin
               for p in placed):
            continue
        placed.append(cand)
    return placed


# ---------------------------------------------------------------------------
# Margin derivation and the cosine diagnostic
# ---------------------------------------------------------------------------

def _unit_rows(field, xs, ys) -> np.ndarray:
    v = np.asarray(field.encoded_Phi[xs, ys], dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


def derive_margin(
    field, rng, *, quantile: float = 0.99, threshold: float = 0.15,
    n_pairs: int = 2048, candidates=None,
) -> int:
    """Smallest gap at which embedding similarity clears ``threshold``.

    Sampled over the **full Chebyshev ring** at each candidate gap, not one
    direction: axis-aligned decorrelates roughly 3x slower than diagonal, and two
    envs side by side are axis-aligned. Derived on a quantile rather than the
    mean, because the mean reaches 0 well before individual pairs do.

    Raises rather than clamping when no candidate clears -- at ``fwhm_ratio=0.5``
    the field plateaus around +0.12 and no finite margin separates it, which a
    hardcoded constant would hide.
    """
    Npos = field.Npos
    if candidates is None:
        candidates = list(range(10, 201, 10))
    for d in candidates:
        if 2 * d >= Npos:
            break
        base = rng.randint(d, Npos - d, size=(n_pairs, 2))
        # Uniform on the Chebyshev ring of radius d: pick a side, then a
        # coordinate along it.
        side = rng.randint(0, 4, size=n_pairs)
        t = rng.randint(-d, d + 1, size=n_pairs)
        disp = np.empty((n_pairs, 2), dtype=np.int64)
        disp[side == 0] = np.stack([np.full((side == 0).sum(), d), t[side == 0]], 1)
        disp[side == 1] = np.stack([np.full((side == 1).sum(), -d), t[side == 1]], 1)
        disp[side == 2] = np.stack([t[side == 2], np.full((side == 2).sum(), d)], 1)
        disp[side == 3] = np.stack([t[side == 3], np.full((side == 3).sum(), -d)], 1)
        other = base + disp
        cos = (_unit_rows(field, base[:, 0], base[:, 1])
               * _unit_rows(field, other[:, 0], other[:, 1])).sum(-1)
        if float(np.quantile(cos, quantile)) < threshold:
            return int(d)
    raise RuntimeError(
        f"no gap up to {candidates[-1]} brings the q{quantile:.2f} cosine below "
        f"{threshold}. This scaffold does not decorrelate with distance -- check "
        f"fwhm_ratio (0.5 plateaus around +0.12) before widening the search."
    )


def _footprint_cells(specs: list[EnvSpec], Npos: int) -> tuple:
    xs, ys, owner = [], [], []
    for i, s in enumerate(specs):
        gx, gy = np.meshgrid(np.arange(s.size), np.arange(s.size), indexing="ij")
        xs.append(np.clip(gx.ravel() + s.offset[0], 0, Npos - 1))
        ys.append(np.clip(gy.ravel() + s.offset[1], 0, Npos - 1))
        owner.append(np.full(s.size * s.size, i))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(owner))


def cosine_report(field, specs_a: list[EnvSpec], specs_b: list[EnvSpec], *,
                  chunk: int = 2048) -> dict:
    """Embedding similarity between two env sets. Reported, never enforced.

    The policy's ``input_encoded_state`` channel *is* ``encoded_Phi[cell]``
    (``policy/channels.py:71``), so two cells at high cosine are near-identical
    policy inputs however far apart they sit. The Euclidean margin controls the
    bulk of this but provably not the tail, which is what this measures.

    ``per_env_max`` is the actionable number: which val env, if any, looks like
    something in training.
    """
    if not specs_a or not specs_b:
        return {}
    ax, ay, _ = _footprint_cells(specs_a, field.Npos)
    bx, by, bowner = _footprint_cells(specs_b, field.Npos)
    A = _unit_rows(field, ax, ay)
    n_b = len(bx)
    per_env = np.full(len(specs_b), -1.0, dtype=np.float64)
    gmax, n_hi3, n_hi5, total = -1.0, 0, 0, 0
    for start in range(0, n_b, chunk):
        sl = slice(start, min(start + chunk, n_b))
        B = _unit_rows(field, bx[sl], by[sl])
        sims = B @ A.T                       # (chunk, cells_a)
        rowmax = sims.max(axis=1)
        np.maximum.at(per_env, bowner[sl], rowmax)
        gmax = max(gmax, float(rowmax.max()))
        n_hi3 += int((sims > 0.3).sum())
        n_hi5 += int((sims > 0.5).sum())
        total += sims.size
    return {
        "max": gmax,
        "per_env_max": [float(v) for v in per_env],
        "frac_above_0.3": n_hi3 / max(total, 1),
        "frac_above_0.5": n_hi5 / max(total, 1),
        "n_pairs": total,
    }


# ---------------------------------------------------------------------------
# Building envs from specs
# ---------------------------------------------------------------------------

def build_envs(specs: list[EnvSpec], env_cfg, movement_mode: str) -> list[GridEnv]:
    """Realize specs as environments.

    Per-env size goes through ``dataclasses.replace`` rather than widening
    ``make_env``: ``make_env`` reads ``env_cfg.size``, so a replaced config is
    the whole change.
    """
    envs = []
    for s in specs:
        cfg = dataclasses.replace(env_cfg, size=s.size)
        env = make_env(cfg, movement_mode, seed=s.wall_seed)
        env.set_goal(s.goal)
        envs.append(env)
    return envs


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _single_size(domains: TraitDomains) -> int:
    if len(domains.size.values) != 1:
        raise NotImplementedError(
            f"mixed sizes within one env set are not supported yet (got "
            f"{domains.size.values}). Size OOD ships as a uniform val size "
            f"differing from train -- pass make_val_set(..., size=N)."
        )
    return int(domains.size.values[0])


def generate_split(
    field, env_cfg, domains: TraitDomains, n_train: int, n_val: int, seed: int,
    *, refresh_goal: bool = False, margin: int | None = None,
    val_frac: float = 0.2, diagnostics: bool = True,
) -> GeneratedSplit:
    """Draw the train and base-validation env sets together.

    Together, not one after the other: sampling validation without knowing the
    train set is exactly what produces today's overlap (§1.3 -- 10/10 val envs
    overlapped a train footprint at the dataclass defaults).
    """
    size = _single_size(domains)
    Npos, period = field.Npos, int(np.prod(field.lambdas))

    if margin is None:
        margin = derive_margin(field, np.random.RandomState(
            dom.stable_hash(seed, "margin")))

    # Preflight: fail with a specific knob rather than a rejection-sampling hang.
    #
    # The bound has to be the *joint* one. An earlier version checked val and
    # train separately -- capacity(margin) >= n_val and capacity(0) >= n_train +
    # n_val -- and passed a configuration that then failed to place a single val
    # env. Train envs placed with no mutual clearance can sit anywhere, including
    # squarely on every slot a margin-separated val env could use, so their
    # capacity at margin 0 says nothing about what is left for validation.
    #
    # Placing train envs on the same margin basis is what makes this sound, and
    # it costs nothing: at the working config, margin 80 admits 289 envs against
    # the 84 used. It also moves train placement *towards* what the historical
    # spread-lattice did, not away from it.
    cap = domains.place.capacity(size, margin, Npos)
    if cap < n_train + n_val:
        raise ValueError(
            f"place domain {domains.place!r} holds ~{cap} envs of size {size} at "
            f"margin {margin}, need {n_train} train + {n_val} val = "
            f"{n_train + n_val}. Raise Npos, enlarge the region, lower the "
            f"margin, or ask for fewer envs.")

    region = domains.goal.cells(size)

    # --- goal cells: two branches, one resolved object (§2.8) --------------
    goal_rng = dom.trait_rng(seed, "goal", role="split")
    ordered = sorted(region)
    if refresh_goal:
        # Refresh would otherwise exhaust the grid: 80 envs consume 80 fresh
        # cells per update and there are only size**2, so after ~5 updates no
        # legal held-out goal exists. Cap the train set up front instead.
        perm = goal_rng.permutation(len(ordered))
        n_val_cells = max(1, int(round(val_frac * len(ordered))))
        cells_val = frozenset(ordered[i] for i in perm[:n_val_cells])
        cells_train = frozenset(ordered[i] for i in perm[n_val_cells:])
    else:
        idx = goal_rng.choice(len(ordered), size=n_train, replace=len(ordered) < n_train)
        cells_train = frozenset(ordered[i] for i in idx)
        cells_val = frozenset(region) - cells_train
    if not cells_val:
        raise ValueError(
            "no goal cells left for validation: training claims every cell in "
            f"{domains.goal!r} at size {size}. Shrink n_train, widen the goal "
            "region, or enable refresh_goal (which caps the train set at "
            f"{1 - val_frac:.0%} of the region).")

    # --- traits -------------------------------------------------------------
    wall_rng = dom.trait_rng(seed, "wall", role="split")
    train_seeds = domains.wall.sample(wall_rng, n_train)
    val_seeds = domains.wall.sample(wall_rng, n_val, exclude=frozenset(train_seeds))

    place_rng = dom.trait_rng(seed, "place", role="split")
    # One draw for train *and* val, split afterwards. Placing them in two calls
    # -- train first, then val excluding train -- looks equivalent and is not:
    # the train envs land wherever they like and can leave no legal slot for the
    # val envs at all. At size 8, margin 20, Npos 132, each placed env blocks a
    # 48x48 zone, so twelve of them cover 27648 cells of a 17424-cell scaffold.
    # Drawing all of them against one another makes train<->val separation a
    # property of the draw rather than something checked afterwards, and makes
    # the capacity preflight exactly the right condition.
    all_off = sample_places(domains.place, place_rng, n_train + n_val,
                            size=size, Npos=Npos, period=period,
                            self_margin=margin)
    train_off, val_off = all_off[:n_train], all_off[n_train:]

    train_cells, val_cells = sorted(cells_train), sorted(cells_val)
    train_goals = [train_cells[i] for i in
                   goal_rng.choice(len(train_cells), n_train,
                                   replace=len(train_cells) < n_train)]
    val_goals = [val_cells[i] for i in
                 goal_rng.choice(len(val_cells), n_val,
                                 replace=len(val_cells) < n_val)]

    train = [EnvSpec(train_seeds[i], size, train_off[i], train_goals[i])
             for i in range(n_train)]
    base_val = [EnvSpec(val_seeds[i], size, val_off[i], val_goals[i])
                for i in range(n_val)]

    split = GeneratedSplit(
        domains=domains, train=train, base_val=base_val,
        goal_cells_train=cells_train, goal_cells_val=cells_val,
        margin=int(margin), period=period, Npos=Npos,
    )
    split.record_used(train)

    if diagnostics:
        split.diagnostics = split_diagnostics(field, env_cfg, split)
    verify_split(split, env_cfg)
    return split


def split_diagnostics(field, env_cfg, split: GeneratedSplit) -> dict:
    """Numbers describing a split. Evidence about it, never an input to it."""
    obs = int(env_cfg.observation_size)
    ego = bool(getattr(env_cfg, "egocentric_heading", True))
    res = int(getattr(env_cfg, "wall_resolution", 1))
    size = split.train[0].size if split.train else 0
    gaps = [toroidal_gap(v.offset, v.size, t.offset, t.size, split.period)
            for v in split.base_val for t in split.train]
    hams = [wall_hamming(v.wall_seed, t.wall_seed, size, obs, ego, res)
            for v in split.base_val for t in split.train]
    live = (np.array(live_wall_bits(size, obs, ego, res), dtype=bool) if size
            else np.zeros((4, 0), bool))
    return {
        "margin": int(split.margin),
        "min_place_gap": int(min(gaps)) if gaps else None,
        "min_wall_hamming": int(min(hams)) if hams else None,
        "live_wall_bits": int(live.sum()),
        "total_wall_bits": int(live.size),
        "cosine": cosine_report(field, split.train, split.base_val),
    }


def verify_split(split: GeneratedSplit, env_cfg) -> None:
    """Assert the four separation properties. Cheap, and the point of all this."""
    size = split.train[0].size if split.train else 0
    res = int(getattr(env_cfg, "wall_resolution", 1))
    for v in split.base_val:
        for t in split.train:
            gap = toroidal_gap(v.offset, v.size, t.offset, t.size, split.period)
            if gap < split.margin:
                raise AssertionError(
                    f"val env at {v.offset} is {gap} cells from train env at "
                    f"{t.offset}, below margin {split.margin}")
    train_seeds = {t.wall_seed for t in split.train}
    for v in split.base_val:
        if v.wall_seed in train_seeds:
            raise AssertionError(f"val env reuses train wall seed {v.wall_seed}")
        if size and np.array_equal(wall_code_for(v.wall_seed, size, res),
                                   wall_code_for(next(iter(train_seeds)), size, res)):
            raise AssertionError("val wall code collides with a train wall code")
    train_goals = {t.goal for t in split.train}
    for v in split.base_val:
        if v.goal in train_goals:
            raise AssertionError(
                f"val goal cell {v.goal} was used by training -- goal cell sets "
                "must be disjoint in env-local coordinates")


# ---------------------------------------------------------------------------
# Post-hoc validation sets
# ---------------------------------------------------------------------------

_LEVELS = ("same", "held_out", "ood")


def make_val_set(
    split: GeneratedSplit, n_envs: int, levels: dict, seed: int, *,
    size: int | None = None, tick: int = 0,
) -> list[EnvSpec]:
    """A validation set at an arbitrary per-trait level.

    ``levels`` maps each of ``place``/``wall``/``goal``/``size`` to
    ``same`` (values training actually used -- a memorization probe),
    ``held_out`` (fresh, inside the train region, disjoint from what training
    used -- this is base validation), or ``ood`` (the region complement).

    ``ood`` exists only for ``place`` and ``goal``: wall and size have no
    bounded universe to complement, so wall novelty *is* ``held_out`` and size
    OOD is named outright via ``size=``.
    """
    for trait, lvl in levels.items():
        if lvl not in _LEVELS:
            raise ValueError(f"{trait}: unknown level {lvl!r}, expected {_LEVELS}")
    d = split.domains
    if size is None and len(split.used["size"]) > 1:
        # Reachable once size refresh exists: `used["size"]` is a set, so
        # picking one arbitrarily would make the val env size depend on set
        # iteration order rather than on anything the caller asked for.
        raise ValueError(
            f"training used sizes {sorted(split.used['size'])}; pass "
            "make_val_set(..., size=N) to say which one this set should use.")
    env_size = int(size if size is not None else next(iter(split.used["size"])))
    used_place = sorted(split.used["place"])
    used_wall = sorted(split.used["wall"])
    used_goal = sorted(split.used["goal"])

    # --- wall ---------------------------------------------------------------
    wl = levels.get("wall", "held_out")
    rng = dom.trait_rng(seed, "wall", tick, role="val")
    if wl == "ood":
        d.wall.complement()          # raises with the explanation
    if wl == "same":
        seeds = [used_wall[i] for i in rng.choice(
            len(used_wall), n_envs, replace=n_envs > len(used_wall))]
    else:
        seeds = d.wall.sample(rng, n_envs, exclude=frozenset(used_wall))

    # --- place --------------------------------------------------------------
    pl = levels.get("place", "held_out")
    rng = dom.trait_rng(seed, "place", tick, role="val")
    if pl == "same":
        offs = [used_place[i] for i in rng.choice(
            len(used_place), n_envs, replace=n_envs > len(used_place))]
    else:
        domain = d.place if pl == "held_out" else d.place.complement(split.margin)
        # self_margin=margin, same as generate_split: the margin is spacing, and
        # it applies to every pair. Envs drawn here must clear each other just as
        # they clear training -- otherwise a val set could stack several envs on
        # one patch of scaffold while still being correctly separated from train.
        offs = sample_places(
            domain, rng, n_envs, size=env_size, Npos=split.Npos,
            period=split.period, self_margin=split.margin,
            exclude=[(o, env_size) for o in used_place], margin=split.margin)

    # --- goal ---------------------------------------------------------------
    gl = levels.get("goal", "held_out")
    rng = dom.trait_rng(seed, "goal", tick, role="val")
    if gl == "same":
        pool = used_goal
    elif gl == "held_out":
        pool = sorted(c for c in split.goal_cells_val
                      if c[0] < env_size and c[1] < env_size)
    else:
        pool = sorted(dom.complement_for(d.goal, env_size).cells(env_size))
    if not pool:
        raise ValueError(
            f"goal level {gl!r} leaves no legal cell at size {env_size} "
            f"(domain {d.goal!r})")
    goals = [pool[i] for i in rng.choice(len(pool), n_envs,
                                         replace=n_envs > len(pool))]

    # --- size ---------------------------------------------------------------
    if levels.get("size", "same") != "same":
        raise ValueError(
            "size supports only level 'same'; env size has no bounded universe "
            "to hold out or complement. Pass make_val_set(..., size=N) for a "
            "uniform OOD size.")

    return [EnvSpec(int(seeds[i]), env_size, tuple(offs[i]), tuple(goals[i]))
            for i in range(n_envs)]


__all__ = [
    "axis_separation", "build_envs", "cosine_report", "derive_margin",
    "generate_split", "live_wall_bits", "make_val_set", "sample_places",
    "split_diagnostics", "toroidal_gap", "verify_split", "wall_code_for",
    "wall_hamming",
]
