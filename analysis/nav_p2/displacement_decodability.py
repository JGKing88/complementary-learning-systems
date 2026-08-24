"""Given two sensory cones in a NEW env, can the vector between them be decoded?

Jack's question. Phase 1's P0.9 (`analysis/nav_tri/sensory_decodability.py`)
asked whether *absolute* position is readable from one cone and got R^2 <= 0.13,
which is what pinned the explore ceiling at the billiard line (0.378) rather
than the lawnmower line (-> 0.50). But absolute position and *relative
displacement* are not the same quantity, and they fail for different reasons:

  * Absolute position cannot transfer across envs **by construction**. The wall
    code is a fresh random +/-1 draw per env (`world/env.py:329`), so the map
    position -> cone is a different hash in every arena. Nothing learned about
    one env's hash says anything about another's.
  * Relative displacement need not be hashed away. `E[s1[i] * s2[j]] = 1` when
    ray i of view 1 and ray j of view 2 land on the *same wall segment* and 0
    otherwise, whatever the codebook is. So the second-order statistic
    `s1 s2^T` is a **codebook-independent geometric measurement** -- it is the
    agreement structure between two views, and it is the same function of
    (position, heading) in every env. That is the only channel through which a
    decoder trained on envs 1..N can possibly work in env N+1, and it is the
    reason the question is worth asking at all.

So this module is really a test of one hypothesis: **is the pairwise agreement
structure between two cones a strong enough measurement of displacement to
path-integrate from?** If yes, the agent can know where it has been from
sensory alone and the lawnmower ceiling reopens. If no, explore stays reactive.

Four things about the actual observation had to be handled explicitly, and
each of them would have silently produced a wrong number:

1. **The cone is egocentric** (`egocentric_heading=True`), so the observation
   is a function of (position, heading), and heading is not a free variable:
   it is `atan2` of the realized displacement (`world/vec_env.py:461`). That
   makes "supply the heading to the decoder" a **leak**, not a control -- at
   lag 1 the heading *is* the direction of travel, so `dpsi` alone answers the
   question and any sensory arm that is handed it scores well for no sensory
   reason. It was measured (`side-only LEAK` row) rather than assumed. Six
   framings are reported separately, and only the first two are leak-free:
     `fixed` -- both views at psi=0. No heading anywhere; the clean case.
     `free`  -- views at the poses actually occupied, no heading supplied.
     `derot` -- view 2's cone re-indexed by `dpsi` so ray i of both views
                looks along the same world bearing, world-frame target. Kept
                because it is instructive that it FAILS: after alignment the
                features are relative to view 1's bearing and `psi1` is never
                supplied, so a world-frame answer is not determined. It is the
                ill-posed version of the next one.
     `derot_ego` -- the same alignment with the target in view 1's frame. The
                properly-posed realistic case, and what an agent chaining its
                own displacements would compute.
     `ego`   -- free headings, `dpsi` supplied, target in view 1's frame, no
                alignment.
     `world` -- free headings, both supplied, world-frame target.
   For the last four, read every row against `side-only LEAK`. In the
   ego-frame framings note also that the CONSTANT predictor is strong -- under
   a persistent walk the egocentric displacement is nearly always straight
   ahead -- so a small angular error there means much less than it looks.
2. **The observation is taken at the SNAPPED cell**, not the float position
   (`world/vec_env.py:373-376`). Sub-cell motion is invisible to the sensor. So
   there is a hard ceiling on decoding the true continuous `dpos` that has
   nothing to do with the decoder, and it is reported as `snap-oracle`: the
   score of a cheating predictor that outputs the exact integer cell
   difference. Any decoder score must be read against that, not against 1.0.
3. **The +/-1 code is a hash of the hit point, not a distance.** To separate
   "the geometry is not there" from "the geometry is there but hashed", every
   arm can also run on a `dist` sensor -- the same 60 rays returning range
   instead of a code. That is the lidar the env does not have, and it bounds
   what restructuring the sensory input could buy.
4. **A prediction of exactly zero has no direction.** The snap oracle emits
   one whenever both views land in the same cell, and a table decoder emits
   one whenever it returns the same entry twice. Dropping those samples (the
   obvious implementation, and the first one here) silently flatters both.
   They are scored 90 deg -- exactly uninformative -- and `frac_zero_pred`
   reports how often it happens.

Controls, both mandatory: `same-env` (fit and test inside one env -- the upper
bound a decoder with unlimited experience in this arena could reach) and
`shuffled` (train with s2 permuted, breaking the pairing -- chance). A
constant-predictor row is printed too, because the failure mode this metric
must be able to see is a decoder that has collapsed to the mean: `dpos` is
near-isotropic, so a mean predictor scores R^2 ~ 0 and **median angular error
~90 deg**. If a "working" number is not far from those, it is not working.

Needs no encoder, no scaffold, no GPU for the ridge arms: like P0.9 this is a
property of the sensor.

    python -m analysis.nav_p2.displacement_decodability --smoke
    python -m analysis.nav_p2.displacement_decodability \
        --train_envs 64 --test_envs 48 --json out.json
    # the properly-posed realistic case, at the settings most favourable to it
    python -m analysis.nav_p2.displacement_decodability \
        --resolution 1 --turn_sd_deg 20 --train_envs 128 \
        --framings fixed derot_ego ego --mlp --no_inenv

See `hopfield_nav/run_nav_p2_disp.sh` for the six probes this was run under,
and `docs/EXPERIMENTS_NAV_P2.md` section 6 for what came out.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from hopfield_nav.world.env import (
    FOVEAL_HALF_ANGLE_DEG, cone_offsets, raycast_codes,
)

# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


def raycast_range(size: int, xs, ys, psi, n_rays: int) -> np.ndarray:
    """Per-ray distance to the wall: ``(N, n_rays)``.

    The sensor the env does *not* have. Same cone, same geometry, same
    `raycast_codes` plane intersections -- but returning the range instead of
    the +/-1 code of the segment that was hit. It exists as an upper bound: if
    displacement decodes from range and not from codes, the bottleneck is the
    hashing and "structure the sensory input differently" is the live move; if
    it decodes from neither, the cone itself is the bottleneck.
    """
    xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
    ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
    psi = np.atleast_1d(np.asarray(psi, dtype=np.float64))
    xs, ys, psi = np.broadcast_arrays(xs, ys, psi)

    angles = psi[:, None] + cone_offsets(n_rays)[None, :]
    dx, dy = np.sin(angles), np.cos(angles)
    cx, cy = xs[:, None], ys[:, None]
    hi = size - 0.5
    inf = np.inf

    def _plane(num, den, keep):
        t = np.full(np.broadcast_shapes(num.shape, den.shape), inf)
        np.divide(num, den, out=t, where=keep)
        t[~keep | (t < 0.0)] = inf
        return t

    t_n = _plane(hi - cy, dy, dy > 0.0)
    t_e = _plane(hi - cx, dx, dx > 0.0)
    t_s = _plane(-0.5 - cy, dy, dy < 0.0)
    t_w = _plane(-0.5 - cx, dx, dx < 0.0)
    hit_n, hit_e = cx + t_n * dx, cy + t_e * dy
    hit_s, hit_w = cx + t_s * dx, cy + t_w * dy
    for t, h in ((t_n, hit_n), (t_e, hit_e), (t_s, hit_s), (t_w, hit_w)):
        t[np.isfinite(t) & ((h < -0.5) | (h > hi))] = inf
    ts = np.stack([t_n, t_e, t_s, t_w], axis=-1)
    return ts.min(axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Trajectory sampling -- mirrors ContinuousVecEnv exactly
# ---------------------------------------------------------------------------


def simulate(rng, *, size, n_walks, steps, min_norm, max_norm,
             turn_sd_deg=None):
    """Random-action walks under the phase-2 movement model.

    Reproduces `ContinuousVecEnv.step_batch`: float position clipped to
    ``[0, size-1]``, heading set to ``atan2(dx, dy)`` of the *realized*
    displacement, observation read at ``round(pos)``. Actions have norm
    uniform in ``[min_norm, max_norm]`` -- the band the phase-2 launcher
    clamps to.

    ``turn_sd_deg`` controls how straight the walk is. ``None`` draws each
    direction uniformly, which is the neutral reference; a finite value draws
    the turn from ``N(0, turn_sd)`` around the previous direction. This is not
    a detail: the whole free-heading result turns on how much two consecutive
    cones overlap, and `PERSISTENCE_BONUS=0.05` pays the policy to go straight.
    A conclusion drawn only from uniform turns would not survive a policy that
    does.
    """
    pos = rng.uniform(0.0, size - 1.0, size=(n_walks, 2))
    psi = np.zeros(n_walks)
    head = rng.uniform(-np.pi, np.pi, size=n_walks)
    P, S = [pos.copy()], [psi.copy()]
    for _ in range(steps):
        if turn_sd_deg is None:
            th = rng.uniform(-np.pi, np.pi, size=n_walks)
        else:
            th = head + rng.normal(0.0, np.deg2rad(turn_sd_deg), size=n_walks)
        head = th
        nrm = rng.uniform(min_norm, max_norm, size=n_walks)
        a = nrm[:, None] * np.stack([np.sin(th), np.cos(th)], axis=1)
        new = np.clip(pos + a, 0.0, float(size - 1))
        moved = new - pos
        pos = new
        spun = np.linalg.norm(moved, axis=1) >= 1e-12
        psi = psi.copy()
        psi[spun] = np.arctan2(moved[spun, 0], moved[spun, 1])
        P.append(pos.copy())
        S.append(psi.copy())
    return np.stack(P), np.stack(S)          # (steps+1, n_walks, 2), (.., n_walks)


def build_env(seed, *, size, obs_size, resolution, sensor, lags, pairs_per_lag,
              min_norm, max_norm, fixed_heading_extra=True, turn_sd_deg=None):
    """One env's paired dataset: observations, headings, positions, geometry.

    Returns a dict of arrays, all of length ``len(lags) * pairs_per_lag``.
    Positions are the *float* ones; the observation is taken at the snapped
    cell, and both are kept so the snap ceiling can be measured.
    """
    rng = np.random.default_rng(seed)
    wall_code = rng.choice([-1.0, 1.0], size=(4, size * resolution))

    maxlag = max(lags)
    steps = maxlag + 4
    per = int(np.ceil(pairs_per_lag / (steps - maxlag + 1)))
    P, S = simulate(rng, size=size, n_walks=per * 2, steps=steps,
                    min_norm=min_norm, max_norm=max_norm,
                    turn_sd_deg=turn_sd_deg)

    def obs_at(pos_f, psi):
        cell = np.clip(np.round(pos_f), 0, size - 1)
        if sensor == "dist":
            return raycast_range(size, cell[:, 0], cell[:, 1], psi, obs_size)
        return raycast_codes(wall_code, size, cell[:, 0], cell[:, 1], psi,
                             obs_size, resolution)

    out = {k: [] for k in ("p1", "p2", "psi1", "psi2", "lag")}
    for L in lags:
        i1 = np.concatenate([np.full(P.shape[1], t) for t in range(steps - L + 1)])
        j1 = np.tile(np.arange(P.shape[1]), steps - L + 1)
        take = rng.permutation(len(i1))[:pairs_per_lag]
        i1, j1 = i1[take], j1[take]
        out["p1"].append(P[i1, j1])
        out["p2"].append(P[i1 + L, j1])
        out["psi1"].append(S[i1, j1])
        out["psi2"].append(S[i1 + L, j1])
        out["lag"].append(np.full(len(i1), L))
    d = {k: np.concatenate(v) for k, v in out.items()}

    d["c1"] = np.clip(np.round(d["p1"]), 0, size - 1)
    d["c2"] = np.clip(np.round(d["p2"]), 0, size - 1)
    d["s1_free"] = obs_at(d["p1"], d["psi1"])
    d["s2_free"] = obs_at(d["p2"], d["psi2"])
    if fixed_heading_extra:
        z = np.zeros(len(d["psi1"]))
        d["s1_fix"] = obs_at(d["p1"], z)
        d["s2_fix"] = obs_at(d["p2"], z)
    hi = size - 0.5
    x, y = d["p1"][:, 0], d["p1"][:, 1]
    d["d_wall"] = np.minimum.reduce([x + 0.5, hi - x, y + 0.5, hi - y])
    d["seed"] = seed
    d["wall_code"] = wall_code
    return d


# ---------------------------------------------------------------------------
# Framings: what is the input, what is the target
# ---------------------------------------------------------------------------


def _rot_into(v, psi):
    """Express world vector ``v`` in the frame of heading ``psi``: (right, fwd).

    psi is clockwise from North, forward = (sin psi, cos psi), right =
    (cos psi, -sin psi) -- the same convention `raycast_codes` uses.
    """
    c, s = np.cos(psi), np.sin(psi)
    return np.stack([v[:, 0] * c - v[:, 1] * s,
                     v[:, 0] * s + v[:, 1] * c], axis=1)


def _rot_from(v, psi):
    """Inverse of `_rot_into`: an egocentric vector back to world frame."""
    c, s = np.cos(psi), np.sin(psi)
    return np.stack([v[:, 0] * c + v[:, 1] * s,
                     -v[:, 0] * s + v[:, 1] * c], axis=1)


def _shift_rays(s, dpsi):
    """Re-index a cone by ``dpsi`` so ray i looks along a fixed world bearing.

    Rays are evenly spaced over 120 deg (`cone_offsets`), so one ray is
    ``2*half/n`` radians and the shift is that many steps. Off-cone rays are
    zero-filled rather than wrapped: the cone is a 120 deg window, not a ring.
    """
    # Ray i of view 2 looks along psi2 + offset[i]; we want the ray that looks
    # along psi1 + offset[i], i.e. offset[j] = offset[i] - dpsi, j = i - k.
    # The opposite sign de-rotates the wrong way and is silent -- `_selftest`
    # below is what pins it down.
    n = s.shape[1]
    step = np.deg2rad(2 * FOVEAL_HALF_ANGLE_DEG) / n
    k = np.rint(dpsi / step).astype(np.int64)
    idx = np.arange(n)[None, :] - k[:, None]
    ok = (idx >= 0) & (idx < n)
    out = np.take_along_axis(np.asarray(s), np.clip(idx, 0, n - 1), axis=1)
    return np.where(ok, out, 0.0).astype(np.float32)


def cone_overlap(envs, lags, obs_size):
    """How much world do two views actually share, per lag?

    The cone is a 120 deg window and heading is the direction of travel, so
    two consecutive views point wherever two consecutive actions pointed. If
    those differ by more than 120 deg the two cones see **disjoint** parts of
    the world and no amount of decoding can relate them -- the pair simply
    does not contain the answer. This is a property of the aperture and the
    movement model, not of the code or the decoder, and it is what has to be
    checked before blaming either.
    """
    step = np.deg2rad(2 * FOVEAL_HALF_ANGLE_DEG) / obs_size
    out = {}
    dp = np.concatenate([d["psi2"] - d["psi1"] for d in envs])
    lg = np.concatenate([d["lag"] for d in envs])
    dp = np.abs((dp + np.pi) % (2 * np.pi) - np.pi)
    for L in lags:
        m = lg == L
        ov = np.clip(obs_size - np.rint(dp[m] / step), 0, obs_size)
        out[int(L)] = {
            "dpsi_med_deg": float(np.degrees(np.median(dp[m]))),
            "overlap_rays_med": float(np.median(ov)),
            "frac_zero_overlap": float((ov == 0).mean()),
        }
    return out


def selftest_shift(*, size=20, obs_size=60, resolution=4, n=4000, seed=0):
    """Two views from the SAME cell at different headings must align.

    De-rotation is the one operation here whose sign is invisible in the
    downstream score -- both signs produce a plausible-looking chance result.
    Standing still and turning changes nothing about the world, so after
    `_shift_rays` the overlapping rays must agree exactly. Returns the mean
    agreement on the overlap for the correct sign and the flipped one.
    """
    rng = np.random.default_rng(seed)
    wc = rng.choice([-1.0, 1.0], size=(4, size * resolution))
    x = rng.integers(0, size, n).astype(np.float64)
    y = rng.integers(0, size, n).astype(np.float64)
    step = np.deg2rad(2 * FOVEAL_HALF_ANGLE_DEG) / obs_size
    dpsi = rng.integers(-20, 21, n) * step          # exact multiples of a ray
    psi1 = rng.uniform(-np.pi, np.pi, n)
    s1 = raycast_codes(wc, size, x, y, psi1, obs_size, resolution)
    s2 = raycast_codes(wc, size, x, y, psi1 + dpsi, obs_size, resolution)
    out = {}
    for name, sgn in (("correct", 1.0), ("flipped", -1.0)):
        a = _shift_rays(s2, sgn * dpsi)
        ov = a != 0.0
        out[name] = float((np.asarray(s1)[ov] == a[ov]).mean())
    return out


def framing_views(d, framing):
    """(s1, s2, heading-side-info, target, snap-oracle target).

    ``d["_target"]``, if present, replaces the target. That is how
    `displacement_adaptation` fits a *residual* on top of a cross-env decoder
    without a second code path for featurizing.
    """
    dp = d["p2"] - d["p1"]
    dc = d["c2"] - d["c1"]
    if framing == "fixed":
        s1, s2, side, y, ys = d["s1_fix"], d["s2_fix"], None, dp, dc
    elif framing == "free":
        s1, s2, side, y, ys = d["s1_free"], d["s2_free"], None, dp, dc
    elif framing == "ego":
        dpsi = d["psi2"] - d["psi1"]
        side = np.stack([np.sin(dpsi), np.cos(dpsi)], axis=1)
        s1, s2 = d["s1_free"], d["s2_free"]
        y, ys = _rot_into(dp, d["psi1"]), _rot_into(dc, d["psi1"])
    elif framing == "world":
        side = np.stack([np.sin(d["psi1"]), np.cos(d["psi1"]),
                         np.sin(d["psi2"]), np.cos(d["psi2"])], axis=1)
        s1, s2, y, ys = d["s1_free"], d["s2_free"], dp, dc
    elif framing == "derot":
        # Free headings, but view 2's cone is shifted by dpsi so that ray i of
        # both views looks along the SAME world direction. This is the thing
        # an agent that knows its own heading would actually do, and it is
        # what separates "the rotation destroyed the information" from "the
        # decoder cannot undo a rotation it was handed". Rays with no
        # counterpart are zero-filled (a zero ray contributes 0 to every
        # product), which does leak |dpsi| through the padding pattern -- so
        # this framing must be read against the side-only control, exactly
        # like `ego` and `world`.
        dpsi = d["psi2"] - d["psi1"]
        side = np.stack([np.sin(dpsi), np.cos(dpsi)], axis=1)
        s1, y, ys = d["s1_free"], dp, dc
        s2 = _shift_rays(d["s2_free"], dpsi)
    elif framing == "derot_ego":
        # `derot` with the target in view 1's frame, which is the only way the
        # question is well posed. After alignment the features are expressed
        # relative to view 1's bearing, and psi1 itself is NOT supplied -- so
        # asking for a world-frame answer (what `derot` does) asks the decoder
        # to rotate by an angle it was never given. Chaining egocentric
        # displacements is also what an agent would actually do with them.
        dpsi = d["psi2"] - d["psi1"]
        side = np.stack([np.sin(dpsi), np.cos(dpsi)], axis=1)
        s1 = d["s1_free"]
        s2 = _shift_rays(d["s2_free"], dpsi)
        y, ys = _rot_into(dp, d["psi1"]), _rot_into(dc, d["psi1"])
    else:
        raise ValueError(framing)
    return s1, s2, side, d.get("_target", y), ys


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _xcorr(s1, s2, lags):
    n = s1.shape[1]
    out = []
    for k in lags:
        if k == 0:
            out.append((s1 * s2).mean(1))
        elif k > 0:
            out.append((s1[:, k:] * s2[:, :n - k]).mean(1))
        else:
            out.append((s1[:, :n + k] * s2[:, -k:]).mean(1))
    return np.stack(out, axis=1)


def featurize(kind, s1, s2, side):
    """Feature block for one chunk of pairs. float32, (n, d)."""
    if kind == "spec":
        # exactly the spec's ridge input
        f = [s1, s2, s1 - s2, s1 * s2]
    elif kind == "xcorr":
        # the shift-invariant compression of the agreement structure: it is
        # what a *translation* does to the cone, and it is codebook-free.
        lags = list(range(-24, 25))
        f = [s1 * s2, _xcorr(s1, s2, lags),
             _xcorr(s1, s1, list(range(1, 25))),
             _xcorr(s2, s2, list(range(1, 25))),
             s1.mean(1, keepdims=True), s2.mean(1, keepdims=True)]
    elif kind == "bilin":
        # the full second-order statistic. E[s1 s2^T] is a pure function of the
        # two poses and is identical in every env -- the complete
        # codebook-independent measurement, of which `xcorr` is a projection.
        f = [(s1[:, :, None] * s2[:, None, :]).reshape(len(s1), -1),
             _xcorr(s1, s1, list(range(1, 25))),
             _xcorr(s2, s2, list(range(1, 25)))]
    elif kind == "raw":
        f = [s1, s2]
    elif kind == "side":
        # the leak control: the heading side-information and NOTHING from the
        # cones. Heading is atan2 of the realized displacement, so at lag 1
        # this alone *is* the answer; the arm exists to make that visible
        # rather than let it inflate a sensory number.
        if side is None:
            raise ValueError("`side` features need a framing that supplies them")
        f = [(side[:, :, None] * side[:, None, :]).reshape(len(s1), -1)]
    else:
        raise ValueError(kind)
    if side is not None:
        f.append(side)
    f.append(np.ones((len(s1), 1)))
    return np.concatenate([np.asarray(x, dtype=np.float32).reshape(len(s1), -1)
                           for x in f], axis=1)


# ---------------------------------------------------------------------------
# Ridge by accumulated normal equations (never materializes X)
# ---------------------------------------------------------------------------

CHUNK = 4096


def _chunks(envs, framing, kind, standardize=None, shuffle_rng=None):
    for d in envs:
        s1, s2, side, y, _ = framing_views(d, framing)
        n = len(y)
        order = shuffle_rng.permutation(n) if shuffle_rng is not None else None
        for a in range(0, n, CHUNK):
            b = min(a + CHUNK, n)
            s2c = s2[order[a:b]] if order is not None else s2[a:b]
            X = featurize(kind, s1[a:b], s2c,
                          None if side is None else side[a:b])
            if standardize is not None:
                X = (X - standardize[0]) * standardize[1]
            yield X, y[a:b]


class Gram:
    def __init__(self, dim, ydim=2):
        self.XtX = np.zeros((dim, dim))
        self.Xty = np.zeros((dim, ydim))
        self.yty = np.zeros(ydim)
        self.ysum = np.zeros(ydim)
        self.n = 0

    def add(self, X, y):
        self.XtX += (X.T @ X).astype(np.float64)
        self.Xty += X.T.astype(np.float64) @ y
        self.yty += (y * y).sum(0)
        self.ysum += y.sum(0)
        self.n += len(y)

    def solve(self, alpha):
        A = self.XtX + alpha * np.eye(len(self.XtX))
        A[-1, -1] -= alpha                      # never penalize the intercept
        return np.linalg.solve(A, self.Xty)

    def sse(self, W):
        return (np.einsum("ij,ik,jk->k", self.XtX, W, W)
                - 2.0 * (self.Xty * W).sum(0) + self.yty)

    def r2(self, W):
        sse = self.sse(W)
        sst = self.yty - self.ysum ** 2 / self.n
        return float(1.0 - sse.sum() / sst.sum())


def fit_ridge(train_envs, framing, kind, alphas, *, val_frac=0.25, seed=0,
              shuffle=False, standardize=None):
    """Ridge with alpha chosen on held-out *envs*, then refit on all of them."""
    rng = np.random.default_rng(seed) if shuffle else None
    nv = max(1, int(round(val_frac * len(train_envs))))
    fit_e, val_e = train_envs[nv:], train_envs[:nv]
    if not fit_e:
        fit_e, val_e = train_envs, train_envs
    dim = None
    G_fit = G_val = None
    for envs, which in ((fit_e, "fit"), (val_e, "val")):
        for X, y in _chunks(envs, framing, kind, standardize, rng):
            if dim is None:
                dim = X.shape[1]
                G_fit, G_val = Gram(dim), Gram(dim)
            (G_fit if which == "fit" else G_val).add(X, y)
    scores = [(G_val.r2(G_fit.solve(a)), a) for a in alphas]
    best = max(scores)[1]
    G_all = Gram(dim)
    G_all.XtX = G_fit.XtX + G_val.XtX
    G_all.Xty = G_fit.Xty + G_val.Xty
    G_all.yty = G_fit.yty + G_val.yty
    G_all.ysum = G_fit.ysum + G_val.ysum
    G_all.n = G_fit.n + G_val.n
    return G_all.solve(best), best, scores


def predict(d, framing, kind, W, standardize=None):
    s1, s2, side, y, ysnap = framing_views(d, framing)
    out = np.empty_like(y)
    for a in range(0, len(y), CHUNK):
        b = min(a + CHUNK, len(y))
        X = featurize(kind, s1[a:b], s2[a:b],
                      None if side is None else side[a:b])
        if standardize is not None:
            X = (X - standardize[0]) * standardize[1]
        out[a:b] = X @ W
    return out, y, ysnap


# ---------------------------------------------------------------------------
# In-env ceiling: template matching against a table of visited cells
# ---------------------------------------------------------------------------


def nn_localize(d, *, size, obs_size, resolution, sensor, anchors=None,
                rng=None):
    """Decode each view by nearest template, then take the difference.

    Ridge is the wrong same-env control and the smoke run showed it: within one
    env the map cell -> cone is a random hash, which is *injective* (so position
    is perfectly recoverable) and *not smooth* (so no linear map recovers it).
    The decoder that matches the structure is a lookup -- which is exactly what
    an associative memory over stored views is. This is therefore both the
    honest in-env upper bound and the model of "experience in the new env":
    `anchors` is the set of cells already visited, and everything else is
    localized to the nearest one that has been.

    Views are read at psi=0, so the table is one entry per cell and the heading
    confound is out of the picture -- the most favourable case for the table.
    """
    wall_code = d["wall_code"]
    gx, gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    if anchors is not None:
        cells = cells[anchors]
    z = np.zeros(len(cells))
    if sensor == "dist":
        tbl = raycast_range(size, cells[:, 0], cells[:, 1], z, obs_size)
    else:
        tbl = raycast_codes(wall_code, size, cells[:, 0], cells[:, 1], z,
                            obs_size, resolution)
    tbl = np.asarray(tbl, dtype=np.float64)
    tn = tbl / (np.linalg.norm(tbl, axis=1, keepdims=True) + 1e-9)

    def loc(s):
        s = np.asarray(s, dtype=np.float64)
        sn = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9)
        return cells[(sn @ tn.T).argmax(axis=1)]

    return loc(d["s2_fix"]) - loc(d["s1_fix"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics(pred, true):
    """R^2 (joint, over both components), angular error, magnitude ratio.

    The angular column is the one that can see a collapsed decoder: `dpos` is
    near-isotropic, so a constant predictor sits at ~90 deg median however good
    its R^2 looks on a skewed sample.
    """
    sse = ((pred - true) ** 2).sum()
    sst = ((true - true.mean(0)) ** 2).sum()
    nrm = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    ok = nrm > 1e-9
    # A zero-length prediction carries no direction. Dropping those samples
    # would flatter a decoder that has collapsed -- and a table decoder DOES
    # collapse, returning the same anchor for every query, which is how this
    # first showed up (an empty `ang` and a crash). Score them as 90 deg,
    # i.e. exactly uninformative, and report how often it happens.
    ang = np.full(len(true), 90.0)
    cos = np.clip((pred[ok] * true[ok]).sum(1) / nrm[ok], -1.0, 1.0)
    ang[ok] = np.degrees(np.arccos(cos))
    return {
        "r2": float(1.0 - sse / sst),
        "frac_zero_pred": float(1.0 - ok.mean()),
        "ang_med": float(np.median(ang)),
        "ang_p90": float(np.percentile(ang, 90)),
        "frac_lt45": float((ang < 45.0).mean()),
        "mag_ratio": float(np.median(np.linalg.norm(pred, axis=1))
                           / max(1e-12, float(np.median(
                               np.linalg.norm(true, axis=1))))),
        "n": int(len(true)),
    }


def spread(rows, key):
    v = np.array([r[key] for r in rows])
    return float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))


# ---------------------------------------------------------------------------
# MLP (torch) -- the gap to ridge says whether the structure is linear
# ---------------------------------------------------------------------------


def fit_mlp(train_envs, framing, kind, *, hidden=512, epochs=30, lr=1e-3,
            device="cpu", seed=0, batch=1024, verbose=False):
    import torch

    torch.manual_seed(seed)
    Xs, ys = [], []
    for X, y in _chunks(train_envs, framing, kind):
        Xs.append(X)
        ys.append(y.astype(np.float32))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    cut = int(0.9 * len(X))
    dev = torch.device(device)
    Xtr = torch.from_numpy(X[:cut]).to(dev)
    ytr = torch.from_numpy(y[:cut]).to(dev)
    Xva = torch.from_numpy(X[cut:]).to(dev)
    yva = torch.from_numpy(y[cut:]).to(dev)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state = np.inf, None
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        net.train()
        for a in range(0, len(Xtr), batch):
            idx = perm[a:a + batch]
            loss = torch.nn.functional.mse_loss(net(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(torch.nn.functional.mse_loss(net(Xva), yva))
        if v < best:
            best = v
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        if verbose:
            print(f"      mlp ep{ep:3d} val_mse {v:.4f}")
    net.load_state_dict(best_state)
    return net, (mu, sd)


def predict_mlp(d, framing, kind, net, norm, device="cpu"):
    import torch

    mu, sd = norm
    s1, s2, side, y, ysnap = framing_views(d, framing)
    out = np.empty_like(y)
    dev = torch.device(device)
    with torch.no_grad():
        for a in range(0, len(y), CHUNK):
            b = min(a + CHUNK, len(y))
            X = featurize(kind, s1[a:b], s2[a:b],
                          None if side is None else side[a:b])
            X = torch.from_numpy((X - mu) / sd).to(dev)
            out[a:b] = net(X).cpu().numpy()
    return out, y, ysnap


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def evaluate(test_envs, framing, predict_fn, breakdown=True):
    """Per-env metrics plus breakdowns pooled over envs."""
    rows, pooled = [], {"pred": [], "true": [], "lag": [], "d_wall": [],
                        "p1": []}
    for d in test_envs:
        pred, true, _ = predict_fn(d, framing)
        rows.append(metrics(pred, true))
        if breakdown:
            pooled["pred"].append(pred)
            pooled["true"].append(true)
            pooled["lag"].append(d["lag"])
            pooled["d_wall"].append(d["d_wall"])
            pooled["p1"].append(d["p1"])
    out = {"per_env": rows}
    if breakdown:
        P = {k: np.concatenate(v) for k, v in pooled.items()}
        by_lag = {}
        for L in np.unique(P["lag"]):
            m = P["lag"] == L
            by_lag[int(L)] = metrics(P["pred"][m], P["true"][m])
            by_lag[int(L)]["dpos_med"] = float(
                np.median(np.linalg.norm(P["true"][m], axis=1)))
        out["by_lag"] = by_lag
        edges = [0.5, 1.5, 3.0, 5.0, 7.5, 10.01]
        by_wall = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (P["d_wall"] >= lo) & (P["d_wall"] < hi)
            if m.sum() > 50:
                by_wall[f"{lo}-{hi}"] = metrics(P["pred"][m], P["true"][m])
        out["by_wall"] = by_wall
        by_quad = {}
        cx = P["p1"][:, 0] > 9.5
        cy = P["p1"][:, 1] > 9.5
        for qx in (False, True):
            for qy in (False, True):
                m = (cx == qx) & (cy == qy)
                if m.sum() > 50:
                    by_quad[f"x{'hi' if qx else 'lo'}_y{'hi' if qy else 'lo'}"] = \
                        metrics(P["pred"][m], P["true"][m])
        out["by_quad"] = by_quad
    return out


def _fmt(m):
    return (f"R2 {m['r2']:>6.3f}  ang_med {m['ang_med']:>5.1f}  "
            f"p90 {m['ang_p90']:>5.1f}  <45deg {m['frac_lt45']:>5.1%}  "
            f"|p|/|t| {m['mag_ratio']:>5.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--obs_size", type=int, default=60)
    p.add_argument("--resolution", type=int, default=4)
    p.add_argument("--min_norm", type=float, default=0.5)
    p.add_argument("--max_norm", type=float, default=2.0)
    p.add_argument("--turn_sd_deg", type=float, default=None,
                   help="per-step turn sd. Omit for uniform directions (the "
                        "neutral reference); a small value is a straight, "
                        "persistent walk, which is what PERSISTENCE_BONUS "
                        "pays the policy for and what decides how much two "
                        "consecutive 120-deg cones overlap")
    p.add_argument("--train_envs", type=int, default=48)
    p.add_argument("--test_envs", type=int, default=48)
    p.add_argument("--train_pairs", type=int, default=1200,
                   help="pairs per lag per training env")
    p.add_argument("--test_pairs", type=int, default=3000,
                   help="pairs per lag per test env")
    p.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--framings", nargs="+",
                   default=["fixed", "free", "derot", "ego", "world"])
    p.add_argument("--inenv_features", nargs="+", default=["xcorr"],
                   help="feature sets for the same-env ridge control; the LU\n"
                        "solve is cubic in the feature count and this loop runs\n"
                        "once per test env, so `bilin` here is expensive")
    p.add_argument("--no_inenv", action="store_true",
                   help="skip the in-env controls (NN table, same-env ridge)")
    p.add_argument("--features", nargs="+", default=["spec", "xcorr", "bilin"])
    p.add_argument("--sensors", nargs="+", default=["code"])
    p.add_argument("--mlp", action="store_true")
    p.add_argument("--mlp_features", nargs="+", default=["raw", "bilin"])
    p.add_argument("--mlp_epochs", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    if args.smoke:
        args.train_envs, args.test_envs = 6, 4
        args.train_pairs, args.test_pairs = 300, 400
        args.lags = [1, 4]
        args.mlp_epochs = 4

    alphas = np.logspace(-1, 6, 8)
    results = {"args": vars(args), "arms": []}

    print(f"{args.size}x{args.size}, {args.obs_size} rays, wall_res "
          f"{args.resolution}, |a| in [{args.min_norm}, {args.max_norm}], "
          f"lags {args.lags}")
    print(f"train {args.train_envs} envs x {args.train_pairs}/lag, "
          f"test {args.test_envs} envs x {args.test_pairs}/lag (HELD OUT)")
    st = selftest_shift(size=args.size, obs_size=args.obs_size,
                        resolution=args.resolution)
    results["selftest_shift"] = st
    print(f"de-rotation sign guard: agreement on the overlap = "
          f"{st['correct']:.3f} correct / {st['flipped']:.3f} flipped "
          f"(must be 1.000 / ~0.5)\n")

    for sensor in args.sensors:
        t0 = time.time()
        tr = [build_env(args.seed * 100000 + i, size=args.size,
                        obs_size=args.obs_size, resolution=args.resolution,
                        sensor=sensor, lags=args.lags,
                        pairs_per_lag=args.train_pairs,
                        min_norm=args.min_norm, max_norm=args.max_norm,
                        turn_sd_deg=args.turn_sd_deg)
              for i in range(args.train_envs)]
        te = [build_env(args.seed * 100000 + 50000 + i, size=args.size,
                        obs_size=args.obs_size, resolution=args.resolution,
                        sensor=sensor, lags=args.lags,
                        pairs_per_lag=args.test_pairs,
                        min_norm=args.min_norm, max_norm=args.max_norm,
                        turn_sd_deg=args.turn_sd_deg)
              for i in range(args.test_envs)]
        print(f"[{sensor}] built {len(tr)}+{len(te)} envs in "
              f"{time.time() - t0:.0f}s")
        ov = cone_overlap(te, sorted(args.lags), args.obs_size)
        results["cone_overlap"] = ov
        print("  cone overlap between the two views (aperture = 120 deg, "
              "heading = direction of travel):")
        print(f"    {'lag':<20s}" + "".join(f"{L:>10d}" for L in ov))
        print(f"    {'median |dpsi| deg':<20s}"
              + "".join(f"{v['dpsi_med_deg']:>10.1f}" for v in ov.values()))
        print(f"    {'median shared rays':<20s}"
              + "".join(f"{v['overlap_rays_med']:>10.0f}" for v in ov.values()))
        print(f"    {'frac DISJOINT views':<20s}"
              + "".join(f"{v['frac_zero_overlap']:>9.1%}"
                        for v in ov.values()))


        for framing in args.framings:
            print(f"\n=== sensor={sensor}  framing={framing} {'=' * 42}")
            arms: dict = {}

            def add(name, res, meta=None):
                arms[name] = res
                results["arms"].append({"sensor": sensor, "framing": framing,
                                        "decoder": name, **(meta or {}), **res})

            has_side = framing_views(te[0], framing)[2] is not None

            # --- instrumentation guards: the metric must separate these ----
            add("snap-oracle CEILING",
                evaluate(te, framing, lambda d, f: (framing_views(d, f)[4],
                                                    framing_views(d, f)[3], None)))
            mn = np.concatenate([framing_views(d, framing)[3] for d in tr]).mean(0)
            add("constant CHANCE",
                evaluate(te, framing,
                         lambda d, f: (np.tile(mn, (len(d["lag"]), 1)),
                                       framing_views(d, f)[3], None)))

            # --- the leak control: heading side-info with NO cone at all ---
            if has_side:
                Wl, _, _ = fit_ridge(tr, framing, "side", alphas, seed=args.seed)
                add("side-only LEAK",
                    evaluate(te, framing,
                             lambda d, f, W=Wl: predict(d, f, "side", W)))

            # --- cross-env decoders + their shuffled controls --------------
            for kind in args.features:
                t0 = time.time()
                W, best_a, _ = fit_ridge(tr, framing, kind, alphas,
                                         seed=args.seed)
                add(f"ridge/{kind}",
                    evaluate(te, framing,
                             lambda d, f, W=W, k=kind: predict(d, f, k, W)),
                    {"alpha": float(best_a), "fit_s": time.time() - t0})
                Ws, _, _ = fit_ridge(tr, framing, kind, alphas, seed=args.seed,
                                     shuffle=True)
                add(f"  shuf/{kind}",
                    evaluate(te, framing,
                             lambda d, f, W=Ws, k=kind: predict(d, f, k, W)))
                print(f"  fitted ridge/{kind} ({time.time() - t0:.0f}s, "
                      f"alpha {best_a:.0e})")

            if args.mlp:
                for kind in args.mlp_features:
                    t0 = time.time()
                    net, nrm = fit_mlp(tr, framing, kind, device=args.device,
                                       epochs=args.mlp_epochs, seed=args.seed)
                    add(f"mlp/{kind}",
                        evaluate(te, framing,
                                 lambda d, f, n=net, m=nrm, k=kind:
                                 predict_mlp(d, f, k, n, m, device=args.device)))
                    print(f"  fitted mlp/{kind} ({time.time() - t0:.0f}s)")

            # --- in-env controls: what unlimited experience here would buy -
            if not args.no_inenv:
                def nn_pred(d, f):
                    w = nn_localize(d, size=args.size, obs_size=args.obs_size,
                                    resolution=args.resolution, sensor=sensor)
                    y = framing_views(d, f)[3]
                    # ego-frame framings need the world-frame table prediction
                    # rotated into view 1's frame to match their target
                    if f in ("ego", "derot_ego"):
                        w = _rot_into(w, d["psi1"])
                    return w, y, None
                add("NN-table in-env", evaluate(te, framing, nn_pred))

                for kind in args.inenv_features:
                    same = []
                    for d in te:
                        n = len(d["lag"])
                        idx = np.random.default_rng(1).permutation(n)
                        a1, a2 = int(0.55 * n), int(0.7 * n)

                        def sub(sl, d=d, idx=idx, n=n):
                            return {k: (v[idx[sl]]
                                        if isinstance(v, np.ndarray) and len(v) == n
                                        else v) for k, v in d.items()}
                        # two pseudo-envs so alpha stays out of sample;
                        # fit_ridge validates on the FIRST, fits on the rest
                        Wi, _, _ = fit_ridge([sub(slice(a1, a2)), sub(slice(a1))],
                                             framing, kind, alphas,
                                             val_frac=0.5, seed=args.seed)
                        pr, tv, _ = predict(sub(slice(a2, None)), framing,
                                            kind, Wi)
                        same.append(metrics(pr, tv))
                    arms[f"  same-env/{kind}"] = {"per_env": same}
                    results["arms"].append({"sensor": sensor, "framing": framing,
                                            "decoder": f"sameenv_{kind}",
                                            "per_env": same})

            # --- report ----------------------------------------------------
            print(f"\n  {'':<22s}{'R2 (median over test envs) [p10,p90]':^34s}"
                  f"{'angular err, deg':^30s}")
            print(f"  {'arm':<22s}{'':^34s}{'':^30s}")
            for name, res in arms.items():
                r2 = spread(res["per_env"], "r2")
                am = spread(res["per_env"], "ang_med")
                f45 = spread(res["per_env"], "frac_lt45")
                z = spread(res["per_env"], "frac_zero_pred")
                print(f"  {name:<22s}{r2[0]:>8.3f} [{r2[1]:>6.3f},{r2[2]:>6.3f}]"
                      f"      {am[0]:>7.1f} [{am[1]:>5.1f},{am[2]:>5.1f}]"
                      f"   <45deg {f45[0]:>5.1%}  zero-pred {z[0]:>5.1%}")

            lags = sorted(args.lags)
            print(f"\n  per-lag  (R2 / median angular error).  ||dpos|| median: "
                  + "  ".join(
                      f"L{L}={arms['snap-oracle CEILING']['by_lag'][L]['dpos_med']:.2f}"
                      for L in lags))
            print(f"  {'arm':<22s}" + "".join(f"{f'lag {L}':>18s}" for L in lags))
            for name, res in arms.items():
                if "by_lag" not in res:
                    continue
                print(f"  {name:<22s}" + "".join(
                    f"{res['by_lag'][L]['r2']:>10.3f} /{res['by_lag'][L]['ang_med']:>6.1f}"
                    for L in lags))

            best = max(
                ((spread(r["per_env"], "r2")[0], n, r) for n, r in arms.items()
                 if n.startswith(("ridge/", "mlp/"))), default=None)
            if best is not None:
                _, name, a = best
                print(f"\n  breakdown of the best cross-env arm ({name})")
                print(f"    {'dist to wall':<16s}" +
                      "".join(f"{k:>18s}" for k in a["by_wall"]))
                print(f"    {'  R2 / ang_med':<16s}" +
                      "".join(f"{m['r2']:>10.3f} /{m['ang_med']:>6.1f}"
                              for m in a["by_wall"].values()))
                print(f"    {'quadrant':<16s}" +
                      "".join(f"{k:>18s}" for k in a["by_quad"]))
                print(f"    {'  R2 / ang_med':<16s}" +
                      "".join(f"{m['r2']:>10.3f} /{m['ang_med']:>6.1f}"
                              for m in a["by_quad"].values()))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
