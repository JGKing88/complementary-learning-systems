"""Sec 21: learning L1 locally and online, inside one environment.

The animal walks inside ONE square box (side 10, 13, 20 or 40, corner at
(700, 900), reflecting at the walls; ``gwn_local_gain.walk_in_env``). The
reference condition ``scat`` uses independent single steps over the 5%
scattered environments, as Sec 20 step 1 did. Every step has an axis e (x or
y); a -e step is read backwards as a +e step, so each step gives a start code
g0 and an end code g1 with g1 = T_e g0 for the module's shift T_e.

Route 1, a velocity-gated temporal Hebbian estimate of the shift. Per module
and axis the walk streams A_e = sum g1 g0^T and C_e = sum g0 g0^T (computed
here as prefix sums, which equal the streamed sums).
  1a: T_e = A_e with each row scaled to sum 1 (Hebbian only).
  1b: T_e = A_e (C_e + rho I)^-1 (the Sec 20 estimate, accumulated online).
  1c: a delta rule for the shift, online and local to each synapse:
      T_e <- T_e + 0.5 (g1 - T_e g0) g0^T / |g0|^2. It tends to 1b's estimate.
Then the left eigenvectors of T_x + 0.618 T_y and their conjugates, offline
(``gwn_l1_learn.eig_pairs``).

Route 2, lock-in rules. Per module, units with complex weights w (z = w^T g)
and intrinsic turns (phi_x, phi_y), drawn uniformly; no unit sees another. On
a step along e, with z1 = w^T g1 and |g0|^2 normalising the input:
  2a  plain Hebbian:  w <- normalise(w + eta e^{-i phi_e} z1 g0 / |g0|^2)
  2b  predictive:     u_e <- u_e + eta_u (z1 - u_e^T g0) g0 / |g0|^2
                      w   <- normalise((1 - eta) w + eta e^{-i phi_e} u_e)
                      (u_e, a second set of the unit's synapses, learns to
                      predict the unit's next output; it tends to T_e^T w)
  2c  2b plus a phase-locked turn:
                      phi_e <- phi_e + eta_phi angle(z1 conj(z0) e^{-i phi_e})
  2d  2a plus the same phase-locked turn (no predictor u).
eta_u = min(4 eta, 1), eta_phi = eta. Each eta group has 6 l^2 units per
module (2 l^2 under GWN_SMOKE); fewer units = the first q l^2 of a group (units are independent).

Measures per learner: purity of each unit or eigenvector (``purity``: the
largest share of its power in one module pattern), the share above 0.99, and
pattern coverage (the share of the l^2 patterns held by a unit above 0.99,
counting conjugates, which L1 also emits). Downstream: L1 assembled with
conjugates (``_assemble``), 400k random triples, gains from this walk's own
steps (``local_slowness``), and ``World.metrics``.

    python -m analysis.hopfield_probe.gwn_l1_online run --task I --out DIR
    python -m analysis.hopfield_probe.gwn_l1_online summary --out DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

from .gwn_check import (DEV, LAMBDAS, NU, R, SMOKE, World, _dump, _line, _noisy, build, l1_out,
                        sample_steps)
from .gwn_encoder import GAIN_RULES, bump_strength, random_triples, save_gwn, unit_waves
from .gwn_l1_learn import OFFS, _assemble, codes_at, eig_pairs
from .gwn_local_gain import walk_in_env
from .rp_encoder import grid_codes_torch

NPOS = 1716
LP = max(l * l for l in LAMBDAS)
BOX = (700, 900)
ETAS = (0.05, 0.5) if SMOKE else (0.01, 0.05, 0.2, 0.5)
QS = (1, 2) if SMOKE else (1, 2, 4, 6)          # units per module = q l^2
QMAX = QS[-1]
KPER = QMAX * LP                                 # units per eta group (padded module)
T_CHECK = (1_000, 4_000) if SMOKE else (1_000, 10_000, 100_000)
RIDGE = 1e-10
LMS_RATE = 0.5
RULES = ("2a", "2b", "2c", "2d")
NOISE = 0.01
# (side, bump, cell noise); side "scat" = single steps over the 5% environments.
# A box of side s never makes the step from phase s-1 to phase s (it would leave
# the box), so a module with l >= s has one transition per row unseen: side 13
# misses module 13's wrap, side 14 is the smallest that sees every transition.
SIDES = (10, 13, 14, 20, 40, "scat")
CONDS = ([(14, 0.10, 0.0), ("scat", 0.25, 0.0), (14, 0.10, NOISE)] if SMOKE else
         [(s, f, 0.0) for f in (0.10, 0.25) for s in SIDES]
         + [(s, 0.10, NOISE) for s in (14, 20, 40, "scat")])
SAVE_SIDES = (14, 20)


def cond_name(side, fwhm, noise):
    return f"side{side}_fwhm{fwhm:g}" + (f"_noise{noise:g}" if noise else "")


# --- the steps ------------------------------------------------------------------

def make_steps(side, fwhm, noise, T, W, seed):
    """Start/end codes of every step (+e orientation), padded to (T, 3, LP), its
    axis (T,), and the positions (p0, p1) as walked (for the gains)."""
    if side == "scat":
        p0, p1 = sample_steps(W.rects, T, seed)
    else:
        pos = walk_in_env(*BOX, side, T, seed)
        p0, p1 = pos[:-1], pos[1:]
    d = p1 - p0
    ax = (d[:, 1] != 0).astype(np.int64)
    fwd = torch.tensor(d.sum(1) > 0, device=DEV)[:, None]
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)

    def codes(p):
        g = grid_codes_torch(torch.as_tensor(p[:, 0], device=DEV), torch.as_tensor(p[:, 1], device=DEV),
                             LAMBDAS, fwhm)
        if noise > 0:
            g = g + noise * torch.randn(g.shape, generator=gen).to(DEV)
        return g

    if side == "scat":
        G0, G1 = codes(p0), codes(p1)
    else:
        G = codes(pos)                           # one noisy code per moment, shared by two steps
        G0, G1 = G[:-1], G[1:]
    IN, OUT = torch.where(fwd, G0, G1), torch.where(fwd, G1, G0)
    del G0, G1

    def pad(X):
        Y = torch.zeros(len(X), len(LAMBDAS), LP, device=DEV)
        for m, l in enumerate(LAMBDAS):
            Y[:, m, :l * l] = X[:, OFFS[m]:OFFS[m + 1]]
        return Y

    return pad(IN), pad(OUT), ax, (p0, p1)


# --- route 1 --------------------------------------------------------------------

def exact_shift(l, ax):
    """The permutation P with g(p + e) = P g(p), cell index a = ix l + iy."""
    P = np.zeros((l * l, l * l))
    for ix in range(l):
        for iy in range(l):
            src = (((ix - 1) % l) * l + iy) if ax == 0 else (ix * l + (iy - 1) % l)
            P[ix * l + iy, src] = 1
    return P


@torch.no_grad()
def lms_shift(IN, OUT, ax, checks, rate=LMS_RATE):
    """1c: T_e <- T_e + rate (g1 - T_e g0) g0^T / |g0|^2 on every step along e (each
    synapse T_ab changes by the error at cell a times the activity of cell b);
    {T: (2, M, LP, LP) numpy}."""
    T, M, Lp = IN.shape
    Ts = torch.zeros(2, M, Lp, Lp, device=DEV)
    inv_n = 1.0 / (IN ** 2).sum(-1)
    snaps = {}
    for t in range(T):
        a = int(ax[t])
        g0 = IN[t]
        err = OUT[t] - torch.bmm(Ts[a], g0[:, :, None])[:, :, 0]
        Ts[a] += rate * (err * inv_n[t][:, None])[:, :, None] * g0[:, None, :]
        if t + 1 in checks:
            snaps[t + 1] = Ts.cpu().numpy().copy()
    return snaps


def route1(IN, OUT, ax, T, lms=None):
    """{'1a', '1b', '1c'}: per module (pairs, max |T - P|)."""
    out = {"1a": [], "1b": [], "1c": []}
    for m, l in enumerate(LAMBDAS):
        L = l * l
        X0 = IN[:T, m, :L].double().cpu().numpy(); X1 = OUT[:T, m, :L].double().cpu().numpy()
        Ts = {"1a": [], "1b": []}
        for e in (0, 1):
            s = ax[:T] == e
            A = X1[s].T @ X0[s]
            C = X0[s].T @ X0[s]
            Ts["1a"].append(A / np.where(A.sum(1, keepdims=True) > 0, A.sum(1, keepdims=True), 1))
            Cr = C + RIDGE * np.trace(C) / L * np.eye(L)
            Ts["1b"].append(np.linalg.solve(Cr, A.T).T)
        Ts["1c"] = [lms[e, m, :L, :L].astype(np.float64) for e in (0, 1)]
        for k in ("1a", "1b", "1c"):
            err = max(float(np.abs(Ts[k][e] - exact_shift(l, e)).max()) for e in (0, 1))
            out[k].append((eig_pairs(*Ts[k]), err))
    return out


# --- route 2 --------------------------------------------------------------------

def _proj(g, W):
    """z = w^T g for every unit: g (M, LP) float, W (M, LP, K) complex -> (M, K)."""
    M, L, K = W.shape
    z = torch.bmm(g[:, None, :], torch.view_as_real(W).reshape(M, L, 2 * K))
    return torch.view_as_complex(z.reshape(M, K, 2))


def _init_units(M, Lp, K, gen):
    """Random unit-norm complex weights (zero on padding), uniform turns, and the
    predictors u (None = zeros)."""
    mask = torch.zeros(M, Lp, 1)
    for m, l in enumerate(LAMBDAS):
        mask[m, :l * l] = 1
    W = torch.complex(torch.randn(M, Lp, K, generator=gen), torch.randn(M, Lp, K, generator=gen)) * mask
    phi = (torch.rand(2, M, K, generator=gen) * 2 - 1) * math.pi
    return W, phi, None


@torch.no_grad()
def lock_in(IN, OUT, ax, rule, seed, checks, phi_ratio=1.0):
    """Run one rule over the steps; {T: (W (M, LP, K) complex numpy, phi (2, M, K))}."""
    T, M, Lp = IN.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    eta = torch.tensor(np.repeat(ETAS, KPER), dtype=torch.float32, device=DEV)
    K = len(eta)
    W, phi, U0 = _init_units(M, Lp, K, gen)
    W, phi = W.to(DEV), phi.to(DEV)
    W /= torch.linalg.vector_norm(W, dim=1, keepdim=True)
    rot = torch.exp(-1j * phi)
    hebb, pll = rule in ("2a", "2d"), rule in ("2c", "2d")
    U = None
    if not hebb:
        U = (torch.zeros(2, M, Lp, K, dtype=torch.complex64) if U0 is None else U0).to(DEV)
    eta_u = (4 * eta).clamp(max=1.0)
    inv_n = 1.0 / (IN ** 2).sum(-1)                                 # (T, M)
    snaps = {}
    for t in range(T):
        a = int(ax[t])
        g0 = IN[t] * inv_n[t][:, None]
        z1 = _proj(OUT[t], W)
        if pll:
            z0 = _proj(IN[t], W)
        if hebb:
            W += (eta * rot[a] * z1)[:, None, :] * g0[:, :, None]
        else:
            pred = _proj(IN[t], U[a])
            U[a] += (eta_u * (z1 - pred))[:, None, :] * g0[:, :, None]
            W.mul_(1 - eta).add_((eta * rot[a])[:, None, :] * U[a])
        if pll:
            phi[a] += (phi_ratio * eta) * torch.angle(z1 * z0.conj() * rot[a])
            rot[a] = torch.exp(-1j * phi[a])
        W /= torch.linalg.vector_norm(W, dim=1, keepdim=True).clamp_min(1e-30)
        if t + 1 in checks:
            snaps[t + 1] = (W.cpu().numpy().copy(), phi.cpu().numpy().copy())
    return snaps


def unit_purity(w, l):
    """Purity and dominant pattern (flat jx l + jy) of complex weight columns (L, K)."""
    F = np.abs(np.fft.fft2(w.T.reshape(-1, l, l))) ** 2
    F = F.reshape(len(F), -1)
    return F.max(1) / F.sum(1), F.argmax(1)


def conj_index(j, l):
    return ((-(j // l)) % l) * l + (-(j % l)) % l


def strengths(l, fwhm):
    b = bump_strength(l, fwhm, np.arange(l))
    return np.outer(b, b).ravel() / b[0] ** 2


def group_stats(Wsnap, g, fwhm):
    """Stats of eta group g at one snapshot, per module and pooled."""
    res = {"modules": {}}
    pure_all = []
    for m, l in enumerate(LAMBDAS):
        L = l * l
        pur, j = unit_purity(Wsnap[m][:L][:, g * KPER + np.arange(QMAX * L)], l)
        S = strengths(l, fwhm)
        weak = np.argsort(S)[:L // 2]
        r = {"purity_median": float(np.median(pur)), "frac_pure_0.99": float((pur > 0.99).mean()),
             "frac_pure_0.9": float((pur > 0.9).mean())}
        for q in QS:
            n = q * L
            found = set(j[:n][pur[:n] > 0.99].tolist())
            both = found | {int(conj_index(x, l)) for x in found}
            r[f"cov_q{q}"] = len(found) / L
            r[f"cov_conj_q{q}"] = len(both) / L
            if q == QMAX:
                r["cov_conj_weak_half"] = len(both & set(weak.tolist())) / len(weak)
        cnt = np.bincount(np.concatenate([j, conj_index(j, l)]), minlength=L)
        r["count_cv"] = float(cnt.std() / cnt.mean())
        r["share_on_const"] = float((j == 0).mean())
        res["modules"][str(l)] = r
        pure_all.append(pur)
    pa = np.concatenate(pure_all)
    res["frac_pure_0.99"] = float((pa > 0.99).mean())
    res["purity_median"] = float(np.median(pa))
    for q in QS:
        res[f"cov_conj_q{q}_min"] = min(res["modules"][str(l)][f"cov_conj_q{q}"] for l in LAMBDAS)
    return res


def route2_pairs(Wsnap, g, dedup):
    """Per-module (wc, ws) lists for eta group g: every unit and its conjugate, or
    (``dedup``) one purest source per pattern."""
    per = []
    for m, l in enumerate(LAMBDAS):
        L = l * l
        w = Wsnap[m][:L][:, g * KPER + np.arange(QMAX * L)]
        if not dedup:
            per.append([p for k in range(w.shape[1])
                        for p in ((w[:, k].real, w[:, k].imag), (w[:, k].real, -w[:, k].imag))])
            continue
        pur, j = unit_purity(w, l)
        best = {}
        for k in range(w.shape[1]):
            for vec, jj in ((w[:, k], int(j[k])), (np.conj(w[:, k]), int(conj_index(j[k], l)))):
                if pur[k] > best.get(jj, (-1.0, None))[0]:
                    best[jj] = (pur[k], vec)
        per.append([(v.real, v.imag) for _, v in best.values()])
    return per


# --- route 1 stats --------------------------------------------------------------

def pairs_stats(per, fwhm):
    res = {"modules": {}}
    pall = []
    for m, l in enumerate(LAMBDAS):
        w = np.stack([c + 1j * s for c, s in per[m]], 1)
        pur, j = unit_purity(w, l)
        found = set(j[pur > 0.99].tolist())
        res["modules"][str(l)] = {"pairs": len(per[m]), "frac_pure_0.99": float((pur > 0.99).mean()),
                                  "purity_median": float(np.median(pur)), "cov_conj": len(found) / (l * l)}
        pall.append(pur)
    pa = np.concatenate(pall)
    res["frac_pure_0.99"] = float((pa > 0.99).mean())
    res["purity_median"] = float(np.median(pa))
    res["cov_conj_min"] = min(r["cov_conj"] for r in res["modules"].values())
    return res


# --- downstream -----------------------------------------------------------------

def local_slowness(l1, triples, steps, fwhm, noise=0.0, chunk=16384):
    """``gwn_check.slowness`` without its across-position variance test, which one
    small box cannot do (a slow wave barely varies inside it): v = mean change per
    step / mean power over this walk's steps; a unit that does not change at all
    (after the at-rest correction when noisy) gets v = inf."""
    fns = [None] * 3
    if noise > 0:
        _noisy.fwhm = fwhm
        fns = [_noisy(noise, s) for s in (101, 202, 303)]
    c0, s0 = (t.float() for t in l1_out(l1, steps[0], fwhm, fns[0]))
    c1, s1 = (t.float() for t in l1_out(l1, steps[1], fwhm, fns[1]))
    if noise > 0:
        c2, s2 = (t.float() for t in l1_out(l1, steps[0], fwhm, fns[2]))
    U = len(triples)
    v = torch.empty(U, dtype=torch.float64, device=DEV)
    pw = torch.empty(U, dtype=torch.float64, device=DEV)
    for a in range(0, U, chunk):
        tr = triples[a:a + chunk].to(DEV)
        C0, S0 = unit_waves(c0, s0, tr)
        C1, S1 = unit_waves(c1, s1, tr)
        p = ((C0 ** 2 + S0 ** 2).mean(0) + (C1 ** 2 + S1 ** 2).mean(0)) / 2
        step = ((C1 - C0) ** 2 + (S1 - S0) ** 2).mean(0)
        ps = p
        if noise > 0:
            C2, S2 = unit_waves(c2, s2, tr)
            rest = ((C2 - C0) ** 2 + (S2 - S0) ** 2).mean(0)
            step = step - rest
            ps = (p - rest / 2).clamp_min(1e-12)
        vk = step / ps
        vk = torch.where(step <= 1e-12 * ps, torch.full_like(vk, float("inf")), vk)
        v[a:a + chunk] = vk.double()
        pw[a:a + chunk] = p.double()
    return v, pw


def triple_n(per, triples, l1):
    """Plane-wave frequency of each triple from its pairs' dominant patterns."""
    pat = []
    for m, l in enumerate(LAMBDAS):
        w = np.stack([c + 1j * s for c, s in per[m]], 1)
        _, j = unit_purity(w, l)
        pat.append(np.stack([j // l, j % l], 1))
    pat = torch.tensor(np.concatenate(pat))
    c = torch.tensor([NPOS // l for l in LAMBDAS])
    n = (pat[triples.cpu()] * c[None, :, None]).sum(1) % NPOS
    return torch.where(n > NPOS // 2, n - NPOS, n)


def downstream(per, codes, steps, fwhm, noise, W, tag, meta, save=None):
    t0 = time.time()
    l1, _ = _assemble(per, codes, fwhm)
    tr = random_triples(l1, NU, seed=0)
    v, pw = local_slowness(l1, tr, steps, fwhm, noise)
    enc, info = build(l1, tr, v, pw, fwhm, meta=meta)
    w2 = (GAIN_RULES["gauss"](v, R) / pw.clamp_min(1e-12).sqrt()).double() ** 2
    n = triple_n(per, tr, l1)
    info["share_w2_n0"] = float(w2[(n == 0).all(1).to(DEV)].sum() / w2.sum())
    info["l1_pairs"] = [len(p) for p in per]
    m = W.metrics(enc)
    _line(tag, info, m)
    if save:
        save_gwn(enc.cpu(), save)
    return {**info, **m, "seconds": time.time() - t0}


def best_group(stats_by_eta):
    """The eta with the most pure units, ties broken by coverage."""
    return max(stats_by_eta, key=lambda e: (round(stats_by_eta[e]["frac_pure_0.99"], 2),
                                            stats_by_eta[e][f"cov_conj_q{QMAX}_min"]))


# --- one condition --------------------------------------------------------------

def run(task, out):
    side, fwhm, noise = CONDS[task]
    name = cond_name(side, fwhm, noise)
    os.makedirs(os.path.join(out, "encoders"), exist_ok=True)
    print(f"=== {name} on {DEV}", flush=True)
    W = World(fwhm=fwhm)
    Tmax = max(T_CHECK)
    seed = 1000 + task
    IN, OUT, ax, (p0, p1) = make_steps(side, fwhm, noise, Tmax, W, seed)
    if side == "scat":
        vis = np.argwhere(W.mask)
        vis = vis[np.random.default_rng(3).permutation(len(vis))[:5_000 if SMOKE else 50_000]]
        cells = vis
        gsteps = W.steps
    else:
        cells = np.unique(np.concatenate([p0, p1[-1:]]), axis=0)
        idx = np.linspace(0, Tmax - 1, 512 if SMOKE else 8192).astype(int)
        gsteps = (p0[idx], p1[idx])
    codes = codes_at(cells, fwhm)
    res = {"side": side, "fwhm": fwhm, "noise": noise, "steps": Tmax, "cells_visited": int(len(cells)),
           "x_steps": int((ax == 0).sum()), "route1": {}, "route2": {}, "downstream": {}}
    save_ok = side in SAVE_SIDES

    def enc_path(k):
        return os.path.join(out, "encoders", f"{name}_{k}.pt") if save_ok else None

    # route 1
    t0 = time.time()
    lms = lms_shift(IN, OUT, ax, set(T_CHECK))
    res["route1_lms_seconds"] = time.time() - t0
    for T in T_CHECK:
        r1 = route1(IN, OUT, ax, T, lms[T])
        for k, mods in r1.items():
            per = [p for p, _ in mods]
            st = pairs_stats(per, fwhm)
            st["max_abs_T_minus_P"] = {str(l): e for l, (_, e) in zip(LAMBDAS, mods)}
            res["route1"].setdefault(k, {})[str(T)] = st
            print(f"{name} route {k} T={T}: pure {st['frac_pure_0.99']:.3f} cov {st['cov_conj_min']:.3f} "
                  f"|T-P| {max(st['max_abs_T_minus_P'].values()):.2e}", flush=True)
            if T == Tmax:
                res["downstream"][k] = downstream(per, codes, gsteps, fwhm, noise, W, f"{name} {k}",
                                                  {"l1": k, "cond": name}, enc_path(k))
    # route 2
    for rule in RULES:
        t0 = time.time()
        snaps = lock_in(IN, OUT, ax, rule, seed=seed + 7, checks=set(T_CHECK))
        res["route2"][rule] = {"seconds": time.time() - t0}
        by_eta = {}
        for gi, eta in enumerate(ETAS):
            for T in T_CHECK:
                st = group_stats(snaps[T][0], gi, fwhm)
                res["route2"][rule].setdefault(f"eta{eta:g}", {})[str(T)] = st
                print(f"{name} route {rule} eta {eta:g} T={T}: pure {st['frac_pure_0.99']:.3f} "
                      f"median {st['purity_median']:.4f} cov {st[f'cov_conj_q{QMAX}_min']:.3f}", flush=True)
            by_eta[eta] = res["route2"][rule][f"eta{eta:g}"][str(Tmax)]
        eta = best_group(by_eta)
        gi = ETAS.index(eta)
        res["route2"][rule]["best_eta"] = eta
        Wf = snaps[Tmax][0]
        for variant, dd in (("all", False), ("dedup", True)):
            k = f"{rule}_{variant}"
            res["downstream"][k] = downstream(route2_pairs(Wf, gi, dd), codes, gsteps, fwhm, noise, W,
                                              f"{name} {rule} eta {eta:g} {variant}",
                                              {"l1": k, "eta": eta, "cond": name},
                                              enc_path(k) if variant == "all" else None)
        del snaps
    _dump(os.path.join(out, f"{name}.json"), res)
    print(f"DONE {name}", flush=True)


# --- summary --------------------------------------------------------------------

def _f(m):
    f = m.get("far", m["unvisited"])
    return f"{f['C1']:.3f} / {f['r_half']:.1f} / {f['far_sd']:.3f} / {f['alias_max']:.3f}" \
        if f["r_half"] is not None else f"{f['C1']:.3f} / - / {f['far_sd']:.3f} / {f['alias_max']:.3f}"


def summary(out):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "side*.json")))]
    order = {str(s): i for i, s in enumerate(SIDES)}
    rows.sort(key=lambda r: (r["noise"], -r["fwhm"], order[str(r["side"])]))
    L = ["# Sec 21: L1 learned locally and online", "",
         "Purity share = share of units (route 2) or eigenvectors (route 1) above 0.99. "
         "Coverage = smallest share over the three modules of patterns held by a unit above 0.99, "
         "conjugates included. Network columns (far cells): C(1) / half-width / far sd / alias max. "
         "Exact L1: 0.998 / 19.3 / 0.032 / 0.041. Sec 20 offline movement L1 (100k scattered steps): "
         "bump 0.25 alias 0.120, bump 0.10 alias 0.064.", ""]
    L += ["## Route 1 (shift estimate, then eigenvectors), and route 2 at its best eta", "",
          "| condition | cells | learner | T | pure share | coverage | network |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        nm = cond_name(r["side"], r["fwhm"], r["noise"])
        Tm = str(r["steps"])
        for k in ("1a", "1b", "1c"):
            for T, st in r["route1"][k].items():
                net = _f(r["downstream"][k]) if T == Tm else ""
                L.append(f"| {nm} | {r['cells_visited']} | {k} | {T} | {st['frac_pure_0.99']:.3f} | "
                         f"{st['cov_conj_min']:.3f} | {net} |")
        for rule in RULES:
            eta = r["route2"][rule]["best_eta"]
            for T, st in r["route2"][rule][f"eta{eta:g}"].items():
                net = ""
                if T == Tm:
                    net = (f"all: {_f(r['downstream'][rule + '_all'])}; "
                           f"dedup: {_f(r['downstream'][rule + '_dedup'])}")
                L.append(f"| {nm} | {r['cells_visited']} | {rule} eta {eta:g} | {T} | "
                         f"{st['frac_pure_0.99']:.3f} | {st[f'cov_conj_q{QMAX}_min']:.3f} | {net} |")
    L += ["", "## Route 2 by eta at the last step count (pure share / coverage with 1, 2, 4, 6 l² units / "
          "weak-half coverage / count CV)", "",
          "| condition | rule | " + " | ".join(f"eta {e:g}" for e in ETAS) + " |",
          "|---|---|" + "---|" * len(ETAS)]
    for r in rows:
        nm = cond_name(r["side"], r["fwhm"], r["noise"])
        for rule in RULES:
            cells = []
            for e in ETAS:
                st = r["route2"][rule][f"eta{e:g}"][str(r["steps"])]
                cv = "/".join(f"{st[f'cov_conj_q{q}_min']:.2f}" for q in QS)
                wk = min(st["modules"][str(l)]["cov_conj_weak_half"] for l in LAMBDAS)
                cvv = max(st["modules"][str(l)]["count_cv"] for l in LAMBDAS)
                cells.append(f"{st['frac_pure_0.99']:.2f} / {cv} / {wk:.2f} / {cvv:.1f}")
            L.append(f"| {nm} | {rule} | " + " | ".join(cells) + " |")
    L += ["", "## Downstream diagnostics (share of gain² on n = 0 units, kept units, n_eff)", "",
          "| condition | learner | share n=0 | kept | strong | n_eff |", "|---|---|---|---|---|---|"]
    for r in rows:
        nm = cond_name(r["side"], r["fwhm"], r["noise"])
        for k, d in r["downstream"].items():
            L.append(f"| {nm} | {k} | {d['share_w2_n0']:.3f} | {d['kept']} | {d['strong']} | {d['n_eff']:.0f} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.mode == "run":
        run(a.task, a.out)
    else:
        summary(a.out)


if __name__ == "__main__":
    main()
