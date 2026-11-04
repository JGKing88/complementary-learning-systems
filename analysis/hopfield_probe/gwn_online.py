"""Sec 22: streaming, stateful pieces for learning the grid wave network online.

Everything here runs step by step over walks that stream in sessions, so a walk
can be millions of steps long without being held in memory, and L1 and L3 can
learn at the same time.

* ``env_set`` / ``sessions`` / ``session_steps``: environments (square boxes),
  sessions of ``BLOCK`` steps cycling through them (each session a reflecting
  walk from the box centre; no step crosses between boxes), and each session's
  codes. Steps are in +e orientation (a -e step read backwards), padded to
  (n, 3, LP) as in ``gwn_l1_online``; with ``rest=True`` a second, independent
  noise draw of each step's start code comes too (the animal at rest).
* ``LockIn``: the lock-in rules of Sec 21 (2a-2e) as a stateful learner, step
  for step the same as ``gwn_l1_online.lock_in`` (pinned by a test), plus each
  unit's settledness (a running mean of cos of its phase-lock error).
* ``ShiftLMS``: the shift rule 1c at several rates, optionally
  bias-compensated, the same as ``gwn_l1_noisefix.lms`` (pinned by a test).
* ``L2Stream``: one lock-in eta group's L1 units -> random triples -> L2 waves,
  online. Each L1 unit is scaled by a running estimate of its own output power
  (a local homeostatic gain). Every step it feeds the same per-step change
  d = |dZ|^2 to several depression-rule variants of L3:

    always  from step 0, w <- w max(0, 1 - kappa_a d), kappa_a = r^2 / (2 T_total)
    late    kappa = 0.005 over the last 25,600 steps only
    self    a self-timed critical period per L2 unit: silent (w = 0) until its
            three L1 inputs have settled (settledness > thr), then w = 1 and
            depression for 25,600 steps, then frozen
    seq     after learning, L1 frozen, a fresh walk of 25,600 steps (Sec 21.1)

  Under noise each variant also recovers at rest by the change it sees there:
  log w += log(1 - kappa d_move) - log(1 - kappa d_rest), the depression form
  of the trace rule's at-rest correction.
* ``depression_l1``: the same depression (with rest recovery) for a fixed L1
  object, e.g. the shift rule's eigenvector pairs.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import gwn_l1_online as G
from .gwn_check import DEV, LAMBDAS, R, l1_out
from .gwn_encoder import L1, random_triples, unit_waves
from .gwn_l1_learn import OFFS
from .gwn_local_gain import KAPPA, walk_in_env
from .rp_encoder import grid_codes_torch

NPOS = 1716
BLOCK = 10_000
T3 = int(round(R * R / (2 * KAPPA)))           # 25,600 steps: sqrt(2 kappa T3) = r = 16
SETTLE_TAU = 2_000.0                            # steps, the settledness running mean
POWER_TAU = 10_000.0                            # steps, the L1 output-power running mean


# --- environments and sessions ----------------------------------------------------

def env_set(n, sides, seed):
    """``n`` boxes ``(x0, y0, side)``; one box sits at ``G.BOX`` (as in Sec 21)."""
    sides = list(sides) if np.ndim(sides) else [int(sides)] * n
    if n == 1:
        return [(G.BOX[0], G.BOX[1], sides[0])]
    rng = np.random.default_rng(seed)
    return [(int(rng.integers(0, NPOS - s)), int(rng.integers(0, NPOS - s)), int(s)) for s in sides]


def sessions(envs, T, seed, block=BLOCK):
    """Yield position arrays (n + 1, 2), sessions cycling through ``envs``, T steps in all."""
    t, k = 0, 0
    while t < T:
        x0, y0, s = envs[k % len(envs)]
        n = min(block, T - t)
        yield walk_in_env(x0, y0, s, n, seed * 1000 + k)
        t, k = t + n, k + 1


def _pad(X):
    Y = torch.zeros(len(X), len(LAMBDAS), G.LP, device=DEV)
    for m, l in enumerate(LAMBDAS):
        Y[:, m, :l * l] = X[:, OFFS[m]:OFFS[m + 1]]
    return Y


def session_steps(pos, fwhm, noise, gen, rest=False):
    """IN, OUT (n, 3, LP), ax (n,), IN2 (rest draw of IN, or None) for one session."""
    clean = grid_codes_torch(torch.as_tensor(pos[:, 0], device=DEV), torch.as_tensor(pos[:, 1], device=DEV),
                             LAMBDAS, fwhm)
    g = clean + noise * torch.randn(clean.shape, generator=gen).to(DEV) if noise > 0 else clean
    d = pos[1:] - pos[:-1]
    fwd = torch.tensor(d.sum(1) > 0, device=DEV)[:, None]
    IN, OUT = torch.where(fwd, g[:-1], g[1:]), torch.where(fwd, g[1:], g[:-1])
    IN2 = None
    if rest:
        c0 = torch.where(fwd, clean[:-1], clean[1:])
        IN2 = _pad(c0 + noise * torch.randn(c0.shape, generator=gen).to(DEV) if noise > 0 else c0)
    return _pad(IN), _pad(OUT), (d[:, 1] != 0).astype(np.int64), IN2


# --- L1 learners ----------------------------------------------------------------------

class LockIn:
    """Rules 2a-2e of ``gwn_l1_online.lock_in``, one step at a time."""

    def __init__(self, etas, rule="2c", phi_ratio=1.0, seed=0, kper=None, track=False):
        self.etas, self.rule, self.phi_ratio = tuple(etas), rule, float(phi_ratio)
        self.kper = int(kper or G.KPER)
        M, Lp = len(LAMBDAS), G.LP
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.eta = torch.tensor(np.repeat(self.etas, self.kper), dtype=torch.float32, device=DEV)
        K = len(self.eta)
        W, phi, U0 = G._init_units(M, Lp, K, gen)
        self.W, self.phi = W.to(DEV), phi.to(DEV)
        self.W /= torch.linalg.vector_norm(self.W, dim=1, keepdim=True)
        self.rot = torch.exp(-1j * self.phi)
        self.hebb, self.pll = rule in ("2a", "2d"), rule in ("2c", "2d")
        self.wpll = rule == "2e"
        self.U = None
        if not self.hebb:
            self.U = (torch.zeros(2, M, Lp, K, dtype=torch.complex64) if U0 is None else U0).to(DEV)
        self.eta_u = (4 * self.eta).clamp(max=1.0)
        self.track = track
        self.settle = torch.zeros(M, K, device=DEV) if track else None
        self.t = 0

    @torch.no_grad()
    def step(self, g0raw, g1, a, inv):
        """One step; returns the units' outputs (z0, z1) before the update."""
        W, eta, rot = self.W, self.eta, self.rot
        g0 = g0raw * inv[:, None]
        z1 = G._proj(g1, W)
        z0 = G._proj(g0raw, W) if (self.pll or self.track) else None
        if self.hebb:
            W += (eta * rot[a] * z1)[:, None, :] * g0[:, :, None]
        else:
            U = self.U
            pred = G._proj(g0raw, U[a])
            U[a] += (self.eta_u * (z1 - pred))[:, None, :] * g0[:, :, None]
            if self.wpll:
                self.phi[a] += (self.phi_ratio * eta) * torch.angle((W.conj() * U[a]).sum(1) * rot[a])
                rot[a] = torch.exp(-1j * self.phi[a])
            W.mul_(1 - eta).add_((eta * rot[a])[:, None, :] * U[a])
        if self.track:
            err = torch.angle(z1 * z0.conj() * rot[a])
            self.settle += (torch.cos(err) - self.settle) / SETTLE_TAU
        if self.pll:
            self.phi[a] += (self.phi_ratio * eta) * torch.angle(z1 * z0.conj() * rot[a])
            rot[a] = torch.exp(-1j * self.phi[a])
        W /= torch.linalg.vector_norm(W, dim=1, keepdim=True).clamp_min(1e-30)
        self.t += 1
        return z0, z1

    def run(self, IN, OUT, ax, on_step=None):
        inv = 1.0 / (IN ** 2).sum(-1)
        for t in range(len(IN)):
            z0, z1 = self.step(IN[t], OUT[t], int(ax[t]), inv[t])
            if on_step is not None:
                on_step(t, z0, z1)

    def snapshot(self):
        return self.W.cpu().numpy().copy(), self.phi.cpu().numpy().copy()


class ShiftLMS:
    """Shift rule 1c at several rates, plain and/or bias-compensated (``gwn_l1_noisefix.lms``)."""

    def __init__(self, rates, sig2=0.0, corrected=(False, True)):
        M, Lp = len(LAMBDAS), G.LP
        self.variants = [(r, c) for r in rates for c in corrected]
        V = len(self.variants)
        self.M, self.V = M, V
        self.rate = torch.tensor([r for r, _ in self.variants], device=DEV).repeat_interleave(M)[:, None, None]
        self.comp = torch.tensor([sig2 if c else 0.0 for _, c in self.variants],
                                 device=DEV).repeat_interleave(M)
        self.Ts = torch.zeros(2, V * M, Lp, Lp, device=DEV)

    @torch.no_grad()
    def step(self, g0raw, g1, a, inv):
        V = self.V
        g0 = g0raw.repeat(V, 1)
        iv = inv.repeat(V)
        err = g1.repeat(V, 1) - torch.bmm(self.Ts[a], g0[:, :, None])[:, :, 0]
        self.Ts[a] += self.rate * ((err * iv[:, None])[:, :, None] * g0[:, None, :]
                                   + (self.comp * iv)[:, None, None] * self.Ts[a])

    def run(self, IN, OUT, ax):
        inv = 1.0 / (IN ** 2).sum(-1)
        for t in range(len(IN)):
            self.step(IN[t], OUT[t], int(ax[t]), inv[t])

    def snapshot(self):
        X = self.Ts.reshape(2, self.V, self.M, G.LP, G.LP).cpu().numpy()
        return {v: X[:, i].copy() for i, v in enumerate(self.variants)}


# --- L2 and L3 online -------------------------------------------------------------------

def _dep(d, kappa):
    return torch.log((1 - kappa * d).clamp_min(1e-300))


def _rest(d_rest, kappa):
    return -torch.log((1 - kappa * d_rest).clamp_min(0.5))


class L2Stream:
    """L2 waves of one lock-in eta group, and depression-rule L3 variants, online."""

    def __init__(self, lock, gi, n_units, seed, T_total, variants, noise=0.0):
        self.lock, self.gi, self.noise = lock, gi, noise
        self.units = [gi * lock.kper + np.arange(G.QMAX * l * l) for l in LAMBDAS]
        self.units_t = [torch.as_tensor(u, device=DEV) for u in self.units]
        mod = np.concatenate([np.full(2 * len(u), m) for m, u in enumerate(self.units)])
        skel = L1(torch.zeros(1, len(mod)), torch.zeros(1, len(mod)), mod, None, LAMBDAS)
        self.tr = random_triples(skel, n_units, seed).to(DEV)
        self.T_total = T_total
        self.power = None
        U = len(self.tr)
        self.spec = variants
        self.logw = {k: torch.zeros(U, dtype=torch.float64, device=DEV) for k in variants}
        self.born = {k: torch.zeros(U, dtype=torch.bool, device=DEV) for k, v in variants.items()
                     if v[0] == "self"}
        self.age = {k: torch.zeros(U, dtype=torch.int32, device=DEV) for k in self.born}
        # pair p -> its L1 unit's flat (module, unit) index, for the settled gate
        self.pair_unit = torch.as_tensor(np.concatenate(
            [np.repeat(m * lock.W.shape[2] + u, 2) for m, u in enumerate(self.units)]), device=DEV)

    def _pairs(self, z):
        """(P,) complex pair outputs: per module, unit k -> (z_k, conj z_k) / sqrt(power)."""
        out = []
        for m, u in enumerate(self.units_t):
            zm = z[m, u] / self.power[m].sqrt()
            out.append(torch.stack([zm, zm.conj()], 1).reshape(-1))
        return torch.cat(out)

    def _waves(self, P):
        return P[self.tr[:, 0]] * P[self.tr[:, 1]] * P[self.tr[:, 2]]

    @torch.no_grad()
    def step(self, t, z0, z1, z2=None, only=None):
        pw0 = [z0[m, u].abs() ** 2 for m, u in enumerate(self.units_t)]   # modules differ in size
        if self.power is None:
            self.power = [p.clamp_min(1e-30) for p in pw0]
        elif only is None:
            self.power = [p + (q - p) / POWER_TAU for p, q in zip(self.power, pw0)]
        Z0 = self._waves(self._pairs(z0))
        d = ((self._waves(self._pairs(z1)) - Z0).abs() ** 2).double()
        dr = ((self._waves(self._pairs(z2)) - Z0).abs() ** 2).double() if z2 is not None else None
        gate = None
        for k, v in self.spec.items():
            if only is not None and k not in only:
                continue
            kind, kappa = v[0], v[1]
            if kind == "always" or (kind == "late" and t >= v[2]) or (kind == "seq" and only is not None):
                inc = _dep(d, kappa) + (_rest(dr, kappa) if dr is not None else 0)
                self.logw[k] += inc
            elif kind == "self":
                if gate is None:
                    s = self.lock.settle.reshape(-1)[self.pair_unit]
                    gate_by = {}
                thr = v[2]
                if thr not in gate_by:
                    ok = s > thr
                    gate_by[thr] = ok[self.tr[:, 0]] & ok[self.tr[:, 1]] & ok[self.tr[:, 2]]
                gate = gate_by[thr]
                self.born[k] |= gate
                act = self.born[k] & (self.age[k] < T3)
                inc = _dep(d, kappa) + (_rest(dr, kappa) if dr is not None else 0)
                self.logw[k] += torch.where(act, inc, torch.zeros_like(inc))
                self.age[k] += act.int()

    def gains(self, k):
        lw = self.logw[k]
        w = torch.where(lw == 0, torch.zeros_like(lw), torch.exp(lw))
        if k in self.born:
            w = torch.where(self.born[k], torch.exp(lw), torch.zeros_like(lw))
        return w

    def l1(self, W=None):
        """L1 of this group: pairs (Re w, Im w), (Re w, -Im w), scaled by 1 / sqrt(power)."""
        W = self.lock.W if W is None else W
        P = sum(2 * len(u) for u in self.units)
        Wc = torch.zeros(int(OFFS[-1]), P, dtype=torch.float64, device=DEV)
        Ws = torch.zeros_like(Wc)
        mod, off = [], 0
        for m, (u, l) in enumerate(zip(self.units_t, LAMBDAS)):
            w = (W[m, :l * l][:, u] / self.power[m].sqrt()[None, :]).to(torch.complex128)
            n = w.shape[1]
            rows = slice(int(OFFS[m]), int(OFFS[m + 1]))
            Wc[rows, off:off + 2 * n:2] = w.real
            Wc[rows, off + 1:off + 2 * n:2] = w.real
            Ws[rows, off:off + 2 * n:2] = w.imag
            Ws[rows, off + 1:off + 2 * n:2] = -w.imag
            mod += [m] * (2 * n)
            off += 2 * n
        return L1(Wc, Ws, mod, None, LAMBDAS).to(DEV)

    def summary(self, k):
        w = self.gains(k)
        r = {"strong": int((w > math.exp(-1)).sum())}
        if k in self.born:
            r["born"] = float(self.born[k].float().mean())
            r["completed"] = float((self.age[k] >= T3).float().mean())
        return r


@torch.no_grad()
def depression_l1(l1, tr, pos, fwhm, noise, seed, kappa=KAPPA, chunk_u=16384, chunk_t=4096):
    """Depression L3 for a fixed L1 object over walk ``pos`` (one noisy code per
    moment), with rest recovery under noise (a second draw at each step's start)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def codes(p):
        c = grid_codes_torch(torch.as_tensor(p[:, 0], device=DEV), torch.as_tensor(p[:, 1], device=DEV),
                             LAMBDAS, fwhm).double()
        return c + noise * torch.randn(c.shape, generator=gen, dtype=torch.float64).to(DEV) if noise > 0 else c

    g = codes(pos)
    c, s = (g @ l1.Wc).float(), (g @ l1.Ws).float()
    if noise > 0:
        g2 = codes(pos[:-1])
        c2, s2 = (g2 @ l1.Wc).float(), (g2 @ l1.Ws).float()
    T = len(pos) - 1
    logw = torch.zeros(len(tr), dtype=torch.float64, device=DEV)
    for a in range(0, len(tr), chunk_u):
        trc = tr[a:a + chunk_u].to(DEV)
        acc = torch.zeros(len(trc), dtype=torch.float64, device=DEV)
        for t0 in range(0, T, chunk_t):
            t1 = min(T, t0 + chunk_t)
            C, S = unit_waves(c[t0:t1 + 1], s[t0:t1 + 1], trc)
            d = ((C[1:] - C[:-1]) ** 2 + (S[1:] - S[:-1]) ** 2).double()
            acc += _dep(d, kappa).sum(0)
            if noise > 0:
                C2, S2 = unit_waves(c2[t0:t1], s2[t0:t1], trc)
                acc += _rest(((C2 - C[:-1]) ** 2 + (S2 - S[:-1]) ** 2).double(), kappa).sum(0)
        logw[a:a + chunk_u] = acc
    return torch.where(logw == 0, torch.zeros_like(logw), torch.exp(logw))
