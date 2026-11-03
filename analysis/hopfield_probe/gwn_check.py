"""Steps 0, 2 and 3 of Sec 20: the grid wave network built and measured.

Walk steps are pairs (p, p + e), e a unit step, both inside one of the
scattered environments (``rp_readout_check.make_envs``: squares of side
10-100 covering ``coverage`` of the scaffold), environments chosen in
proportion to area. A unit's slowness is

    v_k = mean (dC)^2 + (dS)^2  /  mean (C^2 + S^2)

over those steps (the denominator is 1 for a pure wave and keeps impure units
comparable); its gain is a rule g(v_k) at r = 16; the code keeps the fewest
units holding 99.5% of sum w^2 (``gwn_encoder.prune``). Kernel metrics are
``rp_readout_check.kernel_metrics`` on visited, unvisited and far cells.

    python -m analysis.hopfield_probe.gwn_check step0 --out DIR
    python -m analysis.hopfield_probe.gwn_check step2 --out DIR
    python -m analysis.hopfield_probe.gwn_check step3 --out DIR
    python -m analysis.hopfield_probe.gwn_check extra --out DIR   # step3b + step2b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .gwn_encoder import (GAIN_RULES, L1, GWNEncoder, all_triples, exact_l1, prune,
                          random_triples, save_gwn, triple_frequencies, unit_waves)
from .ideal_encoder import IdealEncoder
from .rp_encoder import grid_codes_torch
from .rp_readout_check import (Ideal, grid_aliases, kernel_metrics, make_envs,
                               position_sets)

NPOS = 1716
LAMBDAS = (11, 12, 13)
R = 16.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# GWN_SMOKE=1 shrinks every count so the pipeline runs on a login node
SMOKE = bool(os.environ.get("GWN_SMOKE"))
NU = 5_000 if SMOKE else 400_000


# --- walk steps -----------------------------------------------------------------

def sample_steps(rects, n: int, seed: int):
    """``(p0, p1)`` int arrays (n, 2): unit steps inside one environment."""
    rng = np.random.default_rng(seed)
    s = np.array([r[2] for r in rects], float)
    env = rng.choice(len(rects), n, p=s**2 / (s**2).sum())
    x0 = np.array([rects[e][0] for e in env]); y0 = np.array([rects[e][1] for e in env])
    sz = np.array([rects[e][2] for e in env])
    u = (rng.random(n) * sz).astype(int); v = (rng.random(n) * sz).astype(int)
    d = rng.integers(0, 4, n)
    dx = np.select([d == 0, d == 1], [1, -1], 0); dy = np.select([d == 2, d == 3], [1, -1], 0)
    # reflect at the wall so the step stays inside the environment
    dx = np.where((u + dx < 0) | (u + dx >= sz), -dx, dx)
    dy = np.where((v + dy < 0) | (v + dy >= sz), -dy, dy)
    p0 = np.stack([x0 + u, y0 + v], 1)
    return p0, p0 + np.stack([dx, dy], 1)


def _t(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.long, device=DEV)


def l1_out(l1: L1, pos, fwhm, codes_fn=None):
    """L1 (c, s) at positions, float64."""
    x, y = _t(pos[:, 0]), _t(pos[:, 1])
    g = (codes_fn(x, y) if codes_fn else grid_codes_torch(x, y, LAMBDAS, fwhm)).double()
    return g @ l1.Wc, g @ l1.Ws


def _noisy(sigma, seed):
    """Grid codes plus independent Gaussian cell noise of sd ``sigma``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    def fn(x, y):
        c = grid_codes_torch(x, y, LAMBDAS, _noisy.fwhm).double()
        return c + sigma * torch.randn(c.shape, generator=g, dtype=torch.float64).to(c.device)
    return fn


def slowness(l1, triples, steps, fwhm, mult="exact", param=None, coef=None, chunk=16384,
             noise=0.0, correct=False):
    """Normalised slowness v (U,) and power mean(C^2 + S^2) (U,) of each unit.

    ``noise`` > 0 adds independent cell noise to the grid code at every step
    position, as an animal with noisy grid cells would see while learning.
    ``correct`` subtracts each unit's jitter at rest (two noise draws at the
    same position) from its change per step, and normalises by signal power
    (total minus noise); the returned power stays total, so noisy units are
    still down-weighted by 1/sqrt(power).
    """
    fn0 = fn1 = fn2 = None
    if noise > 0:
        _noisy.fwhm = fwhm
        fn0, fn1, fn2 = _noisy(noise, 101), _noisy(noise, 202), _noisy(noise, 303)
    c0, s0 = (t.float() for t in l1_out(l1, steps[0], fwhm, fn0))
    c1, s1 = (t.float() for t in l1_out(l1, steps[1], fwhm, fn1))
    if correct and noise > 0:
        c2, s2 = (t.float() for t in l1_out(l1, steps[0], fwhm, fn2))
    U = len(triples)
    v = torch.empty(U, dtype=torch.float64, device=DEV)
    pw = torch.empty(U, dtype=torch.float64, device=DEV)
    for a in range(0, U, chunk):
        tr = triples[a:a + chunk].to(DEV)
        cf = None if coef is None else coef[a:a + chunk].to(DEV).float()
        C0, S0 = unit_waves(c0, s0, tr, mult, param, cf)
        C1, S1 = unit_waves(c1, s1, tr, mult, param, cf)
        p = ((C0 ** 2 + S0 ** 2).mean(0) + (C1 ** 2 + S1 ** 2).mean(0)) / 2
        step = ((C1 - C0) ** 2 + (S1 - S0) ** 2).mean(0)
        var = C0.var(0) + S0.var(0)
        ps = p
        if correct and noise > 0:
            C2, S2 = unit_waves(c2, s2, tr, mult, param, cf)
            rest = ((C2 - C0) ** 2 + (S2 - S0) ** 2).mean(0)    # = 2 x noise power
            step = (step - rest).clamp_min(1e-12)
            ps = (p - rest / 2).clamp_min(1e-12)
            var = var - rest / 2
        vk = step / ps.clamp_min(1e-12)
        # a unit that barely varies across positions (the n = 0 wave, or a
        # learned unit that collapsed to a constant) carries no information:
        # infinite slowness, so its gain is 0
        vk = torch.where(var / ps.clamp_min(1e-12) < 0.01, torch.full_like(vk, float("inf")), vk)
        v[a:a + chunk] = vk.double()
        pw[a:a + chunk] = p.double()
    return v, pw


def analytic_slowness(n: torch.Tensor) -> torch.Tensor:
    w = 2 * math.pi * n.double() / NPOS
    return 2 - torch.cos(w[:, 0]) - torch.cos(w[:, 1])


def build(l1, triples, v, pw, fwhm, rule="gauss", r=R, mult="exact", param=None, coef=None,
          meta=None):
    """Gains, pruning and the encoder (on DEV)."""
    w = GAIN_RULES[rule](v, r) / pw.clamp_min(1e-12).sqrt()
    keep = prune(w)
    enc = GWNEncoder(l1, triples[keep.cpu()], w[keep].cpu(), fwhm_ratio=fwhm, mult=mult,
                     param=param, coef=None if coef is None else coef[keep.cpu()],
                     meta=meta).to(DEV).eval()
    w2 = (w.double() ** 2)
    info = {"units": int(len(triples)), "kept": int(len(keep)),
            "kept_mass": float(w2[keep].sum() / w2.sum()),
            "strong": int((GAIN_RULES[rule](v, r) > math.exp(-1)).sum()),
            "n_eff": float(w2[keep].sum() ** 2 / (w2[keep] ** 2).sum())}
    return enc, info


class World:
    def __init__(self, coverage=0.05, seed=0, fwhm=0.25, n_steps=512 if SMOKE else 8192):
        self.mask, self.rects = make_envs(coverage, seed)
        vis, unv, far, _ = position_sets(self.mask, seed)
        self.sets = {"visited": vis[:20000], "unvisited": unv}
        if len(far) >= 1000:
            self.sets["far"] = far
        self.steps = sample_steps(self.rects, n_steps, seed + 5)
        self.aliases, _ = grid_aliases(fwhm)
        self.fwhm = fwhm

    def metrics(self, enc, fwhm=None):
        f = self.fwhm if fwhm is None else fwhm
        with torch.no_grad():
            code = lambda x, y: enc(grid_codes_torch(x, y, LAMBDAS, f))
            return {k: kernel_metrics(code, p, self.aliases, DEV) for k, p in self.sets.items()}


def _dump(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def _line(tag, info, m):
    f = m.get("far", m["unvisited"])
    print(f"{tag}: units {info['units']} kept {info['kept']} strong {info['strong']} "
          f"n_eff {info['n_eff']:.0f} | far C1 {f['C1']:.3f} r_half {f['r_half']} "
          f"far_sd {f['far_sd']:.3f} alias {f['alias_max']:.3f}", flush=True)


# --- step 0 ---------------------------------------------------------------------------

def step0(out, encdir):
    W = World()
    res = {"ideal": W.metrics(IdealEncoder(r=R).to(DEV).eval())}
    l1 = exact_l1(W.fwhm).to(DEV)
    sizes = ([("2k", 2_000), ("400k", NU)] if SMOKE else
             [("25k", 25_000), ("100k", 100_000), ("400k", 400_000), ("all", None)])
    for name, nu in sizes:
        t0 = time.time()
        tr = all_triples(l1) if nu is None else random_triples(l1, nu, seed=0)
        v, pw = slowness(l1, tr, W.steps, W.fwhm)
        va = analytic_slowness(triple_frequencies(l1, tr)).to(DEV)
        enc, info = build(l1, tr, v, pw, W.fwhm, meta={"coverage": 0.05, "units": name})
        fin = torch.isfinite(v)                      # the n = 0 unit is excluded
        info["slowness_err"] = {"max_abs": float((v - va)[fin].abs().max()),
                                "max_rel_slow": float(((v - va).abs() / va.clamp_min(1e-9))[fin & (va < 0.02)].max()),
                                "excluded_constant": int((~fin).sum())}
        m = W.metrics(enc)
        res[f"units_{name}"] = {**info, "seconds": time.time() - t0, **m}
        _line(f"step0 units {name}", info, m)
        if name in ("400k", "all"):
            save_gwn(enc.cpu(), os.path.join(encdir, f"exact_{name}_gauss.pt"))
        if name == "400k":
            tr400, v400, pw400 = tr, v, pw
    for rule in ("cut", "rational", "linear"):
        enc, info = build(l1, tr400, v400, pw400, W.fwhm, rule=rule,
                          meta={"coverage": 0.05, "units": "400k", "rule": rule})
        m = W.metrics(enc)
        res[f"rule_{rule}"] = {**info, **m}
        _line(f"step0 rule {rule}", info, m)
        save_gwn(enc.cpu(), os.path.join(encdir, f"exact_400k_{rule}.pt"))
    for cov in (0.01, 0.2):
        Wc = World(coverage=cov)
        v, pw = slowness(l1, tr400, Wc.steps, W.fwhm)
        enc, info = build(l1, tr400, v, pw, W.fwhm)
        m = W.metrics(enc)
        fin = torch.isfinite(v) & torch.isfinite(v400)
        res[f"coverage_{cov:g}"] = {**info, "slowness_max_abs_diff_vs_5pct": float((v - v400)[fin].abs().max()), **m}
        _line(f"step0 coverage {cov:g}", info, m)
    _dump(os.path.join(out, "step0.json"), res)


# --- step 2 ---------------------------------------------------------------------------

def shifted_codes(x, y, shift, fwhm):
    """Grid codes with module m's bumps displaced by ``shift[:, m]`` (N, 3, 2) cells."""
    blocks = []
    for m, l in enumerate(LAMBDAS):
        idx = torch.arange(l, device=x.device, dtype=torch.float64)
        sig = l * fwhm / (2 * math.sqrt(2 * math.log(2)))
        out = []
        for ax, coord in enumerate((x, y)):
            ph = coord.double()[:, None] + shift[:, m, ax][:, None]
            d = torch.remainder(ph - idx[None, :], l)
            d = torch.minimum(d, l - d)
            out.append(torch.exp(-0.5 * (d / sig) ** 2))
        blocks.append((out[0][:, :, None] * out[1][:, None, :]).reshape(-1, l * l))
    return torch.cat(blocks, 1)


def noise_test(enc, fwhm, kind, level, anchors, radius=32, seed=0):
    """Self-similarity and decoded position error under grid noise."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ax, ay = _t(anchors[:, 0]), _t(anchors[:, 1])
    if kind == "activity":
        clean = grid_codes_torch(ax, ay, LAMBDAS, fwhm).double()
        noisy = clean + level * torch.randn(clean.shape, generator=g, dtype=torch.float64).to(DEV)
    else:
        sh = level * torch.randn(len(anchors), 3, 2, generator=g, dtype=torch.float64).to(DEV)
        noisy = shifted_codes(ax, ay, sh, fwhm)
    zc = enc(grid_codes_torch(ax, ay, LAMBDAS, fwhm)).double()
    zn = enc(noisy.float()).double()
    selfsim = (zc * zn).sum(1)
    offs = torch.arange(-radius, radius + 1, device=DEV)
    ox, oy = torch.meshgrid(offs, offs, indexing="ij")
    ox, oy = ox.reshape(-1), oy.reshape(-1)
    errs = []
    for i in range(len(anchors)):
        cx = torch.remainder(ax[i] + ox, NPOS); cy = torch.remainder(ay[i] + oy, NPOS)
        zw = enc(grid_codes_torch(cx, cy, LAMBDAS, fwhm)).double()
        j = int(torch.argmax(zw @ zn[i]))
        errs.append(math.hypot(int(ox[j]), int(oy[j])))
    errs = np.array(errs)
    return {"selfsim": float(selfsim.mean()), "err_mean": float(errs.mean()),
            "err_median": float(np.median(errs)), "frac_gt5": float((errs > 5).mean())}


def step2(out, encdir):
    W = World()
    l1 = exact_l1(W.fwhm).to(DEV)
    tr = random_triples(l1, NU, seed=0)
    res = {}
    g = torch.Generator().manual_seed(7)
    for s in (0.01, 0.05, 0.1, 0.3):
        coef = 1 + s * torch.randn(len(tr), 8, generator=g, dtype=torch.float64)
        v, pw = slowness(l1, tr, W.steps, W.fwhm, coef=coef)
        enc, info = build(l1, tr, v, pw, W.fwhm, coef=coef, meta={"units": "400k", "coef_sd": s})
        m = W.metrics(enc)
        res[f"coef_{s:g}"] = {**info, **m}
        _line(f"step2 coef {s:g}", info, m)
        if s == 0.1:
            save_gwn(enc.cpu(), os.path.join(encdir, "exact_400k_coef0.1.pt"))
    for p in (1.5, 1.8, 2.2, 2.5):
        v, pw = slowness(l1, tr, W.steps, W.fwhm, mult="quarter", param=p)
        enc, info = build(l1, tr, v, pw, W.fwhm, mult="quarter", param=p,
                          meta={"units": "400k", "quarter_p": p})
        m = W.metrics(enc)
        res[f"quarter_{p:g}"] = {**info, **m}
        _line(f"step2 quarter p={p:g}", info, m)
    anchors = W.sets["unvisited"][:4 if SMOKE else 200]
    for fwhm in (0.25, 0.10):
        l1f = exact_l1(fwhm).to(DEV)
        v, pw = slowness(l1f, tr, W.steps, fwhm)
        enc, _ = build(l1f, tr, v, pw, fwhm)
        ideal = IdealEncoder(r=R).to(DEV).eval()
        for name, e in (("gwn", enc), ("ideal", ideal)):
            for kind, levels in (("activity", (0.001, 0.01, 0.05)), ("phase", (0.05, 0.1, 0.2, 0.4))):
                for lv in levels:
                    with torch.no_grad():
                        r = noise_test(e, fwhm, kind, lv, anchors, radius=3 if SMOKE else 32)
                    res[f"noise_{name}_fwhm{fwhm:g}_{kind}_{lv:g}"] = r
                    print(f"step2 noise {name} fwhm {fwhm:g} {kind} {lv:g}: selfsim {r['selfsim']:.3f} "
                          f"err mean {r['err_mean']:.2f} >5 cells {r['frac_gt5']:.2f}", flush=True)
    _dump(os.path.join(out, "step2.json"), res)


# --- step 3 ---------------------------------------------------------------------------

def step3(out, encdir):
    W = World()
    l1 = exact_l1(W.fwhm).to(DEV)
    a, b, c = (l1.pairs_of(m).cpu() for m in range(3))
    A, B = torch.meshgrid(a, b, indexing="ij")
    pairs = torch.stack([A.reshape(-1), B.reshape(-1)], 1)               # 17,424
    # stage A: pair waves exp(i(th11 + th12)), gain at r / 13
    c0, s0 = (t.float() for t in l1_out(l1, W.steps[0], W.fwhm))
    c1, s1 = (t.float() for t in l1_out(l1, W.steps[1], W.fwhm))
    def pair_wave(cc, ss):
        u = cc[:, pairs[:, 0]] * cc[:, pairs[:, 1]] - ss[:, pairs[:, 0]] * ss[:, pairs[:, 1]]
        v = ss[:, pairs[:, 0]] * cc[:, pairs[:, 1]] + cc[:, pairs[:, 0]] * ss[:, pairs[:, 1]]
        return u, v
    u0, v0 = pair_wave(c0, s0); u1, v1 = pair_wave(c1, s1)
    vA = (((u1 - u0) ** 2 + (v1 - v0) ** 2).mean(0) / ((u0 ** 2 + v0 ** 2).mean(0))).double()
    wA = GAIN_RULES["gauss"](vA, R / 13)
    res = {"pairs": int(len(pairs))}
    # w_A > 0.37 is the 2-sigma band (|m| <~ 34 of 66 per axis); 0.1 keeps |m| <~ 52
    for thr in (0.1, 0.37):
        kept = pairs[(wA > thr).cpu()][:50 if SMOKE else None]
        tr = torch.cat([kept.repeat_interleave(len(c), 0), c.repeat(len(kept))[:, None]], 1)
        v, pw = slowness(l1, tr, W.steps, W.fwhm)
        enc, info = build(l1, tr, v, pw, W.fwhm, meta={"staged": thr})
        m = W.metrics(enc)
        res[f"staged_thr{thr:g}"] = {"kept_pairs": int(len(kept)), "product_synapses": int(len(tr)),
                                     "neurons": int(len(pairs) + len(kept)), **info, **m}
        _line(f"step3 staged thr {thr:g} (kept pairs {len(kept)}, products {len(tr)})", info, m)
    _dump(os.path.join(out, "step3.json"), res)


def step2b(out, encdir):
    """Gains learned with noisy grid cells, then the activity-noise test."""
    tr = random_triples(exact_l1(0.25), NU, seed=0)        # pattern indexing is the same at any width
    res = {}
    for fwhm in (0.25, 0.10):
        W = World(fwhm=fwhm)
        anchors = W.sets["unvisited"][:4 if SMOKE else 200]
        l1f = exact_l1(fwhm).to(DEV)
        for sig in (0.001, 0.01, 0.05):
            for corr in (False, True):
                v, pw = slowness(l1f, tr, W.steps, fwhm, noise=sig, correct=corr)
                enc, info = build(l1f, tr, v, pw, fwhm,
                                  meta={"units": "400k", "learned_noise": sig, "rest_corrected": corr})
                m = W.metrics(enc)
                with torch.no_grad():
                    r = noise_test(enc, fwhm, "activity", sig, anchors, radius=3 if SMOKE else 32)
                tag = f"fwhm{fwhm:g}_noise{sig:g}_{'corrected' if corr else 'raw'}"
                res[tag] = {**info, **m, "noise_test": r}
                _line(f"step2b {tag}", info, m)
                print(f"step2b {tag}: selfsim {r['selfsim']:.3f} err mean {r['err_mean']:.2f} "
                      f">5 cells {r['frac_gt5']:.2f}", flush=True)
                if fwhm == 0.25 and sig == 0.01 and corr:
                    save_gwn(enc.cpu(), os.path.join(encdir, "exact_400k_noise0.01_fwhm0.25.pt"))
    _dump(os.path.join(out, "step2b.json"), res)


def step3b(out, encdir):
    """One neuron per (j11, j12) pair with 169 module-13 branches; branch-level
    slowness keeps the slowest. Per axis, pair m' and the best branch give the
    unique n in (-66, 66] with n = 13 m' (mod 132), so the 17,424 neurons cover
    every frequency with |n_x|, |n_y| <= 66 once."""
    W = World()
    l1 = exact_l1(W.fwhm).to(DEV)
    tr_all = all_triples(l1)                               # module-13 index varies fastest
    nC = len(l1.pairs_of(2))
    v, pw = slowness(l1, tr_all, W.steps, W.fwhm)
    vv = v.view(-1, nC)
    best = torch.argmin(vv, 1)
    idx = torch.arange(vv.shape[0], device=DEV) * nC + best
    tr = tr_all[idx.cpu()]
    n = triple_frequencies(l1, tr)
    enc, info = build(l1, tr, v[idx], pw[idx], W.fwhm, meta={"design": "branch", "neurons": int(len(tr))})
    m = W.metrics(enc)
    cover = {"neurons": int(len(tr)), "branch_products": int(len(tr_all)),
             "distinct_n": int(len({tuple(x) for x in n.tolist()})),
             "max_abs_n": int(n.abs().max())}
    _dump(os.path.join(out, "step3b.json"), {**cover, **info, **m})
    _line(f"step3b branch design ({cover})", info, m)
    save_gwn(enc.cpu(), os.path.join(encdir, "exact_branch17k_gauss.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["step0", "step2", "step3", "extra"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    encdir = os.path.join(a.out, "encoders")
    os.makedirs(encdir, exist_ok=True)
    if a.step == "extra":
        step3b(a.out, encdir)
        step2b(a.out, encdir)
    else:
        {"step0": step0, "step2": step2, "step3": step3}[a.step](a.out, encdir)


if __name__ == "__main__":
    main()
