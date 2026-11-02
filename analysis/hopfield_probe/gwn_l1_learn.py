"""Step 1 of Sec 20: can the grid wave network's L1 be learned?

Two learners, per module (cells of one module only), on cells the animal
visits in the scattered environments:

* ``pca``: eigenvectors of the module's covariance, paired in order as (c, s),
  plus a constant pair (the j = 0 pattern). This is where Hebbian PCA (Oja,
  Sanger) converges; within a tied eigenvalue group any rotation is a fixed
  point, so the exact eigendecomposition is the best a Hebbian rule can do.
* ``move``: one-step shift operators from walk steps, ``g(p + e) ~ T_e g(p)``
  (ridge least squares, e = +x, +y; a -x step is a +x step read backwards),
  then the left eigenvectors w of ``T_x + 0.618 T_y``. Each pattern j has its
  own eigenvalue exp(2 pi i jx/l) + 0.618 exp(2 pi i jy/l), so the tied groups
  separate. The pair is (Re w . g, Im w . g); one of each conjugate pair kept.

Purity of a pair = the largest share of the power of (wc + i ws) on the module
torus in one frequency (1 for an exact pattern). Every pair is scaled to mean
c^2 + s^2 = 1 on visited cells. Each learned L1 then drives the 400k network
(``gwn_check``: random triples, slowness from walk steps, Gaussian gains).

    python -m analysis.hopfield_probe.gwn_l1_learn --out DIR
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .gwn_check import DEV, LAMBDAS, NU, SMOKE, World, _dump, _line, _t, build, sample_steps, slowness
from .gwn_encoder import L1, bump_strength, random_triples, save_gwn
from .rp_encoder import grid_codes_torch

ALPHA = 0.618
OFFS = np.cumsum([0] + [l * l for l in LAMBDAS])


def codes_at(pos, fwhm):
    return grid_codes_torch(_t(pos[:, 0]), _t(pos[:, 1]), LAMBDAS, fwhm).double()


def purity(wc, ws, l):
    F = np.abs(np.fft.fft2((wc + 1j * ws).reshape(l, l))) ** 2
    return float(F.max() / F.sum())


def strength_of_best(wc, ws, l, fwhm):
    """A(j)/A(0) of the pattern the pair is closest to (how weak it is)."""
    F = np.abs(np.fft.fft2((wc + 1j * ws).reshape(l, l))) ** 2
    jx, jy = np.unravel_index(np.argmax(F), F.shape)
    b = bump_strength(l, fwhm, np.array([0, jx, jy]))
    return float(b[1] * b[2] / b[0] ** 2)


def _assemble(per_module, codes, fwhm):
    """Stack per-module (wc, ws) lists into an L1 scaled to mean c^2 + s^2 = 1."""
    ng = int(OFFS[-1])
    Wc, Ws, mod, info = [], [], [], []
    for m, pairs in enumerate(per_module):
        l = LAMBDAS[m]
        for wc, ws in pairs:
            fc = np.zeros(ng); fs = np.zeros(ng)
            fc[OFFS[m]:OFFS[m + 1]] = wc; fs[OFFS[m]:OFFS[m + 1]] = ws
            Wc.append(fc); Ws.append(fs); mod.append(m)
            info.append((m, purity(wc, ws, l), strength_of_best(wc, ws, l, fwhm)))
    Wc, Ws = torch.tensor(np.stack(Wc, 1)).to(DEV), torch.tensor(np.stack(Ws, 1)).to(DEV)
    p = ((codes @ Wc) ** 2 + (codes @ Ws) ** 2).mean(0).clamp_min(1e-300).sqrt()
    return L1(Wc / p, Ws / p, mod, None, LAMBDAS).to(DEV), info


def pca_l1(codes, fwhm):
    per = []
    X = codes.cpu().numpy()
    for m, l in enumerate(LAMBDAS):
        B = X[:, OFFS[m]:OFFS[m + 1]]
        ev, V = np.linalg.eigh(np.cov(B.T, bias=True))
        V = V[:, np.argsort(ev)[::-1]]
        pairs = [(np.ones(l * l), np.zeros(l * l))]                 # the constant pair
        for i in range(0, l * l - 1, 2):
            pairs.append((V[:, i], V[:, i + 1]))
        if (l * l) % 2:
            pairs.append((V[:, -1], np.zeros(l * l)))
        per.append(pairs)
    return per


def shift_ops(g0, g1, d, m, ridge):
    """T_x, T_y for module m from step codes (rows) and step directions d (n, 2)."""
    out = []
    l = LAMBDAS[m]
    a0, a1 = g0[:, OFFS[m]:OFFS[m + 1]], g1[:, OFFS[m]:OFFS[m + 1]]
    for ax in (0, 1):
        fw = d[:, ax] == 1
        bw = d[:, ax] == -1
        X0 = np.concatenate([a0[fw], a1[bw]]); X1 = np.concatenate([a1[fw], a0[bw]])
        G = X0.T @ X0
        G[np.diag_indices(l * l)] += ridge * np.trace(G) / (l * l)
        out.append(np.linalg.solve(G, X0.T @ X1).T)            # X1 ~ X0 T^T
    return out


def move_l1(g0, g1, d, ridge):
    per = []
    for m, l in enumerate(LAMBDAS):
        Tx, Ty = shift_ops(g0, g1, d, m, ridge)
        mu, Wl = np.linalg.eig((Tx + ALPHA * Ty).T)               # left eigenvectors
        pairs = []
        for k in np.argsort(-mu.real):
            if mu[k].imag < -1e-6:
                continue                                         # conjugate of a kept one
            w = Wl[:, k]
            w = w * np.exp(-1j * np.angle(w[np.argmax(np.abs(w))]))
            pairs.append((w.real, w.imag))
        per.append(pairs)
    return per


def online_l1(g0, g1, d, iters=4000, batch=1024, lr=0.02, seed=0):
    """Stretch: online SGD on a rotation-consistency objective, per module.

    K = l^2 complex units z_k = g . w_k, each with learned step angles phi_k,x
    and phi_k,y. Loss per minibatch of steps: |z(p+e) - exp(i phi_k,e) z(p)|^2
    (each unit's own error times its input change: local), plus unit power
    (mean |z|^2 = 1) and decorrelation across units (|C_kk'|^2, the one term
    that needs lateral connections).
    """
    per = []
    for m, l in enumerate(LAMBDAS):
        L = l * l
        a0 = torch.tensor(g0[:, OFFS[m]:OFFS[m + 1]], dtype=torch.float32, device=DEV)
        a1 = torch.tensor(g1[:, OFFS[m]:OFFS[m + 1]], dtype=torch.float32, device=DEV)
        dd = torch.tensor(d, device=DEV)
        sets = []
        for ax in (0, 1):
            fw, bw = dd[:, ax] == 1, dd[:, ax] == -1
            sets.append((torch.cat([a0[fw], a1[bw]]), torch.cat([a1[fw], a0[bw]])))
        gen = torch.Generator(device="cpu").manual_seed(seed + m)
        Wr = (torch.randn(L, L, generator=gen) / L ** 0.5).to(DEV).requires_grad_()
        Wi = (torch.randn(L, L, generator=gen) / L ** 0.5).to(DEV).requires_grad_()
        ph = ((torch.rand(2, L, generator=gen) * 2 - 1) * np.pi).to(DEV).requires_grad_()
        opt = torch.optim.Adam([Wr, Wi, ph], lr=lr)
        scale = float(a0.std())
        for it in range(iters):
            loss = 0.0
            zs = []
            for ax, (X0, X1) in enumerate(sets):
                idx = torch.randint(0, len(X0), (batch,), device=DEV)
                x0, x1 = X0[idx] / scale, X1[idx] / scale
                r0, i0, r1, i1 = x0 @ Wr, x0 @ Wi, x1 @ Wr, x1 @ Wi
                c, s = torch.cos(ph[ax]), torch.sin(ph[ax])
                er, ei = r1 - (c * r0 - s * i0), i1 - (s * r0 + c * i0)
                loss = loss + (er ** 2 + ei ** 2).mean()
                zs.append((r0, i0))
            zr = torch.cat([z[0] for z in zs]); zi = torch.cat([z[1] for z in zs])
            n = len(zr)
            Cr = (zr.T @ zr + zi.T @ zi) / n
            Ci = (zr.T @ zi - zi.T @ zr) / n
            off = (Cr ** 2 + Ci ** 2).sum() - (Cr.diagonal() ** 2 + Ci.diagonal() ** 2).sum()
            loss = loss + 10 * ((Cr.diagonal() - 1) ** 2).mean() + off / L
            opt.zero_grad(); loss.backward(); opt.step()
        Wr_, Wi_ = Wr.detach().cpu().numpy(), Wi.detach().cpu().numpy()
        per.append([(Wr_[:, k], Wi_[:, k]) for k in range(L)])
        print(f"  online module {l}: final loss {float(loss):.4f}", flush=True)
    return per


def summarize(info):
    pur = np.array([i[1] for i in info]); st = np.array([i[2] for i in info])
    return {"pairs": int(len(info)), "purity_mean": float(pur.mean()),
            "purity_median": float(np.median(pur)), "frac_pure_0.9": float((pur > 0.9).mean()),
            "frac_pure_0.99": float((pur > 0.99).mean()),
            "purity_by_strength": {
                f"<{t:g}": float(pur[st < t].mean()) if (st < t).any() else None
                for t in (1e-6, 1e-3, 1e-1, 2)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ridge", type=float, default=1e-10)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    encdir = os.path.join(a.out, "encoders")
    os.makedirs(encdir, exist_ok=True)
    res = {}
    tr = None
    for fwhm in (0.25, 0.10):
        W = World(fwhm=fwhm)
        vis = np.argwhere(W.mask)
        vis = vis[np.random.default_rng(3).permutation(len(vis))[:5_000 if SMOKE else 50_000]]
        codes = codes_at(vis, fwhm)
        learners = {"pca": lambda: pca_l1(codes, fwhm)}
        for n in ((1_000,) if SMOKE else (1_000, 10_000, 100_000)):
            def mk(n=n):
                p0, p1 = sample_steps(W.rects, n, seed=11)
                g0, g1 = codes_at(p0, fwhm).cpu().numpy(), codes_at(p1, fwhm).cpu().numpy()
                return move_l1(g0, g1, p1 - p0, a.ridge)
            learners[f"move{n // 1000}k"] = mk
        def mk_online(n=1_000 if SMOKE else 100_000):
            p0, p1 = sample_steps(W.rects, n, seed=11)
            g0, g1 = codes_at(p0, fwhm).cpu().numpy(), codes_at(p1, fwhm).cpu().numpy()
            return online_l1(g0, g1, p1 - p0, iters=50 if SMOKE else 4000)
        learners["online100k"] = mk_online
        for name, fn in learners.items():
            l1, info = _assemble(fn(), codes, fwhm)
            summ = summarize(info)
            tr = random_triples(l1, NU, seed=0)
            v, pw = slowness(l1, tr, W.steps, fwhm)
            enc, binfo = build(l1, tr, v, pw, fwhm, meta={"l1": name, "fwhm_l1": fwhm})
            m = W.metrics(enc)
            res[f"{name}_fwhm{fwhm:g}"] = {**summ, **binfo, **m}
            print(f"step1 {name} fwhm {fwhm:g}: purity mean {summ['purity_mean']:.3f} "
                  f">0.99 {summ['frac_pure_0.99']:.2f}", flush=True)
            _line(f"step1 {name} fwhm {fwhm:g}", binfo, m)
            if name in ("pca", "move100k", "online100k"):
                save_gwn(enc.cpu(), os.path.join(encdir, f"learned_{name}_fwhm{fwhm:g}.pt"))
    _dump(os.path.join(a.out, "step1.json"), res)


if __name__ == "__main__":
    main()
