"""Step 1 of Sec 19: can a random threshold layer express the ideal code, and does
a readout fitted only where the animal has been carry over to where it hasn't?

The environments are many squares of side 10-100 (log-uniform), dropped at
random on the 1716 x 1716 scaffold until their union covers ``coverage`` of it.
Visited = inside any of them. The readout ``A`` is a ridge regression of the
ideal r=16 code (``ideal:r=16,n_freq=512,seed=0``) on the centred hidden layer,
fitted on every cell of 90% of the environments; the ridge (1e-6 .. 10 of the
mean hidden variance) is chosen on the cells of the other 10%, so it is picked
for transfer, the best case.

Reported on three position sets -- "visited" (fitted cells: does the layer fit
the code where it was trained), unvisited, and "far" (unvisited and more than
50 cells from any visited cell):

* ``r2``: 1 - |y - z|^2 / |z - mean z|^2 of the raw readout against the ideal.
* ``near``: mean cos(z(p), z(p + d e)) over anchors and the 4 axis directions,
  for d = 1 .. 64; ``C1`` is d = 1 and ``r_half`` the interpolated d at 0.5.
* ``far_*``: cos over random pairs more than 64 cells apart (torus metric).
* ``alias_*``: cos at the grid code's own 64 most similar displacements beyond
  50 cells (where every overlap-based code is most similar to itself), mean
  over anchors, then mean / max over the displacements. The ideal code sits
  at ~0 there; the grid code itself is at its highest.

The same metrics for the ideal code on the same anchors are stored beside
every fit (key ``ideal``). One process runs every nonlinearity and coverage
for one (bump width, H).

    python -m analysis.hopfield_probe.rp_readout_check --fwhm 0.25 --H 8192 --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from .ideal_encoder import draw_frequencies
from .rp_encoder import RPEncoder, grid_codes_torch, make_random_layer

NPOS = 1716
LAMBDAS = (11, 12, 13)
VARIANTS = {            # name: (nonlin, gain, active)
    "sig1": ("sigmoid", 1.0, 0.5),
    "sig4": ("sigmoid", 4.0, 0.5),
    "sig16": ("sigmoid", 16.0, 0.5),
    "hard50": ("hard", 1.0, 0.5),
    "hard10": ("hard", 1.0, 0.1),
}
NEAR_D = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


# --- environments --------------------------------------------------------------

def make_envs(coverage: float, seed: int, smin: int = 10, smax: int = 100):
    """Union mask ``(NPOS, NPOS)`` [x, y] and the ``(x0, y0, s)`` squares."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((NPOS, NPOS), bool)
    rects = []
    while mask.mean() < coverage:
        s = int(round(math.exp(rng.uniform(math.log(smin), math.log(smax)))))
        x0, y0 = rng.integers(0, NPOS - s + 1, 2)
        mask[x0:x0 + s, y0:y0 + s] = True
        rects.append((int(x0), int(y0), s))
    return mask, rects


def position_sets(mask, seed: int, n_eval: int = 20_000, far_margin: float = 50.0):
    """Shuffled visited cells, an unvisited sample and a far-unvisited sample."""
    rng = np.random.default_rng(seed + 1)
    vis = np.argwhere(mask)
    vis = vis[rng.permutation(len(vis))]
    unv = np.argwhere(~mask)
    unv = unv[rng.choice(len(unv), min(n_eval, len(unv)), replace=False)]
    dist = distance_transform_edt(~mask)
    far = np.argwhere(dist > far_margin)
    far = far[rng.choice(len(far), min(n_eval, len(far)), replace=False)] if len(far) else far
    return vis, unv, far, float((dist > far_margin).mean())


# --- the ideal code and the grid code's aliases --------------------------------

class Ideal:
    """``ideal:r=R,n_freq=512,seed=0`` straight from positions, on the GPU."""

    def __init__(self, r: float, device, n_freq: int = 512, seed: int = 0):
        self.n = torch.from_numpy(draw_frequencies(r, n_freq, seed, NPOS)).to(device)
        self.scale = 1.0 / math.sqrt(n_freq)

    def __call__(self, xs, ys):
        k = torch.remainder(xs[:, None] * self.n[None, :, 0] + ys[:, None] * self.n[None, :, 1], NPOS)
        th = 2 * math.pi * k.double() / NPOS
        return (torch.cat([torch.cos(th), torch.sin(th)], 1) * self.scale).float()


def grid_aliases(fwhm: float, k: int = 64, min_dist: int = 50):
    """The ``k`` displacements beyond ``min_dist`` (torus) where the grid code is
    most similar to itself, with that cosine."""
    num = np.zeros((NPOS, NPOS))
    den = 0.0
    for l in LAMBDAS:
        i = np.arange(l)
        if fwhm > 0:
            sigma = (l * fwhm) / (2 * math.sqrt(2 * math.log(2)))
            d = np.minimum(i, l - i)
            b = np.exp(-0.5 * (d / sigma) ** 2)
        else:
            b = (i == 0).astype(float)
        o = np.array([np.dot(b, np.roll(b, s)) for s in range(l)])   # 1D overlap
        ox = o[np.arange(NPOS) % l]
        num += np.outer(ox, ox)
        den += o[0] ** 2
    cosg = num / den
    t = np.minimum(np.arange(NPOS), NPOS - np.arange(NPOS))
    close = np.hypot(t[:, None], t[None, :]) < min_dist
    cosg[close] = -np.inf
    flat = np.argsort(cosg.ravel())[::-1][:k]
    dx, dy = np.unravel_index(flat, cosg.shape)
    return np.stack([dx, dy], 1), cosg.ravel()[flat]


# --- metrics ------------------------------------------------------------------------

def _t(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)


def kernel_metrics(code, pos, aliases, device, seed=0, n_anchor=1000):
    """Near profile, far-field spread and alias cosines of a unit-norm ``code``."""
    rng = np.random.default_rng(seed + 7)
    a = pos[rng.choice(len(pos), min(n_anchor, len(pos)), replace=False)]
    ax, ay = _t(a[:, 0], device), _t(a[:, 1], device)
    za = code(ax, ay)
    near = {}
    for d in NEAR_D:
        c = 0.0
        for ex, ey in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            zb = code(torch.remainder(ax + d * ex, NPOS), torch.remainder(ay + d * ey, NPOS))
            c += (za * zb).sum(1).mean().item() / 4
        near[d] = c
    ds, cs = np.array(NEAR_D, float), np.array([near[d] for d in NEAR_D])
    below = np.nonzero(cs < 0.5)[0]
    r_half = None
    if len(below):
        j = below[0]
        r_half = float(ds[0] if j == 0 else
                       ds[j - 1] + (cs[j - 1] - 0.5) / (cs[j - 1] - cs[j]) * (ds[j] - ds[j - 1]))
    # far field: random partners more than 64 cells away on the torus
    n = 20_000
    i = rng.integers(0, len(pos), n)
    off = rng.integers(0, NPOS, (n, 2))
    t = np.minimum(off, NPOS - off)
    keep = np.hypot(t[:, 0], t[:, 1]) > 64
    p, off = pos[i[keep]], off[keep]
    px, py = _t(p[:, 0], device), _t(p[:, 1], device)
    cf = (code(px, py) * code(torch.remainder(px + _t(off[:, 0], device), NPOS),
                              torch.remainder(py + _t(off[:, 1], device), NPOS))).sum(1)
    cf = cf.double().cpu().numpy()
    # aliases: the grid code's own most similar far displacements
    b = a[:256]
    bx, by = _t(b[:, 0], device), _t(b[:, 1], device)
    zb0 = code(bx, by)
    al = []
    for dx, dy in aliases:
        zb = code(torch.remainder(bx + int(dx), NPOS), torch.remainder(by + int(dy), NPOS))
        al.append((zb0 * zb).sum(1).mean().item())
    al = np.array(al)
    return {"near": {str(d): near[d] for d in NEAR_D}, "C1": near[1], "r_half": r_half,
            "far_mean": float(cf.mean()), "far_sd": float(cf.std()),
            "far_p999": float(np.quantile(cf, 0.999)), "far_max": float(cf.max()),
            "alias_mean": float(al.mean()), "alias_max": float(al.max()),
            "alias_top8": float(al[:8].mean())}


# --- accumulation -------------------------------------------------------------------

def hidden_batches(layer, pos, fwhm, device, chunk):
    for s in range(0, len(pos), chunk):
        p = pos[s:s + chunk]
        x, y = _t(p[:, 0], device), _t(p[:, 1], device)
        yield x, y, layer(grid_codes_torch(x, y, LAMBDAS, fwhm))


def fit_readout(layer, ideal, fit_pos, val_pos, fwhm, device, chunk,
                lams=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Ridge readout centred hidden -> ideal code; returns (A, hbar, lam, val_r2s)."""
    H = layer.H
    G = torch.zeros(H, H, dtype=torch.float64, device=device)
    C = torch.zeros(H, 1024, dtype=torch.float64, device=device)
    sh = torch.zeros(H, dtype=torch.float64, device=device)
    sz = torch.zeros(1024, dtype=torch.float64, device=device)
    n = 0
    for x, y, h in hidden_batches(layer, fit_pos, fwhm, device, chunk):
        h = h.double()
        z = ideal(x, y).double()
        G.addmm_(h.T, h)
        C.addmm_(h.T, z)
        sh += h.sum(0)
        sz += z.sum(0)
        n += len(h)
    hbar, zbar = sh / n, sz / n
    G -= n * torch.outer(hbar, hbar)
    C -= n * torch.outer(hbar, zbar)
    scale = G.diagonal().mean().item()
    As = []
    for lam in lams:
        Gl = G.clone()
        Gl.diagonal().add_(lam * scale)
        L = torch.linalg.cholesky(Gl)
        As.append(torch.cholesky_solve(C, L).float())
        del Gl, L
    del G
    hbar32 = hbar.float()

    def r2s(pos):
        se = torch.zeros(len(lams), dtype=torch.float64, device=device)
        zsum, zsq, m = torch.zeros(1024, dtype=torch.float64, device=device), 0.0, 0
        for x, y, h in hidden_batches(layer, pos, fwhm, device, chunk):
            z = ideal(x, y)
            hc = h - hbar32
            for i, A in enumerate(As):
                se[i] += ((hc @ A - z).double() ** 2).sum()
            zsum += z.double().sum(0)
            zsq += (z.double() ** 2).sum().item()
            m += len(h)
        return (1 - se / (zsq - (zsum ** 2).sum().item() / m)).cpu().numpy()

    r2 = r2s(val_pos)                       # held-out environments: picks the ridge
    r2_in = r2s(fit_pos[:20_000])           # fitted cells: how well the layer fits at all
    best = int(np.argmax(r2))
    return (As[best], hbar32, float(lams[best]),
            {str(l): float(v) for l, v in zip(lams, r2)},
            {str(l): float(v) for l, v in zip(lams, r2_in)})


def r2_on(enc_raw, ideal, pos, device, chunk=8192):
    se, zsq, zsum, m = 0.0, 0.0, None, 0
    cos = 0.0
    for s in range(0, len(pos), chunk):
        p = pos[s:s + chunk]
        x, y = _t(p[:, 0], device), _t(p[:, 1], device)
        yh, z = enc_raw(x, y).double(), ideal(x, y).double()
        se += ((yh - z) ** 2).sum().item()
        zsq += (z ** 2).sum().item()
        zsum = z.sum(0) if zsum is None else zsum + z.sum(0)
        cos += (torch.nn.functional.normalize(yh, dim=1) * z / z.norm(dim=1, keepdim=True)).sum().item()
        m += len(p)
    return 1 - se / (zsq - (zsum ** 2).sum().item() / m), cos / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwhm", type=float, required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--coverages", default="0.01,0.05,0.2")
    ap.add_argument("--r", type=float, default=16.0)
    ap.add_argument("--hidden_seed", type=int, default=0)
    ap.add_argument("--env_seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--save_encoder", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chunk = a.chunk or max(1024, min(32768, int(2.5e8 // a.H)))
    os.makedirs(a.out, exist_ok=True)
    ideal = Ideal(a.r, dev)
    aliases, alias_cos = grid_aliases(a.fwhm)
    for cov in [float(c) for c in a.coverages.split(",")]:
        mask, rects = make_envs(cov, a.env_seed)
        vis, unv, far, far_frac = position_sets(mask, a.env_seed)
        # the ridge is chosen on whole held-out environments (10% of them), so
        # neighbouring cells of a fitted environment cannot pass for transfer
        rng = np.random.default_rng(a.env_seed + 2)
        order = rng.permutation(len(rects))
        n_ho = max(1, len(rects) // 10)
        fit_mask = np.zeros_like(mask)
        for j in order[n_ho:]:
            x0, y0, s = rects[j]
            fit_mask[x0:x0 + s, y0:y0 + s] = True
        fit_pos = np.argwhere(fit_mask)
        fit_pos = fit_pos[rng.permutation(len(fit_pos))]
        val_pos = np.argwhere(mask & ~fit_mask)
        if len(val_pos) < 500:                    # held-out envs swallowed by overlaps
            raise RuntimeError(f"only {len(val_pos)} held-out cells at coverage {cov}")
        sets = {"visited": fit_pos[:20_000], "unvisited": unv}
        if len(far) >= 1000:
            sets["far"] = far
        ideal_m = {k: kernel_metrics(ideal, p, aliases, dev) for k, p in sets.items()}
        for name in a.variants.split(","):
            tag = f"fwhm{a.fwhm:g}_H{a.H}_{name}_cov{cov:g}"
            path = os.path.join(a.out, f"{tag}.json")
            if os.path.exists(path):
                print("skip", tag, flush=True)
                continue
            t0 = time.time()
            nonlin, gain, active = VARIANTS[name]
            layer = make_random_layer(a.H, seed=a.hidden_seed, nonlin=nonlin, gain=gain,
                                      active=active, fwhm_ratio=a.fwhm, device=dev)
            A, hbar, lam, val_r2, in_r2 = fit_readout(layer, ideal, fit_pos, val_pos, a.fwhm,
                                                      dev, chunk)
            enc = RPEncoder(layer, hbar, A, fwhm_ratio=a.fwhm,
                            meta={"coverage": cov, "n_envs": len(rects), "variant": name,
                                  "target": f"ideal:r={a.r:g},n_freq=512,seed=0", "lam": lam})

            def raw(x, y):
                return (layer(grid_codes_torch(x, y, LAMBDAS, a.fwhm)) - hbar) @ A

            def unit(x, y):
                return enc(grid_codes_torch(x, y, LAMBDAS, a.fwhm))

            res = {"tag": tag, "fwhm": a.fwhm, "H": a.H, "variant": name, "nonlin": nonlin,
                   "gain_act": gain, "active": active, "coverage": cov,
                   "coverage_actual": float(mask.mean()), "n_envs": len(rects),
                   "n_fit": int(len(fit_pos)), "n_val_heldout_envs": int(len(val_pos)),
                   "far_frac": far_frac, "lam": lam,
                   "val_r2_by_lam": val_r2, "fit_r2_by_lam": in_r2,
                   "fit_r2_best": max(in_r2.values()), "alias_grid_cos": alias_cos[:8].tolist(),
                   "sets": {}, "ideal": ideal_m}
            with torch.no_grad():
                for k, p in sets.items():
                    r2, cosi = r2_on(raw, ideal, p, dev)
                    res["sets"][k] = {"r2": r2, "cos_to_ideal": cosi,
                                      **kernel_metrics(unit, p, aliases, dev)}
            res["seconds"] = time.time() - t0
            with open(path, "w") as f:
                json.dump(res, f, indent=1)
            if a.save_encoder:
                from .rp_encoder import save_rp_encoder
                save_rp_encoder(enc.cpu(), os.path.join(a.out, f"{tag}.pt"))
            s = res["sets"]
            print(f"{tag}: lam {lam:g} r2 fit-best {res['fit_r2_best']:.3f} heldout-env "
                  f"{val_r2[str(lam)]:.3f} vis {s['visited']['r2']:.3f} unv {s['unvisited']['r2']:.3f}"
                  f" | alias_max unv {s['unvisited']['alias_max']:.3f} r_half "
                  f"{s['unvisited']['r_half']} far_sd {s['unvisited']['far_sd']:.3f}"
                  f" | {res['seconds']:.0f}s", flush=True)
            del layer, A, enc
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
