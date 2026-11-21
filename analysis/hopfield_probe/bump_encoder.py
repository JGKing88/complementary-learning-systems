"""An analytic place code of independent Gaussian bumps, read off the grid code.

Each of D units has its own random place fields on the Npos x Npos scaffold torus (scaffold cells
are the unit of length):

* the number of fields is Poisson with mean ``dens * Npos^2 / 1e4`` (``dens`` = fields per unit per
  100 x 100 cells);
* centres are uniform on the torus, continuous;
* each field is a Gaussian bump with peak 1, ``exp(-d^2 / (2 sigma^2))``, with ``d`` the torus
  distance, so fields wrap around the edges;
* widths are log-normal with the MEAN full width at half maximum fixed at ``fwhm`` cells:
  ``sigma = sigma_med * exp(spread * g)``, ``g ~ N(0, 1)``, ``sigma_med = fwhm / 2.3548 *
  exp(-spread^2 / 2)``;
* overlapping fields add.

The output at each place is the D unit rates scaled to unit length, like the other analytic codes,
so cosine storage, recall and the q readout behave as for the ideal codes. This is the model of
``analysis/rect_ideal/place_kernel.html`` (``cellFields``, ``addField``), with its own random
stream (numpy ``default_rng(seed)``, not the page's per-cell mulberry32 streams). The same seed
draws the same counts, centres and unit normals ``g`` at any spread, so changing only the spread
changes only the widths.

For Poisson fields at density ``rho = dens / 1e4`` per cell^2 the mean rate is ``mu = rho E[2 pi
sigma^2]`` and the rate covariance at distance d is ``C(d) = rho E[pi sigma^2 exp(-d^2 / (4
sigma^2))]``. So for large D the raw cosine is ``(C(d) + mu^2) / (C(0) + mu^2)``, with far-field
floor ``2 mu / (1 + 2 mu)``, and after one global mean constant is removed it is ``C(d) / C(0)``.
With one width that is ``exp(-d^2 / (4 sigma^2))``, half-width ``FWHM / sqrt(2)``.

Evaluation uses a tile index, because the probe asks for millions of places and a unit has about
200 fields on the full scaffold. The torus is cut into T x T tiles (T divides Npos). Each tile
holds every field whose disc of radius ``cut * sigma`` reaches one of the tile's cells, with the
field centre stored relative to the tile's origin through the right torus image. A place then
evaluates only its own tile's list, in vectorised torch, and scatter-adds the bumps into the units.
The fields left out are each below ``exp(-cut^2 / 2)`` at that place (3.7e-6 at the default cut of
5). ``exact_rates`` evaluates every field at every place in float64, for checks.

Position is decoded exactly from the grid code: ``IdealEncoder`` with the two frequencies (1, 0)
and (0, 1) gives the angles ``2 pi x / Npos`` and ``2 pi y / Npos`` by the Chinese remainder theorem
(``ideal_encoder.py``, steps 1 and 2), and rounding recovers the integer cell.

``load_probe_encoder`` takes a ``--ckpt`` of the form

    bump:D=1024,fwhm=27,dens=0.66[,spread=0][,seed=0][,cut=5][,tile=T][,gain=100][,lambdas=11-12-13]

``gain`` is carried only because the probe reads it as the default Hopfield beta; it does not touch
the code. The probe's ideal codes use 100.
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch

from .ideal_encoder import DEFAULT_LAMBDAS, IdealEncoder, IdealModelConfig

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))      # 2.3548
BUMP_PREFIX = "bump:"


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


class BumpEncoder(torch.nn.Module):
    """Grid code ``(N, sum l^2)`` -> unit-length rates of D units with Poisson Gaussian fields."""

    def __init__(self, D: int, fwhm: float, dens: float, spread: float = 0.0, seed: int = 0,
                 cut: float = 5.0, tile: int | None = None, gain: float = 100.0,
                 lambdas=DEFAULT_LAMBDAS, budget: int = 1 << 22):
        super().__init__()
        self.D = int(D)
        self.fwhm = float(fwhm)
        self.dens = float(dens)
        self.spread = float(spread)
        self.seed = int(seed)
        self.cut = float(cut)
        self.gain = float(gain)
        self.lambdas = [int(l) for l in lambdas]
        self.npos = int(np.prod(self.lambdas))
        self.budget = int(budget)            # elements per (places x tile list) block
        N = self.npos

        # --- the fields ---------------------------------------------------------------------
        rng = np.random.default_rng(self.seed)
        self.lam = self.dens * N * N / 1e4
        counts = rng.poisson(self.lam, self.D)
        F = int(counts.sum())
        self.unit = np.repeat(np.arange(self.D), counts)
        self.cx = rng.uniform(0.0, N, F)
        self.cy = rng.uniform(0.0, N, F)
        g = rng.standard_normal(F)
        self.sigma_med = self.fwhm / FWHM_PER_SIGMA * math.exp(-0.5 * self.spread ** 2)
        self.sigma = self.sigma_med * np.exp(self.spread * g)
        self.counts = counts
        self.n_fields = F

        # --- the tile index -----------------------------------------------------------------
        R = self.cut * self.sigma
        if tile is None:
            # About half the median cut radius: a place's list covers (T + 2R)^2, the storage
            # grows as (2R / T + 1)^2 per field.
            target = max(4.0, 0.5 * self.cut * self.sigma_med)
            tile = min(_divisors(N), key=lambda d: abs(math.log(d / target)))
        self.T = int(tile)
        if N % self.T:
            raise ValueError(f"tile {self.T} must divide Npos {N}")
        self.nt = N // self.T
        if F and R.max() >= N / 2 - self.T:
            raise ValueError(f"widest field's cut radius {R.max():.0f} reaches round the torus")

        T = self.T
        tiles, fids, rx, ry = [], [], [], []
        step = max(1, 4_000_000 // max(1, int(((2 * R.mean() / T + 2) ** 2))))
        for a in range(0, F, step):
            b = min(F, a + step)
            cx, cy, r = self.cx[a:b], self.cy[a:b], R[a:b]
            tx0 = np.floor((cx - r) / T).astype(np.int64)
            tx1 = np.floor((cx + r) / T).astype(np.int64)
            ty0 = np.floor((cy - r) / T).astype(np.int64)
            ty1 = np.floor((cy + r) / T).astype(np.int64)
            nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
            ncand = nx * ny
            f = np.repeat(np.arange(a, b), ncand)
            start = np.repeat(np.cumsum(ncand) - ncand, ncand)
            k = np.arange(f.size) - start
            nyf = np.repeat(ny, ncand)
            tx = np.repeat(tx0, ncand) + k // nyf
            ty = np.repeat(ty0, ncand) + k % nyf
            # Nearest cell of the tile (cells txT .. txT + T - 1) to the centre.
            fx, fy = self.cx[f], self.cy[f]
            qx = np.clip(fx, tx * T, tx * T + T - 1)
            qy = np.clip(fy, ty * T, ty * T + T - 1)
            keep = (fx - qx) ** 2 + (fy - qy) ** 2 <= R[f] ** 2
            f, tx, ty = f[keep], tx[keep], ty[keep]
            tiles.append(np.mod(tx, self.nt) * self.nt + np.mod(ty, self.nt))
            fids.append(f)
            # Centre relative to the UNWRAPPED tile origin: a place's offset in its wrapped tile
            # minus this is the displacement to the field's nearest torus image.
            rx.append(self.cx[f] - tx * T)
            ry.append(self.cy[f] - ty * T)
        tiles = np.concatenate(tiles) if tiles else np.zeros(0, np.int64)
        fids = np.concatenate(fids) if fids else np.zeros(0, np.int64)
        rx = np.concatenate(rx) if rx else np.zeros(0)
        ry = np.concatenate(ry) if ry else np.zeros(0)
        n_tiles = self.nt * self.nt
        order = np.lexsort((self.unit[fids] if F else fids, tiles))
        tiles, fids, rx, ry = tiles[order], fids[order], rx[order], ry[order]
        per = np.bincount(tiles, minlength=n_tiles)
        self.L = max(1, int(per.max()) if per.size else 1)
        rank = np.arange(tiles.size) - np.repeat(np.cumsum(per) - per, per)
        # Padding: offset 0, inverse width 0 -> value 1, scattered into a spare unit D (dropped).
        RX = np.zeros((n_tiles, self.L), np.float32)
        RY = np.zeros((n_tiles, self.L), np.float32)
        IV = np.zeros((n_tiles, self.L), np.float32)
        U = np.full((n_tiles, self.L), self.D, np.int64)
        RX[tiles, rank] = rx
        RY[tiles, rank] = ry
        IV[tiles, rank] = 1.0 / (2.0 * self.sigma[fids] ** 2)
        U[tiles, rank] = self.unit[fids]
        self.register_buffer("RX", torch.from_numpy(RX))
        self.register_buffer("RY", torch.from_numpy(RY))
        self.register_buffer("IV", torch.from_numpy(IV))
        self.register_buffer("U", torch.from_numpy(U))
        self.list_mean = float(per.mean())
        self.entries = int(tiles.size)

        # Exact position decode: angles 2 pi x / Npos and 2 pi y / Npos from the module phases.
        self.decoder = IdealEncoder(r=1.0, n_freq=2, freqs=np.array([[1, 0], [0, 1]]),
                                    lambdas=self.lambdas)
        self.in_dim = self.decoder.in_dim

    @property
    def out_dim(self) -> int:
        return self.D

    def model_config(self) -> IdealModelConfig:
        return IdealModelConfig(lambdas=list(self.lambdas), out_dim=self.D, gain=self.gain,
                                encoder_type="bump")

    # --- position --------------------------------------------------------------------------

    def positions(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Integer scaffold cell ``(x, y)`` of each grid code, int64 in ``[0, Npos)``."""
        theta = self.decoder._phases(codes) @ self.decoder.harmonics.to(codes.device)
        p = torch.round(theta * (self.npos / (2.0 * math.pi))).to(torch.int64)
        p = torch.remainder(p, self.npos)
        return p[:, 0], p[:, 1]

    # --- rates -----------------------------------------------------------------------------

    def rates(self, xs, ys) -> torch.Tensor:
        """Raw rates ``(N, D)`` float32 at integer cells, from the tile index."""
        def cells(v):
            v = v.to(torch.int64) if torch.is_tensor(v) else torch.from_numpy(
                np.asarray(v, dtype=np.int64))
            return torch.remainder(v.cpu().reshape(-1), self.npos)

        xs, ys = cells(xs), cells(ys)
        n = xs.numel()
        out = torch.zeros(n, self.D, dtype=torch.float32)
        T = self.T
        tx, ty = torch.div(xs, T, rounding_mode="floor"), torch.div(ys, T, rounding_mode="floor")
        t = tx * self.nt + ty
        lx = (xs - tx * T).to(torch.float32)
        ly = (ys - ty * T).to(torch.float32)
        rows = max(1, self.budget // self.L)
        for a in range(0, n, rows):
            b = min(n, a + rows)
            ti = t[a:b]
            d = lx[a:b, None] - self.RX.index_select(0, ti)
            d.square_()
            e = ly[a:b, None] - self.RY.index_select(0, ti)
            e.square_()
            d.add_(e)
            del e
            d.mul_(self.IV.index_select(0, ti)).neg_().exp_()
            acc = torch.zeros(b - a, self.D + 1, dtype=torch.float32)
            acc.scatter_add_(1, self.U.index_select(0, ti), d)
            out[a:b] = acc[:, :self.D]
        return out

    def code(self, xs, ys) -> torch.Tensor:
        """Unit-length rates ``(N, D)`` float32."""
        r = self.rates(xs, ys)
        return r / torch.linalg.norm(r, dim=1, keepdim=True).clamp_min(1e-12)

    def exact_rates(self, xs, ys, chunk: int = 4_000_000) -> np.ndarray:
        """Every field at every place, float64 numpy, nearest torus image. Slow; for checks."""
        xs = np.mod(np.asarray(xs, np.int64).reshape(-1), self.npos).astype(np.float64)
        ys = np.mod(np.asarray(ys, np.int64).reshape(-1), self.npos).astype(np.float64)
        N, h = self.npos, self.npos / 2.0
        out = np.zeros((xs.size, self.D))
        iv = 1.0 / (2.0 * self.sigma ** 2)
        per = max(1, chunk // max(1, self.n_fields))
        for a in range(0, xs.size, per):
            b = min(xs.size, a + per)
            dx = np.mod(xs[a:b, None] - self.cx[None, :] + h, N) - h
            dy = np.mod(ys[a:b, None] - self.cy[None, :] + h, N) - h
            g = np.exp(-(dx * dx + dy * dy) * iv[None, :])
            for i in range(b - a):
                out[a + i] = np.bincount(self.unit, weights=g[i], minlength=self.D)
        return out

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[-1] != self.in_dim:
            raise ValueError(f"expected (N, {self.in_dim}) grid codes, got {tuple(codes.shape)}")
        xs, ys = self.positions(codes)
        out = self.code(xs.cpu(), ys.cpu()).to(codes.device)
        return out.to(codes.dtype if codes.is_floating_point() else torch.float32)

    def describe(self) -> dict:
        return {"D": self.D, "fwhm_mean": self.fwhm, "spread": self.spread, "dens": self.dens,
                "seed": self.seed, "cut": self.cut, "tile": self.T, "npos": self.npos,
                "lambdas": list(self.lambdas), "fields_per_unit_expected": self.lam,
                "fields_per_unit_mean": float(self.counts.mean()),
                "fields_per_unit_min": int(self.counts.min()),
                "fields_per_unit_max": int(self.counts.max()),
                "n_fields": self.n_fields, "sigma_median": self.sigma_med,
                "fwhm_realised_mean": float(self.sigma.mean() * FWHM_PER_SIGMA) if self.n_fields
                else None,
                "tile_list_mean": self.list_mean, "tile_list_max": self.L}


# --- probe-harness hook -----------------------------------------------------------------------

def is_bump_spec(path) -> bool:
    return isinstance(path, str) and path.startswith(BUMP_PREFIX)


def parse_bump_spec(spec: str) -> dict:
    kw: dict = {}
    for part in filter(None, re.split(r"[,;]", spec[len(BUMP_PREFIX):])):
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if k in ("D", "seed", "tile"):
            kw[k] = int(v)
        elif k in ("fwhm", "dens", "spread", "cut", "gain"):
            kw[k] = float(v)
        elif k == "lambdas":
            kw[k] = tuple(int(x) for x in v.split("-"))
        else:
            raise ValueError(f"unknown bump-encoder key {k!r} in {spec!r}")
    for need in ("D", "fwhm", "dens"):
        if need not in kw:
            raise ValueError(f"{spec!r} needs {need}=")
    return kw


def load_bump_encoder(spec: str, *, device="cpu", fwhm_override=None, fwhm_fallback=None):
    """Same return shape as ``harness.load_probe_encoder``.

    Position decodes exactly at any grid-code smoothing, so ``fwhm_ratio`` only decides which grid
    code is fed in; it defaults to production's 0.25. The tile index lives on the CPU.
    """
    enc = BumpEncoder(**parse_bump_spec(spec)).eval()
    fwhm = (fwhm_override if fwhm_override is not None else
            fwhm_fallback if fwhm_fallback is not None else 0.25)
    cfg = enc.model_config()
    header = {
        "path": spec, "name": spec, "coverage": None, "n_patches": None,
        "patch_sizes": None, "gain": enc.gain, "fwhm_ratio": float(fwhm),
        "fwhm_was_overridden": fwhm_override is not None,
        "lambdas": list(enc.lambdas), "out_dim": enc.out_dim,
        "hidden_dim": 0, "num_hidden_layers": 0,
        "encoder_type": cfg.encoder_type, "output_nonlinearity": cfg.output_nonlinearity,
        "n_params": 0, "epoch": None, "val_nav_acc": None, "unique_radius": None,
        "bump": enc.describe(),
    }
    return enc, cfg, enc.gain, float(fwhm), header


__all__ = ["BumpEncoder", "FWHM_PER_SIGMA", "is_bump_spec", "load_bump_encoder",
           "parse_bump_spec"]
