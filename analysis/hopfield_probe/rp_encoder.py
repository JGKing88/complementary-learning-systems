"""Random projection of the grid code, a threshold, then a linear readout.

The candidate biological route to the ideal encoder (docs/EXPERIMENTS_IDEAL_ENCODER.md
Sec 19): a fixed random layer ("hippocampal", genetic) followed by a learned
linear readout.

    pre_k(p) = (g(p) . W_k - mu_k) / sd_k          W ~ N(0, 1), shape (Ng, H)
    h_k(p)   = act(pre_k(p) - tau_k)               fixed, never trained
    y(p)     = (h(p) - hbar) @ A                   the readout, fitted elsewhere
    z(p)     = y(p) / |y(p)|                       unit norm, like the ideal code

``act`` is a sigmoid of slope ``gain`` (``nonlin="sigmoid"``), a hard step
(``"hard"``) or a ReLU (``"relu"``). ``tau_k`` sets each unit's activity: it is
the ``1 - active`` quantile of ``pre_k``, so ``active=0.5`` is a dense layer and
``active=0.1`` a sparse one. ``mu``, ``sd`` and ``tau`` are calibrated once on
uniform scaffold positions (a homeostatic setting independent of where the
animal has been), so one hidden layer is shared by every environment set.

``grid_codes_torch`` is ``encode.grid_codes`` on the GPU: the same wrapped
Gaussian bumps, ``fwhm_ratio`` of the period, float64 arithmetic, float32 out.

The probe loads a saved encoder with the spec ``rpnet:<path.pt>``. Its
``fwhm_ratio`` is the bump width the readout was fitted on, so the harness
feeds the matching grid code.
"""
from __future__ import annotations

import math

import numpy as np
import torch

DEFAULT_LAMBDAS = (11, 12, 13)
RPNET_PREFIX = "rpnet:"


def grid_codes_torch(xs: torch.Tensor, ys: torch.Tensor, lambdas=DEFAULT_LAMBDAS,
                     fwhm_ratio: float = 0.25) -> torch.Tensor:
    """``(N, sum l^2)`` float32 grid codes for integer positions, on ``xs.device``."""
    xs = xs.reshape(-1).long()
    ys = ys.reshape(-1).long()
    blocks = []
    for l in lambdas:
        idx = torch.arange(l, device=xs.device)
        dx = (idx[None, :] - (xs % l)[:, None]).abs()
        dx = torch.minimum(dx, l - dx).double()
        dy = (idx[None, :] - (ys % l)[:, None]).abs()
        dy = torch.minimum(dy, l - dy).double()
        if fwhm_ratio > 0:
            sigma = (l * fwhm_ratio) / (2 * math.sqrt(2 * math.log(2)))
            bx, by = torch.exp(-0.5 * (dx / sigma) ** 2), torch.exp(-0.5 * (dy / sigma) ** 2)
        else:
            bx, by = (dx == 0).double(), (dy == 0).double()
        blocks.append((bx[:, :, None] * by[:, None, :]).reshape(-1, l * l))
    return torch.cat(blocks, 1).float()


class RandomLayer(torch.nn.Module):
    """The fixed random projection + threshold, ``(N, Ng) -> (N, H)``."""

    def __init__(self, W, mu, sd, tau, nonlin: str, gain: float):
        super().__init__()
        if nonlin not in ("sigmoid", "hard", "relu"):
            raise ValueError(f"nonlin {nonlin!r}")
        self.nonlin, self.gain = nonlin, float(gain)
        for k, v in (("W", W), ("mu", mu), ("sd", sd), ("tau", tau)):
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    @property
    def H(self) -> int:
        return int(self.W.shape[1])

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        u = (codes.float() @ self.W - self.mu) / self.sd - self.tau
        if self.nonlin == "hard":
            return (u > 0).float()
        if self.nonlin == "relu":
            return torch.relu(u)
        return torch.sigmoid(self.gain * u)


def make_random_layer(H: int, *, seed: int, nonlin: str, gain: float = 1.0,
                      active: float = 0.5, fwhm_ratio: float = 0.25,
                      lambdas=DEFAULT_LAMBDAS, n_calib: int = 100_000,
                      device="cpu") -> RandomLayer:
    """Draw ``W`` and calibrate ``mu``, ``sd``, ``tau`` on uniform positions."""
    npos = int(np.prod(lambdas))
    ng = int(sum(l * l for l in lambdas))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    W = torch.randn(ng, int(H), generator=g).to(device)
    xy = torch.randint(0, npos, (2, int(n_calib)), generator=g).to(device)
    s1 = torch.zeros(int(H), dtype=torch.float64, device=device)
    s2 = torch.zeros_like(s1)
    keep = None                       # the first 16k rows set the threshold
    for i in range(0, int(n_calib), 4096):
        pre = (grid_codes_torch(xy[0, i:i + 4096], xy[1, i:i + 4096], lambdas,
                                fwhm_ratio) @ W).double()
        s1 += pre.sum(0)
        s2 += (pre ** 2).sum(0)
        if i < 16_384:
            keep = pre.float() if keep is None else torch.cat([keep, pre.float()])
    n = int(n_calib)
    mu = s1 / n
    sd = ((s2 / n - mu ** 2) * n / (n - 1)).sqrt()
    mu, sd = mu.float(), sd.float()
    z = (keep - mu) / sd
    # torch.quantile caps its input at 2^24 elements, so go 512 units at a time
    tau = torch.cat([torch.quantile(z[:, j:j + 512], 1.0 - float(active), dim=0)
                     for j in range(0, int(H), 512)])
    return RandomLayer(W, mu, sd, tau, nonlin, gain).to(device)


class RPEncoder(torch.nn.Module):
    """Grid code -> random layer -> linear readout -> unit norm."""

    def __init__(self, layer: RandomLayer, hbar, A, *, lambdas=DEFAULT_LAMBDAS,
                 fwhm_ratio: float = 0.25, gain: float = 100.0, meta: dict | None = None):
        super().__init__()
        self.layer = layer
        self.register_buffer("hbar", torch.as_tensor(hbar, dtype=torch.float32))
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
        self.lambdas = [int(l) for l in lambdas]
        self.fwhm_ratio = float(fwhm_ratio)
        self.gain = float(gain)          # Hopfield beta default, as IdealEncoder
        self.meta = dict(meta or {})

    @property
    def out_dim(self) -> int:
        return int(self.A.shape[1])

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        y = (self.layer(codes) - self.hbar) @ self.A
        return y / y.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def model_config(self):
        from .ideal_encoder import IdealModelConfig
        return IdealModelConfig(lambdas=list(self.lambdas), out_dim=self.out_dim,
                                encoder_type="rp_readout", output_nonlinearity="unit_norm",
                                hidden_dim=self.layer.H, num_hidden_layers=1,
                                gain=self.gain)


def save_rp_encoder(enc: RPEncoder, path: str) -> None:
    L = enc.layer
    torch.save({"kind": "rp_encoder", "W": L.W.cpu(), "mu": L.mu.cpu(), "sd": L.sd.cpu(),
                "tau": L.tau.cpu(), "nonlin": L.nonlin, "gain_act": L.gain,
                "hbar": enc.hbar.cpu(), "A": enc.A.cpu(), "lambdas": enc.lambdas,
                "fwhm_ratio": enc.fwhm_ratio, "gain": enc.gain, "meta": enc.meta}, path)


def load_rp_encoder(path: str, device="cpu") -> RPEncoder:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    layer = RandomLayer(ck["W"], ck["mu"], ck["sd"], ck["tau"], ck["nonlin"], ck["gain_act"])
    return RPEncoder(layer, ck["hbar"], ck["A"], lambdas=ck["lambdas"],
                     fwhm_ratio=ck["fwhm_ratio"], gain=ck["gain"],
                     meta=ck.get("meta")).to(device).eval()


def is_rpnet_spec(path) -> bool:
    return isinstance(path, str) and path.startswith(RPNET_PREFIX)


def load_rpnet_encoder(spec: str, *, device="cpu", fwhm_override=None, fwhm_fallback=None):
    """``harness.load_probe_encoder``'s return shape for ``rpnet:<path.pt>``.

    The bump width is a property of the fitted readout, so the stored value
    wins unless ``fwhm_override`` forces another.
    """
    enc = load_rp_encoder(spec[len(RPNET_PREFIX):], device=device)
    fwhm = float(fwhm_override) if fwhm_override is not None else enc.fwhm_ratio
    cfg = enc.model_config()
    L = enc.layer
    header = {
        "path": spec, "name": spec, "coverage": enc.meta.get("coverage"),
        "n_patches": enc.meta.get("n_envs"), "patch_sizes": None, "gain": enc.gain,
        "fwhm_ratio": fwhm, "fwhm_was_overridden": fwhm_override is not None,
        "lambdas": list(enc.lambdas), "out_dim": enc.out_dim,
        "hidden_dim": L.H, "num_hidden_layers": 1,
        "encoder_type": cfg.encoder_type, "output_nonlinearity": "unit_norm",
        "n_params": int(enc.A.numel()), "epoch": None, "val_nav_acc": None,
        "unique_radius": None,
        "rp": {"H": L.H, "nonlin": L.nonlin, "gain_act": L.gain, **enc.meta},
    }
    return enc, cfg, enc.gain, fwhm, header


__all__ = ["RPEncoder", "RandomLayer", "grid_codes_torch", "make_random_layer",
           "save_rp_encoder", "load_rp_encoder", "is_rpnet_spec", "load_rpnet_encoder"]
