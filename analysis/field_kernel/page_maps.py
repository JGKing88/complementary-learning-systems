"""2D rate maps of the same units at several fast shares, for docs/run_pages/field_kernel.html.

    python -m analysis.field_kernel.page_maps > maps.json

Unit u's input is ``sqrt(1 - a) y_slow + sqrt(a) y_fast``: y_slow is its input in the one-scale
code (r = 23.2), y_fast its input in a one-scale code at r = 3 with another frequency seed and the
same rotation. This has the same correlation as the ``fast=a,r_fast=3`` spec and keeps a unit's
slow part fixed across shares, so the columns show one unit. Rates are ``relu(y - t)`` at 10%
active, each map scaled to its own peak. Output: base64 RGBA PNGs (alpha = rate / peak).
"""
from __future__ import annotations

import base64
import io
import json
import math

import numpy as np
from PIL import Image

from analysis.field_kernel.measure import build

SIDE = 80
SHARES = (0.0, 0.05, 0.1, 0.3, 0.5)
N_UNITS = 4
T = 1.2816


def drive(spec: str, gx, gy) -> np.ndarray:
    e = build(spec)
    return e.drive(e._waves_np(gx, gy)) * math.sqrt(e.out_dim)


def main() -> None:
    gx, gy = np.meshgrid(np.arange(SIDE) + 700, np.arange(SIDE) + 300, indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    ys = drive("ideal:r=23.2,n_freq=512,seed=0,relu_mix=1,active=0.1", gx, gy)
    yf = drive("ideal:r=3,n_freq=512,seed=1,relu_mix=1,active=0.1", gx, gy)
    # units with one clear field well inside the arena in the one-scale code
    m = ys.reshape(SIDE, SIDE, -1)
    peak = m.reshape(-1, m.shape[2]).argmax(0)
    px, py = peak // SIDE, peak % SIDE
    ok = (ys.max(0) > 2.0) & (px > 20) & (px < 60) & (py > 20) & (py < 60)
    units = np.flatnonzero(ok)[:N_UNITS]
    out = {"side": SIDE, "shares": list(SHARES), "units": [int(u) for u in units], "maps": []}
    for u in units:
        row = []
        for a in SHARES:
            y = math.sqrt(1 - a) * ys[:, u] + math.sqrt(a) * yf[:, u]
            r = np.maximum(y - T, 0).reshape(SIDE, SIDE).T[::-1]      # y up
            al = (255 * r / max(r.max(), 1e-9)).astype(np.uint8)
            img = np.zeros((SIDE, SIDE, 4), np.uint8)
            img[..., 0], img[..., 1], img[..., 2], img[..., 3] = 42, 120, 214, al
            buf = io.BytesIO()
            Image.fromarray(img, "RGBA").save(buf, format="PNG", optimize=True)
            row.append(base64.b64encode(buf.getvalue()).decode())
        out["maps"].append(row)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
