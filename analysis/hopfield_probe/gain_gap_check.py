"""Why is the encoder's gain=100 not nav_p2's g=100?

nav_p2 5.7 stores p = tanh(g*xi) and at g=100 gets 83% saturated / cos 0.954 to
its own binarisation. The L7 encoder computes z = normalize(tanh(gain*net(x)))
with gain=100 and lands at cos 0.895. Same nominal gain, different corner.

The candidate the earlier note skipped: saturation is set by
`gain * |pre-activation|`, not by gain. If nav_p2's xi is order 1 per coordinate
and the encoder's net(x) is much smaller, the two "100"s are not the same
number at all. Measure the pre-activation, then sweep gain to find where this
encoder would actually reach 5.7's corner.
"""
import sys
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np
import torch

from analysis.hopfield_probe.encode import grid_codes
from analysis.hopfield_probe.harness import load_probe_encoder

R = "/orcd/pool/003/jackking/cls_runs"
ENCODERS = [
    ("v35", f"{R}/encoders/run_20260422_185816/encoder_best.pt", None),
    ("L7-s42", f"{R}/sweeps/w53_attract_knee/004_att16_seed=42/encoder_final.pt",
     None),
    ("untrained", f"{R}/encoders/untrained_mlp.pt", 0.25),
]
NPOS, M = 1716, 64
rng = np.random.RandomState(0)


def corner_stats(z):
    """cos to own binarisation, and the fraction of coordinates past 0.9*max."""
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    b = np.sign(z)
    cos_bin = np.sum(z * b, axis=1) / (np.linalg.norm(z, axis=1)
                                       * np.linalg.norm(b, axis=1))
    a = np.abs(z)
    sat = (a > 0.9 * a.max(axis=1, keepdims=True)).mean(axis=1)
    return cos_bin.mean(), sat.mean()


print("The reference: nav_p2 5.7 at g=100 -> cos_bin 0.954, ~83% saturated.\n")
print(f"{'encoder':11s} {'gain':>7s} {'pre-act RMS':>12s} {'gain*RMS':>9s} "
      f"{'cos_bin':>8s} {'sat@0.9':>8s}")
print("-" * 62)

models = {}
for name, path, fb in ENCODERS:
    enc, cfg, gain, fwhm, _ = load_probe_encoder(path, fwhm_fallback=fb)
    gx = rng.randint(0, NPOS, size=M)
    gy = rng.randint(0, NPOS, size=M)
    codes = torch.from_numpy(grid_codes(list(cfg.lambdas), gx, gy, fwhm))

    with torch.no_grad():
        pre = enc.net(codes).numpy().astype(np.float64)   # before gain & tanh
        z = enc(codes).numpy().astype(np.float64)
    rms = float(np.sqrt((pre ** 2).mean()))
    cb, sat = corner_stats(z)
    print(f"{name:11s} {gain:7.2f} {rms:12.5f} {gain * rms:9.3f} "
          f"{cb:8.4f} {sat:8.3f}")
    models[name] = (enc, codes, rms)

print("\nA unit-normal xi has RMS 1.0, so nav_p2's g=100 delivers |arg| ~ 100.")
print("\nWhat gain would this encoder need to reach 5.7's corner?")
print(f"{'encoder':11s} " + " ".join(f"g={g:<7g}" for g in
                                     (5, 100, 1e3, 1e4, 1e5)))
print("-" * 62)
for name, (enc, codes, rms) in models.items():
    row = []
    for g in (5, 100, 1e3, 1e4, 1e5):
        with torch.no_grad():
            z = torch.tanh(g * enc.net(codes))
            z = torch.nn.functional.normalize(z, dim=-1).numpy().astype(float)
        cb, _ = corner_stats(z)
        row.append(f"{cb:7.4f}")
    print(f"{name:11s} " + " ".join(row))
print("\n(cos_bin 0.954 is 5.7's turn-on. 1.0 is an exact corner.)")
