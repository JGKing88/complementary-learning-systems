"""Data for the rectified-ideal tuning page (``analysis/rect_ideal/page/viewer.html``).

    python -m analysis.rect_ideal.export_page [--out analysis/rect_ideal/page/data]
        [--variant m111213 --variant ...] [--results-only]

For each grid-module variant (``codes.VARIANTS``; all of them by default) it writes:

* ``rect_manifest_<variant>.json``: the scaffold (npos, modules, lambda_ref, r_ideal), and for each
  (f, D) code its integer frequencies, amplitude, threshold and r; for each D the rotation's random
  signs. The page computes every map from these with the same transform as
  ``IdealEncoder.closed_form`` (cos and sin of the integer phase, three rounds of sign flips and
  Walsh-Hadamard, threshold, unit length at each cell).
* ``rect_ref_<variant>.b64.txt``: Python's own values (``closed_form``, float64) for 16 units of
  every code on the check arena (16 x 16 cells at (1708, 1710), which wraps both edges of the
  1716 scaffold); the page checks itself against them. Layout: code by code in manifest order, then
  cell ``i = row * 16 + col`` (x = x0 + col, y = y0 + row), then the code's 16 check units.
* ``rect_results_<variant>.json``: the variant's ``grid_results.json`` (``summary.py``), when it
  exists.

and ``rect_variants.json``, the list of variants the page offers. ``--results-only`` refreshes the
results files and the index without recomputing the codes.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess

import numpy as np

from analysis.hopfield_probe.ideal_encoder import IdealEncoder, parse_ideal_spec
from analysis.rect_ideal.codes import DS, FS, VARIANTS, f_tag, field_tasks, matched_r, npos
from cls_paths import results_dir

REF = {"x0": 1708, "y0": 1710, "S": 16}
N_CHECK = 16
# The page's threshold slider: share of units active, 1% to 60% in 0.5% steps (the six computed
# sparsities are on it).
F_GRID = [round(0.01 + 0.005 * i, 4) for i in range(119)]
N_GAUSS = max(DS) // 2        # draws for the largest n_freq; smaller codes use a prefix
LABELS = {"m111213": "11, 12, 13 (r = 16)", "m111523_r16": "11, 15, 23, r = 16",
          "m111523_rs": "11, 15, 23, r scaled (20.8)"}


def export_codes(variant: str, out: str, sha: str) -> None:
    N = npos(variant)
    r_, c_ = np.divmod(np.arange(REF["S"] ** 2), REF["S"])
    xs, ys = (REF["x0"] + c_) % N, (REF["y0"] + r_) % N
    codes, signs, refs = [], {}, []
    for spec, label in field_tasks(variant):
        enc = IdealEncoder(**parse_ideal_spec(spec)).eval()
        assert enc.npos == N
        D = enc.out_dim
        amp = enc.amp.cpu().numpy()
        assert np.allclose(amp, amp[0]), "the page assumes one amplitude for every wave"
        if str(D) not in signs:
            s = np.random.default_rng(enc.rot_seed).choice([-1, 1], size=(3, D))
            signs[str(D)] = s.astype(int).tolist()
        units = np.unique(np.round(np.linspace(0, D - 1, N_CHECK)).astype(int))
        refs.append(enc.closed_form(xs, ys)[:, units].astype(np.float32).ravel())
        codes.append({
            "key": f"{f_tag(enc.active)}_D{D}", "label": label, "spec": spec,
            "f": enc.active, "D": D, "n_freq": enc.n_freq, "r": enc.r,
            "amp": float(amp[0]), "thresh": float(enc.thresh),
            "freqs": enc.freqs.cpu().numpy().astype(int).ravel().tolist(),
            "check_units": units.tolist(),
        })
        print(f"{variant} {label}: D {D}, r {enc.r:.3f}, thresh {enc.thresh:.5f}")
    with open(os.path.join(out, f"rect_ref_{variant}.b64.txt"), "w") as fh:
        fh.write(base64.b64encode(np.concatenate(refs).astype("<f4").tobytes()).decode())
    v = VARIANTS[variant]
    man = {"variant": variant, "label": LABELS[variant], "npos": N, "lambdas": list(v["lambdas"]),
           "lambda_ref": v["lambda_ref"], "r_ideal": v["r_ideal"], "Fs": list(FS), "Ds": list(DS),
           "ref_arena": REF, "n_check": N_CHECK, "codes": codes, "signs": signs, "git_sha": sha,
           # r matched at every slider position, rounded as the codes' specs round it, so the page
           # rebuilds the six computed codes exactly (draw_frequencies from rect_gauss.b64.txt).
           "r_table": {"f": F_GRID, "r": [round(matched_r(f, variant), 4) for f in F_GRID]}}
    with open(os.path.join(out, f"rect_manifest_{variant}.json"), "w") as fh:
        json.dump(man, fh, separators=(",", ":"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "page", "data"))
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS))
    ap.add_argument("--results-only", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    variants = args.variant or list(VARIANTS)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, cwd=here).stdout.strip()
    # draw_frequencies(r, n, seed=0) is rint(sigma * g) with g these standard normals (legacy
    # RandomState.normal is loc + scale * gauss), reduced into (-Npos/2, Npos/2].
    g = np.random.RandomState(0).normal(0.0, 1.0, size=(N_GAUSS, 2))
    with open(os.path.join(args.out, "rect_gauss.b64.txt"), "w") as fh:
        fh.write(base64.b64encode(np.ascontiguousarray(g, dtype="<f8").tobytes()).decode())
    index = []
    for variant in VARIANTS:
        if variant in variants and not args.results_only:
            export_codes(variant, args.out, sha)
        res = results_dir() / VARIANTS[variant]["results"] / "grid_results.json"
        dest = os.path.join(args.out, f"rect_results_{variant}.json")
        if variant in variants and res.exists():
            shutil.copy(res, dest)
        v = VARIANTS[variant]
        index.append({"key": variant, "label": LABELS[variant], "npos": npos(variant),
                      "lambdas": list(v["lambdas"]), "lambda_ref": v["lambda_ref"],
                      "r_ideal": v["r_ideal"], "has_results": os.path.exists(dest)})
    with open(os.path.join(args.out, "rect_variants.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"wrote {args.out}: {[(i['key'], i['has_results']) for i in index]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
