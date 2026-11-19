"""Data for the rectified-ideal tuning page (``analysis/rect_ideal/page/viewer.html``).

    python -m analysis.rect_ideal.export_page [--out analysis/rect_ideal/page/data]
        [--results ROOT/grid_results.json]

Writes ``rect_manifest.json``: for each (f, D) code, its integer frequencies, amplitude, threshold
and r, and for each D the rotation's random signs. The page computes every map from these with the
same transform as ``IdealEncoder.closed_form`` (cos and sin of the integer phase, three rounds of
sign flips and Walsh-Hadamard, threshold, unit length at each cell).

``rect_ref.b64.txt`` holds Python's own values (``closed_form``, float64) for 16 units of every code
on the old viewer's check arena (16 x 16 cells at (1708, 1710), wrapping both scaffold edges); the
page checks itself against them at load. Layout: code by code in manifest order, then cell
``i = row * 16 + col`` (x = x0 + col, y = y0 + row), then the code's 16 check units.

With ``--results`` it also copies the summary's ``grid_results.json`` to ``rect_results.json``.
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
from analysis.rect_ideal.codes import DS, FS, LAMBDA_REF, NPOS, R_IDEAL, f_tag, field_tasks

REF = {"x0": 1708, "y0": 1710, "S": 16}
N_CHECK = 16


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "page", "data"))
    ap.add_argument("--results", default=None)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    r_, c_ = np.divmod(np.arange(REF["S"] ** 2), REF["S"])
    xs, ys = (REF["x0"] + c_) % NPOS, (REF["y0"] + r_) % NPOS
    codes, signs, refs = [], {}, []
    for spec, label in field_tasks():
        kw = parse_ideal_spec(spec)
        enc = IdealEncoder(**kw).eval()
        D = enc.out_dim
        amp = enc.amp.cpu().numpy()
        assert np.allclose(amp, amp[0]), "the page assumes one amplitude for every wave"
        s = np.random.default_rng(enc.rot_seed).choice([-1, 1], size=(3, D))
        if str(D) not in signs:
            signs[str(D)] = s.astype(int).tolist()
        units = np.unique(np.round(np.linspace(0, D - 1, N_CHECK)).astype(int))
        ref = enc.closed_form(xs, ys)[:, units].astype(np.float32)
        refs.append(ref.ravel())
        codes.append({
            "key": f"{f_tag(enc.active)}_D{D}", "label": label, "spec": spec,
            "f": enc.active, "D": D, "n_freq": enc.n_freq, "r": enc.r,
            "amp": float(amp[0]), "thresh": float(enc.thresh),
            "freqs": enc.freqs.cpu().numpy().astype(int).ravel().tolist(),
            "check_units": units.tolist(),
        })
        print(f"{label}: D {D}, r {enc.r:.3f}, thresh {enc.thresh:.5f}")

    with open(os.path.join(args.out, "rect_ref.b64.txt"), "w") as fh:
        fh.write(base64.b64encode(np.concatenate(refs).astype("<f4").tobytes()).decode())
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, cwd=here).stdout.strip()
    man = {"npos": NPOS, "lambda_ref": LAMBDA_REF, "r_ideal": R_IDEAL, "Fs": list(FS),
           "Ds": list(DS), "ref_arena": REF, "n_check": N_CHECK, "codes": codes,
           "signs": signs, "git_sha": sha}
    with open(os.path.join(args.out, "rect_manifest.json"), "w") as fh:
        json.dump(man, fh, separators=(",", ":"))
    if args.results:
        shutil.copy(args.results, os.path.join(args.out, "rect_results.json"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
