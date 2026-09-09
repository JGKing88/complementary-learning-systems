"""Can a GRADED code be an exact fixed point? Yes -- change the recurrence.

Sec 7.1's synthesis argued: an exact fixed point requires every code to be a
fixed point of the saturating map, hence a hypercube corner, hence an EXTREME
point of the convex hull -- and extreme points are never midpoints, so no code
can be interpolated. That argument is correct **for the recurrence we run**,
`x <- normalize(sign(W x))`, and it silently assumes that recurrence.

It is not a fact about graded codes. Three storage rules on the identical
patterns, one step and fifteen:

  hebb    W = (1/D)(Z^T Z - diag)          what the project uses. The stored
                                           pattern is only NEAR an eigenvector,
                                           because the patterns overlap.

  proj    W = Z^T (Z Z^T)^-1 Z             the projection rule (Personnaz et
                                           al.): the orthogonal projector onto
                                           span(Z). Then W z_i = z_i EXACTLY for
                                           any linearly independent patterns,
                                           graded or binary. But a projector is
                                           idempotent and its eigenvalues on the
                                           span are all 1, so every BLEND is
                                           equally fixed -- fixed points without
                                           attraction.

  soft    x <- normalize(Z^T softmax(beta Z x))    modern Hopfield / dense
                                           associative memory (Krotov & Hopfield;
                                           Ramsauer et al.). The softmax selects
                                           the nearest pattern instead of
                                           blending, so stored patterns are
                                           ATTRACTING fixed points -- and for
                                           continuous patterns, which is the
                                           point.

So "binary codes are special" is false; "the saturating pointwise nonlinearity
has the corners as its fixed set" is true, and those are different claims.

Reported: cos(recall(z), z) for the stored patterns, and retrieval of a cue at
radius r, both at 1 and 15 steps.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (ProbeConfig, build_memory,
                                             load_probe_encoder, local_cells,
                                             sample_worlds, scored_envs)

DEFAULT_DIR = os.path.join(
    os.environ.get("CLS_RESULTS", "/orcd/pool/003/jackking/cls_runs/results"),
    "hopfield_probe/20260827/probe_ladder7")


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-30)


def softmax(a, axis=-1):
    a = a - a.max(axis=axis, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=axis, keepdims=True)


def run(rule: str, Z: np.ndarray, X: np.ndarray, steps: int, beta: float,
        D: int) -> np.ndarray:
    """`steps` applications of one recurrence to the batch of cues `X`."""
    if rule == "hebb":
        W = (Z.T @ Z) / D
        np.fill_diagonal(W, 0.0)
    elif rule == "proj":
        W = Z.T @ np.linalg.pinv(Z @ Z.T) @ Z
    x = X.copy()
    for _ in range(steps):
        x = unit(x @ Z.T @ Z) if rule == "soft_lin" else (
            unit(softmax(beta * (x @ Z.T)) @ Z) if rule == "soft"
            else unit(x @ W.T))
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--label", default="10% · ")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--beta", type=float, default=30.0,
                    help="softmax sharpness, on cosines in [-1, 1]")
    args = ap.parse_args()

    with open(os.path.join(args.dir, "manifest.json")) as f:
        man = json.load(f)
    h = next(e["header"] for e in man["encoders"]
             if e["label"].startswith(args.label))
    enc, ecfg, gain, fwhm, _x = load_probe_encoder(
        h["path"], fwhm_fallback=h.get("fwhm_ratio", 0.25))
    gain = float(h.get("gain", gain))
    enc.gain = gain
    cfg = ProbeConfig(n_worlds=4, n_envs_per_world=20, env_size=20, Npos=1716,
                      k_values=(args.k,), steps=(1,), seed=0, basin_radius=0)
    field = Field(enc, list(ecfg.lambdas), fwhm, gain, cfg.Npos)
    D = 1024

    self_cos = {r: {s: [] for s in (1, 15)} for r in ("hebb", "proj", "soft")}
    cue_hit = {r: {s: [] for s in (1, 15)} for r in ("hebb", "proj", "soft")}

    cells = local_cells(cfg.env_size)
    for w in sample_worlds(cfg):
        rng = np.random.RandomState(w.seed * 31 + args.k)
        mem = build_memory(field, w, args.k, cfg, rng)
        Z = unit(mem.Z)
        env = scored_envs(cfg, args.k)[0]
        own = int(np.flatnonzero(mem.owner == env)[0])
        off, goal = w.specs[env].offset, np.array(w.specs[env].goal)
        d = np.sqrt(((cells - goal) ** 2).sum(1))
        pick = cells[np.abs(d - args.radius) < 0.75]
        cues = unit(field.encode(pick[:, 0] + off[0], pick[:, 1] + off[1]))

        for rule in ("hebb", "proj", "soft"):
            for s in (1, 15):
                self_cos[rule][s].append(
                    float(np.mean(np.einsum(
                        "id,id->i", run(rule, Z, Z, s, args.beta, D), Z))))
                out = run(rule, Z, cues, s, args.beta, D)
                cue_hit[rule][s].append(
                    float(np.mean((out @ Z.T).argmax(1) == own)))

    print(f"encoder {args.label.strip()}  K={args.k}  cue radius "
          f"{args.radius:g}  softmax beta {args.beta:g}\n")
    print(f"  {'rule':<8s}{'cos(recall(z), z)':>22s}"
          f"{'cue -> own goal':>20s}")
    print(f"  {'':<8s}{'s=1':>11s}{'s=15':>11s}{'s=1':>10s}{'s=15':>10s}")
    for rule, note in (("hebb", "  <- what we run"),
                       ("proj", "  <- projection rule"),
                       ("soft", "  <- modern Hopfield")):
        print(f"  {rule:<8s}"
              f"{np.mean(self_cos[rule][1]):>11.4f}"
              f"{np.mean(self_cos[rule][15]):>11.4f}"
              f"{np.mean(cue_hit[rule][1]):>10.3f}"
              f"{np.mean(cue_hit[rule][15]):>10.3f}{note}")


if __name__ == "__main__":
    main()
