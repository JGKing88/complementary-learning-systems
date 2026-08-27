"""CLI: run the encoder-Hopfield probe suite and write result JSON.

    python -m analysis.hopfield_probe.run --ckpt PATH [--ckpt PATH ...] \
        [--quick] [--rescue] [--out DIR]

One JSON per encoder, plus a manifest. The report layer
(``analysis.hopfield_probe.report.build``) turns those into pages and never
recomputes anything, so restyling a figure costs no recall.

Nothing here defaults ``gain`` or ``fwhm_ratio``: both are read from the
checkpoint and a checkpoint that carries no ``fwhm_ratio`` is an error unless
``--fwhm_override`` states one, which is then recorded in the header.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cls_paths import results_dir

from .attractor import run_test_a
from .controls import run_controls, run_rescue
from .encode import Field
from .flow import run_test_d
from .harness import (
    MEMORY_MODES, ProbeConfig, load_probe_encoder, sample_worlds, write_json,
)
from .qfield import run_tests_bc

QUICK = dict(
    n_worlds=2, n_envs_per_world=6, k_values=(1, 3, 6), steps=(1, 2, 3),
    env_size=8, Npos=132, n_alias=500, n_cont_samples=4000,
    n_cont_annulus=1000, cos_chunk=1024,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Encoder-Hopfield probe (docs/ENCODER_HOPFIELD_EVAL.md)")
    p.add_argument("--ckpt", action="append", required=True,
                   help="encoder checkpoint; repeatable")
    p.add_argument("--label", action="append", default=None,
                   help="display name per --ckpt, in the same order")
    p.add_argument("--out", default=None,
                   help="output directory (default: <results>/hopfield_probe/<stamp>)")
    p.add_argument("--fwhm_fallback", type=float, default=None,
                   help="used ONLY for checkpoints that carry no fwhm_ratio, "
                        "e.g. untrained_mlp.pt. Recorded in the header as an "
                        "override. Cannot mask a stored value.")
    p.add_argument("--fwhm_override", type=float, default=None,
                   help="force this fwhm_ratio even where the checkpoint has "
                        "one. Rarely what you want.")

    p.add_argument("--quick", action="store_true",
                   help="tiny preset for smoke tests")
    p.add_argument("--rescue", action="store_true",
                   help="Sec 3.1a hyperparameter sweep. Off by default: not "
                        "the production operating point, and its numbers are "
                        "not encoder-quality numbers.")
    p.add_argument("--skip", action="append", default=[],
                   choices=["a", "bc", "d", "controls"], help="skip a test")

    g = p.add_argument_group("sweep")
    g.add_argument("--k", type=int, nargs="+", default=None)
    g.add_argument("--steps", type=int, nargs="+", default=None)
    g.add_argument("--n_worlds", type=int, default=None)
    g.add_argument("--n_envs_per_world", type=int, default=None)
    g.add_argument("--env_size", type=int, default=None)
    g.add_argument("--Npos", type=int, default=None)
    g.add_argument("--memory_mode", default=None, choices=list(MEMORY_MODES))
    g.add_argument("--n_alias", type=int, default=None)
    g.add_argument("--n_cont_samples", type=int, default=None)
    g.add_argument("--n_cont_annulus", type=int, default=None)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cpu")

    d = p.add_argument_group("recall dynamics")
    d.add_argument("--beta", type=float, default=None,
                   help="Hopfield beta. Default: the encoder's gain, which is "
                        "what production does (train_navigate.py:488). Per "
                        "coordinate (Wz)_i ~ D^-1.5, so saturating the recall "
                        "needs beta ~ 3e4 at D=1024 -- and that is the "
                        "difference between a linear matched filter whose "
                        "readout decays with steps and a memory that holds.")
    d.add_argument("--hopfield_scale", type=float, default=None,
                   help="storage scale. Default 1/D. Equivalent to --beta by "
                        "(p -> lambda p, beta -> beta/lambda^2); only the "
                        "product beta*scale*D reaches the tanh.")
    d.add_argument("--encoder_gain", type=float, default=None,
                   help="override the encoder's own gain at inference. Shapes "
                        "the PATTERN (how near a hypercube corner), which is a "
                        "different job from --beta. Changes every embedding, "
                        "so a policy trained on the checkpoint's own gain no "
                        "longer applies.")
    return p


def config_from_args(args) -> ProbeConfig:
    kw = dict(QUICK) if args.quick else {}
    for name, val in (
        ("k_values", tuple(args.k) if args.k else None),
        ("steps", tuple(args.steps) if args.steps else None),
        ("n_worlds", args.n_worlds), ("n_envs_per_world", args.n_envs_per_world),
        ("env_size", args.env_size), ("Npos", args.Npos),
        ("memory_mode", args.memory_mode), ("n_alias", args.n_alias),
        ("n_cont_samples", args.n_cont_samples),
        ("n_cont_annulus", args.n_cont_annulus),
    ):
        if val is not None:
            kw[name] = val
    kw["seed"] = args.seed
    kw["device"] = args.device
    if args.beta is not None:
        kw["beta_override"] = args.beta
    if args.hopfield_scale is not None:
        kw["hopfield_scale"] = args.hopfield_scale

    cfg = ProbeConfig(**kw)
    # Sec 2.3: the world is pinned at the largest K so placement is identical
    # across the sweep and only the load moves. Raise it silently rather than
    # failing on a --k the user picked for good reasons.
    if max(cfg.k_values) > cfg.n_envs_per_world:
        cfg = dataclasses.replace(cfg, n_envs_per_world=max(cfg.k_values))
    cfg.validate()
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    out_dir = (Path(args.out) if args.out else
               results_dir(ensure=True) / "hopfield_probe"
               / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.label or []
    manifest = {"config": cfg.to_json(), "encoders": [],
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    t0 = time.time()
    for i, ckpt in enumerate(args.ckpt):
        label = labels[i] if i < len(labels) else Path(ckpt).parent.name
        print(f"[{time.time() - t0:7.1f}s] === {label} ===", flush=True)

        encoder, mcfg, gain, fwhm, header = load_probe_encoder(
            ckpt, device=cfg.device, fwhm_override=args.fwhm_override,
            fwhm_fallback=args.fwhm_fallback)
        header["label"] = label
        if args.encoder_gain is not None:
            encoder.gain = float(args.encoder_gain)
            gain = float(args.encoder_gain)
            header["gain"] = gain
            header["gain_was_overridden"] = True
        # The recall regime belongs in the provenance bar: two runs of the same
        # encoder at different beta are not the same measurement.
        header["beta"] = (float(cfg.beta_override)
                          if cfg.beta_override is not None else float(gain))
        # Saturation threshold: per coordinate (Wz)_i ~ D^-1.5, so the tanh
        # argument is beta * D^-1.5 and it bends around beta ~ D^1.5.
        embed_dim = int(mcfg.out_dim)
        header["recall_regime"] = (
            "saturated" if header["beta"] >= embed_dim ** 1.5 else "linear")

        if list(mcfg.lambdas) != list(getattr(cfg, "lambdas", mcfg.lambdas)):
            pass  # lambdas come from the encoder; nothing to reconcile

        field = Field(encoder=encoder, lambdas=list(mcfg.lambdas),
                      fwhm_ratio=fwhm, gain=gain, Npos=cfg.Npos,
                      device=cfg.device, chunk=cfg.chunk)
        prod = int(np.prod(mcfg.lambdas))
        if cfg.Npos > prod:
            raise SystemExit(
                f"Npos={cfg.Npos} exceeds prod(lambdas)={prod} for {label}: "
                f"two distinct scaffold positions would share an identical "
                f"grid code in every module.")

        worlds = sample_worlds(cfg)

        def progress(msg, _l=label, _t=t0):
            print(f"[{time.time() - _t:7.1f}s] {_l} {msg}", flush=True)

        payload = {"header": header, "config": cfg.to_json()}
        if "a" not in args.skip:
            payload["test_a"] = run_test_a(field, worlds, cfg,
                                           progress=progress)
        if "bc" not in args.skip:
            payload["test_bc"] = run_tests_bc(field, worlds, cfg,
                                              progress=progress)
        if "d" not in args.skip:
            payload["test_d"] = run_test_d(field, worlds, cfg,
                                           progress=progress)
        if "controls" not in args.skip:
            payload["controls"] = run_controls(field, worlds, cfg,
                                               progress=progress)
        if args.rescue:
            payload["rescue"] = run_rescue(field, worlds, cfg,
                                           progress=progress)

        payload["worlds"] = [w.to_json() for w in worlds[:4]]
        path = out_dir / f"{_slug(label)}.json"
        write_json(path, payload)
        manifest["encoders"].append(
            {"label": label, "file": path.name, "header": header})
        print(f"[{time.time() - t0:7.1f}s] wrote {path}", flush=True)

    write_json(out_dir / "manifest.json", manifest)
    print(f"\nresults: {out_dir}")
    print(f"build pages: python -m analysis.hopfield_probe.report.build "
          f"{out_dir}")
    return 0


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


if __name__ == "__main__":
    sys.exit(main())
