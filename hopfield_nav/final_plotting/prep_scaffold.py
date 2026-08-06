"""Build VectorHash + encoded_Phi from a phased ckpt and write to disk.

Designed to run once before fanning out into N parallel ``agenthash`` subruns
that all share the same scaffold. Each subrun then loads ``encoded_Phi`` via
``--scaffold_cache <dir>``; with ``--mmap`` it's read via the OS page cache so
all subruns share one set of physical pages instead of each holding a private
copy.

Two output modes:
  --out <dir>          Explicit output directory.
  --cache_root <dir>   Content-addressed cache. Resolves to <cache_root>/<hash>/
                       where hash = sha256(lambdas, fwhm_ratio, Npos,
                       encoder_path)[:16]. If that dir already contains a
                       valid encoded_Phi.npy + meta.json with matching params,
                       skip the build (cache hit). Otherwise build and save.

All logging goes to stderr. stdout emits exactly one line: the resolved output
directory, so a bash driver can capture the path via $(...).

Output layout (under the resolved dir):
    encoded_Phi.npy   (Npos, Npos, embed_dim) float32 — the only big array
    meta.json         lambdas / Npos / Ng / fwhm_ratio / embed_dim / encoder_path

``--static_vectorhash`` is required at the agenthash side for the cached path
because we don't persist the raw ``gbook`` (it's intermediate to encoded_Phi
and only used by the non-static register_envs path).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys

import numpy as np
import torch

from ..encoder import load_encoder
from ..evaluation.checkpoint_io import cfg_from_checkpoint
from ..vectorhash import VectorHash


def _log(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


def _scaffold_hash(lambdas: list[int], fwhm_ratio: float, Npos: int,
                   encoder_path: str) -> str:
    """sha256-derived 16-char tag uniquely keying scaffold cache contents."""
    payload = json.dumps(
        {
            "lambdas": [int(x) for x in lambdas],
            "fwhm_ratio": float(fwhm_ratio),
            "Npos": int(Npos),
            "encoder_path": str(encoder_path),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_is_valid(cache_dir: str, lambdas, fwhm_ratio, Npos,
                    encoder_path) -> bool:
    enc_path = os.path.join(cache_dir, "encoded_Phi.npy")
    meta_path = os.path.join(cache_dir, "meta.json")
    if not (os.path.exists(enc_path) and os.path.exists(meta_path)):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return False
    return (
        list(meta.get("lambdas", [])) == [int(x) for x in lambdas]
        and abs(float(meta.get("fwhm_ratio", -1)) - float(fwhm_ratio)) < 1e-6
        and int(meta.get("Npos", -1)) == int(Npos)
        and str(meta.get("encoder_path", "")) == str(encoder_path)
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True,
                   help="Phased checkpoint (final.pt from train_phased*.py).")
    p.add_argument("--out", default=None,
                   help="Explicit output directory. Mutually exclusive with --cache_root.")
    p.add_argument("--cache_root", default=None,
                   help="Content-addressed cache root. Resolves to "
                        "<cache_root>/<hash>/. Skips the build on cache hit.")
    p.add_argument("--encoder_override", default=None,
                   help="Override cfg.encoder_checkpoint from the ckpt.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--static_vectorhash", action="store_true",
                   help="Set cfg.vectorhash.static_vectorhash = True before "
                        "building. Should match the agenthash invocation.")
    args = p.parse_args()

    if (args.out is None) == (args.cache_root is None):
        p.error("exactly one of --out / --cache_root must be set")

    # Route ALL stdout (including bare print()s in vectorhash, load_encoder,
    # etc.) to stderr so bash $(...) capture sees only the final path line.
    out_dir: str
    with contextlib.redirect_stdout(sys.stderr):
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        _log(f"[prep_scaffold] loading {args.ckpt}")
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = cfg_from_checkpoint(ck["config"])
        if args.static_vectorhash:
            cfg.vectorhash.static_vectorhash = True

        encoder_path = args.encoder_override or cfg.encoder_checkpoint
        lambdas = list(cfg.vectorhash.lambdas)
        Npos = (cfg.vectorhash.Npos
                if cfg.vectorhash.Npos is not None
                else int(np.prod(lambdas)))
        fwhm_ratio = float(cfg.fwhm_ratio)

        # Resolve output directory.
        if args.cache_root is not None:
            h = _scaffold_hash(lambdas, fwhm_ratio, Npos, encoder_path)
            out_dir = os.path.join(args.cache_root, h)
            _log(f"[prep_scaffold] cache_root resolved hash={h} → {out_dir}")
            os.makedirs(args.cache_root, exist_ok=True)
            if _cache_is_valid(out_dir, lambdas, fwhm_ratio, Npos, encoder_path):
                _log(f"[prep_scaffold] CACHE HIT at {out_dir} — skipping build")
                # Fall through to the final stdout print below.
                print(out_dir, file=sys.__stdout__)
                return
            _log(f"[prep_scaffold] cache miss — building")
        else:
            out_dir = args.out

        # Build.
        encoder, enc_cfg, enc_gain = load_encoder(encoder_path, str(device))
        embed_dim = int(enc_cfg.out_dim)
        if cfg.hopfield.beta is None:
            cfg.hopfield.beta = float(enc_gain)

        torch.manual_seed(0)
        np.random.seed(0)
        vh = VectorHash(cfg.vectorhash, size=cfg.env.size)
        _log(f"[prep_scaffold] lambdas={vh.lambdas}  Ng={vh.Ng}  Npos={vh.Npos}")
        vh.build_scaffold()
        vh.precompute_encoded_phi(encoder, fwhm_ratio, device=str(device))

        os.makedirs(out_dir, exist_ok=True)
        enc_path = os.path.join(out_dir, "encoded_Phi.npy")
        meta_path = os.path.join(out_dir, "meta.json")
        np.save(enc_path, vh.encoded_Phi)
        with open(meta_path, "w") as f:
            json.dump({
                "lambdas": list(vh.lambdas),
                "Npos": int(vh.Npos),
                "Ng": int(vh.Ng),
                "fwhm_ratio": fwhm_ratio,
                "embed_dim": embed_dim,
                "encoder_path": str(encoder_path),
                "ckpt_path": str(args.ckpt),
                "static_vectorhash": bool(cfg.vectorhash.static_vectorhash),
            }, f, indent=2)
        _log(f"[prep_scaffold] wrote {out_dir}/")
        _log(f"  encoded_Phi.npy: {vh.encoded_Phi.shape} {vh.encoded_Phi.dtype} "
             f"({os.path.getsize(enc_path) / 1e9:.2f} GB)")
        _log(f"  meta.json:       {meta_path}")

    # Single clean line to real stdout: the resolved path.
    print(out_dir)


if __name__ == "__main__":
    main()
