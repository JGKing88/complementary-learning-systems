"""Direction readouts of an encoder + Hopfield memory, on fixed held-out pairs (plan sec 6.4).

    python -m analysis.encoder_readouts --size 50 --max_abs 19,49 --ckpts A.pt,B.pt --out r.json

For each pair (p, g) in the held-out arenas, the goal's embedding is stored in a
one-pattern Hopfield memory exactly as the agent does (`hopfield.core.Hopfield.
input_memory`: W = (1/D) g g^T of the normalised pattern, zero diagonal), and the
direction to the goal is read out five ways:

  frame_goal    W_frame (z_g - z_p)                      what train_encoder_walk scored
  frame_recall  W_frame (recall(z_p) - z_p)              the agent's signal (rollout/signal.py):
                recall = normalize(tanh(beta W z_p)), one step, alpha 1, beta = encoder gain
  grad4         central-difference gradient over the 4 neighbours of the memory's
                negative energy  -E(c) = 1/2 c^T W c  (edges one-sided)
  grad8         the same with Sobel weights over the 8-neighbourhood
  argmax4/8     the offset to the neighbour of lowest energy (quantised; `floor4/8` is
                the error of a perfect argmax on the same pairs)

W_frame is the Gram-Schmidt frame of the encoded +x / +y neighbours at p, as the
harness builds it. The same pairs (fixed seed) are used for every checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from encoder_training.config import EncoderModelConfig
from encoder_training.models import create_encoder
from hopfield_nav.train_encoder_walk import angular_error, build_world, local_frames, sample_pairs

OFF4 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
OFF8 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]])


def load_encoder(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mc = ck["model_config"]
    enc = create_encoder(EncoderModelConfig(**{k: v for k, v in mc.items()
                                               if k in EncoderModelConfig.__dataclass_fields__}), "cpu")
    enc.load_state_dict(ck["state_dict"])
    enc.eval()
    gain = float(ck.get("gain", mc.get("gain", 1.0)))
    return enc, gain, ck.get("env_steps"), ck.get("update")


@torch.no_grad()
def encode(enc, gbook, gain) -> np.ndarray:
    return enc(torch.from_numpy(np.asarray(gbook, dtype=np.float32)), gain).numpy().astype(np.float64)


def neg_energy(Z: np.ndarray, G: np.ndarray) -> np.ndarray:
    """-E(c) for every cell c (rows of Z) under each goal's one-pattern memory (rows of G):
    1/2 c^T W c with W = (1/D)(g g^T - diag(g^2)), g normalised. Returns (cells, goals)."""
    D = Z.shape[1]
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    ov = Z @ Gn.T                                   # (cells, goals)  g . c
    diag = (Z * Z) @ (Gn * Gn).T                    # sum_i g_i^2 c_i^2
    return 0.5 / D * (ov * ov - diag)


def recall(Zp: np.ndarray, G: np.ndarray, beta: float) -> np.ndarray:
    """One recall step per pair, the agent's settings: normalize(tanh(beta W x))."""
    D = Zp.shape[1]
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    h = ((Zp * Gn).sum(1, keepdims=True) * Gn - Gn * Gn * Zp) / D      # W x, zero diagonal
    x = np.tanh(beta * h)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-300)


def gradient_dirs(S_field: np.ndarray, S: int, p: np.ndarray, sobel: bool) -> np.ndarray:
    """Finite-difference gradient of per-pair fields S_field (S*S, n) at cells p (n, 2)."""
    n = len(p)
    F = S_field.T.reshape(n, S, S)                  # F[i, x, y]
    x, y = p[:, 0], p[:, 1]
    xp, xm = np.minimum(x + 1, S - 1), np.maximum(x - 1, 0)
    yp, ym = np.minimum(y + 1, S - 1), np.maximum(y - 1, 0)
    i = np.arange(n)
    if not sobel:
        dx = (F[i, xp, y] - F[i, xm, y]) / (xp - xm)
        dy = (F[i, x, yp] - F[i, x, ym]) / (yp - ym)
    else:
        dx = dy = 0.0
        for w, (yy) in ((1, ym), (2, y), (1, yp)):
            dx = dx + w * (F[i, xp, yy] - F[i, xm, yy]) / (xp - xm)
        for w, (xx) in ((1, xm), (2, x), (1, xp)):
            dy = dy + w * (F[i, xx, yp] - F[i, xx, ym]) / (yp - ym)
    return np.stack([dx, dy], 1)


def argmax_dirs(S_field: np.ndarray, S: int, p: np.ndarray, offs: np.ndarray) -> np.ndarray:
    n = len(p)
    F = S_field.T.reshape(n, S, S)
    best = np.full(n, -np.inf)
    out = np.zeros((n, 2))
    for o in offs:
        q = p + o
        ok = (q[:, 0] >= 0) & (q[:, 0] < S) & (q[:, 1] >= 0) & (q[:, 1] < S)
        v = np.full(n, -np.inf)
        v[ok] = F[np.arange(n)[ok], q[ok, 0], q[ok, 1]]
        better = v > best
        best[better] = v[better]
        out[better] = o
    return out


def floor_err(d: np.ndarray, offs: np.ndarray) -> np.ndarray:
    """Error of a perfect argmax: angle from each true direction to the nearest offset."""
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    o = offs / np.linalg.norm(offs, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((u @ o.T).max(1), -1, 1)))


def score(enc, gain, heldout, pairs) -> dict:
    """Mean angular error per readout per range, over the fixed pairs of every held-out env."""
    out = {}
    for rng_key, per_env in pairs.items():
        errs = {k: [] for k in ("frame_goal", "frame_recall", "grad4", "grad8", "argmax4", "argmax8",
                                "floor4", "floor8")}
        for t, (p, g) in zip(heldout.tensors, per_env):
            S = t.size
            Z = encode(enc, t.gbook, gain)
            zp, zg = Z[p[:, 0] * S + p[:, 1]], Z[g[:, 0] * S + g[:, 1]]
            d = (g - p).astype(float)
            W = local_frames(Z, S, p)
            errs["frame_goal"].append(angular_error(np.einsum("bij,bj->bi", W, zg - zp), d))
            rec = recall(zp, zg, gain)
            errs["frame_recall"].append(angular_error(np.einsum("bij,bj->bi", W, rec - zp), d))
            NE = neg_energy(Z, zg)                          # (S*S, pairs)
            errs["grad4"].append(angular_error(gradient_dirs(NE, S, p, False), d))
            errs["grad8"].append(angular_error(gradient_dirs(NE, S, p, True), d))
            errs["argmax4"].append(angular_error(argmax_dirs(NE, S, p, OFF4), d))
            errs["argmax8"].append(angular_error(argmax_dirs(NE, S, p, OFF8), d))
            errs["floor4"].append(floor_err(d, OFF4))
            errs["floor8"].append(floor_err(d, OFF8))
        out[rng_key] = {k: float(np.concatenate(v).mean()) for k, v in errs.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--max_abs", default="19")
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    ap.add_argument("--pairs", type=int, default=1024, help="pairs per held-out env per range")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    class W:                                            # build_world's args, the runs' own world
        size, observation_size, n_envs, n_val_envs, n_same_envs = a.size, 120, 64, 16, 4
        lambdas, fwhm_ratio, place_margin, place_region = [11, 12, 13], 0.25, 20, "anywhere"
        wall_seeds, lr, n_updates, eval_every, seed, device = "0,10000000", 3e-4, 4000, 50, a.seed, "cpu"

    t0 = time.time()
    heldout = build_world(W, np.random.RandomState(a.seed))[2]
    print(f"world size {a.size}: {len(heldout)} held-out envs; {time.time()-t0:.0f}s", flush=True)
    rng = np.random.RandomState(12345)
    pairs = {r: [sample_pairs(rng, heldout.tensors[0].size, a.pairs, int(r)) for _ in heldout.tensors]
             for r in a.max_abs.split(",")}
    results = {}
    for path in a.ckpts.split(","):
        enc, gain, steps, upd = load_encoder(path)
        r = score(enc, gain, heldout, pairs)
        results[path] = {"gain": gain, "env_steps": steps, "update": upd, "readouts": r}
        for rk, v in r.items():
            print(f"{os.path.basename(os.path.dirname(path))[:34]:34s} {os.path.basename(path):18s} within {rk:>2s}: "
                  + "  ".join(f"{k} {v[k]:5.1f}" for k in ("frame_goal", "frame_recall", "grad4", "grad8",
                                                         "argmax4", "argmax8", "floor4", "floor8")), flush=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
