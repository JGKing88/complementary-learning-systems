"""Export the dynamics collection for the dynamics viewer (``dynamics.html``).

    python -m analysis.tuning_viewer.export_dynamics --src COLLECT_DIR [--out DIR]

``COLLECT_DIR`` is what ``collect_dynamics`` wrote. ``DIR`` defaults to
``dynamics_data/`` beside this module, which git ignores. Arrays are base64 of
their little-endian bytes in ``.b64.txt`` files (artifacts serve no raw binary
type). Cells are x-major: cell ``x * S + y`` is column x (East), row y (North).
Per arena ``a``:

``ctrl_a{a}``
    uint8 ``(G, C, U, S*S)``: the controller's mean activity per cell, goal
    ``g``, condition ``c`` (explore, exploit), unit ``u``. 255 marks a cell no
    trial visited; 0..254 spans the scale in ``ctrl_a{a}_scale``.
``ctrl_a{a}_scale``
    float32 ``(G, C, U, 2)``: the (lo, hi) each map was quantized against,
    its minimum and maximum over visited cells.
``count_a{a}``
    uint16 ``(G, C, S*S)``: steps binned into each cell.
``ex_a{a}_pos``, ``ex_a{a}_q``, ``ex_a{a}_h``, ``ex_a{a}_hscale``
    example trials, concatenated in the order of the manifest's ``examples``
    list: positions and the readout q (float32, two per step), the RNN state
    (uint8 per unit, 0..255 over 0..hscale[u]; the units are ReLU, so >= 0).
``cue_a{a}``
    float32 ``(P, P, D)``: the encoder output (the Hopfield's current state)
    on cells 0..P-1 of both axes; the local frame at a cell reads its North
    and East neighbours, so P = S + 1.
``dist_a{a}``
    float32 ``(R, 10, D)``: R draws of ten distractor patterns.
``chk_a{a}_q``, ``chk_a{a}_x``
    reference recalls from the repo's own Hopfield and frame: q at every cell
    (float32 ``(K, S*S, 2)``) and the state at the manifest's check cells
    (float32 ``(K, n_cells, D)``), for the configurations in the manifest.
    The collector computed these on the scaffold at the model's settings
    (Hebbian storage, alpha 1).
``chk2_a{a}_q``, ``chk2_a{a}_x``
    the same for ``RULE_CHECKS``: projection and Hebbian storage, alpha below
    1, several steps (the iterative recall of EXPERIMENTS_IDEAL_ENCODER.md
    Sec 18). Computed here on the CPU by ``hopfield.Hopfield`` from the
    exported cue patch and distractors, with the frame from
    ``gram_schmidt_2d_batch`` on the patch's North and East neighbours, which
    is what ``VectorHash.gram_schmidt_projection`` does on the scaffold.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EX_EXPLORE = 3          # example explore trials kept per goal
EX_EXPLOIT = 4          # example exploit trials kept per goal
EX_CAP = 120            # steps kept per example trial
# (goal index or -1, distractors, draw, recall steps, storage rule, alpha)
RULE_CHECKS = ((0, 0, 0, 1, "proj", 0.9), (1, 3, 0, 5, "proj", 0.9), (2, 10, 1, 30, "proj", 0.9),
               (-1, 5, 2, 10, "proj", 0.5), (3, 10, 3, 3, "proj", 1.0), (0, 3, 4, 10, "hebb", 0.9))


def rule_checks(cue_patch, dist, goals, beta, cells):
    import torch
    from hopfield import Hopfield
    from hopfield_nav.utils import gram_schmidt_2d_batch
    torch.set_num_threads(4)
    P, D = cue_patch.shape[0], cue_patch.shape[2]
    S = P - 1
    xs, ys = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    xs, ys = xs.ravel(), ys.ravel()
    cues = np.ascontiguousarray(cue_patch[xs, ys])                         # x-major, (S*S, D)
    frame = gram_schmidt_2d_batch(cue_patch[xs, ys + 1] - cues, cue_patch[xs + 1, ys] - cues)
    qs, xk = [], []
    for g, nd, draw, k, rule, alpha in RULE_CHECKS:
        hop = Hopfield(D, beta=beta, storage_rule=rule)
        pats = ([cue_patch[goals[g][0], goals[g][1]]] if g >= 0 else []) + list(dist[draw, :nd])
        for pat in pats:
            hop.input_memory(torch.from_numpy(np.ascontiguousarray(pat)).float())
        x = hop.recall_batch(torch.from_numpy(cues), steps=k, beta=beta, alpha=alpha).numpy()
        qs.append(np.einsum("bij,bj->bi", frame, x - cues).astype(np.float32))
        xk.append(x[list(cells)].astype(np.float32))
    return np.stack(qs), np.stack(xk)


def write_b64(out: Path, name: str, arr: np.ndarray, files: dict) -> None:
    arr = np.ascontiguousarray(arr)
    assert arr.dtype.byteorder in ("<", "=", "|"), arr.dtype
    txt = base64.b64encode(arr.tobytes()).decode("ascii")
    (out / f"{name}.b64.txt").write_text(txt)
    files[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "bytes": len(txt)}


def quantize_maps(mean: np.ndarray, visited: np.ndarray):
    """mean (..., U, N) with NaN where unvisited; visited (..., N) bool."""
    v = np.where(visited[..., None, :], mean, np.nan)
    lo = np.nanmin(v, axis=-1)
    hi = np.nanmax(v, axis=-1)
    span = np.where(hi > lo, hi - lo, 1.0)
    q = np.rint((v - lo[..., None]) / span[..., None] * 254)
    q = np.where(np.isnan(q), 255, q).astype(np.uint8)
    # Round trip: dequantize and compare on visited cells.
    back = lo[..., None] + q.astype(np.float64) / 254 * span[..., None]
    err = np.nanmax(np.abs(np.where(q == 255, np.nan, back - v)))
    assert err <= 0.5 / 254 * float(span.max()) + 1e-6, err
    return q, np.stack([lo, hi], axis=-1).astype(np.float32), float(err)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=str(HERE / "dynamics_data"))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    validation = json.loads((src / "validation.json").read_text())
    files: dict = {}
    arenas = []
    S = int(meta["env_size"])
    for ai, am in enumerate(meta["arenas"]):
        z = np.load(src / f"arena_{ai}.npz")
        count = z["count"]                                         # (G, C, N)
        G, C, N = count.shape
        assert N == S * S
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = z["act_sum"] / count[..., None]                 # (G, C, N, U)
        mean = np.moveaxis(mean, -1, -2)                           # (G, C, U, N)
        q8, scale, qerr = quantize_maps(mean, count > 0)
        write_b64(out, f"ctrl_a{ai}", q8, files)
        write_b64(out, f"ctrl_a{ai}_scale", scale, files)
        assert count.max() < 65536
        write_b64(out, f"count_a{ai}", count.astype(np.uint16), files)

        # Example trials: the first few of each goal and condition, capped.
        starts = np.concatenate([[0], np.cumsum(z["ex_len"])])
        keep, pos, qs, hs = [], [], [], []
        for i in range(len(z["ex_len"])):
            c, t = int(z["ex_cond"][i]), int(z["ex_trial"][i])
            if t >= (EX_EXPLORE if c == 0 else EX_EXPLOIT):
                continue
            s0 = int(starts[i])
            n = min(int(z["ex_len"][i]), EX_CAP)
            keep.append({"goal": int(z["ex_goal"][i]), "cond": c, "trial": t, "len": n,
                         "full_len": int(z["ex_len"][i]), "reached": bool(z["ex_reached"][i])})
            pos.append(z["ex_pos"][s0:s0 + n])
            qs.append(z["ex_q"][s0:s0 + n])
            hs.append(z["ex_h"][s0:s0 + n].astype(np.float32))
        h = np.concatenate(hs)
        assert h.min() >= 0, h.min()
        hmax = h.max(axis=0)
        h8 = np.rint(h / np.where(hmax > 0, hmax, 1.0) * 255).astype(np.uint8)
        write_b64(out, f"ex_a{ai}_pos", np.concatenate(pos).astype(np.float32), files)
        write_b64(out, f"ex_a{ai}_q", np.concatenate(qs).astype(np.float32), files)
        write_b64(out, f"ex_a{ai}_h", h8, files)
        write_b64(out, f"ex_a{ai}_hscale", hmax.astype(np.float32), files)

        write_b64(out, f"cue_a{ai}", z["cue_patch"].astype(np.float32), files)
        write_b64(out, f"dist_a{ai}", z["distractors"].astype(np.float32), files)
        write_b64(out, f"chk_a{ai}_q", z["check_q"].astype(np.float32), files)
        write_b64(out, f"chk_a{ai}_x", z["check_x"].astype(np.float32), files)
        q2, x2 = rule_checks(z["cue_patch"], z["distractors"], meta["goals"], float(meta["hopfield"]["beta"]),
                             z["check_cells"].tolist())
        write_b64(out, f"chk2_a{ai}_q", q2, files)
        write_b64(out, f"chk2_a{ai}_x", x2, files)

        # Behaviour per goal and condition, from every trial.
        beh = [[None] * C for _ in range(G)]
        for g in range(G):
            for c, cond in enumerate(meta["conditions"]):
                tr = [t for t in am["trials"] if t["goal"] == g and t["cond"] == cond]
                ok = [t["steps"] for t in tr if t["reached"]]
                beh[g][c] = {"trials": len(tr), "success": len(ok) / max(len(tr), 1),
                             "mean_steps": float(np.mean(ok)) if ok else None,
                             "steps_binned": int(count[g, c].sum()),
                             "cells_visited": int((count[g, c] > 0).sum())}
        arenas.append({k: am[k] for k in ("name", "kind", "index", "offset", "own_goal", "wall_seed", "q_err")}
                      | {"wall_code": z["wall_code"].tolist(), "examples": keep, "behaviour": beh,
                         "quant_err": qerr,
                         "check_configs": z["check_configs"].tolist(),
                         "check2_configs": [list(c) for c in RULE_CHECKS],
                         "check_cells": z["check_cells"].tolist()})
        print(f"[export] {am['name']}: quantization error {qerr:.2e}, "
              f"{len(keep)} example trials, {int(count.sum())} steps", flush=True)

    manifest = {
        "model": {"ckpt": meta["ckpt"], "encoder": meta["encoder"], "hidden_size": meta["hidden_size"]},
        "env_size": S, "goal_radius": meta["goal_radius"], "goals": meta["goals"],
        "conditions": meta["conditions"], "hopfield": meta["hopfield"],
        "n_draws": meta["n_draws"], "d_max": meta["d_max"], "patch": meta["patch"],
        "wall_resolution": meta["wall_resolution"], "validation": validation,
        "collect": {"git": meta["git"], "args": meta["args"], "seconds": meta["seconds"],
                    "torch": meta["torch"]},
        "arenas": arenas, "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    total = sum(f["bytes"] for f in files.values())
    print(f"[export] wrote {len(files)} files, {total / 1e6:.1f} MB, to {out}", flush=True)


if __name__ == "__main__":
    main()
