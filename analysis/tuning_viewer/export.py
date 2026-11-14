"""Export the six place-code populations for the tuning-map viewer.

    python -m analysis.tuning_viewer.export [--out DIR]

The viewer (``viewer.html``) computes every map in the browser, for any arena
size at any scaffold location, from what this script writes to ``DIR``
(default ``viewer_data/`` beside this module, which git ignores):

``manifest.json``
    every population's parameters, the reference arena, the trained
    encoder's training patches, and the result of each check below.
``<pop>_*.b64.txt``
    arrays as base64 of their little-endian bytes (artifacts serve no raw
    binary type): the folded MLP weights (float32), the hippocampal ``Wpg``
    (float64), and one reference arena per population (float32,
    ``(S*S, n_units)``, cell ``n = row*S + col`` at scaffold
    ``((x0+col) % Npos, (y0+row) % Npos)``), computed by the repo's own code.

The populations, and how the page computes one unit at one cell:

grid
    ``hopfield_probe.encode.grid_codes``: module ``l``, phase ``(i, j)``,
    ``exp(-((wrap(i - x mod l))^2 + (wrap(j - y mod l))^2) / 2 sigma^2)``.
ideal
    ``IdealEncoder`` (``ideal:r=16,n_freq=512,seed=0``): units ``0..511`` are
    ``cos``, ``512..1023`` ``sin`` of ``2 pi ((n . p) mod Npos) / Npos``,
    times ``amp``.
ideal_rot
    the ideal code remixed by one fixed orthogonal matrix
    ``Q = H D3 H D2 H D1 / 1024^1.5`` (``H`` the Sylvester Walsh-Hadamard
    matrix, ``D_t`` diagonal random signs, stored in the manifest). Unit ``u``
    is ``(Q z_ideal)_u``. Because ``Q`` is orthogonal, the similarity between
    any two places equals the ideal's; only the single-unit maps differ. The
    page applies ``Q`` with a fast Walsh-Hadamard transform.
trained, untrained
    the MLP with its first layer folded into per-module tables (see
    ``export_bench``). The untrained encoder is the cosyne figure's control:
    ``create_encoder(cfg)`` after ``torch.manual_seed(0)``, same gain.
hippocampus
    the Vector-HaSH place layer ``relu(Wpg g - thresh)`` on the one-hot grid
    code, with the headline controller's settings (Np 400, thresh 2.0,
    c 0.5). That controller ran with ``static_vectorhash=True`` and never
    built this layer, so ``Wpg`` is a fresh draw: ``build_scaffold``'s recipe
    after ``np.random.seed(0)``, the cosyne evaluation's seed.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from analysis.hopfield_probe.encode import encode_positions, grid_codes
from analysis.hopfield_probe.harness import load_probe_encoder
from analysis.tuning_viewer.export_bench import arena_xy, erf_poly, folded_forward
from encoder_training.models import create_encoder
from gridcode.assoc import nonlin

HERE = Path(__file__).resolve().parent
RUNS = Path("/orcd/pool/003/jackking/cls_runs")
ENCODER = str(RUNS / "sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt")
IDEAL = "ideal:r=16,n_freq=512,seed=0,gain=100"
HPC = {"Np": 400, "thresh": 2.0, "c": 0.5, "seed": 0}
ROT = {"seed": 0, "rounds": 3}
# Reference arena: crosses both scaffold edges, so the check covers the
# page's modular indexing as well as its arithmetic.
REF = {"x0": 1708, "y0": 1710, "S": 16}
N_CHECK = 4096


def write_b64(path: Path, arr: np.ndarray, dtype: str) -> None:
    raw = np.ascontiguousarray(arr, dtype=dtype).tobytes()
    path.write_text(base64.b64encode(raw).decode("ascii"))


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------- MLP
def fold_mlp(encoder, lambdas, fwhm):
    """First layer folded into per-module tables; later layers as [in][out]."""
    linears = [m for m in encoder.net if isinstance(m, torch.nn.Linear)]
    acts = [m for m in encoder.net if isinstance(m, torch.nn.GELU)]
    if len(acts) != len(linears) - 1 or any(a.approximate != "none" for a in acts):
        raise ValueError("expected exact GELU between every pair of Linear layers")
    if encoder.output_nonlinearity != "tanh":
        raise ValueError(f"expected tanh output, got {encoder.output_nonlinearity}")
    Wb = [(m.weight.detach().double().numpy(), m.bias.detach().double().numpy())
          for m in linears]
    W1, b1 = Wb[0]
    parts, row_offsets, col = [], [], 0
    for l in lambdas:
        px, py = np.meshgrid(np.arange(l), np.arange(l), indexing="ij")
        block = grid_codes(lambdas, px.ravel(), py.ravel(), fwhm).astype(np.float64)
        row_offsets.append(sum(p.shape[0] for p in parts))
        parts.append(block[:, col:col + l * l] @ W1[:, col:col + l * l].T)
        col += l * l
    T = np.concatenate(parts, axis=0)
    layers = [(W.T.copy(), b) for W, b in Wb[1:]]
    return T, b1, layers, row_offsets


def export_mlp(name, encoder, gain, lambdas, fwhm, npos, out: Path, label, description):
    T, b1, layers, row_offsets = fold_mlp(encoder, lambdas, fwhm)
    rng = np.random.default_rng(0)
    xs, ys = rng.integers(0, npos, N_CHECK), rng.integers(0, npos, N_CHECK)
    truth = encode_positions(encoder, lambdas, fwhm, xs, ys).astype(np.float64)
    got = folded_forward(xs, ys, (T, b1), layers, lambdas, row_offsets, float(gain), erf_poly)
    diff = float(np.abs(got - truth).max())
    if diff > 1e-4:
        raise RuntimeError(f"{name}: folded forward differs from the encoder by {diff:.2e}")

    arrays = [("T", T), ("b1", b1)]
    for i, (W, b) in enumerate(layers, start=2):
        arrays += [(f"W{i}", W), (f"b{i}", b)]
    meta, offset, flat = [], 0, []
    for an, a in arrays:
        meta.append({"name": an, "shape": list(a.shape), "offset": offset})
        offset += a.size
        flat.append(np.asarray(a, dtype=np.float64).ravel())
    write_b64(out / f"{name}_weights.b64.txt", np.concatenate(flat), "<f4")
    return {
        "kind": "mlp", "label": label, "description": description,
        "n_units": int(layers[-1][0].shape[1]), "signed": True,
        "gain": float(gain), "hidden": int(T.shape[1]), "row_offsets": row_offsets,
        "weights_file": f"{name}_weights.b64.txt", "arrays": meta,
        "fold_check_max_abs_diff": diff,
    }


# ---------------------------------------------------------------- rotated ideal
def hadamard_rotation(signs: np.ndarray) -> np.ndarray:
    """``Q = H D_k ... H D_1 / D^(k/2)``, ``H`` Sylvester (natural order), ``D_t = diag(signs[t])``.

    This is the matrix the page's in-place fast Walsh-Hadamard transform applies.
    """
    D = signs.shape[1]
    H = np.array([[1.0]])
    while H.shape[0] < D:
        H = np.block([[H, H], [H, -H]])
    Q = np.eye(D)
    for s in signs:
        Q = H @ (s[:, None] * Q)
    return Q / D ** (len(signs) / 2)


# ---------------------------------------------------------------- hippocampus
def draw_wpg(Np: int, Ng: int, c: float, seed: int) -> np.ndarray:
    """``VectorHash.build_scaffold``'s Wpg draw, after ``np.random.seed(seed)``."""
    np.random.seed(seed)
    W = np.random.randn(Np, Ng)
    prune = int((1 - c) * Np * Ng)
    mask = np.ones((Np, Ng))
    mask[np.random.randint(0, Np, prune), np.random.randint(0, Ng, prune)] = 0
    return mask * W


def hpc_rates(Wpg, lambdas, thresh, xs, ys) -> np.ndarray:
    """Place-layer activity at scaffold cells: ``nonlin(Wpg g, thresh)``, g one-hot."""
    g = grid_codes(lambdas, xs, ys, 0.0).astype(np.float64)        # one-hot
    return nonlin(g @ Wpg.T, thresh=thresh)


def check_hpc_recipe() -> dict:
    """Recipe against ``VectorHash.build_scaffold`` on a small scaffold, exactly."""
    from hopfield_nav.config import VectorHashConfig
    from hopfield_nav.world.scaffold import VectorHash

    lam, Np, thresh, c, seed = [3, 4, 5], 50, 2.0, 0.5, 0
    cfg = VectorHashConfig(lambdas=lam, Np=Np, thresh=thresh, c=c, static_vectorhash=False)
    vh = VectorHash(cfg)
    np.random.seed(seed)
    vh.build_scaffold()
    W = draw_wpg(Np, vh.Ng, c, seed)
    gx, gy = np.meshgrid(np.arange(vh.Npos), np.arange(vh.Npos), indexing="ij")
    rates = hpc_rates(W, lam, thresh, gx.ravel(), gy.ravel())       # (Npos^2, Np)
    pbook = vh.pbook.reshape(Np, -1).T
    return {"lambdas": lam, "Np": Np, "Npos": vh.Npos,
            "wpg_max_abs_diff": float(np.abs(W - vh.Wpg).max()),
            "pbook_max_abs_diff": float(np.abs(rates - pbook).max())}


# ---------------------------------------------------------------- training patches
def encoder_patches(path: str, lambdas, fwhm) -> list[dict]:
    """The trained encoder's training patches, in the viewer's (x, y).

    ``encoder_training`` names a patch ``(y0, x0)`` but passes ``y0`` as the
    first argument of ``grid_code_batch`` (``data.py``), which indexes the
    codebook's first spatial axis: the axis ``grid_codes`` and this viewer
    call x. So viewer ``x0 = y0s[i]`` and ``y0 = x0s[i]``. The assert pins
    that the two code functions agree on which argument is which.
    """
    from encoder_training.eval_unique_radius import grid_code_batch

    a, b = np.array([3]), np.array([7])
    if not np.allclose(grid_code_batch(lambdas, a, b, fwhm), grid_codes(lambdas, a, b, fwhm)):
        raise RuntimeError("grid_code_batch and grid_codes disagree on axis order")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ys, xs, sizes = ck["y0s"], ck["x0s"], ck["sizes"]
    return [{"group": "Trained encoder's training patches", "label": f"Patch {i + 1}",
             "x0": int(ys[i]), "y0": int(xs[i]), "S": int(sizes[i])}
            for i in range(len(sizes))]


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=HERE / "viewer_data")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    trained, cfg, gain, fwhm, header = load_probe_encoder(ENCODER)
    trained.eval()
    lambdas = [int(l) for l in cfg.lambdas]
    npos = int(np.prod(lambdas))
    ng = sum(l * l for l in lambdas)
    xs_r, ys_r = arena_xy(REF["x0"], REF["y0"], REF["S"], npos)
    pops, refs = {}, {}

    # grid
    pops["grid"] = {
        "kind": "grid", "label": "Grid code",
        "description": f"Smoothed grid code, modules of period {', '.join(map(str, lambdas))} "
                       f"cells, bump FWHM {fwhm:g} of the period. One unit per module phase.",
        "n_units": ng, "signed": False,
    }
    refs["grid"] = grid_codes(lambdas, xs_r, ys_r, fwhm)

    # ideal
    ideal, _, igain, ifwhm, iheader = load_probe_encoder(IDEAL)
    freqs = ideal.freqs.cpu().numpy().astype(int)
    amp = ideal.amp.cpu().numpy()
    if not np.allclose(amp, amp[0]):
        raise ValueError("viewer assumes equal ideal-encoder weights")
    pops["ideal"] = {
        "kind": "ideal", "label": "Ideal encoder",
        "description": f"Closed-form code whose similarity is a Gaussian of width r = "
                       f"{ideal.r:g} cells: {len(freqs)} plane waves, each as a cos unit "
                       f"and a sin unit.",
        "n_units": 2 * len(freqs), "signed": True, "spec": IDEAL,
        "freqs": freqs.tolist(), "amp": float(amp[0]), "r": float(ideal.r),
    }
    refs["ideal"] = encode_positions(ideal, lambdas, ifwhm, xs_r, ys_r)
    cf = ideal.closed_form(xs_r, ys_r)
    pops["ideal"]["closed_form_max_abs_diff"] = float(np.abs(cf - refs["ideal"]).max())

    # ideal, rotated
    D = 2 * len(freqs)
    signs = np.random.default_rng(ROT["seed"]).choice([-1, 1], size=(ROT["rounds"], D))
    Q = hadamard_rotation(signs)
    refs["ideal_rot"] = (refs["ideal"].astype(np.float64) @ Q.T).astype(np.float32)
    zi, zr = refs["ideal"].astype(np.float64), refs["ideal_rot"].astype(np.float64)
    pops["ideal_rot"] = {
        "kind": "ideal_rot", "label": "Ideal, rotated",
        "description": f"The ideal encoder's {D} units remixed by one fixed random rotation "
                       f"({ROT['rounds']} rounds of random signs and a Walsh-Hadamard transform). "
                       f"The rotation is orthogonal, so the similarity between any two places is "
                       f"exactly the ideal's. Only the single-unit maps differ.",
        "n_units": D, "signed": True, "seed": ROT["seed"], "rounds": ROT["rounds"],
        "signs": signs.astype(int).tolist(),
        "orthogonality_max_abs_diff": float(np.abs(Q @ Q.T - np.eye(D)).max()),
        "kernel_max_abs_diff": float(np.abs(zr @ zr.T - zi @ zi.T).max()),
    }

    # trained, untrained
    pops["trained"] = export_mlp(
        "trained", trained, gain, lambdas, fwhm, npos, out, "Trained encoder",
        "Embedding net from the cosyne figures (att0.5, seed 43): 434 inputs, four "
        "256-unit GELU layers, 1024 outputs, tanh(100 z), unit norm.")
    pops["trained"]["encoder"] = ENCODER
    refs["trained"] = encode_positions(trained, lambdas, fwhm, xs_r, ys_r)

    torch.manual_seed(0)
    untrained = create_encoder(cfg, "cpu").eval()
    pops["untrained"] = export_mlp(
        "untrained", untrained, float(untrained.gain), lambdas, fwhm, npos, out,
        "Untrained encoder",
        "Same architecture and gain as the trained encoder, with its random initial "
        "weights (torch seed 0), the control in the cosyne encoder figure.")
    refs["untrained"] = encode_positions(untrained, lambdas, fwhm, xs_r, ys_r)

    # hippocampus
    hpc_check = check_hpc_recipe()
    if hpc_check["wpg_max_abs_diff"] != 0 or hpc_check["pbook_max_abs_diff"] > 1e-12:
        raise RuntimeError(f"hippocampus recipe disagrees with build_scaffold: {hpc_check}")
    Wpg = draw_wpg(HPC["Np"], ng, HPC["c"], HPC["seed"])
    write_b64(out / "hippocampus_wpg.b64.txt", Wpg, "<f8")
    pops["hippocampus"] = {
        "kind": "hpc", "label": "Hippocampus",
        "description": f"Vector-HaSH place layer with the headline controller's settings "
                       f"({HPC['Np']} cells, threshold {HPC['thresh']:g}, c {HPC['c']:g}) on "
                       f"the one-hot grid code. A seed-{HPC['seed']} draw: the cosyne "
                       f"controller ran without this layer.",
        "n_units": HPC["Np"], "signed": False, "thresh": HPC["thresh"],
        "c": HPC["c"], "seed": HPC["seed"], "wpg_file": "hippocampus_wpg.b64.txt",
        "recipe_check": hpc_check,
        "zeroed_fraction": float((Wpg == 0).mean()),
    }
    refs["hippocampus"] = hpc_rates(Wpg, lambdas, HPC["thresh"], xs_r, ys_r)

    for name, ref in refs.items():
        fname = f"{name}_ref.b64.txt"
        write_b64(out / fname, ref, "<f4")
        pops[name]["ref_file"] = fname
        assert ref.shape == (REF["S"] ** 2, pops[name]["n_units"]), (name, ref.shape)

    arenas = encoder_patches(ENCODER, lambdas, fwhm)
    manifest = {
        "git_sha": git_sha(), "lambdas": lambdas, "npos": npos, "fwhm_ratio": float(fwhm),
        "ref_arena": REF, "populations": pops, "arenas": arenas,
        "encoder_header": header,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    total = sum(f.stat().st_size for f in out.iterdir())
    for name, pop in pops.items():
        checks = {k: v for k, v in pop.items() if "diff" in k or k == "recipe_check"}
        print(f"  {name:12s} {pop['n_units']:5d} units  {checks}")
    cover = sum(a["S"] ** 2 for a in arenas) / npos ** 2
    print(f"  {len(arenas)} encoder training patches ({cover:.1%} of the scaffold); "
          f"wrote {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
