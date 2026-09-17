"""Phase 1, encoder side: Agent-HaSH's encoder objective on the decode's walks (plan sec 6.4).

    python -m hopfield_nav.train_encoder_walk --labels odometry --n_envs 64 ...
    python -m hopfield_nav.train_encoder_walk --eval_only PATH/encoder_final.pt

The same walkers, buffer, arenas and env-step count as `train_decode_walk`
(imported from it), and the encoder package's own model and loss with the
att0.5 checkpoint's configuration: a 4x256 GELU MLP to 1024-d, tanh
output, gain annealed 1 -> 100, `mse_attract_repel` (attract 0.5, repel 1)
plus 0.5 x `coding_rate_loss`. What differs from its original training is
only where the positions and the labels come from:

  batch    `batches_per_update` batches of `batch_envs` envs x all walkers x
           `per_walker` moments of each walker's history (4096 positions).
  labels   near = distance < `radius`, far otherwise, over
             odometry: pairs of moments of the SAME WALKER -- the displacement
                       is the walker's own recorded motion; every other pair
                       is left out of the loss (the walker cannot relate
                       another walker's moments to its own);
             coords:   pairs of positions in the SAME ENV, from their true
                       coordinates -- the encoder's own label source; cross-env
                       pairs left out (its `exclude_cross_env_pairs`).
  eval     the harness's readout on the decode's test set: every cell of a
           held-out env encoded, the local frame W at each cell from its
           scaffold neighbours (`gram_schmidt_2d_batch`, as `scaffold.py`),
           q = W (z(g) - z(p)) and the angular error against g - p over pairs
           within `max_abs` -- the metric the decode is scored on.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from cls_paths import run_dir, run_name
import run_manifest
from encoder_training.config import EncoderModelConfig
from encoder_training.losses import coding_rate_loss, mse_attract_repel
from encoder_training.models import create_encoder
from .config import EnvConfig, RNNTrainConfig, RNNBCConfig, RNNAgentConfig
from .train_decode_walk import Buffer, Walkers
from .training.goal_pairs_setup import build_env_sets
from .training.rnn_setup import write_rnn_world_spec
from .utils import gram_schmidt_2d_batch


# ---------------------------------------------------------------------------
# Batches and labels
# ---------------------------------------------------------------------------

def sample_batch(buf: Buffer, rng, batch_envs: int, per_walker: int):
    """`batch_envs` envs x every walker x `per_walker` moments. Returns env ids,
    walker ids and cell positions, `(B,)`, `(B,)`, `(B, 2)`."""
    envs = rng.choice(buf.n_envs, size=batch_envs, replace=False)
    e = np.repeat(envs, buf.W * per_walker)
    w = np.tile(np.repeat(np.arange(buf.W), per_walker), batch_envs)
    t = rng.randint(0, buf.n, size=len(e))
    return e, w, buf._at(e, w, t)


def near_far_masks(e, w, pos, radius: float, labels: str):
    """`(near, far)` boolean `(B, B)`: near = within `radius` (Euclidean), far
    otherwise, both restricted to the pairs the label source can relate --
    the same walker (`odometry`) or the same env (`coords`). numpy in, numpy
    out; torch tensors in (on any device), torch out on that device."""
    xp = torch if isinstance(e, torch.Tensor) else np
    same_env = e[:, None] == e[None, :]
    if labels == "odometry":
        group = same_env & (w[:, None] == w[None, :])
    elif labels == "coords":
        group = same_env
    else:
        raise ValueError(labels)
    d = pos[:, None, :] - pos[None, :, :]
    dist = xp.sqrt((d * d).sum(-1))
    if xp is torch:
        eye = torch.eye(len(e), dtype=torch.bool, device=e.device)
    else:
        eye = np.eye(len(e), dtype=bool)
    near = (dist < radius) & group & ~eye
    far = group & ~near & ~eye
    return near, far


# ---------------------------------------------------------------------------
# The harness's readout: local frame from scaffold neighbours, q = W (z_g - z_p)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_cells(encoder, gbook: np.ndarray, gain: float, device) -> np.ndarray:
    x = torch.from_numpy(np.asarray(gbook, dtype=np.float32)).to(device)
    return encoder(x, gain).float().cpu().numpy()


def local_frames(Z: np.ndarray, S: int, p: np.ndarray) -> np.ndarray:
    """`(B, 2, D)` Gram-Schmidt frames at cells `p (B, 2)` from the +x and +y
    neighbours (the inward neighbour, sign-flipped, on the far edges)."""
    Zg = Z.reshape(S, S, -1)
    x, y = p[:, 0], p[:, 1]
    sx = np.where(x < S - 1, 1.0, -1.0)
    sy = np.where(y < S - 1, 1.0, -1.0)
    xn = np.where(x < S - 1, x + 1, x - 1)
    yn = np.where(y < S - 1, y + 1, y - 1)
    d_rgt = sx[:, None] * (Zg[xn, y] - Zg[x, y])          # East = +x
    d_fwd = sy[:, None] * (Zg[x, yn] - Zg[x, y])          # North = +y
    return gram_schmidt_2d_batch(d_fwd, d_rgt)             # rows: East, North


def readout_direction(Z: np.ndarray, S: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """`q (B, 2)` in (x, y) for cell pairs, as `scaffold.project_displacement`."""
    W = local_frames(Z, S, p)
    disp = Z[g[:, 0] * S + g[:, 1]] - Z[p[:, 0] * S + p[:, 1]]
    return np.einsum("bij,bj->bi", W, disp)


def angular_error(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip((qn * dn).sum(1), -1.0, 1.0)))


def sample_pairs(rng, S: int, n: int, max_abs: int):
    p = rng.randint(0, S, size=(4 * n, 2))
    g = rng.randint(0, S, size=(4 * n, 2))
    r = np.abs(g - p).max(1)
    ok = (r >= 1) & (r <= max_abs)
    return p[ok][:n], g[ok][:n]


def eval_envset(encoder, es, gain: float, device, rng, n_pairs: int, max_abs: int) -> float:
    """Mean angular error of the projected readout over sampled pairs, all envs of the set."""
    errs = []
    S = es.tensors[0].size
    for t in es.tensors:
        Z = encode_cells(encoder, t.gbook, gain, device)
        p, g = sample_pairs(rng, S, n_pairs, max_abs)
        errs.append(angular_error(readout_direction(Z, S, p, g), (g - p).astype(float)))
    return float(np.concatenate(errs).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_only", type=str, default="", help="an encoder checkpoint: readout on this world only")
    # Encoder (att0.5's)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--out_dim", type=int, default=1024)
    p.add_argument("--gain_start", type=float, default=1.0)
    p.add_argument("--gain_end", type=float, default=100.0)
    p.add_argument("--attract", type=float, default=0.5)
    p.add_argument("--repel", type=float, default=1.0)
    p.add_argument("--rate", type=float, default=0.5)
    p.add_argument("--rate_eps", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=20.0)
    p.add_argument("--labels", choices=["odometry", "coords"], default="odometry")
    p.add_argument("--positions", choices=["walk", "iid"], default="walk",
                   help="walk: moments of the walkers' histories (the decode's data); iid: cells drawn "
                        "uniformly from the same envs -- the encoder's own sampling, on our arenas "
                        "(coords labels only; env-steps then count the walks still taken, for the axis)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # Walks (the decode's)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--n_val_envs", type=int, default=16)
    p.add_argument("--n_same_envs", type=int, default=4)
    p.add_argument("--walkers", type=int, default=8)
    p.add_argument("--steps_per_update", type=int, default=64)
    p.add_argument("--buffer_updates", type=int, default=20)
    p.add_argument("--batches_per_update", type=int, default=8)
    p.add_argument("--batch_envs", type=int, default=8)
    p.add_argument("--per_walker", type=int, default=64)
    p.add_argument("--max_abs", type=int, default=19, help="eval pairs within this Chebyshev range")
    # World
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=120)
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--fwhm_ratio", type=float, default=0.25)
    p.add_argument("--place_margin", type=int, default=20)
    p.add_argument("--place_region", type=str, default="anywhere")
    p.add_argument("--wall_seeds", type=str, default="0,10000000")
    # Run
    p.add_argument("--n_updates", type=int, default=4000)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_pairs", type=int, default=2048)
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def build_world(args, rng):
    cfg = RNNTrainConfig(
        env=EnvConfig(size=args.size, observation_size=args.observation_size, movement_mode="discrete"),
        agent=RNNAgentConfig(input_grid_state=True, input_goal_grid_state=True, input_sensory=False,
                             movement_mode="continuous", rnn_cell="mlp"),
        bc=RNNBCConfig(lr=args.lr), mode="pairs", n_envs=args.n_envs, n_val_envs=args.n_val_envs,
        n_updates=args.n_updates, eval_every=args.eval_every, seed=args.seed, device=args.device,
        fwhm_ratio=args.fwhm_ratio, lambdas=list(args.lambdas), env_generator=True,
        place_margin=args.place_margin, goal_val_frac=0.2, region_val_frac=0.1,
        wall_seeds=args.wall_seeds, pairs_per_env=0, place_region=args.place_region)
    train, heldout, same, split, vh, sgb = build_env_sets(cfg, rng, n_same=args.n_same_envs)[:6]
    return cfg, train, heldout, same, split, vh


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    cfg, train, heldout, same, split, vh = build_world(args, rng)
    print(f"world: {len(train)} train / {len(heldout)} {heldout.name} / {len(same)} same envs; "
          f"size={args.size}; Npos={vh.Npos} Ng={vh.Ng}; {time.time()-t0:.1f}s")

    if args.eval_only:
        ck = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        mc = ck["model_config"]
        encoder = create_encoder(EncoderModelConfig(**{k: v for k, v in mc.items()
                                                       if k in EncoderModelConfig.__dataclass_fields__}), str(device))
        encoder.load_state_dict(ck["state_dict"])
        encoder.eval()
        gain = float(ck.get("gain", mc.get("gain", 1.0)))
        ev = np.random.RandomState(0)
        out = {}
        for es in (train, heldout):
            out[es.name] = eval_envset(encoder, es, gain, device, ev, args.eval_pairs, args.max_abs)
        # Axis check: the same readout with (x, y) swapped, which should be much worse.
        S = heldout.tensors[0].size
        Z = encode_cells(encoder, heldout.tensors[0].gbook, gain, device)
        p, g = sample_pairs(ev, S, 2000, args.max_abs)
        q = readout_direction(Z, S, p, g)
        swapped = float(angular_error(q[:, ::-1], (g - p).astype(float)).mean())
        print(f"eval_only {os.path.basename(os.path.dirname(args.eval_only))}: gain {gain}; "
              f"readout error train {out['train']:.2f} / {heldout.name} {out[heldout.name]:.2f} deg "
              f"(axes swapped: {swapped:.1f})")
        return

    encoder = create_encoder(EncoderModelConfig(
        encoder_type="mlp", lambdas=list(args.lambdas), out_dim=args.out_dim, nonlinearity="gelu",
        output_nonlinearity="tanh", gain=args.gain_end, hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers), str(device))
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"encoder: {args.num_hidden_layers}x{args.hidden_dim} gelu -> {args.out_dim}, params={n_params:,}; "
          f"labels={args.labels} radius={args.radius}")
    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.save_dir is None:
        sub = run_name(None)
        if args.tag:
            sub = f"{args.tag}_{sub}"
        args.save_dir = str(run_dir("goal_pairs", sub))
    else:
        sub = os.path.basename(args.save_dir.rstrip("/"))
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"save_dir={args.save_dir}")
    u0, env_steps, history = 0, 0, []
    latest = os.path.join(args.save_dir, "enc_latest.pt")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt_state_dict"])
        u0, env_steps, history = int(ck["update"]), int(ck["env_steps"]), ck.get("history", [])
        print(f"resumed from update {u0} ({env_steps:,} env-steps)")
    else:
        write_rnn_world_spec(cfg, split, vh, generator="declared", save_dir=args.save_dir)
        run_manifest.begin(args.save_dir, kind="goal_pairs", name=sub,
                           config={**asdict(cfg), "argv": vars(args)}, parent=None, wandb_run=None)

    walkers = Walkers(train.envs, args.walkers, args.seed + 1000 * (u0 + 1))
    buf = Buffer(len(train), args.walkers, args.steps_per_update, args.buffer_updates)
    data_rng = np.random.RandomState(args.seed + 1 + u0)
    eval_rng = np.random.RandomState(args.seed + 99)
    steps_per_update = len(train) * args.walkers * args.steps_per_update
    B = args.batch_envs * args.walkers * args.per_walker
    t_train = time.time()

    def gain_at(u: int) -> float:
        return args.gain_start + (args.gain_end - args.gain_start) * (u - 1) / max(args.n_updates - 1, 1)

    def save(name: str, u: int) -> str:
        path = os.path.join(args.save_dir, name)
        torch.save({"state_dict": encoder.state_dict(), "opt_state_dict": opt.state_dict(),
                    "model_config": {"encoder_type": "mlp", "lambdas": list(args.lambdas), "out_dim": args.out_dim,
                                     "nonlinearity": "gelu", "output_nonlinearity": "tanh", "gain": gain_at(u),
                                     "hidden_dim": args.hidden_dim, "num_hidden_layers": args.num_hidden_layers},
                    "gain": gain_at(u), "argv": vars(args), "update": u, "env_steps": env_steps,
                    "history": history}, path)
        return path

    for u in range(u0 + 1, args.n_updates + 1):
        buf.add(walkers.segment(args.steps_per_update))
        env_steps += steps_per_update
        gain = gain_at(u)
        encoder.train()
        losses = []
        for _ in range(args.batches_per_update):
            if args.positions == "iid":
                envs = data_rng.choice(len(train), size=args.batch_envs, replace=False)
                e = np.repeat(envs, args.walkers * args.per_walker)
                w = np.zeros(len(e), dtype=np.int64)
                pos = data_rng.randint(0, args.size, size=(len(e), 2))
            else:
                e, w, pos = sample_batch(buf, data_rng, args.batch_envs, args.per_walker)
            # Gather codes per env (one indexed take per env in the batch).
            x = np.empty((B, vh.Ng), dtype=np.float32)
            ids = pos[:, 0] * args.size + pos[:, 1]
            for k in np.unique(e):
                m = e == k
                x[m] = train.tensors[k].gbook[ids[m]]
            near, far = near_far_masks(torch.from_numpy(e).to(device), torch.from_numpy(w).to(device),
                                       torch.from_numpy(pos).float().to(device), args.radius, args.labels)
            xb = torch.from_numpy(x).to(device)
            z = encoder(xb, gain)
            K = (z @ z.T).clamp(-1.0, 1.0)
            loss = mse_attract_repel(K, near, attract_lambda=args.attract, repel_weight=args.repel, far_mask=far)
            if args.rate > 0:
                loss = loss + args.rate * coding_rate_loss(z, eps=args.rate_eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=args.grad_clip)
            opt.step()
            losses.append(float(loss.item()))

        if u == 1 or u % args.eval_every == 0 or u == args.n_updates:
            encoder.eval()
            tr = eval_envset(encoder, same, gain, device, eval_rng, args.eval_pairs, args.max_abs)
            ho = eval_envset(encoder, heldout, gain, device, eval_rng, args.eval_pairs, args.max_abs)
            history.append({"update": u, "env_steps": env_steps, "loss": float(np.mean(losses)),
                            "gain": gain, "train": tr, "heldout": ho})
            print(f"u={u:5d} steps={env_steps:>11,d} loss={np.mean(losses):.4f} gain={gain:6.1f} | "
                  f"train {tr:.2f} | {heldout.name} {ho:.2f} | {time.time()-t_train:.0f}s", flush=True)
        if u % args.ckpt_every == 0:
            save("enc_latest.pt", u)
            path = save(f"enc_u{u}.pt", u)
            run_manifest.record_checkpoint(args.save_dir, os.path.basename(path), update=u)

    encoder.eval()
    final_rng = np.random.RandomState(12345)
    final = {"train": eval_envset(encoder, same, gain_at(args.n_updates), device, final_rng, 20000, args.max_abs),
             heldout.name: eval_envset(encoder, heldout, gain_at(args.n_updates), device, final_rng, 20000, args.max_abs)}
    reached = {}
    for thr in (10.0, 5.0, 2.0, 1.0):
        hit = [h for h in history if h["heldout"] <= thr]
        reached[str(thr)] = (hit[0]["env_steps"], hit[0]["update"]) if hit else None
    print(f"=== FINAL readout: train {final['train']:.2f} | {heldout.name} {final[heldout.name]:.2f} deg")
    print("held-out env-steps to threshold:", reached)
    with open(os.path.join(args.save_dir, "final.json"), "w") as f:
        json.dump({"final": final, "history": history, "argv": vars(args), "env_steps": env_steps,
                   "steps_to_threshold": reached, "params": n_params}, f, indent=1)
    path = save("encoder_final.pt", args.n_updates)
    run_manifest.record_checkpoint(args.save_dir, "encoder_final.pt", update=args.n_updates)
    run_manifest.finish(args.save_dir)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
