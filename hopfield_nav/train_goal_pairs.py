"""Experiment A: a memoryless network on i.i.d. (start, goal) pairs (plan §3).

    python -m hopfield_nav.train_goal_pairs --mode grid --movement_mode continuous ...

No env stepping, no DAgger, no rollouts. Every cell's encodings are
precomputed once per env; a training batch is an index gather; an update is
one forward and one backward. The static quadrant table (plan §2.5) is
evaluated on the training envs, on held-out envs (`wall=held_out,
place=held_out`, the split's `base_val`), and on a `same` subset of the
training envs -- the env-side memorisation probe -- every `eval_every`
updates, and enumerated in full at the end.

The three input modes are three `RNNAgentConfig`s (plan §2.2):

    xy       [xy(p), xy(g)]            the coordinate ceiling
    grid     [gbook(p), gbook(g)]      the grid code, no sensory
    regular  [omni(p), omni(g)]        the ray-cast, heading-free

They are exposed as `--mode` so a launcher cannot half-configure one.

This is its own trainer rather than a fourth mode of `train_rnn.py`: that
file's modes are the continual-learning protocol and are built around a
rollout batch; this one has no rollout. It borrows the world (`rnn_world`),
the manifest, and the evaluator, and nothing from `updates/` or `rollout/`.
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
from .config import EnvConfig, RNNTrainConfig, RNNBCConfig
from .evaluation.goal_pairs import (
    format_table, pair_inputs, pair_targets, sample_pairs, sample_trajectory_pairs)
from .policy.agent_rnn import compute_rnn_input_dim
from .policy.pair_regressor import PairRegressor
from .training.goal_pairs_setup import (
    MODES, EnvSet, agent_cfg_for_mode, build_env_sets, eval_all, jsonable)
from .training.rnn_setup import write_rnn_world_spec


def train_batch(train: EnvSet, cells, acfg, movement_mode, pairs_per_env, rng, device,
                sampler: str = "iid"):
    draw = sample_pairs if sampler == "iid" else sample_trajectory_pairs
    xs, ys = [], []
    for t in train.tensors:
        p, g = draw(cells, "train", "train", pairs_per_env, rng)
        xs.append(pair_inputs(t, acfg, p, g))
        ys.append(pair_targets(p, g, t.size, movement_mode))
    x = torch.from_numpy(np.concatenate(xs)).to(device)
    y = torch.from_numpy(np.concatenate(ys)).to(device)
    return x, y


def flatten_for_log(tables: dict, prefix: str = "eval") -> dict:
    log = {}
    for es, agg in tables.items():
        for (s, g), row in agg.items():
            for who in ("model", "nn", "random", "teacher"):
                if who in row:
                    log[f"{prefix}/{es}/{s}x{g}/{who}"] = row[who]["metric"]
            if "metric_std" in row["model"]:
                log[f"{prefix}/{es}/{s}x{g}/model_std"] = row["model"]["metric_std"]
    return log


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment A: memoryless goal-conditioned pairs")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--movement_mode", choices=["discrete", "continuous"], default="continuous")
    # Model
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--nonlinearity", choices=["tanh", "relu"], default="relu")
    p.add_argument("--dropout", type=float, default=0.0)
    # Data
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--n_val_envs", type=int, default=16)
    p.add_argument("--n_same_envs", type=int, default=8)
    p.add_argument("--pairs_per_env", type=int, default=512)
    p.add_argument("--pair_sampler", choices=["iid", "trajectory"], default="iid",
                   help="iid: uniform (p, g). trajectory: every cell on the straight line "
                        "from p0 to g, so the displacement distribution matches an ideal "
                        "rollout's (plan A1y).")
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=120)
    p.add_argument("--wall_resolution", type=int, default=1)
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--fwhm_ratio", type=float, default=0.25)
    p.add_argument("--place_margin", type=int, default=20)
    p.add_argument("--goal_val_frac", type=float, default=0.2)
    p.add_argument("--region_val_frac", type=float, default=0.1)
    p.add_argument("--wall_seeds", type=str, default="0,10000000")
    p.add_argument("--place_region", type=str, default="anywhere",
                   help="'anywhere' or 'rect:X0,Y0,W,H' in scaffold cells. With a rect, "
                        "every training env (and base_val) sits inside it.")
    p.add_argument("--n_ood_place", type=int, default=0,
                   help="Mint this many envs OUTSIDE --place_region as a fourth env "
                        "set, heldout_out (plan sec 2.4, corner holdout). 0 = off.")
    # Optimisation
    p.add_argument("--n_updates", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_schedule", choices=["none", "cosine", "step"], default="none",
                   help="step: multiply lr by --lr_step_gamma at --lr_step_at (a fraction of n_updates). "
                        "cosine-from-the-start hurt on grid mode (A1); the models are still "
                        "descending at a constant lr and only destabilise late.")
    p.add_argument("--lr_step_at", type=float, default=0.75)
    p.add_argument("--lr_step_gamma", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # Eval
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_pairs", type=int, default=2048,
                   help="pairs per quadrant per env during training; final eval enumerates")
    p.add_argument("--final_enumerate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ckpt_every", type=int, default=500)
    # Run
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wandb_project", default="train_goal_pairs")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    acfg = agent_cfg_for_mode(args.mode, args.movement_mode,
                              hidden_size=args.hidden_size,
                              num_rnn_layers=args.num_layers,
                              rnn_cell="mlp", rnn_nonlinearity=args.nonlinearity,
                              dropout=args.dropout)
    cfg = RNNTrainConfig(
        env=EnvConfig(size=args.size, observation_size=args.observation_size,
                      movement_mode=args.movement_mode,
                      wall_resolution=args.wall_resolution),
        agent=acfg, bc=RNNBCConfig(lr=args.lr),
        mode="pairs", n_envs=args.n_envs, n_val_envs=args.n_val_envs,
        n_updates=args.n_updates, eval_every=args.eval_every, seed=args.seed,
        device=args.device, use_wandb=args.use_wandb, wandb_project=args.wandb_project,
        fwhm_ratio=args.fwhm_ratio, lambdas=list(args.lambdas),
        env_generator=True, place_margin=args.place_margin,
        goal_val_frac=args.goal_val_frac, region_val_frac=args.region_val_frac,
        wall_seeds=args.wall_seeds, pairs_per_env=args.pairs_per_env,
        place_region=args.place_region,
    )
    # rnn_world builds a scaffold only when grid state is on or the generator
    # is declared; the generator IS declared, so every mode gets a scaffold and
    # every mode's envs are placed identically for the same seed.
    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    built = build_env_sets(cfg, rng, n_same=args.n_same_envs, n_ood_place=args.n_ood_place)
    train, heldout, same, split, vh, sgb = built[:6]
    heldout_out = built[6] if len(built) > 6 else None
    cells = split.cell_sets()
    print(f"world: {len(train)} train / {len(heldout)} {heldout.name} / {len(same)} same"
          + (f" / {len(heldout_out)} heldout_out" if heldout_out else "")
          + f" envs; place={args.place_region}; Npos={vh.Npos} Ng={vh.Ng}; "
          f"cells={cells.summary()}; {time.time()-t0:.1f}s")
    if heldout_out:
        print("  heldout_out offsets:", [tuple(int(v) for v in o) for o in heldout_out.offsets])

    D = compute_rnn_input_dim(acfg, args.observation_size, vh.Ng)
    model = PairRegressor(D, args.hidden_size, args.num_layers, args.movement_mode,
                          nonlinearity=args.nonlinearity, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: mode={args.mode} D={D} hidden={args.hidden_size} layers={args.num_layers} "
          f"{args.nonlinearity} params={n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.n_updates)
    elif args.lr_schedule == "step":
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[int(args.lr_step_at * args.n_updates)], gamma=args.lr_step_gamma)
    else:
        sched = None

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config={**asdict(cfg), "tag": args.tag,
                                                                    "input_dim": D, "params": n_params})
    if args.save_dir is None:
        sub = run_name(wandb_run.name if wandb_run is not None else None)
        if args.tag:
            sub = f"{args.tag}_{sub}"
        args.save_dir = str(run_dir("goal_pairs", sub))
    else:
        sub = os.path.basename(args.save_dir.rstrip("/"))
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"save_dir={args.save_dir}")
    write_rnn_world_spec(cfg, split, vh, generator="declared", save_dir=args.save_dir)
    run_manifest.begin(args.save_dir, kind="goal_pairs", name=sub,
                       config={**asdict(cfg), "argv": vars(args)}, parent=None,
                       wandb_run=wandb_run)

    sets = [train, heldout, same] + ([heldout_out] if heldout_out else [])
    set_names = [es.name for es in sets]
    seen = set()          # (env, p, g) triples, the C12 exposure counter
    history = []
    data_rng = np.random.RandomState(args.seed + 1)
    t_train = time.time()
    for u in range(1, args.n_updates + 1):
        x, y = train_batch(train, cells, acfg, args.movement_mode, args.pairs_per_env,
                           data_rng, device, sampler=args.pair_sampler)
        loss = model.loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        if sched is not None:
            sched.step()

        if u == 1 or u % args.eval_every == 0 or u == args.n_updates:
            tables = eval_all(model, sets, acfg, cells, args.movement_mode, device,
                              n_per_quadrant=args.eval_pairs, seed=u)
            log = {"train/loss": float(loss.item()), "train/grad_norm": float(gn),
                   "train/lr": opt.param_groups[0]["lr"], "update": u,
                   "train/elapsed_s": time.time() - t_train}
            log.update(flatten_for_log(tables))
            history.append({"update": u, "loss": float(loss.item()), "tables": jsonable(tables)})
            tt = tables["train"][("train", "train")]["model"]["metric"]
            ht = tables[heldout.name][("train", "train")]["model"]["metric"]
            hr = tables[heldout.name][("region", "region")]["model"]["metric"]
            nn_hr = tables[heldout.name][("region", "region")]["nn"]["metric"]
            extra = ""
            if heldout_out:
                ot = tables["heldout_out"][("train", "train")]["model"]["metric"]
                orr = tables["heldout_out"][("region", "region")]["model"]["metric"]
                extra = f" | OUT tt={ot:.2f} rr={orr:.2f}"
            print(f"u={u:5d} loss={loss.item():.4f} gn={float(gn):.2f} | train tt={tt:.2f} | "
                  f"{heldout.name} tt={ht:.2f} rr={hr:.2f} (nn {nn_hr:.2f}){extra} | "
                  f"{time.time()-t_train:.0f}s", flush=True)
            if wandb_run is not None:
                wandb_run.log(log)
        if u % args.ckpt_every == 0:
            path = os.path.join(args.save_dir, f"pairs_u{u}.pt")
            torch.save({"model_state_dict": model.state_dict(), "cfg": asdict(cfg),
                        "argv": vars(args), "input_dim": D, "update": u}, path)
            run_manifest.record_checkpoint(args.save_dir, os.path.basename(path), update=u)

    # Final: enumerate every pair in every quadrant.
    final = eval_all(model, sets, acfg, cells, args.movement_mode, device,
                     n_per_quadrant=None if args.final_enumerate else args.eval_pairs,
                     seed=args.n_updates)
    print("\n=== FINAL (enumerated)" if args.final_enumerate else "\n=== FINAL (sampled)")
    for es in set_names:
        print(format_table(final[es], args.movement_mode, title=es))
        print()
    with open(os.path.join(args.save_dir, "final_tables.json"), "w") as f:
        json.dump({"final": jsonable(final), "history": history, "argv": vars(args),
                   "cells": cells.summary(), "input_dim": D, "params": n_params}, f, indent=1)
    path = os.path.join(args.save_dir, "pairs_final.pt")
    torch.save({"model_state_dict": model.state_dict(), "cfg": asdict(cfg),
                "argv": vars(args), "input_dim": D, "update": args.n_updates}, path)
    run_manifest.record_checkpoint(args.save_dir, "pairs_final.pt", update=args.n_updates)
    run_manifest.finish(args.save_dir)
    if wandb_run is not None:
        wandb_run.log({**flatten_for_log(final, "final"), "update": args.n_updates})
        wandb_run.finish()
    print(f"saved {path}")


if __name__ == "__main__":
    main()
