"""Generate a continual-learning history JSON from a phased Hopfield checkpoint.

Loads a phased ``final.pt`` (from ``train_phased*.py``), reconstructs the
training-time eval world, and runs the sequential continual protocol over a
**frozen** policy (only the Hopfield store changes). Writes the standardized
history JSON.

Differences from ``hopfield_nav.evaluation.metrics.evaluate_sequential_episodes``:
  - records ``steps_to_goal`` per mini-episode (the loop step at which the
    agent first sat on the goal, or ``None`` if it timed out);
  - emits the standardized history schema (`reached` / `steps_to_goal` per
    (step, env)), so the same plotter handles both this and the RNN baseline;
  - splits the old combined ``oracle_store_at_goal`` flag into two independent
    knobs:
      * ``--oracle_store_at_goal``         — at goal, force store-fire
        (overrides agent's store head). Off-goal: agent decides.
      * ``--oracle_lock_store_not_at_goal``— off goal, suppress all stores.
        At goal: agent decides.
    To reproduce the old combined ``oracle_store_at_goal=True`` behavior,
    pass **both** flags.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from hopfield_nav.policy.agent import NavAgent
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.world.env import GridEnv, at_goal
from hopfield_nav.evaluation import protocols
from hopfield_nav.world.env import make_env
from hopfield_nav.evaluation.metrics import agent_step, random_start
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint, eval_field, eval_specs,
    load_agent, world_spec_for,
)
from .baseline import merge_iter_traces
from hopfield import Hopfield
from hopfield_nav.world import generate as gen
from hopfield_nav.world.scaffold import VectorHash, place_envs


def _load_scaffold_cache(vh: VectorHash, cache_dir: str, cfg, *,
                         mmap: bool) -> None:
    """Populate ``vh.encoded_Phi`` from a prep_scaffold dump.

    Sanity-checks that ``lambdas``, ``Npos``, ``fwhm_ratio`` in the cache
    match cfg before loading. Raises on mismatch.
    """
    import json as _json
    meta_path = os.path.join(cache_dir, "meta.json")
    enc_path = os.path.join(cache_dir, "encoded_Phi.npy")
    if not os.path.exists(meta_path):
        raise RuntimeError(f"scaffold cache missing {meta_path}")
    if not os.path.exists(enc_path):
        raise RuntimeError(f"scaffold cache missing {enc_path}")
    with open(meta_path) as f:
        meta = _json.load(f)
    if list(meta["lambdas"]) != list(vh.lambdas):
        raise RuntimeError(
            f"scaffold cache lambdas {meta['lambdas']} != cfg {list(vh.lambdas)}"
        )
    if int(meta["Npos"]) != int(vh.Npos):
        raise RuntimeError(
            f"scaffold cache Npos {meta['Npos']} != cfg {int(vh.Npos)}"
        )
    if abs(float(meta["fwhm_ratio"]) - float(cfg.fwhm_ratio)) > 1e-6:
        raise RuntimeError(
            f"scaffold cache fwhm_ratio {meta['fwhm_ratio']} != cfg "
            f"{float(cfg.fwhm_ratio)}"
        )
    mmap_mode = "r" if mmap else None
    vh.encoded_Phi = np.load(enc_path, mmap_mode=mmap_mode)
    print(f"[agenthash] loaded scaffold cache from {cache_dir}/  "
          f"(mmap={mmap})  encoded_Phi.shape={vh.encoded_Phi.shape}",
          flush=True)


class _StoreTrainer:
    """Online BCE trainer for ``agent.store_head``, per-rollout batches.

    Freezes everything in ``agent`` except ``store_head`` on construction.
    Collects ``(features, at_goal_label)`` per ``agent_step`` inside a
    mini-episode; after the rollout finishes, runs ``n_updates`` BCE-with-logits
    steps on that single-rollout batch. No persistent buffer.

    Features are read from ``out["h_rnn"][-1, 0, :]`` in ``mini_episode`` —
    for the GRU this equals the trunk features the store head saw at that
    timestep, captured no-grad (which is fine: BCE backprop only updates
    ``store_head``'s own weights).
    """

    def __init__(self, agent: NavAgent, lr: float, device: torch.device) -> None:
        self.agent = agent
        self.device = device
        for p in agent.parameters():
            p.requires_grad_(False)
        for p in agent.store_head.parameters():
            p.requires_grad_(True)
        self.optimizer = torch.optim.Adam(
            agent.store_head.parameters(), lr=lr,
        )
        self._feat_step: list[torch.Tensor] = []
        self._label_step: list[float] = []

    def begin_episode(self) -> None:
        self._feat_step.clear()
        self._label_step.clear()

    def add(self, features: torch.Tensor, at_goal: bool) -> None:
        self._feat_step.append(features.detach())
        self._label_step.append(1.0 if at_goal else 0.0)

    def end_episode_train(self, n_updates: int) -> dict[str, float]:
        n_steps = len(self._feat_step)
        if n_steps == 0 or n_updates <= 0:
            return {"bce": float("nan"), "n_pos": 0, "n_neg": 0,
                    "n_steps": n_steps, "skipped": True}
        feats = torch.stack(self._feat_step, dim=0)  # (T, hidden)
        labels = torch.tensor(self._label_step, device=self.device,
                              dtype=torch.float32)
        n_pos = float(labels.sum().item())
        n_neg = float(n_steps - n_pos)
        if n_pos < 1.0 or n_neg < 1.0:
            return {"bce": float("nan"), "n_pos": int(n_pos),
                    "n_neg": int(n_neg), "n_steps": n_steps, "skipped": True}
        pos_weight = torch.tensor(n_neg / n_pos, device=self.device)
        last_loss = float("nan")
        for _ in range(n_updates):
            logits = self.agent.store_head(feats).squeeze(-1)  # (T,)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight,
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            last_loss = loss.item()
        return {"bce": last_loss, "n_pos": int(n_pos), "n_neg": int(n_neg),
                "n_steps": n_steps, "skipped": False}


def mini_episode(
    *,
    agent: NavAgent,
    env: GridEnv,
    env_offset: tuple[int, int],
    vectorhash: VectorHash,
    hopfield: Hopfield,
    cfg,
    device: torch.device,
    local_idx: int,
    allow_store: bool,
    max_steps: int,
    rng: np.random.RandomState,
    goal_in_mem: dict[int, bool],
    stored_at_goal_count: dict[int, int],
    deterministic: bool,
    oracle_store_at_goal: bool,
    oracle_lock_store_not_at_goal: bool,
    lock_store_after_goal: bool,
    store_trainer: "_StoreTrainer | None" = None,
) -> tuple[int, int | None, float | None, int, int]:
    """Run one mini-episode from a random non-goal start.

    Thin wrapper over ``evaluation.protocols.run_mini_episode``, which is
    shared with ``eval.evaluate_sequential_episodes``. The tuple return is kept
    so this module's caller and its history schema are unchanged.

    Returns ``(reached, steps_to_goal, path_to_goal, stored_at_goal,
    stored_off_goal)``. ``steps_to_goal`` is the loop step at which the agent
    first sat on the goal at decision time, or ``None`` if the cap was hit;
    ``path_to_goal`` is the cumulative L2 displacement to that moment.
    """
    if store_trainer is not None:
        store_trainer.begin_episode()
    rec = protocols.run_mini_episode(
        agent=agent, env=env, env_offset=env_offset, vectorhash=vectorhash,
        hopfield=hopfield, cfg=cfg, device=device, local_idx=local_idx,
        allow_store=allow_store, max_steps=max_steps, rng=rng,
        goal_in_mem=goal_in_mem, stored_at_goal_count=stored_at_goal_count,
        deterministic=deterministic,
        oracle_store_at_goal=oracle_store_at_goal,
        suppress_off_goal_stores=oracle_lock_store_not_at_goal,
        lock_store_after_goal=lock_store_after_goal,
        on_step=(None if store_trainer is None else store_trainer.add),
    )
    return (rec.reached, rec.steps_to_goal, rec.path_to_goal,
            rec.stored_at_goal, rec.stored_off_goal)


def run_sequential(
    *,
    agent: NavAgent,
    val_envs: list[GridEnv],
    vectorhash: VectorHash,
    env_offsets: list[tuple[int, int]],
    cfg,
    device: torch.device,
    iters_per_block: int,
    max_steps: int,
    seed: int,
    deterministic: bool,
    oracle_store_at_goal: bool,
    oracle_lock_store_not_at_goal: bool,
    lock_store_after_goal: bool,
    store_trainer: "_StoreTrainer | None" = None,
    train_store_updates_per_rollout: int = 1,
) -> tuple[
    list[tuple[int, int, dict[int, dict]]],
    list[tuple[int, int, int]],
    dict[int, int],
]:
    """Outer continual protocol. Hopfield is created once and never reset.

    Returns (trace, blocks, stored_at_goal_count). `blocks` end is inclusive.
    """
    agent.eval()
    N = len(val_envs)
    embed_dim = vectorhash.encoded_Phi.shape[2]
    hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
    rng = np.random.RandomState(seed)
    goal_in_mem: dict[int, bool] = {i: False for i in range(N)}
    stored_at_goal_count: dict[int, int] = {i: 0 for i in range(N)}

    trace: list[tuple[int, int, dict[int, dict]]] = []
    blocks: list[tuple[int, int, int]] = []
    cur_iter = 0
    for i in range(N):
        block_start = cur_iter
        block_bce_sum = 0.0
        block_bce_n = 0
        block_skipped = 0
        block_n_pos_total = 0
        block_n_neg_total = 0
        for _k in range(iters_per_block):
            inner: dict[int, dict] = {}
            for j in range(i + 1):
                env_j = val_envs[j]
                env_offset_j = env_offsets[j]
                allow_store = (j == i)
                reached, stg, ptg, sag, sog = mini_episode(
                    agent=agent, env=env_j, env_offset=env_offset_j,
                    vectorhash=vectorhash, hopfield=hopfield,
                    cfg=cfg, device=device,
                    local_idx=j, allow_store=allow_store,
                    max_steps=max_steps, rng=rng,
                    goal_in_mem=goal_in_mem,
                    stored_at_goal_count=stored_at_goal_count,
                    deterministic=deterministic,
                    oracle_store_at_goal=oracle_store_at_goal,
                    oracle_lock_store_not_at_goal=oracle_lock_store_not_at_goal,
                    lock_store_after_goal=lock_store_after_goal,
                    store_trainer=store_trainer,
                )
                if store_trainer is not None:
                    info = store_trainer.end_episode_train(
                        train_store_updates_per_rollout
                    )
                    if info["skipped"]:
                        block_skipped += 1
                    else:
                        block_bce_sum += info["bce"]
                        block_bce_n += 1
                    block_n_pos_total += int(info["n_pos"])
                    block_n_neg_total += int(info["n_neg"])
                inner[j] = {
                    "reached": reached,
                    "steps_to_goal": stg,
                    "path_to_goal": ptg,
                    "stored_at_goal": sag,
                    "stored_off_goal": sog,
                }
            trace.append((cur_iter, i, inner))
            cur_iter += 1
        blocks.append((block_start, cur_iter - 1, i))
        line = (f"  block {i} done  cur_iter={cur_iter}  "
                f"hopfield_mem={hopfield.num_memories}  "
                f"goal_in_mem={goal_in_mem[i]}")
        if store_trainer is not None:
            mean_bce = (block_bce_sum / block_bce_n) if block_bce_n > 0 else float("nan")
            line += (f"  store_bce_mean={mean_bce:.4f}  "
                     f"trained={block_bce_n}  skipped={block_skipped}  "
                     f"n_pos={block_n_pos_total}  n_neg={block_n_neg_total}")
        print(line, flush=True)

    return trace, blocks, stored_at_goal_count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # I/O
    p.add_argument("--out", required=True,
                   help="Output history JSON path.")
    p.add_argument("--ckpt", required=True,
                   help="Phased checkpoint (final.pt from train_phased*.py).")
    p.add_argument("--run_name", default=None,
                   help="Recorded in metadata.run_name. Default: derived from --out.")
    # Cfg overrides
    p.add_argument("--encoder_override", default=None,
                   help="Override cfg.encoder_checkpoint from the ckpt.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_envs", type=int, default=None,
                   help="Override cfg.num_val_envs (None = use ckpt's).")
    p.add_argument("--static_vectorhash", action="store_true",
                   help="Override cfg.vectorhash.static_vectorhash = True.")
    p.add_argument("--scaffold_cache", default=None,
                   help="Directory containing encoded_Phi.npy + meta.json from "
                        "prep_scaffold. If set: skip build_scaffold + "
                        "precompute_encoded_phi, load from disk instead. "
                        "Requires --static_vectorhash.")
    p.add_argument("--mmap", action="store_true",
                   help="When --scaffold_cache is set, mmap encoded_Phi.npy "
                        "(read-only) so parallel subruns share OS page-cache "
                        "pages instead of each holding a private copy.")
    # Protocol
    p.add_argument("--iters_per_block", type=int, default=800,
                   help="Outer iterations per env-block.")
    p.add_argument("--max_steps", type=int, default=100,
                   help="Mini-episode step cap.")
    p.add_argument("--seed", type=int, default=3000,
                   help="Base seed for the mini-episode RNG (iter k uses seed + k).")
    p.add_argument("--split", type=str, default="recorded",
                   help="Which validation envs to run on. 'recorded' (default) "
                        "is the run's own base_val from world.json. Otherwise "
                        "'trait=level' pairs over place/wall/goal, levels "
                        "same | held_out | ood; unnamed traits default to "
                        "held_out. Ignored when --env_seed is set, which is the "
                        "explicit baseline-matching path.")
    p.add_argument("--val_seed", type=int, default=0,
                   help="Seed for minting a --split env set.")
    p.add_argument("--env_seed", type=int, default=None,
                   help="If set: rebuild val_envs with RandomState(env_seed) using "
                        "baseline-compatible draw order (no envs_per_world skip), "
                        "so iter k of agenthash matches iter k of baseline when "
                        "env_seed_base = baseline's --seed. If None: use the "
                        "ckpt's saved cfg.seed (legacy behavior).")
    p.add_argument("--num_full_iters", type=int, default=1,
                   help="Run the entire protocol N times with seeds [seed, "
                        "seed+1, ..., seed+N-1] (and env_seed+k when env_seed is "
                        "set). Plotter aggregates as mean ± 1σ across iters.")
    p.add_argument("--stochastic_policy", action="store_true",
                   help="Sample actions from the policy distribution instead "
                        "of using greedy argmax action selection.")
    # Store-control flags
    p.add_argument("--lock_store_after_goal", action="store_true",
                   help="Once an env's goal has been stored, suppress further "
                        "stores in that env.")
    p.add_argument("--oracle_store_at_goal", action="store_true",
                   help="At goal: force store-fire (override agent's store head).")
    p.add_argument("--oracle_lock_store_not_at_goal", action="store_true",
                   help="Off goal: suppress all stores.")
    # Online store-head training
    p.add_argument("--train_store", action="store_true",
                   help="Train agent.store_head online via BCE during the "
                        "sequential rollout. Trunk + other heads stay frozen. "
                        "Composes with the oracle flags above (oracle gates "
                        "Hopfield writes; this gates store_head weights).")
    p.add_argument("--train_store_lr", type=float, default=3e-4,
                   help="Adam lr for store_head when --train_store is on.")
    p.add_argument("--train_store_updates_per_rollout", type=int, default=1,
                   help="BCE updates after each mini-episode (rollout). "
                        "Batch is the just-finished episode's per-step "
                        "(features, at_goal) pairs.")
    p.add_argument("--goal_radius", type=float, default=None,
                   help="Override EnvConfig.goal_radius from the saved checkpoint. "
                        "Default: use the value saved at training time (or 0.5 for "
                        "checkpoints saved before the field was added).")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[agenthash] loading {args.ckpt}")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    if args.n_envs is not None:
        cfg.num_val_envs = int(args.n_envs)
    if args.static_vectorhash:
        cfg.vectorhash.static_vectorhash = True
    if args.goal_radius is not None:
        cfg.env.goal_radius = float(args.goal_radius)

    encoder_path = args.encoder_override or cfg.encoder_checkpoint
    encoder, enc_cfg, enc_gain = load_encoder(encoder_path, str(device))
    embed_dim = enc_cfg.out_dim
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(enc_gain)

    # Build the scaffold + encoded_Phi ONCE. Both depend only on
    # cfg.vectorhash.lambdas + the encoder, which are fixed across iters; only
    # val_envs and their offsets need to vary per-iter. Offsets are no longer
    # scaffold state, so re-placing per iter is a list rebuild and cannot
    # perturb gbook / encoded_Phi at all.
    if args.scaffold_cache is not None and not cfg.vectorhash.static_vectorhash:
        raise RuntimeError(
            "--scaffold_cache requires --static_vectorhash (the cache only "
            "stores encoded_Phi; the non-static assoc fit needs gbook too)"
        )

    torch.manual_seed(0)
    np.random.seed(0)
    if args.scaffold_cache is not None:
        # Cached path: skip the heavy scaffold build entirely. Works for both
        # env_seed-set (rebuild val_envs per iter) and env_seed-unset (rebuild
        # val_envs once via the build_eval_world draw order) flows.
        vh = VectorHash(cfg.vectorhash)
        _load_scaffold_cache(vh, args.scaffold_cache, cfg, mmap=args.mmap)
        if args.env_seed is None:
            # Mirror build_eval_world's val_env construction (rng skip + draws).
            rng = np.random.RandomState(cfg.seed)
            for _ in range(cfg.envs_per_world * cfg.num_worlds):
                rng.randint(0, 10_000_000)
            val_envs = [
                make_env(cfg.env, cfg.agent.movement_mode,
                         seed=int(rng.randint(0, 10_000_000)))
                for _ in range(cfg.num_val_envs)
            ]
            offsets = place_envs(cfg.num_val_envs, cfg.env.size, vh.Npos,
                                 np.random, placement="spread")
        else:
            val_envs = []      # rebuilt per iter below
            offsets = []       # ditto
    elif args.env_seed is None:
        # build_eval_world handles scaffold + envs together, preferring the
        # recorded world.json and falling back to the ckpt's training-time seed
        # conventions. Envs constant across iters.
        if args.split != gen.RECORDED:
            spec = world_spec_for(args.ckpt)
            if spec is None:
                raise SystemExit(
                    f"--split needs a world.json and {args.ckpt} has none; it "
                    "predates runs recording their envs.")
            levels = gen.parse_levels(args.split)
            specs = eval_specs(spec, levels, n_envs=cfg.num_val_envs,
                               seed=args.val_seed)
            gen.val_set_report(spec.split, specs, cfg.env, levels)
            vh = eval_field(cfg, encoder, str(device), spec)
            val_envs = gen.build_envs(specs, cfg.env, cfg.agent.movement_mode)
            offsets = [s.offset for s in specs]
            print(f"[agenthash] split={gen.levels_key(levels)}  "
                  f"{len(val_envs)} envs minted from world.json")
        else:
            val_envs, vh, offsets = build_eval_world(cfg, encoder, str(device),
                                                     ckpt_path=args.ckpt)
    else:
        vh = VectorHash(cfg.vectorhash)
        vh.build_scaffold()
        vh.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=str(device))
        val_envs = []      # rebuilt per iter below
        offsets = []       # ditto

    agent = load_agent(cfg, ck["agent_state_dict"], embed_dim, device)

    n_full_iters = max(1, int(args.num_full_iters))
    print(f"[agenthash] iters_per_block={args.iters_per_block}  "
          f"max_steps={args.max_steps}  base_seed={args.seed}  "
          f"env_seed={args.env_seed}  num_full_iters={n_full_iters}")
    print(f"[agenthash] oracle_store_at_goal={args.oracle_store_at_goal}  "
          f"oracle_lock_store_not_at_goal={args.oracle_lock_store_not_at_goal}  "
          f"lock_store_after_goal={args.lock_store_after_goal}")
    print(f"[agenthash] stochastic_policy={args.stochastic_policy}")
    if args.train_store:
        print(f"[agenthash] train_store=True  lr={args.train_store_lr}  "
              f"updates_per_rollout={args.train_store_updates_per_rollout}")

    iter_traces: list[tuple[list, list]] = []
    iter_env_goals: list[list[list[int]]] = []
    iter_stored_at_goal: list[dict] = []
    iter_env_offsets: list[list[list[int]]] = []

    for k in range(n_full_iters):
        # Mini-episode RNG seed, varies per iter.
        seed_k = args.seed + k

        # When training the store head online, each --num_full_iters slot must
        # start from a fresh phase-A head (else iter k+1 inherits a partially
        # trained head from iter k, breaking seed independence).
        store_trainer: _StoreTrainer | None = None
        if args.train_store:
            agent.load_state_dict(ck["agent_state_dict"])
            agent.eval()
            store_trainer = _StoreTrainer(
                agent, lr=args.train_store_lr, device=device,
            )

        # Per-iter env rebuild (only when --env_seed is set).
        env_seed_k: int | None = None
        if args.env_seed is not None:
            env_seed_k = int(args.env_seed) + k
            # Seed global np.random so register_envs's spread-placement draws
            # vary per iter. env_rng is separate so per-env GridEnv seeds match
            # baseline's RandomState(env_seed_k) draw order.
            np.random.seed(env_seed_k)
            env_rng = np.random.RandomState(env_seed_k)
            val_envs = [
                make_env(cfg.env, cfg.agent.movement_mode,
                         seed=int(env_rng.randint(0, 10_000_000)))
                for _ in range(cfg.num_val_envs)
            ]
            # Re-place on the shared field: offsets are a property of this
            # env set, so nothing about gbook / encoded_Phi is touched.
            offsets = place_envs(cfg.num_val_envs, cfg.env.size, vh.Npos,
                                 np.random, placement="spread")

        if n_full_iters > 1:
            print(f"\n=== iter {k + 1}/{n_full_iters}  seed={seed_k}"
                  + (f"  env_seed={env_seed_k}" if env_seed_k is not None else "")
                  + " ===", flush=True)
        if k == 0 or args.env_seed is not None:
            for i, env in enumerate(val_envs):
                print(f"  env {i}: goal={env.goal_location}  "
                      f"offset={offsets[i]}")

        trace, blocks, stored_at_goal_count = run_sequential(
            agent=agent, val_envs=val_envs, vectorhash=vh, env_offsets=offsets,
            cfg=cfg, device=device,
            iters_per_block=args.iters_per_block,
            max_steps=args.max_steps,
            seed=seed_k,
            deterministic=(not args.stochastic_policy),
            oracle_store_at_goal=args.oracle_store_at_goal,
            oracle_lock_store_not_at_goal=args.oracle_lock_store_not_at_goal,
            lock_store_after_goal=args.lock_store_after_goal,
            store_trainer=store_trainer,
            train_store_updates_per_rollout=args.train_store_updates_per_rollout,
        )
        iter_traces.append((trace, blocks))
        iter_env_goals.append([list(env.goal_location) for env in val_envs])
        iter_env_offsets.append([list(offsets[i]) for i in range(len(val_envs))])
        iter_stored_at_goal.append(
            {int(j): int(c) for j, c in stored_at_goal_count.items()}
        )

    trace, blocks = merge_iter_traces(iter_traces)
    run_name = args.run_name or os.path.splitext(os.path.basename(args.out))[0]
    history = {
        "metadata": {
            "model_class": "agenthash",
            "run_name": run_name,
            "n_envs": len(val_envs),
            "env_size": cfg.env.size,
            "iters_per_block": args.iters_per_block,
            "max_steps": args.max_steps,
            "x_axis_label": "outer iteration",
            "raw_metric_is_binary": True,
            "ckpt_path": args.ckpt,
            "num_full_iters": n_full_iters,
            "extra": {
                "encoder_path": encoder_path,
                "static_vectorhash": bool(cfg.vectorhash.static_vectorhash),
                "lock_store_after_goal": bool(args.lock_store_after_goal),
                "oracle_store_at_goal": bool(args.oracle_store_at_goal),
                "oracle_lock_store_not_at_goal": bool(args.oracle_lock_store_not_at_goal),
                "train_store": bool(args.train_store),
                "train_store_lr": float(args.train_store_lr),
                "train_store_updates_per_rollout": int(args.train_store_updates_per_rollout),
                "stochastic_policy": bool(args.stochastic_policy),
                "base_seed": int(args.seed),
                "env_seed_base": int(args.env_seed) if args.env_seed is not None else None,
                "movement_mode": cfg.env.movement_mode,
                "stored_at_goal_count_per_iter": iter_stored_at_goal,
                "env_goals_per_iter": iter_env_goals,
                "env_offsets_per_iter": iter_env_offsets,
            },
        },
        "trace": [
            [s, t, {str(k): v for k, v in inner.items()}]
            for s, t, inner in trace
        ],
        "blocks": [list(b) for b in blocks],
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[agenthash] wrote {args.out}  "
          f"trace={len(trace)} steps  blocks={len(blocks)}")


if __name__ == "__main__":
    main()
