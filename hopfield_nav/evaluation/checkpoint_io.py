"""Reading a training checkpoint back into a runnable world and agent.

Every driver that re-opens a `.pt` -- the eval CLIs, `train_phase_b_only`,
`analysis.trajectories`, the figure pipelines, `analysis.phase_decoding` -- needs the
same four steps: map legacy field names onto the current schema, rebuild the
`TrainConfig` from the saved `asdict`, replay the val-env seed sequence, and
construct a `NavAgent` of the width that config implies. Until this module those
steps existed in three copies (`eval_all.py`, `eval_checkpoints.py`,
`eval_distractors.py`) and downstream code imported them from whichever copy it
happened to know about, so a checkpoint-compat fix meant editing three files and
remembering who imported which.

Checkpoints are dicts keyed by `config.py`'s dataclass *field names*, and 309
agent run directories are only readable through them. That is why
`coerce_legacy_cfg` exists at all: it carries every rename and removal the
dataclasses have absorbed -- `val_envs_per_world` -> `num_val_envs`,
`gbook_only` -> `static_vectorhash`, `agent_can_store` -> `allow_store`, and
the outright removal of `randomize_goal_per_rollout` -- rather than keeping the
old fields. Any future one lands here as another clause.
"""
from __future__ import annotations

import os

import numpy as np

from ..policy.agent import NavAgent, compute_input_dim
from ..config import (
    AgentConfig, EnvConfig, HopfieldConfig, PPOConfig, TrainConfig,
    VectorHashConfig,
)
from ..world.env import make_env
from ..world.generate import (
    RECORDED, build_envs, levels_key, make_val_set, parse_levels,
    val_set_report)
from ..world.scaffold import VectorHash, fit_env_assoc, place_envs
from ..world.spec import WORLD_SPEC_NAME, WorldSpec


def coerce_legacy_cfg(cd: dict) -> dict:
    """Map legacy config fields onto the current schema, in place.

    Applied to the *saved config dict*, not to CLI arguments -- these are
    renames the dataclasses have already absorbed, replayed for checkpoints
    written before them.
    """
    # val_envs_per_world used to be scoped to training worlds; it is now a
    # single global num_val_envs for the dedicated eval world.
    if "val_envs_per_world" in cd and "num_val_envs" not in cd:
        cd["num_val_envs"] = cd.pop("val_envs_per_world")
    vh = cd.get("vectorhash")
    if isinstance(vh, dict) and "gbook_only" in vh and "static_vectorhash" not in vh:
        vh["static_vectorhash"] = vh.pop("gbook_only")
    # agent_can_store -> allow_store. Without this every checkpoint written
    # before 2026-08 raises on load: cfg_from_checkpoint does
    # HopfieldConfig(**cd["hopfield"]), and an unknown key is a TypeError.
    hop = cd.get("hopfield")
    if isinstance(hop, dict) and "agent_can_store" in hop and "allow_store" not in hop:
        hop["allow_store"] = hop.pop("agent_can_store")
    # randomize_goal_per_rollout -> --refresh_goal. Not a rename: the old flag
    # drew uniformly over the arena for explore-regime envs only, the new one
    # draws world-wide from the declared train cell partition. An unknown
    # top-level key is dropped silently by cfg_from_checkpoint, so a resumed run
    # would lose the behavior without saying so -- hence the warning rather than
    # a mapping, which would also demand --env_generator the parent never had.
    if cd.pop("randomize_goal_per_rollout", False):
        print("  WARNING: this checkpoint was written with "
              "--randomize_goal_per_rollout, which no longer exists. Its "
              "replacement is --refresh_goal N (with --env_generator); the "
              "resumed run holds its goals fixed unless you pass it.",
              flush=True)
    return cd


def cfg_from_checkpoint(ck_cfg_dict: dict) -> TrainConfig:
    """Reconstruct a TrainConfig from a checkpoint's saved ``asdict(cfg)``.

    Keys absent from the saved dict keep their current dataclass defaults, so a
    field added after a checkpoint was written reads as its default rather than
    failing the load.
    """
    cd = coerce_legacy_cfg(dict(ck_cfg_dict))
    env = EnvConfig(**cd["env"])
    vh = VectorHashConfig(**cd["vectorhash"])
    hop = HopfieldConfig(**cd["hopfield"])
    ag = AgentConfig(**cd["agent"])
    ppo = PPOConfig(**cd["ppo"])
    cfg = TrainConfig(env=env, vectorhash=vh, hopfield=hop, agent=ag, ppo=ppo)
    for k, v in cd.items():
        if k in {"env", "vectorhash", "hopfield", "agent", "ppo"}:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def world_spec_for(path) -> WorldSpec | None:
    """The recorded world for a checkpoint or run directory, if it has one.

    ``path`` may be the run directory or any checkpoint inside it.
    """
    p = str(path)
    run_dir = p if os.path.isdir(p) else os.path.dirname(p)
    candidate = os.path.join(run_dir, WORLD_SPEC_NAME)
    if not os.path.exists(candidate):
        return None
    return WorldSpec.read(candidate)


def eval_field(cfg: TrainConfig, encoder, device: str,
               spec: WorldSpec | None = None) -> VectorHash:
    """The scaffold, built once, and checked against the record.

    Separate from the env sets because it is the expensive half -- `encoded_Phi`
    is 12 GB at `Npos=1716` -- and because several validation sets at different
    split levels all index into the *same* field. Rebuilding it per level would
    dominate a mix-and-match evaluation entirely.
    """
    field = VectorHash(cfg.vectorhash)
    field.build_scaffold()
    field.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=device)
    if spec is None:
        return field

    recorded = spec.scaffold
    if int(recorded.get("Npos", field.Npos)) != int(field.Npos):
        raise ValueError(
            f"world.json was written against Npos={recorded['Npos']} but this "
            f"config builds Npos={field.Npos}; the offsets would index a "
            f"different scaffold.")
    enc_now = (encoder_identity_hint(cfg) or {}).get("sha256")
    enc_then = (recorded.get("encoder") or {}).get("sha256")
    if enc_now and enc_then and enc_now != enc_then:
        print(f"  WARNING: world.json was written against encoder "
              f"{enc_then[:12]}... but this run loads {enc_now[:12]}.... The "
              f"envs are the same cells; their embeddings are not.", flush=True)
    return field


def eval_specs(spec: WorldSpec, levels: dict | None, *, n_envs: int,
               seed: int, size: int | None = None):
    """The env specs one split level combination asks for.

    ``levels=None`` returns the recorded ``base_val`` verbatim -- the envs the
    run was actually scored against, offsets and all. Any other level mints a
    fresh set from the recorded domains and the union of what training used.
    """
    if levels is None:
        return list(spec.split.base_val)
    return make_val_set(spec.split, n_envs, levels, seed, size=size)


def eval_env_set(cfg: TrainConfig, encoder, device: str, *, ckpt_path,
                 levels: dict | None = None, val_seed: int = 0,
                 n_envs: int | None = None, spec: WorldSpec | None = None,
                 field=None) -> dict:
    """One evaluation env set at one split level: ``{key, envs, field, offsets,
    report}``.

    The single-set entry point every eval CLI shares, so "what does
    ``--split place=ood`` mean" has one answer rather than one per driver.
    ``eval_all`` loops it for the multi-combination table, passing ``spec`` and
    ``field`` so the 12 GB scaffold is built once.

    ``levels=None`` is the recorded ``base_val`` -- the envs this checkpoint was
    actually scored against. Deliberately not a synonym for all-``held_out``,
    which mints a fresh set that is also disjoint from training.
    """
    if spec is None:
        spec = world_spec_for(ckpt_path)
    if spec is None:
        if levels is not None:
            raise SystemExit(
                f"--split needs a world.json, and {ckpt_path} has none: it was "
                "written before runs recorded their envs. Minting a level needs "
                "the declared domains and the union training used, and an RNG "
                "replay recovers neither. Re-run training on the current code.")
        envs, built, offsets = build_eval_world(cfg, encoder, device,
                                                ckpt_path=ckpt_path)
        return {"key": RECORDED, "envs": envs, "field": built,
                "offsets": offsets, "report": {}}

    if field is None:
        field = eval_field(cfg, encoder, device, spec)
    n = int(n_envs if n_envs is not None else cfg.num_val_envs)
    specs = eval_specs(spec, levels, n_envs=n, seed=val_seed)
    # A minted set's separation is the claim the evaluation is making, so it is
    # measured and returned rather than assumed. Raises on a held_out/ood
    # violation -- that would be a generator bug, and a number in a report is
    # not a strong enough place to put one.
    report = val_set_report(spec.split, specs, cfg.env, levels)
    return {
        "key": levels_key(levels),
        "envs": build_envs(specs, cfg.env, cfg.agent.movement_mode),
        "field": field,
        "offsets": [s.offset for s in specs],
        "report": report,
    }


def eval_world_for_split(cfg: TrainConfig, encoder, device: str, *, ckpt_path,
                         split: str = RECORDED, val_seed: int = 0,
                         n_envs: int | None = None):
    """``(envs, field, offsets)`` for one ``--split`` string.

    The drop-in for `build_eval_world` in a CLI that evaluates one env set.
    """
    es = eval_env_set(cfg, encoder, device, ckpt_path=ckpt_path,
                      levels=parse_levels(split), val_seed=val_seed,
                      n_envs=n_envs)
    if es["key"] != RECORDED:
        r = es["report"]
        print(f"  split={es['key']}: {len(es['envs'])} envs, "
              f"place_gap>={r.get('min_place_gap_vs_train')} "
              f"(margin {r.get('margin')}), wall_hamming>="
              f"{r.get('min_wall_hamming_vs_train')}", flush=True)
    return es["envs"], es["field"], es["offsets"]


def eval_world_from_spec(spec: WorldSpec, cfg: TrainConfig, encoder,
                         device: str, *, which: str = "base_val"):
    """Rebuild an env set exactly as recorded -- no RNG replay anywhere.

    This is the point of `world.json`. The replay path below can recover a run's
    val wall codes and goals but *not* their offsets, because placement drew from
    global `np.random` whose state depended on everything built before it. Here
    the offsets are read, not re-derived, so what you evaluate is what trained.
    """
    specs = getattr(spec.split, which)
    field = eval_field(cfg, encoder, device, spec)
    envs = build_envs(specs, cfg.env, cfg.agent.movement_mode)
    offsets = [s.offset for s in specs]
    return envs, field, offsets


def encoder_identity_hint(cfg: TrainConfig) -> dict | None:
    """sha256 of the encoder this config points at, if it is still on disk."""
    try:
        import run_manifest
        return run_manifest.encoder_identity(cfg.encoder_checkpoint)
    except Exception:
        return None


def build_eval_world(cfg: TrainConfig, encoder, device: str,
                     spec: WorldSpec | None = None, ckpt_path=None):
    """Rebuild the eval world a checkpoint was trained against.

    Pass ``ckpt_path`` -- a checkpoint or its run directory -- and the recorded
    world is found and used. That is the point of the parameter: between Phase 3
    and now, every caller here passed three arguments, so every post-hoc
    evaluation took the replay branch below and scored checkpoints on scaffold
    patches training never used, while a truthful ``world.json`` sat unread
    beside them. Discovery lives *here* rather than at the four call sites
    precisely so a fifth cannot quietly rejoin the legacy path.

    ``spec`` given explicitly wins, for callers evaluating one run's checkpoint
    against another run's world.

    With neither, this replays the training-time seed stream: training draws its
    train-env seeds first, then its val-env seeds, from one
    ``RandomState(cfg.seed)``, and the skip loop below reproduces that order.
    That recovers wall codes and goals -- but **not offsets**, which came from
    global ``np.random`` (§1.4). Only pre-Phase-3 runs land here.
    """
    if spec is None and ckpt_path is not None:
        spec = world_spec_for(ckpt_path)
    if spec is not None:
        return eval_world_from_spec(spec, cfg, encoder, device)
    print(
        "  NOTE: no world.json for this checkpoint, falling back to the RNG "
        "replay. Wall codes and goals are exact; **env offsets are not** -- "
        "placement drew from global np.random, so these are a fresh draw, not "
        "the ones training evaluated against (measured deltas up to half an env "
        "width). Re-run training on the current code to get a recorded world.",
        flush=True)
    rng = np.random.RandomState(cfg.seed)
    size = cfg.env.size
    # Skip train-env seeds to keep val-env seeds aligned with training.
    for _ in range(cfg.envs_per_world * cfg.num_worlds):
        rng.randint(0, 10_000_000)
    val_envs = [
        make_env(cfg.env, cfg.agent.movement_mode,
                 seed=int(rng.randint(0, 10_000_000)))
        for _ in range(cfg.num_val_envs)
    ]
    vh = VectorHash(cfg.vectorhash)
    vh.build_scaffold()
    vh.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=device)
    offsets = place_envs(cfg.num_val_envs, size, vh.Npos, np.random,
                         placement="spread")
    fit_env_assoc(vh, val_envs, offsets)
    return val_envs, vh, offsets


def load_agent(
    cfg: TrainConfig,
    state_dict: dict | None,
    embed_dim: int,
    device,
    *,
    eval_mode: bool = True,
) -> NavAgent:
    """Build the NavAgent this config implies and load ``state_dict`` into it.

    The input width comes from `compute_input_dim`, which derives it from the
    same channel specs the observation is assembled from -- constructing the
    agent any other way risks a policy whose first layer disagrees with the
    layout it is fed.

    ``state_dict=None`` leaves the freshly-initialized weights alone, for the
    random-agent control in `phase_decoding_v2`. Callers that want a specific
    init seed the RNG before calling: nothing here consumes it.
    """
    input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    agent = NavAgent(cfg.agent, input_dim).to(device)
    if state_dict is not None:
        agent.load_state_dict(state_dict)
    if eval_mode:
        agent.eval()
    return agent


def scaffold_layout_dict(
    cfg: TrainConfig,
    vh: VectorHash,
    val_envs: list,
    env_offsets: list[tuple[int, int]],
) -> dict:
    """Serializable layout: Npos×Npos grid indices, env footprints, goals.

    ``cfg.vectorhash.Npos`` (when not None) is the checkpoint override; ``Npos``
    is the resolved size used by ``VectorHash`` (same as training when the
    checkpoint was saved with that config).
    """
    prod_lambdas = int(np.prod(cfg.vectorhash.lambdas))
    envs_out: list[dict] = []
    for i in range(len(val_envs)):
        off = env_offsets[i]
        g = val_envs[i].goal_location
        ox, oy = int(off[0]), int(off[1])
        gl0, gl1 = int(g[0]), int(g[1])
        envs_out.append({
            "idx": i,
            "offset": [ox, oy],
            "goal_local": [gl0, gl1],
            "goal_global": [gl0 + ox, gl1 + oy],
        })
    return {
        "Npos": int(vh.Npos),
        "Npos_config": cfg.vectorhash.Npos,
        "prod_lambdas": prod_lambdas,
        "lambdas": list(cfg.vectorhash.lambdas),
        "static_vectorhash": bool(cfg.vectorhash.static_vectorhash),
        "env_size": int(cfg.env.size),
        "placement": "spread",
        "envs": envs_out,
    }


__all__ = [
    "build_eval_world",
    "eval_env_set",
    "eval_field",
    "eval_specs",
    "eval_world_for_split",
    "eval_world_from_spec",
    "cfg_from_checkpoint",
    "coerce_legacy_cfg",
    "load_agent",
    "scaffold_layout_dict",
    "world_spec_for",
]
