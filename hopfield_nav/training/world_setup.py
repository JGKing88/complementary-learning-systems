"""World construction, head freezing and the phase-boundary eval.

`train_phase_a_only` and `train_phase_b_only` need all four of these, and until
this module they imported them from `train_phased` -- one entry point reaching
into another for its internals, which is why the three could not be reasoned
about separately and why moving any of them meant checking two other CLIs.

None of it is phase-specific: `setup_world` builds envs plus a scaffold,
`make_hops` builds the per-env Hopfield arrangement a phase asks for by name,
`set_phase_freeze` toggles `requires_grad` on the four parameter groups, and
`do_eval` runs the standard evaluator set and logs it. The *choice* of role,
freeze mask and schedule stays in each entry point, because that choice is what
distinguishes them.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from ..policy.agent import NavAgent
from ..config import TrainConfig
from ..world.env import make_env
from ..evaluation.metrics import (
    evaluate_exploration, evaluate_goal_discovery, evaluate_navigation,
)
from hopfield import Hopfield
from ..world import domains as dom
from ..world import generate
from ..world.scaffold import VectorHash, goal_encodings
from ..world.spec import EnvSpec, GeneratedSplit, TraitDomains, WorldSpec
from ..world.world import World, build_world


# ---------------------------------------------------------------------------
# World setup (reused across phases)
# ---------------------------------------------------------------------------

def build_field(cfg: TrainConfig, encoder) -> VectorHash:
    """The scaffold field, built once and shared by every world.

    It is a pure function of ``(lambdas, Npos, fwhm_ratio, encoder)``, so the
    per-world copies this used to make were bit-identical -- and 12 GB each at
    ``Npos=1716, out_dim=1024``.
    """
    field = VectorHash(cfg.vectorhash)
    field.build_scaffold()
    field.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=cfg.device)
    return field


def setup_world(cfg: TrainConfig, encoder, embed_dim, rng, role: str = "train",
                field: VectorHash | None = None) -> World:
    """Build envs and place them in a scaffold. Same for train + eval worlds;
    role just controls count (envs_per_world vs num_val_envs).

    ``field=None`` builds one, which is what a single-world caller wants. Pass a
    field to share it -- the train worlds and the eval world should always share
    one.
    """
    n = cfg.envs_per_world if role == "train" else cfg.num_val_envs
    envs = [
        make_env(cfg.env, cfg.agent.movement_mode,
                 seed=int(rng.randint(0, 10_000_000)))
        for _ in range(n)
    ]
    if field is None:
        field = build_field(cfg, encoder)
    return build_world(field, envs, placement="spread", size=cfg.env.size)


def inherited_used(parent_ckpt: str | None) -> dict | None:
    """The ancestor's trait union, read from its ``world.json``.

    What a run continuing from ``--load_checkpoint`` must place clear of, and
    must record as part of its own union -- otherwise a validation set minted
    later is disjoint from the *last* stage of training only.

    One level up is enough: the parent absorbed its own parent's union when it
    was written, so this is transitively the whole chain. A parent that predates
    world recording returns ``None`` and says so, because "cannot exclude the
    earlier stage" is exactly the thing that must not pass silently.
    """
    import json
    import os

    if not parent_ckpt:
        return None
    src_dir = (parent_ckpt if os.path.isdir(parent_ckpt)
               else os.path.dirname(parent_ckpt))
    src = os.path.join(src_dir, "world.json")
    if not os.path.exists(src):
        print(f"  NOTE: parent {parent_ckpt} has no world.json, so this run "
              "cannot place clear of the envs it was trained on, and its own "
              "record cannot name them. A later --split held_out will be "
              "disjoint from THIS run's envs only. Re-run the parent on current "
              "code to fix that.", flush=True)
        return None
    with open(src) as f:
        parent = WorldSpec.from_json(json.load(f))
    used = parent.split.used
    print(f"  inheriting parent world from {src}: "
          f"{len(used.get('place', ()))} placements, "
          f"{len(used.get('wall', ()))} wall seeds, "
          f"{len(used.get('goal', ()))} goal cells held against this run's draw",
          flush=True)
    return used


def setup_worlds_declared(cfg: TrainConfig, field: VectorHash,
                          inherited: dict | None = None):
    """Train worlds + eval world drawn from declared domains.

    One ``generate_split`` call covers every train env across all worlds plus the
    validation set, so separation is enforced across the whole run at once rather
    than per world. ``num_worlds`` then only chunks the result -- which is all it
    has ever been, now that the scaffold field is shared.

    ``inherited`` is an ancestor's ``used`` union; see ``generate_split``.
    """
    domains = TraitDomains(
        place=dom.parse_place(cfg.place_region),
        wall=dom.parse_seed_range(cfg.wall_seeds),
        goal=dom.parse_goal(cfg.goal_region),
        size=dom.Sizes((int(cfg.env.size),)),
    )
    n_train = int(cfg.envs_per_world) * int(cfg.num_worlds)
    split = generate.generate_split(
        field, cfg.env, domains, n_train, int(cfg.num_val_envs),
        seed=int(cfg.seed), margin=cfg.place_margin,
        val_frac=float(cfg.goal_val_frac),
        # Goal refresh consumes fresh cells every tick, so the train share has
        # to be capped up front. Without this, `n_train` envs claim `n_train`
        # cells per update and a size-8 arena's 64 cells are gone in a handful
        # of updates, leaving no legal held-out goal at all.
        refresh_goal=cfg.refresh_goal is not None,
        inherited=inherited,
    )

    train_envs = generate.build_envs(split.train, cfg.env, cfg.agent.movement_mode)
    per = int(cfg.envs_per_world)
    worlds = [
        build_world(field, train_envs[w * per:(w + 1) * per],
                    offsets=[s.offset for s in split.train[w * per:(w + 1) * per]])
        for w in range(int(cfg.num_worlds))
    ]

    # Eval envs are always built with goals_active=True, so nav and discovery
    # have a goal event to measure -- same reason the legacy path toggles it.
    eval_cfg = dataclasses.replace(cfg.env, goals_active=True)
    val_envs = generate.build_envs(split.base_val, eval_cfg, cfg.agent.movement_mode)
    eval_world = build_world(field, val_envs,
                             offsets=[s.offset for s in split.base_val])
    return worlds, eval_world, split


def specs_from_world(world: World) -> list[EnvSpec]:
    """Read a built world back out as resolved specs.

    Possible because ``GridEnv`` now records its seed. This is what lets the
    legacy placement path emit a truthful ``world.json`` without going through
    the generator at all.
    """
    return [EnvSpec(wall_seed=int(env.seed), size=int(env.size),
                    offset=(int(off[0]), int(off[1])), goal=tuple(env.goal_location))
            for env, off in zip(world.envs, world.offsets)]


def legacy_split(cfg: TrainConfig, field: VectorHash, worlds: list[World],
                 eval_world: World,
                 inherited: dict | None = None) -> GeneratedSplit:
    """Describe an unconstrained draw as a split, so it can still be recorded.

    The domains are the permissive defaults, and ``margin=0`` is the honest
    value: the legacy path enforced no separation whatsoever. What it *achieved*
    lands in diagnostics, which is the interesting part -- it is the first time a
    run says out loud how close its val envs came to its train envs.
    """
    size = int(cfg.env.size)
    train_specs = [s for w in worlds for s in specs_from_world(w)]
    val_specs = specs_from_world(eval_world)
    all_cells = frozenset((x, y) for x in range(size) for y in range(size))
    goals_train = frozenset(s.goal for s in train_specs)
    split = GeneratedSplit(
        domains=TraitDomains(
            place=dom.Anywhere(), wall=dom.SeedRange(0, 10_000_000),
            goal=dom.AnyCells(), size=dom.Sizes((size,))),
        train=train_specs, base_val=val_specs,
        goal_cells_train=goals_train, goal_cells_val=all_cells - goals_train,
        margin=0, period=int(np.prod(field.lambdas)), Npos=int(field.Npos),
    )
    split.record_used(train_specs)
    # The legacy path constrains nothing, so inheriting cannot make this run's
    # placements avoid its parent's. Recording the union still matters: it is
    # what a later `make_val_set` excludes against, and leaving the parent out
    # would have the record understate what training saw.
    if inherited:
        split.absorb_used(inherited)
    return split


def write_world_spec(cfg: TrainConfig, field: VectorHash, split: GeneratedSplit,
                     encoder_ident: dict, *, generator: str,
                     extra: dict | None = None) -> tuple[WorldSpec, str]:
    """Assemble and write ``world.json`` beside the run manifest.

    Written on **both** paths. §1.4's bug is that a checkpoint's val offsets are
    unrecoverable, so every post-hoc eval scores it on patches training never
    used; recording the resolved specs fixes that for every new run, whether or
    not the run opted into declared domains.

    Rewritten on the checkpoint cadence when a run refreshes its envs, because
    ``split.used`` grows with every tick and it is the union -- not the current
    ``train`` list -- that a later ``make_val_set`` excludes against. The file is
    replaced atomically, so the latest checkpoint always matches it; an earlier
    checkpoint's recorded ``spec_hash`` describes a prefix that is no longer on
    disk. ``base_val`` never moves, so every *evaluation* use is unaffected.

    ``extra`` merges into diagnostics -- the refresh report, so the file can say
    whether the world stood still.
    """
    split.diagnostics = generate.split_diagnostics(field, cfg.env, split)
    if extra:
        split.diagnostics.update(extra)
    spec = WorldSpec(
        scaffold={
            "lambdas": list(field.lambdas), "Npos": int(field.Npos),
            "fwhm_ratio": float(cfg.fwhm_ratio),
            "static_vectorhash": bool(cfg.vectorhash.static_vectorhash),
            "encoder": encoder_ident,
        },
        generator=generator, split=split,
    )
    path = spec.write(cfg.save_dir)
    d = split.diagnostics
    print(f"  world.json: generator={generator} margin={split.margin} "
          f"min_place_gap={d.get('min_place_gap')} "
          f"min_wall_hamming={d.get('min_wall_hamming')} "
          f"max_cos={d.get('cosine', {}).get('max')}", flush=True)
    return spec, path


@dataclasses.dataclass
class RunWorld:
    """A run's worlds, the record of them, and the thing that moves them.

    All four entry points need the same five steps: draw a split one of two
    ways, build a refresher if a cadence was asked for, say up front what a
    post-hoc eval will still be able to ask for, write ``world.json``, and
    rewrite it as the union grows. Three copies of that is three chances for a
    trainer to record a world it is not actually training on -- which is the
    failure this whole design exists to make unrepresentable.
    """

    worlds: list[World]
    eval_world: World
    split: GeneratedSplit
    field: VectorHash
    kind: str                        # "declared" | "legacy"
    refresher: object | None         # training.refresh.Refresher
    preflight: dict | None
    cfg: TrainConfig
    encoder_ident: dict
    parent_ckpt: str | None = None
    parent_world: dict | None = None
    extra: dict = dataclasses.field(default_factory=dict)
    _parent_done: bool = False

    def record(self) -> dict:
        """Write ``world.json``; return the summary a checkpoint carries.

        Called again on the checkpoint cadence when the run refreshes, because
        ``split.used`` grows every tick and it is the union -- not the current
        ``train`` list -- that a later ``make_val_set`` excludes against.

        A run with a parent records the parent's world too, on the first call.
        Here rather than in ``setup_run_world`` because the copy has to land in
        ``cfg.save_dir``, which no entry point has resolved that early.
        """
        if self.parent_ckpt and not self._parent_done:
            self._parent_done = True
            self.parent_world = record_parent_world(
                self.cfg, self.split, self.parent_ckpt, self.cfg.env)
            if self.parent_world is not None:
                self.extra["parent"] = self.parent_world
        extra = dict(self.extra)
        if self.refresher is not None:
            extra["refresh"] = self.refresher.report()
            extra["preflight"] = self.preflight
        spec, path = write_world_spec(
            self.cfg, self.field, self.split, self.encoder_ident,
            generator=self.kind, extra=extra or None)
        return spec.summary(path)

    def refresh(self, tick: int) -> tuple[str, ...]:
        """Refresh whatever is due, or nothing when the run has no cadence."""
        if self.refresher is None:
            return ()
        return self.refresher.maybe_refresh(tick)


def setup_run_world(
    cfg: TrainConfig, encoder, embed_dim, rng, field: VectorHash, *,
    cadence, n_updates: int, encoder_ident: dict, where: str,
    parent_ckpt: str | None = None,
) -> RunWorld:
    """Draw a run's worlds, either way, plus everything that hangs off them.

    A falsy ``cadence`` means no refresher and no preflight: a run whose envs
    never move needs neither and pays for neither, and its startup record
    describes it for the whole run.

    Raises when the cadence would kill the run partway through. A *shrinking
    eval ceiling* is recorded and the run proceeds -- a run that only ever
    evaluates on ``--split recorded`` is fine with a tight union, and that is
    not the trainer's call to veto. A domain that *runs dry* is different: the
    run raises hours in, at a tick fixed before it started, so the choice is
    between failing now and throwing away the training in between.
    """
    from .refresh import Refresher, format_preflight, preflight as run_preflight

    # Read before the draw, not after: a continuation has to place clear of what
    # its parent trained on, and `RunWorld.record` fires far too late for that.
    inherited = inherited_used(parent_ckpt)

    if cfg.env_generator:
        worlds, eval_world, split = setup_worlds_declared(cfg, field, inherited)
        kind = "declared"
    else:
        worlds = [setup_world(cfg, encoder, embed_dim, rng, role="train",
                              field=field)
                  for _ in range(cfg.num_worlds)]
        # Eval envs are always built with goals_active=True, so nav and
        # discovery have a goal event to measure. Training envs are left as the
        # caller set them: under an explore regime they may run with no goal
        # reward at all.
        saved_goals_active = cfg.env.goals_active
        cfg.env.goals_active = True
        eval_world = setup_world(cfg, encoder, embed_dim, rng, role="eval",
                                 field=field)
        cfg.env.goals_active = saved_goals_active
        split = legacy_split(cfg, field, worlds, eval_world, inherited)
        kind = "legacy"

    refresher = pre = None
    if cadence:
        refresher = Refresher(cadence, split, worlds, cfg.env,
                              cfg.agent.movement_mode, int(cfg.seed))
        pre = run_preflight(split, cadence, int(n_updates), cfg.env,
                            cfg.agent.movement_mode, int(cfg.seed),
                            n_val_envs=int(cfg.num_val_envs))
        if pre["refresh_dies_at_update"] is not None:
            raise SystemExit(f"  ERROR [{where}]: {format_preflight(pre)}")
        print(format_preflight(pre), flush=True)

    return RunWorld(worlds=worlds, eval_world=eval_world, split=split,
                    field=field, kind=kind, refresher=refresher, preflight=pre,
                    cfg=cfg, encoder_ident=encoder_ident,
                    parent_ckpt=parent_ckpt)


PARENT_SPEC_NAME = "world_parent.json"


def world_overlap(parent: GeneratedSplit, child: GeneratedSplit,
                  env_cfg) -> dict:
    """How much of the parent's world this run's world reuses.

    A continuation that draws its own envs is training and evaluating somewhere
    else, so its eval curve does not join onto its parent's. That is a fact
    about the two worlds and it is cheap to state, which is better than leaving
    a reader to diff two files and conclude it themselves.

    ``val_envs_identical`` is the one that matters: it answers "can I put these
    two curves on the same axes" outright.

    Everything here reads the child's **own** ``train`` rather than its ``used``.
    Since ``generate_split`` began absorbing an ancestor's union, ``child.used``
    contains the parent's envs by construction -- so a union-vs-parent
    comparison reports total overlap always, including a box against itself at
    ``gap = -size``. That is a true statement about the union and a useless one
    about this run: the question is what *this* run trains on.
    """
    p_val = [(v.wall_seed, v.size, v.offset, v.goal) for v in parent.base_val]
    c_val = [(v.wall_seed, v.size, v.offset, v.goal) for v in child.base_val]
    p_boxes = parent.used_boxes()
    c_boxes = sorted({(t.offset, int(t.size)) for t in child.train})
    gaps = [generate.toroidal_gap(a, sa, b, sb, child.period)
            for a, sa in c_boxes for b, sb in p_boxes]
    sizes = {s for _, s in p_boxes} | {s for _, s in c_boxes}
    return {
        "val_envs_identical": p_val == c_val,
        "n_val_envs_shared": len(set(p_val) & set(c_val)),
        "n_wall_seeds_shared": len(parent.used.get("wall", set())
                                   & {t.wall_seed for t in child.train}),
        "n_goal_cells_shared": len(parent.used.get("goal", set())
                                   & {t.goal for t in child.train}),
        "n_train_offsets_shared": len(parent.used_offsets()
                                      & {t.offset for t in child.train}),
        # Nearest approach between anything the parent placed and anything this
        # run places. Below `margin` means the inherited exclusion did not hold;
        # negative means the two train sets overlap on the scaffold.
        "min_place_gap_vs_parent": int(min(gaps)) if gaps else None,
        "sizes": sorted(int(s) for s in sizes),
    }


def record_parent_world(cfg: TrainConfig, split: GeneratedSplit,
                        parent_ckpt: str | None, env_cfg) -> dict | None:
    """Copy the parent run's ``world.json`` beside this run's, and compare.

    Verbatim, byte for byte: the file carries its own ``spec_hash``, and a
    re-serialized copy would be a different file claiming to be the parent's.
    Copying rather than pointing means this run directory answers "what did the
    parent train on" on its own, after the parent has been moved or collected.

    ``None`` when there is no parent or the parent predates world recording --
    which is not an error, just the older world being unrecoverable.
    """
    import json
    import os
    import shutil

    if not parent_ckpt:
        return None
    src_dir = (parent_ckpt if os.path.isdir(parent_ckpt)
               else os.path.dirname(parent_ckpt))
    src = os.path.join(src_dir, "world.json")
    if not os.path.exists(src):
        print(f"  NOTE: parent {parent_ckpt} has no world.json, so this run "
              "cannot record what it trained on. Only the child world is "
              "described. Re-run the parent on current code to fix that.",
              flush=True)
        return None

    os.makedirs(cfg.save_dir, exist_ok=True)
    dst = os.path.join(str(cfg.save_dir), PARENT_SPEC_NAME)
    shutil.copyfile(src, dst)
    with open(dst) as f:
        parent = WorldSpec.from_json(json.load(f))     # verifies the copy
    overlap = world_overlap(parent.split, split, env_cfg)
    print(f"  {PARENT_SPEC_NAME}: parent world copied from {src}; "
          f"val_envs_identical={overlap['val_envs_identical']} "
          f"shared_walls={overlap['n_wall_seeds_shared']} "
          f"min_place_gap_vs_parent={overlap['min_place_gap_vs_parent']}",
          flush=True)
    if not overlap["val_envs_identical"]:
        print("  NOTE: this run drew its own validation envs, so its eval "
              "numbers are not on the same axes as the parent's. Both worlds "
              f"are recorded ({PARENT_SPEC_NAME} and world.json) so the "
              "comparison can at least be made knowingly.", flush=True)
    return {
        "checkpoint": str(parent_ckpt),
        "world_json": dst,
        "spec_hash": parent.spec_hash(),
        "generator": parent.generator,
        "overlap": overlap,
    }


def make_hops(
    role: str,
    cfg: TrainConfig,
    world: World,
    embed_dim: int,
    device: torch.device,
    B: int,
):
    """Build a per-env Hopfield setup for one training env.

    role:
      - "pre_stored_shared": one shared Hopfield per env preloaded with the
        goal. No agent writes (agent_can_store=False semantics). Phase 2.
      - "empty_shared": one shared empty Hopfield per env. No agent writes.
        Phase 3.
      - "empty_per_env": B empty Hopfields, agent can write to each. Phases 1
        and 4.
    Returns a list/Hopfield and a flag for whether this is per-env vs shared.
    """
    envs = world.envs
    if role == "pre_stored_shared":
        per_env_templates = []
        for pattern in goal_encodings(world.field, envs, world.offsets):
            hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            hop.input_memory(torch.from_numpy(pattern).float())
            per_env_templates.append(hop)
        return per_env_templates  # one per env; shared across the B trajectories
    if role == "empty_shared":
        return [Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                for _ in envs]
    if role == "empty_per_env":
        # For each env we build a fresh list of B Hopfields each rollout; here
        # we just return a factory.
        def factory():
            return [Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                    for _ in range(B)]
        return factory
    raise ValueError(f"unknown role: {role}")


# ---------------------------------------------------------------------------
# Freezing utilities
# ---------------------------------------------------------------------------

def set_requires_grad(params, flag: bool):
    for p in params:
        p.requires_grad_(flag)


def move_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    if agent.cfg.movement_mode == "discrete":
        return list(agent.movement_head.parameters())
    return list(agent.movement_mean.parameters()) + [agent.movement_log_std]


def store_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    return list(agent.store_head.parameters())


def value_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    return list(agent.value_head.parameters())


def rnn_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    return list(agent.rnn.parameters())


def set_phase_freeze(agent: NavAgent, freeze_move: bool,
                     freeze_store: bool, freeze_value: bool, freeze_rnn: bool):
    set_requires_grad(move_params(agent), not freeze_move)
    # `movement_log_std` is a movement parameter, so unfreezing the movement
    # head re-enables its gradient and silently undoes `--freeze_log_std` --
    # which every caller does, because no phase freezes movement. The flag was
    # therefore a no-op on train_navigate: gentle-terrain-124's lineage ran
    # with a *learnable* log_std despite asking for a frozen one, visible as
    # std drifting 0.166 -> 0.294 over 250 updates. The agent's own config is
    # the authority; a phase mask must not overrule it.
    if getattr(agent.cfg, "freeze_log_std", False) \
            and hasattr(agent, "movement_log_std"):
        agent.movement_log_std.requires_grad = False
    set_requires_grad(store_params(agent), not freeze_store)
    set_requires_grad(value_params(agent), not freeze_value)
    set_requires_grad(rnn_params(agent), not freeze_rnn)


# ---------------------------------------------------------------------------
# Eval wrapper used at phase boundaries
# ---------------------------------------------------------------------------

def do_eval(cfg, agent, eval_world: World, device, update_tag: str,
            use_wandb: bool, max_steps: int = 200) -> None:
    val_envs = eval_world.envs
    val_vh = eval_world.field
    val_offsets = eval_world.offsets
    dist = cfg.val_n_distractors_list
    nt = cfg.n_val_trials

    # Which evaluators this scope asks for. A skipped one stays in the wandb log
    # as an empty dict rather than as a stale value, so a scope switch mid-project
    # cannot be mistaken for a collapse in nav.
    #
    # "nav_expl" exists because `evaluate_goal_discovery` is the only one of the
    # three that is not batched -- it steps one trial at a time at B=1, so its
    # cost is envs x trials x distractor-levels x max_steps *serial* model
    # calls, two orders of magnitude more than the other two put together (at
    # 10 x 32 x 3 x 400 that is ~10 minutes against ~50 seconds). It measures
    # the store head, which `train_navigate` freezes and never trains, so a run
    # scored on coverage and navigation is paying that for a constant.
    scope = getattr(cfg, "eval_scope", "all")
    run_nav = scope in ("all", "nav_expl")
    run_disc = scope == "all"

    t0 = time.time()
    nav = evaluate_navigation(
        agent, val_envs, val_vh, val_offsets, cfg, device,
        num_trials=nt, max_steps=max_steps,
        n_distractors_list=dist, deterministic=True) if run_nav else {}
    disc = evaluate_goal_discovery(
        agent, val_envs, val_vh, val_offsets, cfg, device,
        num_trials=nt, max_steps=max_steps,
        n_distractors_list=dist) if run_disc else {}
    expl = evaluate_exploration(agent, val_envs, val_vh, val_offsets, cfg, device,
                                num_trials=nt, max_steps=max_steps,
                                n_distractors_list=dist)
    eval_s = time.time() - t0
    if run_nav:
        print(f"  [{update_tag}] nav={nav}")
    if run_disc:
        print(f"  [{update_tag}] disc={disc}")
    print(f"  [{update_tag}] expl={expl}")
    # Sizing a run needs the eval's own cost, not just the per-update total it
    # is folded into -- see docs/EXPERIMENTS_SCHEDULE_REPRO.md on how badly a
    # run can be mis-sized when that number has to be inferred after the fact.
    print(f"  [{update_tag}] eval_seconds={eval_s:.1f} scope={scope}",
          flush=True)
    if use_wandb:
        import wandb
        log = {"eval/eval_seconds": eval_s}
        for n_d in dist:
            for k, v in nav.get(n_d, {}).items(): log[f"eval/nav_{n_d}/{k}"] = v
            for k, v in disc.get(n_d, {}).items(): log[f"eval/disc_{n_d}/{k}"] = v
            # union_coverage / redundancy now arrive inside expl.
            for k, v in expl[n_d].items(): log[f"eval/expl_{n_d}/{k}"] = v
        log["phase_tag"] = update_tag
        wandb.log(log)


__all__ = [
    "build_field", "do_eval", "inherited_used", "legacy_split", "make_hops",
    "move_params",
    "rnn_params", "set_phase_freeze", "set_requires_grad", "setup_world",
    "setup_worlds_declared", "specs_from_world", "store_params",
    "value_params", "write_world_spec",
]
