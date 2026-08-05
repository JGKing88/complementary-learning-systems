# Refactor Assessment

Written 2026-08-05 against `main` (`959322f`). Verdict, evidence, a target
layout, and a phased plan. No code was changed to produce this document.

---

## 1. Verdict

**Refactor in place. Do not migrate to a new repository.**

The reason is not sentiment about the code — it's that a migration's main
promise (a clean slate) is unavailable here. Two things pin you to the current
structure:

1. **309 agent run directories and 870 encoder run directories are only
   readable through the current config dataclasses.** `make_cfg_from_checkpoint` rebuilds
   `TrainConfig` by *field name* from `asdict(cfg)` stored inside each `.pt`.
   Any rename or restructure of `EnvConfig` / `VectorHashConfig` /
   `HopfieldConfig` / `AgentConfig` / `PPOConfig` fields silently changes what a
   re-evaluated checkpoint means, or requires another shim (you already carry
   two: `val_envs_per_world → num_val_envs` and `gbook_only →
   static_vectorhash`). A new repo would either inherit the same dataclasses —
   in which case the slate isn't clean — or orphan every existing result.
2. **The experiment record lives in the repo**, in
   `run_phase_a_sweep_evelina.sh`'s 101-variant `case` block and in the wandb
   run names that double as checkpoint directory names. Porting selectively
   would break the ability to say "v18d39_size20_v34 was exactly these flags".

Meanwhile the actual code quality is better than the directory listing suggests.
The *library* layer of `hopfield_nav` is genuinely modular: env, vectorized env,
scaffold, memory, policy, rollout, PPO, BC are separate files with clean
interfaces and unusually good docstrings that record *why* choices were made.
The mess is concentrated in four specific places, all fixable incrementally:

| Problem | Where |
|---|---|
| Five training entry points re-implementing one loop | `train.py`, `train_phased.py`, `train_phase_a_only.py`, `train_phase_b_only.py`, `train_rnn.py` |
| Three eval drivers with copy-pasted checkpoint loading | `eval_all.py`, `eval_checkpoints.py`, `eval_distractors.py` |
| The experiment registry is a 785-line bash `case` | `run_phase_a_sweep_evelina.sh` |
| 75 GB of outputs interleaved with source | `encoders/`, `checkpoint*/`, `wandb/`, `hopfield_nav/{final_plotting,phase_decoding*}/…` |

A fifth, quieter problem: `cls/` is 5,900 lines of which ~1,700 are live, and it
contains a *second, incompatible* `Hopfield` class.

---

## 2. Evidence

### 2.1 Duplication with real divergence risk

**(a) Policy-input assembly exists in three places and must stay byte-identical.**

| Site | Lines |
|---|---|
| `rollout.py:374-397` (per-step) | training input |
| `rollout.py:594-618` (bootstrap) | value-bootstrap input |
| `eval.py:173-209` (`agent_step`) | every evaluator, every figure pipeline |

All three concatenate
`[current_reward, prev_reward?, embedding?, hopfield signal?, multistep q?, prev_action?, sensory?, goal_in_memory?]`
in that exact order, each gated by the same `AgentConfig` flags, with
`compute_input_dim` as a fourth, separate statement of the same layout. If any
one drifts, tensor shapes still match — only the *meaning* of the channels
changes, and nothing raises. This is the single most dangerous duplication in
the repo, because the failure mode is "eval numbers are quietly wrong".

**(b) Checkpoint loading is triplicated.**
`_coerce_legacy_cfg` + `make_cfg_from_checkpoint` + `build_eval_world` appear in
`eval_all.py` (83 lines), `eval_checkpoints.py` (48), `eval_distractors.py` (53).
Downstream consumers then import from inconsistent places:
`train_phase_b_only.py` takes `make_cfg_from_checkpoint` from `eval_checkpoints`,
while `final_plotting/agenthash.py` and `phase_decoding_v2/rollout.py` take it
from `eval_all`. Fixing a checkpoint-compat bug currently means editing three
files and remembering who imports which.

**(c) Distractor sampling is written four times.**
`eval._sample_distractor_goals` (used by all evaluators and
`visualize_trajectories` and `phase_decoding_v2`), plus three inline
`while placed < n: rx, ry = randint...; if inside env patch: continue` loops in
`train.py:279-286` and `train_phase_a_only.py:254-261` and `:286-294`. The
training copies and the eval copy must agree or the "training distribution
matches eval distribution" argument fails.

**(d) The sequential mini-episode is written twice.**
`eval.evaluate_sequential_episodes._mini_episode` and
`final_plotting/agenthash.mini_episode` implement the same protocol; the second
adds `steps_to_goal`/`path_to_goal` recording and splits the store oracle into
two flags. `agenthash.py`'s own docstring documents the divergence — which means
the paper figure and the training-time metric are produced by two different
implementations of "the same" protocol.

**(e) Smaller ones.** `_make_vec` in both `train_rnn.py:88` and
`eval_rnn.py:21`. Two `load_encoder` functions with different signatures
(`hopfield_nav/encoder.py` → `(encoder, cfg, gain)`;
`encoder_training/train.py` → `(encoder, ckpt)`). Two `Hopfield` classes whose
`recall()` return types differ (`cls/hopfield.py` returns `(x, cos_sims)`,
`hopfield_nav/hopfield.py` returns `x`).

### 2.2 The five training entry points

They differ in *what varies per update*, not in structure. Every one of them:
loads the encoder → builds worlds → builds the agent → loops {build Hopfields,
collect rollouts over envs, run one pooled update, log, periodically eval and
checkpoint}. The differences are:

| | Hopfield per env | Frozen heads | Schedules | Update fn |
|---|---|---|---|---|
| `train.py` | per-traj (or shared if `agent_can_store=False`), optional pre-stored template, optional distractors | none | aux anneal, ε anneal, env refresh | ppo **or** bc |
| `train_phased` ph1 | per-env empty + force store | move, value, RNN | — | hand-written BCE |
| `train_phased` ph2/3/4 | `pre_stored_shared` / `empty_shared` / `empty_per_env` | store (2,3) | auto-nav warmup | ppo |
| `train_phase_a_only` | per-env, regime-dependent, distractor curriculum | store | novelty, ε, interleave, log-σ, distractor anneals | ppo |
| `train_phase_b_only` | `empty_shared` | move, value, RNN | — | ppo (only store gradients live) |
| `train_rnn` | none | none | — | bc_rnn |

That is one loop with a per-phase policy object, not five programs. The cost of
the current shape is concrete: `train_phase_a_only` had to re-implement the
rollout-collection loop rather than call `train_phased.collect_one_update`,
because it needs per-env regime decisions; and `train.py` never got
`persistence_bonus`, `novelty_scale_remaining`, `input_hopfield_multistep` or the
distractor curriculum, which exist only in the Phase-A path.

### 2.3 Config surface vs CLI surface

`TrainConfig` and friends expose ~120 fields; `train.py` surfaces 77 flags,
`train_phase_a_only` 73, and they overlap only partially. Consequences visible
in the code:

- PPO internals (`gamma`, `gae_lambda`, `vf_coef`, `ppo_epochs`,
  `n_minibatches`, `max_grad_norm`, `bce_detach_trunk`, `bce_pos_weight_cap`)
  are not settable from **any** training CLI except through the phase presets.
- `train_phase_a_only` needs three bespoke flags (`--move_ent_coef`,
  `--ppo_clip_coef`, `--time_penalty`) that exist purely to poke a single
  dataclass field after construction — a symptom of the CLI being hand-mirrored
  rather than generated from the dataclasses.
- Dead config: `PhasedConfigV2` (no importer), `HopfieldConfig.init_mode` (read
  in exactly one place), `n_train_distractors` (only `train.py`).

### 2.4 Outputs inside the source tree

`hopfield_nav/final_plotting/` is 21 GB, `hopfield_nav/phase_decoding_v2/` is
17 GB — because `histories/`, `scaffold_cache/`, `figures/`, `results/` live
*inside the python packages*. `.gitignore` hides them, which is why this hasn't
hurt yet, but it means `du`, `grep -r`, IDE indexing, and any future `pip
install -e .` all walk tens of gigabytes. It also produced
`hopfield_nav/phase_decoding/` — a 1.1 GB v1 pipeline that exists on disk with
no git history at all, so its provenance is unrecoverable.

### 2.5 Correctness items worth fixing regardless of any refactor

1. `--gbook-only` is passed by `run_eval_all.sh`, `run_continuous.sh` and
   `run_new_sweep.sh`; no parser accepts it (renamed to `--static-vectorhash`).
   All three fail at argparse.
2. `train.py:55-59` builds training envs without `goals_active` / `goal_reward` /
   `goal_radius`, so those three `EnvConfig` fields apply only to eval envs in
   that script. Same for `train_rnn.build_envs` and `goal_radius`.
3. `GridEnv.clone()` calls a `@staticmethod` that raises `NotImplementedError`
   (`env.py:313` → `env.py:324`). Currently unreachable (no caller), but it is a
   trap for anyone who tries to use it.
4. `eval_checkpoints.py` / `eval_distractors.py` hardcode
   `/home/jackking/cls/checkpoints/...` paths and an April-2026 encoder.
5. `bc_update` requires `RolloutBatch.trust_hop_mask`, whose dataclass default
   is `None`, and cats it unconditionally (`bc.py:127`). The live BC path always
   populates it, but the guard is missing and
   `test_audit.py::TestBCStoreCap::test_cap_active` fails because of it — the
   only failing test in the suite (135 pass, 1 fails, verified 2026-08-05).
6. `phase_decoding_v2/README.md` references `hopfield_nav/run_tests.sh`, which
   does not exist; there is no pytest config or `conftest.py` anywhere.
7. A live `WANDB_API_KEY` is committed in plaintext in ~10 sbatch scripts.

---

## 3. Proposed target layout

Deliberately conservative: same top-level package names (so `python -m
hopfield_nav.train …` still resolves and muscle memory survives), restructured
internals, outputs evicted.

```
cls/                                  repo root
├── pyproject.toml                    install all three packages, not just cls
├── docs/
├── gridcode/                         ← extracted live parts of cls/vectorhash
│   ├── codebook.py                   gen_gbook_2d
│   ├── smoothing.py                  smooth_g / smooth_gbook  (from encoder_training.utils)
│   └── assoc.py                      nonlin, train_pbook, train_gcpc, pseudotrain_W{sp,ps}
├── encoder_training/
│   ├── config.py  data.py  losses.py  models.py  train.py
│   ├── nav_eval/                     ← absorbed from cls/eval/nav_eval.py + cls/nav.py + cls/hopfield.py
│   └── experiments/
├── hopfield_nav/
│   ├── config.py                     UNCHANGED field names (checkpoint compat)
│   ├── world/       env.py  vec_env.py  scaffold.py(VectorHash)  memory.py(Hopfield)
│   ├── policy/      agent.py  agent_rnn.py
│   ├── rollout/     collector.py  inputs.py  shaping.py  distractors.py  oracles.py
│   ├── updates/     ppo.py  bc.py  bc_rnn.py
│   ├── evaluation/  protocols.py  metrics.py  checkpoint_io.py  drivers.py
│   └── training/    loop.py  phases.py  cli.py
├── experiments/                      ← replaces the bash case statement
│   ├── phase_a/v18d39_size20_v34.yaml
│   ├── encoder/binary_sweep.yaml
│   └── launch.py                     builds the command, submits sbatch
├── analysis/                         ← moved out of hopfield_nav
│   ├── continual/   (was final_plotting)
│   ├── phase_decoding/  (v2)
│   └── schematics/  (was figures)
├── tests/                            hopfield_nav/tests/* + new smoke tests
└── runs/  → symlink to $CLS_RUNS outside the repo
        encoders/  agent_ckpts/  histories/  scaffold_cache/  results/  wandb/
```

Key structural moves:

- **`rollout/inputs.py` owns the policy-input contract.** One function
  `build_policy_input(cfg, ...) -> (tensor, layout)` used by the collector, the
  bootstrap, and `agent_step`; `compute_input_dim` derived from the same layout
  description instead of restating it. This is the highest-value single change.
- **`evaluation/checkpoint_io.py`** holds `coerce_legacy_cfg`,
  `cfg_from_checkpoint`, `build_eval_world`, `load_agent` — one copy, imported
  by every driver and every analysis pipeline.
- **`training/loop.py`** holds the update loop; `training/phases.py` holds a
  `PhaseSpec` (which Hopfield role per env, which heads frozen, which schedules,
  which update function). The five entry points become five `PhaseSpec` lists
  behind one CLI.
- **`experiments/`** is config-first: each past variant becomes a YAML file
  whose keys are dataclass field names. `launch.py` renders flags and submits.
  Keep `run_phase_a_sweep_evelina.sh` frozen in the repo as the historical
  record.
- **`cls/` disappears from `main`** once `gridcode/` and `encoder_training/nav_eval/`
  absorb the six live functions plus the encoder-side nav eval. Archive it on a
  branch/tag (`legacy-cls`) together with the root `tests/` that only exercise
  it.

---

## 4. Phased plan

Ordered so that each phase is independently valuable and independently
revertible. Phases 0–2 are the ones I'd actually do; 3–5 only if the project
continues past the current paper.

### Phase 0 — stop the bleeding (½ day, no behavior change)
1. Fix `--gbook-only` in the three sbatch scripts (or add
   `--gbook-only` as a hidden alias of `--static-vectorhash` in both parsers —
   cheaper if old scripts must keep working).
2. Pass `goals_active` / `goal_reward` / `goal_radius` into `GridEnv` in
   `train.py:setup_train_world` and `goal_radius` in `train_rnn.build_envs`, or
   delete the flags from those two CLIs. **This changes results** for any run
   that set them, so decide which way deliberately and note it in the sweep
   script.
3. Delete `GridEnv._random_position_with_rng` and the first `_random_position`;
   either fix or delete `clone()`.
4. Guard `bc_update` against a `None` `trust_hop_mask` (one line), which fixes
   the one failing test. Then add `[tool.pytest.ini_options]` + a `run_tests.sh`
   so the suite is runnable by name, and restore the reference in
   `phase_decoding_v2/README.md`.
5. Rotate the wandb key; move it to `~/.netrc` and strip it from the scripts.

### Phase 1 — de-duplicate the dangerous parts (2–3 days)
6. Extract `hopfield_nav/rollout/inputs.py` with `build_policy_input` +
   `input_layout`; rewrite `rollout.py`'s two call sites and `eval.agent_step`
   to use it. Regression test: for a fixed config and fixed state, the three old
   paths and the new one must produce identical tensors (a characterization test
   in the style of the existing `test_at_goal.py`).
7. Extract `evaluation/checkpoint_io.py`; delete the three copies; repoint
   `train_phase_b_only`, `agenthash`, `phase_decoding_v2.rollout`.
8. Extract `sample_distractors(vh, offset, size, n, rng)` into one module and
   call it from `eval.py` and both training paths.
9. Unify `mini_episode`: have `agenthash.py` call the `eval.py` protocol with a
   richer return type, rather than keeping a parallel implementation.
10. Delete `PhasedConfigV2`; either wire `init_mode` into the rollout or drop it.

### Phase 2 — evict the outputs (1 day)
11. Introduce `CLS_RUNS` (env var, default `$HOME/cls_runs`); change default
    `save_dir`s and the analysis pipelines' default output roots to point under
    it; symlink `runs/` in the repo for convenience. Move existing directories
    with `mv`, leaving symlinks behind for a transition period so old absolute
    paths in scripts keep resolving.
12. Decide the fate of `hopfield_nav/phase_decoding/` (v1, untracked, 1.1 GB):
    either commit it as `analysis/phase_decoding_v1/` or archive and delete it.

### Phase 3 — one training loop (3–5 days)
13. `training/phases.py`: `PhaseSpec(name, n_updates, hopfield_role, freeze,
    lr, schedules, update_fn)`. Port `train_phase_a_only`'s per-env regime logic
    as a `RegimeSpec` inside it.
14. `training/loop.py`: run a list of `PhaseSpec`s; a single eval + checkpoint
    path.
15. Re-express the five entry points as thin CLIs over that loop, keeping the
    existing flag names so the sweep script keeps working. Validate by
    reproducing one known Phase-A variant end-to-end (same seed) and diffing
    the logged metrics.

### Phase 4 — config-first experiments (2 days)
16. Generate the CLI from the dataclasses (e.g. one `--set section.field=value`
    mechanism plus the ~20 shorthand flags people actually type), so new config
    fields stop needing hand-mirrored flags.
17. Convert the ~10 sweep variants you still care about into
    `experiments/phase_a/*.yaml`; freeze the bash script as history.

### Phase 5 — retire `cls/` (1–2 days)
18. Move `gen_gbook_2d`, `nonlin`, `train_pbook`, `train_gcpc`,
    `pseudotrain_Wsp`, `pseudotrain_Wps` into `gridcode/`; move
    `cls/eval/nav_eval.py`, `cls/nav.py`, `cls/hopfield.py` into
    `encoder_training/nav_eval/` (and reconcile the two `Hopfield` classes:
    give the encoder-side one the same `recall()` contract, or make the
    encoder-side path use `hopfield_nav`'s).
19. Tag `legacy-cls`, delete `cls/` and the root `tests/` from `main`, update
    `pyproject.toml` to install all packages.

### If you only do three things
Phase 0 items 1–2 (the broken flags and the silently-ignored env config),
Phase 1 item 6 (one policy-input builder), Phase 2 item 11 (outputs out of the
package tree). Those remove the two ways this codebase can currently produce
*quietly wrong* numbers, and the one thing that makes the repo unpleasant to
work in.

---

## 5. What not to do

- **Don't rename `config.py` dataclass fields.** Every checkpoint is a dict
  keyed by those names. If a rename is unavoidable, add it to a single
  `coerce_legacy_cfg` (one copy, after Phase 1) with a comment naming the date
  and the runs affected.
- **Don't rewrite `eval.py`'s protocols while the paper figures are in flight.**
  The at-goal semantics ("the reach is the step where the agent *sits* on the
  goal, pre-action") are subtle, replicated deliberately across seven
  evaluators, and pinned by `tests/test_at_goal.py`. Consolidate the *plumbing*
  (input assembly, checkpoint loading, distractor sampling) and leave the
  protocol semantics alone.
- **Don't merge the RNN baseline into the Hopfield stack.** It is a control; its
  value comes from sharing only the environment. The duplicated `_make_vec` is a
  fair price.
- **Don't delete `run_phase_a_sweep_evelina.sh`**, even after moving to YAML
  configs — it is the only record of what 101 named runs actually were.
- **Don't "clean up" the long docstrings** in `config.py`, `rollout.py` and
  `ppo.py`. They encode experimental findings (why BCE is detached, why ε steps
  are masked out of the surrogate, why novelty and store_bonus conflict) that
  exist nowhere else in executable form.
