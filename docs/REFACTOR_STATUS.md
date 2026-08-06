# Refactor status — 2026-08

Handoff note for the 2026-08 refactor. The original plan lives at
`~/.claude/plans/ok-that-sounds-good-drifting-whale.md`; **it is stale in the
ways listed under "Plan corrections" below.** Read this file first.

Branch `refactor/2026-08`, 29 commits from tag `pre-refactor-2026-08` on `main`.
Suite: 313 passing, tree clean, all 30 entry points import.

**Phases 0-2 and 4-7 are done. Phase 3 (one training loop) is partly done:
`train_navigate` was taken apart along its schedule on 2026-08-06 (see below);
the other four trainers still each own their own loop.**

```bash
./run_tests.sh
```

```bash
python scripts/check_entry_points.py
```

---

## Done

| Phase | What landed |
|---|---|
| 0 | pytest config + `run_tests.sh`; the one failing test fixed; 3 scripts that died at argparse; dead `GridEnv.clone`; `.gitignore` newline; wandb key stripped from 4 scripts |
| 1 | **64 GB → 600 MB.** 25 output dirs moved to `/orcd/pool/003/jackking/cls_runs`, verified by file count, symlinked back. `cls_paths.py` + `scripts/cls_env.sh` |
| 2 | `train.py` uses `make_env`, so `--goal_radius` reaches training envs; `--allow_offcell_store` added |
| 2b | `allow_offcell_store` defaults **False** — a store at goal writes the goal cell, not the neighbour |
| 3 | Golden fixtures (5 files) + `<60s` end-to-end smoke test |
| 4a | `channels.py` — one policy-input layout, `compute_input_dim` derived from it |
| 4b | `signal.py` — one Hopfield-signal implementation |
| 4c | `evaluate_navigation` batched: **13–18× faster**, per-trial *and* aggregate identical |
| 5a | `world/episode.py` — the at-goal contract as a value, declared per site |
| 5b | Inert at-goal move dropped from `move_loss` (**behavior change**) |
| 5c | `evaluation/protocols.py` — one sequential protocol, −106 lines |
| 5d | `training/rnn_sequential.py` + `vec_env.make_vec` — one RNN block loop, −66 lines |
| 6-prep | Smoke tests for the four entry points phase 6 moved |
| 6b | `evaluation/checkpoint_io.py` — one checkpoint-loading path, 5 importers repointed, `eval_checkpoints.py` + `eval_distractors.py` deleted |
| 6b | `training/world_setup.py` — world setup leaves `train_phased`; four privates promoted |
| 6c/6d | Strays deleted, experiment record archived to `docs/archive/`, bulk artifacts to `$CLS_RUNS` |
| 6a/6e | Cycles broken, layered tree, `tests/test_layering.py` |
| 6f | The three figure generators the move left behind go to `analysis/`; two more layering rules |
| 7a | `gridcode/` — `gen_gbook_2d`, `smooth_g*`, the assoc trainers. **Last back-edge removed** |
| 7b | `hopfield/` — one memory model for both stacks; `encoder_training/nav_eval/` absorbs the encoder nav eval |
| 7c | `cls/` and the root `tests/` deleted (tag `legacy-cls`); `pyproject.toml` installs all five packages |
| 8 | `evaluate_goal_discovery` takes the training contract: teleport on arrival (**behavior change**) |
| 8b | `evaluate_exploration` takes `NO_GOALS`, absorbs `evaluate_union_coverage`, and is batched (**behavior change**) |

Each dedup was verified by running the pre-change code in a `git worktree` and
diffing the output, not by trusting a green suite. All byte-identical except
where noted.

---

## The layout, as it now stands

```
gridcode/      codebook smoothing assoc     grid codes + associative trainers
hopfield/      core                         the memory model, shared
hopfield_nav/
  config.py  utils.py  encoder_io.py       leaves; anything may import them
  world/       env vec_env scaffold episode
  policy/      agent agent_rnn channels
  rollout/     collector rnn signal oracles distractors types
  updates/     ppo bc bc_rnn
  evaluation/  metrics rnn protocols batched checkpoint_io
  training/    world_setup rnn_sequential stages explore exploit
  tests/
  train.py train_phased.py train_navigate.py train_store.py
  train_rnn.py eval_all.py                        <- the six CLIs
analysis/
  trajectories.py  encoder_sweep.py
  continual/ phase_decoding/ schematics/ scaffold_experiments/
encoder_training/
  nav_eval/                                 the encoder's own nav metric
```

`cls/` is gone. Everything live was extracted first; the package itself is at
tag `legacy-cls` along with the root `tests/` that only exercised it.

`hopfield_nav/tests/test_layering.py` enforces five rules by AST walk: no
upward imports at module scope; `encoder_training` never imports
`hopfield_nav`; no `_`-prefixed name crosses a module boundary; no module-scope
`import matplotlib` outside `analysis/` (that means a figure generator is filed
in a library package); and nothing imports a CLI. `tests/` is exempt from all
five, and `cls/` from the matplotlib rule until phase 7 deletes it. A function-scoped upward import needs a row in
`DEFERRED_UPWARD_IMPORTS` (one entry: `train_rnn` → `analysis.continual.plotting`,
so matplotlib is not an import-time dependency of every training job), and the
test fails if that list goes stale. Two further tests fail if a new package has
no layer, so the table cannot be dodged by adding a directory.

**Invocation changed only for the analysis pipelines.** The six CLIs are still
`python -m hopfield_nav.train_phase_a_only` etc.; every sbatch driver and
`run_phase_a_sweep_evelina.sh`'s 101 variants are untouched. What moved:

| was | is |
|---|---|
| `hopfield_nav.final_plotting.X` | `analysis.continual.X` |
| `hopfield_nav.phase_decoding_v2.X` | `analysis.phase_decoding.X` |
| `hopfield_nav.visualize_trajectories` | `analysis.trajectories` |
| `hopfield_nav.viz_sensory` | `analysis.schematics.sensory_input` |
| `encoder_training.plot_sweep` | `analysis.encoder_sweep` |

---

## Plan corrections

Things the plan asserts that turned out to be wrong. **Do not follow the plan
on these points.**

1. **`paths.py` cannot live in `hopfield_nav`.** The plan has
   `encoder_training/{config,sweep}.py` importing it while also forbidding
   `encoder_training → hopfield_nav`. It is top-level `cls_paths.py`.
2. **Phase 2's dated note was misdirected.** `train.py` has no
   `--goals_active` flag at all, and the sweep runs `train_phase_a_only`, never
   `train.py`. The real silent drop was `--goal_radius`. Continuous
   action-norm settings *do* reach training envs, via `rollout/collector.py`.
3. **No `grid_state` channel was added to `AgentConfig`.** The plan assumed a
   zero-cost seam reusing `rollout_rnn._grid_state_vec`, but that returns a
   smoothed-gbook column of width `Ng`, not a 2-wide coordinate — scaffold
   dependent, needing `sgb`/`env_offset` plumbing `NavAgent` lacks.
4. **`_make_vec` was not duplicated — the two copies differ.** `train_rnn`'s
   resets positions, `eval_rnn`'s does not (its caller seeds starts, and a
   reset would consume the env RNG and move them). Merging as the plan says
   would have silently changed one evaluator. Now `vec_env.make_vec(reset=...)`.
5. **`make_vec` cannot live in `training/`** — `eval_rnn` needs it, and
   evaluation→training is an upward edge the layering test must reject. It sits
   beside the classes it builds.
6. **Phase 5 must precede Phase 4c.** `VecEnv.step_batch` bundled
   reward+move-ignored+teleport under `goals_active`, so batching the
   coverage-style evaluators onto it changed what they measure. 5a made the
   clauses independent; only then was 4c safe.
7. **Bit-exactness under rebatching is impossible.** B=1 and B>1 agree to
   float32 precision, not bit-for-bit (batched matmuls accumulate differently).
   This is why the evaluator goldens pin per-trial *records*, not aggregates.
8. **Storage was less urgent than stated** — HOME was at 87%, not 99.5%.
9. **`channels.py` cannot live in `rollout/`.** The plan's 6e amendment puts it
   there, but `policy/agent.compute_input_dim` derives the agent's input width
   from it, and `rollout/collector` imports `policy.agent` — so `channels` in
   `rollout/` is a package cycle. It is `policy/channels.py`: the input layout
   is a property of the policy. (`signal.py` in `rollout/` is fine; nothing in
   `policy` needs it.)
10. **The training entry points stay at `hopfield_nav/*.py`.** The assessment's
    target table puts them under `training/` as `loop.py`/`phases.py`/`cli.py`,
    but that collapse is phase 3's work. Moving them in phase 6 would have
    broken 9 sbatch drivers and the sweep record to produce a layout phase 3
    immediately rewrites. Decided 2026-08-06.
11. **`hopfield_nav/phase_decoding/` needed no decision.** Phase 1 had already
    moved it to `$CLS_RUNS/results/phase_decoding_v1` (1.1 GB) and left the
    symlink; it is code-free, results only. The symlink is now
    `analysis/phase_decoding_v1_results`.
12. **`HopfieldConfig.init_mode` is live, not dead.** The plan lists it as
    undecided. It is read by `train.py:74` and set/restored at three sites in
    `train_phased.py`. Kept.
13. **The two `Hopfield` classes could not be reconciled "onto the
    `hopfield_nav` core dynamics"** as phase 7 asks, because that requires
    `encoder_training` to import `hopfield_nav` — the one direction the layering
    test forbids outright. The model moved out from under both instead:
    `hopfield_nav/world/memory.py` became the top-level `hopfield/` package at
    layer 0. Same outcome (one class), legal direction.
14. **`sweep_cosine_width.py` is a repoint target, not an archive candidate.**
    The plan says to move it to `docs/archive/`. It moved into
    `encoder_training/` instead on 2026-08-06 and imports three things
    `gridcode/` now owns, so deleting `cls/` without touching it would have
    broken it.
15. **Three figure generators were left in library packages by 6e.** Phase 6
    sorted entry points by "has a `__main__` guard" rather than by what they
    produce. `visualize_trajectories`, `viz_sensory` and
    `encoder_training/plot_sweep` all moved to `analysis/` in 6f, and layering
    rule 4 now makes it impossible to repeat. `viz_sensory` was additionally
    *broken* by the 6e move — it defaulted its output into a deleted directory.

---

## Open decisions (yours, not the refactor's)

- **`input_hopfield_raw` feeds a spurious direction on an empty Hopfield.**
  `q` is never masked by `memory_mask`, so a memoryless env gets
  `W @ (0 − embedding)` instead of zeros. `train_phase_a_only` defaults the
  flag **True** and the empty Hopfield is the normal phase-A explore case, so
  this was live. Pinned as a characterization test in `test_signal.py`; not
  fixed, because fixing it changes training results.
- **No `SITE_CONTRACTS` row is marked DECISION any more.** Both were resolved
  on 2026-08-06: `evaluate_goal_discovery` took `TRAINING` (one store
  opportunity per arrival, then relocate) and `evaluate_exploration` took
  `NO_GOALS` (the goal is inert — no teleport *and* no reward, since a reward
  spike is itself a signal the goal is there). `evaluate_union_coverage` was
  absorbed into exploration and deleted.
- **The wandb key is still in git history.** Stripping it from the working tree
  does not remove it. Rotate at wandb.ai.
- **`eval_checkpoints`' pass gate has no replacement.** That CLI applied
  `nav_stoch ≥ 0.7 ∧ store_eff ≥ 0.7 ∧ coverage ≥ 0.45` and picked a best
  checkpoint by `mean(nav_stoch, store_eff, coverage)`. Deleting it dropped the
  gate; `eval_all --output-json` gives the same numbers, but nothing scores
  them. Re-add it as a small script over the JSON if you want it back.
- **Phase A writes its numbered checkpoint inside the eval branch**, and
  `analysis.trajectories` skips files whose basename lacks an update number.
  At a large `--eval_every` only `phase_a_only_final.pt` exists, and the run
  cannot be visualized at all. Found by the smoke-test fixture; unrelated to
  the refactor and unfixed.

---

## What is left

**Phase 3 — one training loop.** Partly done.

*Done, 2026-08-06:* `train_navigate` was split along its schedule.
`training/stages.py` holds the `Stage` grammar (`explore:200 ;
interleave:800,empty_frac=1.0->0.5 ; exploit:100`) and the lookup arithmetic;
`training/explore.py` and `training/exploit.py` hold one regime each and return
a `RolloutSpec` rather than mutating `cfg` in place; `train_navigate.py` is the
composer plus the CLI. The four `--phase_a_*` / `--interleave_*` flags are gone,
so `run_phase_a_sweep_evelina.sh` no longer runs and three env-var launchers
(`run_navigate.sh`, `run_explore.sh`, `run_exploit.sh`, sharing
`navigate_job.sh`) replace it. Validated the way this note asked for: four tiny
fixed-seed runs captured before the change, three reproduced bit-for-bit
afterwards, and the fourth differs only in the one intentional way
(`test_schedule.py::test_a_warmup_before_an_anneal_deliberately_differs`).

*Outstanding:* the same treatment for the other four trainers. A `PhaseSpec`
(which Hopfield role per env, which heads frozen, which schedules, which update
function) generalizes what `Stage` + the two regime modules now do for one of
them; `training/loop.py` would run a list of them and the five entry points
would become five specs behind one CLI. That is the point at which the CLIs
would move under `training/` (see correction 10).

**Phase 4c′ — batching the remaining evaluators.** Two of the three are done.
`evaluate_exploration` is batched (2026-08-06) and `evaluate_union_coverage` no
longer exists — it computed a union over its own independent rollouts, and
exploration now computes the same union over the walks it was already doing.
That leaves `evaluate_goal_discovery`, which is the awkward one: it declares
`TRAINING`, so it hand-rolls the teleport at its call site the way
`evaluate_realistic` does. Batching it means routing it through
`step_batch(contract=TRAINING)` and *removing* the hand-rolled boundary rather
than sitting alongside it.

The general point, since it was muddled once already: batching under a site's
declared contract is a pure speedup and is unblocked. Which contract a site
declares is the decision. Those are separate, and only the second moves
numbers.

**`experiments/` as config.** The assessment proposes replacing
`run_phase_a_sweep_evelina.sh`'s 101-variant `case` block with one YAML per
variant plus a `launch.py`. Not started. Do **not** delete the shell script even
then — it is the only record of what those variants were.

Smaller things, none blocking:

- `hopfield_nav/utils.py` is now three unrelated things: Gram-Schmidt, direction
  classification, and a re-export of `gridcode.smoothing`. It could be split or
  the re-export dropped once callers are repointed.
- `analysis/schematics/make_*_schematic.py` have no `__main__` guard — their
  bodies run at import. `check_entry_points.py` runs them in full for that
  reason. Giving them `main()` functions would be tidier.
- `VectorHash.build_scaffold` draws `Wpg` and its prune mask from the **global**
  `np.random`, not from `cfg.seed` (`world/scaffold.py:26-27, 83-86`). Runs
  reproduce only because the entry points call `np.random.seed(cfg.seed)` first.
- `run.sh` still hardcodes `/home/$USER/cls` and runs a legacy `cls`-era sweep.

---

## How the safety net works

**The goldens pin behavior; they do not validate it.** A fixture records what
an evaluator returned last time on an untrained network, so it catches a metric
that *moves* and is blind to one that was wrong from the start — a `mean_coverage`
with the wrong divisor would have been pinned, not caught.
`tests/test_evaluator_correctness.py` is the other half: a `ScriptedAgent` walks
a fixed direction on a small grid, so every expected value is arithmetic rather
than a recording. Five wrong-metric mutations are known to fail it (wrong
coverage divisor, `store_efficiency` over trials instead of arrivals, an
off-by-one in `steps_to_goal`, a `visited` set missing its start cell, and `0`
instead of `-1` as navigation's failure sentinel) — the first two of which no
golden could have flagged.

`tests/golden/` holds 5 `.npz` fixtures generated from pre-refactor behavior.
Regenerate **deliberately only**:

```bash
python -m hopfield_nav.tests.gen_golden --check
```

If a golden changes, either the refactor altered behavior (investigate) or the
change was intended (regenerate, and put the diff in the commit message). Never
regenerate to turn a red test green.

`scripts/check_entry_points.py` walks every `__main__` guard outside tests and
runs it with `--help`. Most of those modules are executed by no test, so a
stale import in one surfaces only when somebody runs it — which for the
analysis scripts has historically been weeks later.

Fourteen mutations are known to fail the suite, which is what makes the coverage
claim real rather than nominal: a channel swap, a contract-table edit, an
impossible contract declaration, the `move_loss` mask, the teleport clause,
either `coerce_legacy_cfg` rename clause, each of the five layering rules, and
the five wrong-metric mutations above.
A tenth, outside the suite: swapping the two `randint` calls in
`sample_distractors` changes phase-A weights, which is what proved the
distractor differential was not vacuous.

Two traps worth remembering:

- The first version of `long_horizon_evaluators.npz` passed while pinning
  nothing — on a 6×6 grid the untrained agent never reached the goal, so the
  at-goal branch never ran. `gen_golden` now raises if that fixture stops
  producing reaches.
- A differential probe that crashes in *both* trees diffs clean. Two probes in
  phase 6 did exactly that (a wrong attribute name, wrong CLI flags) and
  reported "identical" over their first ten lines. Check the exit code and the
  line count, not just the diff.
