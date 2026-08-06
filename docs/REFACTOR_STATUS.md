# Refactor status — 2026-08

Handoff note for the 2026-08 refactor. The original plan lives at
`~/.claude/plans/ok-that-sounds-good-drifting-whale.md`; **it is stale in the
ways listed under "Plan corrections" below.** Read this file first.

Branch `refactor/2026-08`, 22 commits from tag `pre-refactor-2026-08` on `main`.
Suite: 312 passing, tree clean, all 25 entry points import.

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

Each dedup was verified by running the pre-change code in a `git worktree` and
diffing the output, not by trusting a green suite. All byte-identical except
where noted.

---

## The layout, as it now stands

```
hopfield_nav/
  config.py  utils.py  encoder_io.py       leaves; anything may import them
  world/       env vec_env scaffold memory episode
  policy/      agent agent_rnn channels
  rollout/     collector rnn signal oracles distractors types
  updates/     ppo bc bc_rnn
  evaluation/  metrics rnn protocols batched checkpoint_io
  training/    world_setup rnn_sequential
  tests/
  train.py train_phased.py train_phase_a_only.py train_phase_b_only.py
  train_rnn.py eval_all.py                        <- the six CLIs
analysis/
  continual/ phase_decoding/ schematics/ scaffold_experiments/
encoder_training/
cls/                                        legacy, retired in phase 7
```

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

---

## Open decisions (yours, not the refactor's)

- **`input_hopfield_raw` feeds a spurious direction on an empty Hopfield.**
  `q` is never masked by `memory_mask`, so a memoryless env gets
  `W @ (0 − embedding)` instead of zeros. `train_phase_a_only` defaults the
  flag **True** and the empty Hopfield is the normal phase-A explore case, so
  this was live. Pinned as a characterization test in `test_signal.py`; not
  fixed, because fixing it changes training results.
- **Two `SITE_CONTRACTS` rows are marked DECISION** in
  `tests/test_goal_contract.py`: `evaluate_goal_discovery` and
  `evaluate_exploration` decline the teleport. Adopting the training contract
  would make them stricter and match training, but moves
  `store_success_rate` and `mean_coverage`.
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

## Phase 7 — next

Unchanged from the plan, plus what phase 6 left it:

- Move the six live `cls/vectorhash` functions plus `smooth_g` / `smooth_gbook`
  (currently re-exported through `hopfield_nav/utils.py`) into a top-level
  `gridcode/`. That removes the last back-edge,
  `hopfield_nav.utils → encoder_training.utils`.
- Move `cls/eval/nav_eval.py`, `cls/nav.py`, `cls/hopfield.py` into
  `encoder_training/nav_eval/`, reconciling the two `Hopfield` classes onto the
  `hopfield_nav` core dynamics.
- `sweep_cosine_width.py` moved from the repo root into `encoder_training/`
  on 2026-08-06, so it is kept rather than archived. It still imports
  `cls.utils.GridUtils.smooth_gbook` and `cls.vectorhash.{assoc_utils_np,
  assoc_utils_np_2D}`, which makes it a **repoint target for `gridcode/`**, not
  a deletion. `run.sh` invokes it as `python -m encoder_training.sweep_cosine_width`.
- Tag `legacy-cls`, delete `cls/` and the root `tests/`.
- **Update `pyproject.toml`**: it still installs only `cls` (`include =
  ["cls*"]`) and still excludes a `notebooks*` directory that no longer exists.
  `hopfield_nav`, `encoder_training` and `analysis` work only because every
  entry point runs as `python -m` from the repo root.
- Add `gridcode` and the retired `cls` to `LAYERS` in `tests/test_layering.py`.

Do **not** delete `run_phase_a_sweep_evelina.sh` — it is the only record of what
the 101 named variants were.

Phase 3 (one training loop: `training/phases.py` + `training/loop.py`, the five
entry points as `PhaseSpec` lists behind one CLI) remains unstarted, and is the
point at which the CLIs would move under `training/`. Phase 4c′ (batching the
other three evaluators) is deferred — it is a decision about what the metrics
mean, not a refactor; see the plan.

---

## How the safety net works

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

Nine mutations are known to fail the suite, which is what makes the coverage
claim real rather than nominal: a channel swap, a contract-table edit, an
impossible contract declaration, the `move_loss` mask, the teleport clause,
either `coerce_legacy_cfg` rename clause, and each of the three layering rules.
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
