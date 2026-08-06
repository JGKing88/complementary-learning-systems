# Refactor status — 2026-08

Handoff note for the 2026-08 refactor. The original plan lives at
`~/.claude/plans/ok-that-sounds-good-drifting-whale.md`; **it is stale in the
ways listed under "Plan corrections" below.** Read this file first.

Branch `refactor/2026-08`, 17 commits from tag `pre-refactor-2026-08` on `main`.
Suite: 274 passing, tree clean, all 25 entry points import.

```bash
./run_tests.sh
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
| 6-prep | Smoke tests for the four entry points Phase 6 will move |

Each dedup was verified by running the pre-change code in a `git worktree` and
diffing the output, not by trusting a green suite. All byte-identical.

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
   action-norm settings *do* reach training envs, via `rollout.py:108`.
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

---

## Phase 6 — next, and best started in a fresh session

It rewrites nearly every import in the repo and needs the layering test plus all
25 entry points green in one commit. A half-applied file move is the worst state
to leave this in.

**Already done as side effects:** `world/`, `evaluation/`, `training/` packages
exist; `make_vec` deduped; `baseline` no longer imports privates from
`train_rnn`; `_env_xy_float` deduped.

**Still to do**, from the plan's 6a–6e:

- Break the cycle: `hopfield_nav.encoder`/`utils` → `encoder_training`, and
  `encoder_training.experiments.encoder_scaffold` → `hopfield_nav.env`.
- Move `RolloutBatch` out of `ppo.py` and the novelty oracles out of `bc.py`,
  so `rollout.py` stops importing upward.
- `evaluation/checkpoint_io.py`: dedup `coerce_legacy_cfg` / `cfg_from_checkpoint`
  / `build_eval_world` / `load_agent` / `scaffold_layout_dict` from 3 copies and
  repoint **5 importers**, verified 2026-08-05: `train_phase_b_only.py:34`,
  `visualize_trajectories.py:65`, `final_plotting/agenthash.py:38`,
  `final_plotting/prep_scaffold.py:41`, `phase_decoding_v2/rollout.py:20`. Two
  import from `eval_checkpoints` and three from `eval_all`, so both source
  modules must be drained before either can be deleted.
- `rollout/distractors.py`: promote `eval._sample_distractor_goals` /
  `_goal_encoding`, replace 3 inline loops.
- Delete `eval_checkpoints.py`, `eval_distractors.py` (after repointing).
- Delete root strays; archive `hopfield_nav/*.md` and `notebooks/`.
- The `git mv` to the layered tree + `tests/test_layering.py`.

**Prerequisite: done.** These four had no coverage at all, and Phase 6 changes
every import they use — a stale import in an analysis script surfaces only when
someone runs it. `tests/test_smoke_train.py` now covers all four
(`test_eval_all_cli_end_to_end`, `test_train_phase_b_only_end_to_end`,
`test_visualize_trajectories_renders`, `test_agenthash_run_sequential_outer_loop`),
chained off a shared phase-A checkpoint fixture. Mutation-verified: repointing a
`visualize_trajectories` import at a module Phase 6 has not created yet fails
the suite. ~43s.

One thing that fixture exposed, worth knowing independently of the refactor:
phase A writes its numbered checkpoint *inside* the eval branch, and
`visualize_trajectories` skips files whose basename lacks an update number. At a
large `--eval_every` only `phase_a_only_final.pt` exists, and the run cannot be
visualized at all.

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

Five mutations are known to fail the suite, which is what makes the coverage
claim real rather than nominal: a channel swap, a contract-table edit, an
impossible contract declaration, the `move_loss` mask, and the teleport clause.

One trap worth remembering: the first version of `long_horizon_evaluators.npz`
passed while pinning nothing — on a 6×6 grid the untrained agent never reached
the goal, so the at-goal branch never ran. `gen_golden` now raises if that
fixture stops producing reaches.
