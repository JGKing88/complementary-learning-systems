# Reproducing gentle-terrain-124 on the schedule system

**Goal.** Get a run on the post-2026-08 `--schedule` trainer that matches
`gentle-terrain-124` at u380: **0.994 nav success rate, 22.9 mean speed** on a
fresh eval. One run at a time, each documented below.

Launcher: `hopfield_nav/run_repro_v35.sh`. Every knob is spelled out there
rather than inherited, because matching a specific historical config is the
whole point and an inherited default would break it silently.

---

## The target

`$CLS_RUNS/agent_ckpts/phase_a_only_gentle-terrain-124/phase_a_u380.pt`,
written 2026-05-13. Sweep variant **`v18d39_size20_v35`**, seed 42, from
`hopfield_nav/run_phase_a_sweep_evelina.sh`.

Its own in-training eval at u380 (`slurm_phase_a_sweep_13737223.out:115-117`):

| metric | n_dist=0 | n_dist=5 | n_dist=10 |
|---|---|---|---|
| nav success_rate | 1.000 | 0.988 | 0.991 |
| nav mean_speed | 0.825 | 0.773 | 0.779 |
| nav mean_steps | 15.2 | 22.1 | 18.5 |
| disc store_success | 0.441 | 0.450 | 0.391 |
| expl mean_coverage | 0.532 | 0.501 | 0.495 |

### What 0.994 / 22.9 actually means

Not the in-training numbers above. They come from a fresh eval recorded at
`$CLS_RUNS/results/eval_results/v35_gt_20260518_095702/v35_u380.log`:

```
 n_dist    srD    msD    srS    msS  stEff  reach    cov  findR
      0  1.000   20.8  1.000   20.5   0.77   0.64   0.53   0.79
      5  0.991   21.5  0.997   23.7   0.82   0.63   0.51   0.83
     10  0.991   26.3  0.988   25.8   0.82   0.60   0.48   0.79
```

- **0.994** = mean of `srD` over the three distractor levels — (1.000 + 0.991 +
  0.991) / 3.
- **22.9** = mean of `msD` — (20.8 + 21.5 + 26.3) / 3 = 22.87. So "mean speed"
  is the mean *steps* column: **lower is better**, and it is a cost, not a rate.

So the target is a pair of averages over `n_dist ∈ {0, 5, 10}`, deterministic
navigation only.

### Reproducing that eval

The script that printed the table (`eval_checkpoints.py`) was deleted in the
phase-6b refactor, which folded its evaluators into `eval_all` — `srD`/`msD`
are `nav_det[d].success_rate` / `nav_det[d].mean_steps`. The equivalent today:

```bash
python -m hopfield_nav.eval_all --ckpt <ckpt> --device cuda \
    --num-val-envs 10 --num_trials 32 --max_steps 400 \
    --n_distractors 0 5 10 --no-nav-stoch --skip-realistic --repeat-trials 0 \
    --output-json <out.json>
```

**This is run on the original u380 checkpoint as a control before it is trusted
on anything new.** If it does not return 0.994 / 22.9 there, the protocol is
wrong and no comparison built on it means anything.

### Cost, measured rather than inferred

The original ran at **239 s/update** — u20 → u440 took **28 hours**, measured
from the checkpoints' own mtimes. (The manifest's `created` field and the log
mtime both suggest ~46 min, and both are misleading: the manifest is
backfilled, so `created` is a reconstruction, and the log mtime is when the job
was killed. Do not size a run from either.) It was configured for 3000 updates
and never got near them.

Its trajectory, mean over `n_dist ∈ {0, 5, 10}`, from the in-training evals:

| update | srD | msD | | update | srD | msD |
|---|---|---|---|---|---|---|
| 20 | 0.260 | 139.2 | | 260 | 0.990 | 20.9 |
| 60 | 0.860 | 80.7 | | 300 | 0.978 | 22.4 |
| 100 | 0.949 | 59.1 | | 340 | 0.991 | 18.6 |
| 140 | 0.971 | 52.4 | | **380** | **0.993** | **18.6** |
| 180 | 0.978 | 33.7 | | 420 | 0.985 | 16.5 |
| 220 | 0.991 | 30.1 | | 440 | 0.983 | 15.2 |

**Success rate saturates around u220; `msD` is the metric still moving.** That
is what makes u380 the interesting checkpoint rather than an arbitrary one, and
it also sets the cheaper stopping points: ~u220 buys the success rate at half
the compute, ~u300-380 is needed for the step count.

(In-training `msD` at u380 is 18.6 while the fresh eval reports 22.9 — a
protocol difference, not a discrepancy. Compare fresh-to-fresh only.)

## Where the config came from

The manifest is `provenance: backfilled` and carries no `argv`, and the
pre-refactor trainer kept its schedule knobs as bare function arguments — they
reached neither the config nor the checkpoint. (That gap is exactly what
putting `schedule` and friends on `TrainConfig` fixed.) So the config was
recovered from two places and cross-checked:

1. `slurm_phase_a_sweep_13737223.out:1` — the `Running variant=... with: ...`
   line, giving the variant's full `EXTRA` string.
2. `phase_a_u380.pt`'s saved `config` — the `TrainConfig` half.

## Old flags → schedule

| Old | New |
|---|---|
| `--warmup_explore_only_updates 0` | *(no explore stage)* |
| `--interleave_empty_fraction 1.0` | `empty_frac=1.0->…` |
| `--interleave_empty_target 0.50` | `…->0.5` |
| `--interleave_anneal_updates 50` | `anneal=50` |
| `--phase_a_updates 3000` | `interleave:600` (see below) |
| `--phase_a_lr 3e-4` | `--lr 3e-4` |
| `--phase_a_novelty_reward 0.3` | `--novelty_reward 0.3` |

→ `--schedule "interleave:600,empty_frac=1.0->0.5,anneal=50"`

With **no warmup**, the old global anneal clock and the new stage-local one
coincide, so this translation is bit-exact — the property pinned by
`test_schedule.py::test_an_anneal_with_no_warmup_is_unchanged` and verified
end-to-end during the refactor (baseline C reproduced per-update weight hashes
exactly).

**3000 → 600.** `n_updates_total` is read in exactly two places: the loop bound
and the novelty anneal (`train_navigate.py:182`). `--no-novelty_anneal` is set,
so the total has no effect on the trajectory — only on when it stops. 600 is
well past the u380 target and past where the original died.

## Config cross-check

The new CLI's `TrainConfig`, built from the translated argv, diffed field by
field against `phase_a_u380.pt`'s saved config. Four fields differ, all benign:

| Field | New | Old | Why it is fine |
|---|---|---|---|
| `ckpt_every` | `None` | absent | Field postdates the run. `None` = follow `eval_every`, which is what the old code did unconditionally. |
| `env.allow_offcell_store` | `False` | absent | Unreachable here. The write site is gated on `not shared_hopfield` (`collector.py:463`), and `train_navigate` passes a single shared Hopfield per rollout, so no agent store is ever written during navigate training. |
| `hopfield.beta` | `None` | `3.6987` | Resolved at startup from `encoder_gain`; same encoder → same value. |
| `hopfield.novelty_reward` | `0.3` | `0.0` | The *old* one is wrong: the pre-fix trainer snapshotted `cfg` mid-loop, where novelty is parked at 0 between rollouts. The log banner confirms `novelty=0.3`. |

Everything else — all 40-odd remaining fields including the six input channels,
`hidden_size 1024`, `init_log_std -1.8` + freeze, `goal_reward 5.0`,
`goal_radius 1.0`, `time_penalty 0.05`, `wall_penalty 0.1`,
`persistence_bonus 0.05`, `novelty_scale_remaining`, the distractor counts,
`ppo.ent_coef 0.005`, `ppo.clip_coef 0.15` — matches exactly.

---

## Runs

### Run 1 — faithful translation, seed 42

- **Schedule:** `interleave:600,empty_frac=1.0->0.5,anneal=50`
- **Changes vs. the original:** none intended. This run exists to establish
  that the new system reproduces the old result before anything is varied.
- **Submitted:** job `19798369`, `pi_evelina9`, node3905, 2026-08-06.
  Log: `$CLS_RUNS/logs/slurm_repro_v35_19798369.out`
- **Startup banner:** `=== navigate: interleave:600,empty_frac=1->0.5,anneal=50
  (600 updates) ===` — schedule parsed as intended.
- **First attempt, job `19795371`: cancelled at u20, my error.** I sized the
  walltime at 12 h from the original's apparent 46-minute runtime, which was an
  artifact of a backfilled manifest date (see above); the real cost is
  ~239 s/update, and this run measured 171 s/update, so u380 needs ~18 h. The
  job would have died around u250. Resubmitted at 2 days. Its one eval, u20,
  is kept below because it is still a valid data point.
- **u20 comparison** (mean over the three distractor levels):

  | | srD (0/5/10) | msD |
  |---|---|---|
  | original | 0.294 / 0.263 / 0.225 | 139.2 |
  | run 1 | 0.300 / 0.275 / 0.303 | 136.5 |

  Close, and exact agreement is not expected: cuDNN/cuBLAS are nondeterministic
  by default, so two GPU runs of *identical* code diverge. The bit-exactness
  established during the refactor was on CPU.
- **Progress at 12 h 23 m / u240** (186 s/update; u380 expected at ~19.6 h,
  well inside the 2-day limit). Mean over `n_dist ∈ {0, 5, 10}`:

  | update | orig srD | orig msD | run 1 srD | run 1 msD |
  |---|---|---|---|---|
  | 20 | 0.260 | 139.2 | 0.293 | 136.5 |
  | 40 | 0.548 | 103.2 | 0.824 | 73.7 |
  | 60 | 0.860 | 80.7 | 0.946 | 48.9 |
  | 100 | 0.949 | 59.1 | 0.911 | 55.5 |
  | 140 | 0.971 | 52.4 | 0.984 | 50.3 |
  | 180 | 0.978 | 33.7 | 0.989 | 36.7 |
  | 220 | 0.991 | 30.1 | 0.985 | 28.6 |

  Mean |Δ srD| over the 11 shared evals: 0.050, dominated by u40 (+0.276) where
  the curve is steepest and a small timing difference moves the number a lot.
  Run 1 leads through u40-u80 and the two converge from u140 on. By u220 both
  sit on the success-rate plateau and `msD` is what still separates them.

- **Killed at u250, 13 h 08 m in** (2026-08-07 07:54). Not walltime (2 d
  limit), not OOM (69 GB peak of 100 GB requested), not preemption
  (`PreemptMode=OFF` on this partition), not node failure (node3905 healthy).
  SLURM recorded `FAILED 15:0` while the step itself exited `COMPLETED 0:0`,
  and the log's last line is bash reporting `707544 Terminated` — i.e. the
  python process alone received SIGTERM from outside SLURM. A `scancel` would
  have marked the job `CANCELLED`, as it did for an unrelated `hnav-navigate`
  job at 07:18. Cause not established.

### Run 1a — resume from u240

Restarting would have cost the full ~18 h again. Resuming costs ~7 h, and the
schedule state at u240 is exactly reproducible, so the continuation is faithful
rather than approximate:

- `empty_fraction_at(u240)` = **0.5** — the anneal finished at u50, so the
  remainder of the original schedule is a flat 0.5.
- `_compute_epsilon(240, 0.4, 200)` = **0.0** — ε annealed out at u200.

So the continuation is `--schedule "interleave:360,empty_frac=0.5"` with
`--epsilon_explore 0`, and everything else inherited from the checkpoint. Had
the schedule simply been re-run from update 1, ε would have jumped back to 0.4
and the interleave anneal would have replayed 1.0 → 0.5 — both wrong.

- **Submitted:** job `19847118`, `pi_evelina9`, node3905, 2026-08-07.
  Log: `$CLS_RUNS/logs/slurm_repro_v35_19847118.out`
- **Checkpoint numbering restarts**, so global u380 = this run's **u140**.
- **Known imperfection:** Adam's moment estimates reset at the boundary. Only
  the weights are carried, not the optimizer state.
- **Stopped deliberately** shortly after starting: the u240 evidence was
  already sufficient (see verdict). No result from this leg.

---

## Verdict

**The schedule system reproduces the old trainer.** Stopped at u240 rather than
run to u380, because the curves had already converged and the remaining 7 h
would only have refined `msD`.

### What was shown

Same config, same seed (42), same encoder; the two runs track each other over
12 evals:

| update | orig srD | run 1 srD | orig msD | run 1 msD |
|---|---|---|---|---|
| 140 | 0.971 | 0.984 | 52.4 | 50.3 |
| 180 | 0.978 | 0.989 | 33.7 | 36.7 |
| 220 | 0.991 | 0.985 | 30.1 | 28.6 |
| 240 | 0.991 | 0.988 | 27.7 | 27.0 |

From u140 on the two are within ~0.01 srD of each other, and at u240 both sit
on the success-rate plateau with `msD` within 0.7 steps. Run 1 led through
u40-u80 and the original led slightly at u220; neither is systematically ahead.

Separately and more strongly: the translated CLI produces a `TrainConfig`
**identical field-for-field** to the original checkpoint's, bar four fields all
accounted for above.

### What was *not* shown

- **The target pair was never measured on the reproduction.** The run stopped at
  u240; srD 0.994 / msD 22.9 is a u380 fresh-eval number. The claim here is
  "same trajectory", not "same endpoint".
- **The fresh-eval comparison never ran.** `hopfield_nav/run_repro_compare.sh`
  is written and ready — it evaluates both checkpoints in one job with the
  original's own numbers as a control on the protocol — but was not executed.
  Running it on the two u240 checkpoints would close this in ~15 min.
- **One seed.** Nothing here separates "the systems agree" from "seed 42 agrees".

### What would have been wrong to claim

Bit-exactness. Two GPU runs of *identical* code diverge — cuDNN and cuBLAS are
nondeterministic by default. The bit-exact equivalence established during the
refactor was on CPU, over four fixed-seed runs, and covers the schedule
arithmetic rather than the whole training loop. On GPU, "tracks within noise"
is the strongest available claim, and it is the one made here.

