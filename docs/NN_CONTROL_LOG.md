# Goal-conditioned NN control — log

Companion to `NN_CONTROL_PLAN.md`. The plan says what; this says what
happened, in order, with numbers. Newest at the bottom.

Branch `worktree-nn-generalization-control`. Env python:
`$HOME/.conda/envs/cls/bin/python`.

---

## 2026-09-10 — build, §5.9 step 1

**Config (§5.1).** Added `SENSORY_MODES`, `SENSORY_ENCODERS`;
`RNNAgentConfig.input_sensory / sensory_mode / input_xy_state /
sensory_encoder*`; `RNNTrainConfig.region_val_frac / pairs_per_env /
resample_goal_on_reach`. All default to today's behaviour.

**FeedForwardCore (§5.3).** In `policy/recurrent.py`, dispatched on
`rnn_cell == "mlp"`. Returns zeros as its state rather than echoing the
input state, so a lifetime caller that reads it sees a network that
provably keeps nothing. Checked: T-step call equals T single-step calls
(`allclose` True on a 3-layer relu core).

**Decision — A's model does not subclass RNNAgent.** `PairRegressor` is
`FeedForwardCore` + one `Linear`. No `Normal`, no `log_std`, no `act()`.
The plan's reason (§3.1) stands: A is a regression, and a Gaussian head
with nothing to sample is a source of ambiguity, not a model. The cost is
a thin adapter for B later so both go through one evaluator.

**Layout as data (§5.2).** `rnn_input_layout` in `policy/agent_rnn.py`;
`compute_rnn_input_dim` sums it; `build_rnn_input` assembles from it. The
four historical channels keep their lenient `is not None` guard (callers
rely on it); the three new ones raise when enabled-but-missing. Producers
`sensory_vec(omni)`, `goal_sensory_vec(omni|north)`, `xy_vec` in
`rollout/rnn.py`. Existing tests, golden fixtures and layering: green.

**Region holdout (§5.5).** `GeneratedSplit.region_cells` (JSON default:
empty), `generate_split(region_frac=)`, `CellSets` with the invariants
checked at construction, `split.cell_sets()`.

**Found: the goal partition was one-cell-per-env, not a fraction.**
`generate_split` with `refresh_goal=False` draws `n_train` distinct goal
cells (one per env) and puts *everything else* in `goal_cells_val` — the
first build gave `goal_train = 64, goal_heldout = 296`. The `val_frac`
partition the plan assumed only runs under `refresh_goal=True`. Since the
pair sampler draws goals from the whole train partition, that is the
partition it needs; `rnn_world` now passes `refresh_goal=True` whenever
`region_val_frac > 0`. Configured world at size 20, 64 + 16 envs, margin
20, `lambdas = [11, 12, 13]`: `start_train 360 / goal_train 320 /
goal_heldout 40 / region 40`, `region ⊂ goal_cells_val` True. Build 4 s.

**Evaluator (§5.6) smoke test on the configured world, 4 envs.** Teacher
scores 0° / 1.0 in every mode; NN decoder 0° / 1.0 on train × train in
every mode. Two plan numbers corrected from this:

- *Random discrete baseline is 0.48, not 0.37.* Off-axis pairs have a
  2-action optimal set and are the majority; mean set size is 1.9/4.
- *The NN decoder is exact only on train × train.* On held-out cells it
  decodes to the nearest **training** cell, which is a neighbour by
  construction, so on xy it scores ~5° / ~0.97 there. That is correct — a
  held-out cell is not in the dictionary — and it is the calibration for
  the line: wherever the encoding is locally smooth, lookup-then-subtract
  is already nearly perfect, and the model has to beat ~5°, not 90°. C5
  rewritten accordingly.

Enumerated train × train is 114,880 pairs per env; a 2048-pair table with
all four reference lines takes 0.1–0.3 s on CPU.

**Script, launcher, pre-flight (§5.7, §7.1).** `train_goal_pairs.py`
(composer), `training/goal_pairs_setup.py` (modes + env sets, layer 6, so
the pre-flight and the trainer share it without importing a CLI — rule 5),
`run_goal_pairs.sh`, `scripts/goal_nav_preflight.py`. Smoke run on 4 envs:
xy goes 83° → 6.5° in 30 updates, heldout tracks train exactly.

**Pre-flight on the configured world: C1–C8 all PASS.**

| gate | result |
|---|---|
| C1 | split invariants hold; heldout walls/boxes disjoint; same ⊂ train |
| C2 | exact-twin rate omni **0.000**, gbook **0.000** (single-north 0.055) |
| C3 | unit_vectors == bfs_continuous; bfs_discrete ∈ optimal_set; set size 1 iff aligned |
| C4 | random 89.6° / 0.482 (expect 90 / 0.476) |
| C5 | NN decoder exact on train×train in every mode; xy region×region 6.4° / 0.96 |
| C6 | pair_inputs bit-identical to build_rnn_input, all three modes |
| C7 | cos(Δgbook@p1, Δgbook@p2), same d=(5,0), 100 pairs: **mean −0.03**, range [−0.67, 0.36] |
| C8 | min pairwise Chebyshev gap over 80 boxes = 27 ≥ margin 20 |

C7 sharpens the non-invariance finding: over 100 random base-point pairs
the code-difference for the same displacement is *uncorrelated* on
average, not merely imperfect. The grid-mode network cannot learn one
displacement operator; it has to learn the phase geometry.

**Tests (§5.10).** `tests/test_goal_pairs.py`, 26 tests, all pass. Two
tolerances loosened from 1e-3° to 0.1°: float32 `arccos` near 1 gives
~0.02° of noise on an exact match. Layering test: `train_goal_pairs`
registered at layer 7; green.

**A0 launched** on CPU: xy mode, both action modes, 300 updates, the full
64 / 16 / 8 world.

## 2026-09-10 — A0 and the A1 launch

**A0 on the login node failed for a reason worth recording.** Two runs
at 15 GB RSS each (the full smoothed gbook, ~5 GB, plus torch) plus a
probe script exceeded what the login node tolerates; one run and the
probe were OOM-killed (exit 137, no traceback). Fix: `build_env_sets`
now drops `vh.gbook` and `sgb` after the `EnvTensors` are built — every
cell a run reads is in them — unless `keep_field=True` (the pre-flight).
A0 resubmitted on `mit_normal_gpu`. Also: one eval of the 88-env table
is 37 s on CPU, 16 s on GPU; the trainer is GPU-only in practice.

**A0 (xy, mlp-2 h256, 300 updates), jobs 22523174 / 22523175:**

| | u=1 | u=50 | u=100 | u=300 |
|---|---|---|---|---|
| continuous, heldout tt | 80.1° | 1.50° | 0.60° | **0.35°** |
| continuous, heldout rr | 82.3° | 1.65° | 0.65° | **0.37°** |
| discrete, heldout tt | 0.51 | 1.00 | 1.00 | 1.00 |
| discrete, heldout rr | 0.49 | 1.00 | 1.00 | 1.00 |

NN-decoder line on heldout rr: 6.3° / 0.96. **C13 passes** — ≤ 5° /
≥ 0.98 on every cell, heldout tracks train exactly, and the model beats
the lookup line on region cells by 17×, which is what learning the
function rather than the table looks like. Enumerated FINAL pending.

**A1 wave submitted** (jobs 22523453–22523467, 15 runs) before A0's
enumeration finished: the gate is unambiguous at 0.35° / 1.00. Plan
baseline (mlp-2, mlp-4 × both actions × 2 seeds, h256) plus a sweep on
continuous seed 0: h512, h1024, l3h512, l4h512, l3h512+cosine,
l3h512+wd1e-4, l3h512 tanh.

**A0 enumerated FINAL — C13 PASS both modes.** Every pair in every
quadrant, 88 envs:

| | train | heldout | same |
|---|---|---|---|
| continuous, all 6 cells | 0.3–0.4° | 0.3–0.4° | 0.3–0.4° |
| discrete, all 6 cells | 1.000 | 1.000 | 1.000 |

Seed spread across envs 0.0. NN-decoder line on region×region: 6.4° /
0.965. The xy model is 17× better than lookup on cells it never saw,
and heldout equals train to the last digit. The pipeline is correct.

## 2026-09-10 — A1 grid mode, first signal

**The NN-decoder line on grid mode is ~83°, i.e. random.** Probed on one
env's 40 region cells: the nearest *training* cell by gbook cosine is
**12 Manhattan cells away for 36 of the 40** (an alias, not a neighbour;
only 4 decode to an adjacent cell). Cosine vs spatial distance on this
scaffold: d=1 0.86, d=2 0.64, d=3 0.40, d=4 0.21, d=6 **0.04**, then
aliasing back up (d=10 max 0.53) as the 11/12/13 modules wrap. So on grid
mode there is no interpolation crutch: a held-out cell's code is not
*near* any training code in the input metric. Whatever the model gets on
region cells comes from having learned the phase structure. (This is
also why the codebase reads the code with a hand-built local frame and
not a lookup.)

**First A1 run, mlp-2 h256 continuous seed 0 (job 22523453):**

| u | train tt | heldout tt | heldout rr | nn rr |
|---|---|---|---|---|
| 1 | 88.3° | 89.4° | 89.9° | 82.9° |
| 100 | 53.1° | 65.3° | 64.8° | 83.0° |
| 400 | 11.8° | 13.3° | 13.5° | 82.6° |

The train/heldout gap that was 12° at u=100 is 1.5° at u=400, and
region×region equals train×train on held-out envs. Still climbing.

**First A1 enumerated FINAL — mlp-2 h256 continuous seed 0 (22523453):**

| | train | goal_heldout | region |
|---|---|---|---|
| **heldout envs**, start=train | 3.0 ± 0.2 (nn 0.0) | 3.0 ± 0.2 (nn 62.5) | 3.0 ± 0.2 (nn 58.2) |
| **heldout envs**, start=region | 3.0 ± 0.2 (nn 63.3) | 3.0 ± 0.2 (nn 79.8) | **3.0 ± 0.2** (nn 82.9) |
| train envs, start=train | 2.7 | 2.7 | 2.8 |
| train envs, start=region | 2.8 | 2.8 | 2.9 |

Every held-out cell at 3.0°. Train→heldout gap 0.3°; train×train →
region×region gap 0.0°. By §6.3 (≤ 20°, region within 10° of train×train)
this **generalizes**, and the smallest model in the wave does it. The
NN-decoder line it beats is 58–83°: there is no lookup route on this
encoding, so the 3° is learned phase geometry.

**Seed 1 (22523454) agrees:** heldout 3.1–3.2° every cell. Seed spread
~0.2°. **mlp-4 h256 continuous (22523455) at u=1900: heldout rr 1.57°** —
depth halves the error. **mlp-4 discrete (22523460) at u=1100: 1.000 on
every held-out cell**, NN line 0.53 (random floor 0.48). Discrete grid
mode is solved outright.

Full test suite: green (one failure fixed — the test fixture had to keep
the scaffold field after `build_env_sets` started dropping it).

## 2026-09-10 — A1 wave results (14 of 15; tanh pending)

All runs: grid mode, 64 train / 16 heldout / 8 same envs, 2000 updates,
lr 1e-3 AdamW, relu unless noted. Heldout cells are enumerated. Every
continuous run's best checkpoint is its **last** — still descending.

**Discrete: solved outright.** mlp-2 and mlp-4, both seeds: **1.000 on
all six held-out cells** (NN line 0.53; random 0.48).

**Continuous, heldout region×region (the hardest cell):**

| config | params | ho rr | ho tt | tr tt |
|---|---|---|---|---|
| l2 h256 s0 / s1 | 290k | 3.01 / 3.12 | 3.02 / 3.21 | 2.72 / 2.88 |
| l2 h512 | 710k | 2.26 | 2.22 | 1.88 |
| l2 h1024 | 1.9M | 1.78 | 1.77 | 1.26 |
| l4 h256 s0 / s1 | 420k | 1.53 / 1.58 | 1.52 / 1.60 | 1.31 / 1.36 |
| l3 h512 | 970k | 1.26 | 1.23 | 1.01 |
| l3 h512 + wd 1e-4 | 970k | 1.24 | 1.23 | 0.98 |
| l3 h512 + cosine | 970k | 1.86 | 1.85 | 1.46 |
| **l4 h512** | 1.5M | **0.84** | **0.85** | 0.69 |

Reading:
- **Depth beats width per parameter**: l4h256 (420k) < l2h1024 (1.9M).
  They stack: l4h512 is the best at 0.84°.
- **Every held-out cell equals every other** to within 0.05°: start
  side, goal side, region — no cell is harder. Train→heldout gap is
  0.2–0.5° and shrinks with capacity.
- **Weight decay does nothing**; there is no overfitting to regularise.
- **Cosine hurts** (1.86 vs 1.26): it anneals the LR to zero while the
  model is still descending. Same message as "best = last checkpoint":
  the remaining lever is **updates**, not architecture.
- Seed spread ≈ 0.1°.

**A1b submitted** (22526922–24): l4h512 × 2 seeds and l5h768, at 8000
updates (4×), eval every 250, 3 h limit.
