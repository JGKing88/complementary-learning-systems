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

**15th run: l3 h512 tanh — 12.3° heldout rr**, 10× worse than relu at
the same size. ReLU throughout. A1 wave complete.

**Best A1 run so far, l4h512 s0, same-vs-heldout probe:**

| cell | train | same | heldout |
|---|---|---|---|
| train×train | 0.69 | 0.67 | 0.85 |
| region×region | 0.76 | 0.75 | 0.84 |

`same` = train (as it must); env-side gap 0.15°. Held-out region×region:
median **0.63°**, **100% of pairs within 30°**, per-env std 0.06°. There
are no failure cases; the mean is a uniformly small error.

## 2026-09-10 — A1b: 4× updates, and a late instability

**l4h512, 8000 updates, seed 0 (22526922).** Heldout region×region,
sampled every 250:

| u | 2000 | 3750 | 5750 | **6250** | 7750 | 8000 |
|---|---|---|---|---|---|---|
| ho rr | 0.84 | 0.46 | 0.34 | **0.32** | 0.99 | 0.90 |

Four times the updates takes it from 0.84° to **0.32°** — and then a
late instability at constant lr 1e-3 (loss 0.0000 → 0.0002, error back
to ~0.9°). The enumerated FINAL at u=8000 is 0.9°, which understates the
model. By the selection rule (heldout train×train), the best checkpoint
is u=6250 (0.326° tt / 0.323° rr, sampled); the nearest saved one is
`pairs_u6000.pt` (0.37° sampled). **Enumerating it** on a GPU node
(22527723) via the new `eval_goal_pairs` CLI so the headline is not a
sampled number.

**Fix for the instability: a late step decay**, ×0.1 at 70% — not
cosine-from-the-start, which the A1 wave showed hurts. Added
`--lr_schedule step` (`--lr_step_at`, `--lr_step_gamma`). **A1c
submitted**: l4h512, 8k updates, step at 5600, seeds 0 and 1
(22527728 / 22527731). Also `eval_goal_pairs.py` + launcher, so any
checkpoint can be enumerated after the fact; `eval_all`/`jsonable`
moved to `training/goal_pairs_setup.py` so the two CLIs share them
without one importing the other.

**A1b seed 1 (22526923) reproduces both halves.** 0.34° at u=6000, then
a much sharper blow-up at u=6250 (loss 0.40, error 40°), recovering to
2.1° by 8000. Best by rule: u=5500, 0.345° tt / 0.334° rr. The
instability is reproducible and seed-independent in timing (~6000
updates) — Adam at a sharp minimum with a fixed lr, a single overshoot.
A1c's step at 5600 lands just before it.

**l5h768, 8k, seed 0 (22526924), 3.0M params:** 0.68 → 0.39 → 0.32 →
**0.25° at u=6000**, then the same blow-up (7.2° at u=7000, 2.9° final).
Three for three on the ~6000-update instability. The bigger model
reaches lower before it goes. A1c extended: l5h768 with the step decay,
seeds 0 and 1 (22528824 / 22528827).

**Enumerated u6000 checkpoint, l4h512 seed 0 (eval job 22527723):**

| | train | goal_heldout | region |
|---|---|---|---|
| heldout, start=train | 0.4 | 0.4 | 0.4 |
| heldout, start=region | 0.4 | 0.4 | **0.4** |

Held-out region×region: **mean 0.38°, median 0.29°, 100% within 30°**,
24,960 pairs, per-env std 0.0. Train region×region 0.36°; gap 0.02°.
This is the current headline, pending A1c's stable finals.

## 2026-09-10 — A1c: step decay gives a stable final

**l4h512, 8k updates, lr ×0.1 at 5600, seed 0 (22527728).** At the step:
0.42° → 0.30°; then monotone to **0.27° at u=8000**, no blow-up.
Enumerated final, held-out region×region: **mean 0.27°, median 0.20°,
100% within 30°**; train 0.245°; `same` 0.239°; env-side gap 0.03°.
The schedule turns the mid-run pick into a stable final and improves on
it (0.38 → 0.27).

**l4h512 step, seed 1 (22527731):** heldout rr **0.28° / median 0.21°**,
stable. Seeds agree to 0.01°.

**l5h768 step, seed 0 (22528824):** stable, monotone; final held-out
region×region **mean 0.23°, median 0.17°**, 100% within 30°. Train 0.19°,
`same` 0.19°, env-side gap 0.04°. Best final.

**l5h768 step, seed 1 (22528827):** 0.25° / median 0.17°. All four A1c
finals, held-out region×region:

| config | s0 | s1 | median |
|---|---|---|---|
| l4h512 step | 0.270 | 0.279 | 0.20 |
| **l5h768 step** | **0.233** | **0.246** | **0.17** |

## A1 — the table (written to the plan's §0)

Best model: 5 × 768 relu, 8000 updates, lr 1e-3, ×0.1 at 5600. Held-out
envs, every pair enumerated, mean degrees (s0 / s1; NN line):

| start \ goal | train | goal_heldout | region |
|---|---|---|---|
| train | 0.24 / 0.25 (0.0) | 0.23 / 0.25 (62.5) | 0.23 / 0.25 (58.2) |
| region | 0.23 / 0.26 (63.3) | 0.22 / 0.27 (79.8) | **0.23 / 0.25** (82.9) |

Discrete: 1.000 on every held-out cell (mlp-2 and mlp-4, both seeds).

**Grid mode generalizes.** The memoryless network computes the direction
between two grid codes it has never seen — cells held out from both
starts and goals, in scaffold regions and walls it never trained on — to
a quarter of a degree, with no interpolation route (the lookup line is
58–83°). Every cell of the table is the same number: no side of the pair
is harder than the other.

What moved it, in order: depth (l4h256 < l2h1024 per parameter), 4×
updates (0.84 → 0.32), and a late step decay (necessary: constant-lr runs
destabilise at ~6000 updates, three for three; cosine-from-the-start
hurts because they are still descending). Not moving it: weight decay.
Hurting it: tanh (10× worse).

Runs: 15 (A1 wave) + 3 (A1b) + 4 (A1c) + 1 eval = 23 GPU jobs, ~4 h
wall, ~8 GPU-h. Checkpoints under
`CLS_RUNS/agent_ckpts/goal_pairs_a1c_cont_l5h768_u8k_step_s{0,1}_*/`.

**Consequence for B.** A is at the ceiling in every grid-mode cell, so
B's grid arm has nothing to add (§4.5). B is worth running only in
regular mode, only if A2 fails there.

## 2026-09-11 — A1x: corner placement

Added to the plan as A1x (§2.4, §3.5, §6.2, P10). A1's `place = held_out`
scatters training envs over the whole scaffold; every per-module bump
position is seen and only cross-module combinations are new, at random.
A **corner** confines every training env to `rect:0,0,K,K` and mints the
test set outside it (`place = ood`, `OutsideRect` with the margin). Inside
a contiguous corner the phase triples are correlated in a way that does
not hold elsewhere, so a corner offers a shortcut that scattered
placement does not — and then tests whether it was taken.

Code: `--place_region` and `--n_ood_place` on the trainer; a fourth env
set `heldout_out`; `base_val` renamed `heldout_in` (new walls, same phase
stretch) so the phase effect is separated from the wall effect; a launch
gate that every minted box clears the rect by ≥ margin on the torus.

World check on a CPU node: K=400 holds 64+4 envs at margin 20 (train
offsets ≤ 379; out-of-corner envs at e.g. (339, 977), (1191, 1624)).
K=120 cannot hold 6+4; **K=160 holds 12+4**. So: K=400 with 64 envs (23%
of one axis) and K=160 with 12 envs (9%).

**A1x submitted** (22588476 / 79 / 81 / 83): A1's best config (l5h768,
8k, step at 5600), K=400 × 2 seeds and K=160 × 2 seeds, 16 `heldout_out`
envs each.

**A1x K=400 — P10 falsified.** Both seeds, u=8000, held-out region×region:

| | seed 0 | seed 1 |
|---|---|---|
| train | 0.16 | 0.18 |
| `heldout_in` (new walls, inside corner) | **0.21** | **0.30** |
| `heldout_out` (outside corner) | **44.5** | **44.0** |

Same weights, same cells, same wall novelty — the only difference between
`heldout_in` and `heldout_out` is *where on the scaffold the env sits*.
Inside the corner the network is at A1's level; outside it is halfway to
random (median 29.8°, 55% of pairs within 30°, per-env std ±19° — some
outside envs are fine, some are near-random). The OUT curve fell 58 → 44°
over 8000 updates and is not converging.

**The network did not learn the phase-difference function.** It learned
something that works within the corner's stretch of the CRT cycle and
does not transfer. A1's scattered `place = held_out` passed because
scattered training covers the cycle; held-out envs interleave with
training envs in phase space and the network interpolates between them.
That is the "interpolates across the scaffold" reading, not "learned the
function". The per-env variance outside is the fingerprint: envs whose
phase triples happen to resemble the corner's do well.

This reverses the A1 conclusion as I wrote it in the plan's §0. What A1
established is that a memoryless MLP generalizes to unseen cells and
unseen *combinations* when training covers the cycle — not that it
learned the translation-equivariant displacement map the attractor
hand-builds. The corner is the test that separates the two, and it was
not in the original plan.

**A1x K=160, seed 0 (22588481)** — 12 train envs, 9% of the cycle:

| region×region | mean | median | frac<30 | env-std |
|---|---|---|---|---|
| train | 0.4 | 0.3 | 1.00 | 0.1 |
| `heldout_in` | 12.8 | 2.5 | 0.89 | 16.1 |
| `heldout_out` | **71.2** | 65.1 | 0.34 | 15.9 |

Outside: essentially random (random is 90°, and 0.34 within 30° is the
random floor). Inside, with only 12 envs, even new walls in the same
corner degrade — median 2.5° but a few envs fail badly (env-std 16°).
Monotone in cycle coverage: 100% (A1 scattered) → 0.23°; 23% (K=400)
→ 44° outside; 9% (K=160) → 71° outside. There is no extrapolation; the
network's competence is bounded by the stretch of the cycle it was shown.

**K=160 seed 1 (22588483):** in 3.3°, out **57.4°**. All four A1x runs:

| run | train | in | out |
|---|---|---|---|
| K=400 s0 | 0.19 | 0.21 | 44.7 |
| K=400 s1 | 0.21 | 0.30 | 44.0 |
| K=160 s0 | 0.41 | 12.8 | 71.2 |
| K=160 s1 | 0.39 | 3.3 | 57.4 |

Seed spread is wider at K=160 (with 12 envs, which ones fall in the
corner matters more) but the direction is the same in every run. A1x
complete. 4 runs, ~1.5 GPU-h.

## 2026-09-11 — A2 and B1x

**A2 launched** (22599345 / 46 / 47 / 49): regular mode on the standard
scattered world. No corner: `[omni(p), omni(g)]` is a ray-cast of the
wall and scaffold position never enters the input, so `place` is a
no-op there and the test is `wall = held_out`, which every run has.
l5h768 step × 2 seeds continuous, l4h512 step continuous and discrete.
First signal at u=3750, l5h768 s0: train 3.0°, **heldout 6.3°** on new
walls, NN line 38°. P3 (near-random on held-out walls) is already
failing; P11 written into the plan before the run finished.

**B built** (§5.8, with one deviation). Per-row goals as a `_goals`
(B, 2) array *beside* the scalar `_goal` rather than replacing it —
five existing readers do `(vec._goal[0], vec._goal[1])` and would have
silently read the first row. `at_goal`, the oracles and
`goal_channel_vec` are row-wise; a goal pool on the vec is applied on
every reset; the collector reads `vec._goals` fresh each step and passes
the goal channels. Bit-identical under defaults (golden fixture green).
`evaluation/lifetime.py` is readout 2: sampled action, `(episode ×
step)` table with counts, live-row mask on both score and `h`.
`train_goal_lifetimes.py` is B's own composer on `build_env_sets` —
`train_rnn.py`'s mixed mode redraws envs through the legacy builder,
which cannot place them in a declared region, and the corner world is
the point. Fresh lifetimes start from explicit zeros so
`bc_rnn_update`'s all-or-nothing `initial_h` guard is satisfied under
round-robin. Smoke-tested end to end on a toy world.

**B1x launched** (22599732 / 33 / 34): `full` (GRU 1×512 + prev_action),
`rec` (GRU, no prev_action), `dist` (MLP 5×768, A1's architecture, on
rollout data), all on `rect:0,0,400,400` with 16 `heldout_out` envs,
seed 0. 2000 updates, 8 envs × 64 lifetimes per update, 64-step chunks,
32 per lifetime, goal resampled on reach, lifetime eval every 500 on 8
held-out envs × 64 lifetimes × 20 episodes.

**A2 l5h768 seed 0 (22599345), enumerated FINAL on held-out walls:**

| start \ goal | train | goal_heldout | region |
|---|---|---|---|
| train | 5.5 (nn 0.0) | 5.4 (nn 24.0) | 5.5 (nn 25.5) |
| region | 5.6 (nn 25.7) | 5.5 (nn 36.7) | **5.6** (nn 38.4) |

Train 2.3°. Every held-out cell equal; a 3° gap from train to new
barcodes. **P3 falsified** — held-out walls are at 5.5°, not near
random. The memoryless MLP learns the observation→direction map of a
barcode it has never seen, from 480 ±1 ray values per cell, with no
ray-axis architecture. Still descending at 8000 (7.3 → 5.4 over the
second half); a longer run would close more of the gap. Other three A2
runs and all of B1x queued behind the QOS cap.

**A2 complete.** Held-out walls, region×region:

| run | train | heldout | median | frac<30 |
|---|---|---|---|---|
| l5h768 cont s0 | 4.3 | **5.65** | 4.3 | 0.994 |
| l5h768 cont s1 | 4.7 | **5.35** | 4.1 | 0.995 |
| l4h512 cont s0 | 2.8 (tt) | 6.3 | — | — |
| l4h512 disc s0 | — | **0.991** | — | — |

Regular mode generalizes to never-seen barcodes at ~5.5° / 0.99. Same
shape as A1 on the scattered world: every held-out cell equal, a small
train→heldout gap (3°), depth helps. The wall holdout is not the corner
holdout — there is no "corner" of barcode space — so this is the
regular-mode result, full stop: the memoryless MLP learns
observation→direction for a barcode it never saw, with no ray-axis
architecture. A3 (encoders) is unnecessary. 4 runs, ~1.3 GPU-h.

**B1x `full` (22599732), first lifetime eval at u=500.** eps/chunk 1.0
→ 5.2 by u=200 as the policy learns to reach. Readout 1 (h=0): train
48.8°, heldout_in 48.4°, heldout_out 75.5°. Readout 2, by episode:

| | e0 | e1 | e2 | e4 | e9 | e19 |
|---|---|---|---|---|---|---|
| heldout_in | 17.5 | **10.7** | 10.1 | 9.9 | 10.0 | 10.0 |
| heldout_out | 76.4 | 73.5 | 77.6 | 73.4 | 68.9 | 72.7 |

Inside the corner the recurrent net uses history hard: a 7° drop after
one episode, then flat — and readout 1 at 48° says that at u=500 it is
leaning on carried state more than on the static map. Outside the
corner: flat and near-random across 20 episodes. No in-context
local-frame estimation, at this point in training. Early; static is
still moving.

**B1x `full` FINAL (u=2000).** Readout 1, enumerated, h=0, every cell:
train 22–23°, heldout_in 24°, **heldout_out 44–47°**. The recurrent
net's static map outside the corner is the same 44° A1x's memoryless
MLP hit. Readout 2, by episode (20 episodes):

| | e0 | e1 | e2 | e5 | e10 | e19 | slope |
|---|---|---|---|---|---|---|---|
| heldout_in | 11.3 | 9.1 | 8.4 | 8.4 | 8.5 | 8.5 | 0 |
| heldout_out | 59.4 | 59.9 | 61.0 | 64.1 | 59.1 | 54.9 | **−0.24°/ep** |

Flat outside. No map accumulates across episodes. But the by-**step**
marginal is where the mechanism shows:

| step within episode | 0 | 1 | 2 | 5 | 10 | 20 | 40 | 59 |
|---|---|---|---|---|---|---|---|---|
| heldout_in | 27 | 9 | 7 | 7 | 8 | 18 | — | — |
| heldout_out | 46 | **31** | 31 | 34 | 43 | 76 | **110** | 113 |

Outside the corner the net is *better than its static map* for the
first ~5 steps of an episode (46° → 31°: it uses the `prev_action` and
the `Δgbook` it just saw), then drifts, and on the episodes that run
long it is worse than random — 110° at step 40 is an agent that has
lost its heading and is systematically pointing away. Inside, the same
first-steps refinement, no collapse, no episode lasts past ~25 steps.

**This is the middle row of the plan's three-shape table (§4.4):
episode-local history, not a map.** The GRU refines within an episode
from the last few observations and forgets at every reset. It does not
estimate the local frame of a new region and keep it.

**B1x `rec` FINAL (22599733):** identical to `full` at every checkpoint.
R1 heldout_out 44.9°; R2 by episode 53–65° flat, slope −0.25°/ep;
by-step 43 → 31 → 35 → 77 → 97. `prev_action` contributes nothing.

**B1x `dist` FINAL (22599734), MLP 5×768 on rollout data:** R1 train
0.6°, heldout_in 0.6°, **heldout_out 22.5° ± 12** (region×region mean
24°, **median 6°**, 74% < 30%). Trajectory 35 → 34 → 28 → 26 → 22.5
over 2000 updates, still falling. R2 outside flat at 70–76° (sampled;
the MLP's log_std head is uncertain there and sampling scatters it —
the §5.2 effect in the other direction). R2 inside 6° flat from
episode 0: no drop, because no memory.

**B1x complete — three arms, one seed:**

| arm | R1 heldout_out | R2 by episode | slope |
|---|---|---|---|
| A1x ref (MLP, i.i.d.) | 44° | — | — |
| dist (MLP, rollouts) | **22.5°** | flat 70–76 | 0 |
| full (GRU + pa) | 44.5° | flat 55–65 | −0.24 |
| rec (GRU) | 44.9° | flat 53–65 | −0.25 |

Findings: (1) no arm builds a map across episodes — every slope is
zero; the GRUs' one-episode dip is episode-local refinement that
collapses on long episodes. (2) The rollout data distribution halves the
MLP's out-of-corner error with the same architecture, 44 → 22.5, most
outside pairs near-solved and a quarter not. (3) The 1-layer GRU is a
worse function approximator than the 5-layer MLP on the same data and
does not earn it back through memory. P12 falsified for `full`;
confirmed for `dist` in the static-map sense. Written to the plan's §0.
3 runs, ~1.6 GPU-h. B's grid arm on the corner: done.

**Open, not run:** a deeper GRU (the arms were 1×512 against the MLP's
5×768 — not matched); longer BPTT windows; and whether `dist`'s 22.5°
keeps falling with more updates. All three are follow-ups, not part of
the question asked.
