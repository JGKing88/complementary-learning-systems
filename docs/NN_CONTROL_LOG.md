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

## 2026-09-12 — D1 and A1y: why does the rollout-trained MLP extrapolate better?

Two tests, plan §6.2 (D1, A1y) and P13. The sampler check first: at
size 20, i.i.d. pairs have mean `|g − p|` 9.4 with 22% at `d ≤ 5` and 48%
at `d ≥ 10`; the trajectory sampler flips that to 6.3, 50%, 22% (`d = 1`
goes from 2% to 11% of pairs).

**A1y — trajectory-shaped pairs through A's trainer** (`--pair_sampler
trajectory`, A's `1 − cos` loss, A's step schedule, K=400, l5h768):
heldout_out **39.4° / 43.9°** (two seeds). A1x's i.i.d. was 44.7 / 44.0.
**P13 falsified for A1y.** The displacement-weighting and goal-fixed
structure of rollout data does *not* explain B-dist's 22.5°.

**D1 — A1x's out-of-corner error by Chebyshev `|g − p|`**, enumerated:

| d | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| out | **14** | 13 | 16 | 22 | 32 | 43 | 53 | 60 | **63** | 62 | 56 | 46 | 36 | 30 | 30 | 45 |
| in | 0.4 | 0.3 | 0.3 | 0.3 | 0.2 | … | | | | | | | | | | 0.5 |

Non-monotone. Outside the corner the memoryless MLP extrapolates at short
range (14° at `d = 1–2`), fails in the mid-range band `d ≈ 6–11` (peak
63° at `d = 9`), and **recovers to ~30° at `d = 13–15`** — where the
modules (λ = 11, 12, 13) wrap and the phase-difference pattern becomes
locally similar to short-range again. So the network learned the code's
periodicity; what does not transfer is the mid-range Chinese-remainder
disambiguation. The 44° mean is dominated by `d = 6–12`, where i.i.d.
sampling puts most of its mass.

**D1 — B-dist's profile beside A1x's:**

| d | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 9 | 10 | 12 | 14 | 16 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1x (i.i.d., 1−cos) | 14 | 13 | 16 | 22 | 32 | 43 | 60 | **63** | 62 | 46 | 30 | 34 | 45 |
| B-dist (rollouts, Gaussian NLL) | **1.6** | **5** | 12 | 22 | 27 | 28 | 25 | 23 | 23 | 24 | 20 | 26 | 27 |

Two different things. (1) At `d ≤ 2` B-dist is near-perfect outside the
corner where A1x is 14° — a real short-range gain, and A1y says it does
not come from the pair distribution through A's loss. (2) The mid-range
CRT band is **flattened, not solved**: A1x's 63° peak becomes a flat
~25° from `d = 4` to `d = 19`. That is not better disambiguation; it is a
**hedge** — a network that is uncertain and outputs a conservative
average rather than committing to the wrong CRT branch. A Gaussian NLL
with learnable σ does exactly that: where the target is unpredictable
the optimizer raises σ and pulls μ toward the mean, whereas `1 − cos`
has no escape and fits the corner's shortcut with confidence. The
mean/median gap on B-dist's region×region (24° vs 6°) is the same
signature: many near-perfect pairs, a tail of hedges.

**Verdict:** the loss, not the data. B-dist's 22.5° mean is partly a real
short-range gain and partly a hedge that scores better on a mean-angle
metric without knowing the answer. P13 half right (loss), half wrong
(distribution). The remaining open question — whether the short-range
gain is *also* the loss, or something about DAgger's off-path states —
is one run: A's trainer with a Gaussian-NLL head. Not run.

D1: 2 eval jobs. A1y: 2 runs, ~1 GPU-h.

## 2026-09-13 — D2: is A2 real, and how does it work?

Two ways a network could hit 5.5° on an 80-bit wall it never saw: (a)
invert the ray projection view-by-view — run-structure → position,
wall-independently — then subtract; (b) match `omni(p)` against `omni(g)`
through the bits they share. `--mismatched_walls` separates them: feed
`omni(p)` from held-out wall *i* and `omni(g)` from held-out wall *j ≠ i*.
Under (a) the direction survives; under (b) it collapses.

**A2 l5h768 s0, held-out envs:**

| | error |
|---|---|
| matched, same wall (16 envs, per-env 4.2–7.4) | **5.5°** |
| mismatched, wall *i* for p and wall *j* for g (240 ordered pairs) | **22.2° ± 5.5** |
| random | 90° |

**And by displacement, matched:** 5.5° at every `d` from 1 to 19 — flat
to the decimal. No range structure at all, unlike grid mode.

**Reading.** Both mechanisms, with (a) carrying most of the load. With
the shared wall removed the network still gets the direction to 22° from
two views of *different* unseen barcodes, and the only thing those views
share is the ray geometry — so it localizes each view on its own. The
shared bits are worth a further 17°, so it uses them too. The flat
by-distance profile is the signature of decode-then-subtract: the
subtraction does not care how far apart the cells are.

**How it can work at all:** the bits are random, the projection is not.
Rays hitting one segment return one bit, so a view is a sequence of runs
whose boundaries sit at angles fixed by position (a boundary 3 cells away
spans more rays than one 15 away; lateral offset shifts the pattern).
Half the boundaries are invisible (adjacent segments share a bit with
p = 0.5), but four views give ~20–30 visible ones per cell — C2 measured
zero exact twins. Position is identifiable from a single cell's `omni`
without knowing the wall, and the network learned the map.

**Leak audit, for the record:** C1 (train/held-out wall seeds disjoint,
expected Hamming distance 40 of 80 bits; region cells never starts or
goals), C2 (zero twins), C6 (`pair_inputs` bit-identical to the rollout
stack's input), C11 (reference lines on the same enumerated pairs). The
within-env NN lookup line on held-out region×region is 38°. Nothing
per-wall transfers; nothing is looked up. 1 eval job.

## 2026-09-13 — B2: build and gates

**Build (a9d5ac4).** Plan §5.12, as-built notes in §5.12. Unit gates:
B2-C1 `gbook_at(θ=0, s=1)` equals `smooth_gbook` gathered at 300 random
scaffold positions with max |diff| **0.0**, and the one-hot at fwhm 0;
B2-C2 the per-module centroid shift for a unit step equals `R_θ a / s`
to **5e-7 cells** over 100 random (θ, s, position) (bound was 0.05); the
CRT decode recovers rotated displacements up to |d| = 19 to 1e-6;
`ScriptedFrameAgent` recovers θ to < 1° and the direction to < 1° on 32
synthetic lifetimes with and without `prev_action` in the layout, and
retries a clipped first step; the oracle channel is last in the layout
and `pair_inputs` matches `build_rnn_input`; `with_lattice(0, 1)` is the
identity; the collector and the evaluator on `gbook_table` at (0, 1)
reproduce the `sgb` path bit for bit. 11/11, plus the 78 existing
goal-pairs / layering / lifetime tests unchanged.

**CPU smoke, 8×8 toy world (lambdas 5 6 7).** The estimator through
readout 2, `heldout` set, episode 0 by step: **96 → 67 → 13 → 0.0 → 0.0**
— two measuring steps, then exact; by episode 30 / 1.7 / 5.4 / 8.3
(residual is the goal-cell/L2-ball artifact below and 8×8 clipping). The
GRU `full` arm on random lattices trains and evaluates on `heldout`,
`same`, `heldout@45`; θ histogram: 0 draws in the held-out band. The
oracle MLP path trains and evaluates on `heldout@45`.

**One evaluator property, noted for reading every B curve.** Continuous
at-goal is an L2 ball of radius 0.5 on the continuous position, so a row
can stand on the goal cell (which is all a cell-resolution code shows)
without being at goal. The teacher there is the zero vector and every
agent scores 90° on that step. B1x was scored the same way. A
deterministic agent that decodes Δ' = 0 and emits a zero action would sit
there until timeout; the estimator now takes a random unit step in that
case (a sampled policy does this by itself). Not changed in the
evaluator, to keep B1x comparable.

**Gates submitted, 3 jobs.** B2-C3 oracle-θ MLP l5h768, 8000 updates,
step at 70%, θ per env per update from the training set, eval on the
standard lattice (θ = 0), θ = 7° (inside the band) and θ = 45° (a
training θ) (22688913); its no-oracle twin on the same data, the i.i.d.
form of B2-C5 — must sit at ~90° everywhere (22688914); B2-C4 the
scripted estimator through readout 2 on the real world, 64 lifetimes ×
20 episodes on 8 envs of each of `heldout`, `same`, `heldout@7`,
`heldout@45` (22688915).

**B2-C3 PASSES — oracle-θ MLP l5h768 (22688913), 8000 updates, 27 min.**
Enumerated final, held-out envs, region×region: **1.0°** on the
standard lattice (θ = 0, never trained: the band |θ| < 15° was
excluded), **0.9°** at θ = 7° (inside the band), **0.6°** at θ = 45° (a
training θ); train 0.9–1.0°. Trajectory on held-out θ = 0: 89° at
u = 200, 4.4° at u = 1000, 2.4° at u = 2000, 1.2° at u = 5000, 0.95°
at u = 8000. Every piece of §4B.2 — continuous-phase synthesis, the
rotation convention, the CRT within an env, the arena-frame teacher — is
consistent, and interpolation over θ into the held-out band costs
nothing. Told the frame, a memoryless net solves it to A1's precision.

**Its no-oracle twin (22688914), same data:** loss 0.99 and grad-norm
0.00 from u = 100 to the end; enumerated final 78–101° across quadrants
(a constant output), region×region **90.0°** on every lattice. The
i.i.d. form of B2-C5: with the frame withheld the input is undetermined
and a memoryless net has nothing to learn. Rotation leaks through
nothing.

*Two things seen on the way.* (1) The oracle run sat at loss ≈ 1 with
grad-norm 0.01 for the first ~200 updates, numerically identical to the
null — a plateau of the `1 − cos` loss where the output norm inflates
and the tangential gradient shrinks as 1/‖v‖. It escaped on its own by
u = 1000; a toy-world probe (12×12, λ = 7 8 9, l4h512, batch 16 × 128)
stayed on it for 600 updates while an MSE-to-unit-vector loss learned
(9.5° at u = 600). At the real batch size no change was needed and none
was made. (2) The nearest-neighbour reference line on held-out
region×region is 58–92° at θ = 0 and 7° but **5–7° at θ = 45°**: at 45°
adjacent cells are 1/√2 apart in phase per axis, the bumps overlap, and
the nearest *training* cell to a region cell is a true neighbour rather
than the 12-cell alias. A property of the code's effective resolution
at that orientation, not a leak (the model is at 0.6° either way).

**Wave 1 submitted:** `dist` (22689649), `full` 2×512 s0/s1 (22689650,
22689651); the scripted estimator (22688915) is running.

**B2-C4 PASSES — scripted two-step estimator (22688915), readout 2, 64
lifetimes × 20 episodes on 8 envs per set, sampled-policy protocol
(the estimator is deterministic except for its zero-decode step).**

| set | ep0 s0 | s1 | **s2** | s3 | s5 | s10 | by episode e0 | e1 | e2–e19 |
|---|---|---|---|---|---|---|---|---|---|
| heldout (θ = 0) | 91.7 | 87.8 | **5.4** | 0.1 | 0.0 | 1.3 | 15.6 | 2.1 | 1.5–2.3 |
| same (θ = 0) | 91.7 | 87.8 | 5.4 | 0.1 | 0.0 | 1.5 | 15.5 | 1.7 | 1.4–2.4 |
| heldout@7 | 91.7 | 87.8 | 5.4 | 0.1 | 0.0 | 1.3 | 15.6 | 2.0 | 1.8–2.3 |
| heldout@45 | 91.7 | 87.8 | 5.4 | 0.1 | 0.0 | 1.2 | 15.6 | 1.6 | 1.5–2.2 |

Two measuring steps (~90°, it is acting along the arena axes and has no
direction yet), then **exact from step 2 of the lifetime on**, and flat
across 19 goal changes at ~2°. The 2° residual and the s20 values
(5–15°) are the goal-cell/L2-ball steps noted above — a row standing on
the goal cell outside the 0.5 ball scores 90° whatever it does — plus
the random escape step. The curve is the same on every lattice, as it
must be for an agent with no training set. Pass criterion was ≤ 5° from
step 2 on; s2 = 5.4 is the step the frame is first used (the 5° is the
few rows whose measuring step was clipped or teleported and retried),
s3 = 0.1.

**Read together with C3:** told the frame, a memoryless net is at 1°;
the frame is measurable from two steps of the trajectory to < 1° and
lasts the lifetime. The design is sound. Whatever the GRUs do on the
held-out lattice is a statement about the GRUs.

**Wave 1 (rotation only) — stopped at u = 3000: a leak in the design.**
`full` 2×512 s0 (22689650): 88–90° flat on both readouts at u = 1000,
2000, 3000, goal rate 0.001, loss 2.0–2.1 — nothing learned, the
chicken-and-egg plateau (plan §8). But `dist` (22689649), the memoryless
MLP on the same rollouts, moved: loss 3.44 → 1.86 (u = 1000) → 1.00
(u = 2000) → **−0.19 (u = 3000)**, goal rate 0.001 → 0.04 → **0.07**,
episodes/chunk 1.0 → 4.5. A memoryless net cannot predict the direction
under a genuinely unknown θ — its best guess is `R_θ̄ᵀ Δ'` for the
training set's mean rotation θ̄ = 180° (the excluded band is around 0),
worth ~0.01 nats — so 3.6 nats of improvement on training lifetimes
means the input *does* determine θ there. It does: the training envs
sit at 64 fixed scaffold offsets `O`, and the absolute phases
`R_θ(O + p) mod λ_m` across three modules pin θ once `O` is memorised.
The held-out readouts were still nulls — `dist` at θ = 0 was 97–124°,
the θ̄ = 180° anti-alignment showing through on new offsets it cannot
place — but a weights route to θ on the training data makes in-context
estimation unnecessary in training, which is the structural problem B2
exists to remove. Both runs cancelled at u = 3000/3300; a diagnostic
(22690872, `eval_goal_pairs --eval_thetas 45,90` on `dist`'s u = 3000
checkpoint: train@45 vs heldout@45) queued to measure the route
directly.

**Fix (3b3ba15): per-lifetime lattice translation**, `T ~ U[0, 1716)²`
after rotation, so absolute phases are uniform for every θ and only
phase differences carry information (plan §4B.2, as amended). Tests:
absolute phases shift by `T mod λ_m`, pairwise differences invariant to
1e-3, the scripted estimator unaffected, the sampler uniform over the
period. Default on; `--no-lattice_translate` reproduces the runs
above. Wave 1 resubmitted on the corrected design: `dist` 22691028,
`full` s0/s1 22691029/22691030; C3 oracle and null rerun with
translation (22691032/22691033) so the gate matches the arms' design.
The scripted estimator (C4) uses phase differences only and needs no
rerun.

**Diagnostic (22690872), `dist` u = 3000 (rotation only), readout 1
enumerated, train × train:** train 107°, heldout 97°, **train@45 114°,
heldout@45 124°**, train@90 105°, heldout@90 92°. No general offset → θ
map: on its own training envs at a training orientation the memoryless
net is as lost as anywhere (and anti-aligned — the θ̄ = 180° guess). So
the −0.19 training loss (≈ 18° on the rollouts it was scored on, at
σ ≈ 0.37) is not a route it can reuse; it is **within-lifetime weight
memorisation**: each env's lattice is fixed for 32 chunks, the env is
picked every 8 updates, so one code → direction map is fitted across
~256 consecutive updates, 64 maps at a time, and dropped at turnover.
Translation alone does not touch that. **Second design change
(af2adda): one lattice per row**, i.e. per lifetime — 4096 concurrent
lattices, each on one row's data, past what the weights can fit while
it lasts. `--lattice_per_row` default on; the per-env translated wave
(22691028–30, up to u = 400) cancelled unread; wave 1 resubmitted a
third time on the per-row design: `dist` 22691259, `full` s0/s1
22691260/22691261. The C3 reruns (22691032/33, A's trainer: θ per env
per *update*, each lattice seen once) are unaffected and stay queued.

The lesson, for the record: a randomisation that is fixed for longer
than the weights take to fit it is not a randomisation. B1x's fixed
lattice was the extreme case; a per-env per-lifetime lattice was a
milder one; per-row is the first version in which the training data
never hold a lattice still long enough to be learned in weights.

**C3 on the corrected design (22691032/33, translation on):** oracle
**1.2°** on held-out θ = 0 (1.05° at 7°, 0.7° at 45°), a shade slower
than without translation (9.7° vs 4.4° at u = 1000, same by u = 3000);
null 89–90° with zero gradient throughout. The gate holds.

**Per-row design, wave 1 at u = 2000.** `dist` (22691259): loss
3.47 → 2.15 (u = 200) → **2.14 (u = 1000) → 2.04 (u = 2000)**, goal
rate 0.001 throughout, readout 1 at 90 ± 1° on every set, readout 2 at
90° (u = 1000) then ~100° (u = 2000, the θ̄ = 180° guess emerging). The
within-lifetime memorisation is gone: with 4096 concurrent lattices the
memoryless net has nothing to fit. This is gate B2-C5 in its proper form.
`full` 2×512 s0 (22691260): loss 3.31 → 2.13 (u = 1000) → 2.05
(u = 2000), goal rate 0.001, readouts 88–89° flat on every set and every
step — indistinguishable from the null. The plateau, on a clean design.
Decision point per §6.2 is u = 8000.

**Per-row design, wave 1 FINAL (8000 updates, step at 5600).**

| arm | loss (u=8000) | goal rate | R1 heldout tt | R2 heldout e0 / e1 / e19 | ep0 by step s0 / s2 / s3 / s5 / s10 |
|---|---|---|---|---|---|
| `dist` MLP 5×768 (22691259) | 2.15 | 0.001 | **88.9** | 89.9 / 90.1 / 90.2 | 91.7 / 91.5 / 87.0 / 85.9 / 87.2 |
| `full` GRU 2×512 (22691260) | 2.12 | 0.001 | **90.2** | 88.6 / 88.8 / 89.0 | 91.6 / 90.6 / 86.1 / 85.5 / 85.8 |
| scripted estimator (22688915) | — | — | — | 15.6 / 2.1 / 1.9 | 91.7 / 5.4 / 0.1 / 0.0 / 1.3 |

`dist` is a clean null on every lattice (θ = 0, 7°, 45°; readout 1
88.9° on all five sets, readout 2 90.0 ± 0.5° at every episode): with
4096 concurrent lattices there is nothing for the weights to fit, and
nothing but the trajectory carries θ — **B2-C5 passes**. `full` is the
same curve: flat across 20 episodes on every set, no drop within
episode 0 (the 4–5° dip from step 3 on `heldout` is ~2 s.e. and absent
on `same`, `heldout@7`, `heldout@45`), loss never below the null's,
goal rate never above chance, for 8000 updates. 131,072 lifetimes
drawn, 0 in the held-out band.

**Reading (plan §4B.7, row 3):** the estimator shows that two steps of
the trajectory determine the frame to < 1° and that it lasts the
lifetime; the oracle shows a memoryless net does the rest to 1°; this
GRU at this budget found neither half. It is the chicken-and-egg of §8:
with θ unknown the gradient on the displacement decode averages to
zero, and with no decode there is no gradient on θ. **Not "impossible"
— "not learned from scratch at this capacity and budget."** The
foothold runs (§8 contingency (i)) are the next wave: anchor-mix
(22694701, running — half the lifetimes at the fixed training
orientation 90°, so the decode gets a determined target, the rest
random, test still θ = 0) and `full-cap` 3×768 (22695405, queued).
`full` s1 was cancelled unstarted in favour of these (two GPUs).

**Anchor-mix `full` (22694701), u = 2000:** flat at 88–90° on every
set — **including `heldout@90`, the anchor orientation**, where half
the training rows carry a fully determined memoryless target (fixed
θ = 90°, per-row translation). Loss 2.1, goal rate 0.001. `full-cap`
3×768 (22695405) at u = 1000: 89–90° flat too. So the block is below
the θ chicken-and-egg: **this GRU is not learning the
translation-invariant displacement decode from rollout data at all.**
What has been shown: B1x's GRU learned the *absolute* decode from
rollouts (fixed lattice); the A-trainer MLP learned the
translation-invariant decode from i.i.d. pairs (C3t, 1.2°). What has
not: any net learning the translation-invariant decode from rollouts.
`full-cap` cancelled (capacity is not what the anchor points at);
diagnostic queued in its place (22698232): `dist` — the MLP 5×768 that
reached 0.6° on B1x rollouts — on lifetimes *all* at θ = 90° with
per-row translation, 4000 updates. If it learns (`heldout@90` ≪ 90°),
the GRU is the bottleneck and the foothold + capacity line continues;
if it does not, the rollout regime (random-walk data, Gaussian-NLL
head, sampled DAgger) is, and B's trainer needs a direction head /
`1 − cos` loss before B2 can be read.

**Diagnostic `dist` @ θ = 90° (22698232, per-row translation, all
lifetimes at one training orientation), cancelled at u = 1000 as
answered:**

| u | loss | goal rate | R1 heldout@90 | heldout@45 | heldout (θ=0) |
|---|---|---|---|---|---|
| 200 | 2.15 | 0.001 | 88.7 | 89.0 | 89.5 |
| 400 | 0.43 | 0.009 | 45.2 | 55.9 | 88.8 |
| 600 | −1.76 | 0.082 | **1.1** | 45.1 | 89.9 |
| 1000 | −1.75 | 0.083 | **0.7** | **45.1** | **90.0** |

Readout 2 at θ = 90: 6° flat (sampled policy). The memoryless MLP learns
the **translation-invariant** displacement decode from rollouts in 600
updates — and its errors elsewhere are exactly |θ − 90°|: it decodes the
rotated displacement and applies the one rotation it was trained on.
The rollout regime (random-walk data, Gaussian-NLL head, sampled
DAgger) is not the block. **The GRU is**: the same determined target
that the 5×768 ReLU stack fits in 600 updates, the 2×512 GRU fed the raw
code has not fitted in 3000 (anchor-mix `heldout@90` 89.5 / 90.4 / 85.9
at u = 1000 / 2000 / 3000). B1x showed the same gap on the absolute
decode (GRU 22° vs MLP 0.6°): on this input a GRU is a poor function
approximator at its first layer.

**Encoder arm (1bb21f9).** `EncodedRecurrentCore`: the `dist` trunk
(5×768 ReLU) as a per-step encoder in front of the GRU, state shape
unchanged, off by default; `--encoder_layers 5 --encoder_hidden 768`.
Submitted: `enc-full` on the primary design (22699034) — the hypothesis
with the memoryless half of the computation given the architecture that
can do it — and `enc-full` anchor-mix (22699035) — the foothold. The
plain-GRU anchor-mix (22694701) runs on to 8000 for the record.

**Plain-GRU anchor-mix FINAL (22694701, u = 8000).** Readout 1:
`heldout@90` (the anchor) **72.1°**, `heldout@45` 79.1°, `heldout@7`
89.1°, `heldout` (θ = 0) **91.2°**; trajectory on the anchor 89.5 → 90.4
→ 85.9 → 86.0 → 83.4 → 75.3 → 70.9 → 72.1 (the fall came with the lr
step at 5600). Readout 2: `heldout@90` 83°, `heldout@45` 86°, `heldout`
90.5°, all flat across 20 episodes, no within-episode drop anywhere.
The plain GRU does learn the determined half — an order of magnitude
slower than the MLP and only to 72° by the end — and nothing
in-context: on the held-out lattice both readouts stay at 90°.

**First encoder runs (22699034 primary, 22699035 anchor-mix) — the
encoder died.** Both showed the constant-output signature (readout 1
identical to 0.1° across every set, primary loss 2.3–2.6 above the
null's 2.14 floor, anchor-mix `heldout@90` 89.3° at u = 1000 where the
MLP alone was at 1°). Probed the anchor-mix u = 1000 checkpoint on 2000
synthetic codes (`gbook_at`, random θ and shift): encoder ReLU layers
dead fraction **0.55 / 0.42 / 0.84 / 0.999 / 1.000**, output variance
0.0, policy mean constant (0.02, 0.07). The undetermined half's gradient
shrinks the stack's activations toward zero and the deep ReLUs die
before the determined half carves out the decode; the pure MLP `dist`
escaped only because 100% of its data was determined (`dist` on the
primary design, 88.9° on every set, died the same way — harmless for a
null). Both cancelled. **Fix (fa9844f):** LayerNorm before each encoder
nonlinearity (`FeedForwardCore(norm=True)`, off for the historical
trunk) and a skip from the encoder output to the heads beside the
recurrent output (`EncodedRecurrentCore(skip=True)`; `feature_size` ≠
state width, `RNNAgent` reads it), so the memoryless path has the short
gradient route the `dist` arm has. Resubmitted as `enc2`: anchor-mix
22702168, primary 22702190 — queued behind a 7-GPU sweep of Jack's on
the shared quota.

**`enc2` (LayerNorm + skip; 22702168 anchor-mix, 22702190 primary) —
collapsed again, differently.** Readout 1 identical across every set
from u = 200 on; `heldout@90` 89.1° at u = 1000. Probe of the anchor-mix
u = 1000 checkpoint: the encoder is alive in scale (mean activations
0.3–0.5, LayerNorm doing its job) but **input-independent** — std across
inputs 0.05 at layer 1 decaying to 5e-9 at the output; GRU output std
2e-9; the head's contribution std 1e-7 from both the recurrent and the
skip path. The stack has learned to ignore its input. With half the data
undetermined the pull toward "output the constant best guess" wins the
first 200 updates, before any decode exists; `dist@90`, with 100%
determined data, only began to move at u = 400. Both cancelled.

**Warm-up curriculum (8ffc035; plan §8 (ii)):**
`--lattice_mix_warmup_updates N` — every lifetime that starts in the
first N updates is on the fixed anchor lattice (θ = 90°, per-row
translation), then the configured mix. Two runs, `enc2` architecture,
N = 1500 (the decode formed by u = 600 for the MLP): then mix 0.5
(22702687, the foothold kept) and then mix 0 (22702689, the primary
design with a pre-formed decode). The readouts that matter: `heldout@90`
should be ≪ 90° by u = 1000 (the decode forming, as `dist@90`); after
the switch, `heldout@45` and `heldout` (θ = 0) are the in-context
question, with the memoryless prediction being |θ − 90°| = 45° and 90°.

**Warm-up runs (22702687 mix 0.5, 22702689 mix 0; identical through the
warm-up as expected) — collapsed on 100% determined data.** `heldout@90`
90.9° at u = 1000, identical across sets, loss 2.13 → 4.42 → 3.31 →
2.15 → 2.66. The pure MLP on exactly this data (`dist@90`) was at 1° by
u = 600. So it was never the undetermined target: **the GRU's BPTT
gradient into the shared encoder is what destabilises it** (the spikes)
and the encoder collapses to a constant to end the instability. Both
cancelled.

**Decoupling (38d60a3).** `EncodedRecurrentCore(detach=True)`: the core
reads the features but sends no gradient back — the encoder trains
through the skip alone, i.e. exactly as the `dist` arm. `bypass`: the
`prev_action` columns go past the encoder to the core (the action is not
something to decode, and a `dist` trunk never saw it). Trainer:
`--encoder_init <dist checkpoint> --encoder_freeze` (a pretrained, fixed
decode; the recurrent net and heads are all that train — 3.7M of 10.3M
params), `--encoder_detach`, `--no-encoder_norm`. Two runs:

- **A, frozen decode, primary design (22703355):** encoder = the
  `dist@90` trunk (u = 1000; it decodes Δ' for every θ — its errors were
  exactly |θ − 90°|), frozen; GRU 2×512 + prev_action + heads train on
  random-θ per-row lifetimes with translation, no anchor. The cleanest
  form of the in-context question: given features from which the
  rotated displacement is linearly decodable, can a recurrent net learn
  from `(Δ'_t, a_t)` history to un-rotate it? Memoryless prediction on
  the readouts: 90° everywhere (a fixed head can only apply one
  rotation).
- **B, detached encoder, anchor-mix 0.5 at 90° (22703357):** end to end,
  encoder trained like `dist` on the mixed data, GRU on top.

**RUN A — frozen decode + GRU, primary design (22703355), u = 1000 of
8000: IN-CONTEXT LEARNING OF THE UNSEEN LATTICE.** Held-out envs, the
standard lattice θ = 0 (never trained; the band |θ| < 15° excluded):

| readout | value |
|---|---|
| R1, h = 0 (no history) | **119.6°** — worse than random, the memoryless best guess (un-rotate by the training mean θ̄ ≈ 180°) |
| R2 by episode e0 / e1 / e2 / e4 / e9 / **e19** | 89.2 / 67.9 / 51.5 / 38.4 / 22.9 / **18.7** |
| R2 episode 0 by step s0 / s1 / s2 / s3 / s5 / s10 | 104 / 120 / 111 / 103 / 80 / 68 |
| `same` (training envs, θ = 0) e0 / e19 | 90.7 / 18.4 |
| `heldout@7` e0 / e19 | 83.9 / 17.3 |
| `heldout@45` (training θ) e0 / e19 | 51.2 / 9.6 |
| `heldout@90` (the decode's own θ) e0 / e19 | 34.4 / 8.5 |

Loss −1.70, goal rate 0.084 (≈ `dist@90`'s), 3.55M trainable params
(GRU + heads; the 5×768 trunk frozen at `dist@90` u = 1000). The
by-episode curve is monotone across goal changes — the frame is a
lifetime property and the GRU keeps refining it through the lifetime;
the by-step curve within episode 0 says it needs ~10 steps to get to
70° where the scripted estimator needs 2 to get to 5°. A recurrent net
trained from lifetimes, with the decode given, **learns the frame of a
grid code it never saw from its own trajectory**, to 18.7° after 19
episodes at a quarter of the run, against 90° memoryless and 44° for
the corner-trained weights' extrapolation (§4B.7 row 1). Leak check:
the decode was trained at θ = 90° with translation and knows nothing of
θ = 0; the GRU and heads saw only θ ∈ [15°, 345°]; test envs are new
walls at new offsets; `dist` on this design is 90°; and readout 1 —
the same network with no history — is 120°. Everything below 120° is
history.

Run B (detached encoder, anchor-mix, 22703357) at u = 800: 88–91°, the
encoder still forming.

**Run A at u = 2000 / 3000, held-out lattice θ = 0.** By episode:
47 → 34 → 27 → 22 → 15.3 → 15.7 (u = 2000); 50 → 40 → 26 → 18 → 16 →
**15.7** (u = 3000). Episode 0 by step at u = 3000: **s0 128 → s1 57 →
s2 54 → s3 49 → s5 38 → s10 29** — the frame is measured largely from
the first step (one `(Δgbook, prev_action)` pair) and refined over the
next ten; readout 1 (no history) has moved to **138°**: with nothing
observed the network commits hard to the training-mean orientation,
which on θ = 0 is nearly opposite, and one step of evidence flips it.
On the training orientations: `heldout@45` 53 → 8.2, `heldout@90`
55 → 8.2 by episode 19. Goal rate 0.083.

**Run B (detached encoder, anchor-mix 0.5; 22703357) — cancelled at
u = 1000:** 90.9° identical across sets; the skip-trained encoder
collapses on half-undetermined data as `dist` did on fully undetermined
data. The frozen-decode line is the one that works; B was redundant
with it and held a GPU the `rec` control needed.

**`rec` with the frozen decode (22704382), u = 1000:** by episode on
θ = 0: 96 → 92 → 80 → 65 → 47 → **33**; episode-0 by-step flat at
~90. Without `prev_action` it cannot read the frame from one step (it
knows its policy mean, not the sampled action) but it integrates over
the lifetime and gets there more slowly. `prev_action` is worth ~2× in
episodes at this point (A was at 18.7 by e19 at u = 1000). Seed 1 of A
(22704383) queued.

## 2026-09-13 — B2 RESULT: run A final

**Run A — frozen `dist@90` decode + GRU 2×512 + `prev_action`, trained
8000 updates (step at 5600) on random-θ per-row lifetimes with
translation, no anchor (22703355).** Held-out envs; the standard lattice
θ = 0 was never trained (band |θ| < 15° excluded); `same` = training
envs at θ = 0, unseen there too.

| readout | heldout θ=0 | same θ=0 | heldout@7 | heldout@45 | heldout@90 |
|---|---|---|---|---|---|
| R1, h = 0, enumerated, train×train | **150.0** | 150.1 | 149.9 | 134.0 | 92.7 |
| R2 by episode e0 / e1 / e2 / e4 / e9 / e19 | **32.0 / 27.6 / 26.5 / 21.9 / 17.7 / 14.5** | 31.8 / 30.3 / 25.4 / 26.0 / 20.6 / 13.5 | 32.4 / 34.5 / 25.6 / 26.2 / 18.9 / 12.0 | 27.7 / 28.8 / 20.6 / 14.6 / 11.4 / 8.3 | 24.2 / 26.7 / 16.6 / 13.8 / 10.9 / 7.6 |
| R2 e10–e19 mean | **14.2** | 14.1 | 13.6 | 8.5 | 7.7 |
| R2 ep0 by step s0 / s1 / s2 / s3 / s5 / s10 | **122 / 56 / 42 / 34 / 22 / 18** | 127 / 55 / 42 / 32 / 22 / 17 | 126 / 50 / 41 / 33 / 24 / 19 | 117 / 49 / 35 / 25 / 17 / 13 | 92 / 38 / 28 / 21 / 17 / 14 |

Loss −1.77, goal rate 0.083, 131,072 lifetimes drawn, 0 in the band.
Trajectory over training on θ = 0, e19: 18.7 (u = 1000) → 15.7 → 15.7 →
17.4 → 23.1 (u = 5000, drifting at lr 1e-3) → 14.6 (u = 6000, after the
step) → 15.5 → **14.3**; ep0 s10: 68 → 30 → 29 → 43 → 41 → 19 → 18 →
**17**.

**Reading.** The three reference lines on the same lattice: memoryless
null 90° (`dist`, C5); the same network with no history 150° (it
commits to the training-mean orientation, which at θ = 0 is nearly
opposite — the worst possible prior, and exactly what a memoryless
best guess under the training distribution is); the corner-trained
weights' extrapolation 44° (A1x, B1x). **From its own trajectory the
recurrent net gets the unseen lattice to 56° in one step, 18° in ten,
and 14° over the lifetime.** The frame accumulates across goal changes
(by-episode monotone) and is measured within an episode (by-step
monotone): the plan's row-1 reading of §4B.7, on the real code. The
scripted estimator's ceiling is 5° at step 2 and ~2° after; on the
*trained* orientations the GRU sits at ~8°, the floor of a sampled
policy (`dist@90` at its own θ: 6°), so the cost of the held-out band
is ~6° of interpolation and the cost of being a GRU rather than the
estimator is ~10 steps rather than 2. Leak audit as at u = 1000: the
decode was trained at θ = 90° only and never saw θ = 0; the GRU and
heads saw θ ∈ [15°, 345°]; test envs are new walls at new offsets; the
null on this design is 90°; readout 1 is 150°. Everything between 150°
and 14° is the trajectory.

**What did not work, for the record:** the GRU fed the raw code (89°
flat at 8000); the raw-code GRU with an anchor foothold (72° on the
anchor, 90° on θ = 0); a jointly-trained MLP encoder in front of the
GRU with or without LayerNorm, with or without a warm-up on determined
data (collapses to a constant — the GRU's BPTT gradient into the shared
encoder); a detached encoder on mixed data (collapses on the
undetermined half). The decode has to be learned memorylessly, once,
and given; the recurrence then learns the frame from lifetimes.

**`rec` with the frozen decode, FINAL (22704382).** Held-out lattice,
by episode: **100.9 → 97.9 → 95.8 → 91.5 → 90.9 → 88.9 → 85.7 → 85.6 →
82.9 → 78.3 → 75.7 → 72.0 → 69.6 → 65.5 → 61.9 → 58.1 → 55.0 → 53.7 →
49.8 → 44.7**; episode 0 by step flat at 98–105; readout 1 102.6. On
the trained orientations 107 → 80 (θ = 45) and 104 → 77 (θ = 90). The
third shape of the plan's §4.4 table, unblurred: **rises across
episodes, not within** — without `prev_action` the network cannot read
the frame from one `(Δgbook, action)` pair (it knows its policy mean,
not the action it sampled), so it integrates evidence across the
lifetime and needs ~20 episodes to get where `full` gets in ten steps.
Its training was also unstable at lr 1e-3 (33° at e19 by u = 1000, then
lost — 122° at u = 2000, 98–105 through u = 6000 — and back to 45° after
the step at 5600; final loss 1.96 against `full`'s −1.77). `prev_action`
is what turns frame estimation from a lifetime-scale integration into a
one-step measurement, as the estimator's algorithm says it should.

**Seed 1 of A (22704383) at u = 1000 / 2000:** held-out lattice by
episode 66 → 32 → 22 → 19 → 16 → 15.1, then 71 → 37 → 25 → 20 → 16 →
14.9; episode 0 by step at u = 2000: 102 → 83 → 74 → 70 → 56 → 49 (s10).
Replicates seed 0 at the same points (18.7 / 15.7 at e19; 68 / 30 at
s10). Running to 8000.

**Seed 1 of A, FINAL (22704383).** Held-out lattice: by episode 37.9 →
21.4 → 18.1 → 18.2 → 16.4 → 16.3 → 15.8 → 15.9 → 15.0 → 15.2 → **14.2–15.0
(e10–19, mean 14.7)**; episode 0 by step **115 → 81 → 55 → 42 → 30 (s5) →
23 (s10)**; readout 1 139.0. Trained orientations: θ = 45 33 → 8.9, θ = 90
29 → 8.8. Seed 0 at the same cells: e10–19 mean 14.2, s10 18, readout 1
150. **Two seeds agree**: the recurrent net with the decode given reads an
unseen grid lattice to ~15° over a lifetime and to ~20° within ten steps
of its first episode, from 120–150° with no history.

**B2 complete.** Final table (`analysis/b2_results.py --prefix b2`),
held-out envs, held-out lattice θ = 0:

| run | R1 (no history) | R2 e0 / e2 / e9 / e19 | ep0 s1 / s2 / s5 / s10 |
|---|---|---|---|
| `dist` MLP null | 89 | 90 / 90 / 90 / 90 | 92 / 92 / 86 / 87 |
| `full` GRU, raw code | 90 | 89 / 89 / 89 / 89 | 90 / 91 / 86 / 86 |
| `full` GRU, raw code, anchor-mix | 91 | 90 / 91 / 90 / 91 | 86 / 94 / 88 / 90 |
| scripted estimator | — | 16 / 1.5 / 1.7 / 1.9 | 88 / **5.4** / 0.0 / 1.3 |
| `rec` GRU, frozen decode | 103 | 101 / 96 / 78 / 45 | 105 / 101 / 102 / 100 |
| **`full` GRU, frozen decode, s0** | **150** | **32 / 27 / 18 / 14.5** | **56 / 42 / 22 / 18** |
| **`full` GRU, frozen decode, s1** | **139** | **38 / 18 / 15 / 15.0** | **81 / 55 / 30 / 23** |

Compute: ≈ 30 GPU-h across 22 jobs including the cancelled diagnostics.

## 2026-09-14 — the from-scratch question

Jack asked whether a GRU with an MLP in front of it, trained *jointly*
from scratch, would work. Every joint attempt on 09-13 collapsed the
encoder, and the warm-up run (100% determined data) collapsed with loss
spikes where the MLP alone learned the same data to 1° — the GRU's BPTT
gradient into the shared encoder at lr 1e-3. The untried fix:
`--encoder_lr` (7ef0ce6), a separate Adam group for the encoder.

Two runs, MLP 5×768 (LayerNorm, skip) → GRU 2×512 + `prev_action`, all
weights from scratch, encoder lr 1e-4 against 1e-3 for the rest, warm-up
**3000** updates on the fixed orientation (θ = 90°, per-row translation)
then the 0.5 anchor mix, 8000 updates, step at 5600:

- **S1** joint (22749410).
- **S2** joint with the encoder detached from the GRU's gradient
  (22749423) — the encoder then trains through the skip alone.

To read: `heldout@90` at u = 1000–3000 (the decode forming; the MLP
alone: 1° by u = 600); then `heldout` (θ = 0) by episode and by step
after the mix begins — the frozen-decode run A reached 14° by e19 and
56° at s1.

**From-scratch FINALS (8000 updates). Yes — it works.** Held-out envs,
held-out lattice θ = 0:

| run | R1 (no history) | R2 e0 / e2 / e9 / e19 | e10–19 mean | ep0 s1 / s2 / s5 / s10 |
|---|---|---|---|---|
| **S1** joint, encoder lr 1e-4 (22749410) | 93.7 | 22.9 / 18.9 / 16.5 / 15.2 | **15.7** | **31 / 26 / 15 / 12.5** |
| **S2** joint + detached encoder (22749423) | 95.7 | 41.3 / 21.7 / 11.4 / 9.8 | **10.3** | 81 / 67 / 41 / 26 |
| run A, frozen `dist@90` decode, s0 / s1 | 150 / 139 | 32 / 27 / 15 / 14.5 | 14.2 / 14.7 | 56 / 42 / 22 / 18 |

Both learned the decode during the 3000-update warm-up (`heldout@90`
2.0° and 0.5° at u = 3000, readout 2 there 6°, |θ − 90°| elsewhere: the
memoryless signature) and then, once the lattices varied, the frame
in-context: S1 was at 15.3° by e19 within 1000 updates of the switch.
Trained orientations at the end: 8.6 / 7.2 (S1), 9.0 / 7.4 (S2). No
collapse in either (readout 1 differs across sets throughout; loss
−1.7 to −1.9). The two variants trade off: with the GRU's gradient
reaching the encoder (S1) the network measures the frame in one step
(31° at s1, twice as fast as the frozen decode) and plateaus at ~15°;
with it cut off (S2) the per-step measurement is slower but the
lifetime estimate reaches 10°, the best of any run. Readout 1 sits at
~94° rather than the frozen run's 150° because half the training
lifetimes were the anchor orientation: with no history the network
assumes the anchor, which at θ = 0 is 90° off.

**What made the difference from the 09-13 joint runs:** a 10× lower
learning rate for the encoder (`--encoder_lr 1e-4`), and the warm-up
on a fixed orientation so the decode exists before undetermined data
arrives. Everything else was already in place. So the earlier "no
network learned both halves from scratch" was an optimiser statement:
with the encoder on its own learning rate, the joint MLP → GRU learns
the decode and the in-context frame together, and matches or beats the
frozen-decode result.

*Caveat kept honest:* both from-scratch runs keep 50% anchor lifetimes
after the warm-up; the anchor is a *training* orientation and the test
lattice stays unseen, but a run with mix 0 after the warm-up (pure
random lattices) was not repeated with the encoder lr fix — the
09-13 attempt at it failed for the lr reason, not the mix.

## 2026-09-14 — what B2 does and does not show (discussion with Jack)

Jack's question: with translation drawn over the full period, isn't B2
training on the whole scaffold? **Yes.** A shift by `T` is a move of the
env to scaffold position `O + T`; `T ~ U[0, 1716)²` covers every phase
combination the code can produce; over 131k lifetimes a B2 network sees
the equivalent of every scaffold position. B2 holds out *orientation*,
not *region* — by design (scattered placement, "the lattice is what
makes the code unseen"), but the tie-back sentence in §4B.7 read as if
the two holdouts were the same thing. They are not, and a rotation is a
much smaller in-context target (one parameter, two steps for the
estimator) than "a new portion of the code."

Consequences, now written into §0 and §4B.7:

- The corner question (A1x 44°, B1x flat) stands, and B2 sharpens the
  reason rather than overturning it: a contiguous corner's data fit the
  absolute decode (code → position, subtract) and the relative decode
  (phase difference → displacement) equally; only the relative one
  extrapolates; the data cannot separate them because any translation
  that would do so exposes new phase combinations, i.e. new scaffold.
  Which one a plain net finds is inductive bias; A1x says absolute.
  Nothing in a lifetime can fix it in context, since under one lattice
  the frame outside the corner is the frame inside — there is nothing
  to measure. The attractor has the relative-phase invariance built in.
- What B2 establishes: (i) a plain net can represent and learn the
  relative-phase decode when the absolute shortcut is removed — the
  capacity is there, the bias is not; (ii) orientation, which no
  invariance fixes, a recurrent net measures from its trajectory.
- Withdrawn: the corner-plus-translation run I proposed as a closing
  check. It is not a holdout; the translations show the network the
  outside.
- The remaining meaningful corner experiment is architectural (an
  explicit phase-difference front end on a corner-trained net, tested
  outside) and would be handing the network the attractor's ingredient
  rather than a control. Not run.
- Also not run, and still open: the mechanism probes on the from-scratch
  model (θ decodable from the GRU state; Δ′ from the encoder; a
  mid-lifetime lattice swap; a wrong `prev_action`), and a from-scratch
  run with mix 0 after the warm-up.

## 2026-09-14 — range probes: what each model actually computes

Synthetic code pairs via `gbook_at` (no scaffold), 3000 per distance,
angular error vs Chebyshev |Δ|. Scripts `range_probe.py`,
`range_probe_corner.py`, `band_probe.py` in the job tmp dir.

**`dist@90` (B2, translation-trained) and A1 (scattered, fixed
lattice), positions uniform over the whole 1716-cycle:**

| \|Δ\| | 1–17 | 18 | 19 | 20 | 22 | 24 | 26 | 28 | 30 | 35 | 40 | 50 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `dist@90` | 0.5–1.0 | 1.3 | 2.2 | 9.2 | 18 | 22 | 20 | 14 | 16 | 87 | 93 | 140 | 137 | 58 | 65 |
| A1 l5h768 | 0.2–0.4 | 0.6 | 1.4 | 5.2 | 16 | 22 | 18 | 15 | 11 | 101 | 89 | 140 | 144 | 52 | 58 |

Both are **position-free and range-limited**: sub-degree at random
positions anywhere in the cycle through the trained ±19, a step at 20,
10–20° to 30, collapse past 35, anti-alignment at 50–60 (the residues of
50 are (−5, 2, −2): a residue read gives the wrong sign). An absolute
decode-and-subtract would be range-free; neither is. **So A1 — the
scattered fixed-lattice MLP — also learned the rule on differences**,
with no translation trick: scattered coverage was enough. The earlier
statement that only B2's model was known to have learned the rule is
withdrawn.

**A1x (corner K=400, s0), |Δ| ≤ 19, positions split by whether each
coordinate VALUE was ever inside any training env** (seen X: 226 of 400
values, seen Y: 221; seen cells 25,600 of 160,000):

| pair's coordinates | mean | median |
|---|---|---|
| X seen, Y seen (cell itself never seen, X and Y from different envs) | **0.2** | 0.1 |
| X seen, Y unseen | 54 | 36 |
| X unseen, Y seen | 44 | 21 |
| X unseen, Y unseen | **92** | 90 |
| seen X / seen Y at \|Δ\| = 20–40, 40–80, …, 250–400 | 86–89 | — |

The corner model is a **per-axis lookup over the coordinate pairs it
saw within the trained range** — position-limited AND range-limited,
the most literal fit to its training pairs, factorised by axis. It is
not an absolute table (that would be range-free on seen coordinates; it
is 90° at |Δ| ≥ 20 there), and not the rule (that would be
position-free).

**Correction to A1x's "inside 0.2–0.3°, outside 44°".** The generator
places envs on a lattice (pitch 47 here) and draws held-out envs from
the same lattice's free slots with ±3 jitter, so A1x's held-out-inside
envs shared the training envs' coordinate bands; their 0.21° tested
unseen *cells* on seen *coordinate values*. The real boundary is seen
vs unseen coordinate values, wherever they lie: 0.2° vs 44–92°. The
outside number (44°, all coordinates unseen) stands; the inside number
was a placement artifact. In the scattered setting the slack is ±101,
there is no band structure, and A1's random-position probe is a fair
test. *General caveat for any "held-out place inside the region" claim
from this generator under a dense packing.*

**The hierarchy, corrected.** Same architecture, same objective; the
training geometry chose the solution: dense corner → per-axis pair
lookup (neither position- nor range-general); scattered envs → the rule
on differences (position-general, range-limited); translation-randomised
B2 → the rule, provably (the table route was closed). Nothing has made
a network learn the full-range decode (a fixed linear map on module
phase angles mod 2π; unique to |Δ| < 858). Jack's small-MLP hypothesis
for the corner stands as the open test: the pair lookup is the cheapest
fit only while the network has room for it.

## 2026-09-15 — B3: the corner with lifetimes (plan §6.1)

**Build (33f106b).** `LatticeSampler(region=rect)` draws each lifetime's
translation so the env's *rotated* footprint stays inside the rect
(`shift_into_rect`), so training shows only the corner's phase
combinations at every orientation; `--eval_far_rect` adds `far@θ` eval
sets whose rotated footprints sit in `[700, 1200)²`, 300 cells clear of
the corner on both axes — needed because a physical outside env rotated
by 90° can land inside the corner on the torus. 19 lattice tests pass;
toy smoke builds the nine eval sets.

**Submitted.** Corner `rect:0,0,400,400` (A1x's 64 envs), 16 `heldout_out`,
eval sets `heldout_in`, `heldout_out` (θ = 0, physical), `heldout_in@45/90`,
`far@0/45/90`:
- **B3-1** (22783724): `dist` 5×768, every lifetime at θ = 90° with
  corner-confined translation, 1000 updates. The decode question: `far@90`
  ≪ 90° means the rule was learned from a corner; ~44–90° the lookup.
- **B3-2** (22783731): from scratch MLP 5×768 (LN, skip, lr 1e-4) → GRU
  2×512 + prev_action, warm-up 3000 at 90°, then mix 0.5, 8000 updates.
  `far@45`/`far@90` isolate the decode's position generalisation;
  `far@0` and `heldout_out` are the composite.
