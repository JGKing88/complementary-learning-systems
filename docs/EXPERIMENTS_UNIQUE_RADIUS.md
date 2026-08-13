# Unique coding radius — encoder experiments

Running log. The metric is `encoder_training.unique_radius`; the evaluator is
`encoder_training.eval_unique_radius`; sweeps are launched from
`encoder_training.sweep` and read with
`python -m encoder_training.analyze_unique_radius <sweep_dir>`.

**`r_min`** = the per-direction unique radius at the worst of 20 reference
positions, each ≥100 cells from any edge, measured over the whole 1716×1716
arena. Higher is better. Companion columns: `alias_ceiling` (highest cosine
anywhere in the far field — the ceiling the local decay must clear) and
`r_at_cos0.5` / `decay50` (how far before similarity halves — the decay width).
The radius is where the decay curve crosses the ceiling, so those two columns
are the two ways to move it.

---

## 1. Best encoder so far

**`run_20260422_185816/encoder_best.pt`** — `r_min = 16`, but see below: the
sweeps have since beaten it.

**Current best: `ur_loss2_repel_low/029_repel_weight=2_per_env_radius_frac=0.1_seed=44`
— `r_min = 21`, `r_median = 28.5`, `alias_ceiling = 0.814`, `decay50 = 37.5`.**
Tied on `r_min` by `011_repel_weight=0.5_..._seed=44` (21, alias 0.865), but the
repel=2 run has the lower alias ceiling and comes from a far more stable cell.

```
encoder_type mlp      lambdas 11,12,13     out_dim 1024
hidden_dim 512        num_hidden_layers 4
nenv 60               npos 100             (60 patches of 100×100 = 20.4% coverage)
per_env_radius_frac 0.1                    ("near" = within 10 cells)
loss mse_contrastive  attract_lambda 2.0   repel_weight 2.0   uniformity_lambda 0.0
single_env_batch FALSE                     <-- the single most important setting
epochs 1000  lr 1e-4  batch_size 8192  fwhm_ratio 0.25  gain 1.0→5.0
```

For scale: across the 407 archived encoders scored in the audit, the median
`r_min` was 1 and 88% scored ≤3. Only one archived encoder exceeded 7.

---

## 2. Central findings

### 2.1 `single_env_batch=False` is worth ~16 radius units on its own

The loss builds `near = (dist < radius) & same_env` and pushes *everything
else* toward cosine 0. With `single_env_batch=True` a batch holds one
environment, so **pairs from different patches never co-occur and are never
repelled** — nothing stops distant places from having near-identical codes.

Controlled at 3 seeds (`ur_seb_control`, all else fixed):

| `single_env_batch` | r_min | alias_ceiling | decay50 |
|---|---|---|---|
| True  | 2, 2, 2 | 0.988 | 18.0 |
| False | 17, 18, 18 | 0.844–0.903 | 38.5–40.5 |

Arms do not overlap; gap of 16. The True arm has *zero* seed variance — without
cross-environment repulsion the outcome is essentially forced.

This also explains the archive: of 407 encoders, exactly one was trained with
`False`, and it was the only one to score above 7.

### 2.2 `repel_weight` has an optimum near 1.0 — too little is worse than too much

Two sweeps, opposite directions, and the second corrects the first.

**Above 1.0, less is better** (`ur_loss_20260811`, repel 1→40):

| repel | median r_min | alias_ceiling | decay50 |
|---|---|---|---|
| 1  | 15 | 0.903 | 39.0 |
| 5  | 13 | 0.897 | 32.0 |
| 15 | 14 | 0.906 | 28.0 |
| 40 | 10 | 0.884 | 26.0 |

Raising it lowers the ceiling only marginally (0.903→0.884 across 40×) while
shrinking the decay width fast (39→26). Over-repelling flattens the
neighbourhood the radius is measuring.

**Below 1.0 it does not keep improving — it gets erratic** (`ur_loss2_repel_low`,
all 36 runs). Median `r_min` by cell:

| repel ↓ / frac → | 0.10 | 0.15 | 0.20 |
|---|---|---|---|
| 0.25 | 15 | 10 | 8 |
| 0.50 | 18 | 7 | 6 |
| **1.00** | **18** | 13 | 11 |
| **2.00** | 14 | **15** | 10 |

and the seed spread within a cell, which is the real story:

| repel | median within-cell spread (max−min) |
|---|---|
| 0.25 | **9.0** |
| 0.50 | 4.0 |
| 1.00 | **1.0** (at frac 0.10) |
| 2.00 | 6.0 |

At 0.25 the same cell yields 5 and 18; collapsed runs show alias ceilings of
0.993–0.998 with the *widest* decay widths measured. So weak repulsion
sometimes fails to break the basin where everything looks alike, rather than
being uniformly bad — repel 0.5 also produced a joint-best 21.

Reading: repulsion breaks the collapsed basin; too little and some seeds never
escape it, too much and it flattens the neighbourhood. **repel ∈ [1, 2] is the
reliable region**, with repel=1.0/frac=0.10 the most reproducible cell in any
sweep so far (17/18/18, spread 1). I earlier called this a monotone
"repel should go down" trend after seeing only the 1→40 side; it is a peak.

### 2.2c `per_env_radius_frac` ≈ 0.10 is the strongest single setting

Marginal median `r_min` over all 36 runs: **0.10 → 17.5**, 0.15 → 10.5,
0.20 → 8.5. It beats 0.15/0.20 in every repel row. The first sweep put 0.05 at
15 and 0.10 at 18, so 0.10 is a peak rather than an edge.

### 2.3 `repel_weight` and `per_env_radius_frac` trade off along a diagonal

Median `r_min` over 3 seeds:

| repel ↓ / radius_frac → | 0.05 | 0.10 | 0.20 |
|---|---|---|---|
| 1  | 15 | **18** | 11 |
| 5  | 9  | 15 | 14 |
| 15 | 7  | 14 | 15 |
| 40 | 6  | 10 | **15** |

Low repel wants a small radius; high repel wants a large one. Consistent with
them being the two halves of the same crossing point. `per_env_radius_frac` is
a fraction of the *patch side*, so 0.1 of a 100-cell patch = "near means within
10 cells" — and the resulting `r_min` (6–20) lands in the same range, i.e. the
metric largely reports how far the trained notion of "near" generalizes.

### 2.4 Architecture matters far less than the loss/batching

Across the 407-encoder audit: `corr(out_dim, r_min) = +0.286`,
`corr(hidden_dim, r_min) = +0.227`, `corr(layers, r_min) = +0.202` — all weaker
than the single boolean above. `out_dim` sets a ceiling (best achievable 1/3/7/16
at out_dim 64/128/256/1024) rather than a mean. `gain` does nothing
(`corr = −0.05`). 4 hidden layers beat 3.

### 2.5 Patch geometry: more, smaller patches; coverage helps

`corr(coverage_pct, r_min) = +0.445`, `corr(n_patches, r_min) = +0.432`, but
`corr(size_max, r_min) = −0.123` and `corr(size_max, alias) = +0.343`. At equal
coverage, 60×(100×100) beat 15×(200×200).

### 2.6 Training is deterministic; seed variance is real but small

Identical config + seed reproduces the radius trajectory eval-for-eval to three
decimals. So seed replicates measure genuine seed sensitivity, not numerical
noise. Median within-config spread across seeds is 2.0 `r_min` units against a
3.68 spread of config medians — config effects are ~2× seed noise, so **three
seeds is a floor, not a nicety.**

### 2.7 Caveat: `r_min` compresses

88% of the archive sat at `r_min ≤ 3`, so `r_min` identifies a winner but does
not rank a field of similar encoders. Use `r_median`, `alias_ceiling` and
`decay50` for ranking, and remember `r_min` as recorded is a max over ~10
evaluation points during training, which is optimistically biased.

---

## 3. Sweep log

| sweep | grid | runs | result |
|---|---|---|---|
| `unique_radius_20260811_195333` | audit of 407 archived encoders | — | median r_min 1; best 16 (`run_20260422_185816`) |
| `ur_loss_20260811` | repel [1,5,15,40] × frac [0.05,0.1,0.2] × 3 seeds | 36 | best 20; repel 1→40 declines (one side of a peak — see §2.2) |
| `ur_seb_control` | single_env_batch [T,F] × 3 seeds | 6 | True 2/2/2, False 17/18/18 — §2.1 confirmed |
| `ur_loss2_repel_low` | repel [0.25,0.5,1,2] × frac [0.10,0.15,0.20] × 3 seeds | 36 | **best 21**; repel is a peak not a slope; frac 0.10 dominates |
| `ur_seb_C_pairs_vs_dynamics` | False batching; exclude_cross_env_pairs [F,T] × 3 seeds | 6 | **pairs, not dynamics** — 18/18/17 vs 3/3/2 |
| `ur_seb_A_geometry` | True; npos_list [4×400, 2×600, 1×800] × repel × 2 seeds | 12 | **cancelled** — step count not matched across geometries |
| `ur_seb_A2_npos200` | True; 15×200 step-matched, repel [1,5] × 2 seeds | 4 | **no rescue** — r_min 3,3,3,4 |
| `ur_seb_A3_npos400` | True; 4×400 step-matched, repel [1,5] × 2 seeds | 4 | partial — best 9, high seed spread |
| `ur_seb_B_uniformity` | True; uniformity_lambda [0,0.1,0.5,2,8] × 2 seeds | 10 | **no rescue** — ceiling falls to 0.806, decay dies to 1 |

### First: which half of `single_env_batch` actually matters (`ur_seb_C`)

The flag changes two things at once, and §2.1 credits only the first:

1. **Loss composition** — a one-env batch holds no cross-environment pairs, and
   since `far` is just "not near", those pairs *are* the between-patch
   repulsion. There is no dedicated term.
2. **Optimisation dynamics** — each gradient step comes from one environment
   (8192 of that env's 10,000 points, ~82%), cycling through envs. Closer to
   alternating per-env full-batch descent than SGD over pooled data.

`ur_seb_control` flipped both together. `exclude_cross_env_pairs` withholds only
the pairs while keeping batches mixed (73 steps/epoch, each step still drawn
from many envs), isolating (1).

**RESULT — mechanism (1). Dynamics contribute essentially nothing.**

| `exclude_cross_env_pairs` | r_min | alias_ceiling | decay50 |
|---|---|---|---|
| False (normal mixed) | 18, 18, 17 | 0.844–0.903 | 38.5–40.5 |
| True (pairs withheld) | **3, 3, 2** | 0.978–0.989 | 17–18 |
| *(`single_env_batch=True`, for reference)* | *2, 2, 2* | *0.988* | *18* |

Withholding the pairs alone reproduces the `single_env_batch=True` result on
every column — radius, ceiling and decay all land on top of it. So the whole
16-unit effect is loss composition; how many environments a gradient step sees
does not matter. §2.1 is confirmed, and the two rescue attempts below were
aimed at the right mechanism.

### Rescuing `single_env_batch=True` — two hypotheses in flight

**A, geometry.** Under True the only repulsion left is between far pairs
*inside* one patch, so the distance over which codes get separated is bounded
by the patch side — 100 cells against a 1716-cell arena. Growing the patch
should extend that reach; at one arena-sized env, True and False coincide by
construction. Coverage held near 20% and the near-radius pinned at a fixed 10
cells, so neither confounds patch size. Baselines to beat: 60×100 → 2,
15×200 → 2.

**B, uniformity — RESULT: lowers the ceiling as predicted, and still fails.**

`uniformity_loss` is `logsumexp(-t‖zi−zj‖²)` over the batch — a repulsion that
never asks which environment a pair came from, so unlike the far-pair term it
does not need mixed batches to bite.

| `uniformity_lambda` | r_min | alias_ceiling | decay50 |
|---|---|---|---|
| 0.0 | 2 | 0.988 | 18 |
| 0.1 | 2 | 0.979 | 16 |
| 0.5 | 1.5 | 0.932 | 11 |
| 2.0 | **0** | 0.912 | 2 |
| 8.0 | **0** | **0.806** | 1 |

The ceiling falls monotonically 0.988 → 0.806, confirming the high-dimensional
argument: 3M codes spread near-uniformly in 1024 dimensions have a maximum
pairwise cosine of ~0.164 (measured), so spreading means near-*orthogonality*,
not overlap. But the decay width collapses faster still, 18 → 1, and the radius
dies with it.

**Why**: `logsumexp(-t·d²)` is dominated by its smallest `d`, so the gradient
concentrates on the *closest* pairs in the batch — exactly the pairs `attract`
is trying to hold at cosine 1. Uniformity is an indiscriminate repulsion and it
fights hardest precisely where local structure is needed.

That is the same failure mode as `repel_weight=40` (§2.2): both flatten the
neighbourhood. It also isolates what makes cross-environment repulsion special
— it is **selective**, acting only on pairs that are already distant, so it
lowers the ceiling without touching the decay. No other knob tried so far has
that property.

Two of my claims about this were wrong and in opposite directions: the original
dismissal ("spreading makes patches overlap more") is refuted by the ceiling
falling, and the correction ("this could be transformative") is refuted by the
decay collapsing. The mechanism was a third thing.

**A, patch size — RESULT: a real but insufficient effect, and unstable.**
All step-matched at 60,000 steps and ~20% coverage:

| geometry | r_min (per seed) | alias_ceiling | decay50 |
|---|---|---|---|
| 60×100 (baseline) | 2, 2, 2 | 0.988 | 18 |
| 15×200 | 3, 3, 3, 4 | 0.963–0.989 | 16–21.5 |
| 4×400 | 3, 3, 6, **9** | 0.957–0.981 | 19–28.5 |

The direction is right — the reach of within-patch repulsion does grow with the
patch, and 4×400 more than quadruples the baseline. But it is still half the
mixed-batch 18, and the seed spread at 4×400 is large (3 → 9 at the same
repel), which is what §2.5's warning about lumpy patch placement predicted:
with only 4 patches, *where* they land varies a lot between seeds.

Extrapolating, closing the gap would need patches approaching arena size — at
which point there is one environment, `single_env_batch` is vacuous, and
nothing has been rescued so much as sidestepped.

### Conclusion for the True regime

`single_env_batch=True` costs ~16 radius units and neither substitute recovers
it. The reason is now sharp: cross-environment repulsion is the only
**selective** repulsion available — it acts solely on pairs that are already
distant, lowering the alias ceiling without touching the decay width. Every
alternative tried is indiscriminate and pays for a lower ceiling with a
collapsed neighbourhood (uniformity, `repel_weight=40`) or simply does not
reach far enough (patch size).

Best achieved under True: **r_min 9** (4×400, repel 1.0, seed 43), against 21
in the mixed-batch regime.

---

# 4. How good can `exclude_cross_env_pairs=True` get?

§3 asked whether the True regime could be *rescued* and answered no. This asks
the different question of how far it can be *pushed*, under a fixed brief:

* `exclude_cross_env_pairs=True` throughout — the constraint, not a variable;
* no patch larger than **200** cells a side;
* patch sizes **mixed**, not uniform;
* `loss_mode=cka` excluded;
* every other knob free.

Driver: `encoder_training.sweep_ecp` (named size mixes, step-matched epochs,
`ou_bcs_normal` only). Read a wave back with
`python -m encoder_training.collect_ur <sweep_dir>`, which reads the summary
each run already stored in its checkpoint rather than re-scoring on a GPU.

### 4.0 What "under the constraint" is taken to mean

The flag governs the *repel* mask. Taken literally it can be satisfied while
putting the same supervision back through another term, so the campaign is
split and the two halves are never mixed in a headline:

* **Legal** — anything that never asks which environment a pair came from:
  geometry, the near radius, loss weights, distance-graded targets, batch-wide
  spread terms (`uniformity_loss` over *all* pairs, VICReg variance/covariance),
  architecture, optimisation.
* **Loophole, labelled as such** — `uniformity_scope=nonnear` and
  `input_far_tau`. "Not near" is defined with `same_env`, so it contains every
  cross-environment pair; and the smoothed code decorrelates within ~5 cells, so
  "input-dissimilar" is very nearly "far apart anywhere". Both restore the
  withheld signal. They are run to bound the gap, never to claim it closed.

### 4.1 Diagnosis first: where the aliases actually are

`encoder_training.alias_structure` reports *where* the far-field peaks sit, and
how many dimensions the code still uses, rather than only how high the ceiling
is. Three results reframe the problem.

**The raw grid code is not a free ceiling.** The smoothed code's own alias
ceiling is **0.953**, at offsets that are multiples of 143 (= 11·13) and 156
(= 12·13) — two modules exactly aligned, the third off by one phase, which the
smoothing then makes nearly identical. Its similarity also decays to 0.02 by
r=5. So the input is *both* aliased and too sharp: an encoder has to widen the
neighbourhood *and* suppress aliases the input already has. Nothing is
inherited.

**An untrained encoder is fully collapsed** — cos = 1.000 arena-wide at gain 5.
There is no "don't destroy the input" strategy to fall back on.

**The True regime's failure is not grid periodicity.** Its worst peaks sit at
offsets ≈ 786–930 with residues (5, 6, 6) mod (11, 12, 13) — *maximally*
misaligned in every module. Those are the two most dissimilar inputs the arena
contains, and the encoder maps them to cosine 0.98. The failure is that the
encoder is collapsing near-orthogonal inputs, not that the code repeats.

| encoder | r_min | alias peaks (top-12 over 5 refs) | decay50 |
|---|---|---|---|
| raw smoothed code | — | max 0.953, at ±143 / ±780 | ~2 |
| untrained, gain 5 | — | 1.000 everywhere | ∞ |
| `exclude_cross_env_pairs=True`, 60×100 | 3 | max 0.983, **median 0.955** | 17 |
| `single_env_batch=True`, 15×200 | 3 | max 0.979, **median 0.783** | 21.5 |
| `exclude_cross_env_pairs=False`, 60×100 | 18 | max 0.610, median 0.445 | 38.5 |

The 200-cell patches move the *median* peak a long way (0.955 → 0.783) and the
maximum barely at all, which is exactly the shape of `r_median` rising 3 → 7
while `r_min` stays at 3. `r_min` is a worst case over 20 references, so one bad
reference holds it down however much the typical one improves.

### 4.2 The two levers, restated

`r_min` is where the decay curve crosses the alias ceiling, so there are two
ways to move it and the True regime has only ever tried one.

* **Lower the ceiling.** Needs a term that reaches ~800 cells. Within-patch
  repulsion reaches 283 at the largest patch allowed here, so the only legal
  candidates are the batch-wide spread terms.
* **Widen the decay.** Untried. The binary target asks for a *plateau* at cosine
  1 inside the radius, and the radius test is a strictly-*decreasing* one — a
  perfectly satisfied binary target scores zero, and what the metric actually
  reads is the residual slope the network failed to flatten. Naming the slope
  outright is what `graded_sigma` does: the pair target becomes
  `exp(-d²/2σ²)` instead of 1-or-0.

### 4.3 Sweep log

| wave | grid | runs | result |
|---|---|---|---|
| `w1_geometry` | 5 size mixes × near-radius {fixed 10, 0.1·side} × 2 seeds | 20 | *running* |
| `w2_spread` | graded σ {10,25,50}, uniformity {0.1, 1, 0.1@t0.25}, VICReg {×1, ×5}, none × 2 seeds | 18 | *running* |

Size mixes, all ≤200 a side and all held near 20% coverage so the axis is
granularity rather than area:

| mix | envs | sizes | coverage |
|---|---|---|---|
| `u100` | 60 | 100 | 20.4% |
| `u200` | 15 | 200 | 20.4% |
| `mix2` | 33 | 200, 100 | 20.4% |
| `mix5` | 93 | 200, 140, 100, 70, 50 | 20.2% |
| `mixbig` | 41 | 200, 140, 100, 70, 50 | 20.6% |

The near-radius axis is the one with two opposed predictions. At a **fixed** 10
cells every patch teaches the same notion of "near" and size varies only how far
the within-patch repulsion reaches. At a **fraction of the side** the patches
*disagree* about what "near" means — 5 cells in a 50-patch against 20 in a
200-patch — and no translation-invariant code can satisfy both, so the encoder
is pushed toward depending on absolute position. That is the missing ingredient,
if it works; it is also a way to make the task incoherent, if it does not.

### 4.4 Infrastructure notes

* `data.build_patch_codes` builds each patch's codes directly instead of slicing
  the 10.2 GB full codebook, dropping a run's host memory from ~20 GB to ~1 GB
  (`--lazy_codes`, verified against the old path in
  `tests/test_lazy_patch_codes.py`). The two group the Gaussian factors
  differently, so codes agree to float32 rounding rather than bit-for-bit — fine
  within a wave, but a seed-for-seed replay of a §2/§3 run needs it off.
* With that, `sweep_ecp` packs four runs onto one GPU. A run at batch 8192 keeps
  an A100 about 6% busy (the step is bound by kernel launches and mask
  construction, not arithmetic) and `ou_bcs_normal` is GPU-limited, so runs per
  GPU is the throughput lever, not jobs.
