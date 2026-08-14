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

### 2.7 Caveat: `r_min` compresses — and see §4.8, it is also unstable

> **§4.8 supersedes part of this.** Every number in §1–§3 comes from the same
> 20 reference positions. Re-scored at 100, §1's best encoder falls from
> `r_min` 21 to **9** on one draw and 15 on another. The rankings here are
> broadly intact but the absolute values are an upper bound on a noisy
> statistic. `r_median` is the stable companion.


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

**Best config: `mixtop_max` (50.8% coverage, 26×200 + 14×150 + 14×100) + near
radius 0.15·side + `rate_lambda=0.3` — `r_min` 26, 27, 30, 31 over four seeds
(median 28.5), `r_median` 42.8, alias ceiling 0.671.**

Against 2–3 for this regime untreated, 9 for the best §3 rescue attempt (which
used 400-cell patches, outside this brief), and **21 for the best encoder ever
trained *with* the cross-environment pairs**.

**On the honest test — 100 references, two reference draws, nothing selected
against them (§4.8b) — it is 24 / 23 against the unconstrained encoder's
9 / 15**, with `r_median` 43–45 against 28.5, decay50 42 against 36, and an
alias ceiling of 0.74–0.79 against 0.825. Roughly double, on every column.

Checkpoints: `w13_coverage_top/00{2,3}_cov51_seed=4{2,3}` and
`w16_coverage_seeds/00{0,1}_cov51_seed=4{4,5}`.

Four knobs, none of which ever asks which environment a pair came from:
coverage (§4.6b, the largest), the near radius (§4.5b/§4.5e), an env-blind
spread term (§4.4), and a big-heavy size mix (§4.5d).

| | best unconstrained (§1) | best under the constraint |
|---|---|---|
| config | `repel=2, frac=0.1`, 60×100 | `mixtop`, radius 0.15·side, `rate_lambda=0.3` |
| cross-env pairs | kept | **withheld** |
| `r_min`, 20 refs | 21 *(1 seed)* | **17.5 median, 14–22** *(4 seeds)* |
| `r_min`, 100 refs | 9 / 15 | 16 / 15 and 13 / 13 |
| `r_median`, 100 refs | 28.5 / 29 | 28 / 29 |
| `r_median`, 20 refs | 28.5 | 26.0–29.5 *(4 seeds)* |
| alias ceiling | 0.814 | 0.81–0.92 |
| decay50 | 37.5 | 37.5–42.0 |

**Seed spread is 8 on the headline cell** (19, 22, 16, 14), so the claim is
"matches", not "beats". The first pair of seeds reached a 0.81 ceiling and the
second pair 0.89–0.92, and `r_min` follows. §2.6 says three seeds is a floor for
this metric; two were carrying this result until `w12`. `r_median` is far
steadier across the same four runs, 26.0 to 29.5.

Not merely the same score — the same *profile*, column by column, and on the
100-reference re-score (§4.8) the constrained encoder is ahead on one draw and
level on the other. Three knobs do it, none of which ever asks which
environment a pair came from:

1. `rate_lambda=0.3` — the MCR² coding rate, which owns the alias ceiling
   (§4.4) and takes it from 0.96 to 0.81;
2. a near radius of **0.15·patch-side**, which owns the decay (§4.5b) and takes
   decay50 from 21 to 42;
3. a **big-heavy size mix** (`mixtop`: 12×200 + 6×150 + 6×100), worth ~3 units
   over uniform 200 at matched settings (§4.5d).

**This overturns §3.** That section concluded that cross-environment repulsion
is the only *selective* repulsion available and that the True regime's ~16-unit
cost is not recoverable. It is recoverable, and the reason §3 missed it is
§4.4b: every substitute it tried moved one factor of a product at the other's
expense, and no wave had varied the two independently.

*Still to confirm: seeds 44/45 (`w12`), whether the radius fraction turns above
0.15 (`w11`), and a re-score at 100 references and a second reference seed —
every run in this campaign shares the same 20.*

> **`graded_sigma` reached 13/11 and is excluded.** A distance-graded pair
> target replaces the contrastive near/far split with a target *kernel*, which
> is the family `loss_mode=cka` was excluded for — the fit is by MSE rather than
> by centered alignment, but it is the same move. §4.4 keeps the result because
> a knob that works and is out of scope needs to be on the record, not
> rediscovered; nothing downstream of it counts toward the headline, and `w6`
> and `w7` were cancelled before they ran.
>
> What survives is the *decomposition*, §4.4b: the radius is the decay times
> the ceiling, and the decay has a legal knob — the near radius, §4.5.

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

### 4.1b Rank tracks the radius — but does not cause it

> **Read with §4.4.** The ordering below is real and reproduces across every
> encoder measured, and it is what motivated the rank terms. It is also
> *observational*: §4.4 forces rank up by two different routes and the radius
> does not follow. Kept as written because the terms it motivated turned out to
> matter for a different reason — they are the only thing that moves the alias
> ceiling, which is the other factor in §4.4b's law.

The same tool reports the code's effective dimensionality — the participation
ratio `(tr C)² / ‖C‖_F²` of its covariance over 20,000 random arena positions.
It orders the encoders exactly as the radius does:

| encoder | eff. dims (of 1024) | alias median | r_min |
|---|---|---|---|
| `exclude_cross_env_pairs=True`, seed 43 | **24** | 0.981 | 2 |
| `exclude_cross_env_pairs=True`, seed 42 | **59** | 0.952 | 3 |
| `single_env_batch=True`, 15×200 | **117** | 0.783 | 3 |
| `exclude_cross_env_pairs=False`, seed 42 | **202** | 0.407 | 18 |
| `exclude_cross_env_pairs=False`, seed 44 | **202** | 0.538 | 18 |
| *(raw smoothed code, 434 dims)* | *43* | *0.953* | *—* |

`live_frac` is 1.000 in every row: no coordinate has gone dead. The collapse is
in the *spectrum*, not the coordinates — which is why 1024 outputs buys nothing
and why a variance penalty is the wrong instrument.

**Why the pairs supply rank.** Inside a patch the code only has to separate the
places that patch contains, and every patch may reuse the same directions — so
the within-patch objective is satisfied by a code of rank ≈ the number of
distinguishable places in one patch. Cross-environment pairs demand instead that
all 600k training points be mutually near-orthogonal, and *that* costs rank.
Both unconstrained runs landing on 202.3 and 202.4 looks like a fixed point of
the objective rather than a coincidence.

This restates §3's conclusion in a way that suggests what to do. §3 said
cross-environment repulsion is the only *selective* repulsion available. The
measurement says what selectivity was worth: rank. And rank can be asked for
directly, by a term that never mentions an environment.

That reasoning is what produced `coding_rate_loss`, and the term earned its
place — but not for the reason given here. See §4.4.

### 4.2 The two levers, restated

`r_min` is where the decay curve crosses the alias ceiling, so there are two
ways to move it and the True regime has only ever tried one.

* **Widen the decay.** The legal knob is the **near radius**: it is the
  threshold in the original binary contrastive loss, swept in §2.2c but never
  above 0.2 of the patch side. §4.5 measures it working — at 15×200, 10 → 20
  cells takes decay50 from 22 to 35 and `r_min` from 4.5 to 6.5 with the
  ceiling unmoved. `w3_radius` brackets it from 2 to 40 and `w8` crosses it
  with the ceiling term. (`graded_sigma` moves the same factor much harder but
  is out of scope; see the section header.)
* **Lower the ceiling.** Needs a term that reaches ~800 cells. Within-patch
  repulsion reaches 283 at the largest patch allowed here, so the only legal
  candidates are the batch-wide spread terms — and §4.1b says what they should
  ask for. Three are in flight, differing in how they can misfire:
  `uniformity_loss` is the only one that acts on an individual collapsed pair,
  but its `logsumexp` is dominated by the smallest distance and those are the
  pairs `attract` holds at cosine 1 (§3's diagnosis, unchanged); VICReg's
  variance/covariance pair is pair-free and so cannot fight `attract` at all,
  but it asks for per-coordinate variance, which these codes already have;
  `coding_rate_loss` (MCR², `-½·logdet(I + D/(Bε²)·ZᵀZ)`) is pair-free *and*
  spectral, so on the diagnosis it is the one aimed at the measured deficit.
  Why the decay is worth widening at all: the binary target asks for a
  *plateau* at cosine 1 inside the radius, and the radius test is a
  strictly-*decreasing* one, so a perfectly satisfied binary target would score
  zero. What the metric actually reads is the residual slope the network failed
  to flatten — and the near radius sets how far that slope extends.

### 4.3 Sweep log

| wave | grid | runs | result |
|---|---|---|---|
| `w1_geometry` | 5 size mixes × near-radius {fixed 10, 0.1·side} × 2 seeds | 20 | partial — §4.5; mix beats uniform, radius acts on the decay |
| `w2_spread` | none, graded σ {10,25,50}, uniformity {0.1,1}, VICReg, rate {0.3,3} × 2 seeds | 18 | partial — §4.4; **`rate0.3` → r_min 9** (graded rows out of scope) |
| `w3_radius` | near radius {2,3,5,10,20,40} at `mix2` × 2 seeds | 10 | *running* (radius=10 dropped: identical to a `w1` cell) |
| `w4_coverage` | 9.5% … 52.6% coverage × 3 seeds | 15 | queued — justified by §4.6 |
| `w5_input_rank` | fwhm {0.1,0.25,0.5} × hidden_dim {512,1024} × 2 seeds | 12 | queued |
| ~~`w6_graded_wide`~~ | σ {50…150} × coverage | 12 | **cancelled before running** — `graded_sigma` out of scope |
| ~~`w7_decay_x_ceiling`~~ | σ × `rate_lambda` | 10 | **cancelled before running** — same reason |
| `w8_rate_x_radius` | radius frac {0.2, 0.3} and {0.1,0.2,0.3}×`rate0.3`, + one at `mixbig` × 2 seeds | 12 | *running* — `w7`'s question with the legal decay knob |

### 4.5b The near radius is a clean decay knob, and it peaks at 20 (`w3_radius`)

`encoder_final` on `mix2`; the radius-10 row is `w1`'s identical cell.

| near radius | r_min | r_pred | r_median | decay50 | res90 | alias max |
|---|---|---|---|---|---|---|
| 2 | 0, 0 | 1.4 | 1.5 | 7.0 | 3.0 | 0.979 |
| 3 | 1, 1 | 1.6 | 2.0 | 9.25 | 4.0 | 0.983 |
| 5 | 2, 1 | 2.6 | 3.5 | 13.0 | 5.0 | 0.975 |
| 10 | 6, 3 | 5.0 | 7.25 | 21.0 | 8.75 | 0.964 |
| **20** | **7** | 7.1 | 14.0 | 34.5 | 13.5 | 0.971 |
| 40 | 5 | *10.9* | 15.0 | **54.5** | 21.0 | 0.972 |

`decay50` tracks the radius almost exactly linearly — ≈1.3× it, across a
twentyfold range — while the alias ceiling stays within 0.964–0.983. So the
radius is an isolated handle on one factor of §4.4b, which is what makes it the
legal substitute for `graded_sigma`. The optimum is **20**, at `r_min` 7 against
10's 4.5.

**Where the law breaks, and how.** Radius 40 produces the widest decay in the
campaign — 54.5, past the unconstrained regime's 38–40 — and `r_min` *falls* to
5 while `r_pred` says 10.9. The shape check passes (res90/decay50 = 0.385,
Gaussian), so this is not the profile. It is that the spread *across reference
positions* blows up: `r_median` is 15 against `r_min` 5. The law is built from a
median decay and a max ceiling, so it predicts the typical reference and not the
worst one — and those diverge exactly when the radius is pushed past its
optimum. Read `r_pred` as a prediction of `r_median`; the gap between them is
itself the signal that the references have stopped agreeing.

### 4.5d The two factors compose (`w8`, `w9`)

The point of §4.4b was that the radius is a product, so the two knobs should
multiply. They do. `encoder_final`, single seeds where noted:

| config | r_min | decay50 | alias max | which factor it supplies |
|---|---|---|---|---|
| radius 0.1·side alone, `mix2` | 5, 4 | 31 | 0.984 | decay |
| `rate0.3` alone, radius 10 | 7, 9 | 22.75 | 0.919 | ceiling |
| **both**, `w8/f0.1_rate0.3` | **10** | **32** | **0.904** | both |
| *unconstrained (§2.1)* | *18–21* | *38–40* | *0.844–0.864* | |

Neither knob disturbed the other's factor: adding `rate0.3` left decay50 at
31→32, and widening the radius left the ceiling at 0.919→0.904. `alias_mean`
reaches 0.726, against 0.63–0.66 unconstrained.

**And a big-heavy mix finally beats uniform.** `mixtop` — 12×200 + 6×150 +
6×100, three sizes with 71% of the area at 200 cells — reaches `r_min` 8 and 9
at radius 0.1·side with no spread term at all, against `u200`'s 6.5 and `mix2`'s
4.5–5 at the same setting. That is the first evidence in the campaign that
mixing sizes helps, and it is consistent with why `mix5` failed: adding scale
variety is only free while every patch is still large enough for its repulsion
to reach. A 50-cell patch reaches 70 cells and the far field never hears it; a
150-cell patch reaches 212.

### 4.5c Notes on the original motivation for `w3_radius`

`w3_radius` was launched to test a prediction from §4.1b — that a *smaller*
radius buys rank and should therefore win — and the prediction is already dead.
`radius=2` reached 412 effective dimensions, twice the unconstrained regime's
202, and scored `r_min` 1 against the baseline's 4.5.

The wave is now the more important one for the opposite reason. With
`graded_sigma` out of scope the near radius is the **only** legal knob on the
decay, which §4.4b makes one of the two factors in the radius, and §2.2c never
took it above 0.2 of the patch side. The bracket runs to 40 cells, which is 0.2
of a 200-cell patch and 0.4 of a 100-cell one, and `w8` crosses it with the
ceiling term.

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

### 4.4 The decay is the lever, and rank is not (`w2_spread`)

> The `graded*` rows below are **out of scope** — see the note in the section
> header. They are kept because they are what identified the decay as the
> lever, and because a knob that works needs to be recorded as excluded rather
> than left to be rediscovered. The `rate0.3` row and the baseline are the ones
> that count.

A distance-graded target at σ=50 reaches **`r_min` 13** where the identical
config with binary targets reaches 6 — one flag, same geometry, same seed.
(Numbers below are one seed each and the second seeds are still running; the
σ=50 arm's other seed was at `r_min` 0 as of epoch 212, so this cell's seed
spread is not yet known and may be large.)

The arms separate into two families, and each family moves exactly one factor
of §4.4b's law. `encoder_final` throughout; a single value means the second
seed is still running:

| arm | r_min (both seeds) | r_pred | alias max | alias mean | decay50 | res90 |
|---|---|---|---|---|---|---|
| **`rate0.3`** | **9**, — | 8.7 | **0.907** | **0.821** | 22.5 | 9.0 |
| `vicreg` | 6, 7 | 6.4, 7.4 | 0.934, 0.914 | 0.840, 0.847 | 20 | 8 |
| `none` (binary baseline) | 6, 3 | 6.2, 3.7 | 0.946, 0.982 | 0.880, 0.939 | 21 | 8.5, 9 |
| *`graded50` (out of scope)* | *13, 11* | *14.4, 13.8* | *0.956, 0.959* | *0.891, 0.907* | ***60, 59*** | *22* |
| *`graded25` (out of scope)* | *4, 3* | *5.5, 4.6* | *0.978, 0.985* | *0.898, 0.935* | *29.5* | *12* |
| *`graded10` (out of scope)* | *3, 2* | *3.5, 2.9* | *0.949, 0.965* | *0.870, 0.919* | *12* | *5* |

**Every legal spread term moves the ceiling and only the ceiling.** `rate0.3`,
`vicreg` and the baseline all sit at decay50 20–22.75; what separates them is
the alias ceiling, and `r_min` follows it in order. The excluded `graded*` rows
are the mirror image — ceiling pinned near 0.956, decay50 taken to 60. Two
clean, separable factors, one of which now has only the near radius as a legal
handle (§4.5).

`r_pred` is §4.4b's law and is within one cell on all eleven rows.

#### Uniformity works here, and §3's test of it was confounded

§3 swept `uniformity_lambda` over {0, 0.1, 0.5, 2, 8} and concluded the term
fails — at 0.1 it scored `r_min` 2 with an alias ceiling of 0.979. The same λ
here:

Medians over both seeds, all legal arms:

| arm | r_min | r_median | alias max | alias mean | decay50 |
|---|---|---|---|---|---|
| **`unif1`** | **9** (8, 10) | 12 | **0.863** | 0.759 | 20.5 |
| `rate0.3` | 8 (7, 9) | 12 | 0.919 | 0.807 | 22.75 |
| `unif0.1` | 7 (6, 8) | 11.5 | 0.926 | 0.822 | 22.75 |
| `vicreg` | 6.5 | 8.75 | 0.924 | 0.843 | 20 |
| `rate3` | 5.5 | 7.5 | 0.843 | 0.741 | 14.5 |
| `none` (binary baseline) | 4.5 (6, 3) | 7.25 | 0.964 | 0.909 | 21 |

**`uniformity_lambda=1.0` is the best spread term at this radius** — a 0.863
ceiling against `rate0.3`'s 0.919 with the decay held at 20.5, and double the
baseline's `r_min`. Every env-blind spread term beats the baseline. §3's range
covered λ = 0.1 and 2 and found neither worked.

**The ranking flips at the wider radius, though** (`w14`, `mixtop` at
0.15·side, where the final config lives):

| spread term | r_min | r_median | alias max | decay50 |
|---|---|---|---|---|
| `rate0.3` | **17.5** (14–22) | 27.5 | 0.853 | 40.8 |
| `unif1` + `rate0.3` | 13.5 | 23.0 | 0.873 | 36.75 |
| `unif1` | 12.5 | 26.25 | 0.876 | 39.5 |
| `unif3` | 3.5 | 6.25 | 0.888 | 32.75 |

Uniformity holds its ceiling advantage nowhere near as well once the radius is
wide, and combining the two terms is worse than the coding rate alone. So the
final config uses `rate_lambda`; uniformity is a close second at a 10-cell
radius and a clear second at 0.15·side. Both are legal, both work, and §3's
verdict on uniformity is wrong either way.

**What changed is the batching, not the term.** §3 ran its uniformity sweep
under `single_env_batch=True`, where a batch holds a single patch. Uniformity
then has *no cross-patch pairs in the batch to push apart*: it can only act on
within-patch pairs, which the far term already covers, so its `logsumexp`
concentrates on the near pairs and fights `attract` — which is exactly the
collapse §3 recorded. Under `exclude_cross_env_pairs=True` the batches stay
mixed, only the *pairs* are withheld from the repel term, and the same term
finally has the collapsed cross-patch pairs available.

So §3's "uniformity is an indiscriminate repulsion and it fights hardest
precisely where local structure is needed" describes a confounded setup rather
than the term. The confound is the same one §3 itself identified in `ur_seb_C` —
`single_env_batch` changes both the loss composition and what a batch contains
— applied to the rescue attempt instead of to the baseline.

#### The ceiling half of §3's deficit is solved

Medians over both seeds, with the unconstrained regime for scale:

| arm | r_min | alias max | alias mean | decay50 | res90 |
|---|---|---|---|---|---|
| `rate0.3` | **8** | 0.919 | 0.807 | 22.75 | 9 |
| `vicreg` | 6.5 | 0.924 | 0.843 | 20 | 8 |
| `rate3` | 5.5 | **0.843** | **0.741** | **14.5** | 5 |
| `none` (binary baseline) | 4.5 | 0.964 | 0.909 | 21 | 8.75 |
| *unconstrained (§2.1)* | *18–21* | *0.844–0.864* | *0.63–0.66* | *38–40* | *15* |

`rate_lambda=3` reaches an alias ceiling of **0.843** — what encoders reach when
they *keep* their cross-environment pairs. So a term that never asks which
environment a pair came from closes that half of the gap completely, which is a
direct answer to §3's "cross-environment repulsion is the only selective
repulsion available": it is not, for the ceiling.

What `rate3` cannot do is hold the decay — 14.5 against 38–40 — and that is now
the entire remaining deficit. The arithmetic is closable on paper: at a 0.843
ceiling, `r_min` 20 needs decay50 ≈ 40, and `u200` at a 0.1·side radius already
produces 35 (§4.5). Both ingredients exist separately. Whether they compose is
what `w9` and `w10` ask, and the risk is visible in the table — `rate0.3` left
the decay alone (22.75 against a baseline 21) while `rate3` halved it.

**This falsifies rank as the cause of the radius.** §4.1b's ordering was
observational, and forcing rank up does not bring the radius with it. Three
independent demonstrations:

| run | eff. dims | r_min |
|---|---|---|
| `graded10` | **293** | 3 |
| `w3_radius/radius=2` | **412** | 1 |
| `graded50` | **27** | **13** |
| *(unconstrained reference)* | *202* | *18–21* |

`graded10` and a 2-cell near radius both exceed the unconstrained regime's 202
effective dimensions and score near zero, while the best encoder in the section
uses 27. Rank continues to predict the *mean* alias ceiling well and it is a
useful diagnostic; it is simply not what gates the headline.

**A second thing I got wrong, in the other direction.** Read at epoch 100, the
strong spread terms looked like §3's collapse — `vicreg` and `rate3` both at
`r_median` 0, the profile not even locally monotone — and I concluded that
being pair-free does not protect a spread term from fighting `attract`. By
epoch 400 both had recovered (`vicreg` to `r_min` 4–5, `rate3` to 4). The
`r_median` 0 was a training transient: these terms take longer to grow a
neighbourhood, they do not prevent one. **Arms are only comparable at matched,
late epochs**, which is also the lesson behind quoting `graded50` at 14 from a
mid-run selected checkpoint when it finished at 13.

**The win is partly the metric, and should be reported as such.** A wide decay
trades near-field resolution for range: `r_at_cos0.9` is ~23 cells at σ=50
against ~7 at the baseline, so everything within 23 cells is at cosine 0.9 or
better and is correspondingly hard to tell apart. This is legitimate on the
metric's own terms — "a similarity of at least `inner_min` means you are within
r cells" is exactly what a broad monotone decay certifies — and the
unconstrained best is also broad (decay50 37.5). But σ buys some of the number
rather than a strictly better code, so `r_at_cos0.9` belongs in every table.

### 4.4b The radius is a two-parameter formula, to within a cell

§4.2 said `r_min` is where the decay crosses the ceiling. Treating the radial
profile as Gaussian makes that quantitative. With
`decay50 = σ·√(2 ln 2)` (the reported `r_at_cos0.5_median`), the profile
`exp(-d²/2σ²)` reaches the alias ceiling `C` at

```
r_pred = decay50 · sqrt( ln(1/C) / ln 2 )
```

Checked over every checkpoint in the sweeps directory that recorded both
columns — 119 encoders spanning the mixed-batch regime, the single-env regime,
the uniformity and geometry rescue attempts of §3, and the graded runs — this
predicts the measured `r_min` at **corr +0.86, median absolute error 1.0 cell,
82% within 3** (`python -m encoder_training.radius_law`).

That is worth more than any single result here, for two reasons.

**It explains §3 in one line.** Every substitute tried there moved one factor at
the other's expense. Uniformity took the ceiling from 0.988 to 0.806 — a factor
1.9 gain on `√(ln 1/C)` — while collapsing decay50 from 18 to 1, a factor 18
loss. `repel_weight=40` did the same thing more mildly. The graded target is the
first knob found that moves the decay and leaves the ceiling alone.

**It turns the rest of the campaign into design rather than search.**
`radius_law --target R` inverts it. To match the unconstrained best of 21:

| if the alias ceiling is | the decay50 needed is | i.e. σ ≈ |
|---|---|---|
| 0.98 | 123 | 105 |
| 0.955 (what σ=50 gave) | 81.5 | 69 |
| 0.90 | 53.9 | 46 |
| 0.86 (the unconstrained value) | 45.0 | 38 |

So `w6`'s σ=75 should reach ~22 *if* the ceiling holds at 0.955 — and whether it
holds is the real question, since a σ that large leaves almost no within-patch
pair asking for separation. The rank terms matter again here, not for `r_min`
directly but as the only way to buy the left-hand column.

### 4.5 Geometry (`w1_geometry`) — size and radius carry it, mixing does not (yet)

`encoder_final`, both seeds where the cell is complete:

| geometry | near radius | r_min median | spread | decay50 | alias max |
|---|---|---|---|---|---|
| **`u200`** (15×200) | 0.1·side = 20 | **6.5** | 1 | 35 | 0.973 |
| `mixbig` (41 envs, 200→50) | 0.1·side | 5.5 | **3** | 29.25 | 0.960 |
| `mix2` (9×200+24×100) | 0.1·side | 4.5 | 1 | 29.75 | 0.974 |
| `mix2` | fixed 10 | 4.5 | **3** | 21 | 0.964 |
| `u200` | fixed 10 | 4.5 | 1 | 22 | 0.964 |
| `mix5` (93 envs, 200→50) | fixed 10 | 3.0 | 0 | 19.5 | 0.980 |
| `mixbig` | fixed 10 | 3.0 | 0 | 21 | 0.974 |
| `mix5` | 0.1·side | 3.0 | 0 | 18.75 | 0.980 |
| `u100` (60×100) | either | 2.5 | 1 | 17.5 | 0.984 |

The fractional radius beats the fixed one for every geometry that contains a
patch bigger than 100 cells, which is the §4.5b decay effect reaching the large
patches. It does nothing at `u100`, where 0.1·side *is* 10.

**The near radius is a decay knob, and it is the axis that pays.** At 15×200,
going from 10 cells to 20 takes decay50 from 22 to 35 and `r_min` from 4.5 to
6.5, with the alias ceiling essentially unmoved (0.964 → 0.973). That is the
same mechanism `graded_sigma` used, through a knob the brief allows, and it is
the reverse of what §4.1b predicted — the radius is not acting on rank here.
`w3_radius` brackets it 2 → 40 and `w8` crosses it with the ceiling term.

**Patch size pays too**: 200-cell patches beat 100-cell ones, 4.5 against 2.5,
consistent with within-patch repulsion reaching further.

**With no spread term, uniform 200 leads at 6.5**, with `mixbig` next at 5.5 —
and both only at the fractional radius. The gap is inside the seed spread
(`mixbig`'s two seeds are 4 and 7), so this wave does not separate them.

What it *does* separate is the smallest patch a mix contains. Every geometry
whose tail reaches 50–70 cells sits at 3.0 when the radius is fixed, because a
50-cell patch's repulsion reaches 70 cells and the far field never hears about
it — those patches supply attract pairs and no reach, which is the `u100`
failure diluted. Giving them a proportional radius recovers `mixbig` (3.0 →
5.5) but not `mix5` (3.0 → 3.0), which has 48 of them.

**Mixing only pays once a spread term is lowering the ceiling** (§4.5d):
`mixtop`, which stops at 100 cells, gives 16 against `u200`'s 13 at identical
settings *with* `rate0.3`, while here without one it is the uniform set that
leads. So the working rule is narrow: a mix helps if every patch is large enough
for its repulsion to reach **and** something else is handling the ceiling;
otherwise prefer uniform 200.

At `u100`, `per_env_radius_frac=0.1` *is* a 10-cell radius, so those two cells
duplicate the fixed-radius ones — and reproduce them to six decimals, which
doubles as the determinism check on the `--lazy_codes` path against the §3 runs.

### 4.6 Which references hold `r_min` down

`r_min` is the worst of 20 references, and §4.1b showed the mean alias ceiling
falling steadily with rank (0.988 → 0.65) while the max barely moved
(0.989 → 0.86). So the question is *which* reference is bad.
`alias_structure --ref_vs_patch` scores each reference against its distance to
the nearest training patch. On the 15×200 encoder:

    corr(patch_dist, r_monotone) = -0.471
    corr(patch_dist, alias_ceiling) = +0.410

References inside a patch score 6–9; those 100–380 cells from any patch score
3–4. The references holding the headline down are the ones training never
reached — which is the one thing coverage fixes, and is why `w4_coverage` is
worth running. On the 60×100 encoder the correlation is a weaker −0.24, but
every reference there sits at `r_monotone` 3, so the metric has no room to show
a difference.

### 4.5e The radius fraction peaks at 0.15 (`w11`)

On `mixtop` with `rate_lambda=0.3`, medians over two seeds:

| radius fraction | `rate0.3` | `rate0.5` | r_median | alias max |
|---|---|---|---|---|
| 0.10 | 16 (16, 16) | — | 21.5 | 0.85–0.87 |
| **0.15** | **20.5** (19, 22) | 16.5 (16, 17) | 29 | 0.81 |
| 0.20 | 17.5 (17, 18) | 17 (15, 19) | **32** | 0.79–0.92 |
| 0.25 | 15 (15, 15) | — | 28 | 0.86–0.95 |

An optimum in both directions on both axes: 0.15 beats 0.10 and 0.20 at fixed
`rate_lambda`, and 0.3 beats 0.5 at fixed radius.

A clean inverted-U. Note that 0.20 has the *higher* `r_median` and the lower
`r_min`: past the peak the encoder keeps getting better at its typical
reference and starts getting worse at its worst one, which is the same
divergence §4.5b found at a fixed radius of 40 and the reason §4.4b's law
tracks `r_median`. On this metric the peak is where the references stop
agreeing, not where the decay stops widening.

### 4.6b Coverage is the strongest lever, and it removes the seed noise (`w13`)

§4.6 predicted this from the −0.47 correlation between a reference's distance
to the nearest training patch and its radius: `r_min` is the worst of 20
references, the worst are the ones training never reached, and coverage is the
only thing that fixes those. Holding the mix *shape* fixed — same three sizes,
same ~71% of area at 200 cells — and moving only coverage, at the winning radius
and `rate_lambda`:

| coverage | n | r_min by seed | median | spread | r_median | alias max | alias mean | decay50 |
|---|---|---|---|---|---|---|---|---|
| 22.9% (`mixtop`) | 4 | 19, 22, 16, 14 | 17.5 | 8 | 27.5 | 0.853 | 0.607 | 40.8 |
| 38.2% | 2 | 25, 22 | 23.5 | 3 | 37.2 | 0.767 | 0.523 | 41.0 |
| **50.8%** | 4 | 26, 27, 31, 30 | **28.5** | **5** | 42.8 | 0.671 | 0.468 | 42.0 |
| 61.1% | 4 | 30, 30, **0**, 27 | 28.5 | **30** | 44.2 | 0.670 | 0.432 | 42.0 |
| 70.1% | 2 | 23, 16 | 19.5 | 7 | 44.2 | 0.608 | 0.450 | 42.0 |
| *best unconstrained (§1)* | 1 | 21 | 21 | — | 28.5 | 0.814 | ~0.63 | 37.5 |

**The stable columns are monotone in coverage; `r_min` is not.** `r_median`
runs 27.5 → 37.2 → 42.8 → 44.2 and the alias ceiling 0.853 → 0.767 → 0.671 →
0.608, both cleanly, and both pass the unconstrained encoder's values by 38%
coverage. `r_min` peaks somewhere around 50–61% and is erratic throughout —
including a **0** at 61.1% seed 44, whose `r_median` of 46 is the best in the
campaign. One reference out of twenty ruins it, which is §4.8 again.

decay50 is flat at 41–42 across the whole sweep, so coverage buys the *ceiling*
and nothing else — §4.4b's other factor, by §4.6's mechanism: more of the arena
inside a patch is less of it extrapolated to.

**The defensible choice is 50.8%**: median 28.5, spread 5, every seed above the
unconstrained 21. 61.1% ties the median and risks a zero; 70.1% is worse on two
seeds and cannot be placed at four (§4.7).

### 4.8 The 20-reference `r_min` is unstable, and it flatters §1–§3

Every run in §1–§4 is scored at the same 20 reference positions (`ur_seed=0`),
so a headline could be an artefact of which points were drawn. Re-scoring the
leaders at **100** references, and at a second reference seed, says it partly
is:

| encoder | 20 refs | 100 refs, seed 0 | 100 refs, seed 1 | `r_median` (100) |
|---|---|---|---|---|
| `top_f0.15_rate0.3` s42 | 19 | **16** | **15** | 28–29 |
| `top_f0.15_rate0.3` s43 | 22 | 13 | 13 | 27.5–29 |
| `top_f0.10_rate0.3` | 16 | 12 | 11 | 22–23 |
| best unconstrained (§1) | 21 | **9** | 15 | 28.5–29 |
| untreated True baseline | 3 | 2 | 2 | 3 |

**The §1 benchmark of 21 falls to 9 on one draw and 15 on the other.** `r_min`
is a minimum over references, so it can only fall as references are added — but
falling by 12 says the 21 rested on a favourable sample, and *every* headline in
§1–§3 rests on the same 20 points. Treat those numbers as an upper bound on a
noisy statistic, not as measurements. `r_median` is the stable companion: it
barely moves between 20 and 100 references for any encoder here.

The constrained encoders move much less (19 → 16, 15), which is a result in its
own right: they are more uniform across the arena, so the worst reference is
closer to the typical one. That is also why the comparison survives the harder
test and improves on it.

Reproduce with::

    python -m encoder_training.sweep_unique_radius --encoders-dir <sweeps> \
        --pattern '<run>/encoder_final.pt' --n-refs 100 --seed {0,1}

#### 4.8b The final comparison, on references nothing was selected against

Every seed of the two best coverage cells, the unconstrained benchmark and the
untreated baseline, scored together at 100 references and two reference seeds:

| encoder | r_min (s0 / s1) | r_median | alias | decay50 |
|---|---|---|---|---|
| `cov51` seed 42 | 22 / 23 | 43 | 0.792 | 42 |
| `cov51` seed 43 | 24 / 27 | 45 | 0.735 | 42 |
| `cov51` seed 44 | 24 / **4** | 43 | 0.756 | 42 |
| `cov51` seed 45 | 28 / 23 | 44 | 0.751 | 43 |
| `cov61` seed 42 | 32 / **5** | 47 | 0.666 | 42 |
| `cov61` seed 43 | 23 / 21 | 45 | 0.667 | 42.5 |
| `cov61` seed 44 | **0 / 2** | 47 | 0.690 | 43 |
| `cov61` seed 45 | 25 / 29 | 46 | 0.627 | 43 |
| **best unconstrained (§1)** | **9 / 15** | **28.5** | **0.825** | **36** |
| untreated True baseline | 2 / 2 | 3 | 0.977 | 17 |

**`cov51` median `r_min` is 24 (s0) and 23 (s1) against the unconstrained
encoder's 9 and 15** — roughly double, on references neither was tuned for. The
stable columns are not close: `r_median` 43–45 against 28.5, decay50 42 against
36, ceiling 0.74–0.79 against 0.825.

The occasional 4 and 5 are the metric, not the encoder: those runs have
`r_median` 43 and 47 and differ from their siblings by one reference out of a
hundred. `cov61` seed 44 is the exception — 0/2 at `r_median` 47 is a genuine
localised failure, and it is why §4.6b recommends 50.8% over 61.1%.

### 4.7 Infrastructure notes

* `data.build_patch_codes` builds each patch's codes directly instead of slicing
  the 10.2 GB full codebook, dropping a run's host memory from ~20 GB to ~1 GB
  (`--lazy_codes`, verified against the old path in
  `tests/test_lazy_patch_codes.py`). The two group the Gaussian factors
  differently, so codes agree to float32 rounding rather than bit-for-bit — fine
  within a wave, but a seed-for-seed replay of a §2/§3 run needs it off.
* **Packing runs onto one GPU does not work, measured.** Four runs sharing an
  A100 each ran at 5.0 epochs/min against 20 solo — exactly 4× slower, no gain.
  The step is bandwidth-bound on the 8192² pair masks, not launch-bound, and
  bandwidth is the shared resource. `RUNS_PER_JOB` stays 1.
* **The throughput ceiling is a QOS cap, not the partition.** Pending jobs sit
  at `Reason=QOSMaxGRESPerUser` once 16 are running, so queueing more waves
  buys nothing and only the *order* matters. `scontrol update jobid=N nice=…`
  on the pending jobs of a wave is how to reorder; everything still runs.
* What did help before hitting that cap: asking for 1.5 h and 16 GB rather than
  12 h and 80 G, which lets backfill drop a run into a gap. That took the
  concurrent count from 2–3 to 16.

---

# 5. How good can it get at **10% coverage**?

**Status: running.** Step 1 (`w17_lowcov_anchor`, 10 runs) and step 3
(`w18_placement`, 4 runs) are on the cluster. Running log in §5.6.

## 5.1 The goal

§4 found coverage to be the strongest lever it had, and its answer used
**50.8%** of the arena. That is a lot of the world to have visited. The question
now is how much of §4's result survives when the training set is cut to **~10%
coverage**, with everything else about the brief unchanged.

Everything from §4.0 still binds:

* `exclude_cross_env_pairs=True` throughout — the constraint, not a variable;
* no patch larger than **200** cells a side;
* patch sizes **mixed**, not uniform;
* `loss_mode=cka` excluded, and so is `graded_sigma` (§4.4 note — it fits a
  target kernel, which is the same family);
* every other knob free.

Legal/loophole split is unchanged: a term may not consult which environment a
pair came from. `uniformity_scope=nonnear` and `input_far_tau` remain the
labelled loopholes.

## 5.2 Where §4 leaves this

The §4 winner, for reference — this is the config to start from:

```
npos_list            26x200 + 14x150 + 14x100     (mixtop_max, 50.8%, 54 envs)
per_env_radius_frac  0.15
rate_lambda          0.3                          (MCR^2 coding rate)
exclude_cross_env_pairs, single_env_batch=False, lazy_codes
out_dim 1024  hidden_dim 512  num_hidden_layers 4
lr 1e-4  batch_size 8192  fwhm_ratio 0.25  gain 1.0->5.0
epochs step-matched to ~73,000 optimizer steps
```
scoring `r_min` 26, 27, 30, 31 over four seeds; 24/23 at 100 references.

**What the coverage sweep (§4.6b) predicts for 10%.** Extrapolating the
measured curve downward:

| coverage | 61.1% | 50.8% | 38.2% | 22.9% | **10%** |
|---|---|---|---|---|---|
| `r_min` median | 28.5 | 28.5 | 23.5 | 17.5 | **~10–13?** |
| alias ceiling | 0.670 | 0.671 | 0.767 | 0.853 | **~0.90?** |
| decay50 | 42 | 42 | 41 | 40.8 | ~40 |

decay50 was *flat* across the whole coverage sweep, so the prediction is that
10% costs the **ceiling** and only the ceiling — which by §4.4b's law
`r_min = r_at_cos0.9 · sqrt(ln(1/C)/ln(1/0.9))` is worth roughly a factor
`sqrt(ln(1/0.90)/ln(1/0.671)) = 0.51`, i.e. about half. **If the answer comes in
far below ~13, something other than the ceiling has broken and that is the
finding.**

**Why 10% is a different problem, not just a smaller one.** §4.6 measured
`corr(distance from a reference to the nearest training patch, its radius) =
−0.47`: the references that hold `r_min` down are the ones training never
reached. At 50.8% coverage almost every reference is near a patch. At 10% almost
none are, so the task stops being "learn the within-patch structure" and becomes
"extrapolate it to the 90% never seen". Two consequences worth expecting:

1. **Where the patches sit should start to matter as much as how big they are.**
   Placement is currently uniform-random rejection sampling
   (`data.sample_nonoverlapping_patches`), which at 10% leaves large holes. See
   §5.4 step 3.
2. **Overfitting becomes plausible for the first time.** 280k training points,
   1.5M parameters, and ~73,000 steps means each point is seen ~2000 times
   against ~380 at 50.8%. `weight_decay` has been at 1e-4 and untouched all
   campaign.

## 5.3 Geometries, placement-checked

All ≤200 cells, all mixed except the two controls, all verified to place at
seeds 42/43/44/45 (the §4.7 lesson: check every seed a wave will use). Epochs
are step-matched by the driver, so all of these run ~73,000 steps.

| name | spec | envs | coverage | area at 200 | steps/ep | epochs |
|---|---|---|---|---|---|---|
| `lo_big` | 7×200 | 7 | 9.5% | 100% | 34 | 2150 |
| `lo_mixtop` | 5×200 + 3×150 + 3×100 | 11 | 10.1% | 67% | 36 | 2050 |
| `lo_mix2` | 4×200 + 12×100 | 16 | 9.5% | 57% | 34 | 2150 |
| `lo_many` | 29×100 | 29 | 9.9% | 0% | 35 | 2100 |
| `lo_tail` | 3×200 + 4×150 + 6×100 + 8×70 | 21 | 10.5% | 39% | 37 | 1950 |

`lo_mixtop` keeps §4's winning *shape* (three sizes, ~⅔ of the area at 200) at
a tenth of the coverage, so it is the natural anchor. `lo_big` and `lo_many` are
the uniform controls at either end. `lo_tail` carries the 70-cell tail that
§4.5 found actively harmful — included to check that finding still holds when
patches are scarce, since the reasoning there ("a 70-cell patch reaches 99 cells
and the far field never hears it") may change when *nothing* reaches the far
field.

**Watch the wall clock.** 2150 epochs is 2.4× any run in §4. Total steps match,
so time should too (~60–90 min), but per-epoch overhead is now paid 2150 times.
Check `eta.sh`-style projections on the first cell and pass `--time` if needed —
a running job's limit cannot be raised (§4.7).

## 5.4 First steps, in order

**Step 1 — anchor (6 runs).** `lo_mixtop` with §4's winning loss settings
(`per_env_radius_frac=0.15`, `rate_lambda=0.3`), 3 seeds; plus `lo_big` and
`lo_many` at 1 seed each as the uniform brackets. This says how much of §4
survives the cut and whether the ~10–13 prediction holds.

```
python -m encoder_training.sweep_ecp w17_lowcov_anchor
```

**Step 2 — re-tune the two factors at the new coverage (12 runs).** Their optima
were found at 22.9% and need not hold. `per_env_radius_frac ∈ {0.10, 0.15,
0.20}` × `rate_lambda ∈ {0.3, 1.0}` × 2 seeds on `lo_mixtop`. Note §4.4's other
finding: at a 10-cell radius `uniformity_lambda=1.0` beat `rate0.3`, and at
0.15·side it lost — so if the radius optimum moves down here, re-test uniformity
in the winning slot too.

**Step 3 — placement, the new idea (4 runs, *brought forward*).** At 10%
coverage the *arrangement* of patches should matter, and it is the one thing §4
never varied. Random rejection sampling leaves holes; a **jittered lattice**
(partition the arena into a coarse grid, place one patch per cell at a random
offset within it) bounds the worst-case distance from an arena point to the
nearest patch, which is exactly the quantity §4.6 correlated with `r_min` at
−0.47. Built as `--patch_placement {random,stratified}` (§5.6b), so the
comparison is one flag. Run early, at §4's loss settings rather than step 2's
optimum, because step 1 left 6 of the 16 GPU slots idle and w17's `lo_mixtop`
and `lo_many` cells are already the paired random controls.

*Prediction, recorded before it runs:* stratified should beat random by more at
10% than it would at 50%, and the gap should show up in `r_min` and the alias
ceiling while leaving decay50 alone.

**Step 4 — overfitting (6 runs).** `weight_decay ∈ {1e-4, 1e-3, 1e-2}` × 2 seeds
at the step-3 best. Cheap, never explored, and this is the first regime where it
should bite. If it does, also try fewer total steps.

**Step 5 — confirm.** 4 seeds on the leader, then re-score at 100 references and
two `ur_seed`s against the §4 50.8% winner and the untreated baseline
(`sweep_unique_radius --n-refs 100 --seed {0,1}`). §4.8: the 20-reference
`r_min` is unstable to the draw and no claim should rest on it.

## 5.5 Method notes that cost time in §4

* **Read only finished cells.** Four conclusions in §4 were drawn from mid-run
  evaluations and all four were wrong the same way — large-radius and
  strong-spread arms start slowest, so any fixed-epoch comparison reads their
  slow start as collapse. `collect_ur --ckpt final` is the honest view.
* **Two seeds is not enough**, repeatedly demonstrated (§4.9, §4.6b). Budget 4
  on anything that will be quoted.
* **Placement-check every seed** a wave will use, not the first two (§4.7).
* Rank on `r_median` and the alias ceiling when `r_min` is noisy; they moved
  monotonically through the whole coverage sweep while `r_min` did not.
* Tooling: `sweep_ecp.py` (waves, named mixes, step-matched epochs,
  `--only`/`--time`), `collect_ur.py` (grouped read, `--ckpt final`),
  `radius_law.py` (the two-factor law and its `--target` inverse),
  `alias_structure.py` (`--ref_vs_patch` is the diagnostic that motivated
  step 3).

## 5.6 Running log

### 5.6a Step 1 launched, widened to all five geometries (`w17_lowcov_anchor`)

10 runs, `lo_mixtop` / `lo_big` / `lo_many` / `lo_mix2` / `lo_tail` × seeds
42, 43, all at §4's winning loss settings (`per_env_radius_frac=0.15`,
`rate_lambda=0.3`) so this isolates coverage. §5.4 asked for three geometries;
the queue was empty and the cap is 16 GPUs, so ten runs cost the same wall clock
as six and step 2 can then tune on a geometry that was measured rather than
assumed.

Throughput on the first cell: ~46 epochs/min, so 2050 epochs is ~45 min — the
§5.3 wall-clock worry does not bite, but the wave went out at `--time 2:00:00`
anyway because a running job's limit cannot be raised.

### 5.6b Placement: the holes are real at 10% and absent at 50%

`--patch_placement stratified` added to `data.sample_nonoverlapping_patches`,
threaded through `PatchConfig` and `train.py`. Two regimes, because a patch can
be larger than its share of the arena:

* **sparse** — some grid of ≥ `nenv` cells has cells the largest patch fits in.
  One patch per cell, offset to stay wholly inside. *Placement cannot fail and
  no rejection is needed*: the gap between neighbouring cells does the work
  rejection sampling has to search for. All five §5.3 mixes are here.
* **dense** — no such grid. Finest grid whose cells still fit the largest
  patch, patches dealt round-robin largest-first, rejection *within* a cell.
  Cross-cell collisions remain impossible, so the search stays local.

The mechanism is measured, not assumed. Distance from an arena point to the
nearest training patch, mean over seeds 42–45, on a 6-cell lattice:

| mix | coverage | max (rand → strat) | p95 | mean |
|---|---|---|---|---|
| `lo_big` 7×200 | 9.5% | 882 → **802** | 594 → 568 | 227 → 190 |
| `lo_mixtop` | 10.1% | 839 → **461** | 565 → 300 | 198 → 128 |
| `lo_mix2` | 9.5% | 672 → **343** | 420 → 236 | 154 → 107 |
| `lo_many` 29×100 | 9.9% | 475 → **279** | 268 → 167 | 101 → 76 |
| `lo_tail` | 10.5% | 604 → **401** | 354 → 254 | 126 → 104 |
| `mixtop_max` (§4 winner) | 50.8% | 192 → **188** | 66 → 64 | 15 → 15 |

Two things fall out before any GPU time is spent. At 10% stratifying roughly
halves the worst hole. At 50.8% it changes nothing (192 → 188) — random
sampling has already left nowhere far from a patch, and the worst hole there is
*smaller than the best hole any 10% layout achieves*. That is the control the
step-3 claim needs, and it explains why §4 never had to think about placement.

`lo_big` is the exception: 7 patches on a 3×3 grid still leaves 802, because
with so few patches the grid is too coarse for the jitter to matter. If
placement turns out to help, it will help the mixes, not the sparse control.

Launched as `w18_placement`: `lo_mixtop` and `lo_many`, stratified, seeds 42/43,
at §4's loss settings — paired one-flag A/Bs against `w17` 000/001 and 004/005.

Tests in `encoder_training/tests/test_patch_placement.py`: every mix places at
every seed, nothing overlaps, the hole shrinks, unknown modes are rejected, and
— the one that matters for continuity — the `random` path returns bit-identical
layouts, since every number in §1–§4 came from it.

### 5.6c The prediction was wrong, and the law was right

First finished cells of `w17`, all `encoder_final`:

| run | `r_min` | **`r_pred`** | `r_median` | alias | decay50 | res90 |
|---|---|---|---|---|---|---|
| `lo_mixtop` s42 | 5.0 | **5.4** | 15.5 | 0.985 | 37.5 | 14.5 |
| `lo_mixtop` s43 | 8.0 | **7.5** | 15.0 | 0.956 | 31.5 | 11.5 |
| `lo_many` s42 | 6.0 | **5.9** | 9.5 | 0.956 | 23.0 | 9.0 |
| *§4 winner, 50.8%* | *28.5* | | *43–45* | *0.671* | *42* | *~16.3* |

**§5.2 predicted `r_min` 10–13 and got 5–8.** It also said, in advance, that
coming in far below 13 would mean something other than the ceiling had broken.
It had. Two things to separate:

**The law transferred.** `r_pred` is within 0.5 of `r_min` on all three cells —
a regime with a fifth of the coverage the law was fitted on, and it still
predicts to half a cell. §4.4b holds up as the right description.

**The prediction failed because both of its inputs moved, and §5.2 asserted one
of them would not.** The claim was that decay50 was flat across the coverage
sweep, so 10% would cost the ceiling alone. Flat is what it looked like from
61.1% → 22.9% (42, 42, 41, 40.8). Below that it falls off:

| coverage | 61.1% | 50.8% | 38.2% | 22.9% | **10.1%** |
|---|---|---|---|---|---|
| alias ceiling | 0.670 | 0.671 | 0.767 | 0.853 | **0.956–0.985** |
| decay50 | 42 | 42 | 41 | 40.8 | **31.5–37.5** |

Extrapolating a flat curve past the end of its data is exactly the mistake §4
kept making in the time domain (reading unfinished runs); this is the same
mistake in the coverage domain. The ceiling also degraded faster than the
log-linear guess — 0.985, not the 0.90 that would have been on trend.

**Why both moved, and what it changes.** At 50.8% roughly half the metric's
references sit inside a training patch; at 10% almost none do. So below some
coverage the metric stops reporting how well the encoder learned the structure
it was shown and starts reporting how well it *extrapolates* — and both factors
of the law degrade together, because both are being read off the same
untrained far field. That reframes the remaining steps:

* it raises the stakes on **placement** (§5.6b, already running) — the holes are
  no longer a detail of the training set, they are most of what is measured;
* it puts **`fwhm_ratio`** into step 2. It has never been moved in any wave
  (w5 was cancelled before it ran) and it is the one knob that acts directly on
  how smoothly the input varies in space, which is what extrapolation rides on;
* it means the radius fraction has to be swept *upward*, not narrowly around
  0.15 as §5.4 planned, because decay50 now needs raising rather than holding.

### 5.6d Which factor to spend the runs on — the law says the ceiling, by a lot

Both factors fell, but they are not equally worth chasing. `r_min = res90 ·
sqrt(ln(1/C) / ln(1/0.9))` depends on the ceiling through `sqrt(ln(1/C))`,
which is brutally sensitive as `C → 1` and nearly flat away from it. From the
measured 10% cell (res90 14.5, C 0.985, `r_min` 5.5), moving one factor at a
time:

| fix the ceiling, res90 untouched | | fix the decay, C untouched | |
|---|---|---|---|
| C = 0.95 | `r_min` 10.1 | res90 = 25 (decay50 64) | `r_min` 9.5 |
| C = 0.90 | `r_min` 14.5 | res90 = 40 (decay50 103) | `r_min` 15.1 |
| C = 0.80 | `r_min` 21.1 | res90 = 50 (decay50 129) | `r_min` 18.9 |
| C = 0.671 (§4's) | **`r_min` 28.2** | — | — |

**The ceiling alone can recover the entire §4 result at 10% coverage**, with the
decay left exactly where it is. The decay cannot get halfway there even at
values no run in this campaign has ever produced — the best decay50 ever
measured is 42, and the table above needs 103 to match what C = 0.90 gives for
free. So the honest reading of §5.6c is not "both factors fell, fix both"; it is
**the ceiling is the whole game at low coverage**, and the decay is a rounding
error by comparison.

`w19_lowcov_loss` is allocated that way: six arms on the spread term (`rate` 1,
3, 10; `uniformity` 1, 3; and `unif1+rate0.3`), two on the radius fraction
(0.25, 0.40), one on `fwhm_ratio`. 18 runs.

Note this puts **uniformity back in a regime where §4 measured it winning.**
§4.4 found uniformity beating the coding rate on the ceiling at a *narrow*
radius (0.863 against 0.919) and losing at the wide one that eventually won —
so the final config used `rate`. Low coverage is ceiling-limited, which is the
half of that trade where uniformity was ahead.

### 5.6e Being trained on does not save a reference — and that demotes placement

`alias_structure --ref_vs_patch --ur_refs 100`, on three 10% encoders and §4's
50.8% winner. The reference positions are identical across all four (same
lambdas, `n_refs`, border and seed), so only the patch layout differs.

| encoder | inside a patch | median patch dist | max | `r_mono` inside | outside |
|---|---|---|---|---|---|
| `cov51` (50.8%) | 60/100 | 0 | 104 | 44.3 | 40.7 |
| `lo_mixtop` (10.1%) | 15/100 | 114 | 627 | 18.1 | 10.8 |
| `lo_big` (9.5%) | 14/100 | 190 | 627 | 18.3 | 12.7 |
| `lo_many` (9.9%) | 8/100 | 74 | 352 | 14.0 | 10.8 |

Half of §5.6c holds: exposure collapses, 60 references inside a patch down to
8–15. The other half — the inference that the metric is therefore reporting
*extrapolation into the holes* — does not survive the measurement.

**A reference sitting inside a training patch at 10% coverage (18.1) does far
worse than a reference outside every training patch at 50.8% (40.7).** Being
trained on is worth about 7 radius units; having a well-covered arena is worth
about 25. So the damage is not localised to the holes at all — it lands on
well-trained positions just as hard.

The mechanism this points at: `r_min` for a reference is killed by whatever
*other* position aliases to it, and that partner is drawn from the 90% of the
arena training never touched. A reference can have perfect local structure and
still score 3 because something 600 cells away collides with it. Coverage does
not work by teaching each reference its own neighbourhood; it works by leaving
fewer untrained positions available to alias against everything.

That is a second, independent line to the same place as §5.6d — the ceiling is
the lever — and this time it arrives by a different measurement.

**It also argues against the step-3 placement idea, so, before `w18` reports:**
the correlation placement is supposed to exploit is *weaker* here than the −0.47
§4.6 found at 22.9% — `corr(patch_dist, r_mono)` is −0.31, −0.32, −0.14, i.e.
about 10% of the variance. And the reference that actually sets `r_min` is not
the far one: it sits at patch distance 254 (rank 80/100) for `lo_mixtop` but at
**5** (rank 15/100) for `lo_big`. Evening out the holes addresses a weak and
inconsistent predictor.

Revised step-3 prediction, replacing the one in §5.4: stratified placement
should produce **little or no gain** — call it under 2 radius units, inside the
seed spread. §5.6b showed it halves the worst hole, so if the hole were the
mechanism the effect would be large and obvious. `w18` is already running and
will settle it either way; the value of the arm is now that it tests the
mechanism, not that it is expected to win.

### 5.6f Step 1 complete: at 10% coverage, geometry is spent

All ten `w17` cells, `encoder_final`, two seeds each:

| arm | `r_min` med | spread | `r_median` | alias max | alias mean | decay50 | res90 |
|---|---|---|---|---|---|---|---|
| `lo_mixtop` 5×200+3×150+3×100 | **6.5** | 3 | **15.25** | 0.971 | 0.824 | 34.50 | 13.00 |
| `lo_many` 29×100 | 6.0 | 0 | 9.25 | 0.956 | 0.841 | 22.75 | 9.00 |
| `lo_tail` +70-cell tail | 5.0 | 2 | 12.25 | 0.968 | 0.810 | 32.25 | 11.50 |
| `lo_mix2` 4×200+12×100 | 5.0 | 0 | 11.25 | 0.971 | 0.833 | 31.75 | 10.75 |
| `lo_big` 7×200 | 4.5 | 3 | 12.50 | 0.970 | 0.862 | 38.25 | 14.50 |
| *§4 winner, 50.8%* | *28.5* | *8* | *43–45* | *0.671* | | *42* | *~16.3* |

**Every geometry lands in `r_min` 4.5–6.5 with a ceiling of 0.956–0.971.** From
7 large patches to 29 small ones — a 4× range in patch count and the whole
size-mix question §4.5 spent a wave on — the answer does not move. Against §4,
where geometry and coverage were the two biggest levers found, that is the
surprise: at 10% coverage **geometry is spent**, and the ceiling is pinned near
0.97 no matter how the patches are arranged.

The internal evidence for §5.6d is stronger than the argument was. Across these
five arms decay50 ranges 22.75 → 38.25, a 68% spread, and `r_min` moves the
*wrong way* over it: `lo_big` has the best decay50 in the wave (38.25, nearly
§4's 42) and the worst `r_min` (4.5), because its alias mean is the worst
(0.862). A factor that varies by 68% while contributing nothing is not the
binding one.

§4's geometry findings do survive, but only in the factor that no longer pays.
"Bigger patches beat smaller" still holds in decay50 (`lo_big` 38.25 against
`lo_many` 22.75, exactly ordered by patch size, since the near radius is a
fraction of the side). And the 70-cell tail §4.5 found harmful is again the
second-worst arm. Both are real; neither reaches `r_min` any more.

`lo_mixtop` is carried into step 2 on `r_median` — 15.25 against 12.5, 12.25,
11.25, 9.25, the one column that separates the arms cleanly, and the one §5.5
says to rank on when `r_min` is inside the noise (spreads here are 0–3 on
values of 4.5–6.5).

**Where this leaves the question.** §5.2 predicted 10–13 and the anchor gives
4.5–6.5. Nothing about the *training set* recovers it: coverage is fixed at 10%
by the brief, patch size is capped at 200, and every arrangement of those has
now been tried. If 10% coverage is to do better it has to come from the loss,
which is what `w19` is for — and by §5.6d it has to come specifically from the
ceiling, which no geometry moved below 0.956.

### 5.6g Step 3 answered: placement does what it was supposed to, and it is not enough

`w18` against the matching `w17` cells — same loss settings, same seeds, one
flag apart:

| geometry | placement | `r_min` med | spread | `r_median` | alias max | decay50 |
|---|---|---|---|---|---|---|
| `lo_mixtop` | random | 6.5 | 3 | 15.25 | 0.971 | 34.50 |
| `lo_mixtop` | **stratified** | **7.5** | 1 | 12.25 | 0.961 | 34.00 |
| `lo_many` | random | 6.0 | 0 | 9.25 | 0.956 | 22.75 |
| `lo_many` | **stratified** | **6.0** | 0 | 10.25 | 0.932 | 23.50 |

**+1.0 and +0.0.** The revised §5.6e prediction — under 2 radius units, inside
the seed spread — is confirmed; on `lo_mixtop` the gain is smaller than the
random arm's own seed spread of 3, and `r_median` moves the other way.

Worth separating the two predictions, because they scored differently:

* §5.4's original claim was that the gain would show up in `r_min` and the alias
  ceiling and leave decay50 alone. **Directionally that is exactly what
  happened** — the ceiling improves in both geometries (0.971 → 0.961, 0.956 →
  0.932) and decay50 does not move (34.50 → 34.00, 22.75 → 23.50).
* The magnitude was wrong, and §5.6e caught it before the runs landed.

So this is a clean falsification of the *hole* mechanism rather than of the
sampler. §5.6b measured stratifying as halving the worst hole — 839 → 461 on
`lo_mixtop`, 475 → 279 on `lo_many`. If the distance to the nearest patch were
what set `r_min`, halving it would have been unmissable. It bought one unit and
zero. §5.6e's reading — that a reference is killed by an aliasing partner drawn
from the untrained 90%, not by its own distance to a patch — survives a test
that could have refuted it.

It also puts a number on how little the ceiling can be moved from the data side:
the best any placement of any geometry achieves is **0.932**, and §5.6d's table
needs 0.85 or below before the ceiling starts paying properly.

Two smaller things worth keeping. Stratifying cuts the seed spread from 3 to 1,
which is what you would expect once the layout stops being random — useful if a
later wave needs to resolve small differences. And it is not adopted as the
default: it costs `r_median` on `lo_mixtop` (15.25 → 12.25), which is the column
§5.6f ranked the geometries on.
