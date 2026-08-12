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

**Current best: `ur_loss_20260811/016_repel_weight=5_per_env_radius_frac=0.2_seed=43`
— `r_min = 20`, `r_median = 25.5`, `alias_ceiling = 0.879`, `decay50 = 44.0`.**

```
encoder_type mlp      lambdas 11,12,13     out_dim 1024
hidden_dim 512        num_hidden_layers 4
nenv 60               npos 100             (60 patches of 100×100 = 20.4% coverage)
per_env_radius_frac 0.2                    ("near" = within 20 cells)
loss mse_contrastive  attract_lambda 2.0   repel_weight 5.0   uniformity_lambda 0.0
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

**Below 1.0, training goes unstable** (`ur_loss2_repel_low`, first 12 runs) —
so the trend above must *not* be extrapolated:

| repel | frac | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|
| 0.25 | 0.10 | 18 | 15 | **5** |
| 0.25 | 0.15 | 10 | 10 | **1** |
| 0.25 | 0.20 | 9 | 8 | **1** |
| 0.50 | 0.10 | 14 | **0** | **0** |

The collapsed runs have alias ceilings of 0.993–0.9996 — near-total aliasing —
while their decay widths are the *largest* seen (45–64). Seed 44 scored 18 at
repel=1.0 in the previous sweep, so this is an interaction with weak
repulsion, not a bad seed.

Reading: repulsion is what breaks the collapsed basin where everything looks
alike. Too little and some seeds never escape it; too much and it flattens the
neighbourhood. **The optimum sits near repel ≈ 1**, which was the edge of the
first grid. Note the seed spread within a cell (0→14) is far wider here than
the ~2 units seen at repel ≥ 1: this is instability, not noise, and it means a
single run at low repel is uninformative.

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
| `ur_loss_20260811` | repel [1,5,15,40] × frac [0.05,0.1,0.2] × 3 seeds | 36 | **best 20**; repel should go down; diagonal ridge |
| `ur_seb_control` | single_env_batch [T,F] × 3 seeds | 6 | True 2/2/2, False 17/18/18 — §2.1 confirmed |
| `ur_loss2_repel_low` | repel [0.25,0.5,1,2] × frac [0.10,0.15,0.20] × 3 seeds | 36 | *running* |
| `ur_seb_A_geometry` | True; npos_list [4×400, 2×600, 1×800] × repel [1,5] × 2 seeds | 12 | *running* |
| `ur_seb_B_uniformity` | True; uniformity_lambda [0,0.1,0.5,2,8] × 2 seeds | 10 | *running* |

### Rescuing `single_env_batch=True` — two hypotheses in flight

**A, geometry.** Under True the only repulsion left is between far pairs
*inside* one patch, so the distance over which codes get separated is bounded
by the patch side — 100 cells against a 1716-cell arena. Growing the patch
should extend that reach; at one arena-sized env, True and False coincide by
construction. Coverage held near 20% and the near-radius pinned at a fixed 10
cells, so neither confounds patch size. Baselines to beat: 60×100 → 2,
15×200 → 2.

**B, uniformity.** `uniformity_loss` is `logsumexp(-t‖zi−zj‖²)` over the batch —
a repulsion that never asks which environment a pair came from, so unlike the
far-pair term it does not need mixed batches to bite. It is the natural
substitute for what True removes, and `uniformity_lambda` has been 0 in every
run in the archive.

The high-dimensional fact that makes B promising, and that I initially got
backwards: **3M codes spread near-uniformly in 1024 dimensions have a maximum
pairwise cosine of ~0.164** (measured; the sqrt(2 ln N / D) estimate gives
0.171). Spread means near-*orthogonality* at this width, not overlap — 3M
points cannot cover a 1024-sphere. So uniformity pushing codes apart should
drive the alias ceiling *down*, potentially far below the 0.988 True produces.
The contrary intuition ("both patches cover the sphere, so they collide") is a
2-D/3-D picture that does not survive at D=1024.
