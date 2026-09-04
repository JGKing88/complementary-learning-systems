# Encoder-Hopfield probe — results

Running log. The harness is `analysis/hopfield_probe/`, specified in
`ENCODER_HOPFIELD_EVAL.md` (tests) and `ENCODER_HOPFIELD_EVAL_VIZ.md` (report).
Everything below is 2026-08-27, `Npos=1716`, `size=20`, 8 worlds × 5 scored
envs, `steps` 1–15, `K` 1–20, four encoders.

---

## 0. Summary

> ## ▸ Read §10.11 first
>
> **The mechanism, and the open questions.** Everything in §10 reduces to one
> variable — the code's **effective dimension** `d_eff`, the participation ratio
> of its covariance. Far-field cosine spread is `1/√d_eff` to within a few
> percent over a 20× range, so aliasing is a tail event in a `d_eff`-dimensional
> space; `attract_lambda` and the coding-rate term compete to set it; and
> `r_min` fails because it is a **product of two opposing functions of `d_eff`**
> and therefore peaks at a different value than reach does.
>
> §10.11 also lists the **four things this does not derive** — including the
> res90 ≈ 5 floor that bounds the entire trade, which is the least understood
> number in this document.
>
> **Best encoder at each coverage** (§10.12–§10.14), all at `β = gain`:
>
> | coverage | checkpoint | gain | mean reach |
> |---|---|---|---|
> | 10% | `w52_attract_fwhm/001_att0.5_seed=43` | 100 | 0.978 |
> | 5% | `w57_cov5/001_half_a0.5_seed=43` | 75 | 0.977 |
> | 2.5% | `w58_cov2.5/011_q_a1_seed=45` | 100 | 0.965 |
> | *1.25%* | *`w60_cov1.25/014_sm35x_a2_seed=44`* | *100* | *0.870* |
> | *0.75%* | *`w61_cov0.75/014_y50_a2_seed=44`* | *200* | *0.727* |
>
> Means are over three scaffold draws × four training seeds. **Reach is flat
> across a 4× range of coverage; what coverage buys is capacity** — dead goals
> at K=20 go 0.08 → 0.25 → 0.42, with K=5 and K=10 at exactly zero throughout.
> **The floor is 2.5%**: at 1.25% the alias rate passes ~0.02, dead goals reach
> the K=5 operating point, and reach drops 0.10 (§10.15). Below that it is a
> steepening knee, not a cliff, and at 0.75% the failure mode changes — dead
> goals appear at **K=1**, where no cross-talk is possible, so the local chart
> itself is failing (§10.17).
>
> **Every basin number before §10.18 is wrong** — measured over the eval env,
> then over a bank that duplicated the goal, then read off a single training
> seed. Corrected across four seeds: **27.0 / 23.0 / 19.2 / 11.5 / 13.5**, a
> clean decline to 1.25% with the bottom two rungs not separable. Reach,
> direction and flow were never affected.
>
> **Saturation is a square, not a ladder** (§10.20). β = 1e6 gives a real
> attractor (the state holds its landing point to cos 0.998 over 15 steps) but
> the fixed point is a hypercube *corner* 0.958 from the memory, not the memory
> — so the basin, which asks whether the state is nearest the goal **cell**
> among 12,853 continuous cells, falls 27.0 → 24.5 on a 0.003 margin even as
> the dynamics improve. Saturating the **encoder** too puts `cos_self` at
> 1.0000 exactly: the basin
> recovers to 28.2, `exact_hit` hits 0.999, and the direction field is
> **destroyed** — acc45 0.997 → 0.392, reach 0.987 → 0.103, because `q` is a
> finite difference and `sign(z)` has no local derivative. Production's corner
> is the only one where memory and direction both work.
>
> Report page (all five rungs plus both saturated arms, encoder selector, full
> Tests A–D, per-encoder basin failure maps, and the basin-across-seeds
> section):
> https://claude.ai/code/artifact/d7a250c1-5044-4854-b453-61881bd518e7
> · 10% spec sheet:
> https://claude.ai/code/artifact/db70ecb9-ca16-4f8b-a897-5dfa0a01d198


**Production's Hopfield is a linear matched filter that degrades cues.** At the
live operating point a cue corrupted to cos 0.70 comes back *worse* — 0.54 at
K=5, 0.27 at K=10 — and the `q` readout decays monotonically with recall depth.
Both are properties of the linear regime, not of the architecture.

**Two independent conditions have to hold at once**, and every earlier attempt
moved one:

| condition | knob | production | what it buys |
|---|---|---|---|
| loop gain / saturation | `β` *or* storage `scale` (same knob twice) | `β·s` 0.004–0.10, tanh arg `u` 1e-4–3e-3 | a nonzero fixed point instead of decay |
| patterns near a hypercube corner | encoder `gain` only | cos-to-binarisation 0.81–0.90 | that the fixed point is *your* memory |

Raising `β` alone is a **net loss** (§2). Raising both together works, unevenly
(§3). The rescue sweep finds settings with genuine basins that hold to K=10
(§4).

**The rule that survived:** raise inference gain until cos-to-binarisation
≈ 0.96 — encoder-specific in gain (v35 ~100, L7 ~300), universal in `cos_bin` —
**and** saturate `β` alongside it. Not "use gain 300", not "raise β" alone, and
**not** the rescue sweep's `alpha` optimum on its own — alpha is a time
constant that only means anything once the loop gain makes the recall term
comparable to the cue (§4.2–4.3, §7).

**Cross-talk, not coding radius, is what the readout reads** (§4.5). At
identical architecture and seed, `attract_lambda` 16→32 leaves `r_min` flat,
raises worst-pair overlap 0.578→0.699, and loses on every probe metric —
retrieval hardest, from 57.8% to 38.8%.

**Best measured setting: v35 at gain 100 with β=1e6** (§3.1) — acc45 100.0% at
every step count from 1 to 15, retrieval 97.5%, reach 99.6%, for 1.2° of mean
angular error against production.

**And the reach objective decomposes** (§10). Continuous reach is per-environment
and nearly binary: a goal dies when **one** co-stored competitor crosses cos
≈ 0.25, that competitor sits ~370 cells away, and at K=1 no encoder loses a
single environment. `r_min` cannot see any of it — six candidates share
`r_min` 12.0 while their far-field alias rate spans seven-fold — and the rate
of distant pairs above 0.25 predicts measured reach at **ρ = −0.92**, and
**−0.85 once the confirmation run's eight arms are added** (§10.8).

**The screen's nomination was run and it won** (§10.8), and then §10.9 found a
better arm needing no inference override at all: `w52_attract_fwhm/*_att0.5` at
its own gain 100 with **β = gain**. Reach rises monotonically as
`attract_lambda` *falls* — 0.806 / 0.931 / 0.972 / 0.987 at 16 / 2 / 1 / 0.5 —
which is the axis w52–w54 spent three waves climbing the other way because
`r_min` rewards it.

**But §10.10 puts a floor on how finely any of this can be read.** The same arm
varies **0.959–0.988** across three scaffold draws, so `att0.5`'s lead over
Level 6 at gain 300 is +0.009 on average, winning two draws and losing one.
Large orderings hold; gaps under ~0.02 between the top arms do not resolve on
one scaffold, and §10.8–§10.9 quote them to three decimals as though they did.

---

## 1. Production baseline

| encoder | gain | \|err\| | acc45 | exact | basin | reach | acc45 @ s=15 |
|---|---|---|---|---|---|---|---|
| v35 | 3.70 | 8.81° | 99.4% | 74.3% | 14.15 | 87.2% | 86.2% |
| L7-s42 | 100 | 11.81° | 97.0% | 57.8% | 11.03 | 78.4% | 76.5% |
| L7-s43 | 100 | 11.84° | 96.9% | 60.9% | 11.35 | 76.9% | 77.2% |
| untrained | 5 | 94.70° | 22.6% | 0.0% | none | 6.9% | 22.6% |

Controls at K=5: oracle 7.2°, local oracle 6.9°, Gram-Schmidt swap 5.9°,
`use_tanh=False` identical to production, `tanh |arg|` max 3.7e-4. The basis is
sound and the recall is linear.

On this pair `unique_radius` and `q` accuracy **agree**. v35 has the better
radius — `r_min` **16**, named as the best of the 407-checkpoint audit in
`EXPERIMENTS_UNIQUE_RADIUS.md` §1 — and it also leads every probe metric:

| | `r_min` | \|err\| | acc45 | exact | reach | pairwise cos |
|---|---|---|---|---|---|---|
| v35 | **16** | **8.81°** | **99.4%** | **74.3%** | **87.2%** | **0.005** |
| L7-s42 | 9.0 | 11.81° | 97.0% | 57.8% | 78.4% | 0.027 |

> **This reverses an earlier claim in this document.** It previously read
> "`unique_radius` does *not* rank encoders the way `q` accuracy does — v35
> leads every probe metric with the worse published `r_min`." That was
> unsupported: v35's checkpoint carries no stored `unique_radius`, and rather
> than look the number up I inferred a worse radius from v35 predating the
> campaign. The doc had 16 on record the whole time.

Two encoders is not a trend, and they differ on more than radius: v35 has 8×
the parameters and **5× lower cross-talk** (worst pair 0.264 vs 0.524). Since
it leads on radius *and* orthogonality, this cannot separate which one the
readout is actually reading — and §4/§7 argue the binding constraint is
cross-talk, which enters `unique_radius` only through the alias ceiling.
A param-matched pair that differs on one of the two would settle it —
**§4.5 is that pair**, and the answer is cross-talk.

## 2. β alone — a net loss

`--beta 1e6`, each encoder at its own gain.

| K=5, s=1 | \|err\| | acc45 | exact | basin | reach |
|---|---|---|---|---|---|
| v35 | 8.81→10.38 | 99.4→98.7% | 74.3→**43.7%** | 14.15→**5.83** | 87.2→**62.6%** |
| L7-s42 | 11.81→13.23 | 97.0→96.7% | 57.8→**18.6%** | 11.03→**1.75** | 78.4→**45.5%** |
| L7-s43 | 11.84→14.95 | 96.9→94.4% | 60.9→**12.9%** | 11.35→**0.77** | 76.9→**37.0%** |

Buys a partial flattening of the steps decay (L7-s42 76.5→92.6% at s=15) and
costs roughly half the retrieval, most of the basin and a quarter of the flow.
Exactly `EXPERIMENTS_NAV_P2.md` §5.7's prediction: the loop gain "does not make
a continuous vector into a corner", and retrieval is scored against a
*continuous* cell bank, so a binarised recall is far from every cell.

## 3. Both conditions — helps, unevenly

`--encoder_gain 300 --beta 1e6`.

| encoder | \|err\| | acc45 | exact | basin | reach | acc45 @ s=15 |
|---|---|---|---|---|---|---|
| v35 | 8.81→19.15 | 99.4→95.1% | 74.3→**100.0%** | 14.15→**21.62** | 87.2→**97.4%** | 86.2→95.0% |
| L7-s42 | 11.81→**10.32** | 97.0→**99.7%** | 57.8→**72.6%** | 11.03→**13.85** | 78.4→**80.7%** | 76.5→**99.6%** |
| L7-s43 | 11.84→**10.00** | 96.9→**99.5%** | 60.9→52.3% | 11.35→8.18 | 76.9→**81.0%** | 77.2→**99.3%** |

The steps decay is **gone** — flat 99.6% to s=15 — which is the clearest result
in the set. But it is not uniform:

- **L7-s42** improves on every metric.
- **L7-s43** improves direction and flow, *loses* retrieval and basin. Same
  architecture, same config, different seed, opposite sign. Do not read this
  off one encoder.
- **v35** trades: retrieval and flow improve sharply, angular accuracy falls.
  Partly self-inflicted — gain 300 is L7's optimum, and v35's own sweep puts it
  at ~100 (cos_bin 0.968, `|err|` 7.5°, acc45 100.0%) against 300's 0.989 /
  14.0° / 98.4%.

### 3.1 v35 at its *own* optimal gain — the best result in the campaign

§3 used gain 300 for all four encoders. That is L7's optimum; v35's is ~100
(`gain_probe_check.py`). Re-run at gain 100 with β=1e6:

| v35, K=5 | \|err\| | acc45 | exact | basin | reach | acc45 @ s=15 |
|---|---|---|---|---|---|---|
| production (g3.7) | **8.81°** | 99.4% | 74.3% | 14.15 | 87.2% | 86.2% |
| g300 + sat | 19.15° | 95.1% | 100.0% | 21.62 | 97.4% | 95.0% |
| **g100 + sat** | 10.03° | **100.0%** | **97.5%** | **20.20** | **99.6%** | **100.0%** |

The 19° in §3 was the wrong gain, not an intrinsic trade — 9° of angular error
bought nothing. At gain 100: **acc45 is 100.0% at every step count from 1 to
15**, retrieval 74.3→97.5%, basin 14.15→20.20, reach 87.2→**99.6%**.

The only regression against production is mean `|err|`, 8.81→10.03°. Note
`acc45` *rises* (99.4→100.0%), so production has ~0.6% of cells beyond 45° and
this has none: the mean is slightly worse while the tail is gone. Strictly
better-behaved distribution.

**One config change buys near-perfect retrieval, +6 cells of basin, +12 points
of reach and complete step-invariance, for 1.2° of mean angular error.** It
changes every embedding, so it needs a training run rather than a config edit,
and it is one encoder at one seed.

## 4. Rescue sweep — real basins exist, and production is nowhere near them

504 cells per encoder: `zero_diag` × `alpha` × `scale` × `β`, with each
encoder's own gain in the grid so β=gain is an explicit anchor row. Recovery is
a cue corrupted to cos 0.70; **below 0.70 means recall made it worse.**

**Production anchor** — degrades cues, worse with load:

| encoder | loop gain `β·s` | tanh arg `u` | K=3 | K=5 | K=10 |
|---|---|---|---|---|---|
| v35 | 0.0036 | 1.1e-4 | +0.698 | +0.543 | **+0.269** |
| L7-s42 | 0.098 | 3.1e-3 | +0.718 | +0.443 | **+0.218** |
| L7-s43 | 0.098 | 3.1e-3 | +0.700 | +0.436 | **+0.222** |

**Best passing cell** — holds to K=10:

| encoder | K=3 | K=5 | K=10 |
|---|---|---|---|
| v35 | 10 / α0.1 / **+0.980** | 10 / α0.1 / **+0.977** | 10 / α0.1 / **+0.968** |
| L7-s42 | 10 / α0.1 / +0.979 | 10 / α0.1 / +0.969 | 1 / α0.5 / **+0.947** |
| L7-s43 | 10 / α0.1 / +0.971 | 10 / α0.1 / +0.962 | 1 / α0.5 / **+0.939** |
| untrained | none passed | none passed | none passed |

Recovery pooled by the sweep coordinate `β·s·√D` peaks in its **1–10** band
for all three trained encoders (0.72–0.76) and falls at both ends. Two routes
reach it — `β=100, scale=1/D` and `β=1, scale=1/√D` — which is the
`(p → λp, β → β/λ²)` invariance showing up in the data.

> **Units, corrected.** These tables originally called `β·s·√D` the "tanh
> argument". It is **D times** the per-coordinate argument `u = β·s/√D`, so the
> 1–10 band is `u` = 1e-3 to 1e-2 — firmly **linear**, not partial saturation
> as the name implied (§7). The coordinate is monotonic in `β·s` so every
> ranking here stands; only the regime label was wrong. Later runs emit
> `loop_gain` and `tanh_u` instead.

**The L7 pair is already inside the optimal band at β=gain=100**, so
on this test their binding knob looks like `alpha`, not the loop gain — a
different knob from the one this campaign spent most of its time on.

> **Tested, and it does not transfer.** §4.2 runs the full suite at
> `alpha=0.5`: `exact_hit` reads 0.2% at s=1. §4.3 then shows why that is a
> *delay* rather than damage, and why alpha and the loop gain cannot be moved
> independently. Read this section's optima as hypotheses about *this test*,
> not as settings.

### 4.1 The untrained control earns its place

It scores **perfect recovery — +1.000, +0.999, +0.999** — and is correctly
rejected at every cell, because `mean_pairwise_cos` catches the collapse
(pairwise 0.9986, effective rank 1.00 of 25). A collapsed encoder sends every
cue to one vector, so a corrupted cue "recovers" perfectly onto it.

Recovery alone would have named the untrained encoder the best in the study.
This is the third time in this campaign a single attractor statistic was
maximised by a degenerate memory — after collapse and after stasis — and it is
why the criterion is all three of self-consistency, pairwise cosine and
recovery, never one.

### 4.2 Rescue's optimum does not transfer — alpha 0.5 on the full suite

§4 said the L7 pair's binding knob is `alpha`, since their loop gain is
already in the optimal band. Tested directly — L7-s42, `alpha=0.5`, nothing
else changed:

| L7-s42, K=5 | \|err\| | acc45 | exact | basin | reach | acc45 @ s=15 |
|---|---|---|---|---|---|---|
| production | 11.81° | 97.0% | 57.8% | 11.03 | 78.4% | 76.5% |
| **alpha 0.5** | 11.75° | 97.1% | **0.2%** | **0.00** | 77.9% | 93.9% |
| both (g300+sat) | **10.32°** | **99.7%** | **72.6%** | **13.85** | **80.7%** | **99.6%** |

Alpha 0.5 delivers what rescue promised on the steps axis (76.5→93.9%) and
leaves direction untouched — and drops `exact_hit` 57.8→0.2% with basin
11.03→**zero** *at s=1*.

> **Overstated, corrected in §4.3.** That is not annihilation, it is delay:
> alpha moves the retrieval peak from s=1 to s=5 or later. Reading a slow
> process at the one step production happens to use made it look broken.

**The flaw is in the rescue test, not in alpha.** With `alpha=0.5` the update
keeps half the *cue*, so the endpoint is a blend of cue and memory. The two
tests start from different places:

- rescue's cue is the goal pattern corrupted to cos 0.70 — already nearly
  right, so damping refines it toward the goal and recovery reads 0.97;
- the task's cue is a *different cell's* embedding, far from the goal, so
  damping keeps the state near where it started and retrieval returns the cue's
  own cell.

Rescue measures **cleanup of a nearly-correct cue**; navigation needs
**retrieval from an arbitrary position**. Damping helps the first and destroys
the second, and the recovery guard cannot see it because from cos 0.70 the
state genuinely does improve.

This is the fourth degenerate route in this campaign and the first the
three-part criterion missed. The lesson is not a fourth criterion: it is that
**rescue's cue distribution is unrepresentative of the task, so its optima are
hypotheses to test on the full suite, never recommendations.**

### 4.3 The alpha sweep — a time constant, not a destroyer

§4.2 said `alpha=0.5` "annihilates retrieval". Wrong, and the trajectory probe
(§6) is what showed it. Sweeping alpha on L7-s42, `exact_hit` by step at K=5:

| alpha | s=1 | s=2 | s=3 | s=5 | s=10 | s=15 |
|---|---|---|---|---|---|---|
| **1.0** (production) | **57.8** | 42.1 | 34.1 | 24.2 | 9.0 | 6.5 |
| 0.9 | 0.7 | 2.8 | 7.4 | **28.3** | 25.4 | 17.3 |
| 0.75 | 0.3 | 0.4 | 1.1 | 2.8 | 17.5 | **28.3** |
| 0.5 | 0.2 | 0.3 | 0.3 | 0.4 | 1.5 | 3.2 *(still climbing)* |

Alpha moves the retrieval **peak later** — s=1 → s=5 → s=15 → past 15 — and
roughly halves it. Reading it at s=1 alone, which is where production lives,
makes a slow process look like a broken one. Still not a win, but the reason
matters: it is a time constant, not damage.

**Why the cliff is at alpha=1.0 exactly.** In
`x ← (1−α)x + α·tanh(β·Wx)` the retained-cue term has norm `(1−α)`, and the
recall term has norm `≈ β/D` in the linear regime (`delta_scale_check.py`):

| | β/D | measured ‖recall‖ | (1−α) at which the cue ties |
|---|---|---|---|
| v35 production | 0.0036 | 0.0034 | 0.0034 |
| L7-s42 production | 0.0977 | 0.0881 | 0.081 |
| v35 g100+β1e6 | *saturated* | **31.96** (= √D) | 0.97 |

So at production the recall signal is **0.3–9% of the cue** — which is just the
loop gain `β·S/D` restated, with `S=1` from unit-norm storage and `β` three
orders below `D=1024`. Any `(1−α)` above that swamps the memory with the cue,
which is why alpha 0.9 already fell off. Saturated, the recall term is capped
at `√D` = 32 and outweighs the cue 32:1, so alpha only becomes a real
integration knob there.

**Alpha and loop gain are not independent.** That is why the rescue sweep's
alpha optimum always came paired with a higher loop gain, and why lifting alpha
out of that pairing (§4) was meaningless.

### 4.4 β sets step-invariance and nothing else

v35 at gain 300, β swept over three decades:

| arm | \|err\| | acc45 | exact | basin | acc45 @ s=15 |
|---|---|---|---|---|---|
| g300, β1e3 | 19.57 | 93.8% | 100.0% | 21.62 | 79.4% |
| g300, β1e4 | 19.54 | 93.8% | 100.0% | 21.62 | 83.3% |
| g300, β1e6 | 19.15 | 95.1% | 100.0% | 21.62 | **95.0%** |
| **g100, β1e6** | **10.03** | **100.0%** | 97.5% | 20.20 | **100.0%** |

A 1000× change in β leaves `|err|` (19.57→19.15), `exact_hit` (100.0%) and
basin (21.62) **untouched**, and moves only `acc45 @ s=15` (79.4→95.0%).
Changing gain 300→100 at fixed β moves `|err|` 19.15→10.03.

- **encoder gain** sets angular error *and* retrieval
- **β** sets step-invariance *and nothing else*

Retrieval at s=1 depends on where a single application lands, which the
patterns decide; step-invariance needs the dynamics to have a fixed point,
which needs saturation.

## 4.5 attract_lambda 16 vs 32 — cross-talk is the binding variable

`w54_attract_far` extends level 7's `attract_lambda` from 16 to 32. It has 12
finished encoders and appears nowhere in `EXPERIMENTS_UNIQUE_RADIUS.md`.
Architecture is identical to att16 — λ [11,12,13], out 1024, hidden 256, 0.572M
params, gain 100, fwhm 0.25 — so paired by seed, `attract_lambda` is the **only**
thing that differs. That is the controlled comparison the v35/L7 pair could not
be (§1).

| | \|err\| | acc45 | exact | basin | reach | acc45 @ s=15 |
|---|---|---|---|---|---|---|
| att16-s42 | **11.81°** | **97.0%** | **57.8%** | **11.03** | **78.4%** | **76.5%** |
| att32-s42 | 15.00° | 94.2% | **38.8%** | **6.35** | **58.0%** | 62.9% |
| att16-s43 | **11.84°** | **96.9%** | 60.9% | 11.35 | 76.9% | 77.2% |
| att32-s43 | 14.63° | 94.7% | 58.8% | 11.25 | 73.9% | 72.0% |

**att32 is worse on every metric on both seeds**, and the radius gain that
motivated it evaporates when paired: `r_min` 12→12 and 15→14. The 4-seed median
advantage quoted from the stored metrics (13 vs 12) came from seeds 44/45; seed
variance within att16 (12 to 15) is larger than the arm difference.

### The two failure modes read different statistics of cross-talk

Direction degrades on both seeds, retrieval only on s42 — which looks like noise
until sorted by **worst-pair** overlap rather than mean:

| | mean pairwise | worst pair | \|err\| | exact |
|---|---|---|---|---|
| att16-s43 | 0.0188 | 0.440 | 11.84° | 60.9% |
| att16-s42 | 0.0275 | 0.578 | 11.81° | 57.8% |
| att32-s43 | 0.0337 | 0.575 | 14.63° | 58.8% |
| att32-s42 | 0.0387 | **0.699** | 15.00° | **38.8%** |

`exact_hit` is monotone in the **worst pair** with a cliff past ≈0.6;
`|err|` is monotone in the **mean**. That is what §7's two conditions predict:
retrieval is an argmax, so it fails when the single worst competitor beats the
self term in condition (b); direction is an average, so it tracks average
interference. s43 survives because its worst pair stays under the cliff even
though its mean rose 79%.

**So cross-talk, not coding radius, is what the readout reads.** §1 could not
separate them because v35 led on both. Here radius is flat and cross-talk is
41% worse, and every probe metric follows cross-talk.

Practical: **`attract_lambda` 16 → 32 is a regression for navigation** even
where `r_min` calls it flat or better. The arm worth trying instead is
`w54_attract_far`'s `rep0.25`, whose alias ceiling (0.833) is the lowest in
w53/w54 — the direction that lowers the worst pair.

Raw: `$CLS_RESULTS/hopfield_probe/20260827/att16_vs_att32/`.

## 5. Caveats

- One scaffold, one seed per encoder, `K ≤ 20`. The two L7 seeds already
  disagree on retrieval sign in §3.
- `β = 1e6` and `alpha = 0.1` are far outside anything run end-to-end.
- An inference-gain change alters every embedding a trained policy was fitted
  to. Nothing here is a config edit for an existing checkpoint; the next step
  is a training run.
- §4's grid never varies encoder gain, so it cannot see §3's regime at all. The
  two sections answer different questions and their optima are not comparable.

## 6. Real-space trajectory — retrieval is a jump, not a walk

`attractor.trajectory_probe` decodes **every step** of the recall trajectory to
its nearest cell and measures the motion in cells. Nothing else in A–D does
this: A decodes only the endpoint, B measures `q`'s angle, D follows the `q`
field with an agent rather than following the recall. Mean distance from the
goal, K=5, start ≈ 8.2 cells:

| arm | s1 | s2 | s3 | s5 | s8 |
|---|---|---|---|---|---|
| v35 production | **0.00** | 0.00 | 0.35 | 0.60 | 1.00 |
| v35 g100+β1e6 | **0.00** | 0.00 | 0.00 | 0.00 | **0.00** |
| L7 production | 0.01 | 0.29 | 0.46 | 1.00 | 1.75 |
| L7 alpha 0.9 | **6.01** | 3.08 | 1.66 | 0.60 | 0.67 |
| L7 g300+β1e6 | 0.35 | 0.50 | 0.50 | 0.50 | 0.50 |

**The first application travels ~10 cells and lands on the goal.** Retrieval is
a one-step jump. Production then *drifts away* — 0.00 → 1.00 by s=8 — which is
the matched-filter decay of §0 made concrete in real space. The saturated arm
lands and **stays at 0.00**: a fixed point, visible as a distance rather than
inferred from a cosine.

And alpha 0.9 turns the jump into a walk — 6 cells out after one application,
arriving around s=5. That is §4.3's correction in one row: the retrieval is not
gone, it has not happened yet.

`ProbeConfig.trajectory_steps` (default 15) runs it on `n_map_worlds`, since it
costs one bank retrieval per step.

## 7. When is it an attractor network? The analysis

Clean in both limits, not in the middle. `regime_check.py` checks the algebra
against the measurements.

**The regime parameter.** For unit-norm patterns `W = s(ZᵀZ − diag)` and
`‖Wx‖ ≈ s` for a cue with support in the memory subspace, so per *coordinate*
the tanh argument is

    u = β·s/√D          transition at u ≈ 1, i.e. β·s ≈ √D

At production's `s = 1/D` that is **β ≈ D^1.5 = 32 768**. The linear-regime
prediction `‖tanh(βWx)‖ ≈ β·s` and the saturated cap `√D` match measurement to
three significant figures:

| | `u` | predicted | measured |
|---|---|---|---|
| v35 production | 1.1e-4 | 0.0033 | 0.0034 |
| L7 production | 3.1e-3 | 0.0879 | 0.0881 |
| v35 g100+β1e6 | 30.5 | 32.0 (`= √D`) | 31.96 |

**Linear limit, `u ≪ 1`.** `x ← normalize(Wx)` is power iteration: the only
fixed points are eigenvectors of `W`, and only the leading one is stable.
Stored patterns are **saddles**. There is exactly one attractor and it is not a
memory. Production sits here, at `u` = 1e-4 to 3e-3.

**Saturated limit, `u ≫ 1`.** `x ← sign(Wx)/√D`, so a stored `z₁` is a fixed
point iff both:

- **(a)** `z₁` is itself a corner, `z₁ = sign(z₁)/√D` — an *encoder* property,
  measured as cos-to-binarisation;
- **(b)** `sign(Wz₁) = sign(z₁)` — a *capacity* property. For corner patterns
  `Wz₁ = s[(1 − K/D)z₁ + Σ_{k≠1} c_k z_k]`, and a coordinate flips when the
  cross-term beats the self-term there. Union-bounding over `D` coordinates
  gives capacity ≈ **D/(2 ln D) = 74** at D=1024.

That is the two-condition structure of §0 *derived* rather than observed, and
(b) predicts a number it was not fitted to: `nav_p2` §5.7 measured capacity
between **50 and 100**.

**What the analysis does not cover.**

1. The intermediate regime `u ≈ 1`. No clean treatment; partial saturation.
2. **The continuous cell bank.** Retrieval is scored against non-corner cell
   embeddings, so a binarised recall is far from *all* of them. This is what
   collapsed `exact_hit` in §2 while the dynamics behaved exactly as predicted,
   and the fixed-point analysis says nothing about it. It is the single largest
   gap between "is an attractor network" and "is a useful position readout".
3. The angular cost of raising encoder gain (§3.1) — that is the geometry of
   the encoder's chart, not the Hopfield.
4. The correlation penalty on (b). Overlap is measured (`alias_ceiling` 0.88)
   but its effect on capacity is not derived.

## 8. Where things live

Raw result JSON, permanent:
`$CLS_RESULTS/hopfield_probe/20260827/` — ten arms plus a `README.md` naming
each. `report.build <dir>` regenerates a page from any of them; nothing is
recomputed, so restyling costs no recall.

The derivation in §7 is also written up as a standalone page, source at
`analysis/hopfield_probe/report/analysis_page.html`.

## 9. Reproducing

    ./analysis/hopfield_probe/run_probe.sh                      # Sec 1
    ... --beta 1e6                                              # Sec 2
    ... --beta 1e6 --encoder_gain 300                           # Sec 3
    ... --beta 1e6 --encoder_gain 100                           # Sec 3.1
    ... --rescue --skip a --skip bc --skip d --skip controls    # Sec 4
    ... --alpha 0.5 / 0.75 / 0.9                                # Sec 4.2-4.3
    ... --encoder_gain 300 --beta 1e3 / 1e4 / 1e6               # Sec 4.4

Per-encoder diagnostics: `gain_gap_check.py`, `gain_probe_check.py`
(`PROBE_CKPT` / `PROBE_GAINS`), `gain_crosstalk_check.py`, `crosstalk_check.py`,
`localchart_check.py`, `contamination_check.py`, `steps_beta_check.py`,
`decouple_check.py`, `corner_check.py`, `delta_scale_check.py`,
`traj_compare_check.py`, `regime_check.py` (Sec 7's algebra vs
measurement).

Sec 10's measurements: `dead_goal_check.py` (per-env reach, dead fraction vs K),
`dead_overlap_check.py` (which goal dies, and why not distance),
`dead_threshold_check.py` (where the overlap line sits),
`alias_lattice_check.py` (the module-realignment falsification),
`gain_tradeoff_check.py` (alias rate against res90),
`nav_screen_check.py` (the pool screen), `nav_screen_validate.py`
(the screen against measured reach), `load_curve_check.py`
(dead fraction against load, on a constant env set),
`l6_confirm_check.py` (Sec 10.8's out-of-sample test). The run itself is
`run_l6.sh`.

## 10. What sets continuous reach — the synthesis

Written 2026-08-30, stepping back from the arms above to ask what would make a
**10% coverage, `exclude_cross_env_pairs=True`** encoder best on *continuous*
reach. Re-analysis of the archived arms plus new measurements over the encoder
pool. Nothing here needed a new probe run.

> **The `reach` column in §1–§4.5 is the discrete rate.** The continuous one is
> the stated objective and it differs — usually higher, because sub-cell steps
> escape sinks a cardinal classifier falls into, but *lower* at the top
> (`v35 g100+β1e6`: 0.996 discrete, 0.974 continuous).

### 10.1 Reach is per-environment and nearly binary

`Scalars` keeps one value per scored environment, so the distribution is
recoverable from the archive. It is a mixture, not a spread:

| arm, K=5 | mean | envs at ≥0.95 | envs <0.5 | worst |
|---|---|---|---|---|
| v35 g100+β1e6 | 0.974 | 90% | **0%** | 0.56 |
| production v35 | 0.867 | 80% | 12% | 0.003 |
| production att16-s42 | 0.806 | 72% | 20% | 0.003 |
| att32-s42 | 0.631 | 53% | 40% | 0.003 |
| β1e6 alone, att16-s42 | 0.525 | 38% | 47% | 0.003 |

Reach against **start distance is flat** — att16-s42 scores 0.83 from cells
adjacent to the goal and 0.77 from 21 cells away. So the failures are not
accumulated error and not a last-half-cell precision problem: in a fifth of
environments the goal is unreachable *from the cell next to it*.

### 10.2 Every failure is interference between stored goals

At **K=1 the dead fraction is 0.00 for every arm**, including the worst. It
then climbs with load — but the load axis needs care, because `scored_envs`
measures `min(K, n_score_envs)` environments, so K=1 and K=3 score a *smaller
env set* than K≥5. That is the same population-grows-with-K confound
`n_score_envs` was added to kill on the retrieval axis, and it reaches the K<5
columns anyway. Fixed by selecting a constant env subset at every K
(`load_curve_check.py`; all 8 worlds, since this needs reach values only):

| dead fraction, envs 0–2, n=24 | K=1 | K=3 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| **v35 g100+β1e6** | 0.00 | **0.00** | **0.00** | **0.00** | **0.00** |
| v35 g300+β1e6 | 0.00 | 0.04 | 0.04 | 0.04 | 0.00 |
| v35 production | 0.00 | 0.12 | 0.17 | 0.12 | 0.33 |
| att16-s42 g300+β1e6 | 0.00 | 0.21 | 0.17 | 0.17 | 0.42 |
| att16-s42 production | 0.00 | 0.17 | 0.25 | 0.29 | 0.75 |
| att16-s43 production | 0.00 | 0.17 | 0.25 | 0.33 | 0.71 |
| att32-s42 production | 0.00 | 0.21 | 0.46 | 0.42 | 0.75 |
| att16-s42, β1e6 alone | 0.50 | 0.50 | 0.54 | 0.58 | 0.71 |

**The encoder changes the slope, not the intercept.** Every arm starts at 0.00
and they diverge, so this is a per-competitor probability that differs by
encoder rather than a fixed penalty. Inverting `P(dead) = 1 − (1−p)^(K−1)`:

| implied per-competitor `p` | K=5 | K=10 | K=20 |
|---|---|---|---|
| v35 g100+β1e6 | <0.01 | <0.01 | <0.01 |
| v35 g300+β1e6 | 0.011 | 0.005 | <0.01 |
| v35 production | 0.045 | 0.015 | 0.021 |
| att16-s42 g300+β1e6 | 0.045 | 0.020 | 0.028 |
| att16-s42 production | 0.069 | 0.038 | 0.070 |
| att32-s42 production | 0.142 | 0.058 | 0.070 |

`p` spans about seven-fold, it is roughly K-stable within an arm, and it tracks
the far-field alias rate of §10.5 at a **ratio of 1.6–3.0** across every arm —
which is what identifies the screen as measuring exactly this quantity. (The
ratio exceeds 1 because "dead" is reach <0.5, which catches partial collapse
as well as outright aliasing.)

So the two factors are **`p` from the encoder and K in the exponent**.
Saturation does not merely lower the rate: `v35 g100+β1e6` is **flat at 0.00
across the whole range to K=20** — condition (b) of §7 paying off, and the
capacity bound `D/(2 ln D) = 74` says that should hold well past 20.

### 10.3 A goal dies when one co-stored competitor crosses cos ≈ 0.25

`multi_env_goals` stores the first K goals of a world, so the competitors are
computable from the checkpoint. Worst overlap separates dead from live at
**AUC 0.83–1.00** (`dead_overlap_check.py`), and a single threshold
misclassifies 0–2 of 20 (`dead_threshold_check.py`, best split 0.18–0.36):

| encoder | max_cos, dead | live | min_sep, dead | live | AUC |
|---|---|---|---|---|---|
| att16-s42 | 0.327 | 0.090 | 383 | 350 | 0.96 |
| att16-s43 | 0.362 | 0.067 | 371 | 354 | 0.89 |
| att32-s42 | 0.325 | 0.103 | 365 | 358 | 0.83 |
| att32-s43 | 0.386 | 0.077 | 371 | 354 | 1.00 |

**Real-space separation predicts nothing** — 383 against 350, and the wrong
sign. The killer sits ~370 cells away: far-field aliasing, hundreds of cells
outside any coding radius. It is one competitor and not an average, which is
§4.5's argmax-versus-mean split again.

> **Falsified: the killers are not on the module lattice.** The obvious story —
> aliases where a grid module realigns exactly — looks strong in the 20 probe
> pairs and dies at scale. Of 3,861 far pairs, 42% of the high-overlap ones
> have an exact module alignment against a **40% base rate**
> (`alias_lattice_check.py`). That is evidence against λ being the visible
> lever, contrary to `EXPERIMENTS_UNIQUE_RADIUS.md` §10.3 item 4.

### 10.4 `r_min` is priced for a different problem

Six constrained candidates carry the **same stored `r_min` of 12.0** while
their far-field alias rate spans seven-fold. And the campaign's last axis moved
the wrong way on purpose: `attract_lambda` 8 → 64 buys res90 12 → 18, exactly
what §4.4b's law rewards, and pays alias rate 0.0123 → 0.0610.

Inference gain is the same trade on a fixed checkpoint
(`gain_tradeoff_check.py`) — att16-s42, gain 1 → 1000: far-field alias rate
0.111 → 0.0094 (12×), res90 22 → 8.

**So `r_min` and continuous reach pull in opposite directions along both of the
axes that have been swept most recently.** `r_min` prices res90 and the alias
ceiling as comparable factors; navigation reads the ceiling almost alone,
because the Gram-Schmidt basis only needs one-cell neighbours.

### 10.5 A screen that costs seconds

If dead goals are goals with a competitor above ~0.25, the rate of *distant*
pairs above that line should predict reach. Against the arms where reach was
measured — nine encoder × setting combinations, two encoder families, gain 3.7
to 300, β 100 to 1e6 — **Spearman = −0.917, p = 0.0005**
(`nav_screen_validate.py`). The threshold came from §10.3, not from this fit.

Over the pool (`nav_screen_check.py`, seed 42, res90 as the local guard):

| encoder | `r_min` | far>.25 @100 | res90 | far>.25 @300 | res90 |
|---|---|---|---|---|---|
| *v35 (out of brief)* | — | *0.0036* | *8* | *0.0031* | *6* |
| *w21 arena-spread ×2 (out of brief)* | *5.0* | *0.0009* | *3* | *0.0009* | *3* |
| L5 `sm50_b4096` | 12.0 | 0.0043 | 6 | 0.0042 | 5 |
| **L6 `eps1_rate0.5`** | 12.0 | 0.0088 | 10 | **0.0057** | **8** |
| w54 `rep0.25` | 12.0 | 0.0106 | 12 | 0.0083 | 9 |
| w53 `att8` | 12.0 | 0.0123 | 12 | 0.0094 | 10 |
| **L7 `att16` (headline)** | 12.0 | **0.0170** | 14 | 0.0125 | 11 |
| w54 `att32` | 12.0 | 0.0269 | 16 | 0.0170 | 12 |
| w54 `att64` | 12.0 | 0.0610 | 18 | 0.0388 | 13 |

The ranking is unchanged at any threshold from 0.15 to 0.35.

### 10.6 What to do

1. **Select on the alias rate, not `r_min`**, with a res90 floor. Free, and it
   ranks a pool where `r_min` is constant.
2. **Best available legal encoder: Level 6 at inference gain 300**, with
   saturated β. Nominated off the screen at (0.0057, res90 8), the only
   constrained arm inside v35's box — **and confirmed in §10.8**: continuous
   reach 0.981 over four seeds, zero dead goals at K=3/5/10 on every seed.
   Unsaturated it already reaches 0.931, which beats level 7 *with* the full
   treatment.
3. **Reverse the last two waves.** `attract_lambda` down (8 beats 16 beats 32
   beats 64; 4/2/1 untested), `repel_weight` down (`rep0.25` already beats
   `att16`, and `EXPERIMENTS_UNIQUE_RADIUS.md` §10.2 flagged the axis).
4. **Re-open §5.6l — its verdict was priced in `r_min`.** The arena-spread
   diagnostic scores **0.0009**, the best measured anywhere, four times better
   than v35. It was dismissed because `r_min` stayed at 6, all of the gap being
   res90. It is out of brief (the spread term sees positions outside the
   patches), so treat it as the bracket — and if res90 6 is survivable, the job
   becomes reaching that ceiling legally.
5. **Settle the res90 floor.** Direction is fine at res90 8, 11 and 14 and
   collapses (10.0° → 19.2°) at 6 — four points from two encoders, and
   everything above is bounded by it. A gain sweep with the full probe at each
   step replaces it with a curve.
6. **Scoping the memory to the current environment is a fallback, not a
   lever.** At K=1 every encoder solves every environment, so it would work —
   but §10.2 shows the encoder route already gets there without it: the good
   arm is flat at 0.00 through K=20. K sets the exponent, the encoder sets the
   base, and only one of the two is free of cost. Worth keeping in mind if a
   configuration turns out to need loads far beyond 20, where §7's capacity
   bound of 74 starts to bind.

### 10.7 What this rests on

- The 0.25 line is 20 goals per encoder across four encoders; the *ranking* is
  threshold-insensitive but the number is soft.
- The res90 = 8 floor is four data points. §10.6 item 5 exists for this.
- **v35 is a target, not a candidate**: 20.4% coverage, 60 patches,
  `hidden_dim` 1024, 4.64M params, no cross-env constraint. It shows the box is
  reachable, not that it is reachable under the brief.
- One scaffold, one seed per screened arm, K ≤ 20. The dead-goal *joins* need
  goal positions, so they use the four worlds serialised into the archive
  (n = 20 per encoder); the load curves need only reach values and use all
  eight (n = 24).
- Raising inference gain changes every embedding: none of this is a config edit
  for a trained policy.

Raw: `$CLS_RESULTS/hopfield_probe/20260827/l6_production/` and
`l6_g300_sat/`.

Page: https://claude.ai/code/artifact/d65ad39b-eb99-4ae3-808c-52223bec874a

### 10.8 The confirmation run — Level 6 probed, and it is the best in the campaign

§10.6 item 2 nominated `w49_g100_knee/*_eps1_rate0.5` off the screen, with a
prediction on record before the run: continuous reach 0.92–0.95 saturated,
0.86–0.91 unsaturated. Four seeds, both arms, settings identical to every
archived arm.

| K=5, s=1 | \|err\| | acc45 | exact | basin | disc | **cont** | acc45 @ s=15 |
|---|---|---|---|---|---|---|---|
| **L6 production** (β=gain=100) | | | | | | | |
| s42 | 7.13° | 99.9% | 90.4% | 17.57 | 0.948 | 0.951 | 92.9% |
| s43 | 7.57° | 99.2% | 88.0% | 18.43 | 0.939 | 0.910 | 92.6% |
| s44 | 6.33° | 99.9% | 90.3% | 18.70 | 0.957 | 0.960 | 95.1% |
| s45 | 8.32° | 99.2% | 85.6% | 17.55 | 0.901 | 0.880 | 87.2% |
| *median* | | | | | | **0.931** | |
| **L6 gain 300 + β1e6** | | | | | | | |
| s42 | 9.09° | 100.0% | 97.4% | 20.38 | 1.000 | 0.986 | 100.0% |
| s43 | 9.20° | 99.8% | 90.2% | 18.23 | 0.997 | 0.976 | 99.8% |
| s44 | 8.59° | 100.0% | 93.1% | 18.90 | 1.000 | 0.964 | 100.0% |
| s45 | 9.15° | 99.4% | 93.6% | 19.25 | 0.974 | 0.987 | 99.3% |
| *median* | | | | | | **0.981** | |

Reference, same settings: att16 production 0.806, v35 production 0.867,
att16 g300+sat 0.890, v35 g100+sat 0.974.

**Both predictions were too low, in the same direction.** The screen ranked
correctly and the mapping from alias rate to reach was pessimistic, which is
what calibrating it on worse encoders would do.

Three things this settles.

**Level 6 unsaturated already beats every constrained arm ever measured** —
0.931 against att16's 0.806 at the same settings, and against att16's 0.890
*with* the full gain-300 + saturation treatment. It leads every other column
too: 6.3–8.3° against 11.8°, retrieval 85.6–90.4% against 57.8%, basin 17.6–18.7
against 11.0. So the campaign's step from level 6 to level 7 cost about 0.12 of
continuous reach, and `r_min` scored that step as an improvement (floor 7 → 9).

**And it does this in the linear regime.** `tanh` argument max **0.0044**, four
orders below the `u ≈ 1` transition, loop gain `β·s` = 0.098 — identical to
att16 production, the arm §0 calls a matched filter that degrades cues. What
level 6 has is condition (a) *baked in at training time*: it is trained at final
gain 100 (§6.10h). Since §6 established retrieval is a one-step jump rather
than a relaxation, a linear filter makes that jump fine when the patterns are
separable enough, and saturation is what stops the state drifting afterwards.

**Saturation still adds, and it is not redundant.** 0.931 → 0.981, and it flattens
the load curve outright: dead fraction at K=20 goes 0.29–0.50 → 0.00–0.12, and
**all four saturated seeds have zero dead goals at K=3, 5 and 10**, matching
`v35 g100+β1e6`. So both conditions earn their place — (a) from the encoder, (b)
from the loop gain.

#### The screen, out of sample

Adding the eight new arms to §10.5's nine:

| | ρ | p | n |
|---|---|---|---|
| original nine arms | −0.917 | 0.0005 | 9 |
| the eight L6 arms alone | −0.762 | 0.028 | 8 |
| **all seventeen** | **−0.853** | **0.00001** | 17 |

It held. The degradation from −0.917 is the expected cost of an out-of-sample
test rather than a fit.

Its rank prediction at the very top did **not** resolve, though: L6 g300+sat has
a *worse* alias rate than `v35 g100+sat` (0.0054–0.0062 against 0.0036) and
scores at least as well (0.981 against 0.974). v35 is one seed and 0.974 sits
inside L6's four-seed range of 0.964–0.987, so this is a tie rather than a
reversal — but the screen has no resolving power left at this level, and it
should be read as a filter, not a ranking, below about 0.006.

> **A §10.2 claim does not survive.** That section reported `p` tracking the
> alias rate at a ratio of 1.6–3.0 over seven arms. The eight new arms give
> 0.60, 1.09, 1.70 and 2.82 for the unsaturated seeds, and the saturated ones
> have `p = 0` exactly, so there is no ratio to take. `p` at K=10 is estimated
> from 24 samples and is quantised in steps of 0.005, which the original seven
> arms did not make obvious. The honest statement is that `p` is within a factor
> of about three of the alias rate, not that it tracks it.

#### What binds now

At the top the failure mode has moved off aliasing entirely. The saturated arms
post **discrete reach 1.000 with continuous 0.964–0.987**, and continuous reach
from cells adjacent to the goal is 0.995–0.998. So 1.4–3.6% of starts reach the
goal *cell* and never get within 0.5 of the goal *point*.

That is the sub-cell approach, and `ARRIVAL_RADIUS = 0.5` against a float
position is as much a property of the measurement as of the encoder. Anything
further on this arm should establish which before spending runs on it.

### 10.9 Selecting on the nav objective: `attract_lambda` **down**, and gain is spent

§10.8 confirmed the screen's nomination. This round asks what is left, and the
answer changes the recommendation: the best encoder is not Level 6 at all, and
it needs no inference override.

Everything below is at **β = gain** — no saturation anywhere.

#### The pool screen

`nav_screen_all.py` walks every arm in w39/w45–w54 at gains 100/300/1000. Two
things fall out immediately.

**The spread term does nearly all the far-field suppression.** `w48 rate0` —
the same config with the coding-rate term off — has an alias rate of **0.2059**
against 0.004–0.06 for everything else. A 25–50× effect, and the largest number
in the screen. That is §5.6j's mechanism confirmed from the other side.

**And `attract_lambda` is monotone the wrong way.** All eight arms share one
base config, so this is a clean axis:

| `attract_lambda` | 0.5 | 1 | 2 (L6) | 4 | 8 | 16 (L7) | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| alias @ g100 | 0.0059 | 0.0072 | 0.0082 | 0.0095 | 0.0123 | 0.0170 | 0.0269 | 0.0610 |
| res90 @ g100 | 7 | 9 | 10 | 11 | 12 | 14 | 16 | 18 |
| `r_min` | 6.0 | 10.0 | 12.0 | 12.0 | 12.0 | 12.0 | 12.0 | 12.0 |

w52 → w53 → w54 climbed from 2 to 64 because `r_min` rewards res90. On the
alias rate that is a 7× regression, and the untested direction was down.

> **A one-seed reading retracted.** `fwhm_ratio` 0.5 looked 15% better than
> Level 6 at seed 42. Over four seeds it is 0.0080 against 0.0082 — nothing,
> and Level 6's own seed range (0.0071–0.0088) is wider than the effect
> (`fwhm_seeds_check.py`). §6.9's "0.25 and 0.5 are equivalent" holds on this
> metric too. `steps3x` survives at gain 100 (0.0073 vs 0.0082) and vanishes at
> gain 300, where every arm converges near 0.0055.

#### The gain ladder: an interior optimum, not a floor

Four seeds of Level 6, β = gain (`run_ladder.sh`):

| L6 | res90 | alias | \|err\| | exact | **cont** | dead @ **K=1** |
|---|---|---|---|---|---|---|
| g100 | 10 | 0.0082 | 7.3° | 88% | 0.931 | 0.00 |
| **g300** | 8 | 0.0056 | 9.0° | 96% | **0.971** | 0.00 |
| g1000 | 6 | 0.0052 | 19.3° | 99% | 0.954 | 0.03 |
| g3000 | 5 | ~0.005 | 41.7° | **100%** | **0.608** | **0.34** |

Gain 3000 collapses to 0.608 while retrieval hits 100% and the basin maxes out:
the Hopfield finds the right goal every time and the readout cannot use it. The
`K=1` column is the clean diagnostic — no cross-talk is possible there, so those
failures are pure local chart.

**It is an optimum because the alias rate saturates.** res90 8 → 6 → 5 buys
0.0056 → 0.0052 → ~0.005 while `|err|` goes 9° → 19° → 42°. Past gain 300 the
trade stops being a trade.

> **The `res90 ≥ 8` guard of §10.6 was calibrated on the wrong quantity** —
> angular error, not reach — and it is also too high. The cross-encoder ladder
> has `att0.5` fine at res90 7 and `L5` at 0.977 with res90 6; only `rate3` at
> res90 3 dies (`|err|` 80.5°, reach 0.077, **dead fraction 1.00 at K=1**). The
> real floor is nearer 4–5. It lands on the right answer for the wrong reason,
> because the alias rate flattens at the same place.
>
> It also conflated two failures. An encoder *trained* into a short chart is
> fine (`L5`, res90 6, 0.977); an encoder *read through* a chart it was not
> trained for degrades much faster (L6 forced to res90 6 by gain, 0.954, and to
> res90 5, 0.608).

#### `att0.5` is the best encoder measured

Four seeds, own gain 100, β = 100 — no override of any kind
(`run_att_low.sh`):

| | \|err\| | acc45 | exact | basin | **cont** (4 seeds) |
|---|---|---|---|---|---|
| **`att0.5`** | 7.0–8.7° | 99.5–100% | 96.2–98.9% | 20.7–21.4 | **0.987** (0.984–0.993) |
| `att1` | 6.5–7.6° | 99.6–100% | 90.6–96.9% | 18.5–20.0 | 0.972 (0.963–0.993) |

`att0.5` has the tightest seed spread in the campaign. `att1`'s 0.993 was one
lucky seed — the fifth time a one-seed reading has not survived here.

| | cont | needs |
|---|---|---|
| **`att0.5` @ own gain 100** | **0.987** | nothing |
| L6 @ g300 + β1e6 | 0.981 | override + saturation |
| `att1` @ g100 | 0.972 | nothing |
| L6 @ g300 | 0.971 | override |
| L6 @ g100 | 0.931 | nothing |
| att16 (level 7) @ g100 | 0.806 | nothing |

Reach rises monotonically as `attract_lambda` falls — 0.806 / 0.931 / 0.972 /
0.987 at 16 / 2 / 1 / 0.5 — while `r_min` falls from 12.0 to 6.0.

**Attract and inference gain are substitutes.** Each arm's optimal gain falls as
its `attract_lambda` falls: `att2` and `att1` want 300, `att0.5` wants 100
(0.987 at g100 against 0.977 at g300). Both knobs shorten the chart, so an
encoder that spent the budget in training gains nothing by spending it again at
inference — which is why the best config is also the simplest.

**Saturation is unnecessary.** L6 at g300 scores 0.971 with β = gain against
0.981 saturated, a gap smaller than either arm's seed spread. What saturation
buys is step-invariance (s15 0.90–0.96 against 0.99–1.00), and the policy reads
s=1.

Raw: `$CLS_RESULTS/hopfield_probe/20260827/` — `l6_g300`, `l6_g1000`,
`l6_g3000`, `ladder_g100`, `attlow_g100`, `attlow_g300`, plus
`screen_all_seed42.txt` and `screen_fwhm_steps_4seed.txt`.

### 10.10 The scaffold is a variance source, and it was never varied

Every reach number in §1–§10.9 comes from `--seed 0`: one draw of 8 worlds × 20
envs. Encoder-training seed spread has been measured repeatedly here; the
probe's own world and goal sampling never had been. Two more draws, on the two
leading arms, four encoder seeds each (`run_probeseed.sh`):

| probe seed | `att0.5` @ g100 | L6 @ g300 | leader |
|---|---|---|---|
| 0 | 0.987 | 0.971 | att0.5 |
| 1 | 0.988 | 0.961 | att0.5 |
| **2** | **0.959** | **0.972** | **L6** |
| mean | 0.978 | 0.968 | att0.5, +0.009 |

> **§10.9 overstated this.** It said `att0.5`'s worst encoder seed beats L6's
> median "on either draw", which was true of draws 0 and 1 and was written as a
> property of the arms. Draw 2 breaks it: `att0.5`'s worst is 0.939 against L6's
> median 0.972. The defensible statement is that **`att0.5` leads on two of
> three scaffolds by ~0.02 and trails on the third by 0.013**, averaging +0.009
> — a real but small edge, inside scaffold variation.

**The variance itself is the result.** One arm swings **0.959–0.988** across
three world draws, which is larger than most of the differences §10.9 ranks. So:

- the large orderings are safe — level 7 at 0.806 against everything else near
  0.97 survives any draw;
- **gaps of 0.01–0.02 between the top arms are not resolvable on one scaffold**,
  and §10.8–§10.9 report them to three decimals as though they were.

Anything meant to separate the leaders needs all three draws. `n_worlds` 8 is
also low; raising it is the cheaper fix than repeating draws, and the probe
costs ~3 min per encoder either way.

Raw: `att0.5_ps1`, `att0.5_ps2`, `l6_g300_ps1`, `l6_g300_ps2`.

### 10.11 The mechanism: one variable, two metrics, two different optima

**This is the section to read if you read one.** §10.1–§10.10 are a sequence of
measurements; this is the account that makes them a single thing, and it is what
makes the result transferable to a config nobody has run.

#### Effective dimension is the variable

`why_attract_check.py`, one seed per arm, 4000 random positions. **Effective
dimension** is the participation ratio of the code covariance,
`(Σλ)² / Σλ²` — how many of the 1024 output directions the code actually
occupies:

| arm | eff dim | PR/D | far-cos sd | **1/√d_eff** | alias rate |
|---|---|---|---|---|---|
| `att16` (level 7) | 131 | 0.128 | 0.0825 | 0.0874 | 0.0194 |
| `att4` | 193 | 0.189 | 0.0675 | 0.0720 | 0.0109 |
| `att2` (level 6) | 211 | 0.206 | 0.0659 | 0.0688 | 0.0114 |
| `att1` | 235 | 0.230 | 0.0639 | 0.0652 | 0.0106 |
| **`att0.5`** | **297** | 0.290 | 0.0577 | 0.0580 | 0.0083 |
| `att0.25` | 345 | 0.337 | 0.0534 | 0.0538 | 0.0083 |
| `rate0` (no spread term) | **16** | 0.016 | 0.2259 | 0.2500 | 0.2115 |

**The far-field cosine spread is `1/√d_eff` to within a few percent over a 20×
range.** Distant pairs behave like random vectors in a `d_eff`-dimensional
space, which is what §10.3's "one competitor above 0.25" is a tail event of.

#### Why `att0.5` works

`attract_lambda` and the coding-rate term **compete for `d_eff`**. Attract asks
nearby positions to be similar, so the code varies slowly and collapses into few
directions; the rate term pushes it to spread over the sphere. Lowering attract
shifts the balance: 16 → 0.25 raises `d_eff` 131 → 345. Removing the rate term
altogether collapses the code to **16 of 1024 directions**, which is the whole
explanation of `rate0`'s 0.21 alias rate.

The chain to reach is then:

1. a goal dies when **one** co-stored competitor exceeds cos ≈ 0.25 (§10.3);
2. competitor cosines are ≈ `N(0, 1/√d_eff)`;
3. so raising `d_eff` pushes that tail down fast, and the dead-goal rate with it.

`att0.5` sits where `d_eff` is high enough that the tail is nearly empty and not
so high that the chart is too short to build a local basis from.

#### Why `r_min` is the wrong signal

§4.4b's law is `r_min ≈ res90 · √(ln(1/C)/ln(1/0.9))` — **a product of two
opposing functions of `d_eff`**. Raising `d_eff` shortens the chart (res90 down,
first factor down) and suppresses aliases (`C` down, second factor up). So
`r_min` has an **interior optimum in `d_eff`** and peaks around `att2`–`att16`,
while reach keeps improving to `att0.5`. One underlying variable, two metrics,
two different peaks — and w52–w54 climbed toward `r_min`'s peak for three waves.

And navigation does not need res90 the way `r_min` does. The readout is
`W_basis @ (recalled − current)` with the basis built from **±1-cell**
neighbours, so the chart only has to be valid at one cell. res90 7 leaves the
1-cell cosine at ~0.998. `r_min` maximises res90 because it asks a *decoding*
question — how far can a position move and stay identifiable; navigation asks a
*gradient* question and needs res90 only as a floor.

#### What this explains that was previously just observed

* **Every knob is the same trade** (§10.9) — attract, gain, `rate_lambda` and
  patch count all move `d_eff`, so all of them buy alias rate with res90.
* **Attract and inference gain are substitutes** (§10.9) — same variable, so a
  budget spent in training cannot be spent again at inference.
* **Combinations do not compound** (§10.10) — `a0.5_sm30` is worse than either
  ingredient because you cannot move one variable twice and then trim back.
* **res90 and the alias rate are rank-correlated across every arm measured** —
  they are two readouts of `d_eff`.

#### Open questions — three things this does *not* derive

1. **Why `attract_lambda` sets `d_eff` quantitatively.** The qualitative account
   is solid and the relation is monotone over six arms across a 64× range, but
   there is no derivation of `d_eff(attract_lambda)`, so the optimum has to be
   found empirically for any new geometry or loss.
2. **Where the 0.25 threshold comes from.** It should fall out of the argmax
   against the self-term at given K — the same condition (b) algebra as §7 — but
   it was fitted from 20 goals per encoder, not derived. Deriving it would give
   a *predicted* dead-goal rate from `d_eff` and K alone, which would remove the
   probe from the loop entirely.
3. **Why direction collapses below res90 ≈ 5.** At res90 5 the one-cell cosine
   is still ~0.996, so the Gram-Schmidt basis should be fine — and yet `|err|`
   reaches 42° (§10.9's gain-3000 row). Something about the basis conditioning,
   or about the `recalled − current` difference vector, degrades faster than the
   raw cosine implies. This is the floor that bounds the whole trade, and it is
   the least understood number in §10.

A fourth, from §10.9: an encoder **trained** into a short chart behaves better
than one **read through** a short chart (`L5` at res90 6 scores 0.977; level 6
forced to res90 6 by gain scores 0.954, and to res90 5 collapses to 0.608). If
`d_eff` were the whole story those would match, so inference gain and training
are not perfectly interchangeable and something beyond `d_eff` distinguishes
them.

### 10.12 Halving coverage to 5% costs almost nothing — and the §10.11 prediction fails

`w57_cov5`: six arms × four seeds at ~5% coverage, `exclude_cross_env_pairs`
throughout, β = gain. Two ways to halve the incumbent's 118 × 50 patches held
against each other, and the attract axis swept rather than transferred.

**Prediction recorded before the results.** §10.11 says attract and the
coding-rate term trade against one `d_eff` budget, and lower coverage gives the
spread term fewer distinct arena positions — so the attract optimum should move
**down** from 0.5. It does not.

#### Screen, at matched res90 7

| arm | gain | alias | `d_eff` |
|---|---|---|---|
| `att0.5` @ **10%** (reference) | 75 | **0.0059** | 286.9 |
| `sm35_a0.5` — 120 × 35 | 50 | **0.0084** | 229.6 |
| `half_a0.5` — 59 × 50 | 75 | 0.0088 | 210.2 |
| `half_a1` | 150 | 0.0090 | 200.8 |
| `half_rate1` | 20 | 0.0101 | 201.6 |
| `sm70_a0.5` — 30 × 70 | 75 | 0.0104 | 182.5 |
| **`half_a0.25`** | 10 | **0.0140** | 169.8 |

> **The prediction failed, with the sign backwards.** `a0.25` is the *worst*
> arm at 5% — 59% worse than `a0.5` — where at 10% it was level with it. The
> optimum stays at 0.5–1.0 (0.0088 against 0.0090, a tie).
>
> The `d_eff` column shows why, and it is consistent with §10.11's *mechanism*
> even though it breaks its *extrapolation*. To land at res90 7, `a0.25` has to
> be read at gain **10**, and low gain costs `d_eff` — 169.8, the lowest in the
> wave. Attract-down and coverage-down are **substitutes**, the way attract and
> gain are: lower coverage already shortens the chart, so there is *less*
> attract-lowering available, not more. Alias rate and `d_eff` stay tightly
> rank-ordered across all seven rows, so the variable is right and the direction
> was wrong.

#### Probe, two arms × four seeds × three scaffold draws

| arm | draw 0 | draw 1 | draw 2 | mean |
|---|---|---|---|---|
| `sm35_a0.5` (120 × 35, gain 50) | 0.972 | 0.984 | 0.931 | 0.962 |
| **`half_a0.5` (59 × 50, gain 75)** | **0.981** | **0.985** | **0.948** | **0.971** |
| `att0.5` @ 10% (gain 100) | 0.987 | 0.988 | 0.959 | 0.978 |

**Halving coverage costs 0.007 of continuous reach** — 0.978 → 0.971. For
comparison, `EXPERIMENTS_UNIQUE_RADIUS.md` §10.3 puts coverage as *the* largest
lever on `r_min`, worth a factor of 2.5. On this objective it is worth almost
nothing, which is the same divergence §10.11 explains: coverage moves `d_eff`
only weakly, and `r_min` prices res90 heavily.

> **The screen got this ordering backwards.** `sm35_a0.5` has the better alias
> rate (0.0084 vs 0.0088) and loses on **all three draws**. The gap is small on
> both axes, but the reach ordering is consistent 3/3 while the alias ordering
> is opposite — so the screen's discriminating power is spent well above the
> ~0.007 floor §10.9 identified. **Treat the screen as a filter down to a
> shortlist, then probe; do not rank inside a shortlist with it.**

#### The 5% answer

`sweeps/w57_cov5/001_half_a0.5_seed=43/encoder_final.pt`, read at **inference
gain 75** (not its trained gain of 100), β = gain. Continuous reach **0.987** on
draw 0, **0.977** as the mean over three draws and four seeds.

Config is `att0.5` with the patch set halved — 59 × 50 instead of 118 × 50 —
and nothing else changed. Per-seed at draw 0:

| seed | \|err\| | acc45 | exact | basin | disc | cont |
|---|---|---|---|---|---|---|
| 42 | 8.67° | 99.7% | 91.8% | 18.93 | 1.000 | 0.983 |
| **43** | 9.44° | 99.5% | 97.4% | 20.73 | 0.996 | **0.987** |
| 44 | 9.04° | 99.8% | 92.6% | 19.77 | 0.939 | 0.964 |
| 45 | 9.90° | 99.1% | 91.2% | 19.15 | 0.980 | 0.979 |

Dead-goal fraction 0.00 through K=10, **0.21 at K=20** — against `att0.5` @ 10%'s
0.04–0.12. The load curve is where halving coverage actually shows up, not in
the K=5 headline.

Also worth recording: **more, smaller patches wins at 5% on the screen but not
on reach.** `sm35` (120 × 35) beats `half` (59 × 50) beats `sm70_lo` (30 × 70)
on the alias rate, so §6.6's "choose the size that gives ~30 environments" rule
is wrong here — but the probe reverses the top two, so the honest statement is
only that the ~30-env geometry is clearly worst.

Raw: `$CLS_RESULTS/hopfield_probe/20260827/w57_{half,sm35}_a0.5_ps{0,1,2}/`.

### 10.13 2.5% coverage — viable, and the attract optimum moves the other way

**Predictions were recorded before any result was read.** §10.11 says attract
and the coding-rate term compete for `d_eff`, and halving coverage twice shrinks
the distinct arena positions the env-blind spread term samples — so `d_eff`
should fall and attract should have to give up more of its share. Predicted:
optimum moves **down** from 0.5; alias 0.010–0.020; continuous reach 0.90–0.96.

#### The attract optimum moves UP

Screened at matched res90 7, gain swept per arm, four seeds
(`screen_w58_check.py`):

| `attract_lambda` @ 2.5% | 0.25 | 0.5 | **1.0** | 2.0 | 4.0 |
|---|---|---|---|---|---|
| alias rate | 0.0253 | 0.0212 | **0.0178** | 0.0179 | 0.0195 |
| gain for res90 7 | 3 | 20 | 100 | 200 | 300 |

> **Prediction 1 is falsified, with the sign reversed.** The optimum is 1.0–2.0
> at 2.5% coverage against 0.5 at 10% — it moves **up** as coverage falls, not
> down. w58 swept 0.25/0.5/1.0 and returned a boundary value, so `w59` was added
> to continue the axis; 4.0 turns over, which makes 1.0–2.0 a genuine interior
> optimum rather than an edge.

The mechanism reading that survives: attract **holds the near field up**
(§6.11), and at 2.5% coverage there is less local structure for it to hold, so
the code needs more of it. The low-attract arms are already maximally
stretched — `q_a0.25` reaches res90 7 only at gain 3, the lowest swept, and its
`res90 max` is exactly 7.0. So §10.11's "attract and the rate term compete for
`d_eff`" is right about the competition and wrong about which way coverage
pushes the balance.

Geometry sub-prediction also failed: `sm25_a0.5` (count held, 118 × 25 cells)
was predicted to beat `q_a0.5` (size held, 30 × 50) on the alias rate and is
worse — 0.0253 against 0.0212. The 25-cell patches are under §6.3's ~50-cell
floor and `res90 max` is 7.0, the shortest of the size-held arms, so the res90
half of that prediction held while the alias half did not.

#### The winner

`w58_cov2.5/*_q_a1` at **inference gain 100** (its own trained gain), β = gain.
Three scaffold draws × four seeds:

| probe seed | s42 | s43 | s44 | s45 | median |
|---|---|---|---|---|---|
| 0 | 0.991 | 0.994 | 0.928 | 0.986 | **0.989** |
| 1 | 0.991 | 0.931 | 0.964 | 0.969 | 0.967 |
| 2 | 0.893 | 0.933 | 0.943 | 0.945 | 0.938 |
| **mean** | 0.958 | 0.953 | 0.945 | **0.967** | **0.965** |

Runners-up on scaffold 0: `q_a0.5` at gain 20 → 0.945; `q_a2` at gain 200 →
0.962. The a1/a2 tie on the screen (0.0178 vs 0.0179) breaks to a1 by 0.027 on
reach, just outside the ~0.02 scaffold noise, so a1 stands — and it stands on
three draws against a2's one.

#### Is 2.5% viable? Yes at K=5, and much less so under load

| | 10% (`att0.5` @ g75–100) | 2.5% (`q_a1` @ g100) |
|---|---|---|
| alias rate at res90 7 | 0.0059 | **0.0178** (3×) |
| continuous reach, mean of 3 draws | 0.977 | **0.965** |
| dead goals at K=20 | 0.04–0.12 | **0.12–0.42** |

**Quartering the coverage costs 0.012 of mean continuous reach** — at or inside
the scaffold noise §10.10 measured — while tripling the alias rate. Prediction 4
(0.90–0.96) was too pessimistic; prediction 2 (alias 0.010–0.020) held.

The cost is real but it is **in load tolerance, not in K=5 reach**. That is the
sharpest thing this section adds to §10.9: the alias rate stopped discriminating
between the 10% leaders because they were all under ~0.007, but at 0.0178 it
still predicts something — just not the headline number. It predicts how fast
the encoder dies as goals accumulate.

So: at K≤10, 2.5% coverage is a near-free saving. At K=20 it loses a third of
its environments where 10% loses a tenth. Which of those matters depends on how
many goals the memory actually holds.

Raw: `w58_q_a1_g100_ps{0,1,2}`, `w58_q_a0.5_g20_ps0`, `w58_q_a2_g200_ps0`.
Screen: `screen_w58_check.py`.

### 10.14 The coverage ladder — reach is flat, capacity is not

§10.12 and §10.13 found the best encoder at 5% and 2.5% coverage. This runs all
three through **one** probe invocation so they are directly comparable, each at
its own optimal gain, β = gain, no saturation.

| | checkpoint | gain | `attract_lambda` | patches |
|---|---|---|---|---|
| 10% | `w52_attract_fwhm/001_att0.5_seed=43` | 100 | 0.5 | 118 × 50 |
| 5% | `w57_cov5/001_half_a0.5_seed=43` | **75** | 0.5 | 59 × 50 |
| 2.5% | `w58_cov2.5/011_q_a1_seed=45` | 100 | **1.0** | 30 × 50 |

| K=5, s=1 | \|err\| | acc45 | exact | basin | disc | **cont** | s15 | dead @ K=20 |
|---|---|---|---|---|---|---|---|---|
| 10% | 7.78° | 99.5% | 98.2% | 21.12 | 0.997 | **0.987** | 96.5% | 0.08 |
| 5% | 9.44° | 99.5% | 97.4% | 20.73 | 0.996 | **0.987** | 98.2% | 0.25 |
| 2.5% | 10.98° | 98.8% | 88.2% | 17.62 | 0.980 | **0.986** | 88.6% | 0.42 |

All three are 0.00 dead at K = 1, 3, 5 and 10.

**Reach is flat across a 4× range of coverage. Capacity is not.** Retrieval
falls 98.2 → 88.2%, the basin shrinks 21.1 → 17.6 cells, and the dead-goal
fraction at K=20 goes 0.08 → 0.25 → 0.42. Coverage buys the ability to hold
*many* goals apart, not the ability to point at one — which is what §10.11
predicts, since capacity is the tail of the competitor-overlap distribution and
K=5 reach is not.

`EXPERIMENTS_UNIQUE_RADIUS.md` §10.3 calls coverage "the largest single lever,
a factor of 2.5, and nothing else in the campaign comes close." That is true of
`r_min`. On continuous reach at K≤10 it is worth approximately nothing.

> **Read the three-draw means, not this table.** These are one scaffold draw,
> and §10.10 measured a fixed arm swinging up to 0.03 across draws. Over three
> draws × four seeds the arms are **0.978 / 0.977 / 0.965**; the 0.001 spread
> above is the draw being kind.

**Both agents' §10.11 extrapolation failed, sign reversed, independently.** The
prediction was that the attract optimum moves *down* as coverage falls. It moves
**up** — 0.5 at 10%, 0.5 at 5%, 1.0–2.0 at 2.5%. The mechanism survives: alias
rate and `d_eff` stay tightly rank-ordered at every coverage. What was wrong is
the direction of the interaction. Attract holds the near field *up*, and at low
coverage there is less local structure to hold, so the code needs more of it —
and the low-attract arms are already maximally stretched, since reaching res90 7
from `a0.25` at 2.5% requires gain 3, its lowest.

**And the screen is a filter, not a ranking.** Twice, independently: `sm35_a0.5`
at 5% and `sm25_a0.5` at 2.5% both had the better alias rate and lost the probe
— the 5% case on all three draws. Use it to cut a wave to a shortlist; use the
probe to order the shortlist.

Report page, all three behind an encoder selector, full Tests A–D:
https://claude.ai/code/artifact/d7a250c1-5044-4854-b453-61881bd518e7

Raw: `$CLS_RESULTS/hopfield_probe/20260827/probe_three/`. Reproduce with
`analysis/hopfield_probe/run_three.sh`.

### 10.15 The coverage floor is between 2.5% and 1.25%

§10.14 left reach flat over a 4× range of coverage while capacity drained, and
asked where it runs out. One more rung answers it, and lets the corrected
mechanism predict instead of explain.

#### The attract trend, predicted and confirmed

§10.14 recorded a prediction before `w60` ran: the optimum at 1.25% is **2.0**,
continuing 0.5 / 0.5 / 1.0. Among the size-held arms (15 × 50), it is:

| 1.25%, 15 × 50 | gain | alias | `d_eff` |
|---|---|---|---|
| `x_a1` | 20 | 0.0477 | 54.4 |
| **`x_a2`** | 200 | **0.0353** | 61.3 |
| `x_a4` | 300 | 0.0408 | 56.4 |

Interior, bracketed both sides. After §10.11's extrapolation failed twice with
the sign reversed (§10.12, §10.13), the corrected account — attract holds the
near field *up*, and low coverage leaves less structure to hold — now has a
prediction it made and got right.

#### `d_eff` across the whole ladder

| coverage | best arm | gain | alias | `d_eff` |
|---|---|---|---|---|
| 10% | `att0.5` | 75 | 0.0059 | 120.4 |
| 5% | `half_a0.5` | 75 | 0.0088 | 102.9 |
| 2.5% | `q_a2` | 200 | 0.0179 | 84.8 |
| 1.25% | `sm35x_a2` | 100 | 0.0286 | 64.9 |

Coverage sets `d_eff`, `d_eff` sets the alias rate (§10.11: far-field cosine
spread is `1/√d_eff`), the alias rate sets dead goals. The chain holds over an
8× range of coverage.

#### And reach finally breaks

Both leading arms, four seeds, three scaffold draws:

| | draw 0 | draw 1 | draw 2 | **mean** |
|---|---|---|---|---|
| `x_a2` (15 × 50) | 0.900 | 0.859 | 0.831 | **0.863** |
| `sm35x_a2` (30 × 35) | 0.888 | 0.862 | 0.861 | **0.870** |

Against the ladder above: **0.978 / 0.977 / 0.965 / ~0.87**. Flat to 2.5%, then
a 0.10 drop — three times the scaffold spread, on every draw.

The diagnostics say why, and it is not a new failure mode:

| | 2.5% | 1.25% |
|---|---|---|
| dead @ K=5 | **0.00** | 0.04–0.25 |
| dead @ K=10 | **0.00** | 0.08–0.29 |
| `exact` | 88.2% | 67–83% |
| basin | 17.62 | 12.0–15.9 |

Every coverage down to 2.5% had **exactly zero** dead goals at K=5 and K=10, and
paid only at K=20. At 1.25% the alias rate passes ~0.02 and dead goals cross
into the K=5 operating point. That is the same mechanism running out, not
something else appearing.

**So the usable floor is 2.5%.** At K≤10, 2.5% coverage costs 0.013 of reach
against 10% — four times less data for nothing. At 1.25% it costs 0.10 and the
failures reach the operating load.

> **Third strike for the screen as a ranking.** `sm35x_a2` has the better alias
> rate (0.0286 against 0.0353) and the two arms tie on reach, with `sm35x_a2`
> ahead only on the mean. §10.12 and §10.13 each had a case where the better
> alias rate lost the probe outright. Three independent instances: **the alias
> rate filters a wave to a shortlist and does not order one.**

Geometry crossover, worth noting for anything below 2.5%: at 2.5% the size-held
mix won (30 × 50 over 118 × 25); at 1.25% the count-held mix has the better
alias rate (30 × 35 over 15 × 50) and the better mean. `EXPERIMENTS_UNIQUE_RADIUS.md`
§6.6's "choose the size that gives ~30 environments" rule was derived for
`r_min`, and it looks right at the bottom of this ladder specifically.

Raw: `$CLS_RESULTS/hopfield_probe/20260827/w60_ps{0,1,2}/`. Reproduce with
`analysis/hopfield_probe/run_w60probe.sh`; screen with `screen_w60_check.py`.

### 10.16 Saturating the 10% winner — it becomes a real attractor, and reach does not care

Every encoder in §10.8–§10.15 runs at `β = gain`, tanh argument ~0.003, which
the report labels `regime: linear`. So none of them is an attractor network: the
stored pattern is a saddle and the only stable fixed point of
`x ← normalize(Wx)` is `W`'s leading eigenvector, which is not a memory. They
land on the goal in one step and drift off it.

`att0.5` saturated, four seeds (`run_sat.sh`). `cos_bin` first, because §7's
condition (a) is an encoder property and β cannot buy it:

| gain | `att0.5` cos_bin | res90 | v35 cos_bin | res90 |
|---|---|---|---|---|
| **100** | **0.9546** | **7** | 0.9682 | 8 |
| 300 | 0.9841 | 5 | 0.9891 | 6 |
| 1000 | 0.9952 | 4 | 0.9967 | 5 |

`att0.5` at its own gain is already at the corner condition — near where v35 sat
when its saturated arm worked — so β can be saturated **without touching gain**,
which matters because gain 300 would drop res90 7 → 5, past the reach optimum.

#### It is an attractor now

> **This section stands. §10.20 sharpens one word.** The fixed point is real and
> the state holds it — `cos(recall¹⁵(z), recall(z))` is 0.998 saturated against
> 0.813 (min 0.46) unsaturated. What it is *not* is the stored pattern: the
> saturated map's image is always a hypercube corner, so a continuous pattern
> cannot be its own fixed point, and `cos_self` is capped at `cos_bin` = 0.956
> by construction. The attractor is a corner 0.958 from the memory.

Mean distance of the recalled state from the goal cell, K=5:

| | tanh arg | s1 | s2 | s3 | s5 | s8 | s15 |
|---|---|---|---|---|---|---|---|
| β = gain = 100 | 0.004 | 0.00 | 0.00 | 0.00 | 1.21 | 1.41 | 1.41 |
| **g100 + β1e6** | **36.8** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |

#### And reach does not move

| `att0.5`, K=5 | \|err\| | acc45 | exact | basin | **cont** | **s15** | dead @ K=20 |
|---|---|---|---|---|---|---|---|
| β = gain = 100 | 7.78° | 99.5% | 98.2% | 21.12 | **0.987** | 96.5% | 0.04–0.12 |
| g100 + β1e6 | 7.4–8.9° | 99.7–100% | 90.5–98.5% | 19.2–21.1 | 0.973 | **99.8–100%** | **0.00–0.08** |
| g300 + β1e6 | 14.0–16.7° | 97.2–98.9% | **99.3–99.9%** | **21.3–21.6** | 0.984 | 97.3–98.9% | **0.00** |

Saturation buys **step-invariance** (s15 96.5% → ~100%), the fixed point, and
**load tolerance** (dead goals at K=20 halve, and reach zero at gain 300). It
does not buy reach: 0.973 and 0.984 against 0.987, all inside the ±0.03 scaffold
spread of §10.10.

That is §10.15 restated from the other side. The residual failures are walkers
parked outside the 0.5-cell arrival radius, not field failures, so fixing the
*dynamics* cannot reach them. **Saturation is the right tool for a memory that
has to survive iteration, and irrelevant to a readout consumed at `s=1`.**

Four predictions were recorded in `run_sat.sh` before the run and all four held:
the drift stops, s15 goes to ~100%, reach is unchanged, and `exact_hit` holds
rather than collapsing.

> **Why `exact_hit` held this time.** §2 saturated β on the old encoders and
> retrieval fell 74% → 44%, because recall binarises while the cell bank stays
> continuous, so a binarised state is far from *every* cell. `att0.5` at cos_bin
> 0.955 is most of the way to a corner already, so the gap is small — and at
> gain 300 (cos_bin 0.984) retrieval reaches **99.3–99.9%**, the highest in the
> campaign. §7's gap 2 is therefore not a flaw in the dynamics but a statement
> about how close the encoder's patterns sit to corners.

The g300 arm also separates the two effects that move together: it is more
corner-like *and* shorter-charted, and it trades 7° of angular error for near
perfect retrieval and zero dead goals at K=20 — the same trade §10.9's gain
ladder found, with saturation not changing its shape.

Raw: `$CLS_RESULTS/hopfield_probe/20260827/att0.5_g{100,300}_sat/`.

### 10.17 0.75% — the falloff is a knee, and the failure mode changes

§10.15 put the floor between 2.5% and 1.25% and left open whether 1.25% was the
edge of a shelf or a point on a decline. One rung lower answers it.

Screen (gain per arm to land res90 7, four seeds):

| 0.75% arm | geometry | gain | alias | `d_eff` |
|---|---|---|---|---|
| **`y35_a2`** | 18 × 35 | 100 | **0.0541** | 46.7 |
| `y50_a2` | 9 × 50 | 200 | 0.0600 | 45.7 |
| `y27_a4` | 30 × 27 | 50 | 0.0649 | 42.8 |
| `y27_a2` | 30 × 27 | **3** | 0.0705 | 44.2 |

Probe, both leaders, four seeds, three draws:

| median reach | draw 0 | draw 1 | draw 2 | **mean** |
|---|---|---|---|---|
| `y35_a2` | 0.700 | 0.738 | 0.670 | 0.703 |
| **`y50_a2`** | 0.723 | 0.727 | 0.731 | **0.727** |

#### The ladder, complete

| coverage | best arm | reach | Δ | alias | `d_eff` | dead @ **K=1** |
|---|---|---|---|---|---|---|
| 10% | `att0.5` | 0.978 | — | 0.0059 | 120 | 0.00 |
| 5% | `half_a0.5` | 0.977 | −0.001 | 0.0088 | 103 | 0.00 |
| 2.5% | `q_a1` | 0.965 | −0.012 | 0.0179 | 85 | 0.00 |
| 1.25% | `sm35x_a2` | 0.870 | −0.095 | 0.0286 | 65 | 0.00 |
| **0.75%** | `y50_a2` | **0.727** | **−0.143** | 0.0541 | 47 | **0.00–0.25** |

**A steepening knee, not a cliff.** Each halving costs more than the last, so
1.25% was partway down rather than at an edge.

**And the failure mode changes at the bottom.** Dead goals at **K=1** are
nonzero for the first time. With one stored pattern no cross-talk is possible,
so those are local-chart failures, not interference — every rung down to 1.25%
was exactly 0.00 there. Below ~1% the binding constraint stops being the alias
rate and becomes whether the code can support a direction readout at all, which
is `d_eff` 47 running out of directions to be locally informative in.

> **§10.15's geometry reading, corrected.** It called 1.25% a count-vs-size
> crossover. Across three rungs the winners are 30×50, 30×**35**, 9×**50**, and
> 27- and 25-cell patches lose wherever they appear — so it is **a patch-size
> floor near 35 cells**, not a count preference. `EXPERIMENTS_UNIQUE_RADIUS.md`
> §6.3 was right that a floor exists and put it at ~50; on this objective it is
> nearer 35.

> **And a one-draw over-call, corrected.** Draw 0 had `y50` ahead of `y35` and
> was reported here as the screen losing a fourth time. Draw 1 reversed it. The
> two arms differ by 0.006 in alias rate and the probe cannot separate them —
> neither a win nor a loss for the screen. Sixth time in this campaign a
> single-draw reading has not survived.

Attract continues up: within the 27-cell geometry `a4` beats `a2`, so the ladder
is 0.5 / 0.5 / 1.0 / 2.0 / 4.0. **Untested:** the winning 35- and 50-cell
geometries were only run at `a2`, so their own optima may be higher.

Report page, five rungs behind the encoder selector:
https://claude.ai/code/artifact/d7a250c1-5044-4854-b453-61881bd518e7

Raw: `$CLS_RESULTS/hopfield_probe/20260827/w61_ps{0,1,2}/` and `probe_five/`.

### 10.18 The basin, corrected twice

**Every basin figure in §10.8–§10.17 is wrong.** Two problems, one of design and
one a bug, and the seed selection on top of both. The reach, direction and flow
numbers in those sections are unaffected.

#### 1. It was measured over the evaluation environment

`run_test_a` drew its cues from `local_cells(env_size)`, so no cue sat more than
~27 cells from the goal at `env_size` 20 and `r_exact_95` could not report a
larger number however large the true basin was. The basin is a property of the
encoder and the stored memory; the arena has nothing to do with it.

`basin_probe` now measures it over **every cell in a scaffold disc** of radius
`cfg.basin_radius` (64) around the goal, as both cue and retrieval bank, with no
env involved. It is the default — `r_exact_all` / `r_exact_95` come from it, and
the env-bounded values are kept as `*_envcues` so archived runs stay visibly
distinct rather than silently reinterpreted. The reported basin is
**`r_exact_all`**: the radius within which *every* cue retrieves the goal, a
guarantee rather than a 95% rate.

#### 2. The first fix had a duplicate-bank bug

The bank was `concat([disc_cells, mem.Z])`, which put the test env's goal in
**twice** — once as the disc centre, once as its stored copy. Same position, so
the same vector to within float noise, and `argmax` returns whichever copy
carries the marginally larger dot product. When the duplicate won, a correct
retrieval scored as a miss.

It corrupted **6 of 16 values per encoder**, not just the 2 that surfaced as
`-1`: the duplicate could also win at radius 2 or 3 and truncate the basin
there, which reads as a small number rather than a sentinel. Intermittent,
absent from the env-bounded path (no duplicates there), and not reproducible by
recalling the goal cue alone — different batching, different rounding. Fixed by
taking only the **other** K−1 stored goals.

#### 3. And one seed is not the arm

The ladder page probes the **reach-winning** seed per rung. Reach has a tight
seed spread; the basin's is wider than the entire ladder-wide effect, and it
prefers different seeds. Median `r_exact_all` over 16 (world, env) pairs, all
four training seeds:

| coverage | s42 | s43 | s44 | s45 | **median** | page used | self-fail |
|---|---|---|---|---|---|---|---|
| 10% `att0.5` | 25.5 | **30.5** | 28.0 | 26.0 | **27.0** | s43 (best) | 0.00 |
| 5% `half_a0.5` | 23.0 | **23.0** | 23.0 | 19.5 | **23.0** | s43 | 0.00 |
| 2.5% `q_a1` | 23.0 | 19.0 | 19.5 | **12.0** | **19.2** | s45 (worst) | 0.06 |
| 1.25% `sm35x_a2` | 10.0 | 13.0 | **18.5** | 8.0 | **11.5** | s44 (best) | 0.22 |
| 0.75% `y50_a2` | 12.5 | 6.0 | **14.5** | 16.0 | **13.5** | s44 | 0.12 |

2.5% drew its worst seed and 1.25% its best, which between them **inverted the
ladder** and made 2.5% look worse than 1.25%. 10% was flattered the same way,
30.5 against 27.0. `q_a2` at 2.5% agrees independently at 19.5, so the
correction is not an artefact of which arm is taken.

#### On the page

The report now groups by encoder **type**: the selector offers one entry per
rung, the tiles show the **best of four seeds** for each, and the coverage
charts plot **every seed** rather than one. Per-encoder bodies still render a
single seed — 20 sets of Tests A–D would exceed the artifact size limit — so
each header names which seed its figures are. The representative is the
**median** by continuous reach, deliberately not the best.

The 20 results were produced by recomputing `basin_probe` alone over the
existing per-seed runs (`splice_basin.py`, `splice_one.sh` as a 20-task array)
rather than re-probing everything: reach, direction and flow were already
correct at four seeds per rung, and only the basin column was broken.

| coverage | basin best | median | worst | self-fail | reach best |
|---|---|---|---|---|---|
| 10% | 30.5 | **27.0** | 25.5 | 0.00 | 0.993 |
| 5% | 23.0 | **23.0** | 19.5 | 0.00 | 0.987 |
| 2.5% | 23.0 | **19.2** | 12.0 | 0.06 | 0.994 |
| 1.25% | 18.5 | **11.5** | 8.0 | 0.06 | 0.938 |
| 0.75% | 16.0 | **13.5** | 6.0 | 0.00 | 0.836 |

#### The corrected reading

**27.0 / 23.0 / 19.2 / 11.5 / 13.5.** A clean decline with coverage down to
1.25%; the bottom two rungs are **not separable** (11.5 against 13.5, per-seed
ranges 8–18.5 and 6–16).

Self-retrieval failure runs the other way — 0.00 / 0.00 / 0.06 / 0.22 / 0.12.
At low coverage the goal increasingly fails to retrieve *itself*, which is a
different failure from a shrinking basin and is now reported as its own metric
rather than folded in as a `-1`.

> **What this does not disturb.** Reach is a rate over whatever starts exist and
> was never censored, so §10.8–§10.17's reach numbers, the coverage floor at
> 2.5%, the `d_eff` account in §10.11 and the attract ladder all stand. What
> falls is §10.14's claim that basin declines smoothly across the whole ladder,
> and every specific basin value quoted before this section.

Raw: `$CLS_RESULTS/hopfield_probe/20260827/probe_basin2/`. Checks:
`basin_seeds_check.py` (across seeds), `basin_confound_check.py` (patch size),
`env_basin_check.py` (the superseded env-size comparison).
Page: https://claude.ai/code/artifact/d7a250c1-5044-4854-b453-61881bd518e7

### 10.19 Saturating the recall shrinks the basin

> **Read §10.20 with this.** Everything below is confirmed, and the apparent
> paradox in the first sentence is a conflation rather than a result. β = 1e6
> *does* make the network an attractor — the state holds its landing point to
> cosine 0.998 over 15 steps. It does not make the **stored pattern** the fixed
> point, which is structurally impossible under a map whose image is a
> hypercube corner. The basin probe asks whether the state is nearest the goal
> *cell*, not whether it stays put, so the two can and do move in opposite
> directions. §10.20 has the arm that settles the mechanism.

§10.16 showed saturation turns the 10% encoder into a genuine attractor: the
recalled state lands on the goal and holds `goal_dist` 0.00 to s=15, where the
linear arm drifts to 1.41. So the basin — the radius from which a cue still
retrieves the goal exactly — was predicted to grow. **It shrinks.**

Same encoders, same encoder gains (100 / 75 / 100 / 100 / 200); only the recall
loop gain changes. Median `r_exact_all` over four seeds:

| coverage | β = gain | β = 1e6 | change | self-fail, gain → 1e6 |
|---|---|---|---|---|
| 10% | 27.0 | 24.5 | −2.5 | 0.00 → 0.03 |
| 5% | 23.0 | **13.0** | −10.0 | 0.00 → 0.09 |
| 2.5% | 19.2 | 12.2 | −7.0 | 0.06 → 0.12 |
| 1.25% | 11.5 | **4.0** | −7.5 | 0.22 → **0.44** |
| 0.75% | 13.5 | 10.2 | −3.3 | 0.12 → 0.22 |

Self-retrieval failure roughly doubles at every rung: saturated, the goal cue
itself increasingly fails to come back as the goal.

**Why — §7's gap 2, biting exactly where it was predicted to.** The saturated
update is `x ← sign(Wx)/√D`, so the recalled state is a hypercube corner, while
the retrieval bank is **continuous cell encodings**. A binarised state is far
from *every* cell, and the argmax then turns on which continuous cell happens to
sit nearest a corner rather than on which one the memory is. §2 measured the
same thing on the old encoders (retrieval 74% → 44% under β alone); §10.16 found
it did *not* bite for `att0.5`, because its `cos_bin` is 0.955 and the bank was
400 env cells plus aliases. Against **12,853 dense cells**, including every near
neighbour of the goal, 0.955 is no longer close enough.

So "basin" splits into two things that saturation moves in opposite directions:

* **as dynamics** — does the state stay put once it arrives? Saturation makes it
  perfect (§10.16, `goal_dist` 0.00 forever).
* **as a readout** — is the state nearest the goal cell among all cells?
  Saturation makes it worse, because binarising costs more against a dense
  continuous bank than the fixed point buys.

The `s=1` navigation readout wants the second. That is consistent with §10.16's
other finding — saturation buys step-invariance and load tolerance and **not**
reach — and it sharpens it: the fixed point is real, and it is not what the
readout is asking for.

Raw: `$CLS_RESULTS/hopfield_probe/20260827/probe_spliced_b1e06/`. Run with
`splice_sat.sh`; the page carries it as its Saturated section.

### 10.20 The attractor is a corner near the memory, and making it *be* the memory costs the gradient

§10.19 asked how self-retrieval could get *worse* once the network is a real
attractor. The attractor is real — that part of §10.16 stands, and an earlier
draft of this section wrongly denied it. `att0.5-s43`, K=5, 40 stored patterns,
one step versus fifteen:

| | β = gain = 100 | β = 1e6 |
|---|---|---|
| `cos(recall(z), z)` | 0.9966 | 0.9582 |
| `cos(recall(x), x)`, `x = recall(z)` | 0.9971 | **0.9989** |
| `cos(recall¹⁵(z), x)` | **0.813** (min 0.46) | **0.9981** (min 0.989) |

Saturated, the state lands in one step and holds it to s=15. Unsaturated it
drifts most of the way off. That is an attractor by any useful definition.

What it is **not** is the stored pattern. `cos_self = 0.958`, and that is
structural, not a shortfall: the saturated update is `x ← normalize(sign(Wx))`,
whose image is always a hypercube corner, so a *continuous* pattern can never be
its own fixed point and `cos_self` is capped at `cos_bin` = 0.956 by
construction. Reading that ceiling as "no attractor" is the error the earlier
draft made — `fixed_point_probe` answers "is the **pattern** a fixed point", not
"is there one".

So there is no paradox in §10.19 to resolve, only a conflation. **The basin
probe asks a different question than the trajectory probe.** Not "does the state
stay put" but "is it nearest the goal cell among 12,853 *continuous* cells".
The attractor sits 0.958 from the goal cell and about 0.955 from that cell's
near neighbours, so a **0.003 margin** decides the argmax. The dynamics get
better and the readout gets worse, at the same time, for the same reason.

Making the pattern itself the fixed point is §7's condition (a), and β cannot
buy it — the *encoder* gain can. At gain 1e6 the output is `tanh(1e6·z)` =
`sign(z)` to float precision, so the pattern **is** a corner. Full suite,
`att0.5`, four training seeds per arm, K=5, s=1 (`run_sat10.sh`):

| arm | `cos_self` | basin p50 | self-fail | exact | acc45 | \|err\| | `qnorm` | cont reach |
|---|---|---|---|---|---|---|---|---|
| β = gain = 100 | 0.9968 | 27.0 | 0/64 | 0.979 | 0.997 | 7.8° | 0.328 | **0.987** |
| β = 1e6, gain 100 | 0.9570 | 24.5 | 2/64 | 0.945 | 0.998 | 8.0° | 0.332 | 0.973 |
| **β = 1e6, gain 1e6** | **1.0000** | **28.2** | **0/64** | **0.999** | 0.392 | 66.4° | 0.184 | 0.103 |

**The basin does not shrink under saturation. It shrinks under half of it.**
With both saturated the fixed point is exact, the goal cue never fails in 64
(world, env) pairs, `exact_hit` is 0.999 — the highest in the campaign — and the
basin is 28.2 against 27.0 unsaturated. §10.19's *mechanism* is therefore
confirmed rather than merely surviving: what costs the argmax is the mismatch
between a binarised state and a continuous bank, and binarising the bank too
removes it — the one manipulation that closes the gap is the one that fixes the
basin. Nothing in §10.19 has to be withdrawn except its rhetorical framing: the
lost basin is not the price of an attractor, it is the price of reading a
binarised attractor against a continuous menu.

**And the attractor costs the entire direction field.** acc45 falls 0.997 →
0.392 against 0.25 chance, |err| 7.8° → 66.4°, `qnorm` halves, and continuous
reach goes 0.987 → 0.103. That is not a memory failure — retrieval is *better*
than any other arm.

#### Why: the code stops moving in a straight line

`q` is `basis @ (z_goal − z_here)` with `basis = gram_schmidt(d_fwd, d_rgt)` and
`d_fwd = z(x, y+1) − z(x, y)` (§6.2). Two things can break: the **frame**, or
the **displacement**. Measured at 1500 scaffold positions
(`binary_geometry_check.py`), it is the displacement.

The frame survives binarisation. `||d_fwd||` is 0.267 against 0.086 — *larger*,
and zero at no position — and the two axes go from `|cos|` 0.11 to 0.29: worse
conditioned, nowhere near parallel. So "`sign(z)` has no local derivative", the
first explanation offered here, is wrong; the difference vectors are bigger than
before.

What breaks is how displacement **accumulates**. `||z(x + k·N) − z(x)||`:

| k cells north | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| gain 100 | 0.086 | 0.167 | 0.314 | 0.559 | 0.908 | 1.281 |
| *per doubling* | | ×1.94 | ×1.88 | ×1.78 | ×1.62 | ×1.41 |
| gain 1e6 | 0.267 | 0.373 | 0.523 | 0.732 | 1.011 | 1.310 |
| *per doubling* | | ×1.40 | ×1.40 | ×1.40 | ×1.38 | ×1.30 |

The linear code moves **ballistically** at short range — ×2 per doubling, i.e.
`‖Δ‖ ∝ k`, a straight line in code space — and only turns diffusive (×√2) past
k ≈ 16. The binarised code is **diffusive from the first cell**: ×1.40 ≈ √2 at
every scale. Successive one-cell steps flip independent sets of coordinates, so
displacement accumulates as a random walk rather than a line.

The arithmetic closes to 1.5%: 1.8% of 1024 coordinates change sign per cell,
each by `2/√D` = 0.0625, so one cell gives `√18.4 × 0.0625` = 0.268 against
0.267 measured, and two give `√2 × 0.267` = 0.378 against 0.373.

A random walk has no direction, and direction is the whole of what `q` reads —
`cos(z(x + k·N) − z(x), d_fwd)`:

| k | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| gain 100 | 1.000 | 0.965 | 0.875 | 0.699 | 0.461 |
| gain 1e6 | 1.000 | **0.699** | 0.494 | 0.347 | 0.237 |

One extra cell costs the binarised code the alignment that eight cells cost the
linear one. Hence |err| is already **55.6° in the nearest distance bin** rather
than degrading with range, and `qnorm` sits flat at 0.18 from d=1 outward where
the working arms climb 0.09 → 0.37 — a `q` whose length carries no information
about how far the goal is. `acc90` holds at 0.71, so a coarse signal survives;
nothing at 45° does.

The sign pattern is not where the smoothness lives. **Both regimes flip the same
1.8% of coordinates per cell** — what the linear code carries in the *magnitudes*
is exactly the differentiable part, and binarising keeps the part that random
walks and discards the part that does not.

So §7's two conditions are not two steps toward one goal, and this is the square
they span:

| | encoder linear | encoder saturated |
|---|---|---|
| **β linear** | production: memory 0.979, direction 7.8° | §10.9's gain ladder, monotone trade |
| **β saturated** | fixed point near-miss, basin −2.5 | perfect memory, **no direction** |

Condition (b) alone buys step-invariance (§10.16) and costs a little basin.
Condition (a) at its limit buys a perfect fixed point and a perfect readout and
destroys the gradient the policy navigates on. The production corner is not a
compromise between them — it is the only corner of the square where both work.

This is §10.9's gain ladder at its endpoint, and it is why that ladder has an
interior optimum at all: raising gain trades angular error for retrieval, and
gain 1e6 is that trade taken to where retrieval is perfect and angle is gone.

#### Would *training* at gain 1e6 fix it? No, and not for a training reason

`att0.5` is trained at `gain_end=100` with the `linspace(1, gain_end, epochs)`
anneal, then arm B evaluates it at 1e6 — so the obvious objection is that the
encoder was never fitted to a binary output. Two answers
(`train_at_high_gain_check.py`), and the second one closes it.

**It would train, but half-dead.** Gradient coverage through `tanh(g·u)`:

| gain | 100 | 300 | 1000 | 1e4 | 1e6 |
|---|---|---|---|---|---|
| params with nonzero grad | 100% | 100% | 100% | 87% | **54.2%** |
| max/mean \|grad\| | 43× | 47× | 51× | 147× | **1400×** |

The anneal is the standard mitigation and is already there, but at
`gain_end=1e6` the ramp is past 1e4 within ~1% of epochs, so nearly all of
training sits in the half-dead regime.

**But gain does not touch the sign pattern at all.** `sign(tanh(g·u)) =
sign(u)` for every `g`, and the Hamming trajectory `H(k)/(k·H(1))` is
bit-identical at gain 100 and 1e6 — 1.000, 0.975, 0.958, 0.938, 0.893, 0.743,
0.431 for k = 1…64. Same code, same signs. Binarisation removes only the
magnitudes, and the magnitudes are the entire difference: `‖Δk‖/(k‖Δ1‖)` is
0.965 / 0.910 / 0.814 at k = 2/4/8 for gain 100 against 0.701 / 0.492 / 0.345
at 1e6, the latter being `1/√k` to three decimals.

**And `√k` is a hard ceiling for any binary code.** `‖Δk‖ = 2√(H(k)/D)`, and
translation invariance caps `H(k) ≤ k·m`: each unit step flips `m` coordinates
on average and they can at best *stay* flipped. So `‖Δk‖ ∝ √k` at best —
ballistic growth is impossible for a binary code, trained or not — and the
current one already sits at 89–100% of that cap out to k=16, i.e. coordinates
essentially never flip back. There is no headroom for training to recover.

Alignment above `1/√k` is buyable only by *slowing* `H(k)`. With `H(k) ∝ k^α`,
`cos ∝ k^(−α/2)` while `‖Δk‖ ∝ k^(α/2)`: better bearing costs a `q` whose length
says even less about distance, and costs discriminability between far cells,
which is what the basin needs. **For a binary code those two are in direct
opposition.** The continuous code escapes by walking the *same* sign trajectory
with a norm that grows linearly on top of it — `cos·√k` reaches **1.97** at k=8,
nearly double anything binary can reach.

> **Open, and it is the readout not the encoder.** Gain is not the lever. A
> basis built from something other than a finite difference of binarised codes
> — the pre-nonlinearity activations, say — would let the memory see corners
> while `q` still sees magnitudes, and is the only route to both. It changes the
> production contract (`gram_schmidt_projection`, §6.2), so it is not a sweep.

Raw: `probe_sat10`; the ladder page carries both as the tabs `10% β=1e6` and
`10% gain=1e6, β=1e6`.
