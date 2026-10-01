# The ideal encoder as Hopfield input

*2026-09-30. Branch `worktree-agent-a792a678d5a963123`. Explainer page: "The Ideal Grid Encoder" (claude.ai artifact C5p2GC3F6j1RxCQjGx6X1h).*

## Summary

We built the encoder that the grid code *should* map to, written in closed form rather than trained, and scored it with the standard Hopfield probe suite next to the production trained encoder (w52 att0.5).

- **It works, if its similarity peak is wide enough.** At r = 16 the ideal encoder matches or beats att0.5 on every probe metric: direction error ~1° (att0.5 ~8°), exact retrieval 0.999–1.000, basin 29–32, reach 0.987–0.998, no dead goals up to K = 20.
- **It is position-independent.** Every metric is identical across the whole arena, the corner, the centre and the opposite region. att0.5's exact retrieval ranges 0.81–0.98 across the same regions.
- **Narrow peaks fail.** At r = 2 direction is at chance (acc45 0.28) and at r = 4 it is 0.50. The direction readout only has signal within about 2r of the goal. So r is set by the distance the agent must navigate from, not by separation.
- **The trained encoder's aliasing is structure, not dimension.** att0.5's near field matches ideal r = 16 almost exactly, but its alias ceiling is 0.79 against 0.13, and its far-field spread is 0.058 against 0.032 (d_eff 297 against 639). In the same 1024 dimensions, a code can have the same near field with six times less worst-case aliasing. This answers THEORY_ENCODER_HOPFIELD §3.6: the excess over chance is (c), structured.

## 1. What the ideal encoder is

Target similarity: a Gaussian bump in displacement on the 1716 × 1716 scaffold torus,

    k(Δ) = exp(−|Δ|² / 2r²).

Any translation-invariant, positive-definite k is a sum of plane waves (Bochner):
k(Δ) = Σ_n P_n cos(ω_n·Δ), with ω_n = 2π n / 1716 for integer n = (n_x, n_y). Splitting each cosine
with cos(a − b) = cos a cos b + sin a sin b turns k into a dot product, so the code is one cos/sin pair per frequency:

    z(p) = [ …, a_i cos(ω_i·p), a_i sin(ω_i·p), … ].

In 2D the frequencies that carry weight fill a disc of ~66,000 distinct waves at r = 4 (radius ~3·1716/(2πr) in n), against 512 slots in 1024 dimensions. Keeping only the lowest 512 with exact weights would make the peak far too wide. So we **sample**: draw 512 frequencies with n_x, n_y ~ round(N(0, σ²)), σ = 1716/(2πr), and give each a_i = 1/√512. P_n then decides which frequencies are picked, not how big each one is. In expectation z(p)·z(p′) = k(Δ), with a far-field wobble of ~1/√(2·512) ≈ 0.031.

**Computed from the grid code, not from p.** The encoder only sees the three module codes, which know p mod 11, 12 and 13. Every arena wave is a product of one wave from each module, because the frequencies add: j₁/11 + j₂/12 + j₃/13 = (156 j₁ + 143 j₂ + 132 j₃)/1716. The Chinese remainder theorem gives the unique triple for each frequency component n:

    j₁ = 6n mod 11,   j₂ = −n mod 12,   j₃ = 7n mod 13.

So the angle of wave i is Σ_m (j_{m,x} φ_{m,x} + j_{m,y} φ_{m,y}), where φ_{m,·} is module m's phase along each axis.

## 2. Implementation

`analysis/hopfield_probe/ideal_encoder.py`: `IdealEncoder(r, n_freq=512, seed=0, gain=100)`, a `torch.nn.Module` mapping the (N, 434) grid code to (N, 1024).

1. For each module and axis, marginalise the λ × λ block over the other axis and take the circular mean over the λ cells to get the phase. This uses the block layout of `encode.py::grid_codes`.
2. Form each wave's angle from the phases with the CRT harmonics above.
3. Output cos and sin of each angle, scaled by 1/√512. The rows are exactly unit norm.

Reading the phase directly (circular mean) is equivalent, in the noise-free case, to reading each harmonic j with cosine weights and dividing by its strength. It avoids that division, which would multiply the weak high-j readouts by up to ~250 (strength 0.004 at |j| = 5).

The probe loader (`harness.load_probe_encoder`) accepts the spec `ideal:r=R[,n_freq,seed,gain]` in place of a checkpoint path, so every probe script runs it unchanged. `gain=100` matches the default Hopfield β, which is taken from the encoder gain. Per `project-hopfield-is-linear`, β has no effect at this scale anyway.

**Tests** (`hopfield_nav/tests/test_ideal_encoder.py`; all 54 probe tests pass after the merge with main):
- output = cos/sin(2π n·p/1716)/√512 on 300 random scaffold positions to within 2e-6, for r = 2, 4, 8, 16 and fwhm 0.25 and 0;
- rows are unit norm to within 1e-5;
- the CRT harmonics reconstruct n;
- the similarity at r = 8 matches exp(−d²/2r²) at short range;
- the loader resolves an `ideal:` spec.

## 3. Setup

- **Encoders:** ideal at r ∈ {2, 4, 8, 16} (seed 0), and the reference `w52_attract_fwhm/000_att0.5_seed=42` and `001_…seed=43`. Per their `train_config`: gain 100, fwhm 0.25, attract 0.5, 118 patches of 50, batch 4096, exclude_cross_env_pairs=True.
- **Probe** (`run.py`, the same settings as `run_corner.sh` / `run_proj.sh`): 8 worlds × 20 envs, K ∈ {1, 3, 5, 10, 20}, recall steps up to 15, `--seed 0`. Four regions via `--world_region`: whole arena, corner `0 0 500`, centre `608 608 500`, opposite `1216 1216 500`.
- **Scan** (`ideal_scan_check.py`): d_eff, far-field spread and the >0.25 rate, sampled as in `why_attract_check.py`. Also the per-reference region scan from `corner_scan.py`: C(1), r_0.9, r_0.5, r_mono, r_u16, and the alias ceiling (max cosine beyond 50 cells over the whole scaffold), over 8 references in each of 5 distance bands from the 500² corner.
- **Slurm:** job 24374455 (probe, array 0–23) and job 24374456 (scan, array 0–5) on ou_bcs_normal. 29/30 tasks completed. The att0.5 s42 scan timed out at 3 h after 4 of its 5 bands, so the scan table uses s43 as the reference. Both seeds' probes completed.

## 4. Results

### Probe, K = 5, one recall step

The ideal encoders give identical numbers in every region (to ~0.01), so each gets one row showing the range over regions.

| encoder | region | \|err\| (°) | acc45 | exact | basin | reach disc | reach cont | acc45 @15 steps | dead @ K = 1/3/5/10/20 |
|---|---|---|---|---|---|---|---|---|---|
| ideal r = 2 | all 4 | 88.6 | 0.28 | 0.37 | 4.0 | 0.26–0.27 | 0.26–0.27 | 0.26–0.27 | 1.00 / 0.92 / 0.92 / 0.92 / 0.92 |
| ideal r = 4 | all 4 | 63.5–63.7 | 0.50 | 0.67–0.68 | 8.5–8.6 | 0.57–0.58 | 0.57–0.58 | 0.48 | 0.62 / 0.54 / 0.54 / 0.54 / 0.54 |
| ideal r = 8 | all 4 | 10.9–11.0 | 0.95 | 0.97 | 17.0–17.3 | 0.98 | 0.96–0.98 | 0.94 | 0 |
| **ideal r = 16** | all 4 | **1.0–1.1** | **1.000** | **0.999–1.000** | **29.2–32.1** | **1.000** | **0.987–0.998** | **0.998–0.999** | **0** |
| att0.5 s42 | whole | 8.65 | 0.997 | 0.982 | 25.4 | 1.000 | 0.993 | 0.986 | 0 / 0 / 0 / 0 / 0.12 |
| | corner | 8.12 | 1.000 | 0.924 | 21.8 | 0.990 | 0.980 | 0.934 | 0 / 0.04 / 0.04 / 0.08 / 0 |
| | centre | 8.61 | 1.000 | 0.976 | 19.8 | 0.978 | 0.995 | 0.960 | 0 / 0 / 0 / 0 / 0.04 |
| | opposite | 10.53 | 0.996 | 0.859 | 14.7 | 0.984 | 0.986 | 0.858 | 0 |
| att0.5 s43 | whole | 7.78 | 0.995 | 0.982 | 27.7 | 0.997 | 0.987 | 0.965 | 0 / 0 / 0 / 0 / 0.08 |
| | corner | 6.55 | 1.000 | 0.930 | 20.6 | 0.983 | 0.970 | 0.938 | 0 / 0.04 / 0.04 / 0.04 / 0.04 |
| | centre | 8.48 | 0.996 | 0.891 | 11.9 | 0.918 | 0.931 | 0.902 | 0 / 0.12 / 0.12 / 0.21 / 0.21 |
| | opposite | 8.27 | 0.999 | 0.813 | 14.9 | 0.960 | 0.962 | 0.826 | 0 |

Sanity check: the att0.5 s42 corner row reproduces THEORY §3.7's published 1.000 / 0.924 / 21.8 / 0.980 (acc45 / exact / basin / reach).

### Scan: far field and near field

| encoder | d_eff | far sd | 1/√d_eff | >0.25 | C(1) | r_0.9 | r_0.5 | r_mono | r_u16 | alias ceiling (median / max) | predicted (5.45·sd) | across bands: C(1) / alias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ideal r = 2 | 811 | 0.0315 | 0.035 | 0 | 0.886 | 1 | 3 | 4 | 3 | 0.150 / 0.150 | 0.171 | constant / constant |
| ideal r = 4 | 814 | 0.0307 | 0.035 | 0 | 0.970 | 2 | 5 | 9 | 6 | 0.146 / 0.146 | 0.169 | constant / constant |
| ideal r = 8 | 754 | 0.0318 | 0.036 | 0 | 0.992 | 4 | 10 | 18 | 10 | 0.133 / 0.133 | 0.173 | constant / constant |
| ideal r = 16 | 639 | 0.0318 | 0.040 | 0 | 0.998 | 8 | 19 | 38 | 13 | 0.133 / 0.133 | 0.176 | constant / constant |
| att0.5 s43 | 297 | 0.0582 | 0.058 | 0.0096 | 0.996 | 6 | 19 | 35 | 4 | **0.786 / 0.883** | 0.297 | 0.996 / 0.765–0.813 |

(1/√D = 0.0312 for every encoder.)

## 5. Interpretation

**Direction needs a wide peak.** The readout q = basis·(z_goal − z(p)) projects the code difference onto the local frame. Its signal is the slope of the similarity to the goal, which is non-zero only within about 2r (r_mono ≈ 2.2r for the ideal codes). Beyond that, only the ±0.03 far-field wobble is left, and the heading is random. At r = 2 the basin (4.0) and r_mono (4) coincide, and acc45 sits at the 0.25 chance level. Exact retrieval collapses too (0.37): a start several r from the goal has cue–goal similarity below the cross-talk from the other K − 1 goals. Separation never limits the ideal code; its alias ceiling is 0.13–0.15 at every r.

**The trained encoder has picked about the right r.** The att0.5 near field matches ideal r = 16 to within a few cells: C(1) 0.996 against 0.998, r_0.5 19 against 19, r_mono 35 against 38. What it lacks is the clean far field: alias 0.79 against 0.13, far sd 0.058 against 0.032, and d_eff 297 against 639. It also has a smaller r_u16 (4 against 13), a local roughness the ideal code does not have. Both the aliasing excess and the region dependence belong to the trained function, not to the target or the dimension.

**r = 16 is not necessarily optimal.** Reach, basin and direction were still improving from r = 8 to 16. What limits large r (smaller d_eff; exact arrival once neighbouring cells become nearly identical) has not been measured.

## 6. Predictions made before the run

| prediction | outcome |
|---|---|
| far-field sd ≈ 1/√1024 ≈ 0.031 | **held**: 0.031–0.032 |
| alias ceiling ≈ 7.6·sd ≈ 0.24 (revised to 5.45·sd ≈ 0.17 for a single reference over ~2.9M cells) | **held**, below either value: 0.13–0.15 |
| small r → better exact / separation; large r → longer reach / basin | **half wrong**: large r is better on everything measured, and small r is worse on exact too |
| ideal metrics identical inside and outside any region | **held**: identical to 3 decimals across regions and bands |

## 7. Caveats

- These are probe metrics (memory plus the q readout with scripted following), not navigation with a trained policy. A policy trained on att0.5 cannot be reused, because it is tied to that encoder's q statistics.
- The ideal encoder uses one frequency draw (seed 0) per r. Variation across draws is not measured, though at 512 frequencies it should be small.
- Only one reference scan (s43) completed; the s42 scan needs ~4 h.
- The ideal encoder reads phases noise-free. With noisy cells, the high harmonics (via the phase or via divided readouts) would amplify the noise; that is not tested.

## 8. Reproduce

```bash
# from the worktree / branch checkout, on a compute node (never the login node: 5 GB scaffold)
bash analysis/hopfield_probe/run_ideal.sh          # submits probe (24 tasks) + scan (6 tasks)
python -m analysis.hopfield_probe.ideal_summary \
    /orcd/pool/003/jackking/cls_runs/results/hopfield_probe/ideal_encoder
```

Results: `/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/ideal_encoder/{probe,scan,logs}`.

## 9. Open questions

- Where does r stop helping? (r = 24, 32; not run yet.)
- How many goals K can the ideal code hold before cross-talk kills exact retrieval?
- Can a network learn this function? The ideal code is cos/sin of (integer-weighted sum of module phases), which suggests a phase decoder, then a linear layer with integer weights, then a cos/sin activation. Test it with frozen random integer frequencies (should match exactly), then with trainable frequencies under the usual loss and patch sampling: does gradient descent find integer j from 10% coverage?
- Is the trained encoder's structured aliasing (0.79 at the triple near-realignments 780/792) a failure to use the weak high-j harmonics? Measure its effective frequency content against the ideal's.

## 10. Recall dynamics and saturated recall (2026-09-30, second run)

Page: "Ideal Encoder Probe" (claude.ai artifact XZHyrhhWjXocQHRWSumohn), built by
`python -m analysis.hopfield_probe.ideal_report OUT --lede OUT/lede.html`.
Runs: `run_ideal_dyn.sh dyn` (job 24446458, `ideal_dynamics_check.py`, 4 headers)
and `run_ideal_dyn.sh sat` (job 24446459, run.py with `--beta 1e6`). Results in
`OUT/dynamics/*.json` and `OUT/sat/t*/`.

**Projection storage gives exact fixed points without saturation.** Self-recall
at α = 1, cos to its own stored pattern at steps 1 / 5 / 15 / 30 (K = 5):

| encoder, recall | hebb | proj |
|---|---|---|
| att0.5 s42, β = 100 | 0.998 / 0.960 / 0.830 / 0.689 | 1.0000 throughout (K = 20 too) |
| ideal r = 16, β = 100 | 0.998 / 0.960 / 0.795 / 0.617 | 1.0000 throughout (K = 20 too) |
| ideal r = 16, β = 1e6 | 0.906 / 0.902 / 0.901 / 0.901 | 0.906 → 0.905 |
| arm B, gain = β = 1e6 | 1.0000 | 1.0000 |

`proj` stores the orthogonal projector onto span(Z) with the diagonal kept, so
P z = z. Every vector in span(Z) is fixed, so these are fixed points but not
isolated attractors. Saturated recall moves the ideal code's patterns to the
nearest sign pattern (a hypercube corner), at cos ≈ (2/π)/(1/√2) ≈ 0.90 for a
cos/sin code.

**α walk, K = 5, cues ~10.5 cells out** (decoded distance at steps 1, 2, 3, 5, 8, 12, 20, 30; min cos over steps 1–12; cos at step 30):

| encoder · storage · α | walk | min cos 1–12 | cos s30 |
|---|---|---|---|
| ideal r=16 · proj · 0.9 | 6.82 3.63 1.79 0.41 0.00 0.00 0.00 0.00 | 0.993 | 0.999 |
| ideal r=16 · hebb · 0.9 | 6.81 3.60 1.78 0.40 0.00 0.00 0.24 0.34 | 0.955 | 0.811 |
| att0.5 · proj · 0.9 | 7.91 3.57 1.40 0.13 0.03 0.02 0.02 0.02 | 0.972 | 0.992 |
| att0.5 · hebb · 0.9 | 7.89 3.53 1.37 0.16 0.12 0.23 0.31 0.36 | 0.945 | 0.846 |
| ideal r=16 β=1e6 · hebb · 0.01 | 8.48 6.59 4.93 2.71 1.19 0.43 0.28 0.34 | 0.910 | 0.902 |
| arm B · hebb · 0.01 | 10.46 10.46 0.00 0.00 … | 0.919 | 1.000 |

The att0.5 hebb row reproduces THEORY §7.1 exactly.

**Chord** x(t) = (1−t) z(here) + t z(goal), decoded distance at t = 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1:

| encoder | start 10 | min cos | start 20 | min cos |
|---|---|---|---|---|
| ideal r = 16 | 10.0 8.1 6.0 5.0 4.1 1.9 0.0 | 0.998 | 20.0 17.5 13.0 10.0 7.0 2.5 0.0 | 0.969 |
| att0.5 s42 | 10.0 8.5 6.4 4.9 3.8 1.6 0.0 | 0.990 | 20.0 18.6 15.3 11.0 5.3 1.8 0.0 | 0.930 |
| arm B | 10.0 10.0 10.0 2.7 0.0 0.0 0.0 | 0.918 | 20.0 20.0 20.0 1.7 0.0 0.0 0.0 | 0.827 |

**Full probe with saturated recall (β = 1e6)**, K = 5, one step:

| encoder | region | \|err\| ° | acc45 | exact | basin | reach cont |
|---|---|---|---|---|---|---|
| ideal r = 16 | whole / corner / centre / opposite | 2.72 / 2.50 / 2.93 / 2.73 | 1.000 | 0.719 / 0.928 / 0.762 / 0.790 | 10.2 / 18.4 / 15.3 / 11.9 | 0.977 / 0.939 / 0.924 / 0.996 |
| att0.5 s42 | whole | 8.87 | 0.998 | 0.941 | 22.4 | 0.987 |

**Reading.**
1. Walking in and stopping is available without saturation: projection storage plus α < 1. On the ideal code it is cleaner than on att0.5 (cos ≥ 0.993 against ≥ 0.972, rest at 0.00 against 0.02 cells).
2. Snapping is a property of a binary CODE, not of saturated recall. The ideal code's chord is on-manifold at every start distance tested (predicted up to ~2r = 32). Saturated recall on the ideal code still walks at small α, but off-manifold at a constant cos ≈ 0.90, and it rests near, not on, the goal.
3. Saturated recall costs the ideal code precision (exact and basin) and position-independence, but not direction.

## 11. Stage A: does gradient descent find the integers? (2026-09-30)

`IdealNet` with layers 1–2 fixed and the 6 → 512 harmonic layer trainable
(`encoder_training/train_ideal_net.py`, `run_ideal_net_stageA.sh`, job 24487677;
task 7 re-run as 24488275 after two failures on node3804, "CUDA device busy").
Data = the att0.5 recipe (118 random 50² patches, 10% of the scaffold, fwhm 0.25,
batch 4096, within-patch pairs only). Loss = mean (z(p)·z(q) − exp(−d²/2·16²))²
on those pairs. Adam lr 1e-2, 500 epochs, about 4.5 min per run on one GPU.
Diagnostics: `encoder_training/ideal_net_diagnostics.py`. Results:
`/orcd/pool/003/jackking/cls_runs/results/ideal_net/stageA/<init>_s<seed>/`.

| start | loss | rows within 0.05 of an integer | r_eff (integral rows) | \|n\| p10/50/90 | C(1) | r_half | kernel RMSE, 0–71 cells | far sd | alias ceiling |
|---|---|---|---|---|---|---|---|---|---|
| exact integers (3 seeds) | 0.0001–0.0002 | 0.95–0.96 | 16.5–16.7 | 7 / 20 / 35 | 0.992–0.995 | 19 | 0.009–0.015 | 0.037 | 0.13–0.14 |
| integers ± 0.3 (3 seeds) | 0.0001 | 0.955–0.959 (by epoch 10) | 15.4–16.6 | 7 / 20 / 34 | 0.994–0.995 | 19 | 0.008–0.011 | 0.036 | 0.13–0.15 |
| uniform [−6, 6] (3 seeds) | 0.107–0.133 | 0.55–0.62, still rising | 11.1–11.9 | ~24 / 32 / 39 | 0.67–0.71 | 10 | 0.24–0.25 | 0.033 | 0.18–0.22 |

Integral fraction by epoch, random starts: 0.10–0.15 (10) → 0.23–0.29 (50) →
0.33–0.36 (100) → 0.44–0.48 (200) → 0.51–0.55 (300) → 0.55–0.62 (500). Not
converged at 500 epochs.

**Reading.**
1. **Integers are strong local attractors.** From ±0.3 every seed reaches ~96%
   integral within 10 epochs, with the same kernel as the ideal encoder.
2. **From a random start, gradient descent finds integers for only about half
   the rows, and the kernel is poor.** Loss 0.11–0.13 against 0.0001;
   similarity at one cell 0.67–0.71; half-height at 10 cells against 19. The
   integral rows sit in a narrow band of frequencies (|n| ≈ 21–40), not the
   Gaussian's spread (median 20).
3. **Non-integrality is being used as a volume knob.** Even from the exact
   integers, ~4.5% of rows (23–25) leave, and they are the highest-frequency
   waves of the draw (median |n| 39 against 20), moving ~0.2 off. The layer has
   no per-row amplitude, so detuning a row is the only way to turn a wave down.
   98% of the rows that stay integral keep the ideal integers they started on.
   This is also the likely story for the random start: rows whose nearby integers
   give an unhelpful frequency stay detuned instead of becoming integral.

**Next.** Give each row a learnable non-negative amplitude, so a wave can be
turned down without breaking integrality, and train longer (the random runs are
still improving at 500 epochs). Then repeat the random start. Stage B (the
campaign's own loss) and Stage C (corner-only training) wait on that.

### 11.1 Correction: why the high-frequency rows drift (least-squares and local-gradient check)

`encoder_training/ideal_net_lsq_check.py`. Fixed frequencies = the ideal r=16
draw; features cos(ω_i·Δ) on ~1M within-patch pairs (the Stage A distribution);
target exp(−|Δ|²/2·16²).

- **Measured damping.** Each drifted row's contribution to pair similarity, as a
  fraction of its ideal wave's: 0.80–0.85 at every distance from 3 to 71 cells,
  and 0.90–0.92 below 3 cells. Integral rows: 1.000. So the drift is an amplitude
  cut done through the frequency weights (the layer has no amplitude), nearly
  flat in distance. My earlier claim that it damps mainly mid-distance pairs was
  wrong.
- **Global optimum (non-negative least squares).** Loss ≈ 0 (equal weights:
  0.00048) using **34 of 512** waves, with far-field sd **0.19** (equal weights:
  0.036). The within-patch objective sees only offsets inside a 99×99 window and
  can be fit exactly by a sparse set that rings elsewhere: the §9 blind spot again.
- **Local gradient at equal weights** (z-scored, + = wants the weight lower):
  |n| 0–10 +1.01, 10–20 +0.58, 20–30 −0.42, 30–40 −0.57, 40–50 −0.66; corr(z, |n|)
  = −0.52. The loss wants low frequencies down and high ones up. **This
  contradicts the explanation in §11** ("the loss sees the high-frequency waves'
  wobble but not their benefit"). The drifted rows are enriched for individually
  positive gradients (78% vs 54%; 5 of the top 23 against ~1 by chance), but they
  are not a band effect. Detuning can only turn a wave down, so the network acted
  on particular high-|n| rows. Why those rows and not equally penalised low-|n|
  rows is not explained.

**Consequence for the next run.** Learnable per-row amplitudes alone would let
the network collapse toward the sparse, ringing optimum. The objective needs a
far-field term (random scaffold pairs with target ≈ 0, or the coding-rate term)
alongside the amplitudes.

## 12. Objective first: the optimum of kernel MSE + coding rate (2026-09-30)

`encoder_training/ideal_net_rate_optimum.py`, GPU job 24504580; weights and
`rate_optimum.json` in `/orcd/pool/003/jackking/cls_runs/results/ideal_net/rate_optimum/`.
Frequencies fixed, amplitudes a on the simplex (unit-norm codes). Loss =
within-patch kernel MSE (r = 16) − λ·rate, with the rate exactly
`losses.coding_rate_loss` (ε = 1) on the recipe's training positions. MSE is
convex in a, and the log-det is concave in a (Sylvester), so the problem is
convex; solved by exponentiated gradient from equal weights.

**First, the least-squares encoder in the probe** (§11.1's 34-wave optimum,
`ideal:r=16,...,weights=lsq_weights_r16_seed0.npy`, job 24503516): acc45
0.989–0.995, direction error 7–8°, but exact 0.12–0.15, basin ≈ 0, reach
0.58–0.72, 75% dead goals at K ≥ 10. So minimising the within-patch loss
alone can give a bad Hopfield input.

| menu | λ | MSE | waves (PR) | C(1) | r_half | r_eff | max beyond 71 | far sd |
|---|---|---|---|---|---|---|---|---|
| 512 draw | 0 – 0.1 | 2e-6 – 4e-5 | ~470 | 0.998 | 19 | 15.9–16.1 | 0.05–0.06 | 0.035–0.038 |
| 512 draw | 0.5 | 0.00047 | 471 | 0.998 | 19 | 15.5 | 0.059 | 0.035 |
| 512 draw | equal | 0.00048 | 512 | 0.998 | 20 | 16.2 | 0.050 | 0.037 |
| flat \|n\| ≤ 40 (2513) | 0 – 0.1 | 1.3e-5 – 1.2e-4 | 1265–1600 | 0.998 | 19 | 16.2–16.3 | 0.12–0.16 | 0.020–0.021 |
| flat | 0.5 | 0.0013 | 1731 | 0.998 | 18 | 15.5 | 0.090 | 0.019 |
| flat | equal | 0.035 | 2513 | 0.997 | 16 | – | 0.041 | 0.015 |

**Reading.**
1. **With the rate term, the optimum is a good code at every λ on both menus.**
   On the flat menu (no Gaussian shape built in) the MSE reshapes the spectrum
   to the target (r_eff 15.5–16.3, half-height 18–19) and the rate keeps
   1300–1700 waves in use.
2. **The within-patch MSE is not simply broken; it is degenerate.** Exponentiated
   gradient from equal weights at λ = 0 also reaches near-zero MSE with a dense,
   good code (470 waves, far sd 0.038). NNLS's 34-wave solution is a vertex of a
   flat set of near-optimal solutions. The rate term breaks that tie toward the
   dense one.
3. My prediction that "at λ = 0.5 the rate dominates and the target barely
   matters" held on the 512 draw (the optimum ≈ equal weights) but not on the
   flat menu, where the MSE still shapes the kernel.

**Objective chosen:** kernel MSE (r = 16) + coding rate, λ = 0.5, ε = 1, as in
the recipe. **Next:** Stage A from random starts with this loss: does SGD find
integer weights implementing it?

## 13. Stage A with the coding-rate term: the rate term breaks integrality (2026-09-30)

Job 24506866 (`RATE_LAMBDA=0.5 run_ideal_net_stageA.sh 0,6,7,8`; results in
`.../ideal_net/stageA_rate/`). Same as §11 plus 0.5 × `coding_rate_loss(z, eps=1)`
on each batch's codes.

| start | loss | integral (< 0.05) | C(1) | r_half | kernel RMSE < 71 | far sd | alias |
|---|---|---|---|---|---|---|---|
| exact integers | −0.135 | **0.50 from epoch 10 on** | 0.90 | 18 | 0.049 | 0.032 | 0.136 |
| random (3 seeds) | −0.036 to −0.048 | 0.45–0.48 | 0.60–0.64 | 8–9 | 0.23–0.24 | 0.032 | 0.15–0.19 |
| random, no rate (§11) | – | 0.55–0.62 | 0.67–0.71 | 10 | 0.24–0.25 | 0.033 | 0.18–0.22 |

**Reading.**
1. **The rate term pushes rows off the integers.** From the exact ideal integers,
   half the rows leave within 10 epochs and stay off. The likely mechanism: a
   detuned row adds a phase jump at every module wrap, which looks like noise
   across positions and raises the batch's spread, which is what the rate term
   rewards. The cost is near-field roughness: C(1) 0.998 → 0.90.
2. **Random starts are not helped** (slightly worse than without the rate term).
3. **§12's objective check was incomplete.** It held the frequencies at integers
   and varied amplitudes. The trainable network has a third way to lower the
   loss, breaking integrality, and under the full network the loss's optimum is
   not the integral ideal code.

**Next (proposed).** Make integrality architectural instead of learned: a fixed
integer frequency table (e.g. every |n| ≤ 40, ~2500 waves) with learnable
amplitudes only. That is exactly §12's convex model, whose optimum is a good
code, so SGD reaches it by construction. Open design question: the 1024-number
budget (sparsity pressure, or keep the top 512 amplitudes), then the probe.

## 14. Fixed integer table + learned amplitudes (2026-10-01)

Make integrality architectural: a fixed table of integer frequencies (every
half-plane |n| ≤ 40, 2513 waves) with learned amplitudes only. Training is then
§12's convex problem, whose optimum (kernel MSE + 0.5 × rate) is a good code:
1731 effective waves, MSE 0.0013, r_half 18, far sd 0.019.

**Top-512 by weight does not fit the 1024-number budget well**
(`encoder_training/ideal_net_topk_refit.py`; `.../ideal_net/topk/`). The 512
largest weights are the low-|n| centre of the disc (45% of the weight), so the
peak comes out too wide:

| | MSE | waves (PR) | r_half (target 19) | r_eff | max beyond 71 | far sd |
|---|---|---|---|---|---|---|
| top-512, renormalised | 0.095 | 499 | 34 | 30 | 0.15 | 0.034 |
| top-512, refit | 0.027 | 233 | 26 | 24 | 0.32 | 0.048 |

This is the 2D budget problem of the explainer page's step 06 ("exact lowest N").
Probe of the refit code (job 24510166; `table top512 refit`): whole arena
(stored goals ~350 cells apart) exact 0.977, basin 26, reach 0.99, direction
error 0.8°; but every 500-cell region (goals ~99 cells apart) exact 0.42, basin
4.5, reach 0.82, 33–38% dead goals at K ≥ 10. The same failure as r = 48 (§9):
the too-wide peak and high far tail let neighbouring goals leak into recall.

### One possible solution (not yet run)

1. **Train amplitudes over the fixed table.** Convex, so SGD reaches §12's
   optimum.
2. **Sample 512 frequencies with probability ∝ amplitude** (importance sampling,
   as in random Fourier features), merging repeated draws.
3. **Refit the amplitudes of the sampled 512** under the same objective (convex).

Why it should work: sampling ∝ the learned spectrum keeps the disc's high
frequencies in proportion, which is how the ideal encoder itself is built. That
construction already works at r = 16 (§4–5). Steps 1 and 3 are convex, and step
2 is the random-features construction, so nothing relies on SGD finding discrete
structure.

Caveats: the frequency table is a strong inductive bias (it assumes the answer
is a set of integer waves); duplicates in step 2 reduce the effective count
below 512; and the result is only as good as a single random draw of 512.

## 15. A translation-invariance penalty (2026-10-01)

`train_ideal_net.py --inv_lambda` (variance of z(p)·z(q) across within-batch pairs
with the same displacement; Δ and −Δ pooled; within-patch pairs only),
`--near_weight` (MSE weighted by 1/pairs per 2-cell distance bin) and
`--inv_ramp_epochs`. All runs add 0.5 × rate. `run_ideal_net_inv.sh`; results in
`.../ideal_net/inv_{a,b,c}/`. Penalty size: ~1e-31 at the ideal integers,
2.8e-4 for the rate-seamed codes of §13.

**(a) Is the ideal stable?** Exact-integer start, 100 epochs (job 24511536):

| λ_inv | integral at 10 / 100 (near off; near on) | C(1) | r_half | kernel RMSE < 71 |
|---|---|---|---|---|
| 0 (§13) | 0.49 / 0.50 | 0.90 | 18 | 0.049 |
| 10 | 0.58 / 0.59; 0.66 / 0.67 | 0.97 | 17 | 0.046–0.047 |
| 30 | 0.64 / 0.62; 0.70 / 0.71 | 0.98 | 17 | 0.039–0.046 |
| 100 | 0.77 / 0.72; 0.79 / 0.76 | 0.99 | 18 | 0.026–0.030 |
| 300 | 0.95 / 0.91; 0.92 / 0.89 | 0.997 | 20 | 0.0125 |

λ_inv = 300 largely cancels the rate term's incentive to seam.

**(b) Random starts, full strength** (λ_inv 300 / 1000 × near off/on × 3 seeds;
jobs 24513157–9): 97–100% integral by epoch 10, but MSE 0.21, C(1) ≈ 0,
r_half 1, r_eff 0.6 — a hash-like code. Every integer is translation-invariant,
and the penalty freezes each row on the nearest one.

**(c) Random starts, annealed** (λ_inv 0 → 300 over 250 of 500 epochs; jobs
24540147–9): 76–96% integral already at epoch 10 (λ_inv ≈ 11), 99–100% at the
end; MSE 0.15–0.20, C(1) 0.08–0.45, r_half 1. Same failure.

**It is an optimisation failure, not the loss.** Under rate 0.5 + invariance 300,
the trained exact-integer start sits at total loss ≈ −0.126 (MSE 0.00046), while
the random starts end at ≈ −0.02 to +0.05 (MSE 0.15–0.22). The rate term alone
prefers the hash-like code (−0.33 vs −0.255), but the MSE gap dominates. A much
better solution exists; SGD does not reach it. (Not shown: that the ideal is the
global minimum.)

**Why random starts fail: exact periodicity, not frequency learning.** Within one
module period a row is a plane wave with local frequency j₁/11 + j₂/12 + j₃/13
cycles per cell per axis, linear in the weights, so gradient descent does move a
row's frequency smoothly. What is hard is exactness: only integer weights line
the wave up at every module wrap, and the integer points with the low arena
frequencies a width-16 peak needs (|n| ≲ 50 in n = 156 j₁₁ + 143 j₁₂ + 132 j₁₃
mod 1716; e.g. (1, −1, 0) → 13) are rare and far apart. So rows reach roughly the
right local frequency and then either sit between integers (seams, which the rate
term rewards) or, under the invariance penalty, snap to the nearest integer,
usually a wrong, high frequency.

**Options.** (1) The fixed integer table with sampling by weight (§14).
(2) Decode position from the three phases (exact by the CRT), then learn
continuous frequency vectors with a cos/sin layer. This helps not because it
learns frequencies "directly" (the current net already does, locally) but because
a non-integer frequency on a decoded position seams once, at the 1716 arena edge,
instead of at every module wrap, so integrality barely matters. (3) A discrete
search over n under the convex objective.

## 16. The probe with projection storage (2026-10-01)

Every probe number above used Hebbian storage; projection storage had only been
used in the dynamics check (§10). `STORAGE=proj run_ideal.sh probe 12-19,24-31`
(job 24547428), the same settings plus `--storage_rule proj`; results in
`.../ideal_encoder/probe_proj/`.

| (K = 5, s = 1) | storage | \|err\| ° | acc45 | exact | basin | reach (cont) | dead K 3/5/10/20 |
|---|---|---|---|---|---|---|---|
| ideal r = 48, 500-cell regions | hebb | 7.76 | 0.975 | 0.355 | 7.4 | 0.52 | 0.42/0.54/0.54/0.54 |
| | **proj** | **0.85** | **1.000** | **0.733** | **14.5** | **0.99** | **0/0/0/0** |
| ideal r = 48, whole | hebb / proj | 1.16 / 0.62 | 1.000 | 1.000 | 54.6 / 50.2 | 0.99 | 0 |
| ideal r = 32, regions | hebb / proj | 0.87 / 0.59 | 1.000 | 1.000 | 44.4 / 44.0 | 0.99 | 0 |
| ideal r = 16, regions | hebb / proj | 1.12 / 1.02 | 1.000 | 0.999 | 29.2 / 28.9 | 0.99 | 0 |
| att0.5 s42, regions | hebb | 8.1–10.5 | 0.996–1.000 | 0.86–0.98 | 14.7–21.8 | 0.98–0.995 | up to 0.08 |
| | proj | 7.8–10.0 | 0.997–1.000 | 0.97–0.99 | 18.1–23.7 | 0.975–0.995 | ≤ 0.04 |

**Reading.**
1. **The navigation width limit of §9 was Hebbian cross-talk.** Projection
   storage takes r = 48 in the regions from 54% dead goals and reach 0.52 to no
   dead goals, reach 0.99 and ~1° direction error. "Usable r ≲ goal spacing / 3"
   is a property of Hebbian storage, not of wide codes.
2. **What remains is a precision limit from the kernel itself.** Exact retrieval
   of the goal cell reaches only 0.73 (basin 14.5) at r = 48, against 1.00 at
   r = 32: neighbouring cells differ by 1 − k(1) ≈ 2×10⁻⁴, so the exact cell is
   fragile. Projection storage removes the other goals' leak but not that flatness.
3. **Projection storage also helps the trained encoder:** att0.5's
   region-dependent exact retrieval rises from 0.86–0.98 to 0.97–0.99.
4. At r = 16 and 32 the ideal code barely changes (direction error drops slightly).

### 16.1 Dimension cannot fix kernel overlap; the kernel shape can (noted for later)

In the linear recall regime (β = 100, tanh inert), every step of the probe is a
function of inner products between codes only:
- Hebbian recall Σₖ (zₖ·x) zₖ has similarity Σₖ k(x−gₖ) k(c−gₖ) to a cell c;
- projection recall P x, P = Z(ZᵀZ)⁻¹Zᵀ, has similarity k_cᵀ G⁻¹ k_x;
- normalisation, the direction readout and decoding are dot products too.

So a code that matches a target kernel exactly gives the same probe results at
any dimension. The r = 48 overlap (similarity 0.12 between goals ~99 cells
apart) is the kernel's own value, and more dimensions cannot remove it.

Caveats: (1) a saturating recall nonlinearity (β = 1e6) acts element-wise and does
depend on the representation; (2) at finite D the code only approximates the
kernel, with random far-field wobble ~1/√(2·waves); more D shrinks that, but it is
not what limits r = 48; (3) **the lever is the kernel's shape** — a target with
a sharper fall-off or slight negative lobes, instead of a Gaussian, could cut
neighbour overlap at a given peak width. More dimensions matter there only in that
sharper kernels need more frequencies to realise. **To do later:** compare kernel
shapes at fixed near-field width.
