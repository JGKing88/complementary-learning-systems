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
