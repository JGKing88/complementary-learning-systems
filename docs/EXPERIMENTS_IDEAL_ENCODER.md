# The ideal encoder as Hopfield input

*Started 2026-09-30; map current as of 2026-11-04 (through §22). Branch
`worktree-agent-a792a678d5a963123`. Pages: "The Ideal Grid Encoder" (teaching,
claude.ai/artifact/FSKXEs5KkvwNxk7Ad7PTeZ, republished 2026-10-31 under the
current account; the original C5p2GC3F6j1RxCQjGx6X1h is on the old account);
"Grid Wave Network" (claude.ai/artifact/AhAWfeS2AB546xayRMj8gj); "Learning the
Grid Wave Network" (teaching: L1 from movement, L3 local rules;
claude.ai/artifact/RBo3jTbaPEtaweGyu3fF3G); "Ideal Encoder
Probe" (results, old account only, XZHyrhhWjXocQHRWSumohn, not republished).*

## Status and map (read this first)

**The question.** What should the encoder from the grid code to the Hopfield
memory compute, and can a biologically plausible network reach it?

**Headline results**

1. **The ideal encoder exists and is the best Hopfield input tested** (§1–10,
   §16). It is random Fourier features of a Gaussian kernel, r = 16, 512
   waves, computed from grid phases through the Chinese remainder theorem
   (CRT). It has ~1° direction error, exact retrieval ~1.0, and is the same in
   every region. The trained att0.5 encoder's excess aliasing is structure,
   not dimension.
2. **Gradient descent cannot find it** (§11–17). Training the integer layer of
   IdealNet fails because the correct integers are isolated local minima (the
   kick test, §17.1): a search problem. Kernel MSE plus coding rate prefers a
   seamed code (§13.1). An invariance penalty fixes the loss ranking but not
   the search (§15).
3. **A random projection, a threshold and a learned readout cannot either**
   (§19). A readout fitted to the ideal memorises and transfers nothing.
   Slowness learning finds each module's own period, which is aliased.
4. **The grid wave network can** (§20). It has three layers:
   - L1: per-module sinusoidal readouts;
   - L2: random triples, one pattern per module, multiplied into one plane wave;
   - L3: a slowness gain per unit.

   It is the ideal encoder with selection in place of search. 400k units match
   the ideal. A 17,424-neuron branch design gives the exact kernel and beats
   the ideal in the probe (basin 64 against 32).
   - L3 is local and incremental: per-step depression gives r = √(2κT)
     (§20.1).
   - L3 can instead be stochastic pruning, which is the ideal's own
     construction (§20.2).
5. **What the circuit needs** (§20):
   - L1 purity well above 0.99. Learnable from movement (eigenvectors of each
     module's one-step shift operators); not from Hebbian PCA, where patterns
     tie in groups of 4–8.
   - L1 can be learned inside one 14 × 14 box (§21). A local online delta rule
     learns the shift; with the eigenvector step the network equals exact L1
     (probe basin 34). A fully online lock-in rule with a slow turn learns exact
     pairs too, but holds patterns with uneven numbers of units, which aliases
     (0.19–0.34 across runs; an oracle keeping one unit per pattern gives
     0.07–0.10). In the probe that alias costs little: with L1 and L3 both
     from local rules, the fully online network matches the ideal on reach
     (0.98–0.995) and exact retrieval (1.000), with a basin about 2 cells
     smaller (29–30 against 32) and direction error 1.5–2.6° against 1.0°
     (§21.1, §22.1).
   - L1 and L3 can learn on the same walk if L3's plasticity opens late (a
     fixed window long after L1 has settled; a window timed by the units'
     own settling gets most of the way) (§22.1). Under cell noise both layers' learning
     rates must wind down as experience accumulates (§22.2); trained that way
     under noise, the shift-rule network navigates like exact L1 (§22.3).
   - With noisy cues and stored goals (readout basis clean), the network is
     unaffected by test-time cell noise 0.01; the ideal is hurt more, mostly
     because of how it reads harmonics (§22.4, §23).
   - Multiplication accurate to a few percent.
   - Narrow grid bumps (0.10 of the period) to survive cell noise, with gains
     corrected for jitter at rest.

**Models at a glance** (the ones worth comparing; probe numbers are whole /
corner, K = 5, clean codes unless stated)

| model | how it is built or learned | better at | worse at | biological plausibility |
|---|---|---|---|---|
| Ideal encoder (§1–10) | Not learned. Reads each module's phase (atan2), then 512 random integer frequencies through the CRT, Gaussian r = 16 | Best clean probe (basin 32 / 30, error 1°), smallest code | More fragile to test-time noise at narrow bumps: as implemented (harmonics up to l − 1) reach 0.58 at noise 0.01; with symmetric harmonics (|j| ≤ l/2) basin 17 and reach 0.99 at noise 0.01 (network 32 and 1.0) and reach 0.23 at noise 0.03 (network 0.98) (§22.4, §23); its integer frequencies cannot be found by gradient descent (§11–17) | Low: atan2 and chosen integers |
| Trained att0.5 (production baseline) | Gradient descent on kernel objectives | Learned end to end | Direction error ~8°, exact retrieval 0.81–0.98 depending on region, aliasing | Low: backprop |
| Random projection + threshold + readout or slowness (§19) | Random weights, learned linear readout | The canonical plausible circuit | Does not work: readout memorises, slowest features alias ≥ 0.92 | High, but fails |
| Grid wave network, exact L1 (§20) | L1 hand-wired (every pattern of every module); random triples; L3 gains | Equals the ideal kernel; robust to test noise (§22.4) | L1 not learned | Reference only |
| Branch design (§20) | Exact L1; one neuron per (j11, j12) pair keeping the slowest of 169 module-13 branches | Exact kernel with 17,424 neurons; best probe (basin 64 / 53) | Needs selection among dendritic branches; no learning rule tested | Medium (dendritic branch selection) |
| Shift rule + eigenvectors (route 1c, §21–22) | A delta rule learns each module's one-step shift from the walk (local, online); the eigenvectors of that shift give exactly one pair per pattern | Exact without noise from ~30k steps in one 14-box (basin 33–35); under noise, with a slow rate and ~1M–3M steps, matches its exact twin and navigates like exact L1 (basin 34 / 35, §22.3) | The eigenvector step is linear algebra, not a learning rule | Medium: one non-neural step |
| Lock-in, fully online (route 2c, §21–22) | Each L1 neuron learns one wave by itself (predictor synapses, an intrinsic turn that locks 1000 times more slowly than the weights); depression L3 on the same walk, window opening late | No offline learning step in the §22.1 version (pair scales by a running power; pruning only drops near-zero units for compute); reach and exact retrieval like the ideal, basin about 2 cells smaller (29–30), direction error ~2.3° | Needs ~6× more L1 neurons than patterns (2,604 for 434) and holds patterns unevenly (kernel alias 0.19–0.34 across runs, about 2–4 cells of basin); under training noise 9° error and up to 17% dead goals after 1M steps (§22.3) | Highest: needs paired (complex) outputs, predictor synapses, an oscillator-like turn per direction, and velocity input |
| Lock-in + lateral competition (proposed) | As above, with units that hold the same pattern suppressing each other | Should hold one unit per pattern (434), removing the uneven counts (an oracle keeping one unit per pattern gives alias 0.07–0.10) | Untested | High (anti-Hebbian lateral inhibition is local) |

The tuned settings (lock-in η 0.2 and turn ratio 0.001, shift-rule rates)
were chosen by comparing against the true patterns or shift (§23); the
networks' weights never see the truth.

L3 options, all equivalent once L1 has settled: **depression** (the default:
one multiplicative update per step, nothing stored, kernel width set by how long
plasticity runs), **trace** (running mean of the change, then exp(−r²v̄/2);
width a parameter), **stochastic pruning** (keep each unit with probability
P(n) at full weight, the ideal's own construction, like cell death; §20.2).
Ruled out for L1: Hebbian PCA (patterns tie in groups of 4–8), the plain
Hebbian lock-in rules 2a and 2d (never purify), a fast adaptive turn (units
collapse onto coarse patterns).

**Routes**

| § | question | verdict |
|---|---|---|
| 1–9 | Does the analytic ideal encoder work in the Hopfield probe? | Yes at r = 16 (also 8, 32); r ≤ 4 fails |
| 10 | Recall dynamics, saturation, projection storage | Projection storage gives exact fixed points; saturation snaps; binary codes cannot walk |
| 11 | Train IdealNet's integer harmonics by kernel MSE (Stage A) | Exact start stays; random starts only 55–62% integral |
| 12 | The objective's optimum over amplitudes | Within-window MSE is degenerate; MSE minus rate gives a good dense code |
| 13, 13.1 | Add the coding-rate term | Fails: rate rewards seams, and a seamed code beats the ideal under MSE + rate |
| 14 | Fixed integer table plus learned amplitudes | Proposed; a top-512 refit works on the whole arena, fails in regions |
| 15 | Translation-invariance penalty | Fixes the loss ranking; random starts snap to the wrong integers |
| 16, 16.1 | Projection storage in the probe; dimension against kernel | Rescues r = 48 for navigation; more dimensions cannot fix kernel overlap, the kernel shape can (open) |
| 17, 17.1 | Training dynamics (delay, noise, uniform-Δ pairs); kick test | Best gap 0.019, still worse; the ideal sits in a separate basin |
| 18 | 16 conditions: encoder × saturation × storage × readout | Gradient readout navigates everywhere; only unsaturated projection walks in and stops; Hebbian recall stops on a cue-independent state |
| 19 | Random projection + threshold + learned readout or slow feature analysis | Fails (memorises; slowest features alias ≥ 0.92) |
| 20 | Grid wave network | Works; requirements above |
| 20.1 | Is the gain local and incremental? | Yes; a few steps in one 10-cell box suffice; depression rule r = √(2κT) |
| 20.2 | Pruning instead of weighting | Stochastic pruning at keep-probability ∝ P(n) = the ideal's construction |
| 21 | Can L1 be learned locally and online, inside one environment? | Yes in one 14-box: online shift rule + eigenvectors = exact; fully online lock-in exact per unit but uneven counts alias; noise doubles the shift learners' alias and breaks lock-in; side 13 fails (wrap step unseen) |
| 21.1 | More boxes, longer walks; L1 and L3 both local in the probe | Fully online network (lock-in L1 + depression L3, one 14-box) navigates like the ideal; more boxes barely help the lock-in counts (a strong-pattern bias); under noise a slow rate over 1M steps brings the shift rule to its twin, lock-in still fails |
| 22 | Fully online and noise-robust: concurrent L1/L3, noise scaling, noise-trained networks, test-time noise | Concurrent works with a late L3 window (basin 29–30); under noise steps matter most and the best rate falls with steps; the noise-trained shift-rule network navigates like exact; the fully online one navigates with 9° error and some dead goals; the network shrugs off test noise 0.01, the ideal does not |

**Open**

- **Next: lateral competition among lock-in L1 units.** Units whose outputs
  match (the same pattern) inhibit each other, so about one unit per pattern
  stays active: 434 units instead of 2,604, and no uneven-count aliasing
  (0.27–0.34 now; the one-per-pattern reference gives 0.07–0.10). An
  anti-Hebbian lateral rule (Földiák) is local. Uneven counts currently cost
  about 4 cells of probe basin, not retrieval (§21.1).
- The fully online network under noise: 9° direction error and up to 17%
  dead goals after 1M steps at η 0.02 (§22.3); a longer walk at a falling
  rate is the expected fix (alias 0.42–0.51 at 3M steps, η 0.01, §22.2), not
  yet probed.
- A self-timed L3 critical period that matches the late window: units born
  before their inputs are fully pure are over-depressed (§22.1).
- Why a single box of side 20 leaves module 13 ten times noisier than boxes of
  side 14, 40 or 80 (§22.2).
- The eigenvector step of route 1 as a network (§21).
- Stochastic pruning simulated as an online process (§20.2).
- Kernel shape as a lever for wide kernels, r ≥ 48 (§16.1).
- Whether real neurons multiply accurately enough (a few percent).

**Where things are**

- Encoders and probe loader specs (`analysis/hopfield_probe/`):
  - `ideal_encoder.py` (`ideal:r=…`);
  - `ideal_net.py` (`idealnet:`);
  - `rp_encoder.py` (`rpnet:`, §19);
  - `gwn_encoder.py` (`gwn:`, §20).

  All load through `harness.load_probe_encoder`.
- Checks and launchers:
  - §10: `ideal_dynamics_check.py`.
  - §18: `grid16_check.py`, `grid16_summary.py`, `run_grid16.sh`.
  - §19: `rp_readout_check.py`, `rp_sfa_check.py`, `rp_summary.py`,
    `run_rp.sh`.
  - §20–22 (the grid wave network): see the table below.
  - Probe: `run_ideal.sh` (`STORAGE=proj`); the grid wave network's probe
    tasks are in `run_gwn.sh probe`.
- Training (§11–17): `encoder_training/train_ideal_net.py`, plus
  `ideal_net_*` diagnostics.
- Tests: `hopfield_nav/tests/test_ideal_encoder.py`, `test_rp_encoder.py`,
  `test_gwn_encoder.py`, `test_gwn_l1_online.py`, `test_gwn_online.py`,
  `test_probe_noise.py`.
- Results, under `/orcd/pool/003/jackking/cls_runs/results/`:
  - `hopfield_probe/ideal_encoder/` (probe, probe_proj, scan, dynamics,
    grid16);
  - `hopfield_probe/rp_readout/`;
  - `hopfield_probe/gwn/` (`summary.md` with every probe row, step JSON,
    encoders, `probe/tN`; `l1_online/` for §21–21.1; `online/` for §22);
  - `ideal_net/` (training runs).
- Conventions:
  - Probe settings are those of `run_ideal.sh`, with Hebbian storage unless
    stated. The ideal r = 16 reference rows are `ideal_encoder/probe/t12`–`t15`.
  - A learned L1 is judged against its **exact twin** (same triples, exact
    pairs): absolute alias varies 0.03–0.08 with the triple sample (§21).
  - L3 is the depression rule by default (Jack's preference, §22).
  - Learning tests use many scattered environments of side 10–100 covering
    1–20% of the scaffold, and report unvisited cells separately.

**Grid wave network scripts (§20–22)**, all in `analysis/hopfield_probe/`,
launched by `run_gwn.sh <mode>`; results under `.../hopfield_probe/gwn/`.

| script | question | § | output |
|---|---|---|---|
| `gwn_encoder.py` | the network itself (`gwn:` loader spec) | 20 | |
| `gwn_check.py` | built and measured: units, gain rules, multiplication, noise | 20 | `step0/2/3.json`, `step2b/3b.json` |
| `gwn_l1_learn.py` | L1 offline: Hebbian PCA vs movement (shift eigenvectors) | 20 | `step1.json` |
| `gwn_local_gain.py` | L3 local: trace and depression rules in one box | 20.1 | `local_gain.json` |
| `gwn_prune_vs_grade.py` | stochastic pruning vs graded gains | 20.2 | stdout |
| `gwn_summary.py` | tables for §20 and every probe row | 20–22 | `summary.md` |
| `gwn_l1_online.py` | L1 in one box: routes 1a–1c, 2a–2e (16-task array) | 21 | `l1_online/side*.json`, `summary.md` |
| `gwn_l1_twin.py` | learned L1 vs its exact twin; triple-seed spread | 21 | `l1_online/twin.json` |
| `gwn_l1_lms.py` | learning rate of the shift rule 1c | 21 | `l1_online/lms_rate.json` |
| `gwn_l1_noisefix.py` | noise-bias correction of the shift estimate | 21 | `l1_online/noisefix.json` |
| `gwn_l1_decay.py` | 1c with a per-synapse rate 1/(1 + n) | 21 | `l1_online/decay.json` |
| `gwn_l1_lockin2.py` | lock-in turn fixes: slow turn, rule 2e | 21 | `l1_online/lockin2.json` |
| `gwn_l1_mult.py` | uneven units per pattern (simulated counts) | 21 | `l1_online/mult.json` |
| `gwn_l1_multienv.py` | 6 boxes or one 40-box, 1M steps | 21.1 | `l1_online/multienv.json` |
| `gwn_l1_noise_slow.py` | slow rates plus correction under noise | 21.1 | `l1_online/noise_slow.json` |
| `gwn_l1l3_local.py` | L1 and L3 both local, L3 after L1 | 21.1 | `l1_online/l1l3_local.json`, `encoders/local_*` |
| `gwn_online.py` | streaming learners and online L2/L3 (a module) | 22 | |
| `gwn_noise_scaling.py` | noise robustness vs steps, environments, box size | 22.2 | `online/scaling_*.json`, `scaling_summary.md` |
| `gwn_concurrent.py` | L1 and L3 together; noisy-trained networks | 22.1, 22.3 | `online/concurrent.json`, `online/encoders/conc_*` |

Probe tasks (`run_gwn.sh probe`, rows in `gwn/summary.md`): 0–25 §20, 26–31
§21, 32–37 §21.1, 38–45 §22.4 (test-time noise), 46 on §22.1/22.3.

## First result: the ideal encoder in the probe (2026-09-30)

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
2. **The within-patch MSE does not match the kernel; it matches the kernel inside
   one patch window.** It only sees pairs inside a 50×50 patch, i.e. displacements
   in a 99×99 window (up to 71 cells), so it constrains the kernel there and says
   nothing beyond. Many codes fit the window equally well: over 99 cells the
   low-frequency waves have barely turned and look alike, so different amplitude
   combinations give nearly the same window kernel but completely different far
   kernels. Exponentiated gradient from equal weights at λ = 0 lands on a dense,
   good one (470 waves, far sd 0.038); NNLS lands on the 34-wave one that rings
   outside the window (far sd 0.19). By §16.1, a loss that matched the kernel at
   **all** displacements would have a unique, good optimum; that would need
   cross-patch pairs, which `exclude_cross_env_pairs` forbids. The rate term is the
   in-constraint substitute: it constrains the far field on average (the batch's
   spread), not pair by pair, which breaks the tie toward the dense code but also
   rewards seams (§13).
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

### 13.1 Paired loss check: the rate term prefers the seamed code (2026-10-01)

`encoder_training/ideal_net_loss_compare.py`: four harmonic layers evaluated on
the same 40 training batches (recipe patches, seed 0):

| network | integral | MSE | rate | inv | MSE + 0.5·rate | + 300·inv |
|---|---|---|---|---|---|---|
| exact ideal | 1.00 | 0.00049 | −0.2531 | ≈ 0 | −0.1261 | −0.1261 |
| seamed (§13, rate from the ideal) | 0.50 | 0.00482 | −0.2789 | 2.8e-4 | **−0.1346** | −0.0518 |
| near-ideal (§15a, rate + inv 300) | 0.91 | 0.00046 | −0.2549 | 1.9e-6 | −0.1270 | −0.1264 |
| random start (§13, rate) | 0.48 | 0.110 | −0.3142 | 3.0e-4 | −0.0475 | +0.0413 |

- **Under MSE + rate the seamed code scores 0.0086 below the ideal** (paired
  s.e. < 1e-4) while being a worse code (MSE 10×, C(1) 0.90). A flaw of the loss,
  not of SGD. (An earlier estimate put the ideal at −0.132; measured −0.126.)
- **Adding 300 × invariance reverses it:** ideal −0.126 vs seamed −0.052.
- **The exact ideal is still not that loss's minimum:** the 91%-integral
  near-ideal scores slightly lower (−0.1264).
- Random starts are far worse under every version, so they remain an SGD failure.

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

## 17. Different training dynamics, same loss (2026-10-01)

Loss fixed at MSE (r = 16) + 0.5 × rate + 300 × invariance; random starts in
[−6, 6]; 3 seeds each. New trainer options: `--inv_delay_epochs` (invariance off,
then ramped over `--inv_ramp_epochs`), `--noise_std` (Gaussian noise on the
harmonic weights after each step, decaying linearly to 0), and
`--pair_sampling uniform_delta` (displacements drawn uniformly from the 99×99
window, 16 anchors each, ~65k pairs per step; checked in-loop that each pair lies
in one patch at the drawn displacement). Schedule for all: 250 epochs off, 250
ramp, 250 full (750 epochs). Jobs 24570517–25 (`inv_d/`) and 24573525–42
(`inv_e/`).

Paired loss on the same 40 batch-sampled pairs (`ideal_net_loss_compare.py`,
seed 0 of each), plus each run's own diagnostics (3 seeds):

| network | total loss | gap to ideal | C(1) | r_half |
|---|---|---|---|---|
| exact ideal | −0.126 | — | 0.998 | 19 |
| near-ideal (from the ideal, §15a) | −0.1264 | −0.001 | 0.997 | 20 |
| delayed invariance, no noise | ≈ −0.035 | ≈ +0.09 | 0.51–0.54 | 6–8 |
| batch, noise 0.01 | ≈ −0.047 | ≈ +0.08 | 0.58–0.59 | 9–10 |
| batch, noise 0.03 | −0.093 | +0.033 | 0.75–0.78 | 12–14 |
| **uniform-Δ, noise 0.03** | **−0.107** | **+0.019** | 0.60–0.61 | 8–10 |
| batch, noise 0.06 | −0.082 | +0.044 | 0.50–0.53 | 1–2 |
| uniform-Δ, noise 0.06 | – | – | 0.35–0.37 | 1 |
| batch or uniform-Δ, noise 0.1 | ≈ +0.02 | – | 0.03–0.09 | 1 |
| batch + near weighting, noise 0.03 | −0.014 | +0.112 | 0.95 | 8 |

(Rows without a paired loss give the total from each run's own logged
components, MSE + 0.5·rate, under its own sampling.)

**Reading.**
1. **Uniform-Δ sampling + noise 0.03 is the best random start so far.** It cuts the
   remaining gap to the ideal by ~40% (0.033 → 0.019), consistently across seeds,
   but the code is still not the ideal (C(1) 0.6, r_half 9 vs 20).
2. **More noise hurts:** σ = 0.03 is the sweet spot; at 0.06 the code roughens and
   at 0.1 it is hash-like under either sampling.
3. **Near weighting fits the near field (C(1) 0.95) but not the shoulder**
   (r_half 8), and is the worst on the actual loss. Each change captures a
   different part of the kernel; none gets all of it.

No training-dynamics change so far brings a random start to the ideal.

### 17.1 Kick test: these are local minima, not slow convergence

`train_ideal_net.py --init_ckpt --kick K` (uniform ±K added to every weight, then
200 epochs under MSE + 0.5 rate + 300 inv at full strength, no noise; jobs
24590500–07, `.../ideal_net/kick/`). Bases: the best random start (uniform-Δ,
noise 0.03, s0; retrained with uniform-Δ sampling) and the near-ideal (§15a;
batch sampling). Losses are each run's own sampling, so compare within a base.

| base | kick | rows back on original integers | integral | loss after 200 epochs |
|---|---|---|---|---|
| best random | 0 / 0.2 / 0.5 / 1.0 | 1.00 / **1.00** / 0.59 / 0.02 | 0.93 / 0.94 / 0.93 / 0.87 | −0.1340 / −0.1339 / −0.1272 / −0.1058 |
| near-ideal | 0 / 0.2 / 0.5 / 1.0 | 1.00 / **1.00** / 0.47 / 0.04 | 0.90 / 0.91 / 0.88 / 0.66 | −0.1265 / −0.1265 / −0.1130 / −0.0673 |

A ±0.2 kick is undone exactly (same integers, same loss); larger kicks move rows
into other integer basins and the loss always gets worse, never better. Both
solutions sit at the bottom of their basins; the near-ideal is a better local
minimum, reached only by starting there. Training is not failing to converge:
each row settles in the integer basin nearest where it is when invariance takes
over. Reaching the ideal from a random start is a basin-selection (search)
problem, not an optimisation-speed problem.

## 18. The 16-condition grid: encoder × saturation × storage × readout (2026-10-06)

**Readout (b) navigates in every condition. Only projection storage gives a slow
walk that ends on the goal.**

- **Saturation leaves the memory exact but breaks readout (a).** Every stored
  goal is a correct fixed point, and the ideal code's basin grows from ~49 to
  ~57 cells. But readout (a) falls to acc45 0.39–0.44 and reach 0.10–0.14, the
  §7.1 offset, which the binary ideal code shows as well. Readout (b) brings it
  back to acc45 0.97–1.00 and reach 0.93–0.98.
- **A saturated memory cannot walk at any α.** At α ≤ 0.01 it stalls on the cue
  and then jumps to the goal.
- **An unsaturated code walks (interpolates) at K = 5 under both storage rules,
  but only projection storage keeps the goal as the fixed point.**
  - Under Hebbian storage the goal is not a fixed point. Self-recall at α = 1
    does stop, but on a cue-independent state (linear recall is power
    iteration, so it lands on the top eigenvector): cos 0.36–0.44 to the goal,
    mostly decoding to another env.
  - At K = 20 the Hebbian walk's basin (ii) is ~2 cells, and after-walk
    navigation fails for att0.5.
- **Recall-only saturation (rsat) has fixed points, but slightly wrong ones.**
  Each stored goal settles about 1 cell from its own cell.
  - Ideal code: 20–25% of goals correct, cos 0.90 to the goal; reach 0.6–0.7.
  - att0.5: 93–97% of goals correct, and navigation is unaffected.

### Conditions

- **Encoder.** Ideal r = 32 (`ideal:r=32,n_freq=512,seed=0,gain=100`), and
  trained att0.5 s42 (checkpoint config: gain 100, fwhm 0.25, tanh output,
  attract 0.5, batch 4096, exclude_cross_env_pairs).
- **Saturation.**
  - *unsat* (production): encoder gain 100, recall β = 100. The ideal code has
    no tanh.
  - *sat*: the encoder output is binarised AND the recall is saturated. For
    att0.5 this is arm B (gain = β = 1e6). For the ideal code it is the new
    `binary=1` (z → sign(z)/√D) with β = 1e6.
  - Side row *rsat*: recall-only saturation, β = 1e6 with the encoder
    unsaturated.
  - Note on the binary ideal kernel: it is mean_i (2/π)·arcsin(cos ω_i·Δ),
    linear in |Δ| at 0. It is not exactly (2/π)·arcsin(k); the two are tested in
    `test_ideal_encoder.py`.
  - The binary ideal code is **not injective**: 2.5% of cells share their exact
    sign pattern with a cell within 2 cells (3000-cell check).
- **Storage.** `hebb` and `proj`.
- **Readout.**
  - (a) the production q = basis·(recalled − current).
  - (b) the THEORY §7.1 (iii-c) central difference of s(p) = ⟨ẑ, z(p)⟩ over the
    four neighbours. This is the readout of `potential_readout_check.py`; there
    is no implementation in the nav/agenthash code.
- **Timing.** One recall step at α = 1, and the converged walk at the
  condition's best α (up to 600 steps; stopped once every cue moves less than
  1e-6 per step).
- **Layout.** multi_env_goals, 8 worlds × 20 envs, env 20. Whole arena, and the
  region `0 0 500`. K = 5 and K = 20.

### Metrics

- **α sweep.** α ∈ {1, 0.95, 0.9, 0.8, 0.5, 0.2, 0.05, 0.01, 0.003, 0.001},
  2 worlds × 5 envs, every non-goal cell as a cue (start ~10.5 cells out). Each
  step is decoded against the K envs' cells plus 5000 alias cells.
  - *Interpolates* requires all of:
    - ≥ 90% of cues monotone on the inbound segment (start to first closest
      approach);
    - mean per-cue inbound min cos ≥ 0.95, which is the "(min cos)" column;
    - median largest single step ≤ 50% of the path;
    - mean closest approach ≤ 1 cell.
  - *Snap* means a stall, then a single step covering ≥ 80% of the path at
    step ≥ 2. The dip cos is printed beside it.
- **Best α.**
  - If some α interpolates: among those, the most final states exactly on the
    goal, then the higher min cos, then the larger α.
  - Otherwise, the α **closest to interpolating**, tagged "(none interp.)": the
    least snap (smallest median, then mean, largest-step fraction), then the
    most states exactly on the goal, then min cos. Only α whose walk gets in are
    candidates.
  - α = 1 now wins only if it genuinely jumps least. In the first version it won
    every saturated row, because a one-step jump has inbound min cos 1.000.
- **Fixed point** (`self_fixed_point`). Self-recall from each stored goal
  (8 worlds × K), at α = 1 and at the best α, up to 600 steps.
  - *Goal is fixed pt*: y only if, for every goal, the state stops (cos of the
    last two states > 0.99999) AND decodes exactly to the goal's own cell (cell
    bank). "The state stops" alone is not reported as a fixed point: under
    Hebbian storage it always stops, on a cue-independent state.
  - *Settles at*: where self-recall ends, over the goals that are not fixed
    points: "X cells off" (mean decoded distance, same env), "N% other env"
    (fraction of all goals decoding outside the goal's env), "N% still moving",
    and the cos to the goal's code. The share of goals that are fixed points is
    in its own column.
  - *Near-goal exact* is the second line of evidence: starts within 2 cells,
    converged walk, fraction ending exactly on the goal (8 goals, disc bank).
- **Basin (ii).** Every scaffold cell within 64 of the goal is a cue (8 goals),
  run through the converged walk; first-failure radius at 100% / 95%.
- **Navigation.** Each cell reads acc45 / |err|° / reach / basin (iii).
  - acc45, |err| and reach (`continuous_flow`, arrival within 0.5 cell) come from
    the cells of 8 worlds × 5 envs.
  - Basin (iii) is the 100% first-failure radius of "reached", flowing on an
    81 × 81 scaffold box around each of 8 goals (boxes are now taken from all
    worlds, so both layouts have 8).
- **Caps.** "≥" marks a mean in which at least one goal hit the cap (64 for
  basin ii, 40 for basin iii), so it is a lower bound. A radius of −1 (the goal
  cell itself fails) pulls the mean down.

### Results

Run: `run_grid16.sh` with OUT=`.../grid16/v2` (Slurm 25093041, 48 tasks, all
completed). Tables: `python -m analysis.hopfield_probe.grid16_summary OUT [--doc]`.
`summary.md` holds every α sweep. The first run is kept in `.../grid16/json`.

Controls reproduce:
- Arm B with readout (a) gives 0.389 / 0.10 (published 0.392 / 0.103).
- Arm B with readout (b) gives 0.998 / 11.8° (Stage 0: 0.998 / 11.8°).
- Ideal unsat, readout (a), one step gives |err| 0.6–0.8° (§16: 0.59–0.87).

#### K = 5, whole: memory

| encoder · sat · storage | best α | interpolates (min cos) | snaps at α (dip cos) | goal is fixed pt α=1 | settles at α=1 | goal is fixed pt best α | settles at best α | goals correct α=1 / best | near-goal exact | basin ii 100/95 |
|---|---|---|---|---|---|---|---|---|---|---|
| ideal · unsat · hebb | 0.2 | y (0.984) | none | n | 2.1 cells off, 62% other env, cos 0.412 | n | 1.5 cells off, cos 0.768 | 0.00 / 0.28 | 0.39 | 15.2 / ≥18.1 |
| ideal · unsat · proj | 0.9 | y (0.999) | none | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 49.4 / ≥56.8 |
| att0.5 · unsat · hebb | 0.05 | y (0.961) | none | n | 1.6 cells off, 68% other env, cos 0.364 | n | 1.0 cells off, cos 0.954 | 0.12 / 0.97 | 0.99 | 13.0 / 16.4 |
| att0.5 · unsat · proj | 0.9 | y (0.969) | none | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 24.9 / 28.9 |
| ideal · sat · hebb | 0.01 (none interp.) | n (0.967) | 0.01 (0.967), 0.003 (0.959), 0.001 (0.958) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 56.8 / ≥63.8 |
| ideal · sat · proj | 0.01 (none interp.) | n (0.967) | 0.01 (0.967), 0.003 (0.959), 0.001 (0.958) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 57.1 / ≥63.8 |
| att0.5 · sat · hebb | 0.001 (none interp.) | n (0.902) | 0.01 (0.919), 0.003 (0.904), 0.001 (0.902) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 25.5 / 30.2 |
| att0.5 · sat · proj | 0.001 (none interp.) | n (0.902) | 0.01 (0.919), 0.003 (0.904), 0.001 (0.902) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 25.5 / 30.2 |
| ideal · rsat · hebb | 0.001 (none interp.) | n (0.922) | none | n | 1.2 cells off, cos 0.902 | n | 1.2 cells off, cos 0.903 | 0.25 / 0.25 | 0.12 | ≥7.1 / ≥7.1 |
| ideal · rsat · proj | 0.001 (none interp.) | n (0.924) | none | n | 1.1 cells off, cos 0.906 | n | 1.1 cells off, cos 0.906 | 0.20 / 0.20 | 0.12 | ≥7.1 / ≥7.1 |
| att0.5 · rsat · hebb | 0.001 (none interp.) | n (0.946) | none | n | 1.0 cells off, cos 0.941 | n | 1.0 cells off, cos 0.941 | 0.97 / 0.97 | 1.00 | 26.9 / 31.4 |
| att0.5 · rsat · proj | 0.001 (none interp.) | n (0.946) | none | n | 1.0 cells off, cos 0.947 | n | 1.0 cells off, cos 0.947 | 0.93 / 0.93 | 1.00 | 26.9 / 31.4 |

#### K = 5, whole: navigation

| encoder · sat · storage | best α | (a) q, 1 step | (a) q, after walk | (b) grad, 1 step | (b) grad, after walk |
|---|---|---|---|---|---|
| ideal · unsat · hebb | 0.2 | 1.000 / 0.8 / 0.99 / ≥31.0 | 0.990 / 6.4 / 0.55 / ≥25.0 | 1.000 / 1.4 / 0.99 / ≥40.0 | 0.990 / 6.4 / 0.58 / ≥25.0 |
| ideal · unsat · proj | 0.9 | 1.000 / 0.6 / 0.99 / ≥40.0 | 1.000 / 0.6 / 0.99 / ≥40.0 | 1.000 / 1.2 / 0.98 / ≥35.9 | 1.000 / 1.2 / 0.98 / ≥40.0 |
| att0.5 · unsat · hebb | 0.05 | 0.997 / 8.7 / 0.99 / 27.9 | 0.997 / 8.9 / 0.99 / 27.8 | 1.000 / 8.3 / 0.97 / 27.8 | 1.000 / 8.5 / 0.99 / 27.6 |
| att0.5 · unsat · proj | 0.9 | 0.998 / 8.6 / 0.99 / 27.9 | 0.998 / 8.6 / 0.99 / 27.9 | 1.000 / 8.2 / 0.96 / 27.8 | 1.000 / 8.2 / 0.96 / 27.8 |
| ideal · sat · hebb | 0.01 (none interp.) | 0.413 / 61.4 / 0.12 / 0.0 | 0.444 / 58.0 / 0.14 / 0.0 | 0.966 / 15.2 / 0.97 / ≥40.0 | 0.966 / 15.2 / 0.97 / ≥40.0 |
| ideal · sat · proj | 0.01 (none interp.) | 0.413 / 61.4 / 0.12 / 0.0 | 0.444 / 58.0 / 0.14 / 0.0 | 0.967 / 15.2 / 0.97 / ≥40.0 | 0.967 / 15.2 / 0.97 / ≥40.0 |
| att0.5 · sat · hebb | 0.001 (none interp.) | 0.389 / 66.5 / 0.10 / 0.0 | 0.389 / 66.5 / 0.11 / 0.0 | 0.998 / 11.8 / 0.98 / 25.5 | 0.998 / 11.8 / 0.98 / 25.1 |
| att0.5 · sat · proj | 0.001 (none interp.) | 0.389 / 66.5 / 0.10 / 0.0 | 0.389 / 66.5 / 0.11 / 0.0 | 0.998 / 11.8 / 0.98 / 25.5 | 0.998 / 11.8 / 0.98 / 25.2 |
| ideal · rsat · hebb | 0.001 (none interp.) | 0.995 / 4.5 / 0.72 / ≥25.0 | 0.994 / 4.7 / 0.68 / ≥25.0 | 0.994 / 4.5 / 0.72 / ≥25.0 | 0.995 / 4.7 / 0.66 / ≥25.0 |
| ideal · rsat · proj | 0.001 (none interp.) | 0.995 / 4.5 / 0.74 / ≥30.0 | 0.995 / 4.5 / 0.66 / ≥19.9 | 0.995 / 4.5 / 0.72 / ≥30.0 | 0.995 / 4.5 / 0.69 / ≥20.0 |
| att0.5 · rsat · hebb | 0.001 (none interp.) | 0.998 / 8.9 / 0.99 / 26.9 | 0.998 / 8.8 / 0.99 / 26.5 | 1.000 / 8.5 / 0.98 / 26.8 | 1.000 / 8.5 / 0.98 / 26.6 |
| att0.5 · rsat · proj | 0.001 (none interp.) | 0.998 / 8.9 / 0.99 / 26.9 | 0.998 / 8.9 / 0.98 / 26.4 | 1.000 / 8.5 / 0.99 / 26.6 | 1.000 / 8.5 / 0.98 / 26.4 |

#### K = 5, region: memory

| encoder · sat · storage | best α | interpolates (min cos) | snaps at α (dip cos) | goal is fixed pt α=1 | settles at α=1 | goal is fixed pt best α | settles at best α | goals correct α=1 / best | near-goal exact | basin ii 100/95 |
|---|---|---|---|---|---|---|---|---|---|---|
| ideal · unsat · hebb | 0.05 | y (0.976) | none | n | 2.2 cells off, 60% other env, cos 0.437 | n | 1.0 cells off, cos 0.954 | 0.05 / 0.85 | 0.88 | 4.8 / 5.9 |
| ideal · unsat · proj | 0.9 | y (0.999) | none | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 46.6 / 53.9 |
| att0.5 · unsat · hebb | 0.9 | y (0.956) | none | n | 1.6 cells off, 75% other env, cos 0.416 | n | 1.6 cells off, 75% other env, cos 0.416 | 0.03 / 0.03 | 0.00 | -1.0 / -1.0 |
| att0.5 · unsat · proj | 0.9 | y (0.976) | none | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 24.9 / 30.9 |
| ideal · sat · hebb | 0.01 (none interp.) | n (0.968) | 0.01 (0.968), 0.003 (0.960), 0.001 (0.959) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 49.4 / ≥58.5 |
| ideal · sat · proj | 0.01 (none interp.) | n (0.968) | 0.01 (0.968), 0.003 (0.960), 0.001 (0.959) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 49.8 / ≥58.6 |
| att0.5 · sat · hebb | 0.003 (none interp.) | n (0.915) | 0.01 (0.929), 0.003 (0.915), 0.001 (0.914) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 28.6 / 37.9 |
| att0.5 · sat · proj | 0.001 (none interp.) | n (0.914) | 0.01 (0.929), 0.003 (0.915), 0.001 (0.914) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 29.2 / 38.5 |
| ideal · rsat · hebb | 0.001 (none interp.) | n (0.919) | none | n | 1.2 cells off, cos 0.899 | n | 1.2 cells off, cos 0.898 | 0.20 / 0.17 | 0.25 | 13.5 / ≥15.2 |
| ideal · rsat · proj | 0.001 (none interp.) | n (0.920) | none | n | 1.3 cells off, cos 0.904 | n | 1.3 cells off, cos 0.905 | 0.35 / 0.33 | 0.38 | 20.1 / ≥22.6 |
| att0.5 · rsat · hebb | 0.01 | y (0.953) | none | n | 1.1 cells off, cos 0.945 | n | 1.1 cells off, cos 0.945 | 0.90 / 0.90 | 0.88 | 25.1 / 32.1 |
| att0.5 · rsat · proj | 0.01 | y (0.955) | none | n | 1.0 cells off, cos 0.957 | n | 1.0 cells off, cos 0.957 | 0.97 / 0.97 | 1.00 | 30.9 / 38.4 |

#### K = 5, region: navigation

| encoder · sat · storage | best α | (a) q, 1 step | (a) q, after walk | (b) grad, 1 step | (b) grad, after walk |
|---|---|---|---|---|---|
| ideal · unsat · hebb | 0.05 | 1.000 / 0.9 / 0.99 / ≥40.0 | 1.000 / 2.0 / 0.98 / ≥40.0 | 1.000 / 1.4 / 0.99 / ≥40.0 | 1.000 / 2.3 / 0.98 / ≥40.0 |
| ideal · unsat · proj | 0.9 | 1.000 / 0.6 / 0.99 / ≥40.0 | 1.000 / 0.6 / 0.99 / ≥36.1 | 1.000 / 1.3 / 0.99 / ≥40.0 | 1.000 / 1.3 / 0.97 / ≥40.0 |
| att0.5 · unsat · hebb | 0.9 | 1.000 / 8.1 / 0.98 / ≥34.8 | 0.749 / 38.4 / 0.16 / 4.1 | 0.999 / 8.3 / 0.97 / ≥35.1 | 0.811 / 30.9 / 0.24 / 9.0 |
| att0.5 · unsat · proj | 0.9 | 1.000 / 7.8 / 0.98 / ≥34.2 | 1.000 / 7.8 / 0.98 / ≥30.6 | 0.999 / 8.0 / 0.98 / ≥34.9 | 0.999 / 8.0 / 0.98 / ≥34.9 |
| ideal · sat · hebb | 0.01 (none interp.) | 0.417 / 61.1 / 0.11 / 0.0 | 0.446 / 58.2 / 0.13 / 0.0 | 0.977 / 14.8 / 0.93 / ≥40.0 | 0.977 / 14.8 / 0.93 / ≥40.0 |
| ideal · sat · proj | 0.01 (none interp.) | 0.417 / 61.1 / 0.11 / 0.0 | 0.445 / 58.2 / 0.13 / 0.0 | 0.977 / 14.8 / 0.93 / ≥40.0 | 0.977 / 14.8 / 0.93 / ≥40.0 |
| att0.5 · sat · hebb | 0.003 (none interp.) | 0.393 / 66.2 / 0.10 / 0.0 | 0.394 / 66.1 / 0.10 / 0.0 | 0.996 / 12.0 / 0.97 / 25.5 | 0.995 / 12.1 / 0.97 / 25.2 |
| att0.5 · sat · proj | 0.001 (none interp.) | 0.393 / 66.2 / 0.10 / 0.0 | 0.394 / 66.1 / 0.12 / 0.0 | 0.995 / 12.1 / 0.97 / 26.0 | 0.994 / 12.1 / 0.97 / 25.4 |
| ideal · rsat · hebb | 0.001 (none interp.) | 0.994 / 4.8 / 0.58 / ≥20.0 | 0.994 / 4.9 / 0.63 / ≥25.0 | 0.995 / 5.0 / 0.65 / ≥20.0 | 0.995 / 5.1 / 0.62 / ≥23.1 |
| ideal · rsat · proj | 0.001 (none interp.) | 0.994 / 4.7 / 0.64 / ≥20.0 | 0.994 / 5.0 / 0.64 / ≥25.0 | 0.995 / 4.9 / 0.65 / ≥21.6 | 0.994 / 5.2 / 0.63 / ≥20.0 |
| att0.5 · rsat · hebb | 0.01 | 0.999 / 8.7 / 0.97 / ≥33.0 | 0.998 / 8.8 / 0.97 / 26.4 | 0.998 / 8.8 / 0.96 / ≥33.8 | 0.998 / 8.9 / 0.96 / 27.2 |
| att0.5 · rsat · proj | 0.01 | 0.998 / 8.5 / 0.96 / ≥32.1 | 0.997 / 8.5 / 0.96 / 26.8 | 0.998 / 8.7 / 0.96 / ≥33.8 | 0.997 / 8.8 / 0.96 / 26.5 |

#### K = 20, whole: memory

| encoder · sat · storage | best α | interpolates (min cos) | goal is fixed pt α=1 | settles at α=1 | goal is fixed pt best α | settles at best α | goals correct α=1 / best | near-goal exact | basin ii 100/95 |
|---|---|---|---|---|---|---|---|---|---|
| ideal · unsat · hebb | 0.05 | y (0.951) | n | 4.2 cells off, 91% other env, cos 0.207 | n | 1.3 cells off, cos 0.912 | 0.00 / 0.54 | 0.50 | 1.6 / 2.0 |
| ideal · unsat · proj | 0.9 | y (0.998) | y | goal | y | goal | 1.00 / 1.00 | 0.97 | 17.6 / 21.2 |
| att0.5 · unsat · hebb | 0.5 (none interp.) | n (0.855) | n | 3.0 cells off, 92% other env, cos 0.181 | n | 3.0 cells off, 92% other env, cos 0.186 | 0.00 / 0.00 | 0.00 | -1.0 / -1.0 |
| att0.5 · unsat · proj | 0.9 | y (0.958) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 11.0 / 14.2 |
| ideal · sat · hebb | 0.01 (none interp.) | n (0.967) | n | 1.5 cells off, cos 1.000 | n | 1.5 cells off, cos 1.000 | 0.99 / 0.99 | 1.00 | 41.5 / ≥54.2 |
| ideal · sat · proj | 0.01 (none interp.) | n (0.967) | n | 1.5 cells off, cos 1.000 | n | 1.5 cells off, cos 1.000 | 0.99 / 0.99 | 1.00 | 42.0 / ≥56.8 |
| att0.5 · sat · hebb | 0.001 (none interp.) | n (0.898) | n | 2.0 cells off, cos 0.752 | n | 2.0 cells off, cos 0.752 | 0.99 / 0.99 | 1.00 | 17.8 / 21.8 |
| att0.5 · sat · proj | 0.001 (none interp.) | n (0.902) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 21.1 / 25.8 |
| ideal · rsat · hebb | 0.001 (none interp.) | n (0.918) | n | 1.7 cells off, 3% other env, cos 0.847 | n | 1% still moving, 1.7 cells off, cos 0.873 | 0.12 / 0.16 | 0.12 | 1.2 / 1.8 |
| ideal · rsat · proj | 0.001 (none interp.) | n (0.922) | n | 1.6 cells off, cos 0.897 | n | 1.6 cells off, cos 0.897 | 0.19 / 0.19 | 0.01 | -1.0 / -1.0 |
| att0.5 · rsat · hebb | 0.001 (none interp.) | n (0.932) | n | 1.7 cells off, 6% other env, cos 0.802 | n | 1% still moving, 1.5 cells off, 6% other env, cos 0.815 | 0.76 / 0.76 | 0.75 | 14.4 / 16.9 |
| att0.5 · rsat · proj | 0.001 (none interp.) | n (0.945) | n | 1.0 cells off, cos 0.948 | n | 1.0 cells off, cos 0.948 | 0.97 / 0.97 | 1.00 | 21.6 / 26.0 |

#### K = 20, whole: navigation

| encoder · sat · storage | best α | (a) q, 1 step | (a) q, after walk | (b) grad, 1 step | (b) grad, after walk |
|---|---|---|---|---|---|
| ideal · unsat · hebb | 0.05 | 0.996 / 1.7 / 0.97 / ≥40.0 | 0.994 / 3.9 / 0.91 / ≥40.0 | 0.997 / 2.2 / 0.97 / ≥40.0 | 0.994 / 4.1 / 0.88 / ≥40.0 |
| ideal · unsat · proj | 0.9 | 1.000 / 0.7 / 0.99 / ≥40.0 | 1.000 / 0.7 / 0.99 / ≥40.0 | 1.000 / 1.4 / 0.97 / ≥40.0 | 1.000 / 1.4 / 0.97 / ≥40.0 |
| att0.5 · unsat · hebb | 0.5 (none interp.) | 0.982 / 11.2 / 0.89 / 18.9 | 0.487 / 61.9 / 0.12 / 4.6 | 0.990 / 10.0 / 0.94 / 21.8 | 0.640 / 48.2 / 0.14 / 4.6 |
| att0.5 · unsat · proj | 0.9 | 0.989 / 9.6 / 1.00 / 25.1 | 0.989 / 9.6 / 0.99 / 25.1 | 0.993 / 9.0 / 0.96 / 25.5 | 0.993 / 9.0 / 0.96 / 25.5 |
| ideal · sat · hebb | 0.01 (none interp.) | 0.417 / 61.0 / 0.12 / 0.0 | 0.445 / 58.0 / 0.14 / 0.0 | 0.966 / 15.2 / 0.97 / ≥39.6 | 0.966 / 15.2 / 0.97 / ≥40.0 |
| ideal · sat · proj | 0.01 (none interp.) | 0.412 / 61.4 / 0.12 / 0.0 | 0.444 / 58.0 / 0.14 / 0.0 | 0.967 / 15.2 / 0.97 / ≥39.9 | 0.967 / 15.2 / 0.97 / ≥39.4 |
| att0.5 · sat · hebb | 0.001 (none interp.) | 0.384 / 67.2 / 0.10 / 0.0 | 0.387 / 66.9 / 0.11 / 0.0 | 0.987 / 12.8 / 0.98 / 23.4 | 0.984 / 13.1 / 0.97 / 19.9 |
| att0.5 · sat · proj | 0.001 (none interp.) | 0.387 / 66.8 / 0.10 / 0.0 | 0.389 / 66.6 / 0.11 / 0.0 | 0.988 / 12.7 / 0.98 / 21.8 | 0.990 / 12.7 / 0.98 / 21.8 |
| ideal · rsat · hebb | 0.001 (none interp.) | 0.992 / 5.0 / 0.77 / ≥35.0 | 0.983 / 7.2 / 0.40 / ≥20.0 | 0.992 / 5.1 / 0.76 / ≥30.0 | 0.983 / 7.3 / 0.44 / ≥20.0 |
| ideal · rsat · proj | 0.001 (none interp.) | 0.994 / 4.8 / 0.73 / ≥30.0 | 0.984 / 6.7 / 0.45 / ≥14.9 | 0.993 / 4.8 / 0.72 / ≥30.0 | 0.984 / 6.8 / 0.46 / ≥15.0 |
| att0.5 · rsat · hebb | 0.001 (none interp.) | 0.988 / 10.6 / 0.95 / 21.2 | 0.973 / 12.1 / 0.88 / 16.8 | 0.990 / 9.9 / 0.95 / 21.2 | 0.975 / 11.5 / 0.87 / 17.0 |
| att0.5 · rsat · proj | 0.001 (none interp.) | 0.991 / 9.6 / 0.98 / 24.6 | 0.991 / 9.8 / 0.97 / 23.9 | 0.993 / 9.2 / 0.98 / 24.6 | 0.991 / 9.4 / 0.98 / 21.9 |

#### K = 20, region: memory

| encoder · sat · storage | best α | interpolates (min cos) | goal is fixed pt α=1 | settles at α=1 | goal is fixed pt best α | settles at best α | goals correct α=1 / best | near-goal exact | basin ii 100/95 |
|---|---|---|---|---|---|---|---|---|---|
| ideal · unsat · hebb | 0.05 (none interp.) | n (0.938) | n | 2.8 cells off, 90% other env, cos 0.213 | n | 1.2 cells off, cos 0.905 | 0.02 / 0.51 | 0.66 | 1.5 / 2.0 |
| ideal · unsat · proj | 0.9 | y (0.998) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 20.0 / 23.4 |
| att0.5 · unsat · hebb | 0.5 (none interp.) | n (0.841) | n | 2.7 cells off, 93% other env, cos 0.218 | n | 1% still moving, 2.7 cells off, 93% other env, cos 0.224 | 0.00 / 0.00 | 0.00 | -1.0 / -1.0 |
| att0.5 · unsat · proj | 0.9 | y (0.970) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 13.0 / 17.1 |
| ideal · sat · hebb | 0.01 (none interp.) | n (0.968) | n | 2.2 cells off, cos 1.000 | n | 2.2 cells off, cos 1.000 | 0.99 / 0.99 | 1.00 | 48.2 / ≥57.5 |
| ideal · sat · proj | 0.01 (none interp.) | n (0.968) | n | 2.2 cells off, cos 1.000 | n | 2.2 cells off, cos 1.000 | 0.99 / 0.99 | 1.00 | 49.5 / ≥58.5 |
| att0.5 · sat · hebb | 0.001 (none interp.) | n (0.911) | n | 1% other env, cos 0.652 | n | 1% other env, cos 0.652 | 0.99 / 0.99 | 1.00 | 20.6 / 25.4 |
| att0.5 · sat · proj | 0.001 (none interp.) | n (0.914) | y | goal | y | goal | 1.00 / 1.00 | 1.00 | 25.4 / 31.9 |
| ideal · rsat · hebb | 0.001 (none interp.) | n (0.914) | n | 1.5 cells off, 1% other env, cos 0.870 | n | 1.5 cells off, 1% other env, cos 0.874 | 0.17 / 0.19 | 0.12 | -0.1 / 0.0 |
| ideal · rsat · proj | 0.001 (none interp.) | n (0.920) | n | 1.5 cells off, cos 0.898 | n | 1.5 cells off, cos 0.898 | 0.21 / 0.21 | 0.12 | -0.6 / -0.5 |
| att0.5 · rsat · hebb | 0.001 (none interp.) | n (0.937) | n | 1.5 cells off, 27% other env, cos 0.623 | n | 6% still moving, 1.6 cells off, 19% other env, cos 0.720 | 0.46 / 0.47 | 0.62 | 12.0 / 14.8 |
| att0.5 · rsat · proj | 0.01 | y (0.954) | n | 1.1 cells off, cos 0.954 | n | 1.1 cells off, cos 0.954 | 0.96 / 0.96 | 1.00 | 25.6 / 31.9 |

#### K = 20, region: navigation

| encoder · sat · storage | best α | (a) q, 1 step | (a) q, after walk | (b) grad, 1 step | (b) grad, after walk |
|---|---|---|---|---|---|
| ideal · unsat · hebb | 0.05 (none interp.) | 1.000 / 1.2 / 1.00 / ≥40.0 | 0.998 / 3.2 / 0.87 / ≥35.0 | 1.000 / 1.6 / 0.99 / ≥40.0 | 0.999 / 3.3 / 0.88 / ≥35.0 |
| ideal · unsat · proj | 0.9 | 1.000 / 0.5 / 1.00 / ≥40.0 | 1.000 / 0.5 / 1.00 / ≥35.6 | 1.000 / 1.2 / 0.97 / ≥40.0 | 1.000 / 1.2 / 0.98 / ≥35.4 |
| att0.5 · unsat · hebb | 0.5 (none interp.) | 0.996 / 9.2 / 0.99 / 33.6 | 0.328 / 78.4 / 0.04 / 0.0 | 0.999 / 9.1 / 0.98 / ≥34.6 | 0.447 / 66.9 / 0.03 / 0.2 |
| att0.5 · unsat · proj | 0.9 | 0.997 / 8.3 / 0.98 / ≥33.6 | 0.997 / 8.3 / 0.98 / ≥33.6 | 0.999 / 8.4 / 0.98 / 32.2 | 0.999 / 8.4 / 0.98 / 32.2 |
| ideal · sat · hebb | 0.01 (none interp.) | 0.421 / 60.8 / 0.11 / 0.0 | 0.446 / 58.2 / 0.13 / 0.0 | 0.977 / 14.8 / 0.93 / ≥39.0 | 0.977 / 14.8 / 0.93 / ≥39.4 |
| ideal · sat · proj | 0.01 (none interp.) | 0.417 / 61.0 / 0.11 / 0.0 | 0.445 / 58.2 / 0.13 / 0.0 | 0.977 / 14.8 / 0.93 / ≥39.2 | 0.977 / 14.8 / 0.93 / ≥40.0 |
| att0.5 · sat · hebb | 0.001 (none interp.) | 0.391 / 67.1 / 0.10 / 0.0 | 0.393 / 66.8 / 0.12 / 0.0 | 0.990 / 12.6 / 0.97 / 24.9 | 0.977 / 13.9 / 0.95 / 22.2 |
| att0.5 · sat · proj | 0.001 (none interp.) | 0.391 / 66.5 / 0.10 / 0.0 | 0.394 / 66.3 / 0.12 / 0.0 | 0.992 / 12.4 / 0.97 / 25.8 | 0.989 / 12.7 / 0.96 / 24.6 |
| ideal · rsat · hebb | 0.001 (none interp.) | 0.994 / 5.2 / 0.63 / ≥25.0 | 0.988 / 7.2 / 0.39 / ≥15.0 | 0.994 / 5.4 / 0.60 / ≥20.4 | 0.988 / 7.4 / 0.41 / ≥20.0 |
| ideal · rsat · proj | 0.001 (none interp.) | 0.994 / 4.8 / 0.64 / ≥20.0 | 0.987 / 7.2 / 0.45 / ≥20.0 | 0.994 / 4.9 / 0.65 / ≥25.0 | 0.986 / 7.3 / 0.41 / ≥20.0 |
| att0.5 · rsat · hebb | 0.001 (none interp.) | 0.996 / 9.1 / 0.94 / ≥31.2 | 0.958 / 12.8 / 0.85 / 20.2 | 0.999 / 9.1 / 0.95 / 33.1 | 0.968 / 12.1 / 0.89 / 19.4 |
| att0.5 · rsat · proj | 0.01 | 0.997 / 8.8 / 0.96 / 29.8 | 0.992 / 9.1 / 0.96 / 21.4 | 0.998 / 8.9 / 0.96 / 31.4 | 0.992 / 9.4 / 0.95 / 25.2 |

### Reading

1. **Saturation breaks the readout, not the memory.**
   - In every *sat* row at K = 5, every stored goal is a correct fixed point at
     both α values, near-goal starts end exactly on the goal, and the ideal
     basin (ii) is 57/≥64, against 49/≥57 unsaturated.
   - Readout (a) still fails, at acc45 0.39–0.44, on the ideal code as on arm B.
     So the §7.1 offset failure belongs to any binary code.
   - Readout (b) navigates: ideal 15° and basin (iii) ≥ 40; arm B 12° and ~25.
2. **No saturated memory interpolates.** Even the "closest" α (ideal 0.01,
   att0.5 0.001–0.003) sits on the cue's cell for 2–23 steps and then covers
   the whole path in one step. The cos dip is 0.96 for the ideal code and
   0.90–0.92 for arm B. Taking the readouts after that walk changes nothing
   material (readout (a) 0.41 → 0.44 on the ideal code).
3. **Under Hebbian storage the goals are not fixed points.**
   - Self-recall at α = 1 always stops, but on a cue-independent state (power
     iteration onto the top eigenvector): cos
     0.36–0.44 to the goal at K = 5 and 0.18–0.22 at K = 20, decoding mostly
     into another env.
   - At the best (small) α the walk stops earlier and nearer: 28% (ideal) and
     97% (att0.5) of goals correct at K = 5, whole arena.
   - Projection storage gets every goal exactly at both α. The walk then ends
     on the goal, and after-walk navigation equals one-step navigation.
   - Under Hebbian storage, after-walk navigation degrades where the drift is
     large: att0.5 region K = 5 reach 0.98 → 0.16, and K = 20 → 0.03–0.14.
4. **Recall-only saturation: right place, wrong exact cell.** Self-recall
   stops about 1 cell off the goal.
   - The ideal cos/sin code is far from any sign corner. Its stored patterns
     move to cos 0.90 and are correct for only 20–25% of goals, so reach drops
     to 0.6–0.7 even though direction stays good (acc45 0.99, ~5°).
   - att0.5's tanh code is already near-binary. 93–97% of goals stay on their
     cell, and navigation matches unsat.
5. **K = 20.**
   - Projection storage keeps every unsaturated goal exact; its basin (ii)
     shrinks (ideal 49 → 18, att0.5 25 → 11).
   - Saturated memories keep the largest basins (ideal 42–50, att0.5 18–25).
     Their few "wrong" fixed points are of two kinds:
     - Ideal: 1–2 of 160 goals sit at cos 1.000 to the goal's code, i.e. the
       memory is exact, but a cell 1–2 away has the same binary code (the
       non-injectivity above).
     - att0.5 · sat · hebb: 1 of 160 is a spurious state, cos 0.65–0.75. Under
       proj, none.
6. **Changes from the first version of this section.**
   - Best α for *sat* moved from 1 to 0.01 (ideal) or 0.001–0.003 (att0.5); for
     att0.5 *rsat* in the whole arena it moved from 0.2 to 0.001.
   - The old "fixed point y/n" (cos ≥ 0.999 to the stored pattern) is replaced
     by "goal is fixed pt" (stops AND decodes to the goal) plus "settles at"
     (where it ends instead). Hebbian rows stay "n"; "settles at" shows the
     cue-independent end state.
   - att0.5 *rsat*, previously "correct y" on 8 goals, is n: 1–3 of 40 goals end
     1 cell off.
   - Region basin (iii) now rests on 8 goals instead of 1.

## 19. A random projection, a threshold and a learned linear readout (2026-11-02)

**No tested random threshold layer gets to the ideal code.** A fixed random
layer on the grid code fits the ideal code only where it was trained, and
transfers nothing to held-out environments. Slow feature analysis on the same
layer finds each module's own period as its slowest feature, and that is the
most aliased function on offer. A positive control, the same pipeline on a
layer of pure waves, passes. So the failure belongs to the random layer, not
to the readout or the objective.

The question was whether a biologically plausible route exists: a fixed random
"hippocampal" layer (genetic), then a linear readout learned from experience.
The environments follow the ecological regime: squares of side 10–100
(log-uniform), dropped at random on the 1716 × 1716 scaffold until their union
covers 1%, 5% or 20% of it (13, ~68 and ~270 environments). Results are split
into visited cells, unvisited cells, and far cells (unvisited, more than 50
cells from any visited one).

### Setup

- **Layer** (`rp_encoder.py`): `h = act((g W - mu)/sd - tau)`, with `W ~ N(0, 1)`
  of shape 434 × H and H ∈ {2048, 8192, 32768}. `mu`, `sd` and `tau` are
  calibrated once on uniform positions. `tau` sets the activity (50%, or 10% for
  `hard10`).
- **Nonlinearities**: sigmoid of slope 1, 4 or 16 (`sig1`, `sig4`, `sig16`), and
  a hard step (`hard50`, `hard10`).
- **Bump widths**: the grid code at 0.25 (production), 0.15 and 0.10 of the
  period.
- **Readout**: `z = (h - hbar) A`, unit-normalised. The probe can load it as
  `rpnet:<path.pt>`.
- **Metrics**: C(1) and the half-width of the near-field profile; far-field
  spread; and "alias max", the cosine at the grid code's own 64 most similar
  far displacements, where every overlap-based code is most similar to itself.
  For the ideal r = 16 code these are C(1) 0.998, half-width 19, far sd 0.032
  and alias max 0.06. For the grid code itself, alias max is 0.95.

### Step 1: best-case readout (`rp_readout_check.py`)

A ridge regression of the ideal r = 16 code on the hidden layer, fitted on every
cell of 90% of the environments. The ridge is chosen on the other 10%, so it is
picked for transfer. 135 fits: 5 nonlinearities × 3 bump widths × 3 H × 3
coverages.

**The layer fits the code only by memorising cells.** In-sample R², best ridge,
H = 2048 / 8192 / 32768:

| bump | coverage | sig1 | sig4 | sig16 | hard50 | hard10 |
|---|---|---|---|---|---|---|
| 0.25 | 5% | 0.14 / 0.35 / 0.67 | 0.12 / 0.30 / 0.58 | 0.09 / 0.24 / 0.50 | 0.07 / 0.21 / 0.47 | 0.10 / 0.22 / 0.46 |
| 0.15 | 5% | 0.05 / 0.18 / 0.46 | 0.04 / 0.15 / 0.41 | 0.03 / 0.12 / 0.36 | 0.02 / 0.11 / 0.34 | 0.04 / 0.14 / 0.35 |
| 0.10 | 5% | 0.01 / 0.07 / 0.26 | 0.01 / 0.06 / 0.25 | 0.01 / 0.06 / 0.24 | 0.01 / 0.05 / 0.24 | 0.01 / 0.07 / 0.26 |
| 0.25 | 20% | 0.03 / 0.10 / 0.21 | 0.03 / 0.08 / 0.18 | 0.02 / 0.06 / 0.15 | 0.02 / 0.05 / 0.14 | 0.02 / 0.06 / 0.14 |
| 0.15 | 20% | 0.01 / 0.04 / 0.13 | 0.01 / 0.04 / 0.11 | 0.01 / 0.03 / 0.10 | 0.00 / 0.03 / 0.09 | 0.01 / 0.03 / 0.09 |
| 0.10 | 20% | 0.00 / 0.02 / 0.07 | 0.00 / 0.01 / 0.06 | 0.00 / 0.01 / 0.06 | 0.00 / 0.01 / 0.06 | 0.00 / 0.02 / 0.07 |

- The fit falls as the covered area grows and rises less than linearly with H
  (about H^0.66 at 20% coverage). At 1% coverage with H = 32768 (more units than
  fitted cells) every layer fits R² 0.958.
- Narrower bumps and harder thresholds fit *worse*. The prediction that they
  would help (they make the fine module harmonics available) was wrong. A
  sharper unit spreads its variance over more frequencies, so less of it lands
  in the slow band the ideal is made of.

**Nothing transfers.** In every fit, R² on held-out environments falls
monotonically as the ridge shrinks (e.g. bump 0.25, sig1, H = 32768, 5%: −0.12 at
the largest ridge, −1.34 at the smallest). Every fit therefore picks the largest
ridge, which predicts almost nothing. On far cells the cosine to the ideal code
is at most 0.00 over all 135 fits, and alias max never falls below 0.57.

The layer's own similarity is a function of grid-code overlap: high for nearby
cells and for grid aliases. So whatever a readout learns on visited cells
spreads only to their neighbours and to their aliases.

### Step 2: slowness readout (`rp_sfa_check.py`)

Exact slow feature analysis: the covariance of h over all visited cells, pooled
across environments, and the covariance of its one-step changes within each
environment; output i has unit variance and slowness λ_i. Two readouts, each
randomly projected to 1024 dims when wider:

- `top1024`: the 1024 slowest features.
- `gauss16`: every feature, weighted `exp(-λ_i 16² / 2)`, which is exactly the
  Gaussian kernel of width 16 when the features are waves.

**The random layer has no slow cross-module features.** A complete set of waves
has 3661 features slower than a wave of the r = 16 scale. Every random layer at
H = 8192 has none, at every coverage. Rows below are 5% coverage, far cells:

| layer | bump | slowest feature, as a wavelength | gauss16: C(1) / half-width / far sd / alias max | top1024: C(1) / half-width / far sd / alias max |
|---|---|---|---|---|
| sig1 | 0.25 | 31 cells | 0.976 / 34 / 0.46 / 0.98 | 0.790 / 1.9 / 0.034 / 0.85 |
| hard50 | 0.25 | 13 | 0.941 / 3.2 / 0.51 / 1.00 | 0.495 / 1.0 / 0.032 / 0.67 |
| sig1 | 0.15 | 13 | 0.942 / 3.2 / 0.51 / 1.00 | 0.530 / 1.1 / 0.032 / 0.74 |
| hard50 | 0.10 | 12 | 0.934 / 3.2 / 0.51 / 0.99 | 0.108 / 1.0 / 0.031 / 0.28 |
| *control: pure waves (r = 8), 20%* | — | — | 0.999 / 28 / 0.065 / 0.045 | 0.999 / 28 / 0.062 / 0.047 |

- The slowest feature is a wave of 11–18 cells in 14 of the 15 settings: each
  module's own period (11–13), a function of one module's phase. Such a function
  changes slowly along the sheet and is identical on neighbouring strands, so
  slowness prefers exactly the aliased functions. Only the smoothest layer at
  the widest bump reaches 31 cells, and its slow features still alias (0.98).
- The two readouts fail in opposite ways. `gauss16` keeps a near field but
  aliases at 0.92–1.00. `top1024` aliases less but has no near field (half-width
  about 1 cell).
- Visited and far cells give the same numbers. Slow features are global
  functions, so they transfer; they are simply the wrong functions.
- The control passes: a layer of pure waves gives an unaliased large-field code
  (alias 0.045, the ideal's level). The method works when the layer contains
  slow waves.

**A bigger layer helps the smooth layer a little, and it still aliases.** H
trend at 5% coverage, far cells (cuSOLVER's eigensolver refuses n = 32768 in
float64 and float32, and MAGMA segfaults there, so 24576 is the large point):

| layer | H | slowest feature, as a wavelength | slower than an r = 16 wave (of 3661) | gauss16: far sd / alias max |
|---|---|---|---|---|
| sig1, bump 0.25 | 2048 / 8192 / 24576 | 13 / 31 / 86 cells | 0 / 0 / 4 | 0.51 / 0.46 / 0.24 and 1.00 / 0.98 / 0.96 |
| hard50, bump 0.10 | 2048 / 8192 / 24576 | 10 / 12 / 13 cells | 0 / 0 / 0 | 0.56 / 0.51 / 0.51 and 0.97 / 0.99 / 1.00 |

The smooth layer's slow features get slower and its far field quieter as H
grows, but they still alias at 0.96. A smooth unit is smooth in every direction
of phase space, so its slow combinations are smooth across strands too: they
take nearly the same value on strands 0.82 cells apart. The ideal's waves need
the opposite, slow along the sheet and fast across it. The hard layer never
gets past the module period.

### Step 3: not run

Every learned code either aliases at ≥ 0.9 or has no near field. One far-field
alias is enough to kill reach (PROBE §10), so the Hopfield probe would only
restate that.

### Reading

- A random threshold layer dilutes the slow cross-module waves. Each unit puts
  a tiny share of its variance on any one of them. A linear readout can only
  assemble them by memorising cells (step 1), and slowness never finds them
  (step 2).
- The positive control says what the layer needs instead: units that are close
  to pure waves. In network terms, one unit would be a product of one harmonic
  of each module's phase, `cos(j₁φ₁ + j₂φ₂ + j₃φ₃)`. Per-module harmonics are
  linear readouts of a module's cells (amplitude 0.28 at j = 6 with 0.10 bumps),
  and the cross-module combination is multiplicative. Random integer `j` with
  slowness selecting among them is the untested next step. By a rough count
  (uniform integer frequencies against the r = 16 disc), about 1 in 800 random
  combinations is slow enough, and the control shows slowness then builds the
  right kernel.

Results: `/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/rp_readout/`
(`step1/`, `step2/`, `logs/`; `step1_v0/` is a first pass whose ridge was picked
on cells interleaved with the fitted ones; full tables in `step1_summary.md` and
`step2_summary.md`). Launcher: `run_rp.sh all`. Tables:
`python -m analysis.hopfield_probe.rp_summary step1|step2 DIR`.

## 20. The grid wave network (plan written 2026-11-02, before any run)

**The network.** L0 grid cells (434) → L1 per-module waves → L2 wave units →
L3 slowness gains → code. Page: "Grid Wave Network"
(claude.ai/artifact/AhAWfeS2AB546xayRMj8gj).

- **L1**: for module m and pattern j = (jx, jy) ∈ Z_λ², the pair
  c = Σ_ab cos(2π(jx a + jy b)/λ) g_ab / A_m(j) = cos θ, s = the same with sin.
  θ = 2π(jx x + jy y)/λ. Exact L1 holds all λ² patterns per module.
- **L2**: unit k picks one L1 pair per module at random (the only random
  wiring) and multiplies: u = c₁₁c₁₂ − s₁₁s₁₂, v = s₁₁c₁₂ + c₁₁s₁₂,
  C = u c₁₃ − v s₁₃, S = v c₁₃ + u s₁₃. So C = cos Θ, Θ = 2π n·p/1716, with
  n = 156 j₁₁ + 143 j₁₂ + 132 j₁₃ mod 1716 per axis.
- **L3**: each unit's slowness v_k = mean over walk steps of (ΔC)² + (ΔS)²
  (≈ |ω|²/2 for a pure wave, the same at every position). Gain w_k = g(v_k).
  The code is [w C, w S] over the units with w above 10⁻⁴, normalised.

**Regime throughout.** Walk steps are sampled inside scattered square
environments of side 10–100 (log-uniform) covering 5% of the scaffold (1% and
20% as a check). Target r = 16. Metrics as in §19 (C(1), half-width, far sd,
alias max at the grid code's 64 strongest far displacements) on visited,
unvisited and far cells. The probe is `run.py` with the `run_ideal.sh`
settings, Hebbian storage, against the existing ideal r = 16 rows
(`probe/t12`–`t15`).

**No gating.** Every step runs with exact L1, whether or not L1 can be learned.
Learned L1 variants are run alongside wherever they apply.

### Step 0: exact network
1. Units N_u ∈ {25k, 100k, 400k, all 1716² = 2.94M}, exact L1, exact products,
   Gaussian gains: survivors, kernel metrics, and the gain estimate's error
   against the analytic |ω|²/2.
2. Gain rules at N_u = 400k, each scaled to r = 16: Gaussian
   exp(−r²v/2); cut 1[v < 2/r²]; rational 1/(1 + r²v/2); linear
   max(0, 1 − r²v/4). Kernel metrics.
3. Coverage 1/5/20% for the walk steps (expected not to matter).
4. Probe: 400k Gaussian (4 regions); all-triples Gaussian, and 400k with cut,
   rational and linear gains (whole + corner).

### Step 1: learning L1 (bump widths 0.25 and 0.10)
1. **Hebbian PCA baseline**: per-module covariance from visited cells,
   eigenvectors paired in order as (c, s), plus a constant unit (the j = 0
   pattern). Purity = the share of a pair's power (as c + i s on the module
   torus) in one frequency; exact L1 = 1.
2. **Movement (offline)**: estimate per-module shift operators T_x, T_y
   from walk steps (ridge least squares g(p + e) ≈ T_e g(p)), take the left
   eigenvectors of T_x + α T_y (α = 0.618), and use their real and imaginary
   parts as (c, s), normalised to mean c² + s² = 1. Purity against the number
   of steps (1k, 10k, 100k) and bump width.
3. **Local online rule** (stretch, time-boxed): a velocity-gated update that
   learns rotating pairs step by step. Reported as not done if not reached.
4. Plug learned L1 (PCA, movement) into the 400k network: survivors and kernel
   metrics; probe both (whole + corner).

### Step 2: robustness (exact L1; repeated with movement-learned L1)
1. **Imperfect multiplication**:
   - static coefficient errors, each ±1 coefficient × (1 + δ), δ ~ N(0, s²),
     s ∈ {0.01, 0.05, 0.1, 0.3};
   - products by quarter squares, xy = (f(x+y) − f(x−y))/4 with f = |·|^p,
     p ∈ {1.5, 1.8, 2.2, 2.5} (p = 2 is exact).

   Kernel metrics. Probe s = 0.1 (whole + corner).
2. **Grid noise**, at bump widths 0.25 and 0.10, for the network and for the
   ideal code:
   - additive cell-activity noise σ ∈ {0.001, 0.01, 0.05};
   - a coherent phase shift per module, ε ∈ {0.05, 0.1, 0.2, 0.4} cells.

   Measured: self-similarity cos(z_clean(p), z_noisy(p)), the decoded
   position error (best match among clean codes within 64 cells), and the
   share of errors beyond 5 cells.

### Step 3: cost
1. Unit count against kernel quality (from step 0.1).
2. **Staged selection**: pair units over all 121 · 144 = 17,424 (j₁₁, j₁₂)
   with gain scale r/13 (modules 11 and 12 repeat every 132 = 1716/13 cells),
   then each kept pair is multiplied with all 169 module-13 patterns and gained
   at r. Report neurons and product synapses against 400k random triples and
   the 2.94M full set, and the final kernel.

### Deliverables
`gwn_encoder.py` (with the `gwn:` probe spec and tests), `gwn_check.py`
(steps 0, 2, 3), `gwn_l1_learn.py` (step 1), `run_gwn.sh`, results under
`.../hopfield_probe/gwn/`, this section's results, and a results section on
the page.

### Results (2026-11-02)

**The network works, and the circuit-level questions now have numbers.**

- **The exact network matches the ideal.** 400k random units leave 527 strong
  survivors (predicted about 500). The code's far-cell kernel is C(1) 0.998,
  half-width 19.3, far sd 0.032, alias max 0.041; the ideal's is 0.998 / 19.3 /
  0.032 / 0.058. All 1716² triples give the exact Gaussian kernel (far sd 0.000).
- **L1 can be learned from movement, not from variance.**
  - Diagonalising the module's one-step shift operators gives pure patterns
    (purity 1.000) from 10k walk steps. With bumps 0.10 wide, its network
    equals the exact one (alias 0.064 at 100k steps).
  - Hebbian PCA fails (purity 0.37–0.39, no strong units).
  - The online rule reaches purity 0.95, which is not enough: no strong
    units. Impurities multiply across the three modules, so L1 must be
    much purer than 0.99.
- **Multiplication must be accurate to a few percent.** Coefficient errors of
  1% are harmless; 5% gives alias 0.18 and 10% gives 0.41. Products built from
  squaring units tolerate an exponent of 1.8–2.2.
- **Bump width decides noise tolerance.** With 0.10 bumps, gains learned under
  cell noise and corrected for each unit's jitter at rest rebuild the exact
  network at noise 0.01 (alias 0.054, decode error 0, self-similarity 0.97).
  At noise 0.05 decoding is still perfect. With 0.25 bumps the fine patterns
  are amplified 10⁴–10⁷ times and no rule recovers them. The ideal encoder is
  the reverse: its atan2 decode pools a whole module, so it tolerates noise at
  0.25 and fails at 0.10.
- **17,424 neurons are enough for the exact kernel.** Give each module-11 ×
  module-12 pair one neuron with 169 module-13 branches, and keep the slowest
  branch. The neurons cover every frequency with |n_x|, |n_y| ≤ 66 once. That
  reproduces the full 2.94M-unit network with no randomness.
  - Staged selection by pair slowness fails. The final wave has
    n ≡ 13 m′ (mod 132), so slow pairs (small m′) only make multiples of 13.
    That spectrum repeats every 132 cells: alias 0.66.

**Correction during the run.** The first pass of step 1 kept one of each
conjugate pair in the movement method: pattern j but not −j. Its network
aliased at 0.40 despite purity 1.000, because only a structured subset of
frequencies could be built. Every learner now also emits (c, −s), which is
the −j pair. The first pass is kept in `step1_v0/`.

#### Step 0: exact network (far cells; 5% coverage unless stated)

| condition | strong units | n_eff | C(1) / half-width / far sd / alias max |
|---|---|---|---|
| ideal r = 16 | 512 waves | 512 | 0.998 / 19.3 / 0.032 / 0.058 |
| 25k random units | 30 | 30 | 0.998 / 19.7 / 0.129 / 0.183 |
| 100k | 128 | 128 | 0.998 / 17.5 / 0.062 / 0.066 |
| 400k | 527 | 512 | 0.998 / 19.3 / 0.032 / 0.041 |
| all 1716² triples | 3672 | 3393 | 0.998 / 19.8 / 0.000 / 0.000 |
| 400k, cut gain (keep / prune) | 527 | 524 | 0.998 / 17.8 / 0.031 / 0.036 |
| 400k, rational gain | 883 | 1295 | 0.993 / 11.1 / 0.020 / 0.035 |
| 400k, linear gain | 634 | 565 | 0.998 / 18.5 / 0.030 / 0.043 |
| 400k, steps from 1% or 20% coverage | 527 | 512 | identical to 5% |

The slowness estimated from 8192 walk steps matches |ω|²/2 to a relative
0.02% on slow units. (All-triples and branch codes keep the 6000 largest gains,
96.2% of the total; the rest keep 99.5%.) Coverage does not matter, because a pure wave's change per
step is the same everywhere. The gain rule sets the kernel shape. Gaussian,
cut and linear gains all work here. The rational gain's heavy tail narrows the
peak (half-width 11).

#### Step 1: learned L1 in the 400k network (far cells)

| L1 | bump | purity mean | pairs > 0.99 | strong | C(1) / half-width / far sd / alias max |
|---|---|---|---|---|---|
| PCA | 0.25 | 0.390 | 0.01 | 0 | 0.917 / 3.1 / 0.690 / 1.000 |
| movement, 1k steps | 0.25 | 0.958 | 0.86 | 335 | 0.998 / 19.4 / 0.039 / 0.284 |
| movement, 10k | 0.25 | 0.999 | 0.98 | 476 | 0.998 / 19.5 / 0.033 / 0.132 |
| movement, 100k | 0.25 | 1.000 | 1.00 | 486 | 0.998 / 19.7 / 0.034 / 0.120 |
| online rule, 100k | 0.25 | 0.376 | 0.00 | 0 | 0.948 / 7.6 / 0.374 / 0.805 |
| PCA | 0.10 | 0.370 | 0.01 | 0 | 0.924 / 3.0 / 0.570 / 1.000 |
| movement, 1k | 0.10 | 0.762 | 0.09 | 0 | 0.997 / 21.3 / 0.200 / 0.924 |
| movement, 10k | 0.10 | 1.000 | 1.00 | 523 | 0.998 / 19.5 / 0.031 / 0.079 |
| movement, 100k | 0.10 | 1.000 | 1.00 | 553 | 0.998 / 18.9 / 0.029 / 0.064 |
| online rule, 100k | 0.10 | 0.952 | 0.60 | 0 | 0.986 / 17.4 / 0.213 / 0.857 |

With 1k steps and 0.10 bumps, some phase-to-phase transitions are never seen,
so the shift operator is wrong (max error 0.96 against the exact permutation;
2e-10 at 10k steps). Wide bumps overlap and interpolate from fewer steps. At
0.25 the residual alias (0.12) probably comes from the weakest fine patterns,
which the ridge (1e-10 of the mean eigenvalue) suppresses.

#### Step 2: imperfect multiplication (400k, far cells)

| condition | strong | C(1) / half-width / far sd / alias max |
|---|---|---|
| coefficient error sd 0.01 | 505 | 0.998 / 19.3 / 0.032 / 0.041 |
| coefficient error sd 0.05 | 206 | 0.997 / 18.8 / 0.041 / 0.179 |
| coefficient error sd 0.1 | 40 | 0.996 / 18.9 / 0.079 / 0.408 |
| coefficient error sd 0.3 | 2 | 0.996 / 27.7 / 0.368 / 0.735 |
| quarter squares, p = 1.5 | 9 | 0.994 / 18.7 / 0.064 / 0.081 |
| quarter squares, p = 1.8 | 399 | 0.997 / 19.3 / 0.032 / 0.041 |
| quarter squares, p = 2.2 | 370 | 0.997 / 19.2 / 0.033 / 0.044 |
| quarter squares, p = 2.5 | 0 | 0.992 / 17.9 / 0.076 / 0.114 |

A wrong coefficient mixes in the wave with one module's phase subtracted
instead of added (a fast frequency). That makes the unit look faster, so its
gain drops. Hence fewer strong units and a noisier far field.

#### Step 2: grid noise, gains learned on clean codes (200 unvisited anchors)

Cells give self-similarity / mean decoded error (cells) / share of errors
beyond 5 cells. Decoding is the best match among clean codes within 32 cells.

| noise | level | network, bump 0.25 | ideal, bump 0.25 | network, bump 0.10 | ideal, bump 0.10 |
|---|---|---|---|---|---|
| cell activity | 0.001 | 0.000 / 35.6 / 1.00 | 1.000 / 0.0 / 0.00 | 1.000 / 0.0 / 0.00 | 0.997 / 0.0 / 0.00 |
| cell activity | 0.01 | 0.000 / 35.7 / 1.00 | 0.987 / 0.0 / 0.00 | 0.971 / 0.0 / 0.00 | 0.725 / 2.1 / 0.06 |
| cell activity | 0.05 | 0.000 / 35.5 / 1.00 | 0.726 / 2.1 / 0.06 | 0.548 / 0.8 / 0.00 | 0.019 / 25.4 / 0.85 |
| module phase | 0.05 | 0.979 / 0.0 / 0.00 | 0.924 / 0.2 / 0.00 | 0.994 / 0.0 / 0.00 | 0.943 / 0.1 / 0.00 |
| module phase | 0.1 | 0.920 / 0.1 / 0.00 | 0.731 / 1.9 / 0.04 | 0.974 / 0.0 / 0.00 | 0.787 / 0.9 / 0.01 |
| module phase | 0.2 | 0.710 / 0.7 / 0.00 | 0.312 / 12.5 / 0.32 | 0.836 / 0.4 / 0.00 | 0.370 / 11.1 / 0.29 |
| module phase | 0.4 | 0.268 / 10.2 / 0.38 | 0.044 / 22.5 / 0.76 | 0.341 / 8.7 / 0.34 | 0.051 / 22.8 / 0.78 |

#### Step 2b: gains learned under cell noise

"Raw" uses the change per step. "Corrected" subtracts the change at rest (two
noise draws at one position) and normalises by signal power.

| bump | noise | rule | strong | C(1) / half-width / far sd / alias max | self-sim / err / > 5 |
|---|---|---|---|---|---|
| 0.10 | 0.001 | raw | 428 | 0.998 / 19.3 / 0.032 / 0.094 | 1.000 / 0.0 / 0.00 |
| 0.10 | 0.001 | corrected | 526 | 0.998 / 19.3 / 0.032 / 0.054 | 1.000 / 0.0 / 0.00 |
| 0.10 | 0.01 | raw | 0 | 0.998 / 20.0 / 0.188 / 0.919 | 0.992 / 0.3 / 0.00 |
| 0.10 | 0.01 | corrected | 516 | 0.998 / 18.6 / 0.032 / 0.054 | 0.971 / 0.0 / 0.00 |
| 0.10 | 0.05 | raw | 0 | dead | 0.000 / 32.8 / 1.00 |
| 0.10 | 0.05 | corrected | 2202 | 0.973 / 5.2 / 0.018 / 0.101 | 0.535 / 0.0 / 0.00 |
| 0.25 | 0.001 | corrected | 11171 | 0.849 / 4.1 / 0.025 / 0.224 | 0.201 / 0.8 / 0.01 |
| 0.25 | 0.01 | corrected | 15324 | 0.764 / 2.1 / 0.027 / 0.480 | 0.167 / 1.2 / 0.05 |
| 0.25 | 0.05 | corrected | 14693 | 0.811 / 2.6 / 0.041 / 0.629 | 0.162 / 0.8 / 0.03 |

Raw slowness under noise makes every unit look fast, so it keeps only a few
coarse, aliased units. With 0.25 bumps the corrected rule is swamped. The
at-rest estimate of a 10⁴–10⁷-times-amplified noise is itself noisy, and
thousands of noise-dominated units come out looking slow.

#### Step 3: cost

| design | neurons | product branches | strong | C(1) / half-width / far sd / alias max |
|---|---|---|---|---|
| random triples | 400,000 | 400,000 | 527 | 0.998 / 19.3 / 0.032 / 0.041 |
| all triples | 2,944,656 | 2,944,656 | 3672 | 0.998 / 19.8 / 0.000 / 0.000 |
| staged, pair gain > 0.1 | 31,783 | 2,426,671 | 3004 | 0.998 / 19.3 / 0.008 / 0.163 |
| staged, pair gain > 0.37 | 21,937 | 762,697 | 916 | 0.998 / 18.8 / 0.029 / 0.660 |
| branch: 1 neuron per (j₁₁, j₁₂), slowest of 169 branches | 17,424 | 2,944,656 (pruned to 17,424) | 3672 | 0.998 / 19.8 / 0.000 / 0.000 |

The branch design's 17,424 neurons carry 17,424 distinct frequencies. The one
with |n| = 132 is the pair whose slowest branch would be the excluded constant.

#### Hopfield probe (K = 5, one step, Hebbian; `run_ideal.sh` settings)

| encoder | region | abs err (°) | acc45 | exact | basin | reach | dead goals at K = 5 / 10 / 20 |
|---|---|---|---|---|---|---|---|
| ideal r = 16 | whole | 1.04 | 1.000 | 1.000 | 32.1 | 0.998 | 0 / 0 / 0 |
| ideal r = 16 | corner, centre, opposite | 1.12 | 1.000 | 0.999 | 29.2–29.5 | 0.987–0.988 | 0 / 0 / 0 |
| network, 400k random units | whole | 1.16 | 1.000 | 1.000 | 34.2 | 0.995 | 0 / 0 / 0 |
| network, 400k random units | corner, centre, opposite | 1.13 | 1.000 | 1.000 | 34.6–34.8 | 0.979–0.981 | 0 / 0 / 0 |
| network, all triples | whole / corner | 0.67 | 1.000 | 1.000 | 64.0 / 52.9 | 0.993 / 0.998 | 0 / 0 / 0 |
| branch design, 17,424 neurons | whole / corner | 0.67 | 1.000 | 1.000 | 64.0 / 52.9 | 0.993 / 0.998 | 0 / 0 / 0 |
| movement-learned L1, bump 0.10 | whole / corner | 1.07 / 1.09 | 1.000 | 1.000 | 32.1 / 32.8 | 0.995 / 0.989 | 0 / 0 / 0 |
| movement-learned L1, bump 0.25 | whole / corner | 1.33 | 1.000 | 1.000 | 33.1 / 31.7 | 0.988 / 0.991 | 0 / 0 / 0 |
| 400k, cut gain | whole / corner | 0.81 / 0.84 | 1.000 | 1.000 / 0.998 | 26.0 / 24.9 | 0.994 / 0.982 | 0 / 0 / 0 |
| 400k, linear gain | whole / corner | 1.23 / 1.20 | 1.000 | 1.000 | 29.8 / 29.7 | 0.993 / 0.979 | 0 / 0 / 0 |
| 400k, rational gain | whole / corner | 7.08 / 7.03 | 1.000 | 1.000 | 33.1 / 33.7 | 0.993 / 0.979 | 0 / 0 / 0 |
| 400k, coefficient error 0.1 | whole | 10.0 | 0.999 | 0.974 | 19.7 | 0.945 | 0.04 / 0.12 / 0.08 |
| 400k, coefficient error 0.1 | corner | 10.6 | 0.995 | 0.785 | 11.9 | 0.821 | 0.12 / 0.25 / 0.42 |
| Hebbian PCA L1, bump 0.25 / 0.10 | whole / corner | 87–89 | 0.26 | 0.000 | none | 0.02 | 1 / 1 / 1 |
| bump 0.25, gains learned at cell noise 0.01 (corrected) | whole / corner | 92.3 | 0.23 | 0.95 | 14.8 | 0.08 | 1 / 1 / 1 |

- The 400k network performs like the ideal in every region, with a slightly
  larger exact-retrieval basin (34 against 29–32).
- The all-frequency codes (all triples, branch design) beat the ideal: basin
  53–64 and direction error 0.67°. They contain every slow frequency once, so
  they have no far-field noise; the ideal's 512 random waves leave sd 0.032.
- Movement-learned L1 performs like exact L1 at both bump widths. These probe
  codes are clean, so the 0.25 bump's noise problem does not show here.
- A 10% multiplication error costs most in a region: exact 0.79, reach 0.82,
  42% dead goals at K = 20.
- PCA-learned L1, and gains learned under noise with 0.25 bumps, fail
  completely.

The branch design's whole-arena run timed out at 4 h on the first try
(12,000-dimensional code) and finished in 3 h on the rerun (job 27747697).

### 20.1 The gain is local and has an incremental rule (2026-11-03)

**A walk inside one small environment sets the gains.** A single per-step
depression rule gives the Gaussian gain, and its width grows with time spent
learning.

A unit whose (C, S) pair is one wave changes by 2 − 2cos(ω·e) on a step in
direction e, at every position. So its slowness is measured exactly by a few x-
and y-steps anywhere. The many-environment averaging in step 0 was not needed.
(A lone cosine unit's change depends on its phase, and would need the walk to
cover the wave's whole cycle.)

Setup (`gwn_local_gain.py`): the exact 400k network (bump 0.25), and a random
walk of 57,600 steps inside one square of side 20, and of side 10, reflecting at
the walls. Kernel metrics on far cells. Results in `gwn/local_gain.json`.

- **Locality**: each unit's change per step differs across positions by at most
  2.6e-6 (float rounding), in both environments.
- **Trace rule**, gain = exp(−r² × mean change / 2) with r = 16:

  | steps | half-width | far sd | alias max |
  |---|---|---|---|
  | 10 | 18.1–18.4 | 0.030–0.031 | 0.035–0.044 |
  | 100 | 19.1–19.2 | 0.031 | 0.038–0.039 |
  | 1k–10k | 19.2 | 0.032 | 0.039 |
  | many environments (step 0) | 19.3 | 0.032 | 0.041 |

- **Depression rule**: every step w ← w (1 − κ |ΔC|² − κ |ΔS|²), κ = 0.005,
  w₀ = 1. After T steps w ≈ exp(−κ T × mean change), the Gaussian gain with
  r = √(2κT):

  | T | √(2κT) | Gaussian half-width | measured half-width | strong units | far sd | alias max |
  |---|---|---|---|---|---|---|
  | 6,400 | 8 | 9.4 | 9.6 | 2,038 | 0.016 | 0.036 |
  | 14,400 | 12 | 14.1 | 14.3 | 918 | 0.024 | 0.043 |
  | 25,600 | 16 | 18.8 | 19.2 | 529 | 0.032 | 0.039 |
  | 57,600 | 24 | 28.3 | 28.5 | 241 | 0.047 | 0.029 |

  Side 10 gives the same numbers.

The depression rule needs no built-in transfer function: the exponential comes
from multiplying many small decrements. Its kernel keeps widening as long as
plasticity runs. As it widens, fewer waves stay strong, so the far field gets
noisier (0.016 at r = 8, 0.047 at r = 24) unless the pool of units grows. When
plasticity stops therefore sets both the field size and the noise.

### 20.2 L3 can be pruning instead of weighting (2026-11-03)

**Keeping each unit with probability ∝ P(n), at full weight, builds the ideal
encoder itself.** The survivors are a sample drawn in proportion to P(n) with
equal weights, which is how the ideal draws its 512 waves. Graded gains are not
needed. A hard keep/drop threshold is not the same thing: it gives a flat disc
of frequencies, and its kernel rings.

Uniform random waves, r = 16, keep probability = exp(−r² v), the Gaussian
spectrum shape (`gwn_prune_vs_grade.py`; frequencies only, no grid code):

| pool | rule | units used | half-width | far sd | most negative |
|---|---|---|---|---|---|
| 400k | graded, w = exp(−r² v / 2) | n_eff 481 | 19 | 0.032 | |
| 400k | stochastic pruning, keep w.p. w² | 219 kept | 19 | 0.046 | |
| 400k | hard cut, inside the 2σ disc | 480 kept | 18 | 0.032 | −0.121 |
| 820k | graded | n_eff 986 | 20 | 0.022 | |
| 820k | stochastic pruning | 519 kept | 20 | 0.031 | |
| 820k | hard cut | 1003 kept | 18 | 0.023 | −0.118 |
| ideal | 512 drawn from the Gaussian, equal weights | 512 | 20 | 0.030 | |

- From 820k units, stochastic pruning keeps 519 and matches the ideal.
- Graded gains use the same pool about twice as well (n_eff 481 against 219
  kept), because partly weighted units still count. Pruning keeps fewer neurons;
  grading needs fewer drawn.
- The hard cut has the right width but a −0.12 lobe. In the step 0 probe the
  cut gain had basin 25–26 against 34 for the Gaussian.

**Incremental form.** The depression rule of 20.1, applied to survival instead of
gain: on every step, eliminate the unit with probability κ |Δz|². After T steps
its survival probability is ∏(1 − κ d_t) ≈ exp(−κ T v), which is ∝ P(n) when
κT = r². This is activity-driven elimination of cells or synapses. Like the
depression rule, the field width grows while it runs. Not yet simulated as an
online process; the table samples the final survival probabilities directly.

## 21. Learning L1 locally and online (plan written 2026-11-03, before any run)

**Goal.** Learn L1, the per-module readout pairs, with a local and incremental
rule, inside one environment where possible. A local rule uses only a unit's
inputs, its own output at two moments, and the step direction. The target is
what §20 needs: every pair pure (purity > 0.99), every pattern j of every module
present, and the −j pairs as well. Today's method (§20 step 1) works but is
offline: a least-squares fit over stored steps, then an eigendecomposition.

### Why it should be possible

- **Coverage, not environments.** Estimating a module's shift needs steps out
  of every phase of that module. A box at least 13 × 13 cells covers every phase
  of all three modules. A 10-cell box covers only 100 of module 13's 169
  phases.
- **Covariance and shift share eigenvectors.** With phases visited uniformly,
  the covariance C of a module's activity is translation-invariant on the module
  torus, and so is the shift T. The two commute, and their shared eigenvectors
  are the patterns. So a plain Hebbian product g(t) z(t+1), whose average is
  C Tᵀ w, has the patterns as its fixed directions, even with wide bumps. It is
  biased, though: pattern j's growth is scaled by its power c_j, which spans
  10⁹ at bump 0.25 and 5–20× at 0.10.
- **Lock-in selectivity.** Rotate a unit's update back by its own intrinsic
  turn φ. Then pattern j's part is multiplied each step by about
  (1 − η) + η e^{i(θ_j − φ)}. That has magnitude 1 only when the pattern's turn
  θ_j matches φ, and is smaller otherwise. With η = 0.05 and the smallest
  angular gap between patterns (2π/13 ≈ 0.48 rad), a competing pattern shrinks
  about 0.6% per step, so about 2,000 steps should purify a unit. x-steps use
  φ_x and y-steps use φ_y, so only one 2D pattern survives.

### Common setup

- **Walks.** A continuous 4-direction random walk, reflecting at the walls,
  inside one square box of side s ∈ {10, 13, 20, 40} placed on the scaffold.
  Reference: single steps over the 5% scattered environment set (as §20).
  Steps T ∈ {1k, 10k, 100k}. Bump widths 0.10 and 0.25. Noise-free first,
  then cell noise 0.01 at bump 0.10.
- **Purity** as in `gwn_l1_learn.purity`. Report the share of pairs above 0.99,
  and **pattern coverage**: the share of each module's λ² patterns matched by at
  least one pair above 0.99.
- **Downstream.** `gwn_l1_learn._assemble` (adding (c, −s) conjugates), 400k
  random triples, trace-rule gains from the same walk (`gwn_local_gain` style),
  and `World.metrics`. Exact L1 gives alias 0.041 and far sd 0.032.

### Route 1: a velocity-gated temporal Hebbian estimate of the shift

Per module and axis, stream two sums over the walk: A_e += g(t+1) g(t)ᵀ on +e
steps (−e steps are reversed), and C += g(t) g(t)ᵀ. Two readouts:

- **1a, Hebbian only.** T_e ≈ A_e with each row normalised. Near-exact for
  narrow bumps, where C ≈ cI.
- **1b, ridge.** T_e = A_e (C + ρI)⁻¹ with one solve at the end. This is §20's
  estimate, accumulated online.

Then the eigenvectors of (T_x + 0.618 T_y)ᵀ, offline as in §20.

Measure purity, pattern coverage and the error max|T − P| against the exact
permutation, as functions of s, T and bump width.

Expectation: s ≥ 13 is pure from about 10k steps; at s = 10, module-13
patterns are missing or impure. Route 1 makes the shift estimate local; the
eigenvector step is still offline.

### Route 2: the lock-in rule (local, online, no matrix)

Per module, K units (K ∈ {2λ², 6λ²}), each with complex weights w_k (random
start, length λ²) and intrinsic turns (φ_x, φ_y) drawn uniformly. On each step
in direction e (a −e step is treated as +e read backwards), with z = wᵀ g:

- **2a, plain Hebbian lock-in.**
  w ← normalise(w + η e^{−iφ_e} g(t) z(t+1)).
  The simplest form. Expect a bias toward strong, coarse patterns at bump 0.25.
- **2b, predictive lock-in.** A fast delta rule learns u_e to predict the
  unit's own next output from its current input:
  u_e ← u_e + η_u (z(t+1) − u_eᵀ g(t)) g(t), which converges to Tᵀ w.
  Then w ← normalise((1 − η) w + η e^{−iφ_e} u_e). This removes C, so the
  power bias goes away.
- **2c, 2b plus an adaptive turn.** A phase-locked loop moves φ_e toward the
  turn the unit actually sees:
  φ_e ← φ_e + η_φ · angle(z(t+1) z(t)* e^{−iφ_e}).

Sweep η ∈ {0.01, 0.05, 0.2} (η_u = 4η, η_φ = η). Measure purity against steps
(convergence speed), pattern coverage against K (fixed random turns need about
λ² ln λ² ≈ 850 units for module 13; 2c should need fewer), and the bias of 2a
at each bump width.

### Route 3 (low priority): the existing online rule, trained longer

`gwn_l1_learn.online_l1` with 20k iterations and a decaying learning rate. Run
it only if routes 1–2 leave a gap. It needs lateral decorrelation, so it is
less local.

### Then

- The best two learners (s ≥ 13, T = 100k) through the 400k network: kernel
  metrics, then the probe (whole + corner) via `run_gwn.sh`-style tasks.
- Repeat the best learner with cell noise 0.01, bump 0.10.
- **Success**: purity > 0.99 for at least 99% of pairs, every pattern covered,
  network alias ≤ 0.08 and far sd ≤ 0.035, from one environment.

### Implementation notes

- **New script** `analysis/hopfield_probe/gwn_l1_online.py`, with routes 1–3.
  Reuse `gwn_check` (World, kernel metrics, `build`), `gwn_l1_learn` (purity,
  `_assemble`, `move_l1`'s eigen step) and `gwn_local_gain.walk_in_env`.
  Support `GWN_SMOKE=1` for a login-node smoke test.
- **Route 2 cost.** Vectorise over units and modules on the GPU; the steps are
  a sequential loop. 100k steps × about 1,000 units × 169 weights is about a
  minute.
- **Launcher.** Add a `l1online` mode to `run_gwn.sh`. Partitions:
  pi_evelina9 first. Throttle probe arrays to 4. Arm a watch on every sbatch.
- **Outputs**: `.../hopfield_probe/gwn/l1_online/` (JSON per route,
  `summary.md`, encoders). Results go into this section. Update
  "Learning the Grid Wave Network" A8 and the memory.

### Results (run 2026-11-03)

**Answer.** L1 can be learned inside one 14 × 14 box, from a walk alone.

- **Shift rule plus eigenvectors (route 1c).** A local, online delta rule
  learns each module's one-step shift. The eigenvectors of T_x + 0.618 T_y
  then give every L1 pair pure and every pattern covered, from about 30k steps.
  The network built on it is identical to its exact twin and navigates like
  exact L1 (probe basin 34, reach 0.98–0.99). The eigenvector step is still
  offline.
- **Fully online lock-in (route 2c with a slow turn).** Single units, no
  matrix. With the turn adapting 1000 times more slowly than the weights,
  every unit ends pure and 94–98% of patterns are held. The learned weights
  are exact (learned network = twin), but the network wired from every unit
  aliases at 0.27–0.34 because patterns are held by unevenly many units. An
  oracle that keeps one unit per pattern (chosen with the true patterns) gives
  0.07–0.10; that is a what-if, not something the route does.
- **Cell noise 0.01** roughly doubles the alias of the shift learners and
  breaks the lock-in rules. **Bump 0.25** makes the finest patterns too weak
  to learn locally.

#### What the plan got wrong

- **The box needs side 14, not 13.** A box of side s never makes the step from
  phase s − 1 to phase s, because that step would leave the box. For module 13
  in a 13-box that is the wrap from phase 12 to phase 0, so one transition per
  row is never seen. That column of the shift stays unknown and every
  eigenvector of the module is impure (1b at side 13: modules 11 and 12 pure,
  module 13 not, 61% of pairs). Under the lock-in rules the ring becomes an
  open chain, and on an open chain a wave of any angle fits every step. Side
  14 sees every transition of every module. Both facts are pinned in
  `test_gwn_l1_online.py`.
- **The lock-in rules are about λ² times slower than estimated.** With narrow
  bumps a step updates only the weights at the current phase, so each weight
  changes about once per λ² steps. Convergence takes 30k–300k steps, not
  2,000.
- **1a is not near-exact for narrow bumps.** A reflecting walk visits phases
  unevenly. Edge cells get half the visits, and a 20-box sees phases 0–6 of
  module 13 twice as often as phases 7–12. The covariance then no longer
  commutes with the shift. 1a reaches 0.21–0.97 pure at 100k steps and its
  networks alias at 0.43–0.71.
- **2c does not need fewer units; it collapses.** Its turn follows the unit's
  output, and the output is dominated by the strongest-input pattern long
  before that pattern dominates the weights. 27–44% of units end on the
  constant pattern and no weak pattern is held.
- **"Alias ≤ 0.08" cannot be a pass mark.** Over 8 random-triple seeds the
  exact network's alias ranges 0.032–0.083 (bump 0.10; far sd 0.031–0.032),
  because each seed samples a different set of frequencies. A learned L1 lists
  its pairs in its own order, so it samples its own set; the same learner gave
  0.058 in one job and 0.085 in another when GPU rounding reordered tied
  eigenvalues. The right test is the **exact twin**: the same triples, with
  each learned pair replaced by the exact pair of its dominant pattern
  (`gwn_l1_twin.py`).

#### Route 1: the shift, estimated in one box (bump 0.10, no noise)

Share of pairs above purity 0.99 after 1k / 10k / 100k steps, and the 400k
network's alias at 100k steps (learned; the exact twin gives the same value to
three decimals wherever every pair is pure).

| box | cells | 1a | 1b | 1c | alias, 1b / 1c |
|---|---|---|---|---|---|
| 10 | 100 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | code collapses |
| 13 | 169 | 0 / 0 / 0.54 | 0 / 0.61 / 0.61 | 0 / 0.60 / 0.61 | 0.686 / 0.674 |
| 14 | 196 | 0 / 0.06 / 0.77 | 0 / 1.00 / 1.00 | 0 / 0.94 / 1.00 | 0.058 / 0.044 |
| 20 | 400 | 0 / 0 / 0.21 | 0 / 1.00 / 1.00 | 0 / 0.79 / 1.00 | 0.076 / 0.097 |
| 40 | 1600 | 0 / 0.18 / 0.91 | 0 / 1.00 / 1.00 | 0 / 0.92 / 1.00 | 0.084 / 0.056 |
| 5% scattered steps | 50,000 | 0 / 0.68 / 0.97 | 0.03 / 1.00 / 1.00 | 0 / 1.00 / 1.00 | 0.048 / 0.064 |

- **Twins.** For boxes 14, 20, 40 and the scattered steps, learned and twin
  networks agree to three decimals for 1b and 1c (C(1) 0.998, half-width
  18.3–19.5, far sd 0.029–0.031). L1 learned in one 14-cell box adds no
  measurable error.
- **Rate of 1c** (`gwn_l1_lms.py`). Rate 0.5 is best without noise: |T − P| is
  2×10⁻³ after 30k steps in a 14-box and 10⁻⁷ after 100k. Rates 0.1, 0.02 and
  0.005 are slower.
- **Probe** (whole / corner, K = 5, one step):

  | network | abs err | exact | basin | reach |
  |---|---|---|---|---|
  | 1c, one 14-box | 1.70 / 1.70 | 1.000 / 1.000 | 34.1 / 34.8 | 0.994 / 0.983 |
  | 1b, one 14-box | 1.25 / 1.24 | 1.000 / 1.000 | 33.4 / 33.3 | 0.993 / 0.984 |
  | exact 400k | 1.16 / 1.13 | 1.000 / 1.000 | 34.2 / 34.8 | 0.995 / 0.980 |
  | ideal r = 16 | 1.04 / 1.12 | 1.000 / 0.999 | 32.1 / 29.5 | 0.998 / 0.987 |

#### Route 2: lock-in rules (fully online)

First run (`gwn_l1_online.py`, 100k steps, 6 l² units per module, η 0.01–0.5):

- **2a** (plain Hebbian) and **2d** (2a plus the adaptive turn) never purify
  (median purity 0.14–0.36). Each update changes only the visited weights, by a
  phase that is not exactly the pattern's, so the relative phases diffuse; and
  uneven visits tilt the mean fixed point.
- **2b** (predictive, fixed turns) has no power bias (1% of units on the
  constant pattern, coverage 0.62–0.78 in boxes at η 0.2) but purifies slowly: 14–21%
  of units above 0.99 after 100k steps in a box, 43% with scattered steps. With
  the turn not exactly the pattern's, the weights' overall phase drifts every
  step and the predictor lags behind.
- **2c** (2b plus a phase-locked turn) makes every unit pure but collapses
  (coverage 5–10%).

Turn rate (`gwn_l1_lockin2.py`; 300k steps, η 0.2, boxes 14 and 20 and the
scattered steps):

| rule, turn rate ÷ weight rate | pure | coverage | weak-half coverage | on constant pattern |
|---|---|---|---|---|
| 2c, 1 (first run) | 1.00 | 0.07–0.10 | 0 | 0.36–0.44 |
| 2c, 0.1 | 1.00 | 0.12–0.15 | 0 | 0.18–0.19 |
| 2c, 0.01 | 1.00 | 0.54–0.65 | 0.13–0.29 | 0.02 |
| 2c, 0.001 | 1.00 | 0.94–0.98 | 0.88–0.95 | 0.01–0.02 |
| 2e (turn read from the weights), 1 | 1.00 | 0.20–0.34 | 0–0.05 | 0.09–0.30 |
| 2e, 0.1 | 1.00 | 0.65–0.79 | 0.36–0.57 | 0.02–0.03 |

- **Why the slow turn works.** Each unit first selects the pattern nearest its
  starting turn, as under 2b. The turn then locks onto that pattern, which
  makes it an exact fixed point (`test_gwn_l1_online.py` pins that a pattern
  with its own turn and a learned predictor stays put step by step).
- **2e** reads the turn as angle(Σ_a w̄_a u_a), which weights each pattern by
  its share of the unit's weights instead of its input power. It helps, but
  less than slowing the turn.
- **Network for 2c at 0.001** (box 14 / box 20 / scattered steps): wired from
  every unit, learned 0.335 / 0.300 / 0.269 and the exact twin identical; one
  unit per pattern 0.095 / 0.093 / 0.066. The learned weights are exact; the alias comes from how many
  units hold each pattern (count CV 0.35–0.56 per module).
- **Uneven counts** (`gwn_l1_mult.py`, exact L1 with simulated counts).
  Random (Poisson) counts cost little: mean 12 per pattern (CV 0.27) gives
  alias 0.083–0.114, mean 50 or more 0.05–0.09, one per pattern 0.052–0.096.
  The 2c counts have a similar CV but cost 0.27–0.34, so they are structured: weak,
  fine patterns are held by fewer units (weak-half coverage 0.88–0.95). A
  per-module bias that varies smoothly with |j| puts a periodic comb on the
  spectrum, which lands on the grid code's self-similar displacements.
  Equalising units per pattern (for example lateral competition between L1
  units with the same output) is the missing piece of the fully online route.

#### Cell noise 0.01 (bump 0.10)

Network gains here are the trace rule with the at-rest correction. The 100k rows were rerun after the
`local_slowness` fix (§23; before: 1b 0.115–0.205 / 0.073–0.097); the other rows predate it, which
narrows every kernel (half-width ~17) on both sides of each comparison.

| learner | steps | pure | network alias, learned / twin |
|---|---|---|---|
| 1b | 100k | 0.93–1.00 | 0.101–0.208 / 0.053–0.106 |
| 1b | 300k | 1.00 | 0.081–0.145 / 0.087–0.124 |
| 1c, rate 0.5 | 100k | 0.03–0.07 | 0.90–0.93 |
| 1c, rate 0.1 | 300k | 0.95–0.97 | 0.62–0.66 |
| 1c, rate 0.02 | 300k | 1.00 | 0.13–0.19 / 0.07–0.10 |
| 1c, rate 1/(1 + n) | 300k | 0.97–0.99 | 0.17–0.24 / 0.07–0.12 |

- **1b reaches its twin** given 300k steps (`gwn_l1_noisefix.py`,
  `gwn_l1_decay.py`).
- **1c needs a slow rate.** At rate 0.5 each noisy step overwrites half a
  synapse's estimate and nothing averages. At 0.02 every pair is pure, but the
  network still aliases at about twice its twin.
- **The bias is not the problem.** Noise adds σ²I to the input covariance but
  not to the cross-time product, so the estimate shrinks pattern j by
  c_j / (c_j + σ²), unevenly in a box. Subtracting it (1b with C − Nσ²I; 1c
  with T ← T + rate[(g₁ − T g₀) g₀ᵀ + σ² T] / |g₀|², σ² measurable at rest)
  cuts |T − P| 3–10 times but leaves the network unchanged. The remaining
  error is variance.
- **A per-synapse rate 1 / (1 + n)** (online least squares; n = the
  presynaptic cell's summed activity) is worse than a fixed 0.02. It freezes
  before the coupling through the bump tails has settled.
- **Probe of 1b at noise** (one 14-box, 100k steps): basin 28.9 / 30.3, reach
  0.986 / 0.973, abs err 3.9°. It still navigates.
- **Lock-in at noise** (2c and 2e, box 14, 300k steps). At η 0.2 no variant
  has a unit above 0.99: each noisy step moves the weights by a fifth. At
  η 0.05, 2c with the 0.001 turn reaches 97% pure and coverage 0.85, but the
  network aliases at 0.67 (exact twin 0.34, one unit per pattern 0.63). Under
  noise the fully online route fails at these rates.

#### Bump 0.25

1b reaches 96–98% pure with network alias 0.12–0.18 (twins 0.05–0.11). 1c
stays at 31–63% pure, alias 0.86–0.89 (twins 0.05–0.08). The finest patterns
carry about 10⁻⁹ of the coarsest one's power, so a delta rule learns them too
slowly and the ridge suppresses them. Every lock-in rule collapses onto coarse
patterns (2c coverage 3–5%). Bump 0.10 is the setting for local learning, the
same conclusion as for noise in §20.

#### Code and results

- Code (`analysis/hopfield_probe/`): `gwn_l1_online.py` (routes 1a–1c and
  2a–2e, summary), `gwn_l1_twin.py`, `gwn_l1_lms.py`, `gwn_l1_noisefix.py`,
  `gwn_l1_decay.py`, `gwn_l1_lockin2.py`, `gwn_l1_mult.py`. `run_gwn.sh` modes
  `l1online` (16-task array), `l1twin`, `l1lms`, `l1noisefix`, `l1decay`,
  `l1lockin2`, `l1mult`; probe tasks 26–31. Tests:
  `hopfield_nav/tests/test_gwn_l1_online.py`.
- Results (`.../hopfield_probe/gwn/l1_online/`): `side*.json` and
  `summary.md` (main array), `twin.json`, `lms_rate.json`, `noisefix.json`,
  `decay.json`, `lockin2.json`, `mult.json`, `encoders/` (boxes 14 and 20);
  probe rows in `gwn/summary.md` (t26–t31).

#### Next

- Equalise units per pattern in the fully online route: lateral competition
  among L1 units, or more units with an unbiased selection.
- 1c under noise: a rate schedule between fixed 0.02 and 1 / (1 + n), over
  longer walks.
- The eigenvector step of route 1 as a network (it is what route 2 does per
  unit; combining 1c's shift with 2c's units is untested).

### 21.1 More environments, longer walks, and both layers local (2026-11-04)

Two questions after §21: would a bigger box or a few more environments help
the online rules, and does the network navigate when L1 and L3 both come from
their local rules?

**More environments trim the lock-in unevenness a little; the cause is a
bias toward strong patterns** (`gwn_l1_multienv.py`; 1M steps; "6 boxes" =
sessions of 10k steps cycling through boxes of side 14, 18, 24, 30, 36 and 40
at random scaffold positions; 2c with the 0.001 turn, η 0.2; no noise).

| condition | all units (= exact twin) | one unit per pattern | units per pattern, strong half / weak half | corr(count, log strength) |
|---|---|---|---|---|
| 14-box, 300k steps (§21) | 0.335 | 0.095 | | |
| 40-box | 0.301 | 0.069 | 13.4–16.3 / 7.6–10.7 | 0.57–0.68 |
| 6 boxes | 0.227 | 0.094 | 13.2–15.9 / 8.1–10.8 | 0.39–0.68 |

- Network alias, far cells. The weak half of each module's patterns (by
  A(j)/A(0)) gets 30–50% fewer units than the strong half, and the bias grows
  at smaller η (correlation 0.60–0.78 at η 0.02). This measures directly what
  §21 inferred. Each condition is one run, and the same 14-box recipe gave
  0.192 in another run (§22.1) against 0.335 here, so run-to-run spread (at
  least 0.14) is larger than any effect of more boxes. The bias itself (fewer
  units on weak patterns) is in every run; it comes from the bump's power
  spectrum.
- The shift rule 1c is exact in both conditions without noise (alias 0.064
  and 0.056, equal to the twins).

**Under noise, a slow rate over more steps rescues the shift rule**
(`gwn_l1_noise_slow.py`; cell noise 0.01; 1M steps). Network alias, learned /
exact twin. Rerun 2026-11-08 after the `local_slowness` fix (§23); the rate 0.02
row predates it (narrower kernels on both sides):

| learner | 6 boxes | 14-box |
|---|---|---|
| 1c, rate 0.02 | 0.225 / 0.090 | 0.13–0.19 / 0.07–0.10 (300k steps, §21) |
| 1c, rate 0.005 | 0.071 / 0.052 | 0.058 / 0.059 |
| 1c, rate 0.005, bias-corrected | 0.074 / 0.062 | 0.057 / 0.053 |
| 1c, rate 0.002 (91–95% pure) | 0.173 / 0.074 | 0.166 / 0.081 |
| 1b (offline least squares) | 0.078 / 0.072 | 0.073 / 0.070 |
| 1b, bias-corrected | 0.062 / 0.058 | 0.060 / 0.059 |

- At rate 0.005 the local rule equals offline least squares and sits within
  0.00–0.02 of its exact twin (in one 14-box, 0.058 against 0.059). One 14-box does as well as six boxes, so the
  number of steps matters, not the number of places. The bias correction cuts
  |T − P| (1c 0.047 to 0.019, 1b 0.03 to 0.002) but changes the network
  little.
- The lock-in rule under noise does not recover with more steps. Across 6
  boxes at η 0.05 every unit is above 0.99, yet the network aliases at 0.697
  (exact twin 0.340, one unit per pattern 0.634): the units' residual
  impurity, not their counts, sets the alias. One 40-box is the same (0.662;
  twin 0.326; one unit per pattern 0.588).

**Both layers from local rules** (`gwn_l1l3_local.py`; one 14-box, bump
0.10, no noise). L1 is learned first; L3 then runs the step-by-step
depression rule w ← w·max(0, 1 − κ|Δz|²) over 25,600 further steps
(√(2κT) = r = 16), and the encoder keeps the units holding 99.5% of Σw².

| L1 | depression L3: alias / far sd | trace L3: alias | probe (whole / corner): basin, reach |
|---|---|---|---|
| exact | 0.054 / 0.032 | 0.054 | 34.4 / 34.9, 0.994 / 0.981 |
| 1c (100k steps) | 0.094 / 0.029 | 0.094 | 33.4 / 34.1, 0.984 / 0.996 |
| 2c, every unit (300k steps) | 0.340 / 0.037 | 0.340 | 30.2 / 29.4, 0.995 / 0.981 |

- **All three navigate.** Exact retrieval is 1.000 everywhere, direction
  error 1.1–1.6°, no dead goals. The fully online network learns everything
  from the walk (lock-in L1, depression L3, one 14-box, 325k steps in all);
  here each pair's scale is a batch average over the visited cells
  (`_assemble`), and §22.1's running-power version reproduces the result
  (basin 29.9 / 29.0). Against the ideal encoder (basin 32.1 / 29.5, reach
  0.998 / 0.987) it matches reach and exact retrieval; its basin is about 2
  cells smaller in a paired comparison (−1.9 ± 0.7, §23). Its kernel alias of
  0.34 costs about 4 cells of basin against exact L1 and nothing else, so the
  alias metric overstates what uneven units per pattern cost the memory.
- Depression and trace give the same kernel metrics to three decimals (alias
  within 3e-4, strong counts within 3), without noise; the gains themselves
  were not compared one by one. L3 learned after L1 had finished; learning both at once is
  untested, and early, impure L1 units would be depressed hard, so L3
  plasticity probably has to start late.
- Probe tasks 32–37 (`run_gwn.sh probe`).

## 22. Toward a fully online, noise-robust network (plan written 2026-11-04, before any run)

§21.1 showed that a network whose L1 and L3 both come from local rules, learned
in one 14-cell box, navigates like the ideal. Three things stop that from being
the full answer. This section tests them, plus how noise robustness scales.
L3 is the depression rule throughout (Jack's preference; the trace rule only as
a comparison).

**Questions**

- **22.1 Concurrent learning.** Can L1 and L3 learn at the same time, from step
  0, on one walk? The depression rule never recovers, so L2 units built from
  early, impure L1 units may be worn down before L1 settles.
- **22.2 Noise scaling.** Under cell noise 0.01, how do the shift rule (1c) and
  the lock-in rule (2c) improve with the number of steps, the number of
  environments and the box size?
- **22.3 Noisy-trained networks in the probe.** Does the lock-in network trained
  under noise (kernel alias 0.66–0.70) still navigate? Alias 0.34 cost only
  about 4 cells of basin.
- **22.4 Noisy grid codes at test time.** Every probe so far fed clean codes.

**Shared pieces** (`analysis/hopfield_probe/gwn_online.py`)

- Walks stream in sessions of 10k steps, each a reflecting walk inside one box,
  cycling through the environments. No step crosses between boxes. Codes are
  made per session, so walks can reach 3M steps without holding them in memory.
- Stateful learners with the same updates as §21: `LockIn` (rules 2a–2e, slow
  turn) and `ShiftLMS` (1c at several rates, optionally bias-compensated). A
  test pins `LockIn` to `gwn_l1_online.lock_in` exactly.
- L2 outputs are computed online from the current L1 units, with each L1 unit
  scaled by a running estimate of its own output power (a local homeostatic
  gain), so concurrent L3 sees what the network sees.
- **Depression with rest recovery under noise.** Noise adds jitter to every
  step's change, so plain depression would wear every unit down by an amount
  that depends on its noise, not its frequency (§20 step 2b found the
  uncorrected gains fatal). The animal also rests. At rest a unit's change is
  pure noise, and the synapse recovers by that amount:
  log w += log(1 − κ d_move) − log(1 − κ d_rest). This is the depression form
  of the trace rule's at-rest correction.

**22.1 design** (bump 0.10, 14-box; no noise, then noise 0.01)

- L1 = lock-in 2c with the 0.001 turn: η 0.2 for 300k steps without noise;
  η 0.05 and 0.02 for 1M steps at noise 0.01.
- L3 variants, all with κ chosen so the exposure gives r = 16:
  - (a) always on from step 0, κ = r² / (2 T_total);
  - (b) a late window: κ = 0.005 for the last 25,600 steps;
  - (c) a self-timed critical period per L2 unit: the unit starts silent
    (w = 0), is "born" (w = 1) once its three L1 inputs have settled (each L1
    unit's own phase-lock error stays small), depresses for 25,600 steps, then
    freezes.
- Reference: §21.1's sequential version (L3 on a fresh walk after L1).
- Also at noise: 1c (rate 0.005, 1M steps) plus eigenvectors, then depression
  with rest recovery, for a "local except the eigenvector step" network.
- Kernel metrics, then the probe for every variant that passes.

**22.2 design** (noise 0.01, bump 0.10; checkpoints 100k, 300k, 1M, 3M steps)

- Number of environments: 1, 3, 6 and 12 boxes of side 20.
- Box size: one box of side 14, 20, 40 or 80.
- Learners: 1c at rates 0.02, 0.005, 0.002, 0.001; 2c at η 0.05, 0.02, 0.01.
- At each checkpoint the best rate per learner goes through the network,
  learned and as its exact twin. Also reported: |T − P| (1c) and the mean
  impurity 1 − purity (2c), which is more sensitive than the share above 0.99.

**22.3–22.4 design**

- `run.py --grid_noise σ` adds fresh Gaussian noise of sd σ to every grid code
  the probe encodes: stored goals and every cue. The direction readout's local
  basis stays clean by default (`--grid_noise_basis` adds noise there too),
  because a single-step difference of two noisy codes is mostly noise and would
  test the readout, not the memory.
- Test-time noise σ = 0.01 and 0.03, whole region, for the ideal (bump 0.10),
  the three §21.1 networks, and the noisy-trained networks of 22.1.

**Outputs**: `.../gwn/online/` (one JSON per script, encoders); probe tasks
from 38 on in `run_gwn.sh`. Results go here, with a short verdict per question.

### 22 results (run 2026-11-04)

**Verdicts**

- **22.1 Concurrent learning works if L3 starts late.** With L1 and L3 on
  the same walk, a depression window fixed over the last 25,600 steps (L1 had
  settled by ~100k of 300k steps) gives the sequential network to three
  decimals, and it navigates nearly like the ideal (basin 29–30 against 32,
  reach 0.99, exact 1.000). A window timed by each unit's own inputs settling
  gets most of the way (basin 27–28). L3 from step 0 destroys the code.
- **22.2 Steps matter most; places and box size barely.** Under cell noise
  0.01 both rules approach their exact twins as steps grow. SCALING_VERDICT
- **22.3 Noise-trained networks navigate; the fully online one with a cost.**
  Trained under cell noise 0.01 in one 14-box, the shift-rule network
  navigates like exact L1, with noisy test codes too (basin 31–35, reach
  0.99). The fully online lock-in network at η 0.02 navigates with a 9°
  direction error and up to 17% dead goals; at η 0.05 (kernel alias 0.68) a
  third to a half of the goals die. A slower rate over more steps (22.2) is
  the remedy.
- **22.4 With noisy cues and stored goals, the network navigates as if the
  codes were clean.** At test noise 0.01 (readout basis clean) basin 32 and
  reach 0.99–1.0. The ideal encoder is hurt more. As implemented it falls to reach 0.58 at 0.01,
  but most of that is its choice of harmonic (j up to l − 1); with |j| ≤ l/2
  it keeps reach 0.99 but loses exact retrieval (0.77) and half its basin
  (17). At 0.03 the network still reaches 98% of goals and the symmetric ideal
  23%. All of this is with a clean readout basis (§23).

#### 22.1 Concurrent learning (`gwn_concurrent.py`; one 14-box, bump 0.10)

Without noise (lock-in η 0.2, 300k steps; L1 ends 100% pure, coverage 0.96;
99.9% of units count as settled by 100k steps). Kernel alias / far sd /
strong units, and the probe (whole / corner):

| L3 | kernel | probe basin | probe reach | exact |
|---|---|---|---|---|
| always on from step 0 | 0.860 / 0.594 / 0 | | | |
| late window (last 25,600 steps) | 0.192 / 0.034 / 480 | 29.9 / 29.0 | 0.986 / 0.991 | 1.000 |
| self-timed, gate 0.9 | 0.458 / 0.123 / 0 | | | |
| self-timed, gate 0.99 | 0.365 / 0.040 / 151 | 28.2 / 27.2 | 0.984 / 0.985 | 0.996 / 1.000 |
| sequential (fresh walk after) | 0.192 / 0.034 / 480 | (= late) | | |
| shift-rule L1, sequential | 0.060 / 0.030 / 540 | | | |

- Always on: depression never recovers, and early depression differs from
  unit to unit (impure L1 outputs, and a running power estimate that starts
  from a single sample), so only a scrambled handful of units survives (9).
- The late window and the sequential walk give the same kernel to three
  decimals (alias 0.1924 / 0.1923, kept 1306 / 1307; different walks, so not
  identical): once L1 has settled, running L3 during or after the walk makes
  no difference. The window opens at a fixed time chosen in advance, not by a
  settling signal.
- The self-timed critical period (each L2 unit silent until its three inputs
  settle, then 25,600 steps of depression) works at a strict gate. Units
  born before their inputs are fully pure are over-depressed, so fewer units
  are strong (151 against 480); a stricter gate or a delay after settling is
  the obvious refinement. Units born too late to finish their 25,600 steps keep
  partial gains (1.1% of units in the noisy η 0.02 run).

Under cell noise 0.01 (1M steps; depression with rest recovery):

| L1 | L3 | kernel alias / far sd / strong |
|---|---|---|
| lock-in η 0.05 (impurity 3.6e-3) | late / sequential | 0.681 / 0.057 / 25; 0.678 |
| lock-in η 0.02 (impurity 1.5e-3) | late / sequential | 0.502 / 0.041 / 249; 0.495 |
| lock-in η 0.02 | self-timed 0.99 | 0.446 / 0.060 / 47 |
| lock-in, either η | always on | 0.83–0.85, nearly no units |
| shift rule, rate 0.005 | sequential, rest recovery | 0.097 / 0.032 / 457 |

- Depression with rest recovery gives a working network: the shift-rule
  network reaches alias 0.097 and probe basin 34 / 35 (one walk; there is no
  same-L1 control with the trace rule).
- Under noise only 76% of lock-in units pass the 0.99 settledness bar by 1M
  steps, so the self-timed version leaves many units unborn.

#### 22.2 Noise scaling (`gwn_noise_scaling.py`; cell noise 0.01, bump 0.10)

SCALING_TABLES_NOTE

Full tables: `gwn/online/scaling_summary.md`. Network alias, learned / exact
twin, with the best rate chosen at each checkpoint:

| condition | shift rule 1c: 100k | 300k | 1M | 3M |
|---|---|---|---|---|
| 1 box, side 20 | 0.663 / 0.077 | 0.335 / 0.076 | 0.253 / 0.096 | 0.151 / 0.060 |
| 3 boxes, side 20 | 0.426 / 0.093 | 0.269 / 0.088 | 0.175 / 0.072 | 0.082 / 0.073 |
| 6 boxes, side 20 | 0.453 / 0.103 | 0.159 / 0.133 | 0.125 / 0.076 | 0.115 / 0.080 |
| 12 boxes, side 20 | 0.407 / 0.090 | 0.215 / 0.069 | 0.198 / 0.092 | 0.079 / 0.082 |
| 1 box, side 14 | 0.413 / 0.141 | 0.187 / 0.105 | 0.179 / 0.076 | 0.066 / 0.060 |
| 1 box, side 40 | 0.341 / 0.086 | 0.155 / 0.096 | 0.226 / 0.084 | 0.115 / 0.126 |
| 1 box, side 80 | 0.353 / 0.107 | 0.164 / 0.070 | 0.208 / 0.084 | 0.104 / 0.092 |

| condition | lock-in 2c: 100k | 300k | 1M | 3M |
|---|---|---|---|---|
| 1 box, side 20 | 0.844 / 0.416 | 0.757 / 0.397 | 0.575 / 0.421 | 0.509 / 0.407 |
| 3 boxes, side 20 | 0.739 / 0.322 | 0.648 / 0.338 | 0.534 / 0.388 | 0.481 / 0.425 |
| 6 boxes, side 20 | 0.792 / 0.372 | 0.673 / 0.354 | 0.515 / 0.371 | 0.488 / 0.438 |
| 12 boxes, side 20 | 0.817 / 0.333 | 0.660 / 0.327 | 0.495 / 0.367 | 0.439 / 0.362 |
| 1 box, side 14 | 0.780 / 0.348 | 0.674 / 0.361 | 0.548 / 0.415 | 0.417 / 0.359 |
| 1 box, side 40 | 0.805 / 0.398 | 0.678 / 0.396 | 0.545 / 0.403 | 0.465 / 0.416 |
| 1 box, side 80 | 0.829 / 0.444 | 0.725 / 0.435 | 0.551 / 0.455 | 0.434 / 0.366 |

- **Steps.** In every condition the gap to the twin shrinks with steps. The
  shift rule's mean impurity falls from 4e-3–9e-2 at 100k steps to
  2e-4–3e-3 at 3M (about 4 times per tenfold more steps between 300k and
  3M); lock-in's from 7e-2–9e-2 to 6e-4–2e-3.
- **The best rate falls with steps.** The shift rule's best rate is 0.02 up
  to 1M steps and 0.005 at 3M; lock-in's best η goes 0.05, 0.05, 0.02, 0.01.
  A rate that winds down as experience accumulates is what noise asks for,
  the same winding down the depression rule already needs.
- **Number of environments and box size** are second order. At 3M steps
  every condition is within the sampling spread of its twin for the shift
  rule. One exception: a single box of side 20 leaves module 13 ten times
  more impure (5.0e-3 against about 2.2e-4 for boxes of side 14, 40 and 80),
  and several side-20 boxes reduce it (0.7–1.8e-3). A per-axis count of visits
  per phase does not separate side 20 from side 14, so the cause is not
  pinned down.
- The lock-in twins (0.36–0.44) are the uneven units per pattern, which cost
  only basin in the probe (§21.1).

#### 22.3 Noise-trained networks in the probe (cell noise 0.01, one 14-box, 1M steps)

Probe tasks 50–58; clean test codes unless marked.

| network | kernel alias | abs err (°) | exact | basin (whole / corner) | reach | dead (K = 5 / 10 / 20) |
|---|---|---|---|---|---|---|
| shift-rule L1 + depression L3 | 0.097 | 3.37 / 3.36 | 1.000 | 33.9 / 35.2 | 0.993 / 0.985 | 0 / 0 / 0 |
| same, test noise 0.01 | | 3.89 | 1.000 | 31.1 | 0.987 | 0 / 0 / 0.04 |
| lock-in η 0.02 + late L3 | 0.502 | 8.85 / 8.75 | 0.996 / 0.988 | 29.4 / 25.6 | 0.932 / 0.979 | 0.08 / 0.08 / 0.17 (whole) |
| same, test noise 0.01 | | 9.11 | 0.995 | 28.3 | 0.967 | 0.04 / 0 / 0.17 |
| lock-in η 0.02 + self-timed L3 (0.99) | 0.446 | 9.24 | 0.991 | 22.2 | 0.971 | 0 / 0 / 0.17 |
| lock-in η 0.05 + late L3 | 0.681 | 13.64 / 13.77 | 0.992 / 0.952 | 24.7 / 18.6 | 0.753 / 0.709 | 0.29 / 0.29–0.54 / 0.50 |

- The shift-rule network is the robust one: local except the eigenvector
  step, learned under noise, and tested under noise, it navigates like exact
  L1.
- For these lock-in networks, kernel alias ranks the probe cost (but clean
  all-unit networks spread 0.19–0.34 across runs, so it is a coarse guide). Alias 0.34
  from uneven counts alone (no noise) cost only basin (§21.1); alias 0.50 under
  noise costs direction (9°) and some goals; alias 0.68 costs a third to a
  half of the goals. At 3M steps and η 0.01 the scaling study reached alias
  0.42–0.51 (22.2); that network was not probed.

#### 22.4 Test-time noise (`run.py --grid_noise`, whole region)

The decoding bank (the probe's ruler for which cell a recalled state is) stays
clean, and so does the direction readout's local basis; stored goals and cues
are noisy. A first run with a noisy ruler scored
exact retrieval and basin wrongly (neighbouring cells' codes differ by less than
the noise) and was rerun; reach and direction were unaffected.

| network | test noise | abs err (°) | exact | basin | reach | dead (K = 5 / 10 / 20) |
|---|---|---|---|---|---|---|
| ideal r = 16 (clean code identical at any bump) | 0 | 1.04 | 1.000 | 32.1 | 0.998 | 0 / 0 / 0 |
| ideal r = 16, bump 0.10 | 0.01 | 25.97 | 0.328 | none | 0.583 | 0.29 / 0.42 / 0.38 |
| ideal r = 16, bump 0.10 | 0.03 | 84.24 | 0.020 | none | 0.029 | 1 / 1 / 1 |
| exact L1 + depression L3 | 0 | 1.12 | 1.000 | 34.4 | 0.994 | 0 / 0 / 0 |
| exact L1 + depression L3 | 0.01 | 1.66 | 1.000 | 31.7 | 0.997 | 0 / 0 / 0 |
| exact L1 + depression L3 | 0.03 | 3.35 | 0.811 | 10.8 | 0.976 | 0 / 0 / 0.04 |
| shift-rule L1 + depression L3 | 0.01 | 1.88 | 1.000 | 32.1 | 0.993 | 0 / 0 / 0 |
| shift-rule L1 + depression L3 | 0.03 | 3.65 | 0.845 | 15.1 | 0.961 | 0.08 / 0 / 0.08 |
| lock-in L1 + depression L3 | 0.01 | 1.94 | 1.000 | 28.4 | 0.987 | 0 / 0 / 0 |
| lock-in L1 + depression L3 | 0.03 | 3.33 | 0.892 | 13.6 | 0.941 | 0.04 / 0 / 0.04 |

- Both encoders sum all cells of a module, so averaging is not the
  difference. The ideal decodes each module's phase, then multiplies the
  noisy phase by the harmonic j, so its error grows with |j|; as implemented
  j runs up to l − 1, where |j| ≤ l/2 gives the same clean code (§23). The
  network reads the j-th Fourier coefficient directly, so its error goes as
  noise / A(j). At 0.01 the network is unaffected; at 0.03 it loses exact
  retrieval at a distance (basin 11–15) but still reaches 94–98% of goals.

Audit rows (§23): the ideal with symmetric harmonics, and noise in the readout
basis too.

| network | test noise | abs err (°) | exact | basin | reach | dead (K = 5 / 10 / 20) |
|---|---|---|---|---|---|---|
| ideal r = 16, symmetric harmonics | 0.01 | 3.12 | 0.768 | 16.8 | 0.987 | 0 / 0 / 0.04 |
| ideal r = 16, symmetric harmonics | 0.03 | 38.96 | 0.079 | none | 0.233 | 0.75 / 0.50 / 0.88 |
NOISY_BASIS_ROWS

## 23. Audit of the final model and its claims (2026-11-08)

Scope: everything behind §21–22, from L1 learning to the probe. Three
independent reviews, each given the code and the claims rather than my
conclusions (L1 learning and leakage; L2/L3, network construction and kernel
metrics; the Hopfield evaluation), plus checks of my own.

**Bottom line.** No supervision reaches any network's weights: every learner
reads only the walk's grid codes and each step's direction, and the exact
patterns, exact shift and positions appear only in diagnostics and evaluation.
The evaluation is wired correctly. There was one real bug (noisy trace-rule
gains), several overstated claims (corrected in place above), hyperparameters
tuned against the truth (now stated), and two evaluation choices that narrow
the noise claims.

**Bug**

| finding | consequence | status |
|---|---|---|
| `local_slowness` under noise dropped slow units whose at-rest-corrected change per step came out ≤ 0 by sampling error (v = inf, gain 0) | 8–13% of the kernel's weight lost, all from its broadest units: half-width ~17 instead of 18.5–19, alias slightly higher. Affects every noisy trace-rule network (§21 noise rows, §21.1 noise table, §22.2), learned and twin alike, so learned-vs-twin comparisons stayed fair. Depression with rest recovery (the probed §22.3 networks) was not affected | Fixed (clamp at 0; only units that never change are dropped), with a regression test that fails on the old rule (3 of 10 slow units dropped). Reruns: AUDIT_RERUNS |

**Supervision and tuning**

- The learners (`lms_shift` / `ShiftLMS`, `lock_in` / `LockIn`, `_init_units`,
  `eig_pairs`) read only the start and end codes of each step and its axis.
- Tuned against the truth: lock-in η 0.2 and turn ratio 0.001 (chosen by
  purity and coverage, `gwn_l1_lockin2.py`), shift-rule rate 0.5 (by
  |T − P|). These are global scalars, chosen on other seeds and stable across
  the 14-box, 20-box and scattered steps. The turn ratio 0.001 is at the edge
  of the tested grid, and no label-free way to choose it was tested.
- Oracle picks now labelled as such: §22.2's per-checkpoint best rate (by
  |T − P|) and best η (by impurity), and "one unit per pattern" (each unit's
  pattern found with the true patterns). The §22.2 rerun also builds networks
  at a fixed rate with no selection.
- Pair scale under noise: the shift-rule pairs were scaled to unit power with
  clean codes over the visited cells. An at-rest-corrected estimate from noisy
  codes gives the same scale within 0.5% (the raw noisy estimate is off by up
  to 5.6%), so the clean scale stands in for a local mechanism and changes
  nothing.

**Overstated claims, corrected in place**

- "No offline step anywhere" (§21.1): that network's pair scale was a batch
  average over visited cells; §22.1's running-power version is the online one.
- "L3's window opens once L1 has settled": it opens at a fixed time chosen in
  advance (long after L1 settled); the self-timed version does worse.
- "Exactly the sequential gains": the same to three decimals, from different
  walks.
- "Rest recovery works as the trace rule did": single runs, no same-L1 control.
- "More boxes trim the alias from 0.30 to 0.23": run-to-run spread is at least
  0.14 (the same recipe gave 0.192 and 0.335).
- "Depression gives the same gains as the trace rule": the kernel metrics
  agree, without noise; gains were not compared one by one.
- "Navigates like the ideal": reach and exact retrieval are like the ideal's;
  basin is about 2 cells smaller in paired comparisons (concurrent late window
  −2.2 ± 0.6, §21.1 lock-in −1.9 ± 0.7), direction error 1.5–2.6° against 1.0°.
- "One unit per pattern gives 0.07–0.10" read as a result of the route: it is
  an oracle what-if.
- The fixed-point test showed that a pattern with its own turn and a learned
  predictor is a fixed point, not that the rule converges to it (attraction
  rests on the runs). The test now requires purity above 1 − 1e-6; a wrong
  turn drifts to 0.996–0.9999 in the same steps.

**Evaluation**

- Verified: each probe task loads the encoder it names; grid wave network
  encoders use their stored bump (0.10), and the ideal's override applies;
  noise is a fresh draw on every encode; the decoding bank is clean in both
  places it is built; reruns overwrote their JSON and the summary holds no
  stale rows; the ideal reference rows used the same settings.
- **The ideal's harmonics.** `IdealEncoder` multiplies each decoded phase by
  j in 0 … l − 1; the representative with |j| ≤ l/2 gives the identical clean
  code and amplifies phase noise about half as much. Option added
  (`ideal:r=16,sym=1`, off by default so older results reproduce).
  With it, at test noise 0.01 the ideal reaches 0.987 of goals
  (exact 0.77, basin 17) instead of 0.58; at 0.03 it reaches 0.23 against the
  network's 0.976. The ordering holds, but "the ideal fails at noise 0.01" was
  mostly an artefact of the harmonic choice.
- **The readout basis stays clean** in every noisy-test task. The noise claims
  cover noisy cues and stored goals; a basis built from single-step
  differences of noisy codes is mostly noise (code noise 0.24 against a
  one-cell step of 0.06). NOISY_BASIS_AUDIT
- **Regions.** At K = 5 the scored environments lie in one row along the
  x = 0 edge: "whole" is an edge strip (x 4–232), "corner" lies inside it, and
  "centre" and "opposite" are translates of "corner". Cues past the edge are
  dropped from the basin disc, which can only inflate basin (35 of 40 corner
  goals, 10 of 40 whole goals). This holds for every encoder, so comparisons
  stay paired and fair; absolute basins and whole-vs-corner contrasts are
  affected.
- Test D follows a field computed once per cell with one noise draw, not
  redrawn on each visit.
- Statistics: at K = 5, basin rests on 16 (world, env) pairs, reach on 40 ×
  400 starts, dead goals on 24 goals (steps of 0.042; "0.17" is 4 goals).
  Settings and seeds are identical across tasks, so comparisons are paired.
  Each learned network comes from one training walk; training-seed variance
  was not sampled, so close learned variants (basin 29.9 against 28.2) cannot
  be ranked.

**Assumptions, not bugs**

- An idealised lattice world: unit, axis-aligned steps; integer periods;
  square cell sheets. The animal knows each step's axis and sign (velocity
  input) and which module each cell belongs to.
- L1 units have paired (complex) outputs, and L2 can read either sign of the
  sine part (the conjugate pair, pattern −j).
- L2 multiplies its three inputs exactly (§20: a few percent error is
  tolerated).
- Pruning to 99.5% of Σw² is a global compute economy; its 6,000 cap never
  binds.
- r = 16 and κ = 0.005 are fixed by design.

**Minor**

- α = 0.618 leaves an eigenvalue gap of 0.043 for module 11, comparable to the
  noisy |T − P| (0.03–0.06), so near-degenerate eigenvectors can mix under
  noise.
- `G.ETAS` was set at import by three scripts; now set at run time. The
  `LockIn` equivalence test now covers rules 2a and 2d too.
