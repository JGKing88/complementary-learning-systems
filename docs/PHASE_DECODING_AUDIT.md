# Audit of the RNN explore/exploit abstraction analysis (`analysis/phase_decoding/`)

2026-10-14. Scripts in `analysis/phase_decoding/audit/`, outputs in
`$CLS_RUNS/results/phase_decoding_audit/`.

## Verdict

The idea behind the analysis is the accepted one: test whether the
explore/exploit variable has a consistent coding direction across contexts
(arenas), using a parallelism score and cross-context decoding (Bernardi et al.
2020, Cell). Neither the implementation nor the existing results support any
conclusion. The collector never produced exploit behaviour, so every number in
`results/` compares two kinds of polluted Hopfield memory, not explore against
exploit. Several metric choices also differ from the published method in ways
that inflate the numbers. Redo it from fixed collection; nothing in the
existing `results/` tree is reusable.

## 1. The collector does not produce exploit behaviour

`RolloutEngine.rollout` (`rollout.py:244`) writes `store_embedding` into the
Hopfield every step the store head fires. Neither training eval does that:
`evaluate_navigation` never stores, and `evaluate_goal_discovery` lets the
head fire only on the step that begins at the goal. The store head fires on
about 55% of steps, so within a few steps both conditions hold dozens of
self-written memories and differ by one buried goal pattern.

Same checkpoint, same arenas, 20 trials per arena, sampled policy, 100-step cap:

| checkpoint | `evaluate_navigation` success | collector exploit | collector explore |
|---|---|---|---|
| d0_base u725 (recorded world, 6 arenas), 0 distractors | 1.00 (12.1 steps) | 0.16 | 0.18 |
| d0_base u725, 5 distractors | 1.00 (14.3 steps) | 0.19 | 0.10 |
| glamorous-field-97 u360 (10 arenas), 0 distractors | 0.98 (30.4 steps) | 0.16 | 0.13 |
| glamorous-field-97 u360, 5 distractors | 0.90 (30.3 steps) | 0.16 | 0.14 |

`store_diag.py` isolates the cause on d0_base u725. Exploit success is 0.21
with the collector's stores, 1.00 with stores suppressed, and 1.00 with stores
suppressed plus a reset of heading and previous displacement. The existing
exp1 trials show the same failure: exploit reach 0.11 and explore reach 0.10
for a checkpoint whose training eval reported 0.97.

`TwoEpisodeTrajectoryCollector` (Exp 2) uses the same `rollout`, so its
episodes are polluted the same way.

## 2. Other implementation defects

1. `run_exp1.sh:39` hardcodes `CKPT=.../glamorous-field-97/phase_a_u360.pt`.
   The `CKPT=... sbatch run_exp1.sh` usage in the README silently analyses
   glamorous-field-97 whatever is passed. `run_exp2.sh` reads `CKPT` correctly.
2. Exp 2 Part 1 (`exp2_regular_pca.png`) was computed on the random-agent
   control's trials. Its `trials_dir` is
   `exp1_..._stochastic_random_init_seed0/trials`, and that exp1 run has
   `random_agent: true`. `TrialsDataset.meta` does not record `random_agent`, so
   nothing flagged the mismatch.
3. The May results used an RNG-replayed world. Until a9b6c6a (2026-08-12)
   `rollout.py` replayed the world, and offsets came out up to half an env
   width off. glamorous-field-97 has no `world.json`, so current code still
   replays it. Its navigation eval still reaches 0.98 on the replayed world, so
   this is not what broke exploit. It does mean the checkpoint cannot be scored
   on its own training world.
4. The headline run is deterministic. The stochastic trained-agent run never
   produced `metrics.json`. The random-agent control ran at 50 starts × 50
   steps while the trained run used 100 × 100, so the control is not matched.
5. `set_position` resets neither `_last_displacement` nor heading, so each
   trial's first step sees the previous trial's displacement and sensory
   heading. Measured effect on success: none. Fix it anyway.
6. The bar error bars are SD across folds. The 20 random 80/20 folds share
   most of their arenas and the quadrant families have 4 folds, so these are
   not confidence intervals.

## 3. Method compared with the published one

Bernardi et al. (2020) and later work (Courellis et al. 2024, Nature; the
`decodanda` package) define the measures as follows.

| | published method | this pipeline |
|---|---|---|
| Parallelism | one coding vector per matched condition pair (here: per arena, exploit minus explore); mean cosine over all pairs of coding vectors; 1,000-shuffle null | cosine between two pooled centroid differences (80 arenas vs 20); no null |
| CCGP | train a linear decoder on a few conditions, test on held-out conditions; geometric random-model null | train on 80–99 arenas, test on the rest; no null |
| Resampling unit | trials (pseudo-populations); temporal neighbours never straddle train and test | individual timesteps; the within-arena "ceiling" splits timesteps of one trial across train and test |
| Confounds | explore/exploit states checked against reward history, outcome, and kinematics (Ebitz et al. 2018, Neuron) | none |
| Visualisation | unsupervised projections, or supervised ones on held-out data next to a shuffled-label version | PCA of the hidden layer of an MLP trained on the phase labels, plotted on its own training data |

Measured on the existing trials (`geometry_check.py`; trained = glamorous-field-97
deterministic, random = its random-init control):

| quantity | trained | random init | shuffle null (trained) |
|---|---|---|---|
| per-arena pairwise parallelism (published) | 0.86 | 0.00 | 0.00 (max 0.01) |
| pooled cosine, 1 arena per group | 0.86 | -0.02 | |
| pooled cosine, 10 arenas per group | 0.99 | -0.05 | |
| pooled cosine, 80/20 (pipeline) | 0.996 | -0.19 | 0.04 (max 0.42) |
| within-arena decoding, timestep split (pipeline) | 0.94 | 0.83 | |
| within-arena decoding, trial split | 0.82 | 0.65 | |
| train on one arena, test on others | 0.67 | 0.53 | |
| time-in-trial only, cross-arena | 0.50 | 0.50 | |

Three things follow. Pooling turns 0.86 into 0.996, because averaging over
arenas removes the arena-specific part of each coding vector. The pooled null
also widens to a maximum of 0.42. Timestep splits add 0.12 to 0.18 to
within-arena accuracy, so the "ceiling" bar is mostly leakage for the random
agent. Training on one arena gives 0.67, against 0.79 for the pipeline's
80-arena training set, which is the gap between CCGP and ordinary held-out
decoding. These numbers come from the broken collector. They show how the
metrics behave, not anything about explore and exploit.

## 4. Conceptual issues the redo must handle

- **The phase variable is an input difference.** Phase is set by what the
  Hopfield holds, and the policy reads raw `q`. With no distractors, explore
  `q` is exactly zero. Decodable phase in `h` can therefore be the input passed
  through. The redo needs an input-only baseline: run the same metrics on the
  policy input vector, and claim a representation only where `h` exceeds it,
  or where `h` keeps phase while the input does not (for example in steps
  with matched `||q||`).
- **Behaviour differs once exploit works.** A working exploit episode is short
  and straight and ends near the goal. Explore is long and visits walls. Phase
  is then confounded with time in episode, distance to goal, speed, and wall
  contact. Bin by these variables and compute the metrics within matched bins,
  or regress them out first (Ebitz-style GLM).
- **CCGP should train on few contexts.** Report it as a function of the number
  of training arenas (1, 2, 5, 10), with the geometric random-model null.
- **Exp 2's switch is also a teleport.** The switch coincides with an oracle
  store and a jump in position, and 1,677 of 2,000 trajectories were dropped
  because episode 1 had not reached the goal within 500 steps. The
  task-faithful models (search, store once at the goal, continue) give a
  natural within-episode switch with no teleport.

## 5. Proposed redo

1. Fix the collector. Exploit episodes honour no stores (as in
   `evaluate_navigation`). Explore episodes end at goal arrival, or store only
   on that step (as in `evaluate_goal_discovery`). Add a test that pins the
   collector's per-trial success to `evaluate_navigation` on a fixture.
2. Record per step: policy input vector, `h`, time, position, distance to
   goal, speed, `||q||`, and memory count. Record `random_agent` and the git
   SHA in `meta.json`.
3. Checkpoints: d0_base u725 and task3r_k2_h128 (recorded worlds), plus a
   random-init control at matched settings. The recorded d0_base world has 6
   validation arenas. Reaching around 100 arenas means minting a `--split`
   arena set (`held_out` levels).
4. Metrics: per-arena pairwise parallelism with a trial-shuffle null; CCGP
   against the number of training arenas, with trial-grouped resampling and a
   geometric null; the same metrics on the input vector; and behaviour-matched
   versions. Sampled policy throughout.
5. Visualisation: unsupervised PCA of `h`, and `h` projected on a decoder axis
   fitted on held-out arenas.

## References

- Bernardi et al. 2020, Cell, https://pmc.ncbi.nlm.nih.gov/articles/PMC8451959/
- Courellis et al. 2024, Nature, https://pmc.ncbi.nlm.nih.gov/articles/PMC11338822/
- Johnston & Fusi 2023, Nat Commun, https://pmc.ncbi.nlm.nih.gov/articles/PMC9950464/
- decodanda (Posani), https://github.com/lposani/decodanda
- Ebitz, Albarran & Moore 2018, Neuron, https://pmc.ncbi.nlm.nih.gov/articles/PMC5774994/
- Varoquaux et al. 2017, NeuroImage (cross-validation on autocorrelated data), https://arxiv.org/abs/1606.05201
- Kriegeskorte et al. 2009, Nat Neurosci (circular analysis), https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/
