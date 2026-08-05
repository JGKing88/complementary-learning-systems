# Phase decoding v2 — parallelism, decodability, PCA viz

Two analyses on a trained Hopfield-nav controller's hidden state:

- **Exp 1** — bar plots of cross-arena **parallelism score** and **decodability**
  across four split families (LOO, Random 80/20, Quadrant 1v3, Quadrant 3v1).
- **Exp 2** — PCA visualizations of MLP hidden activations:
  1. **Regular PCA** on per-trial explore vs. exploit data.
  2. **Trajectory PCA** on two-episode trajectories
     (start → run-to-goal → oracle-store → teleport without resetting `h_rnn`/Hopfield → run-to-goal).

## Definitions

- **Parallelism score** (per fold):
  - `v_train = centroid(exploit | train_arenas) − centroid(explore | train_arenas)`
  - `v_test = centroid(exploit | test_arenas) − centroid(explore | test_arenas)`
  - `score = cos(v_train, v_test)`
- **Decodability** (per fold): L2 logistic regression fit on pooled `(h, phase)`
  over `train_arenas`; balanced accuracy on pooled `test_arenas`.

Both metrics are averaged across folds within a split family for the bar height,
SD across folds for the error bars.

## Splits

- **LOO** — N folds, train on N−1 arenas, test on the held-out one.
- **Random 80/20** — `n_random_splits` permutations, first 20% as test.
- **Quadrant 1v3** — 4 folds; each goal-quadrant takes a turn as the *training*
  set, the other 3 quadrants are the test set.
- **Quadrant 3v1** — 4 folds; each goal-quadrant takes a turn as the held-out
  test set, the other 3 are training.

Goal quadrant assignment uses `q = 2 * (r >= env_size/2) + (c >= env_size/2)`.

## Layout

| File | Role |
| --- | --- |
| `rollout.py` | `RolloutEngine` (ckpt + envs + agent) and `EnvBundle` |
| `collect_trials.py` | `ExploreExploitCollector`, `TrialsDataset`, `TrialData` |
| `collect_trajectory.py` | `TwoEpisodeTrajectoryCollector`, `TrajectoryDataset` |
| `splits.py` | `Split`, `Fold`, `split_loo` / `split_random` / `split_quadrant_*` |
| `metrics.py` | `parallelism_score`, `decodability` |
| `classifier.py` | `MLPPhaseClassifier`, `train_mlp`, `extract_hidden` |
| `viz.py` | `plot_bars`, `plot_pca_scatter`, `plot_trajectory_pca` |
| `exp1.py` | Exp 1 entry point |
| `exp2.py` | Exp 2 entry point |
| `run_exp1.sh`, `run_exp2.sh` | sbatch launchers |

Tests at `hopfield_nav/tests/test_phase_decoding_v2.py`.

## Running

### Exp 1 (bars)

```bash
CKPT=/path/to/ckpt.pt sbatch hopfield_nav/phase_decoding_v2/run_exp1.sh
```

Defaults: 100 arenas, 100 starts per condition, max 200 steps, stochastic policy,
distractor count `Uniform[0, 5]`, 20 random 80/20 splits, seed 0.

Override with env vars: `NUM_ARENAS`, `N_STARTS`, `MAX_STEPS`, `N_DIST_MIN`,
`N_DIST_MAX`, `N_RAND`, `TEST_FRAC`, `SEED`, `POLICY_FLAG=--deterministic|--stochastic`,
`OUT`.

Outputs in `OUT`:

```
trials/per_arena/<idx>.npz
trials/meta.json
scaffold.json
metrics.json     # per-fold + summary
bars.png
```

### Exp 2 (PCA)

```bash
# Reuse trials from an existing exp1 run (cheaper):
CKPT=/path/to/ckpt.pt TRIALS_DIR=/path/to/exp1/trials \
sbatch hopfield_nav/phase_decoding_v2/run_exp2.sh

# Or collect fresh:
CKPT=/path/to/ckpt.pt sbatch hopfield_nav/phase_decoding_v2/run_exp2.sh
```

Defaults: 100 arenas, 50 starts (Part 1 if collecting fresh), 4 trajectories per
arena (Part 2), MLP hidden dim 64, 30 epochs.

Outputs:

```
exp2_regular_pca.png        # Part 1
exp2_trajectory_pca.png     # Part 2
trajectories/<idx>.npz
trajectories/meta.json (under OUT/meta.json)
exp2_meta.json
```

## Stochastic vs deterministic

Both entry points expose `--stochastic` and `--deterministic` (default
**stochastic**, sampled actions). Override via the launcher:

```bash
POLICY_FLAG=--deterministic CKPT=... bash run_exp1.sh
```

## Tests

```bash
./run_tests.sh hopfield_nav/tests/test_phase_decoding_v2.py
```

End-to-end smoke test runs in seconds on CPU with a tiny synthetic ckpt-shaped
fixture and exercises every module.

## Relationship to `phase_decoding/`

The original `phase_decoding/` directory remains untouched. `phase_decoding_v2/`
is a separate object-oriented redesign tailored to the Exp-1/Exp-2 spec:

- explicit parallelism-score (not just decoder LLR);
- by-quadrant arena splits;
- MLP-with-hidden-layer classifier (rather than a single linear axis);
- two-episode trajectory mode with explicit oracle store + no Hopfield/`h_rnn`
  reset across the teleport.
