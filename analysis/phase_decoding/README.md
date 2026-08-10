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

### Comparing across recurrent cells

`decodability` standardizes `h` before fitting (`StandardScaler` on the train
rows), so it is comparable between policies whose hidden states live on
different scales. **`parallelism_score` does not.** It is a raw cosine between
centroid differences: the difference cancels any constant offset, but not a
per-feature rescaling, so the score is weighted toward whichever units happen
to carry the most variance.

That distinction only started to matter when the trunk became selectable
(`--rnn_cell` / `--rnn_nonlinearity`, see `hopfield_nav/policy/recurrent.py`).
Measured on this pipeline at `hidden_size=128`, the share of total variance
sitting in the top 8 units is ~15% under `gru`/`tanh` and `rnn`/`tanh`, ~26%
under `rnn`/`relu`, and ~36% under `rnn`/`softplus` — which also idles at a
positive DC offset of `softplus(0) = 0.693` per unit and a hidden norm roughly
8-12x the tanh cells'.

So: decodability is comparable across cells; **parallelism is not**, without
standardizing first. `parallelism_score` is deliberately left unnormalized
rather than "fixed", because changing it would silently move every number in
the existing `results/` tree. If you want the cross-cell comparison, standardize
`h` at the pooling boundary on train-arena statistics, as an explicitly separate
path.

`hopfield_nav/tests/test_phase_decoding_cells.py` runs this pipeline end-to-end
against all four cells.

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

Tests at `hopfield_nav/tests/test_phase_decoding.py`.

## Running

### Exp 1 (bars)

```bash
CKPT=/path/to/ckpt.pt sbatch analysis/phase_decoding/run_exp1.sh
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
sbatch analysis/phase_decoding/run_exp2.sh

# Or collect fresh:
CKPT=/path/to/ckpt.pt sbatch analysis/phase_decoding/run_exp2.sh
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
./run_tests.sh hopfield_nav/tests/test_phase_decoding.py
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
