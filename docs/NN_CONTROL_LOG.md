# Goal-conditioned NN control — log

Companion to `NN_CONTROL_PLAN.md`. The plan says what; this says what
happened, in order, with numbers. Newest at the bottom.

Branch `worktree-nn-generalization-control`. Env python:
`$HOME/.conda/envs/cls/bin/python`.

---

## 2026-09-10 — build, §5.9 step 1

**Config (§5.1).** Added `SENSORY_MODES`, `SENSORY_ENCODERS`;
`RNNAgentConfig.input_sensory / sensory_mode / input_xy_state /
sensory_encoder*`; `RNNTrainConfig.region_val_frac / pairs_per_env /
resample_goal_on_reach`. All default to today's behaviour.

**FeedForwardCore (§5.3).** In `policy/recurrent.py`, dispatched on
`rnn_cell == "mlp"`. Returns zeros as its state rather than echoing the
input state, so a lifetime caller that reads it sees a network that
provably keeps nothing. Checked: T-step call equals T single-step calls
(`allclose` True on a 3-layer relu core).

**Decision — A's model does not subclass RNNAgent.** `PairRegressor` is
`FeedForwardCore` + one `Linear`. No `Normal`, no `log_std`, no `act()`.
The plan's reason (§3.1) stands: A is a regression, and a Gaussian head
with nothing to sample is a source of ambiguity, not a model. The cost is
a thin adapter for B later so both go through one evaluator.

**Layout as data (§5.2).** `rnn_input_layout` in `policy/agent_rnn.py`;
`compute_rnn_input_dim` sums it; `build_rnn_input` assembles from it. The
four historical channels keep their lenient `is not None` guard (callers
rely on it); the three new ones raise when enabled-but-missing. Producers
`sensory_vec(omni)`, `goal_sensory_vec(omni|north)`, `xy_vec` in
`rollout/rnn.py`. Existing tests, golden fixtures and layering: green.

**Region holdout (§5.5).** `GeneratedSplit.region_cells` (JSON default:
empty), `generate_split(region_frac=)`, `CellSets` with the invariants
checked at construction, `split.cell_sets()`.

**Found: the goal partition was one-cell-per-env, not a fraction.**
`generate_split` with `refresh_goal=False` draws `n_train` distinct goal
cells (one per env) and puts *everything else* in `goal_cells_val` — the
first build gave `goal_train = 64, goal_heldout = 296`. The `val_frac`
partition the plan assumed only runs under `refresh_goal=True`. Since the
pair sampler draws goals from the whole train partition, that is the
partition it needs; `rnn_world` now passes `refresh_goal=True` whenever
`region_val_frac > 0`. Configured world at size 20, 64 + 16 envs, margin
20, `lambdas = [11, 12, 13]`: `start_train 360 / goal_train 320 /
goal_heldout 40 / region 40`, `region ⊂ goal_cells_val` True. Build 4 s.
