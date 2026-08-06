# Archive

Documents and modules that are no longer current but are worth keeping, moved
here by phase 6 of the 2026-08 refactor. Nothing here is maintained. Read it as
a record of what was believed and tried at the time, not as a description of the
code as it stands.

**Where to look instead**, for anything about the code as it is now:

| For | Read |
|---|---|
| What each module is and how they fit together | `docs/CODEBASE_MAP.md` |
| What training and evaluation actually do, flag by flag | `docs/TRAINING_AND_EVAL_REFERENCE.md` |
| Why the repo is shaped this way, and what is left to do | `docs/REFACTOR_ASSESSMENT.md`, `docs/REFACTOR_STATUS.md` |
| Grid/world coordinate conventions | `docs/coordinate_conventions.md` |

## What is here

### Experiment record (was `hopfield_nav/*.md`)

Nine files, ~104 KB. These are the running log of the project's experiments:
which variants were tried, what they scored, and what was concluded. They cite
each other by bare filename and those links still resolve, since they moved as a
set.

| File | What it is |
|---|---|
| `AUTORESEARCH.md` | The autonomous-research protocol: which knobs were tunable, how runs were to be logged |
| `EXPERIMENTS.md` | The main run log written under that protocol |
| `EXPERIMENTS_BC.md` | The behavior-cloning / DAgger run log (runs A–AO) |
| `PHASE_A_SIZE20.md` | The size-20 Phase-A recipe and workflow |
| `PHASE_A_SIZE20_FINDINGS.md` | Its append-only findings log |
| `PHASE_A_TRAINING.md` | What every Phase-A CLI flag does, as of `merge-workspace` |
| `RNN_BASELINE.md` | The RNN control model's design and results |
| `CODE_REFERENCE.md` | An earlier code map, superseded by `docs/CODEBASE_MAP.md` |
| `TRAINING_AND_EVALUATION.md` | An earlier training/eval reference, superseded by `docs/TRAINING_AND_EVAL_REFERENCE.md` |

Two caveats a reader should carry:

- **`CODE_REFERENCE.md` and `TRAINING_AND_EVALUATION.md` predate the refactor**
  and describe module paths, duplicated helpers and at-goal behavior that no
  longer exist. The 2026-08 docs replace them outright.
- **`PHASE_A_SIZE20.md` step 9 instructs the reader to append findings to
  `PHASE_A_SIZE20_FINDINGS.md`.** That instruction is retired along with the
  protocol; it is recorded here, not to be followed.

### Retired config schedules

`phase_schedules.md` holds the docstring of `PhasedConfigV2`, a dataclass with no
importer that phase 6 deleted. Its schedule is not recorded anywhere else -- the
`EXPERIMENTS_PHASE2_V2.md` it cited was never written.

### Orphan modules

`orphan_modules/` holds two `encoder_training` modules that nothing imported:

- `encoder_training_trajectory.py` (was `encoder_training/trajectory.py`) --
  single-trajectory visualization over `cls.nav` / `cls.hopfield`
- `encoder_training_viz.py` (was `encoder_training/viz.py`)

They are kept as text, not as importable modules: both depend on `cls/`, which
phase 7 retires.

## What went to pool storage instead

Bulk artifacts are under `$CLS_RUNS` (`/orcd/pool/003/jackking/cls_runs` by
default), not in the repo:

| Path under `$CLS_RUNS` | What |
|---|---|
| `archive/notebooks/` | The 19 tracked notebooks plus their jupyter launcher and log (52 MB) |
| `archive/action_classifiers/` | `prime-thunder-220/model_final.pt` (4.6 MB) |
| `archive/root_strays/` | `image.png` and the four `smoke_traj*` figures |
| `refs/` | Wang et al. 2024, *Rapid Learning without Catastrophic Forgetting in the Morris Water Maze* |
| `results/phase_decoding_v1/` | The v1 phase-decoding results (1.1 GB), symlinked back as `hopfield_nav/phase_decoding` |
