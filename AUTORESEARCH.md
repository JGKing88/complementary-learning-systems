# autoresearch: encoder

This is an autonomous research loop for improving the distance encoder used in Hopfield-based grid cell navigation.

## Background

The encoder maps smoothed grid cell codes (g_hot with Gaussian smoothing, `fwhm_ratio=0.25`) to unit-sphere embeddings via an MLP. These embeddings are consumed by a Hopfield associative memory network for goal-directed navigation: given a goal position's encoding, the Hopfield network recalls it and the agent navigates by following the gradient of similarity in embedding space (Gram-Schmidt projection).

The core tension in the loss design is:
- **Local fidelity**: nearby grid positions must map to similar embeddings so Hopfield recall produces a useful gradient toward the goal
- **Global separation**: distant positions must be distinguishable so the agent doesn't confuse far-away locations

The metric that matters is **`val_nav/accuracy`** — the fraction of navigation episodes where the agent reaches the goal platform. This is evaluated on environments placed in regions of the grid that were NOT used during encoder training.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr5`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current HEAD.
3. **Read the in-scope files** for full context:
   - `AUTORESEARCH.md` — this file
   - `notebooks/train_dist_encoder.py` — the file you modify. Encoder architecture, loss functions, training loop, config.
   - `cls/encoder.py` — encoder model definitions (GridEncoder MLP). Read-only context.
   - `cls/eval/nav_eval.py` — navigation evaluation. Read-only context.
   - `cls/hopfield.py` — Hopfield network used in nav eval. Read-only context.
   - `run_sweep.py` — sweep entry point showing how config maps to `train()`. Read-only context.
   - `sweep_encoder_mlp.yaml` — existing sweep parameter ranges (shows what's been explored). Read-only context.
   - `eval_encoder.py` — standalone eval script. Read-only context.
4. **Verify GPU**: confirm CUDA is available (`python -c "import torch; print(torch.cuda.is_available())"`)
5. **Initialize tracking files**:
   - Create `results.tsv` with the header row
   - Create `research_log.md` for detailed notes
6. **Confirm and go**: confirm setup looks good with the user.

## Research phase

Before running any experiments, conduct a thorough research phase:

1. **Read all in-scope files** listed above, thoroughly
2. **Study the loss landscape**: understand what each loss term does (CKA, mod_loss, uniformity, plane loss) and how they interact
3. **Understand Hopfield recall**: what embedding geometry actually produces good navigation? Read `cls/hopfield.py` and `cls/eval/nav_eval.py` to understand what properties matter
4. **Review prior work**: look at sweep configs, commented-out code, and git history to understand what's been tried and what worked
5. **Form hypotheses**: build a ranked list of experiments with reasoning for each
6. **Document**: write all findings and the experiment plan into `research_log.md`

Only after completing the research phase and documenting your plan should you begin experimenting.

## Experimentation

Each experiment trains the encoder for **100 epochs** on a single GPU. You launch it as:

```
python notebooks/train_dist_encoder.py > run.log 2>&1
```

Navigation eval runs **only at the final epoch** (epoch 100) to save time. Set `eval_every` equal to the number of epochs so it fires once at the end.

**What you CAN modify:**
- `notebooks/train_dist_encoder.py` — this is the only file you edit. Loss functions, loss weighting, training hyperparameters, config dict, gain schedules, batch size, etc. You may add new loss functions or modify existing ones.

**What you CANNOT modify:**
- `cls/encoder.py` — encoder architecture is fixed (MLP with smoothed inputs)
- `cls/eval/nav_eval.py` — evaluation is the ground truth
- `cls/hopfield.py` — the Hopfield network is fixed
- `run_sweep.py`, `eval_encoder.py` — read-only
- Do not install new packages or add dependencies

**Constraints:**
- Encoder type: **MLP only** (no CNN)
- Input type: **smoothed** (Gaussian-smoothed grid codes)
- You may vary: loss functions, loss weights (`cka_alpha`, `cka_topk`, `mod_loss_lambda`, `uniformity_lambda_end`), learning rate, batch size, gain schedule, number of training environments (`Nenv`), architecture hyperparameters (`hidden_dim`, `num_hidden_layers`, `out_dim`), and any other training parameter
- You may add entirely new loss functions or restructure the loss computation

**Wandb:** All runs should log to the project **`autoresearch-encoder`** (not `dist-encoder`). Set `wandb_project` accordingly in the config.

**The goal is simple: maximize `val_nav/accuracy`.** Everything else (training loss, Pearson correlation, triplet accuracy) is diagnostic — only val_nav/accuracy determines keep/discard.

## Output format

The training script logs nav eval results like:

```
Val nav: acc=0.542 | steps=34.2 | speed=0.285
```

Extract the key metric:
```
grep "Val nav:" run.log | tail -1
```

If the grep is empty, the run crashed. Run `tail -n 50 run.log` to diagnose.

## Logging results

### results.tsv

Tab-separated, 5 columns (do NOT use commas in descriptions — they break TSV):

```
commit	val_nav_acc	status	description	notes
```

1. git commit hash (short, 7 chars)
2. val_nav/accuracy achieved (e.g. 0.542) — use 0.000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short description of what this experiment tried
5. brief notes on what was learned

Example:
```
commit	val_nav_acc	status	description	notes
a1b2c3d	0.542	keep	baseline	starting point for comparison
b2c3d4e	0.571	keep	increase uniformity_lambda to 0.8	uniformity helps; more separation
c3d4e5f	0.538	discard	remove mod_loss entirely	mod_loss is needed
d4e5f6g	0.000	crash	experimental contrastive loss	index error in new loss fn
```

### research_log.md

Maintain a detailed research log throughout the run. This is the primary documentation artifact. It should contain:

- **Research phase findings**: what you learned about the codebase, prior work, and the problem
- **Experiment plan**: ranked hypotheses with reasoning
- **Per-experiment entries**: for each experiment, document:
  - What you changed and why (hypothesis)
  - The result (val_nav/accuracy and any diagnostic metrics)
  - What you learned (does this confirm or refute the hypothesis?)
  - What this suggests for next experiments
- **Running summary**: periodically update a summary of key findings, what works, what doesn't

The research log should be detailed enough that someone reading it can understand the full arc of reasoning and experimentation.

**Do not commit `results.tsv` or `research_log.md`** — leave them untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr5`).

LOOP FOREVER:

1. Look at the current git state: branch, latest commit, current config
2. Form a hypothesis and modify `notebooks/train_dist_encoder.py`
3. `git commit` the change (short descriptive message)
4. Run the experiment: `python notebooks/train_dist_encoder.py > run.log 2>&1`
5. Read the results: `grep "Val nav:" run.log | tail -1`
6. If grep is empty, the run crashed:
   - Run `tail -n 50 run.log` to read the traceback
   - If it's a trivial fix (typo, wrong shape), fix and re-run
   - If the idea is fundamentally broken, log as crash and move on
7. Record in `results.tsv` AND write an entry in `research_log.md`
8. If val_nav/accuracy **improved** (higher than current best): keep the commit, advance the branch
9. If val_nav/accuracy is **equal or worse**: `git reset --hard HEAD~1` to revert
10. Update the running summary in `research_log.md`
11. Plan the next experiment based on what you've learned

**The first run**: always establish the baseline by running the script as-is (with only the eval_every and wandb_project changes).

**Timeout**: each run should take a few minutes. If a run exceeds 15 minutes, kill it (`kill %1` or find the PID) and treat as a crash.

**Crashes**: use judgment. Typos and shape mismatches — fix and retry. Fundamental issues — log, revert, move on.

**NEVER STOP**: once the loop begins, do NOT pause to ask the user anything. The user may be away from the computer. You are autonomous. If you run out of ideas:
- Re-read the in-scope files for angles you missed
- Try combining two near-miss changes
- Try more radical loss function redesigns
- Revisit discarded ideas with different parameterizations
- Look at the diagnostic metrics (Pearson, triplet accuracy) for clues about what's wrong
- The loop runs until the user interrupts you
