> **Archived.** Moved out of `hopfield_nav/` by phase 6 of the 2026-08
> refactor. Not maintained; describes what was believed and tried at the time,
> which in places is no longer true of the code. Start from `docs/archive/README.md`
> for what replaced it. Step 9's instruction to keep appending to `PHASE_A_SIZE20_FINDINGS.md` is retired with the protocol.

# Phase A on 20×20 — getting started

**This document is the only thing you need.** Ignore the many `EXPERIMENTS_*.md`, `experiment_tracking_*` etc files in this directory — they're stale logs from earlier waves.

## Terminology

The **bit** = the `--input_goal_in_memory` CLI flag. When on, the policy receives a 1-bit input that is True iff the env's goal pattern has been stored in the Hopfield. **Bitless** = without this flag — the policy must infer the same thing from natural inputs (Hopfield recall, sensory, prev_reward, etc).

## The problem

Train a **bitless** RNN policy that can:
1. **Explore** when the env's goal is **not** in the Hopfield (no useful recall signal).
2. **Follow** the Hopfield direction signal when the goal **is** stored.

Phase A trains both behaviors together (different rollouts), without training the store action. Phase B handles store later — empirically, learning to store at goal on top of a working follow/explore policy is easy.

## Metrics that matter

Run eval against several distractor counts (e.g. d=0/5/10):
- **`mean_coverage`** — fraction of the grid the agent visits during empty-Hopfield rollouts. Want >0.5. Just needs to be high enough to be plausibly random over the grid; the absolute number isn't sacred, but anything <0.3 means the agent is getting stuck (corners, repeated paths).
- **`success_rate`** — probability the agent reaches the goal during nav rollouts (goal pre-stored).
- **`mean_steps`** — average steps to goal (success-conditioned). Beware: a low mean_steps with low sr is misleading because failures are excluded; combine the two via censored mean steps `cst = (mean_steps × n_succ + max_steps × n_fail) / n_total` for a fair follow-speed metric.

## Why it's hard, and the fix

The hard case for the agent is distinguishing "Hopfield is recalling something useful" (follow it) from "Hopfield is recalling distractor noise" (ignore it, explore). With `--input_goal_in_memory`, this bifurcation is trivially learnable — the agent gets a 1-bit label. **We don't want to give that bit.**

Without the bit, the only natural cue is the structure of the recall itself. The recipe that works (8×8 + size 20):

- **`--input_hopfield_multistep 1 2 3`** — adds the projected recall at Hopfield iterations 1, 2, and 3 as 6 extra input dims (each is 2-D in continuous mode). Recall trajectory carries "is this attractor real" information that a single converged recall doesn't.
- **`--input_hopfield_raw`** — the main `hopfield_signal` becomes raw projected q (with magnitude) instead of a unit-vector direction. **Open question whether raw vs normalized is better here**; we haven't ablated it carefully on size 20.
- **`--input_sensory`** — required. Without sensory the agent corner-traps in explore mode (V18d27/V18d38/V18d41 all confirmed this).
- **No `--input_goal_in_memory`** — bitless.
- Reward shape: `--phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05`. Novelty 0.3 was better than 0.1 on size 20.
- **`--freeze_log_std`** is NOT required — unfreezing didn't hurt (V18d42). May be worth exploring with size 20.

## Best 20×20 model so far

`sunny-dew-92`. TIMEOUT at u180 of a 200u job that itself was a continuation of earlier resumes (this is why we accumulated training-length signal).

| ckpt | cov | sr |
|---|---|---|
| `/orcd/home/002/jackking/cls/checkpoint/phase_a_only_sunny-dew-92/phase_a_u180.pt` | 0.533 | 0.92 |
| `/orcd/home/002/jackking/cls/checkpoint/phase_a_only_sunny-dew-92/phase_a_u140.pt` | 0.515 | 0.96 |

`u180` if you want highest cov; `u140` if you want best follow + decent cov.

For reference: oracle-store + lock-after-first-store sequential continual eval on `u140` gives mean_primary ≈ 0.97, mean_revisit ≈ 0.99 across both seeds tested. Effectively V18d20-class continual performance.

## Recommended next steps — single long training run

The strategy is **not** chained-resume. Instead, run a single long job. `pi_evelina9` allows up to **7 days** wall-clock per job. Use it.

Starting config (variant `v18d39_size20_v15` in `run_phase_a_sweep_evelina.sh`, but with much higher `--phase_a_updates`):

```bash
EXTRA="--warmup_explore_only_updates 0 \
  --phase_a_updates 3000 \
  --phase_a_novelty_reward 0.3 \
  --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 \
  --interleave_empty_fraction 1.0 \
  --interleave_empty_target 0.50 --interleave_anneal_updates 50 \
  --move_ent_coef 0 --no-novelty_anneal \
  --size 20 --steps_per_rollout 400 \
  --eval_every 20 \
  --init_log_std -1.8 --freeze_log_std \
  --epsilon_explore 0.4 --epsilon_anneal_updates 200 \
  --no-input_prev_action --input_hopfield_raw \
  --input_hopfield_multistep 1 2 3 \
  --novelty_scale_remaining --novelty_scale_cap 10 \
  --explore_goals_off \
  --envs_per_world 80 \
  --n_train_distractors_min 0 --n_train_distractors_max 10 \
  --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 \
  --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
```

Update `run_phase_a_sweep_evelina.sh`'s `#SBATCH --time` to `--time=7-00:00:00` for the long run.

## Things to avoid (already tested, didn't work)

- **`--phase_a_novelty_reward 0.5`** on size 20 (V18d39_size20_v14): regressed cov from 0.40 to 0.36. 0.3 is the sweet spot; pushing higher destabilizes a converged policy.
- **Sustained ε** (`--epsilon_anneal_updates 750` instead of 200) on size 20 (V18d39_size20_v8): hurt sr (0.68 vs 0.89) — too much random action during follow phase.
- **Curriculum size 14 → size 20** (resume size-14 ckpt on size 20, V18d39_size20_v10): NaN-crashed at u40. Size shift on resume is unstable.
- **`--no-input_sensory`**: corner-traps. Sensory is load-bearing.
- **New reward shape alone** (`wall_penalty 0.1 + persistence_bonus 0.05` *without* novelty 0.3 + multistep): cov caps around 0.15. The shape needs to be paired with the multistep input + sensory.
- **`--no-freeze_log_std` on bare V18d27** (V18d35): PPO collapsed log_std → 0.13, made follow sharper but corner-trap remained. With multistep input though (V18d42), unfreezing was safe — worth re-exploring on size 20.

## Key files

- **`train_phase_a_only.py`** — training entry point. Look here for CLI flags (`p.add_argument(...)`).
- **`config.py`** — `AgentConfig`, `EnvConfig`, etc. New flag I added: `input_hopfield_multistep: list[int]`.
- **`hopfield.py`** — `recall_batch_trajectory` returns recall states at requested iterations. This is what powers the multistep input.
- **`rollout.py`** — `_compute_multistep_q` projects each snapshot via Gram-Schmidt. The main rollout loop concatenates these to the RNN input.
- **`agent.py`** — `compute_input_dim` accounts for multistep dims when continuous mode.
- **`eval.py`** — `_agent_step` (used by all evals) also applies the multistep recall when `cfg.agent.input_hopfield_multistep` is non-empty.
- **`run_phase_a_sweep_evelina.sh`** — sbatch wrapper. New variant cases get added under the case statement; submit with `VARIANT=v18d39_size20_<your_tag> sbatch run_phase_a_sweep_evelina.sh`.
- **`eval_distractors.py`** — quick post-hoc nav/disc/expl eval at d=0,5,10.
- **`run_seq_continual_oneseed.sh`** — sequential continual eval (oracle store + lock variant). Good for the V18d20-class comparison once Phase B is done; for Phase-A-only ckpts, use `--oracle-store-at-goal --lock-store-after-goal` flags so the eval bypasses the untrained store policy.

## How to iterate (running experiments yourself)

Workflow per experiment:

**1. Add a new variant to `run_phase_a_sweep_evelina.sh`.** Find the `case $VARIANT in` block; add a new case at the top (next to the existing `v18d39_size20_*` entries):

```bash
v18d39_size20_<your_tag>)
  EXTRA="--phase_a_updates 3000 ... --size 20 ..."  # see Recommended config above
  ;;
```

**2. Set the SBATCH time appropriately.** Edit `#SBATCH --time=` near the top of `run_phase_a_sweep_evelina.sh`:
- For exploratory short jobs: `0-08:00:00` (the existing default, fits ~280 updates on size 20).
- For long runs: `7-00:00:00` (max on `pi_evelina9`).

If you want to run on a different partition (e.g. `mit_normal_gpu`, max 6h), copy `run_phase_a_sweep_evelina.sh` to `run_phase_a_sweep_normal.sh` and `sed -i 's/pi_evelina9/mit_normal_gpu/'` + adjust `--time`.

**3. Submit:**

```bash
cd /home/jackking/cls
VARIANT=v18d39_size20_<your_tag> sbatch /orcd/home/002/jackking/cls/hopfield_nav/run_phase_a_sweep_evelina.sh
# returns: Submitted batch job <jobid>
```

The job's slurm log lands at `/home/jackking/cls/hopfield_nav/logs/slurm_phase_a_sweep_<jobid>.out`.

**4. Monitor.** Quick options:

```bash
# what's running and how long
squeue -u $USER -o '%.10i %.20j %.8T %.10M'

# tail the eval lines as they come in (each evaluation is ~every 20 updates)
tail -F /home/jackking/cls/hopfield_nav/logs/slurm_phase_a_sweep_<jobid>.out \
  | grep -E "phase_a_u[0-9]+\] expl=|nav=|Traceback|Error|FAILED|OOM|Killed|Done\. Saved"
```

The wandb run name (e.g. `bright-flower-95`) prints near the top of the slurm log. Checkpoints save under `/orcd/home/002/jackking/cls/checkpoint/phase_a_only_<run_name>/phase_a_u<N>.pt` every `--eval_every` updates.

**5. Find the best checkpoint after training.** Parse the slurm log:

```python
# inline python to find best-cov ckpt
import re, ast
log = "/home/jackking/cls/hopfield_nav/logs/slurm_phase_a_sweep_<jobid>.out"
expl = {}
with open(log) as f:
    for line in f:
        m = re.search(r"\[phase_a_u(\d+)\] expl=(\{.*\})\s*$", line.strip())
        if m: expl[int(m.group(1))] = ast.literal_eval(m.group(2))
best_u = max(expl, key=lambda u: sum(v["mean_coverage"] for v in expl[u].values())/len(expl[u]))
print(f"best u={best_u}, cov={sum(expl[best_u][d]['mean_coverage'] for d in expl[best_u])/3:.3f}")
```

The training-time eval already runs `evaluate_navigation/exploration` against d=0,5,10 every `eval_every` updates, so the slurm log contains per-update sr/cov per distractor without re-evaluating.

**6. Post-hoc eval (more thorough than training-time eval).** From `/home/jackking/cls`:

```bash
module load miniforge/24.3.0-0 && source activate cls
python -m hopfield_nav.eval_distractors \
    --checkpoint /orcd/home/002/jackking/cls/checkpoint/phase_a_only_<run_name>/phase_a_u<best>.pt \
    --device cuda --num_trials 32 --max_steps 400 \
    --distractors 0 5 10
```

Prints a single-row table with sr / cst / cov / store_efficiency etc.

**7. Sequential continual eval (the V18d20-class comparison).** Phase A ckpts have an untrained store head, so the seq eval needs `--oracle-store-at-goal --lock-store-after-goal` to bypass it. The wrapper `run_seq_continual_oneseed_size20.sh` (or generate your own) takes a checkpoint + seed:

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=/home/jackking/cls/hopfield_nav/eval_results/seq2par_<your_tag>_$TS
mkdir -p "$OUT"
cd /home/jackking/cls
for s in 1000 2000; do
  CKPT=/orcd/home/002/jackking/cls/checkpoint/phase_a_only_<run_name>/phase_a_u<best>.pt \
  OUT_DIR="$OUT" SEED_OFFSET=$s \
  sbatch /orcd/home/002/jackking/cls/hopfield_nav/run_seq_continual_oneseed_size20.sh
done
```

The plot lives at `$OUT/seq_seed{1000,2000}_sequential.png`. For a clean version (no store markers, forgetting-style aesthetic), use `eval_all`'s `--no-show-stores` or call `save_sequential_episodes_plot(results, out_path, show_stores=False)` directly.

Note `run_seq_continual_oneseed_size20.sh` is set up with `--seq-iters-per-block 800 --seq-max-steps 100 --num-val-envs 5`. Adjust if needed (e.g. raise `--seq-max-steps` for size 20 if you want each rollout to allow more goal-finding).

**8. Iterate.** Skim the slurm log for the new variant's per-update cov/sr trajectory; if it's tracking the prior best, let it run. If it's regressing or NaN-crashing, `scancel <jobid>`, debug, resubmit. Aggressive single-knob ablations off the current best config are usually more informative than multi-knob changes.

**9. Log findings.** Append to a single living doc — `hopfield_nav/PHASE_A_SIZE20_FINDINGS.md` (create on first run if it doesn't exist). One short entry per variant:

```markdown
## v18d39_size20_<your_tag> — <one-line hypothesis>

- **Run:** wandb name `<run_name>`, ckpt `phase_a_u<best>.pt`, slurm job `<jobid>`.
- **Diff vs prior best:** "novelty 0.3 → 0.4", "+ unfreeze log_std", etc.
- **Result:** cov=X / sr=Y / cst=Z. Compare to prior best.
- **Verdict:** kept / discarded / inconclusive. One sentence why.
```

Don't add new `EXPERIMENTS_*.md` files (those are the stale ones to ignore). Don't write per-experiment markdown files. Just this one doc, append-only, dated entries. If you discover a finding worth promoting (e.g. "actually unfreezing log_std is required on size 20"), update the **Things to avoid** or **Recipe** section of *this* doc (`PHASE_A_SIZE20.md`) at the same time so the canonical guidance stays current.

## Open questions

1. **Raw vs normalized hopfield_signal** with multistep inputs. We have `--input_hopfield_raw` on in current best, but never ablated against normalized + multistep on size 20.
2. **Unfreezing log_std on size 20.** Worked on 8×8 (V18d42); not tested on size 20.
3. **Larger hidden_size** (1024 vs 512). Untested.
4. **Higher epsilon early but with sharper anneal.** Untested.
