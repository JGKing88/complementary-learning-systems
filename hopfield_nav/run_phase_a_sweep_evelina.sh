#!/bin/bash -l
#SBATCH --job-name=hnav-phase-a-sweep
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_evelina9
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_phase_a_sweep_%j.out

# ---------------------------------------------------------------------------
# 2026-08-05 -- behavior note for anyone re-running an older variant.
#
# Every variant below runs through `hopfield_nav.train_phase_a_only`, which has
# always honored --goal_radius. The sibling entry point `hopfield_nav.train`
# did NOT: setup_train_world built its GridEnvs by hand and dropped
# goals_active / goal_reward / goal_radius, so VecEnv (which reads them off the
# base env) silently used the GridEnv defaults during training while eval used
# the configured values. Fixed on this date. Variants recorded here are
# unaffected; runs launched from `train.py` with --goal_radius != 0.5 before
# this date trained at radius 0.5 regardless of the flag.
#
# Also added on this date: --allow_offcell_store, default True, which preserves
# existing behavior. See EnvConfig.allow_offcell_store.
# ---------------------------------------------------------------------------

SEED=${SEED:-42}
VARIANT=${VARIANT:-v6b}

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate cls
# wandb auth comes from ~/.netrc (machine api.wandb.ai). Run `wandb login`
# once if it is missing; never paste an API key into a tracked script.
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

source scripts/cls_env.sh
case $VARIANT in
  v6a)
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 600 --phase_a_novelty_reward 0 --interleave_empty_fraction 1.0 --randomize_goal_per_rollout --no-novelty_anneal"
    ;;
  v6b)
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 600 --phase_a_novelty_reward 0.3 --interleave_empty_fraction 1.0 --randomize_goal_per_rollout --no-novelty_anneal"
    ;;
  v6c)
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 600 --phase_a_novelty_reward 0.3 --interleave_empty_fraction 1.0 --randomize_goal_per_rollout --no-novelty_anneal --epsilon_explore 0.2 --epsilon_anneal_updates 600"
    ;;
  v7)
    # Pure-explore: no goals (no +1, no teleport), no entropy bonus on movement, novelty=0.3
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 600 --phase_a_novelty_reward 0.3 --interleave_empty_fraction 1.0 --no-goals_active --move_ent_coef 0 --no-novelty_anneal"
    ;;
  baseline)
    # Untrained-policy baseline: phase_a_updates=0, only the post-loop eval fires on a random-init agent.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 0 --phase_a_novelty_reward 0 --interleave_empty_fraction 1.0 --no-novelty_anneal"
    ;;
  v8)
    # Short rollouts (80 steps) + coverage shaping (revisit penalty=0.05). 50 updates only.
    # Otherwise V7: no goals, ent_coef=0, novelty=0.3, pure explore.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 50 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --no-goals_active --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 80 --eval_every 10"
    ;;
  v9)
    # 100-step rollouts + matching 100-step eval (max_steps=cfg.steps_per_rollout). 100 updates.
    # Otherwise V8 settings: revisit penalty=0.05, no goals, ent_coef=0, novelty=0.3.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 100 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --no-goals_active --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10"
    ;;
  v10)
    # V9 + freeze log_std at -1.5 (std=0.22) + epsilon=0.2 annealed over 100 updates.
    # Pinned variance forces PPO loss to pressure the policy mean directly;
    # ε-greedy provides exploration floor independent of learned std.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 100 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --no-goals_active --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100"
    ;;
  v10_long)
    # V10 with 200 updates and eval_every=20 (10 eval points).
    # ε anneal stretched proportionally to 200u.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 200 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --no-goals_active --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 200"
    ;;
  v11)
    # V10 substrate + V2/V3-style interleaved follow/explore (50/50).
    # goals_active=True. Drop input_prev_action + use normalized signal.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 100 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 0.5 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw"
    ;;
  v12)
    # V11 + skewed interleave (75% explore) + remaining-scaled novelty.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 100 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 0.75 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10"
    ;;
  v13)
    # V12 + persistent ε (no anneal).
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 100 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 0.75 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 0 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10"
    ;;
  v14)
    # V13 with lower persistent ε (0.1). V13 cov shot up but importance
    # ratio exploded → NaN. ε=0.1 = half as many tail-action events.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 100 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 0.75 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.1 --epsilon_anneal_updates 0 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10"
    ;;
  v15)
    # V12 with wider frozen std (init_log_std=-0.8 → std=0.45).
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 100 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 0.75 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -0.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10"
    ;;
  v16)
    # Load V10 checkpoint (explore solved, cov 0.76) + interleave
    # curriculum: start 1.0 (pure explore, V10's setting), anneal to
    # 0.5 over 50 updates. ε=0 because V10 was trained with ε=0 by
    # u160; non-zero ε at u1 causes catastrophic PPO ratio explosion.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 100 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.5 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0 --epsilon_anneal_updates 0 --novelty_scale_remaining --novelty_scale_cap 10 --load_checkpoint checkpoint/phase_a_only_giddy-morning-15/phase_a_u160.pt"
    ;;
  v17)
    # V12 base + (1) explore_goals_off (no goal reward in emp regime —
    # cleaner mode separation), (2) envs_per_world=40 (2x more rollouts
    # per update), (3) 500 updates (5x longer training). Tests if V12
    # plateaued because of mode reward overlap and/or training budget.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 0.75 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 25 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 40"
    ;;
  v18)
    # V17 + envs_per_world=80 (2x V17). More rollouts per update →
    # lower variance per PPO update → may reduce u100+ drift while
    # keeping the explore_goals_off win.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 100 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 0.75 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80"
    ;;
  v18d)
    # V18 + per-rollout random distractors in pre_stored Hopfield
    # (N ~ U[0, 10]). Tests if follow-mode generalizes to noisy
    # memories instead of overfitting to "exactly one pattern".
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 100 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 0.75 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10"
    ;;
  v18d2)
    # V18d hurt cov (0.50→0.25). Two changes:
    # (1) Bigger trunk (hidden=256, was 128) — more capacity for the
    #     harder noisy-recall follow task without starving explore.
    # (2) Slow nav-in: interleave_empty 1.0 → 0.75 over first 50u, so
    #     the trunk first establishes pure-explore behavior, then
    #     gradually layers in distractor-noisy follow.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 150 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.75 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --hidden_size 256"
    ;;
  v18d3)
    # V18d2 + emp distractors. V18d2 evals showed cov@d>0 stuck at
    # ~0.30 vs 0.50@d=0 — explore policy never trained on non-empty
    # memory. Add U[0,10] distractors (no goal) to empty-regime
    # rollouts so trunk learns to ignore non-goal recall signals.
    # Risk: blurs the regime cue (Hopfield emptiness was the implicit
    # mode signal). If nav drops, mode confusion is the cost.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 150 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.75 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 256"
    ;;
  v18d4)
    # V18d2 + bumped novelty (0.3 → 0.5). Pushes explore harder so
    # the policy explores regardless of memory contents. Cheaper
    # change than V18d3 (no architecture/distribution shift). Should
    # not degrade nav since pre_stored rollouts have novelty=0.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 150 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.75 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --hidden_size 256"
    ;;
  v18d5)
    # V18d3 + 50% pre_stored (was 25% in V18d3). V18d3 won cov@d>0
    # but tanked nav (0.5). Hypothesis: trunk needs more follow
    # training to re-learn the regime discriminator under noisy
    # memory. Doubling pre_stored share gives the goal-recall
    # direction more PPO updates. Anneal interleave 1.0 → 0.50.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 150 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 256"
    ;;
  v18d6)
    # V18d3 + smaller emp distractor range (0-3 vs 0-10). Less
    # aggressive non-empty memory in explore mode → smaller
    # perturbation to the regime cue. Hopes nav recovers while
    # keeping enough cov@d>0 to clear target.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 150 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.75 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 3 --hidden_size 256"
    ;;
  v18d7)
    # V18d5 settings + 250u (was 150u). V18d5 is showing nav climb
    # of +0.10-0.15 per 10u; if it plateaus before u150, V18d7 gives
    # extra runway. Also more time for mean_steps to drop as nav
    # solidifies.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 250 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 256"
    ;;
  v18d8)
    # Hot-start from V18d2 u70 (nav 0.95-1.00 / mean_steps 5.7-6.7
    # / cov 0.46/0.27/0.27 at d=0/5/10). nav and speed already meet
    # targets; only cov@d>0 is the gap. Continue training with emp
    # distractors to teach explore policy to ignore non-goal recall.
    # Skip the curriculum (interleave_anneal=0); start at the V18d5
    # final ratio (50% pre_stored). Short run: 60u.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 60 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 0.50 --interleave_anneal_updates 0 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 5 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.05 --epsilon_anneal_updates 30 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 256 --load_checkpoint checkpoint/phase_a_only_divine-firefly-31/phase_a_u70.pt"
    ;;
  v18d9)
    # V18d8 + novelty 0.3 → 1.0. Hot-start preserved nav but didn't
    # bump cov; need stronger explore signal to overcome the V18d2
    # trunk's settled-in conservative explore policy. 3.3x novelty.
    # Risk: nav degradation from PPO weighting explore higher.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 60 --phase_a_novelty_reward 1.0 --revisit_penalty 0.1 --interleave_empty_fraction 0.50 --interleave_anneal_updates 0 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 5 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.05 --epsilon_anneal_updates 30 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 256 --load_checkpoint checkpoint/phase_a_only_divine-firefly-31/phase_a_u70.pt"
    ;;
  v18d10)
    # V18d8 + epsilon_explore 0.05 → 0.3. Random actions force
    # coverage independent of policy gradient. Different mechanism
    # than V18d9 (novelty bump). Both probe whether the V18d2 trunk
    # can be pushed off its conservative explore plateau.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 60 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 0.50 --interleave_anneal_updates 0 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 5 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.3 --epsilon_anneal_updates 60 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 256 --load_checkpoint checkpoint/phase_a_only_divine-firefly-31/phase_a_u70.pt"
    ;;
  v18d11)
    # V18d3 from-scratch + goal_reward=10 (was 1). Diagnosis: V18d3
    # mode-collapsed onto pure-explore because follow's +1 goal
    # reward was outweighed by ~15 explore reward (novelty + revisit)
    # per rollout. With +10 goal, follow PPO gradient should
    # dominate when goal is reachable, breaking the collapse. If
    # the trunk learns to discriminate goal-present vs distractor-
    # only memory, this should yield non-collapsed follow + cov.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 150 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 256 --goal_reward 10.0"
    ;;
  v18d12)
    # V18d11 + input_goal_in_memory=True. Bypass: explicit 1-bit cue
    # ("goal is in Hopfield") instead of forcing the trunk to infer
    # regime from prev_reward + recall direction. Should make the
    # discrimination trivial: bit=1 → follow; bit=0 → explore. If
    # this works AND V18d11 doesn't, that confirms regime-inference
    # is the bottleneck. If V18d11 works without it, the bit was
    # unnecessary and bifurcation can be learned from rewards alone.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 150 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 256 --goal_reward 10.0 --input_goal_in_memory"
    ;;
  v18d13)
    # V18d12 stabilized: goal_reward 10 → 5 (smaller value targets,
    # avoid the value_loss runaway that NaN'd V18d11 and degraded
    # V18d12 after u50). Faster epsilon anneal (100 → 50) so noise
    # quiets by the time policy stabilizes. Tighter PPO clip
    # (0.2 → 0.15) caps update size. Goal: extend the V18d12 u50 win
    # past u50 + push cov higher.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 150 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 10 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 50 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 256 --goal_reward 5.0 --input_goal_in_memory --ppo_clip_coef 0.15"
    ;;
  v18d14)
    # V18d12 + bigger trunk (256→512) + longer training (150→300u)
    # + ppo_clip_coef=0.15 for stability over the longer run.
    # V18d12 hit the bifurcation but cov stuck at 0.45 vs V10's 0.76
    # in pure explore — single trunk lost ~30% cov capacity to share
    # with follow. Tests whether 4× params + 2× updates lets each
    # mode reach its full-strength version.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 10.0 --input_goal_in_memory --ppo_clip_coef 0.15"
    ;;
  v18d15)
    # V18d14 NaN'd at ~u60 from value_loss runaway (goal_reward=10).
    # V18d15 fixes that AND rebalances: goal_reward 10→5 (stable per
    # V18d13), slower nav curriculum (50→100u), more explore share
    # (final 65% empty vs 50%), bigger novelty (0.3→0.5) to give
    # explore more gradient. Hidden=512 + 300u kept. Targets the
    # "more capacity" hypothesis while preventing the trunk from
    # over-committing capacity to follow.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --input_goal_in_memory --ppo_clip_coef 0.15"
    ;;
  v18d16)
    # V18d15 NaN'd at u90 from PPO move_loss explosion. Two fixes:
    # (1) ε / override actions now masked out of move_loss (ppo.py
    # change). They were the trigger: log_prob of ε actions under
    # frozen narrow std is huge negative, ratio = exp(new−old) blows
    # up after a few gradient steps of mean drift.
    # (2) Drop --freeze_log_std so log_std can adapt. Previously
    # frozen at -1.5 (σ=0.22); now allowed to drift.
    # Otherwise V18d15 settings.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.5 --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --input_goal_in_memory --ppo_clip_coef 0.15"
    ;;
  v18d17)
    # V18d16 + re-freeze log_std. The ε-action mask removed the
    # instability that originally forced unfreeze. With log_std pinned
    # narrow (σ=0.22), PPO loss directly pressures the policy mean
    # → faster, more direct paths. V18d16 mean_steps stuck at 14-17;
    # V18 baseline with frozen std hit 6.4. Goal: keep V18d16's
    # nav 0.94 / cov 0.60 across distractors but drop mean_steps.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --input_goal_in_memory --ppo_clip_coef 0.15"
    ;;
  v18d18)
    # Pure-explore variant: V10-style (interleave_empty=1.0 always,
    # explore_goals_off) but with emp distractors during training.
    # Inputs: prev_reward + sensory + hopfield_signal (no prev_action,
    # no encoded_state, no hopfield_raw, no goal_in_memory bit).
    # Tests: can the explore-only policy learn cov >0.6 even when
    # the Hopfield contains distractors at eval? V10 hit 0.76 with
    # empty memory; this checks whether emp-distractor training
    # closes the gap to OOD distractor eval.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 200 --phase_a_novelty_reward 0.3 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_anneal_updates 0 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --no-input_encoded_state --input_hopfield_signal --input_prev_reward --input_sensory --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 128 --ppo_clip_coef 0.15"
    ;;
  v18d19)
    # V18d17 + revisit_penalty 0.05 → 0.1. V18d17 looks great on
    # nav (0.86/0.83 at u120) and cov (0.58/0.54) but mean_steps
    # is 12-14 (target <10, V18 baseline 6.4). Higher revisit
    # penalty makes wandering costlier per step → should push the
    # follow policy toward more direct paths. Risk: cov drops
    # because explore-mode agent has less freedom to revisit cells
    # for routing.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.5 --revisit_penalty 0.1 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.5 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --input_goal_in_memory --ppo_clip_coef 0.15"
    ;;
  v18d20)
    # V18d17 final: nav 0.93/0.88/0.89, mean_steps 13.2/13.3/12.3.
    # nav target (>0.9) hit at d=0 only; mean_steps below 14
    # consistently across distractors but still above <10 target.
    # V18d19 (revisit_penalty=0.1) was strictly worse — kills the
    # revisit-bump path. V18d20 = V18d17 + init_log_std -1.5 → -1.8
    # (σ 0.22 → 0.165). Tighter frozen policy → PPO mean pressure
    # is sharper → more direct nav paths. ε-mask makes the
    # narrower std safe (no PPO ratio explosion). Otherwise V18d17
    # settings unchanged.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --input_goal_in_memory --ppo_clip_coef 0.15"
    ;;
  v18d21)
    # V18d20 minus --input_goal_in_memory bit. Tests whether sensory
    # + prev_reward + hopfield_signal + ε-mask + bilateral distractors
    # + goal_reward=5 + tight log_std are enough to learn regime
    # discrimination from natural cues alone — i.e., whether the bit
    # is genuinely necessary or just an accelerator. V18d11 (without
    # the bit, but also without ε-mask + missing fixes) plateaued at
    # nav 0.65; V18d21 retests with all the other infrastructure
    # fixes in place.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d36)
    # V18d21_long recipe + --input_hopfield_raw. Bitless, slow anneal,
    # 500u. Tests whether q_full magnitude (the recall-confidence
    # scalar already implemented) lets the bitless cov-champion close
    # the follow-speed gap to V18d20. If yes, hopfield_raw alone can
    # replace the bit.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d37)
    # V18d30 recipe + --input_hopfield_raw. Bitless, fast-follow
    # recipe (50/50 anneal 50u, persistence_bonus, wall_penalty,
    # sensory on, goal-in-explore on), 500u. Tests whether q_full
    # magnitude lets the fast-follow recipe expand its cov.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d20_size20)
    # V18d20 (with bit) on a 20x20 env (vs default 8x8). Tests
    # whether the bit recipe transfers to a larger world. 6.25x more
    # cells means more steps per rollout (250 vs 100) and more updates
    # (500). Same VectorHash scaffold (Npos already supports >=20).
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --input_goal_in_memory --ppo_clip_coef 0.15"
    ;;
  v18d39)
    # V18d37 recipe (V18d30 base + sensory + goal-in-explore + raw-q)
    # but augmented with multi-step Hopfield recall trajectory inputs.
    # Adds q at iterations 1, 2, 3 (6 extra dims) so policy can read
    # recall convergence dynamics as a confidence signal — clean
    # attractors converge fast, diffuse landscapes wander. 500u.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d40)
    # V18d39 minus --input_hopfield_raw. Drops the iter-1 raw signal
    # that was duplicated via multistep[1]. Policy now sees: normalized
    # unit-vector q at iter 1 (from input_hopfield_signal) + raw q at
    # iters 1, 2, 3 (from multistep). Tests whether removing the
    # duplication helps or hurts.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d41)
    # V18d39 minus sensory. Tests whether multistep recall is enough
    # signal to learn the bifurcation without sensory's wall-detection
    # backstop. Hypothesis: should corner-trap like V18d27/d38.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --no-input_sensory --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d42)
    # V18d39 with unfrozen log_std. Tests whether multistep recall
    # changes how PPO drifts log_std; previous unfreeze (V18d35)
    # collapsed it. Maybe with multistep info the agent can stably
    # widen its policy.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --no-freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v2)
    # 20x20 attempt 2: longer steps_per_rollout (400 vs 250).
    # Tests whether v1's slow learning is due to rollouts being too
    # short to ever reach goals on the larger grid. 500u still.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v3)
    # 20x20 attempt 3: more updates (1000) with same 250 rollouts.
    # Tests whether 500u is just under-trained for the bigger grid.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 750 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v7)
    # 20x20 attempt 7: v3 + higher novelty (0.3 vs 0.1). Single-knob
    # ablation off v3 to isolate novelty contribution.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 750 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v8)
    # 20x20 attempt 8: v3 + slower epsilon anneal (anneal over full
    # training instead of just first 200u). Keeps exploration noise
    # high throughout. Tests if size20 needs more sustained eps.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 750 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 750 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v37)
    # 20x20 attempt 37 — v35 (time_penalty 0.05) + v36 (log_std anneal -1.8→-2.2 over u300-u600) stack.
    # v35 u380 hit fresh-eval champion 0.994/22.9, v35 u440 hit 0.988/21.3 — best new family.
    # Hypothesis: log_std squeeze on top of time_penalty pressure breaks the ms=20 floor.
    # Tighter noise (σ 0.165→0.111) lets the time-penalty-pressured policy commit precisely
    # to short paths late training. Both levers proven; this tests their composition.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --goal_radius 1.0 --time_penalty 0.05 --log_std_anneal_start_update 300 --log_std_anneal_end_update 600 --log_std_anneal_target -2.2"
    ;;
  v18d39_size20_v36)
    # 20x20 attempt 36 — v33 base (goal_radius 1.0) + log_std anneal -1.8 → -2.2 over u300-u600.
    # v33 u540 hit champion 0.982/20.3 fresh-eval. Hypothesis: noise floor σ=exp(-1.8)=0.165
    # blocks tighter approach near goal — late-training squeeze to σ=exp(-2.2)=0.111 lets
    # converged policy exploit precision without disrupting early bootstrap (u1-300 unchanged).
    # Mirrors v29's log_std anneal but in opposite direction (v29 went -1.8→-1.4 looser).
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --goal_radius 1.0 --log_std_anneal_start_update 300 --log_std_anneal_end_update 600 --log_std_anneal_target -2.2"
    ;;
  v18d39_size20_v35)
    # 20x20 attempt 35 — v33 base (goal_radius 1.0) + time_penalty 0.05 (5x default).
    # Hypothesis: v33/v34 hit sr=0.99 but ms plateaued at 26-32 because the
    # wider goal ball removed precision pressure near goal. Stronger time penalty
    # restores the gradient pushing policy to minimize steps. Targets v27's
    # ms=18 floor while keeping v33's sr=0.99.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --goal_radius 1.0 --time_penalty 0.05"
    ;;
  v18d39_size20_v34)
    # 20x20 attempt 34 — v33 + v32 stacked: goal_radius 1.0 + action L2 clamp [0.5, 1.25].
    # Tests whether goal-radius relaxation and action-magnitude clamp stack
    # additively. v33 should reduce near-goal back-and-forth via larger goal
    # ball; [0.5, 1.25] clamp prevents bang-bang overshoot from large mean
    # actions (v24 std drift 0.166 → 0.359).
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --goal_radius 1.0 --max_action_norm 1.25 --min_action_norm 0.5"
    ;;
  v18d39_size20_v33)
    # 20x20 attempt 33 — v24 base + goal_radius 1.0.
    # Default radius is 0.5 (snap-equality on integer cells); 1.0 includes
    # 4-connected neighbors. Hypothesis: enlarging the goal acceptance
    # ball relaxes the near-goal precision the policy needs and should
    # cut ms (less back-and-forth at goal cell) without sacrificing sr.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --goal_radius 1.0"
    ;;
  v18d39_size20_v32)
    # 20x20 attempt 32 — clamp action L2 to [0.5, 1.25].
    # Combines min floor (forces non-trivial step magnitude) with max cap
    # (prevents overshoot). Hypothesis: small-mean policy actions get
    # boosted to 0.5 (so they make progress), large-mean actions clipped
    # to 1.25 (preventing bang-bang oscillation near goal).
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --max_action_norm 1.25 --min_action_norm 0.5"
    ;;
  v18d39_size20_v31)
    # 20x20 attempt 31 — variant of v27/v28: max_action_norm=1.0.
    # v27 used continuous_normalize=True (forces ‖action‖ = 1 always).
    # v28 used max_action_norm=1.5 (allows variable step up to 1.5).
    # v31 uses max_action_norm=1.0 (allows variable step up to 1.0,
    # so small-mean actions stay small unlike v27's hard normalize).
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --max_action_norm 1.0"
    ;;
  v18d39_size20_v28)
    # 20x20 attempt 28 — Exp A: v24 base + soft action-magnitude cap.
    # Targets the late-training near-goal overshoot identified in the deep
    # analysis (action std drift 0.166 → 0.359 over u1→u520 in v24).
    # max_action_norm 1.5 keeps direction but caps step size, should avoid
    # bang-bang oscillation around goal cell. v25 unit-norm proved too tight;
    # 1.5 is a soft middle ground.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --max_action_norm 1.5"
    ;;
  v18d39_size20_v29)
    # 20x20 attempt 29 — Exp B: v24 base + scheduled log_std anneal.
    # Stays frozen at -1.8 until u300, then ramps to -1.4 (σ 0.165 → 0.247)
    # by u500. Targets the d=10 stochastic-rescue gap (srS-srD ~0.03) by
    # giving the policy more usable late-training noise to escape distractor
    # local minima. Keeps freeze_log_std so the new value is set programmatic.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --log_std_anneal_start_update 300 --log_std_anneal_end_update 500 --log_std_anneal_target -1.4"
    ;;
  v18d39_size20_v30)
    # 20x20 attempt 30 — Exp C: v24 base + distractor curriculum 10→20.
    # v26 had this curriculum on v20 base, plateaued at d=10 sr ~0.95.
    # Test whether v24's recipe (h=1024 + ent_coef) lets the curriculum
    # actually push d=10 sr higher.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --n_train_distractors_max_end 20 --n_train_emp_distractors_max_end 20 --distractor_curriculum_updates 1500 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v27)
    # 20x20 attempt 27: v25 recipe (action norm + h=1024) + ent_coef=0.005.
    # Combines v25's bootstrap acceleration with v24's late-training
    # ms-floor lever. Best of both — predicted to dominate all prior.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --continuous_normalize"
    ;;
  v18d39_size20_v26)
    # 20x20 attempt 26: v20 base (raw + h=1024) + distractor curriculum.
    # Ramps n_train_distractors_max from 10 → 20 (and emp from 10 → 20)
    # over the first 1500 updates. Targets the d=10 failure rate
    # (currently 4-5%) which dominates avg cst via the 400-step penalty.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --n_train_distractors_max_end 20 --n_train_emp_distractors_max_end 20 --distractor_curriculum_updates 1500 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v25)
    # 20x20 attempt 25: v20 base (raw + h=1024) + --continuous_normalize.
    # Env unit-normalizes the action vector before applying so step
    # magnitude is fixed at continuous_scale=1.0. Should accelerate
    # bootstrap dramatically (current u20 step ≈ 0.08, would jump to 1.0)
    # and close the 0.65 → 1.0 mean_speed gap.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --continuous_normalize"
    ;;
  v18d39_size20_v24)
    # 20x20 attempt 24: v21 recipe (raw + ent_coef=0.005) + hidden_size 1024.
    # Best of both: ent_coef=0.005 unlocked v21 to top sr (0.987) at h=512;
    # h=1024 gave v20 fast bootstrap. Combined should beat both individually.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v23)
    # 20x20 attempt 23: v20 base (raw + h=1024) + time_penalty 0.05
    # (5× default 0.01). Direct lever for cst — every extra step
    # costs −0.05, vs goal_reward 5.0. Pressures policy toward
    # short trajectories. Risk: too high → agent gives up / wanders
    # less in explore. Start mild and iterate.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15 --time_penalty 0.05"
    ;;
  v18d39_size20_v22)
    # 20x20 attempt 22: v20 base (raw + h=1024) but with goals_active
    # ON in explore rollouts (--no-explore_goals_off). Tests whether
    # giving +goal_reward in empty rollouts helps the agent generalize
    # navigation across both regimes. May need novelty/goal_reward
    # rebalancing if goal-hit reward dominates the explore signal.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --no-explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v21)
    # 20x20 attempt 21: v16 recipe + small move_ent_coef (0.005) as
    # soft entropy floor to replace ε once it anneals. Both v16 and v17
    # showed sr degradation post-ε=0; hypothesis is that some entropy
    # pressure is needed to keep policy from over-committing. frozen
    # log_std prevents the V6-era log_std-blowup failure mode.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0.005 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v20)
    # 20x20 attempt 20: v16 recipe (raw signal) + hidden_size 1024
    # (open question #3 from docs/archive/PHASE_A_SIZE20.md, on the winning recipe).
    # Post-epsilon-anneal v16 hit sr=0.95 d=10 vs v17 norm 0.77 — raw
    # is decisively better for follow. Test if capacity helps v16 too.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v19)
    # 20x20 attempt 19: v17 recipe (norm signal) + hidden_size 1024
    # (open question #3 from docs/archive/PHASE_A_SIZE20.md). v17 is winning on
    # both cov and sr at u140 — does more capacity push it further?
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 1024 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v18)
    # 20x20 attempt 18: v16 with --no-freeze_log_std. Open question #2
    # from docs/archive/PHASE_A_SIZE20.md — does unfreezing log_std help on size 20
    # the way it did on 8x8 V18d42? Single-knob ablation off v16.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --no-freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v17)
    # 20x20 attempt 17: v16 with --no-input_hopfield_raw. Open
    # question #1 from docs/archive/PHASE_A_SIZE20.md — does normalized hopfield
    # signal (unit-vector direction, no magnitude) beat raw projected
    # q when the multistep recall channels already carry magnitude
    # info? Pure single-knob ablation off v16.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v16)
    # 20x20 attempt 16: single long-run from scratch (no checkpoint
    # resume) of the canonical v15-style recipe per docs/archive/PHASE_A_SIZE20.md
    # "Recommended next steps". Phase_a_updates 3000, ppo_clip 0.15
    # (matches v7's clip; v15's 0.05 was too tight for fresh training),
    # explore_goals_off (true bitless explore signal), 400-step rollouts.
    # Job runs up to 7d on pi_evelina9 — supersedes chained-resume.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 3000 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v15)
    # 20x20 attempt 15: resume v13 (gentle-water-90 u20, cov 0.41) +
    # longer rollouts (steps_per_rollout 400 vs 250). More goal-finding
    # opportunity per episode without destabilizing the converged policy.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 200 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.05 --load_checkpoint /orcd/home/002/jackking/cls/checkpoint/phase_a_only_gentle-water-90/phase_a_u20.pt"
    ;;
  v18d39_size20_v14)
    # 20x20 attempt 14: resume v13 (gentle-water-90, plateau cov 0.40)
    # + novelty 0.5 (vs 0.3) to push past the cov ceiling. Higher
    # novelty reward, same other settings.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.5 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.05 --load_checkpoint /orcd/home/002/jackking/cls/checkpoint/phase_a_only_gentle-water-90/phase_a_u20.pt"
    ;;
  v18d39_size20_v13)
    # 20x20 attempt 13: resume v12 (devoted-pyramid-89 u300, cov 0.40
    # sr 0.96) for next +300u. Cumulative u840 → u1140.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.05 --load_checkpoint /orcd/home/002/jackking/cls/checkpoint/phase_a_only_devoted-pyramid-89/phase_a_u300.pt"
    ;;
  v18d39_size20_v12)
    # 20x20 attempt 12: resume v11 (rare-lion-88, best size20 ckpt at
    # cumulative u540, cov 0.35 sr 0.96) + 300 more updates with tight
    # clip. Push toward bigger cov.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.05 --load_checkpoint /orcd/home/002/jackking/cls/checkpoint/phase_a_only_rare-lion-88/phase_a_u260.pt"
    ;;
  v18d39_size20_v11)
    # 20x20 attempt 11: resume v7 (leafy-voice-86 u280, cov 0.27 sr 0.98)
    # with tight clip 0.05 to extend without NaN crash. v7's recipe
    # (nov 0.3) was the best size20 so far.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.05 --load_checkpoint /orcd/home/002/jackking/cls/checkpoint/phase_a_only_leafy-voice-86/phase_a_u280.pt"
    ;;
  v18d39_size20_v10)
    # 20x20 attempt 10: curriculum. Resume v5's u320 ckpt (size=14,
    # cov 0.37) and continue training on size=20. Tests if skills
    # learned on smaller grid transfer.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.05 --load_checkpoint /orcd/home/002/jackking/cls/checkpoint/phase_a_only_fast-pine-82/phase_a_u320.pt"
    ;;
  v18d39_size20_v9)
    # 20x20 attempt 9: resume v3 again (smooth-star-80 u280) but with
    # ppo_clip=0.05 (vs 0.15) to keep update steps smaller — v6 crashed
    # with NaN logits, likely from fresh optimizer momentum on resume.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.05 --load_checkpoint /orcd/home/002/jackking/cls/checkpoint/phase_a_only_smooth-star-80/phase_a_u280.pt"
    ;;
  v18d39_size20_v6)
    # 20x20 attempt 6: resume v3 (smooth-star-80 u280) + 300 more updates.
    # Effectively ~580u on the same agent. Tests if v3 was just under-
    # trained at u280 (cov 0.23, sr 0.89) and would keep climbing.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15 --load_checkpoint /orcd/home/002/jackking/cls/checkpoint/phase_a_only_smooth-star-80/phase_a_u280.pt"
    ;;
  v18d39_size20_v4)
    # 20x20 attempt 4: combine v2 (longer rollouts=400) + v3 (more
    # updates=750) + higher novelty (0.3 vs 0.1) for the bigger grid.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 750 --phase_a_novelty_reward 0.3 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 400 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v5)
    # 20x20 attempt 5: smaller intermediate grid (size=14) as a
    # stepping stone. Tests if difficulty is sharp or smooth in size.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 14 --steps_per_rollout 200 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d39_size20_v1)
    # V18d39 ported to 20x20 grid. First attempt. steps_per_rollout
    # 250 (vs 100 on 8x8) since Manhattan distances scale ~2.5x.
    # Same 500 updates as 8x8 baseline.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --size 20 --steps_per_rollout 250 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --input_hopfield_multistep 1 2 3 --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d38)
    # V18d27 recipe + --input_hopfield_raw. Bitless, no-sensory, fast-
    # follow recipe (50/50 anneal 50u, persistence, wall, no goal-in-
    # explore), 500u. Tests whether q_full magnitude alone (no sensory
    # to dilute the lock) gives both V18d27's sharp follow AND non-
    # degenerate explore. The cleanest possible bit-replacement test.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --input_hopfield_raw --no-input_sensory --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d26_long)
    # V18d26 (V18d21 base + 50/50 follow share + faster anneal,
    # sensory ON) extended to 500u. V18d26 was cov 0.56 / sr 0.88 /
    # cst 37 at u300 — best balanced bitless. V18d21_long showed
    # extending old-shape variants closes the cov gap; V18d26_long
    # tests whether the better-sr V18d26 ALSO climbs with extension.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d20_long)
    # V18d20 (with bit) extended to 500 updates. Apples-to-apples
    # comparison against V18d21_long which hit cov 0.71 at u460
    # (exceeding V18d20's u300 cov of 0.63). Tests whether V18d20
    # also climbs further with extra training.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --input_goal_in_memory --ppo_clip_coef 0.15"
    ;;
  v18d21_long)
    # V18d21 extended to 500 updates: cov was still climbing
    # monotonically at u=300 (.46→.49→.49→.50→.55→.56→.58 over
    # u180→u300). Schedules (epsilon, interleave) finish by u100,
    # so the extra 200 updates train the fully-annealed regime.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d30)
    # V18d29 + goal reward in explore (remove --explore_goals_off).
    # New shape + 50/50 + anneal 50u + sensory on + goal reward in
    # explore. Tests if goal reward helps cov when combined with
    # the V18d27/V18d29 follow-acceleration recipe.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d35)
    # Ablation: V18d27 with --no-freeze_log_std. Tests whether
    # PPO-tuned global log_std helps the corner-trap explore policy
    # without breaking V18d27's sharp Hopfield readout. Single
    # one-knob change off V18d27. Decides if --no-freeze_log_std
    # should be baked into V18d31-d34 wave.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --no-freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --no-input_sensory --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d31)
    # 2x2 ablation off V18d27. V18d31 = V18d27 - persistence_bonus.
    # Tests "is persistence the corner-attractor on its own?" If yes
    # and follow holds, ship. If follow drops, persistence was
    # load-bearing for sharp readout. Sensory off (V18d27 base).
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --no-input_sensory --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d32)
    # V18d31 + revisit_penalty=0.05 (still no-sensory).
    # Adds the "leave the corner" gradient that V18d26 has but V18d27
    # lacked.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0.05 --wall_penalty 0.1 --persistence_bonus 0 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --no-input_sensory --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d33)
    # V18d31 + sensory ON (no persistence, no revisit).
    # Tests "does sensory rescue cov on its own once persistence is
    # gone?" Compared to V18d29 which had sensory but kept persistence
    # — this isolates sensory effect cleanly.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d34)
    # V18d31 + sensory ON + revisit_penalty=0.05.
    # Combined fix; closest to V18d20 ergonomics. Expected best on
    # explore among the four.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0.05 --wall_penalty 0.1 --persistence_bonus 0 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d30_long)
    # V18d30 extended to 500 updates: cov was climbing strongly at
    # u=300 (.25→.28→.34→.35→.38→.40→.42 over u180→u300). Most
    # under-trained variant of the d20-d30 batch. Anneal schedules
    # (epsilon=200u, interleave=50u) finish by u200; the extra 200
    # updates train the fully-annealed regime.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 500 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d29)
    # V18d27 + sensory on (remove --no-input_sensory). V18d27 = new
    # shape + 50/50 + anneal 50u + no sensory. V18d29 keeps the new
    # shape and follow-accel but restores sensory. Tests sensory's
    # contribution under the new shape recipe.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d28)
    # V18d23 (V18d21 base + goal reward in explore) + V18d25's
    # follow-acceleration: 50/50 split + anneal 50u. Sensory stays
    # ON (default). Tests if goal reward in explore + faster follow
    # ramp gives V18d25-style speed without V18d25's cov collapse.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d27)
    # V18d22 (new shape: persistence + wall + low novelty + high ε,
    # bit off) + V18d25's three changes: 50/50 split, anneal 50u,
    # --no-input_sensory. Tests whether V18d22 base + the V18d25
    # follow-acceleration recipe lands somewhere on the cov/speed
    # frontier different from V18d25 alone.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --no-input_sensory --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d26)
    # V18d25 - the --no-input_sensory flag. Keep 50/50 split + faster
    # anneal, but restore sensory input. Hypothesis: no-sensory was
    # what killed V18d25's cov (0.14 vs V18d20's 0.63) by stripping
    # the explore policy of per-cell state info. With sensory back +
    # 50/50 follow share, hoping to keep V18d25's nav/speed gains
    # while recovering V18d20-level cov.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d25)
    # V18d21 base + 50/50 follow/explore split (was 65/35) annealed
    # over 50u (was 100u) + remove sensory input. Tests:
    # 1) more follow training share might compensate for missing bit
    # 2) faster anneal lets follow training start earlier
    # 3) no sensory: forces trunk to use only prev_reward + hopfield_signal
    #    for state representation. Cleaner test of whether the agent
    #    can discriminate regime from those alone.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.50 --interleave_anneal_updates 50 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --no-input_sensory --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d24)
    # V18d22 (no bit, new explore shape: persistence + wall, lower
    # novelty, higher ε) PLUS goal reward in explore mode (remove
    # --explore_goals_off so explore rollouts have goals_active=True,
    # +5 + teleport on goal). Hopfield in explore still has only
    # distractors. Combines V18d22's reward shape with V18d23's goal
    # signal — gives the explore policy a real goal to chase, not
    # just shaping rewards.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d23)
    # V18d21 (no bit) + remove --explore_goals_off so explore-mode
    # rollouts have goals_active=True (agent gets +5 reward + teleport
    # on reaching goal). Hopfield contents unchanged: empty regime
    # still has only distractors, no goal pattern. Tests whether
    # giving the explore policy a real goal signal (without help from
    # memory) bootstraps the trunk into discriminating regimes more
    # cleanly without the bit. V18d21 plateaued at nav 0.44 by u100.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.5 --revisit_penalty 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.2 --epsilon_anneal_updates 100 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  v18d22)
    # V18d21 (no bit) + reshape explore reward: novelty 0.5→0.1
    # (still a nudge, not the dominant signal), revisit_penalty
    # dropped, persistence_bonus 0.05 (stateless straight-line
    # reward replacing revisit), wall_penalty 0.1 (now actually
    # wired after rollout.py refactor), epsilon up 0.2→0.4 with
    # slower anneal (100→200u). Hypothesis: a simpler explore
    # behavior (random walk biased toward straight + away from
    # walls) frees trunk capacity for follow, and might also
    # generalize better to OOD memory at eval.
    EXTRA="--warmup_explore_only_updates 0 --phase_a_updates 300 --phase_a_novelty_reward 0.1 --revisit_penalty 0 --wall_penalty 0.1 --persistence_bonus 0.05 --interleave_empty_fraction 1.0 --interleave_empty_target 0.65 --interleave_anneal_updates 100 --move_ent_coef 0 --no-novelty_anneal --steps_per_rollout 100 --eval_every 20 --init_log_std -1.8 --freeze_log_std --epsilon_explore 0.4 --epsilon_anneal_updates 200 --no-input_prev_action --no-input_hopfield_raw --novelty_scale_remaining --novelty_scale_cap 10 --explore_goals_off --envs_per_world 80 --n_train_distractors_min 0 --n_train_distractors_max 10 --n_train_emp_distractors_min 0 --n_train_emp_distractors_max 10 --hidden_size 512 --goal_reward 5.0 --ppo_clip_coef 0.15"
    ;;
  *)
    echo "Unknown VARIANT: $VARIANT"; exit 1 ;;
esac

echo "Running variant=$VARIANT with: $EXTRA"

python -u -m hopfield_nav.train_phase_a_only \
    --encoder_checkpoint encoders/run_20260422_185816/encoder_best.pt \
    --fwhm_ratio 0.25 \
    --size 8 \
    --observation_size 12 \
    --movement_mode continuous \
    --hopfield_mode continuous \
    --lambdas 11 12 13 \
    --Np 400 \
    --static-vectorhash \
    --no-input_encoded_state \
    --input_hopfield_signal \
    --input_prev_reward \
    --input_prev_action \
    --input_hopfield_raw \
    --input_sensory \
    --init_log_std -0.5 \
    --phase_a_lr 3e-4 \
    --batch_envs 16 \
    --steps_per_rollout 400 \
    --num_worlds 1 \
    --envs_per_world 20 \
    --num_val_envs 10 \
    --n_val_trials 32 \
    --val_distractors 0 5 10 \
    --eval_every 50 \
    --seed $SEED \
    --use_wandb \
    --wandb_project hopfield-nav-phase-a-sweep \
    --device cuda \
    $EXTRA
