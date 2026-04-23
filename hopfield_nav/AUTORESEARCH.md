You will be training models on a navigation task with PPO, as is set up in hopfield_nav/. I need you to iterate on this framework to reach the best model possible.

The models are evaluated along three different types of evaluation. Make sure you understand these. Look at the code, and you can look at TRAINING_AND_EVALUATION.md. Of the different measures, what you need to particularly focuson on are:
	success_rate (want higher) and mean_speed (want higher) in deterministic evaluate navigation
	store_efficiency (want higher), and reach_success_rate (want higher) 
	mean_steps_to_goal (want lower) in evaluate_exploration
But you can potentially use the other measures to inform what may be failing. You can also potentially write new diagnostics if you think that could be useful, but only if you have a really good idea for how it would be useful.

Make sure you understand the code very well, then develop a well motivated plan. There are many steps to process — environment, vector hash, encoder, hopfield. In making the plan and iterating, recruit an understanding of what types of representations and knowledge this specific task will require. Also use an understanding of the mechanics of PPO.

You will iterate on training runs, keeping the additions that work and throwing out those that don’t. Document every run and what you’ve learned from it in EXPERIMENTS.md. If you write a new diagnostic, make sure to include as well.

Every 10 runs, do a comprehensive reevaluation of progress so far to make sure the plan is on the right track. Document this evaluation in EXPERIMENTS.md.

DO NOT STOP EVER. Keep running.

The base script is run_continuous.sh. Iterate off of the params in that one. Keep those SBATCH params, although you can also try  --partition=pi_evelina9. But run no more than two jobs at a time on pi_evelina9.

Iteration will be choosing different params. Here are the ones you can change. Of course only change them if it is in the plan, and therefore there is good reason to do so. Leave everything else as it is in run_continuous.sh.

--store_cost
--store_bonus
--store_bc_weight
--auto_store_warmup
--auto_nav_warmup
--aux_anneal_updates
--num_rnn_layers
--hidden_size
--init_log_std
--ent_coef
--store_ent_coef
--batch_envs
--steps_per_rollout
--n_updates
--lr
--explore_steps
—clip_coef
—ent_coef
—gae_lambda
—gamma
—n_minibatches
—ppo_epochs

