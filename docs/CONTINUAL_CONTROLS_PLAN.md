# A continual-learning control suite for the Hopfield-nav claim

Written 2026-08-30. Plan only — nothing here is implemented yet.

Companion reading: `docs/CODEBASE_MAP.md` §"final_plotting", `analysis/continual/`,
`hopfield_nav/training/rnn_sequential.py`.

---

## 0. What this document is for

`analysis/continual/` currently supports one comparison:

| run | protocol | env0 | env1 | env2 | env3 | env4 (current) |
|---|---|---|---|---|---|---|
| `agenthash_w_oracle` | frozen policy, Hopfield store only | **0.98** | 1.00 | 1.00 | 1.00 | 1.00 |
| `baseline_regular_final` (30 seeds) | pretrain → sequential BC finetune | 0.05 | 0.18 | 0.22 | 0.28 | 0.99 |
| `baseline_regular_200steps` (30 seeds) | same, 200-step rollouts | 0.08 | 0.18 | 0.24 | 0.28 | 1.00 |
| `20x20_pretrained_10_full_iters` (10 seeds) | pretrain → finetune, 100 upd/env | 0.06 | 0.26 | 0.35 | 0.29 | 0.99 |

(`reached` averaged over the last 20 % of the final block; 5 envs × 200 updates,
`size=20`. Extracted from `analysis/continual/histories/`.)

The RNN control reaches **0.99 on the env it is currently training on** and
**~0.19 on everything else**, with a clean recency gradient. Plasticity is
fine; retention is gone. The Hopfield agent is at ~1.0 everywhere.

That is a real result, but as it stands it is one control — a *naive* one — and
a referee's first question writes itself: *"you compared against SGD with no
continual-learning method at all; a 2019 replay buffer would close this."*
This document is the answer to that question: what the strongest possible
classic continual learner looks like on this protocol, and how to build it.

### 0.1 The strategic point, stated up front

**A replay buffer with an unbounded budget will probably close most of this
gap, and the plan is built on the assumption that it will.** The arithmetic:
one trajectory is `200 steps × 60 floats = 48 KB`; the entire 5-env × 200-update
stream is **~192 MB**. Storing *every observation the agent has ever seen* is
free at this scale, and Experience Replay over a perfect buffer is, by
construction, joint multi-task training with a delay. If it retains, that is
not a surprise and it is not a failure — it is the expected result.

So the deliverable is **not** "no classic method retains." It is a
**cost-matched frontier**. Every method gets scored on five axes at once:

| axis | what it means | Hopfield agent's value |
|---|---|---|
| **retention** | mean `reached` on envs `0..i-1` at end of block `i` | ~1.0 |
| **gradient steps at deployment** | weight updates needed to acquire a new env | **0** |
| **episodes to acquire** | rollouts needed before the new env is solved | **1** (one store) |
| **stored bytes** | replay data + per-task params + importance matrices | fixed `O(D²)` Hopfield matrix, no data |
| **task-ID requirement** | does the method need to be told which env it is in | **none** |

A method that matches retention but needs 200 gradient steps, 200 episodes and
a growing archive of raw trajectories has not refuted the claim — it has
*located* it. The suite exists to draw that frontier honestly and to make sure
nothing sitting at the Hopfield agent's corner of it has been missed.

### 0.2 The one result that would genuinely hurt

If a **pretrained recurrent policy adapts to a new env in-context, with no
weight updates at all** (§5.4, RL²-style), the claim that a separate
associative memory is *needed* weakens sharply. That control is in the plan
deliberately, in Wave 4, and it should be run rather than avoided.

---

## 1. What the setting actually is, in continual-learning terms

This matters because it determines which algorithms are even applicable and
which literature results transfer.

**The task.** `GridEnv` (`hopfield_nav/world/env.py:278`) is an empty
`size × size` arena. Its four boundary walls carry a **random ±1 barcode**,
`(4, size)`, drawn from the env's seed. The observation is a foveal ray-cast:
`observation_size=60` rays fanned around the heading, each returning the ±1 code
of the wall segment it hits. The goal is a random cell and **is never
observed** — only the BFS oracle sees it.

So env `i` differs from env `j` in two ways:

1. **a different random re-encoding of position** (different barcode), and
2. **a different target function** (different goal).

Item (1) makes this structurally a **permuted-MNIST-style domain-incremental**
problem: the input space is randomly relabelled per task while the output space
(4 move actions, or a 2-D unit displacement) is shared. Item (2) makes it
strictly harder than permuted MNIST, where the label function is constant.

**Scenario classification** (van de Ven & Tolias taxonomy):

- **Domain-incremental.** Shared output space, shifting input distribution, task
  identity not provided at test.
- **Task identity is inferable in principle.** 60 rays against an 80-bit
  barcode is highly discriminative; two random envs agree on ~half their
  segments. An agent *could* recognise which maze it is in from a single
  observation. It could not recover the *goal* that way — the goal is pure
  episodic content.
- **Task boundaries are known at training time** (the block structure in
  `run_sequential_blocks`). We will hand this to the baselines for free; it is a
  generosity that only strengthens the control.
- **The learner is recurrent** — and this is the single most consequential fact
  for method selection (§4.2).

**Why this classification is good news for the controls.** In domain-IL
permuted MNIST, EWC is a *serious* method, not a strawman: 94.3 % vs 78.5 % for
naive fine-tuning against a 97.6 % joint ceiling (van de Ven & Tolias 2019,
Table 5). This is not the class-incremental regime where EWC collapses to
chance. Expect EWC to do real work here.

**Why it is bad news.** Ehret et al. (ICLR 2021), the definitive study of CL in
*recurrent* networks, found that weight-importance methods (EWC/SI/MAS) degrade
specifically as **working-memory demand** rises — not with sequence length, but
with how much sample-specific state the recurrent hidden vector must carry.
Their explanation: Fisher importance values spike with memory load, so the
network is forced into stability and loses plasticity. A navigation policy that
must integrate a trajectory to localise itself is exactly a high-working-memory
task. Their recommendation — **task-conditioned hypernetworks** — is in Wave 3
and is the most important single competitor in this document (§4.6).

---

## 2. Tier 0 — the measurements that must exist before any method is judged

None of these are continual-learning methods. They are the axes of the plot.
**Nothing in Tier 1–4 is interpretable without them**, and three of the four do
not currently exist as recorded runs.

| # | run | what it bounds | status |
|---|---|---|---|
| **T0.1** | **Joint / multi-task ceiling** — train one policy on all N envs simultaneously (pool one rollout per env each update, `train_rnn --mode mixed` on the *evaluation* envs) to convergence, then evaluate all N | The **ceiling for every weight-based method**. No CL algorithm can beat joint training on its own tasks. | ✗ missing |
| **T0.2** | **Per-env expert** — N independent policies, one per env, each trained alone | Per-task ceiling; the gap `T0.1 − T0.2` measures how much **capacity interference** there is at N envs | ✗ missing |
| **T0.3** | **BFS oracle** — `teacher_force=True` in `collect_rollout_rnn` | The task's own ceiling under the eval step cap; catches "the eval is impossible" | ✗ not recorded |
| **T0.4** | **Sequential from scratch** — no pretraining, naive SGD | The **floor**. Every recorded history has `mode=finetune`; the from-scratch control the paper describes appears never to have been run at final settings | ✗ missing |

**T0.1 is the most important run in this entire document.** If the joint ceiling
at N=5, hidden=128 is (say) 0.6, then no continual method can exceed 0.6, the
"CL methods fail" result is really a *capacity* result, and the honest fix is a
bigger network — which a referee will ask for. If the joint ceiling is ~1.0,
then every point of the retention gap is genuinely attributable to forgetting
and the comparison is clean. **Run T0.1 first, before writing any method code.**

T0.1 must be swept over capacity — `hidden_size ∈ {128, 256, 512, 1024}`,
`num_rnn_layers ∈ {1, 2}` — and over N. The result is a curve: *joint ceiling
vs. number of envs, at each capacity*. That curve is a figure in its own right,
and it is the fair way to state what the Hopfield store buys: the Hopfield
agent's ceiling does not bend down as N grows because nothing about its policy
is per-env.

---

## 3. Tier 1 — making the two existing controls as strong as they can be

The request was that the two basic controls be "the absolute best they can be."
Here is the concrete audit. `analysis/continual/run_baseline.sh` currently
overrides the library defaults *downward* on almost every training knob.

### 3.1 Confirmed weaknesses in the current naive control

| # | current | problem | fix |
|---|---|---|---|
| **W1** | `BATCH_ENVS=1`, `EPOCHS=1`, `N_MINIBATCHES=1` | **One gradient step per update, on one 200-step trajectory.** The library defaults are `batch_envs=16, epochs=4, n_minibatches=4` = 16 steps/update. The shell script turned all of them off. The control is being run at 1/16 of its intended gradient budget on a batch of one highly autocorrelated trajectory. | Restore `batch_envs ∈ {16, 32}`. Per `project_hopfield_nav_explore_min`, `batch_envs` is within-env and vectorised, so wall-clock cost is near-zero. Sweep `epochs × n_minibatches` under a matched-compute constraint. **Highest expected value of anything in Tier 1.** |
| **W2** | `optimizer = Adam(...)` constructed **once per seed**, outside the block loop (`baseline.py:337`) | Adam's second moments carry across env boundaries. Stale `v` from env `i-1` makes the first steps in env `i` maximally destructive — this is the mechanism behind the *stability gap* (De Lange et al. 2023). | Sweep: (a) reset optimizer state at each block boundary, (b) SGD+momentum, (c) lower `β₂`, (d) LR warm-up over the first K updates of each block. Free, and (d) alone often removes most of the transient drop. |
| **W3** | `LR=1e-3` constant for all 1000 updates | LR is the single strongest forgetting/plasticity knob and is untuned. | Sweep `lr ∈ {1e-4 … 3e-3}` per decade, plus within-block cosine decay so late-block steps are small. Report the retention/plasticity trade-off curve — it is itself a result. |
| **W4** | `HIDDEN_SIZE=128`, 1 layer, GRU | Untested capacity. At N=5 the net must memorise 5 barcode→position decoders *and* 5 goals. It may simply be too small, which would mean the "forgetting" figure is partly a capacity figure. | Tied to T0.1. Sweep `hidden ∈ {128,256,512,1024}`, `layers ∈ {1,2}`, GRU/LSTM. **Report the control at its best capacity, not at 128.** |
| **W5** | `MOVEMENT_MODE=continuous` with `init_log_std` **not exposed on `baseline.py`'s CLI** → stuck at the `RNNAgentConfig` default of `0.0`, i.e. σ = 1.0 on a unit-magnitude action, and learnable | The DAgger student explores with noise the size of the action itself. State coverage during collection is being wrecked by an unswept default that cannot even be set from the run script. | Expose `--init_log_std` / `--freeze_log_std` on `analysis/continual/baseline.py` and sweep. **Also run the whole suite in `discrete` mode** — see §3.3. |
| **W6** | no weight decay, no dropout (`num_rnn_layers=1` makes `dropout` a no-op anyway) | Standard regularisation untested. | Small sweep; low expected value but cheap and forecloses the objection. |
| **W7** | `ONLY_TRAIN_ON_REACHED` | Measured, and it **hurts**: current-env `reached` drops to 0.09–0.25 in every `only_reached` history. | Keep off. Report as a recorded ablation, not a candidate. |
| **W8** | `INPUT_PREV_ACTION=0`, `INPUT_PREV_REWARD=0` | A recurrent policy without a previous-action channel is a weaker policy than the standard recipe. See open question Q3 — `feedback_hopfield_nav_bc_inputs` freezes the input set for the *bc-AQ* line and I do not know whether that binds here. | **Ask before changing.** |

### 3.2 The pretraining control, strengthened

Currently: `train_rnn --mode mixed` on a pool of envs → `final.pt` →
`baseline.py --load_checkpoint`. Improvements, in increasing order of ambition:

- **P1 — more, and more diverse, pretraining envs.** The structure worth
  learning ("ray-cast pattern → where I am → which way is the goal-ish
  direction") is only extractable from many envs. Sweep pretraining pool size
  over decades (10, 100, 1000 envs) and pretraining length.
- **P2 — measure the transfer, don't assume it.** Report **forward transfer**
  explicitly: `FT_i = (AUC_i − AUC_i^scratch) / (1 − AUC_i^scratch)`
  (CRL-survey definition). Right now we have no evidence pretraining helps at
  all — `20x20_pretrained_10_full_iters` retains ~0.06 on env 0, which is *worse*
  than the from-scratch-unknown case is likely to be. Pretraining may currently
  be doing nothing, and we would not know.
- **P3 — optimizer-state handling.** `baseline.py:285` deliberately drops Adam
  moments ("fresh Adam moments"). Sweep both ways.
- **P4 — freeze depth.** Freeze the recurrent trunk and adapt only the head; the
  opposite; and adapt-everything. A frozen trunk + per-env head is a much
  stronger and much more common transfer recipe than full finetuning, and it
  bridges naturally to the parameter-isolation methods in §4.5.
- **P5 — the limit case: meta-pretraining.** §5.3. Pretraining "to learn
  something about the structure of the task" has a principled maximum, and it
  is OML/ANML: pretrain *so that subsequent online SGD does not interfere*.
  This is the strongest possible version of control #2 and it belongs in the
  paper.

### 3.3 One structural recommendation: run the suite in `discrete` mode

`run_baseline.sh` sets `MOVEMENT_MODE=continuous`. Nearly every method in §4
is defined against a **softmax over actions**: DER++ replays logits, LwF
distils a categorical, EWC-DR's Logits Reversal is literally a softmax
manipulation, CLEAR's cloning term is a KL between categorical policies. Under
a `Normal(μ, σ)` head, each of these needs a bespoke translation, and each
translation is a place for a referee to say the method was not faithfully
implemented.

**Recommendation:** run the whole control suite in `discrete` mode, where every
method has its canonical form, and keep `continuous` as a robustness check on
the two or three methods that survive. This costs one extra pass over the
existing baselines to establish the discrete-mode reference points.

---

## 4. Tier 2 — the continual-learning algorithm suite

Organised by family. For each: why it is here, what to implement against this
codebase, and what to expect. Grouped into waves at §6.

### 4.1 Family A — Replay (expected strongest; implement first)

The literature is blunt about this. van de Ven & Tolias: *"the only strategy
among the top performers in all three incremental learning scenarios is
replay."* Recent surveys: replay beats regularisation *"consistently."* And at
this scale the buffer is free (§0.1).

| method | ref | what to implement | notes |
|---|---|---|---|
| **ER** (reservoir) | Chaudhry et al. 2019 | Keep an `RNNRolloutBatch` buffer. Each update: train on the new rollout **plus** a sampled batch of stored trajectories. Reservoir sampling for eviction. | **The workhorse.** Sweep buffer size across decades *including unbounded*, and replay ratio. Unbounded-buffer ER ≈ joint training and should approach T0.1. Report bytes. |
| **ER-balanced** | — | Same, but sample uniformly *per env* rather than uniformly over the stream. | Removes the recency bias in the buffer; usually worth 5–10 points over plain reservoir. |
| **CLEAR** | Rolnick et al. 2019 | ER at a **50:50 new:replay ratio**, plus behaviour-cloning to the *past self* on replayed states: KL(π_old ‖ π_now), coefficient 0.01. (The value-cloning term has no counterpart — there is no value head on `RNNAgent`.) | **The continual-RL reference method.** Requires no task boundaries or identity. Rolnick report it beats P&C and substantially beats EWC. In our setting the "past self" targets can be logged at collection time. |
| **DER++** | Buzzega et al. 2020 | Store the **move logits** alongside the teacher action at collection time; replay loss = CE(teacher) + α·MSE(stored logits, current logits). | Strong, simple, boundary-free, and the standard modern rehearsal baseline. Discrete mode makes this natural (§3.3). |
| **A-GEM** | Chaudhry et al. 2019 | Project the new-task gradient so its inner product with the buffer gradient is non-negative. | Weaker than ER in practice but a required citation and cheap. |
| **MIR** | Aljundi et al. 2019 | Retrieve the buffer samples whose loss would rise most under the pending update. | Nice-to-have; the best-known "smart retrieval" variant. |
| **GDumb** | Prabhu et al. 2020 | Ignore the stream; at eval time train from scratch on the buffer alone. | The **embarrassing control**. If GDumb matches ER, then the online stream contributes nothing beyond its buffer, which is a sharp and quotable finding. |

**Replay-specific design decisions to fix and document:**
- *What is one buffer item* — a whole `(T, D)` trajectory (preserves the
  recurrent state's context; the right choice for an RNN) or a single
  timestep (breaks BPTT context). Use trajectories.
- *Hidden-state handling on replay* — replayed trajectories must be re-run
  from `h=0`, matching how `bc_rnn_update` already treats each `(B, T, D)`
  trajectory. This is consistent and needs no extra state storage.
- *Budget accounting* — a replay update sees more data per step than a naive
  update. **Every comparison must be matched on gradient steps AND on
  environment interactions**, and where they cannot both be matched, report both.

### 4.2 Family B — Weight-importance regularisation (the EWC family)

Do this properly; it is what was explicitly asked for, and it is the family
most likely to be dismissed as a strawman if done casually.

| method | ref | notes |
|---|---|---|
| **EWC** | Kirkpatrick et al. 2017 | Diagonal Fisher penalty `λ Σ F_k (θ_k − θ*_k)²`, one anchor per completed block. **Compute the Fisher correctly**: the *true* Fisher `E_s E_{a∼π}[(∇ log π(a\|s))²]` sampling actions from the model, not the "squared gradient of the BC loss at the teacher action" (which is the empirical Fisher and is a different, worse estimator). Document which one is used. |
| **Online EWC** | Schwarz et al. 2018 | Single running Fisher with decay γ. Cheaper, usually ≥ vanilla EWC, and it is the version used in the continual-RL benchmarks (CORA). |
| **SI** | Zenke et al. 2017 | Path-integral importance accumulated *online* during training — needs no separate Fisher pass and no clean task boundary. Beat EWC on domain-IL split MNIST (65.4 vs 64.0) and permuted MNIST (95.3 vs 94.3). |
| **MAS** | Aljundi et al. 2018 | Importance = sensitivity of the output's L2 norm; unsupervised, so it can be estimated from unlabelled rollouts. |
| **EWC-DR** (Logits Reversal) | Liu et al., **CVPR 2026** | The current best-known fix to EWC. Diagnosis: once the model is confident, `(p_c − 1) → 0`, the Fisher vanishes and importance is systematically underestimated for exactly the weights that matter. Fix: compute importance against **reversed logits** `z̃ = −z`, giving `Ω = E[(y_k − p̃_k)²(∂z̃_k/∂w_k)²]`, `p̃ = softmax(−z)`. Reported to consistently beat EWC and variants. **~20 lines on top of an EWC implementation, and it is what makes "we ran the best version of EWC" true.** |
| **L2-init / regenerative reg.** | Kumar et al. 2023 | Penalty toward the *initialisation* rather than the previous task. Cheap, and it preserves plasticity rather than trading it away. |
| **MESU** | Bruno et al., **Nature Comms 2025** | Bayesian metaplasticity: scale each parameter's learning rate by its posterior uncertainty. **Needs no task boundaries.** Beats classical consolidation on 200 sequential permuted-MNIST tasks and gives OOD detection for free. The strongest *modern* member of this family and a good fit for the "boundary-free" column of the results table. |

**Calibration expectation.** Per van de Ven & Tolias's domain-IL permuted-MNIST
numbers, EWC-family methods should recover a large fraction of the joint
ceiling *if* the setting behaves like input-permutation. Per Ehret et al., the
recurrent working-memory demand should claw a lot of that back. Where this
lands between the two is an empirical result worth having, and it is the most
interesting single number in the regularisation family.

**λ must be swept over decades** (1e-1 … 1e5). Under-tuned λ is the standard
way EWC gets accidentally strawmanned, and the resulting stability/plasticity
curve (retention vs. current-env performance as λ rises) is a figure.

### 4.3 Family C — Functional / distillation regularisation

| method | ref | notes |
|---|---|---|
| **LwF** | Li & Hoiem 2016 | Distil the *pre-block* model's action distribution on the **current** env's rollouts. **No buffer at all.** Repeatedly reported as a strong domain-IL baseline despite its simplicity — and it is the cleanest way to show what pure functional regularisation buys with zero storage. |
| **P&C** | Schwarz et al. 2018 | Active column + knowledge base, compressed by distillation, protected by online EWC. The other continual-RL reference method alongside CLEAR. Heavier to implement; include only if CLEAR and online-EWC both land in the interesting region. |

### 4.4 Family D — Gradient projection

| method | ref | notes |
|---|---|---|
| **GPM** | Saha et al. 2021 | After each block, SVD the layer activations to get the block's core representation subspace; constrain later gradients to its orthogonal complement. Stores **bases, not data** — a genuinely different point on the memory axis, which is exactly what the frontier in §0.1 needs. For a GRU, apply to `weight_ih` and `weight_hh`. |
| **OGD** | Farajtabar et al. 2020 | GPM's predecessor; store gradient directions instead of representation bases. Mention, do not implement — GPM dominates it at orders-of-magnitude less memory. |

### 4.5 Family E — Parameter isolation (requires a task signal)

These are the methods that *win* continual-RL benchmarks, and they all need to
know which task they are in. On Continual World, **PackNet reaches 0.80 with
~0.00 forgetting against 0.05 for fine-tuning** — it is not close. Excluding
them because they need a task ID would be exactly the kind of convenient
omission this suite exists to prevent.

Run them in two conditions:
- **(a) oracle task ID** — told the env index at train and test. This is an
  explicit upper bound on the whole family; label it as such in every figure.
- **(b) inferred task ID** — a small learned env classifier over the
  observation stream (recall: the barcode makes this feasible), then route.
  This is the *fair* condition, and the gap `(a) − (b)` measures how much of
  the problem is task inference rather than forgetting — which is precisely
  the job the Hopfield store does in one shot.

| method | ref | notes |
|---|---|---|
| **Multi-head** | — | Shared trunk, per-env output head. The cheapest member and the natural partner to §3.2's P4. |
| **PackNet** | Mallya & Lazebnik 2018 | Iterative prune + per-task binary masks; zero forgetting by construction, capacity runs out at some N. **Best method on Continual World.** |
| **HAT** | Serra et al. 2018 | Learned hard attention masks per task. |
| **XdG** | Masse, Grant & Freedman, **PNAS 2018** | Random sparse gating: for each task, zero a fixed fraction of hidden units. Trivial to implement, neuro-flavoured, and **composes with EWC/SI** — XdG+EWC learned 100 sequential permuted-MNIST tasks at 52.4 % where either alone is near-chance. The closest classical analogue to sparse content-addressing, which makes it the most *conceptually* interesting comparison for this paper. |
| **Progressive Nets** | Rusu et al. 2016 | A new column per task with lateral connections. Zero forgetting by construction, parameters grow linearly in N. Include as **the honest "you can always win by growing" bound**, and report the parameter count next to it. |

### 4.6 Family F — Task-conditioned hypernetworks — *the most important competitor*

**HNET** (von Oswald et al., ICLR 2020): do not store the policy's weights;
store a small per-task **embedding** and generate the weights from it with a
hypernetwork, regularised so that old embeddings still produce old weights.

Two reasons this is the headline competitor:

1. **It is the empirical winner for recurrent networks.** Ehret et al. (ICLR
   2021) benchmarked online EWC, SI, masking, masking+SI, generative replay,
   coresets, multitask and from-scratch across four RNN benchmarks, and found
   HNET *"consistently outperformed weight-importance methods,"* especially as
   task complexity rose. Their stated recommendation is to prefer it. We have a
   recurrent policy; this is the method the literature points at.
2. **It is the closest classical thing to what the Hopfield agent does.**
   Both keep a small, addressable per-task code and recover task-specific
   behaviour from it, instead of overwriting shared weights. The scientific
   difference is *how the code gets written*: HNET learns each embedding by
   gradient descent over a whole block; the Hopfield store writes one in a
   **single Hebbian outer-product update from one episode**. Framing the
   comparison this way turns "our thing beats a baseline" into "our thing
   occupies the same functional niche as the best classical method at
   0 gradient steps instead of 200," which is a far better claim.

Caveat from Ehret et al.: hypernetworks *"introduce additional optimization
challenges, especially in conjunction with vanilla RNNs."* Budget tuning time.

At test the task embedding must come from somewhere — same oracle/inferred
split as §4.5.

### 4.7 Family G — Plasticity maintenance (needed only if N grows)

At N=5 the control is not plasticity-limited (`reached ≈ 0.99` on the current
env). If the stream is extended to N=20–50 (open question Q1), plasticity loss
becomes a real second failure mode and these become necessary:

- **Continual backprop** (Dohare, Hernandez-Garcia, Lan, Rahman & Sutton,
  **Nature 2024**) — continually reinitialise a small fraction of least-used
  units. Their headline finding: gradient descent alone provably loses
  plasticity, and only algorithms injecting fresh diversity maintain it
  indefinitely.
- **Shrink & Perturb**, **L2-init**, **ReDo**, **layer-norm / CReLU** — cheap
  variants of the same idea; run as a small sweep, not as separate headline
  methods.

---

## 5. Tier 3 — the bounds that make the claim precise

### 5.1 Perfect-memory ER
Unbounded buffer, balanced sampling, replay ratio 1:1. The empirical
realisation of T0.1's ceiling *under the streaming protocol*. If this does not
reach T0.1, something is wrong with the optimisation, not with continual
learning — a useful internal consistency check.

### 5.2 Oracle-task-ID isolation
Best of {PackNet, HAT, XdG+EWC, multi-head} with the env index handed over.
Zero forgetting is achievable here almost by definition; the interesting number
is what it costs in parameters and in per-task gradient steps.

### 5.3 Meta-pretraining — the strongest form of control #2
**OML** (Javed & White, NeurIPS 2019) and **ANML** (Beaulieu et al., ECAI 2020):
meta-learn a representation (ANML: a neuromodulatory gating network) *such that
subsequent online SGD does not interfere*. ANML sequentially learned 600 classes
over ~9 000 SGD updates without catastrophic forgetting.

This is the principled maximum of "do pretraining first to learn something
about the structure of the task": rather than hoping features transfer, meta-
learn features whose *update dynamics* are non-interfering. Meta-train across a
large pool of `GridEnv`s with the inner loop being exactly the sequential BC
protocol, then meta-test on the held-out N-env stream. It is the most expensive
item in this document and the one most likely to genuinely compete.

### 5.4 In-context adaptation with **zero** weight updates — the dangerous one
Pretrain the recurrent policy across many envs **with the hidden state carried
across episodes within an env** (RL²/in-context). At test, freeze the weights
entirely and let the agent adapt to a new env purely through recurrent
activity — matching the Hopfield agent's "no gradient steps at deployment"
condition exactly.

Forgetting is then **impossible by construction** for both models, and the
comparison becomes the sharp one: *can an RNN's activation memory do what an
explicit associative store does?* If yes, the paper's framing has to change. If
no — and the specific failure mode is legible (capacity? interference?
horizon?) — that is the strongest positive result available here, because it is
the only control that competes on the Hopfield agent's own terms.

**Run this.** It is the control a good referee will ask for.

---

## 6. Sequencing

Each wave gates the next. Do not start a wave before its predecessor's runs are
in `histories/`.

**Wave 0 — the axes (blocking).**
T0.1 joint ceiling incl. capacity sweep · T0.2 per-env experts · T0.3 oracle ·
T0.4 from-scratch sequential. Plus the discrete-mode reference points (§3.3).
*Nothing else is interpretable until these land.*

**Wave 1 — strongest naive + strongest pretrained + the two obvious methods.**
Tier-1 fixes W1–W6 (W1 and W2 first — cheapest, largest expected effect) ·
pretraining P1–P4 · **ER** (buffer-size sweep incl. unbounded) · **EWC +
online EWC** (λ swept over decades, Fisher done correctly).
*Exit criterion: we know the ceiling, the floor, and where a good buffer and a
good regulariser sit between them.*

**Wave 2 — the modern competition.**
CLEAR · DER++ · SI · MAS · **EWC-DR** · LwF · A-GEM · GDumb.
*Exit criterion: the best classic method on each of the five cost axes in
§0.1 is identified.*

**Wave 3 — the structural methods.**
**HNET** (both oracle and inferred embeddings — the headline competitor) ·
GPM · PackNet/XdG/multi-head under both task-ID conditions · Progressive Nets
as the parameter-growth bound · MESU as the boundary-free regulariser.

**Wave 4 — the bounds that change the framing.**
§5.3 meta-pretraining (OML/ANML) · §5.4 in-context zero-update control ·
Family G if N is extended.

---

## 7. Evaluation protocol

### 7.1 Metrics
Report all of these for every method; the existing `history` JSON schema
already carries enough to compute them offline from the per-update trace.

| metric | definition |
|---|---|
| **Average performance** `A_N` | mean `reached` over all N envs at the end of the stream |
| **Forgetting** `FG` | `mean_i max(p_{i,i} − p_{N,i}, 0)` — peak-minus-final per env |
| **Backward transfer** `BWT` | `1/(N−1) Σ (p_{N,i} − p_{i,i})` |
| **Forward transfer** `FT_i` | `(AUC_i − AUC_i^scratch)/(1 − AUC_i^scratch)` — the only metric that scores pretraining, and the one currently missing |
| **Stability gap** | the *transient* drop on env `i-1` in the first updates of block `i`. The per-update trace already records this at full resolution; nobody has plotted it. De Lange et al. 2023 show it survives in methods whose *final* forgetting looks fine. |
| **Episodes-to-criterion** | rollouts in env `i` before `reached ≥ 0.9`. **The axis on which the Hopfield agent's advantage is largest** (1 vs ~200) and which the current figures do not show at all. |
| **Stored bytes** | replay data + per-task params + Fisher/importance matrices + masks |
| **Parameters** | flat for most, linear in N for ProgNets |

### 7.2 Matched budgets
Every method is reported at **matched gradient steps** and **matched
environment interactions**. Where a method inherently uses more of one (replay:
more data per step), report both and say which is matched. A retention number
obtained at 4× the compute is not a comparison.

### 7.3 Hyperparameter protocol
Wang et al. (2024), *Hyperparameters in Continual Learning: A Reality Check* —
across 8 000+ experiments, *"most state-of-the-art algorithms fail to replicate
their reported performance"* once hyperparameters are tuned on one dataset and
evaluated on another, because the conventional protocol tunes on the very
stream it evaluates and thereby violates continual learning's own premise.

**Adopt their two-phase protocol.** Tune every method's hyperparameters on a
**tuning stream** of envs drawn from the same generator, then evaluate,
frozen, on a disjoint **evaluation stream**. `rnn_world` already supports
declared domains and held-out envs (`--n_val_envs`), so this is cheap to do
correctly. Doing it makes the eventual claim survive review; not doing it
inflates every baseline *and* our own model, unevenly.

Report the per-method HP search grid in an appendix. An under-tuned EWC is not
a result.

### 7.4 Seeds
The existing runs use 30 seeds via `NUM_FULL_ITERS`, and single-seed histories
swing violently (`baseline_regular200steps_final_iter*` ranges from
`{0,0,0,0.098,1}` to `{0.049,0.976,0.805,0.317,1}` on the same config). Keep
≥ 20 seeds and report SEM. Per `feedback_eval_point_threshold`, do not make a
directional claim from a handful of eval points.

---

## 8. Implementation shape

The protocol loop already exists and is already parameterised correctly:
`hopfield_nav/training/rnn_sequential.py:run_sequential_blocks` takes
`on_update` / `on_block_start` callbacks and drives
`collect_rollout_rnn → bc_rnn_update → evaluate_nav_all`. Almost every method
in §4 is a modification of exactly two things: **what loss the update adds**,
and **what happens at a block boundary**.

Proposed minimal surface — a `ContinualMethod` protocol:

```python
class ContinualMethod(Protocol):
    def on_block_start(self, block: int, agent, envs) -> None: ...
    def extra_batches(self, rollout) -> list[RNNRolloutBatch]: ...   # replay
    def penalty(self, agent) -> torch.Tensor: ...                    # EWC/SI/MAS/LwF
    def after_step(self, agent) -> None: ...                         # SI path integral, masks
    def on_block_end(self, block: int, agent, envs) -> None: ...     # Fisher, GPM bases, pruning
    def state_bytes(self) -> int: ...                                # for the memory axis
```

- Replay methods implement `extra_batches` (+ `on_block_end` for eviction).
- Regularisers implement `penalty` (+ `on_block_end` to snapshot θ* and importances).
- SI additionally implements `after_step`.
- Isolation methods implement `on_block_start`/`on_block_end` (masks, pruning).
- GPM implements `on_block_end` (SVD the bases) and hooks the gradient.
- HNET is the one method that does not fit — it replaces the agent, not the
  update. Give it its own driver that reuses `run_sequential_blocks`.

`bc_rnn_update` needs one change: accept an optional list of extra batches and
an optional penalty term. Everything else is additive.

Suggested layout, respecting the layering rules in
`hopfield_nav/tests/test_layering.py` (`analysis/` may import `hopfield_nav`,
never the reverse):

```
hopfield_nav/continual/           # the methods — importable by training code
    base.py                       # ContinualMethod protocol + a no-op
    replay.py                     # ER, ER-balanced, CLEAR, DER++, A-GEM, MIR, GDumb
    regularize.py                 # EWC, online EWC, EWC-DR, SI, MAS, L2-init, MESU
    distill.py                    # LwF, P&C
    project.py                    # GPM
    isolate.py                    # multi-head, XdG, PackNet, HAT
    hypernet.py                   # HNET (its own agent + driver)
analysis/continual/
    baseline.py                   # gains --method / --method_args
    metrics.py                    # NEW: A_N, FG, BWT, FT, stability gap, bytes
    run_suite.sh                  # sweeps the suite over seeds and methods
```

`analysis/continual/baseline.py` grows `--method` and `--method_args` and
records both in `history["metadata"]`, so every existing plotting and merging
path keeps working unchanged and old histories stay readable.

---

## 9. Risks

| risk | mitigation |
|---|---|
| **The joint ceiling is low** — the RNN cannot represent 5 envs at once at any tested capacity | Then this is a capacity result, not a forgetting result, and it must be *reported as such*. T0.1's capacity sweep is what distinguishes them. This is why Wave 0 blocks everything. |
| **Unbounded ER matches the Hopfield agent on retention** | Expected (§0.1). The claim moves to the cost frontier: 0 gradient steps, 1 episode, no stored data. Build the figures for the frontier from the start rather than retrofitting them after the result comes in. |
| **In-context RL² also retains** (§5.4) | The most serious risk to the framing. Better to find it ourselves in Wave 4 than in review. |
| **HNET is the real competitor and beats everything** | Fine, and interesting — it is the same functional niche (§4.6). The differentiator is one-shot Hebbian writes vs. a per-task gradient-descent inner loop. Frame it that way from the start. |
| **Suite explodes combinatorially** | ~20 methods × HP grids × 20 seeds does not fit in a 12 h job. Gate strictly on wave exit criteria; carry only the best 2–3 configs per family forward; use the tuning/evaluation stream split (§7.3) to keep the eval runs small. |
| **N=5 is too short to separate methods** | Most CL methods look similar at 5 tasks and separate at 20–50. See Q1. |

---

## 10. Open questions

1. **Stream length.** Keep N=5 (matches every existing `agenthash` history), or
   extend to N=20–50? *Recommendation: keep N=5 as the headline figure for
   comparability, add an N=20 stream as a scaling panel.* Methods separate at
   length, and the Hopfield agent's flat-in-N story is most compelling there.
2. **Task boundaries.** Confirm we give the baselines free block boundaries.
   *Recommendation: yes* — it strengthens them, and CLEAR/SI/MESU/GDumb are
   reported separately as the boundary-free column.
3. **Input channels.** May the control gain `--input_prev_action` (and possibly
   a previous-reward channel)? `feedback_hopfield_nav_bc_inputs` freezes the
   input set for the **bc-AQ** line; I do not know whether that binds the
   continual-controls line. *This is the one item I will not change without an
   answer.*
4. **Movement mode.** Adopt `discrete` for the suite (§3.3)? It gives every
   method its canonical softmax form. *Recommendation: yes, with `continuous`
   as a robustness check on the survivors.*
5. **Compute envelope.** How many GPU-hours per wave? The current
   `run_baseline.sh` is 12 h / 32 CPU / 1 GPU for 30 seeds × 5 envs × 200
   updates; that number sets how wide each HP sweep can be.
6. **Scope check on Wave 4.** §5.3 (meta-pretraining) and §5.4 (in-context) are
   each a project-sized effort. Are they in scope for this paper, or noted as
   future work with §5.4's risk stated explicitly in the discussion?

---

## 11. Reading list

**Surveys / positioning**
- Wang et al. 2023, *A Comprehensive Survey of Continual Learning* — arXiv 2302.00487
- *A Survey of Continual Reinforcement Learning* 2025 — arXiv 2506.21872 (taxonomy + the metric definitions used in §7.1)
- *Advancements and Challenges in Continual RL* 2025 — arXiv 2506.21899
- van de Ven, Tuytelaars & Tolias 2022, *Three types of incremental learning*, Nat. Mach. Intell. — the scenario taxonomy
- van de Ven & Tolias 2019, *Three scenarios for continual learning* — arXiv 1904.07734 (the domain-IL numbers in §1)

**Replay**
- Rolnick et al. 2019, *Experience Replay for Continual Learning* (CLEAR) — arXiv 1811.11682
- Buzzega et al. 2020, *Dark Experience for General Continual Learning* (DER++) — arXiv 2004.07211
- Chaudhry et al. 2019 (A-GEM); Aljundi et al. 2019 (MIR); Prabhu et al. 2020 (GDumb)

**Regularisation**
- Kirkpatrick et al. 2017 (EWC); Schwarz et al. 2018 (online EWC, P&C); Zenke et al. 2017 (SI); Aljundi et al. 2018 (MAS)
- Liu et al., **CVPR 2026**, *Elastic Weight Consolidation Done Right* — arXiv 2603.18596
- Bruno et al., **Nat. Commun. 2025**, *Bayesian continual learning and forgetting* (MESU) — arXiv 2504.13569

**Recurrent-specific — read these first**
- **Ehret et al., ICLR 2021, *Continual Learning in Recurrent Neural Networks* — arXiv 2006.12109.** The single most relevant paper to this setting.
- Cossu et al. 2021, *Continual Learning for RNNs: an Empirical Evaluation* — arXiv 2103.07492
- von Oswald et al., ICLR 2020, *Continual learning with hypernetworks* — arXiv 1906.00695

**Isolation / projection**
- Mallya & Lazebnik 2018 (PackNet); Serra et al. 2018 (HAT); Rusu et al. 2016 (ProgNets)
- Masse, Grant & Freedman, **PNAS 2018**, *Alleviating catastrophic forgetting using context-dependent gating* — arXiv 1802.01569
- Saha et al., ICLR 2021, *Gradient Projection Memory* — arXiv 2103.09762
- Wołczyk et al. 2021, *Continual World* — arXiv 2105.10919 (the PackNet 0.80 vs 0.05 result)

**Meta / plasticity**
- Javed & White 2019, *Meta-Learning Representations for Continual Learning* (OML) — arXiv 1905.12588
- Beaulieu et al. 2020, *Learning to Continually Learn* (ANML) — arXiv 2002.09571
- Dohare et al., **Nature 2024**, *Loss of plasticity in deep continual learning*

**Evaluation methodology**
- Wang et al. 2024, *Hyperparameters in Continual Learning: A Reality Check* — arXiv 2403.09066
- De Lange et al. 2023, *Continual evaluation for lifelong learning: identifying the stability gap* — arXiv 2205.13452
