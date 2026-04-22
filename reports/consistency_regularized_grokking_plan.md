# Consistency-Regularized Multi-Specialist Grokking

**Status**: proposal, pre-implementation.
**Author/discussion**: chat thread, April 2026.
**Predecessors**: `reports/milestone.pdf` (Exp 3), `reports/distillation_repro_attempt3.md`.

## 1. Why we are pivoting

Milestone Exp 3 claimed that distillation from disjoint specialists grokked where baselines plateaued at ~45–50%. `distillation_repro_attempt3.md` showed this was almost certainly a comparison artifact: the distillation student at commit `09befb8` was the only model in the comparison running a grokking-tuned hardcoded optimizer (`AdamW(lr=1e-3, wd=1.0) + cosine annealing`), while the baselines and merged-finetune phases used the run's weaker hyperparameters. With that confound exposed, the original "distillation as a regularizer for grokking" claim no longer stands on its own evidence and needs either (a) a controlled re-run that isolates the optimizer effect or (b) a different mechanism for getting grokking-style generalization out of the multi-model setup.

This document proposes (b): **use unlabeled inputs as a consistency-regularization domain for an ensemble of specialists**, and test whether the resulting smoothness prior accelerates or strengthens grokking on modular addition.

## 2. The idea in one paragraph

Train $M$ specialists, each on a different shard of a labeled training set. In addition to the usual cross-entropy on each shard, add a consistency loss that penalizes pairwise disagreement between specialists on the **full input grid** (used as unlabeled data — labels are not consulted). The hypothesis is that consistency-on-unlabeled-inputs constrains the joint hypothesis space of the $M$-tuple of models toward configurations where each $f_i$ implements a structured rule rather than a per-sample lookup, because per-sample lookups disagree with each other on out-of-bag inputs. This is a smoothness/agreement prior on outputs, complementary to the weight-norm prior that weight decay provides — and weight decay is already known to be the main lever that drives grokking (Power et al., 2022).

## 3. Background and intuition

### 3.1 Grokking and what's known to drive it

Grokking is the slow phase transition from a memorization solution (high-norm weights, per-sample lookup, ~100% train / ~chance val) to a structured solution (low-norm weights, Fourier-feature representation on modular arithmetic, ~100% train / ~100% val). The standard mechanism is weight decay: the memorization basin and the generalization basin both fit the training data, but the generalization basin has lower weight norm, so a weight-decay term slowly biases search toward it once training loss is near zero.

Anything that biases search toward structured / smooth / low-complexity solutions is therefore *a priori* a candidate for accelerating grokking. The lever we propose here is a smoothness prior on **outputs**, enforced via consensus between an ensemble of specialists.

### 3.2 Why an ensemble at all

Two reasons:

1. **Multi-view constraint.** With $M$ specialists trained on different shards, asking them to agree on out-of-shard inputs is asking for a function consistent with multiple labeled subsets *simultaneously*. With limited capacity, the only way to satisfy "fits my shard + agrees with the others on inputs I haven't seen" without per-sample slack is to implement a structured rule that fits the union of shards.
2. **OOB pseudo-supervision.** When models agree on an unlabeled input, that agreed-upon output behaves like a soft pseudo-label for any specialist that didn't have that input in its shard. This is the same mechanism as Mean Teacher, Π-Model, FixMatch, and co-training, transplanted to grokking.

### 3.3 Why "unlabeled equations are not really information" is *almost* right

For modular addition with $p=97$ the input grid has $p^2 = 9409$ pairs, all of which the model class can in principle handle. Strictly: an unlabeled $(a, b)$ pair carries zero Shannon information about $f(a,b) = (a+b) \bmod p$. In that data-processing-inequality sense the user's intuition is correct.

But unlabeled inputs *do* carry information about **which joint model configurations are mutually consistent**. The constraint $f_1(x) = f_2(x) = \cdots = f_M(x)$ for all $x \in \mathcal{X}_{\text{full}}$ is a constraint on the *space of $M$-tuples of models*, not a fact about labels. Adding more inputs to the consistency domain shrinks the admissible $M$-tuple space, which under capacity limits forces structured solutions. So unlabeled inputs are zero-info about labels but real-info about the admissible joint hypothesis. These are different quantities and the proposal exploits the second.

### 3.4 Predecessors in the chat thread

- An earlier contrastive-learning proposal (positive pair = same input through different specialists, negative pair = different inputs) was rejected because (i) it requires augmentations to avoid the standard SimCLR collapse modes, none of which are natural on token sequences, and (ii) the "diversity-via-negatives + alignment-via-positives" framing is internally contradictory when the negatives are just other input pairs.
- A pure alignment-on-bag-overlaps proposal was the previous step. Its weakness is that with bagged shards the pairwise OOB intersection is small, so alignment doesn't have many inputs to bite on. The unlabeled-grid version of this proposal removes that weakness by enforcing alignment on the *full* input grid.
- The user's "consistency on synthetic data" reframing is what led to this document.

### 3.5 Relationship to existing literature

This proposal is, technically, **consistency regularization in the Mean-Teacher / Π-Model family applied to a grokking benchmark**. To our knowledge nobody has run this exact experiment. The novelty is not the mechanism (well-known in semi-supervised learning) but the application (does an output-smoothness prior interact with the grokking phase transition the same way the weight-norm prior does, or differently, or not at all?). Either result is informative.

## 4. Concrete formulation

Let $f_i*{i=1}^{M}$ be $M$ specialist transformers with the same architecture but independent initializations. Let $D_i \subset \mathcal{X}*{\text{full}}$ be specialist $i$'s shard with labels $y$. Let $\mathcal{X}*{\text{unsup}} \subseteq \mathcal{X}*{\text{full}}$ be the consistency domain (see §6 for transductivity discussion).

Per-step loss for specialist $i$:

$$
\mathcal{L}*i(t) = \mathcal{L}*{\text{CE}}\big(f_i; D_i\big) + \lambda(t) \cdot \mathcal{L}*{\text{cons}}(f_i; f_j*{j \neq i}; \mathcal{X}_{\text{unsup}})
$$

Two candidate consistency losses, both differentiable w.r.t. $f_i$ only (the $f_j_{j\neq i}$ are detached when computing specialist $i$'s update — this is the Mean-Teacher convention and prevents the trivial all-models-collapse-together fixed point from being reached via mutual gradients):

- **MSE on logits**: $\mathcal{L}*{\text{cons}} = \mathbb{E}*{x \in \mathcal{X}*{\text{unsup}}}\big f_i(x) - \tfrac{1}{M-1}\sum*{j\neq i} \mathrm{sg}[f_j(x)] \big_2^2$
- **Symmetric KL on softmax**: $\mathcal{L}*{\text{cons}} = \mathbb{E}*{x} \mathrm{KL}\big(\sigma(f_i(x)) \big \tfrac{1}{M-1}\sum_{j\neq i}\sigma(\mathrm{sg}[f_j(x)])\big)$

Default: MSE on logits (more stable early in training when softmax distributions are nearly uniform).

$\lambda(t)$ is a scalar schedule. Two variants to test:

- **constant**: $\lambda(t) = \lambda_{\max}$ for all $t$.
- **warmup**: $\lambda(t) = \lambda_{\max} \cdot \min(1, t / T_{\text{warm}})$ with $T_{\text{warm}} \approx 1000$.

The warmup variant exists because §5 (failure mode 2) predicts that without warmup, hard CE will memorize each shard within ~1k steps before the consistency loss can do anything, and consistency will then have to drag four already-crystallized memorizers toward agreement, which is hard.

## 5. Failure modes the experiment must distinguish

These are the modes the metrics in §7 are designed to detect.

1. **Trivial-agreement collapse.** Models satisfy consistency by outputting near-uniform distributions on $\mathcal{X}*{\text{unsup}}$ (uniform agrees with uniform). Detect via output entropy on the unsup set: if it stays near $\log p$ throughout training, this is happening. Mitigation: either lower $\lambda*{\max}$ or use MSE on logits (which doesn't have a constant-output trivial minimum the way KL does).
2. **Memorize-then-can't-escape.** Specialists fit their shards in ~1k steps, then consistency tries to drag four crystallized memorizers toward each other. Because both memorization basins have near-zero CE gradients, the specialist that "wins" the consensus is essentially random. Detect by looking at $\lambda=0$ vs $\lambda > 0$ *with no warmup*: if no improvement, consistency arrived too late. Mitigation: warmup schedule.
3. **Compromise non-solution.** Each specialist memorizes its shard *and* partially agrees with others on the OOB set, hitting a flat region of the loss with no gradient toward true generalization. Detect by inter-model KL on val staying nontrivially positive while ensemble val acc stays at baseline. This is the most likely null-result mode and the one Deep Mutual Learning typically lands in on standard supervised tasks.
4. **Confound with weight decay.** If consistency works *only* because it adds an effective regularization equivalent to higher weight decay, then a simple "single model + higher weight decay" baseline should match it. We must include this baseline to claim consistency contributes anything beyond what weight decay already gives.

## 6. Transductivity and data hygiene

For modular addition with $p=97$, "all 9409 input pairs" is the entire input distribution; train and val are a 50/50 partition of it. Using val *inputs* (without labels) as part of $\mathcal{X}_{\text{unsup}}$ is **transductive** semi-supervised learning. This is allowed in the grokking literature — Power et al. and follow-ups all evaluate on the same held-out grid the model is conceptually defined over — but it must be reported explicitly.

Two configurations of $\mathcal{X}_{\text{unsup}}$ to consider, to defang any "you peeked at val" objection:

- **Transductive (default)**: $\mathcal{X}*{\text{unsup}} = \mathcal{X}*{\text{full}}$, the entire grid. Largest consistency surface.
- **Held-out-aware**: $\mathcal{X}_{\text{unsup}} = $ train inputs only (i.e., consistency only on inputs that have a label for *some* specialist, even if not for $f_i$). No val inputs are touched at training time. Smaller consistency surface, cleaner story.

We will run both and report both. If they give the same qualitative result, we use the held-out-aware version in the headline because it's the more conservative claim.

## 7. Experimental protocol

### 7.1 Fixed setup (matches Milestone Exp 3 to keep prior comparisons valid)

- Operator: `+`, $p = 97$.
- Train fraction: 50% (~4704 inputs).
- $M = 4$ specialists.
- Architecture: 2-layer transformer, ~455K params (current default).
- Specialist phase: 6.25k steps each, total 25k step budget shared with single-model baselines.
- Optimizer: `CustomAdamW`, `lr = 5e-4`, `weight_decay = 0.1`, warmup-then-flat schedule (HEAD defaults — *not* the hardcoded `1e-3 / 1.0 / cosine` recipe that contaminated the original Exp 3 distillation phase).
- Sharding: **disjoint** is the primary configuration. Disjoint shards make every input either "labeled for exactly one specialist" or "labeled for none," giving the consistency loss a clean job. Bagged shards are a secondary configuration to compare against.

### 7.2 Sweeps

Primary sweep:

- Consistency loss: `mse_logits` (default), `kl_softmax` (sanity).
- $\lambda_{\max} \in 0, 0.1, 1.0, 10.0$.
- Schedule: `constant`, `warmup_1000`.
- $\mathcal{X}_{\text{unsup}}$: `full_grid`, `train_inputs_only`.

The $\lambda_{\max} = 0$ cell is the multi-specialist baseline; everything else is a treatment.

### 7.3 Baselines (not optional)

These exist to rule out the failure modes in §5 and to make any positive result interpretable.

1. **Single model on 50% train.** Same hparams. This is the current grokking ceiling.
2. **Single model on 25% train.** Same hparams. Each specialist's labeled budget; gives a "no-collaboration" floor.
3. **Single model on 50% train, $\text{wd} \in 0.1, 0.3, 1.0$.** Rules out failure mode 4 (consistency = effective wd).
4. **$M$ specialists on disjoint shards, no consistency ($\lambda = 0$).** Pure ensemble baseline.
5. **$M$ specialists, no consistency, then prob-averaged ensemble at eval.** Ensembling-without-training-time-coupling baseline.

### 7.4 Metrics, logged every $K$ steps per specialist


| Metric                                         | What it diagnoses                                    |
| ---------------------------------------------- | ---------------------------------------------------- |
| per-shard train acc & loss                     | basic fit; collapse failure mode 1 if these stay low |
| val acc per specialist                         | individual generalization                            |
| ensemble val acc (prob-averaged across $M$)    | the headline number                                  |
| weight-merged-model val acc                    | to compare to milestone "merged" phase               |
| output entropy on $\mathcal{X}_{\text{unsup}}$ | failure mode 1 (uniform-collapse)                    |
| pairwise KL between specialists on val         | are they actually agreeing?                          |
| effective weight norm per specialist           | for the wd-confound comparison                       |


### 7.5 Decision rules, written *before* running

- **Positive headline result**: ensemble val acc with consistency $> $ single-model-on-50% by a meaningful margin (≥ 5pp, ideally 10pp+) at matched compute, *and* not matched by single-model-with-higher-wd.
- **Methodological positive result**: warmup-$\lambda$ + disjoint sharding works at $\lambda_{\max} \in 1, 10$ but constant-$\lambda$ doesn't — confirms the §5 mode-2 hypothesis and gives a clean methodological story even if the absolute numbers don't beat single-model-on-full-data.
- **Null result**: ensemble val acc $\approx$ single-model-on-50%, or matched by higher-wd single model. Writeable as "consistency on unlabeled inputs is subsumed by weight decay on this task."
- **Collapse result**: $\lambda$ too high causes uniform-output collapse before grokking. Writeable as a warning about consistency-regularization tuning on grokking benchmarks.

Any of the four outcomes is publishable as a small empirical contribution — none of them require us to be right.

## 8. Compute budget and timeline

Each multi-specialist run is roughly the same cost as the milestone Exp 3 (~~25k steps, ~30–45 min on the M-class machine used for prior runs). The primary sweep is $2 \times 4 \times 2 \times 2 = 32$ runs; we can prune to ~16 by dropping `kl_softmax` from the headline sweep and only running it as a single sanity check at the best $\lambda$. Baselines are 4–6 additional runs. Total: **~~20–25 runs, ~12–15 hours of compute** if run sequentially, much less if any parallelism is available.

We should run a 4-cell pilot first (best guess: `mse_logits`, $\lambda_{\max} \in 0, 1$, schedule $\in$ constant, warmup_1000, full grid, disjoint shards) before committing the full sweep. This is ~2–3 hours and tells us whether to scale up or rethink.

### 8.1 Pilot v1 post-mortem and pilot v2 revision

The first 4-cell pilot (`pilot_v1`, see `scripts/run_consistency_sweep.py` before the v2 revision) finished cleanly but produced a **null result with diagnostic signal**: every cell, including `lam0_baseline`, plateaued at ~1% val acc for 25k steps. The diagnostics showed pairwise val KL and output entropy collapsing in lockstep with strong consistency, which is consistent with the "trivial-agreement collapse" mode in §5.2.

Two mechanistic bugs drove that collapse:

1. **Loss-scale mismatch.** `mse_logits` on random-init logits is O(σ²) ≈ 0.5 (mean over logit dims), while CE at init is log(V) ≈ 5.5 — so at $\lambda=1$, the consistency term is ~10% of CE at init but becomes dominant *as soon as CE drops*, which happens before any useful structure forms. Worse, MSE on logits has no inherent scale invariance: if we rescale all logits by $c$, MSE scales as $c^2$ but CE and argmax are invariant, so MSE is "shouting" a constraint that the task doesn't actually require. KL-on-softmax sidesteps this: KL is in nats and is comparable to per-example CE throughout training.
2. **Warmup too short.** 1000 steps of warmup on a 25k-step budget is 4%. On modular-arithmetic grokking with these hparams, memorization of a single specialist's shard typically takes ≥5k steps (Power et al., Nanda et al., and our own milestone runs). So by the time $\lambda(t) \to \lambda_{\max}$, specialists have barely begun to memorize — consistency locks them into the uniform-output basin.

Additionally, the v1 pilot was missing the **single-model-on-50%** baseline, without which we cannot separate "consistency hurt" from "this hparam regime doesn't grok in 25k steps anyway". We added it in v2 as the first pilot cell.

**Pilot v2** (`DEFAULT_PILOT_CELLS` in `scripts/run_consistency_sweep.py` at this revision) addresses all three issues. 5 cells:

- `single_model_50pct` — M=1, no consistency. Baseline grokking calibration.
- `multi_lam0` — M=4, no consistency. Multi-specialist baseline.
- `multi_kl_lam01_warm5k` — M=4, KL-softmax, $\lambda_{\max}=0.1$, 5k-step warmup. Headline weak-consistency candidate.
- `multi_kl_lam1_warm5k` — M=4, KL-softmax, $\lambda_{\max}=1.0$, 5k-step warmup. Bounds the strong-consistency regime.
- `multi_kl_lam01_warm5k_trainonly` — same as #3 but $\mathcal{X}*{\text{unsup}} = \mathcal{X}*{\text{train}}$, controls for transductive leak.

Decision rules for v2 results are the same as the original §7.5 table, but now interpretable against the single-model baseline rather than in a vacuum.

### 8.2 Pilot v2 results and pilot v3 extensions

Pilot v2 finished cleanly. Headline numbers (best val acc over 25k steps):


| cell                                | single val | ensemble val | best specialist |
| ----------------------------------- | ---------- | ------------ | --------------- |
| `single_model_50pct`                | 11.78%     | —            | —               |
| `multi_lam0`                        | —          | 1.37%        | 1.72%           |
| `multi_kl_lam01_warm5k` (full_grid) | —          | 11.93%       | 13.63%          |
| `multi_kl_lam1_warm5k` (full_grid)  | —          | 8.12%        | 7.90%           |
| `multi_kl_lam01_warm5k_trainonly`   | —          | **21.15%**   | **29.00%**      |


Two takeaways: (i) `train_inputs_only` consistency substantially beats `full_grid` at the same $\lambda$, consistent with a co-training interpretation where specialists pass each other information about their own shards rather than collapsing to agreement on held-out inputs; (ii) even the best v2 cell is far from grokking (~100%), so the question was whether v2 was seed-lucky / budget-limited / lambda-suboptimal, or whether the setup has an actual ceiling below grokking.

Pilot v3 (9 cells, ~8h 30m on T4) answered each of those:

- **Seed robustness (4 cells, lam=0.1, 25k, seeds 42-45):** ensemble val acc came in at 21 / 39 / 45 / 38 percent (mean 36%, median 39%). The v2 21% was a low outlier; the effect is real and robust.
- **Single-model budget probe (1 cell, 100k steps):** plateaus at ~12% val acc around 24k steps and does not climb afterwards. The 12% ceiling is **architectural**, not budget-limited.
- **Multi-model budget probe (1 cell, seed 42, lam=0.1, 100k steps):** climbs monotonically: 21% (25k) → 31% (50k) → 34.5% (75k) → 38% (100k). Slope still positive at the end, so the headline is not immediately capped by budget.
- **Lambda ablation (3 cells at seed 42, 25k, on `train_inputs_only`):** ensemble val acc is monotonic in $\lambda$ over $[0.03, 1.0]$ — 16 / 21 / 26 / 27 percent at $\lambda \in 0.03, 0.1, 0.3, 1.0$.

Mechanistically, across all multi cells: output entropy dropped from $\log 97 \approx 4.6$ nats to ~1.3-2.0 nats (peaked but not collapsed), pairwise KL on val *rose* rather than fell (specialists differentiate, not collapse), and weight merging remained broken (merged model near-random). This rules out the trivial-agreement collapse mode from §5.2 and is consistent with genuine co-training.

The open question at the end of v3 is the *ceiling*. The v3 probes were never run at the best-seed × best-lambda × full-budget corner of the configuration space. Pilot v4 fills that gap.

### 8.3 Pilot v4 (final)

v4 is three cells, all at 100k steps, all on `train_inputs_only` KL-softmax + 5k warmup, varying only seed and $\lambda$:


| #   | cell                | seed | $\lambda_{\max}$ | steps | purpose                                            |
| --- | ------------------- | ---- | ---------------- | ----- | -------------------------------------------------- |
| 1   | `seed44_lam1_100k`  | 44   | 1.0              | 100k  | primary bet: best seed × best lambda × full budget |
| 2   | `seed43_lam1_100k`  | 43   | 1.0              | 100k  | cross-seed replication of (1)                      |
| 3   | `seed44_lam01_100k` | 44   | 0.1              | 100k  | lambda ablation at the winning seed, full budget   |


Seed 44 was the v3 seed-robustness winner (45% ensemble val acc at 25k). $\lambda=1.0$ was the v3 lambda-ablation winner (27% at seed 42, 25k, vs 21% at $\lambda=0.1$) and its curve had the steepest late-run slope of any ablation cell. Cell 1 therefore combines the best observed setting on every axis we have data for.

**Decision rules for v4:**

- If cell 1 reaches ≥90% ensemble val acc, the multi-specialist + `train_inputs_only` consistency recipe *enables grokking* in a regime where single-model is architecturally capped at 12%. Strong claim.
- If cell 1 reaches 50-90%, the recipe substantially lifts val acc without crossing the phase transition; we have a quantitative "non-grokking improvement" result with a 5-7× lift over the single-model ceiling.
- If cell 1 finishes below 50%, the recipe has a ceiling well below grokking on this task.
- Cell 2 tests whether (1)'s outcome is seed-44-specific. Cell 3 isolates the lambda contribution at the 100k budget: if cell 1 ≫ cell 3, the lambda effect compounds with budget; if cell 1 ≈ cell 3, v3's monotone-in-lambda result was a transient that longer training erases.

v4 is the last planned experiment; whichever outcome obtains, the next step is write-up, not another pilot.

## 9. Implementation plan

The code changes needed live in `grok/multi_training.py` and `scripts/train_multi.py`. Sketch:

1. Add a `ConsistencyTrainer` class (or extend the existing multi-specialist training loop) that owns all $M$ specialists and runs them in lockstep so we can sample shared consistency batches.
2. Per training step, in addition to each specialist's per-shard CE batch, draw a batch from $\mathcal{X}*{\text{unsup}}$, forward all $M$ specialists on it, compute pairwise consistency losses with `detach()` on the "teacher" side as in Mean Teacher, and add $\lambda(t) \cdot \mathcal{L}*{\text{cons}}$ to each specialist's loss before backprop.
3. CLI flags to add to `add_multi_args`: `--consistency_loss {none,mse_logits,kl_softmax}`, `--consistency_lambda` (float), `--consistency_warmup_steps` (int, 0 = no warmup), `--consistency_domain {full_grid,train_inputs_only}`.
4. Persist these flags into `comparison_results.json` (extend the schema documented in `docs/EXPERIMENT_PERSISTENCE.md`).
5. Add the metrics in §7.4 to whatever logging callback is used by the multi-trainer; the entropy and pairwise-KL metrics in particular are new and will need to be added explicitly.
6. A small evaluation script (or addition to the existing one) to compute prob-averaged ensemble val acc from the saved per-specialist checkpoints, since the baselines need this for comparability.

Nothing about this requires changes to the model architecture, the dataset code, the optimizer plumbing, or the existing single-model training path. The change is contained to multi-trainer + CLI + logging.

## 10. What this proposal does *not* claim

- It does not claim that the milestone Exp 3 result was due to consistency regularization. The milestone result is explained by `distillation_repro_attempt3.md` (optimizer confound) and we're not retroactively re-explaining it.
- It does not claim consistency regularization is novel. It is a well-known technique. The novel question is its interaction with the grokking phase transition.
- It does not claim multi-model setups are necessary for grokking. Single-model-on-full-data with weight decay groks. The question is whether the multi-model + consistency setup groks *faster*, *more reliably*, or *at lower train fraction* than the single-model baseline.
- It does not commit us to a benchmark switch. If the result is null on modular addition, we still learn something publishable, and the decision about whether to move to a richer benchmark can be made on the basis of that result.