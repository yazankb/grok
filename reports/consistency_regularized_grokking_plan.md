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