#!/usr/bin/env python
"""
Consistency-regularized multi-specialist training for grokking.

This implements the experiment described in
``reports/consistency_regularized_grokking_plan.md``. M specialists are trained
in lockstep, each on its own shard of the labeled training data, with an
additional consistency loss that penalises pairwise disagreement on an
unlabeled-input domain (the full input grid by default).

Per-step loss for specialist i:

    L_i = CE(f_i; D_i) + lambda(t) * Cons(f_i, {sg(f_j)}_{j != i}; X_unsup)

where ``Cons`` is either MSE on logits or KL on softmax, both evaluated at
the answer-prediction position only. Teacher copies are stop-gradient'ed
(Mean Teacher convention) so each specialist's update only writes to its own
parameters.

Logging produced per run:
- consistency_metrics.csv  - per-step training metrics
- consistency_eval.csv     - per-eval metrics (val acc, entropy, pairwise KL,
                             weight norms, ensemble val acc)
- consistency_results.json - final summary + best-of-run + run config
- checkpoints/             - per-specialist init.pt, periodic and final state dicts
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import time
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from grok.data import ArithmeticDataset
from grok.training import TrainableTransformer
from grok.multi_training import (
    _experiment_artifacts_dir,
    _json_safe,
    _write_json,
    average_state_dicts,
    build_specialist_shards,
    save_environment_metadata,
    save_run_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(hparams: Namespace) -> str:
    if torch.cuda.is_available() and getattr(hparams, "gpu", -1) >= 0:
        return f"cuda:{hparams.gpu}"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _eq_token_position_in_target(target_tensor: torch.Tensor, eq_token_id: int) -> int:
    """Position of the '=' token inside the *target* sequence (data[:, 1:]).

    This matches the convention used by ``DistillationTrainer._rhs_accuracy``
    and the existing distillation eval loop in ``multi_training.py``: they
    locate '=' inside ``targets`` and then slice ``[..., eq_pos + 1:]`` to
    get the answer + final-EOS positions. We follow the same convention so
    the metrics are comparable across trainers.
    """
    row = target_tensor[0]
    pos = int(torch.nonzero(row == eq_token_id, as_tuple=False)[0].squeeze())
    return pos


def _weight_norm(model: torch.nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_sq += float(p.detach().pow(2).sum().item())
    return math.sqrt(total_sq)


def _rhs_accuracy(logits: torch.Tensor, targets: torch.Tensor, eq_pos_in_target: int) -> float:
    """Accuracy on the answer (post-'=') tokens. Matches DistillationTrainer
    exactly: ``eq_pos_in_target`` is the position of '=' inside the target
    sequence (data[:, 1:]); the slice ``[..., eq_pos + 1:]`` then covers the
    answer token and the trailing EOS."""
    y_hat = logits.transpose(-2, -1)
    y_hat_rhs = y_hat[..., eq_pos_in_target + 1:]
    y_rhs = targets[..., eq_pos_in_target + 1:]
    preds = torch.max(y_hat_rhs, dim=-2).indices
    row_acc = torch.min((preds == y_rhs), dim=-1).values
    return float(row_acc.float().mean().item()) * 100.0


def _answer_logit_position(eq_pos_in_target: int) -> int:
    """The position in the model's logit tensor where the next-token prediction
    *is the answer*. Targets and logits both have length seq_len-1 because
    data has shape [N, seq_len] and we work with data[:, :-1] / data[:, 1:].
    The '=' token sits at ``eq_pos_in_target`` inside targets, so the answer
    sits at ``eq_pos_in_target + 1`` inside both targets and the logit tensor's
    sequence dimension."""
    return eq_pos_in_target + 1


# ---------------------------------------------------------------------------
# Consistency trainer
# ---------------------------------------------------------------------------


class ConsistencyTrainer:
    """Lockstep trainer for M specialists with output-consistency regularisation.

    Each specialist owns its own optimizer and LR scheduler, both built via
    ``TrainableTransformer.configure_optimizers()`` so that the recipe matches
    the run hparams (``max_lr``, ``weight_decay``, warmup, etc.). The lockstep
    loop performs, at each step:

      1. Sample one labeled batch per specialist from its own shard.
      2. Sample one shared unlabeled batch from ``X_unsup``.
      3. Forward each specialist on (a) its labeled batch and (b) the shared
         unsup batch. Both forwards retain grad.
      4. Build per-specialist loss CE(shard) + lambda(t) * Cons(unsup), sum
         across specialists into one scalar, backward once. Because the
         consistency loss uses ``detach()`` on every "teacher" copy, gradients
         only flow into the corresponding specialist's parameters.
      5. Per-specialist grad clip + optimizer step + scheduler step.
    """

    def __init__(
        self,
        specialists: List[TrainableTransformer],
        shards: List[ArithmeticDataset],
        full_train_ds: ArithmeticDataset,
        val_ds: ArithmeticDataset,
        device: str,
        log_dir: str,
        consistency_loss: str = "mse_logits",
        consistency_lambda: float = 0.0,
        consistency_warmup_steps: int = 0,
        consistency_domain: str = "full_grid",
        consistency_batch_size: int = 256,
        shard_batch_size: int = 256,
        total_steps: int = 25000,
        eval_every: int = 500,
        checkpoint_every: int = 5000,
        log_every: int = 50,
    ) -> None:
        assert len(specialists) == len(shards)
        assert len(specialists) >= 2, "Consistency requires at least 2 specialists."
        if consistency_loss not in ("none", "mse_logits", "kl_softmax"):
            raise ValueError(f"Unknown consistency_loss: {consistency_loss!r}")
        if consistency_domain not in ("full_grid", "train_inputs_only"):
            raise ValueError(f"Unknown consistency_domain: {consistency_domain!r}")

        self.specialists = specialists
        self.shards = shards
        self.full_train_ds = full_train_ds
        self.val_ds = val_ds
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

        self.consistency_loss = consistency_loss
        self.consistency_lambda = float(consistency_lambda)
        self.consistency_warmup_steps = int(consistency_warmup_steps)
        self.consistency_domain = consistency_domain
        self.consistency_batch_size = int(consistency_batch_size)
        self.shard_batch_size = int(shard_batch_size)
        self.total_steps = int(total_steps)
        self.eval_every = int(eval_every)
        self.checkpoint_every = int(checkpoint_every)
        self.log_every = int(log_every)

        self.M = len(specialists)
        self.tokenizer = full_train_ds.tokenizer
        eq_token_id = self.tokenizer.stoi["="]
        # All datasets share the same encoding so eq_pos is consistent.
        # We follow the existing _rhs_accuracy convention: locate '=' inside
        # the target sequence (data[:, 1:]). The model's answer-prediction
        # logit then lives at index (eq_pos_in_target + 1) along the seq dim.
        target_tensor = full_train_ds.data[:, 1:]
        self.eq_pos = _eq_token_position_in_target(target_tensor, eq_token_id)
        self.answer_pos = _answer_logit_position(self.eq_pos)

        # Move all specialists to device and build optimizers
        self.optimizers: List[torch.optim.Optimizer] = []
        self.schedulers: List[Any] = []
        for spec in self.specialists:
            spec.to(self.device)
            opts, scheds = spec.configure_optimizers()
            self.optimizers.append(opts[0])
            if scheds and isinstance(scheds[0], dict):
                self.schedulers.append(scheds[0]["scheduler"])
            else:
                self.schedulers.append(None)

        # Build the unsup tensor once. ``data`` is a [N, seq_len] tensor of
        # token ids; we slice off the trailing token to match the model's
        # input convention (training uses ``data[:, :-1]``).
        if consistency_domain == "full_grid":
            unsup_full = torch.cat(
                [full_train_ds.data, val_ds.data], dim=0
            )
        else:  # train_inputs_only
            unsup_full = full_train_ds.data
        # Move to CPU; we'll index and transfer per batch
        self.unsup_inputs = unsup_full[:, :-1].clone()
        self.unsup_size = self.unsup_inputs.shape[0]

        # Shard data tensors for fast random sampling
        self.shard_inputs = [s.data[:, :-1].clone() for s in shards]
        self.shard_targets = [s.data[:, 1:].clone() for s in shards]
        self.shard_sizes = [s.data.shape[0] for s in shards]

        # Val tensors (cached on device)
        self.val_inputs = val_ds.data[:, :-1].to(self.device)
        self.val_targets = val_ds.data[:, 1:].to(self.device)
        self.full_train_inputs = full_train_ds.data[:, :-1].to(self.device)
        self.full_train_targets = full_train_ds.data[:, 1:].to(self.device)

        # Per-step CSV logger
        self._train_csv_path = os.path.join(log_dir, "consistency_metrics.csv")
        self._eval_csv_path = os.path.join(log_dir, "consistency_eval.csv")
        self._train_csv = open(self._train_csv_path, "w", newline="")
        self._train_writer = csv.writer(self._train_csv)
        self._train_writer.writerow(
            [
                "step",
                "lambda_t",
                "lr",
                *[f"ce_spec_{i}" for i in range(self.M)],
                *[f"cons_spec_{i}" for i in range(self.M)],
                *[f"shard_acc_spec_{i}" for i in range(self.M)],
                "elapsed_sec",
            ]
        )
        self._eval_csv = open(self._eval_csv_path, "w", newline="")
        self._eval_writer = csv.writer(self._eval_csv)
        self._eval_writer.writerow(
            [
                "step",
                *[f"val_acc_spec_{i}" for i in range(self.M)],
                "val_acc_ensemble",
                "val_acc_merged",
                "train_acc_ensemble",
                *[f"weight_norm_spec_{i}" for i in range(self.M)],
                "unsup_entropy_mean",
                "pairwise_kl_val_mean",
            ]
        )
        self._t0 = time.time()
        # Eval history for "best" tracking
        self.best_ensemble_val: float = -1.0
        self.best_ensemble_step: int = -1
        self.best_merged_val: float = -1.0
        self.best_merged_step: int = -1

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def lambda_t(self, step: int) -> float:
        if self.consistency_loss == "none" or self.consistency_lambda == 0.0:
            return 0.0
        if self.consistency_warmup_steps <= 0:
            return self.consistency_lambda
        return self.consistency_lambda * min(1.0, step / self.consistency_warmup_steps)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_shard(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.shard_sizes[i]
        bs = min(self.shard_batch_size, n)
        idx = torch.randint(0, n, (bs,))
        text = self.shard_inputs[i][idx].to(self.device)
        target = self.shard_targets[i][idx].to(self.device)
        return text, target

    def _sample_unsup(self) -> torch.Tensor:
        bs = min(self.consistency_batch_size, self.unsup_size)
        idx = torch.randint(0, self.unsup_size, (bs,))
        return self.unsup_inputs[idx].to(self.device)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _consistency_loss(
        self, all_unsup_logits: List[torch.Tensor], i: int
    ) -> torch.Tensor:
        if self.consistency_loss == "none":
            return torch.tensor(0.0, device=self.device)
        # Build teacher target = mean of detached others, at answer position
        others = [
            all_unsup_logits[j][:, self.answer_pos, :].detach()
            for j in range(self.M)
            if j != i
        ]
        teacher_ans = torch.stack(others, dim=0).mean(dim=0)
        student_ans = all_unsup_logits[i][:, self.answer_pos, :]
        if self.consistency_loss == "mse_logits":
            return F.mse_loss(student_ans, teacher_ans)
        if self.consistency_loss == "kl_softmax":
            student_log = F.log_softmax(student_ans, dim=-1)
            teacher_p = F.softmax(teacher_ans, dim=-1)
            return F.kl_div(student_log, teacher_p, reduction="batchmean")
        raise RuntimeError(f"Unhandled consistency_loss: {self.consistency_loss}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, Any]:
        for spec in self.specialists:
            spec.train()

        for step in range(1, self.total_steps + 1):
            lam = self.lambda_t(step)

            # Sample one shared unsup batch
            unsup_text = self._sample_unsup()

            # Forward all specialists on the unsup batch (with grad)
            unsup_logits: List[torch.Tensor] = []
            for spec in self.specialists:
                logits, _, _ = spec(unsup_text)
                unsup_logits.append(logits)

            # Sample per-specialist shard batches and compute losses
            shard_ce: List[torch.Tensor] = []
            shard_acc: List[float] = []
            cons_losses: List[torch.Tensor] = []

            total_loss = torch.tensor(0.0, device=self.device)
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            for i, spec in enumerate(self.specialists):
                text, target = self._sample_shard(i)
                logits, _, _ = spec(text)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                )
                shard_ce.append(ce.detach())
                with torch.no_grad():
                    shard_acc.append(_rhs_accuracy(logits, target, self.eq_pos))
                cons = self._consistency_loss(unsup_logits, i)
                cons_losses.append(cons.detach())
                if lam > 0 and self.consistency_loss != "none":
                    total_loss = total_loss + ce + lam * cons
                else:
                    total_loss = total_loss + ce

            total_loss.backward()

            for i, spec in enumerate(self.specialists):
                torch.nn.utils.clip_grad_norm_(spec.parameters(), 1.0)
                self.optimizers[i].step()
                if self.schedulers[i] is not None:
                    self.schedulers[i].step()

            # Periodic logging
            if step % self.log_every == 0 or step == 1:
                lr = float(self.optimizers[0].param_groups[0]["lr"])
                self._train_writer.writerow(
                    [
                        step,
                        f"{lam:.6f}",
                        f"{lr:.6e}",
                        *[f"{c.item():.6f}" for c in shard_ce],
                        *[f"{c.item():.6f}" for c in cons_losses],
                        *[f"{a:.4f}" for a in shard_acc],
                        f"{time.time() - self._t0:.1f}",
                    ]
                )
                self._train_csv.flush()

            # Periodic eval
            if step % self.eval_every == 0 or step == 1:
                eval_metrics = self.evaluate(step)
                self._eval_writer.writerow(
                    [
                        step,
                        *[f"{eval_metrics[f'val_acc_spec_{i}']:.4f}" for i in range(self.M)],
                        f"{eval_metrics['val_acc_ensemble']:.4f}",
                        f"{eval_metrics['val_acc_merged']:.4f}",
                        f"{eval_metrics['train_acc_ensemble']:.4f}",
                        *[f"{eval_metrics[f'weight_norm_spec_{i}']:.4f}" for i in range(self.M)],
                        f"{eval_metrics['unsup_entropy_mean']:.6f}",
                        f"{eval_metrics['pairwise_kl_val_mean']:.6f}",
                    ]
                )
                self._eval_csv.flush()
                if eval_metrics["val_acc_ensemble"] > self.best_ensemble_val:
                    self.best_ensemble_val = eval_metrics["val_acc_ensemble"]
                    self.best_ensemble_step = step
                if eval_metrics["val_acc_merged"] > self.best_merged_val:
                    self.best_merged_val = eval_metrics["val_acc_merged"]
                    self.best_merged_step = step
                print(
                    f"  [step {step:>6}/{self.total_steps}] "
                    f"lam={lam:.4f}  "
                    f"shard_ce_mean={float(torch.stack(shard_ce).mean()):.4f}  "
                    f"cons_mean={float(torch.stack(cons_losses).mean()):.4f}  "
                    f"val_ens={eval_metrics['val_acc_ensemble']:.2f}%  "
                    f"val_merged={eval_metrics['val_acc_merged']:.2f}%  "
                    f"H={eval_metrics['unsup_entropy_mean']:.3f}  "
                    f"pwKL={eval_metrics['pairwise_kl_val_mean']:.4f}",
                    flush=True,
                )

            # Periodic checkpoint
            if (
                self.checkpoint_every > 0
                and step % self.checkpoint_every == 0
                and step != self.total_steps
            ):
                self._save_specialists_checkpoint(step)

            for spec in self.specialists:
                spec.train()

        # Final checkpoints + summary
        self._save_specialists_checkpoint(self.total_steps, suffix="final")
        final_eval = self.evaluate(self.total_steps + 1, full_train_eval=True)
        # Also persist final eval row
        self._eval_writer.writerow(
            [
                self.total_steps + 1,
                *[f"{final_eval[f'val_acc_spec_{i}']:.4f}" for i in range(self.M)],
                f"{final_eval['val_acc_ensemble']:.4f}",
                f"{final_eval['val_acc_merged']:.4f}",
                f"{final_eval['train_acc_ensemble']:.4f}",
                *[f"{final_eval[f'weight_norm_spec_{i}']:.4f}" for i in range(self.M)],
                f"{final_eval['unsup_entropy_mean']:.6f}",
                f"{final_eval['pairwise_kl_val_mean']:.6f}",
            ]
        )
        self._eval_csv.flush()
        if final_eval["val_acc_ensemble"] > self.best_ensemble_val:
            self.best_ensemble_val = final_eval["val_acc_ensemble"]
            self.best_ensemble_step = self.total_steps
        if final_eval["val_acc_merged"] > self.best_merged_val:
            self.best_merged_val = final_eval["val_acc_merged"]
            self.best_merged_step = self.total_steps

        self._train_csv.close()
        self._eval_csv.close()

        return {
            "final_eval": final_eval,
            "best_val_acc_ensemble": self.best_ensemble_val,
            "best_val_acc_ensemble_step": self.best_ensemble_step,
            "best_val_acc_merged": self.best_merged_val,
            "best_val_acc_merged_step": self.best_merged_step,
            "train_metrics_csv": self._train_csv_path,
            "eval_metrics_csv": self._eval_csv_path,
            "elapsed_sec": time.time() - self._t0,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_in_batches(
        self, model: TrainableTransformer, inputs: torch.Tensor, batch_size: int = 1024
    ) -> torch.Tensor:
        """Forward through ``inputs`` in chunks, returning concatenated logits."""
        model.eval()
        outs = []
        for i in range(0, inputs.shape[0], batch_size):
            chunk = inputs[i:i + batch_size]
            logits, _, _ = model(chunk)
            outs.append(logits)
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def evaluate(self, step: int, full_train_eval: bool = False) -> Dict[str, float]:
        """Compute val/train metrics + ensemble/merged + diagnostics."""
        # Per-specialist val logits (answer position only after compute)
        val_logits_list: List[torch.Tensor] = []
        train_logits_list: List[torch.Tensor] = []
        for spec in self.specialists:
            val_logits = self._forward_in_batches(spec, self.val_inputs)
            val_logits_list.append(val_logits)
            if full_train_eval:
                tr_logits = self._forward_in_batches(spec, self.full_train_inputs)
                train_logits_list.append(tr_logits)

        out: Dict[str, float] = {}

        # Per-specialist val acc
        for i, vl in enumerate(val_logits_list):
            out[f"val_acc_spec_{i}"] = _rhs_accuracy(vl, self.val_targets, self.eq_pos)

        # Ensemble val acc (probability-averaged across specialists, then argmax)
        avg_probs = torch.stack(
            [F.softmax(vl, dim=-1) for vl in val_logits_list], dim=0
        ).mean(dim=0)  # [B, seq, vocab]
        ens_y_hat = avg_probs.transpose(-2, -1)[..., self.eq_pos + 1:]
        ens_preds = torch.max(ens_y_hat, dim=-2).indices
        ens_y_rhs = self.val_targets[..., self.eq_pos + 1:]
        ens_row_acc = torch.min((ens_preds == ens_y_rhs), dim=-1).values
        out["val_acc_ensemble"] = float(ens_row_acc.float().mean().item()) * 100.0

        # Merged val acc (weight average)
        merged_state = average_state_dicts(
            [spec.transformer.state_dict() for spec in self.specialists]
        )
        merged = copy.deepcopy(self.specialists[0])
        merged.transformer.load_state_dict(merged_state)
        merged_logits = self._forward_in_batches(merged, self.val_inputs)
        out["val_acc_merged"] = _rhs_accuracy(merged_logits, self.val_targets, self.eq_pos)
        del merged

        # Train ensemble acc (used for collapse diagnostic)
        if full_train_eval:
            avg_train_probs = torch.stack(
                [F.softmax(tl, dim=-1) for tl in train_logits_list], dim=0
            ).mean(dim=0)
            t_y_hat = avg_train_probs.transpose(-2, -1)[..., self.eq_pos + 1:]
            t_preds = torch.max(t_y_hat, dim=-2).indices
            t_rhs = self.full_train_targets[..., self.eq_pos + 1:]
            t_row = torch.min((t_preds == t_rhs), dim=-1).values
            out["train_acc_ensemble"] = float(t_row.float().mean().item()) * 100.0
        else:
            out["train_acc_ensemble"] = -1.0

        # Per-specialist weight norm
        for i, spec in enumerate(self.specialists):
            out[f"weight_norm_spec_{i}"] = _weight_norm(spec.transformer)

        # Output entropy on consistency domain (averaged across specialists)
        # Sample a chunk of unsup inputs to keep this cheap
        max_n = min(2048, self.unsup_size)
        idx = torch.randperm(self.unsup_size)[:max_n]
        u_inputs = self.unsup_inputs[idx].to(self.device)
        ent_total = 0.0
        for spec in self.specialists:
            logits = self._forward_in_batches(spec, u_inputs, batch_size=1024)
            p = F.softmax(logits[:, self.answer_pos, :], dim=-1)
            logp = torch.log(p.clamp_min(1e-12))
            ent = -(p * logp).sum(dim=-1).mean().item()
            ent_total += float(ent)
        out["unsup_entropy_mean"] = ent_total / self.M

        # Pairwise KL on val (answer position, symmetric KL)
        val_probs_ans = [
            F.softmax(vl[:, self.answer_pos, :], dim=-1) for vl in val_logits_list
        ]
        pair_kls: List[float] = []
        for i in range(self.M):
            for j in range(i + 1, self.M):
                p = val_probs_ans[i]
                q = val_probs_ans[j]
                lp = torch.log(p.clamp_min(1e-12))
                lq = torch.log(q.clamp_min(1e-12))
                kl_pq = (p * (lp - lq)).sum(dim=-1).mean().item()
                kl_qp = (q * (lq - lp)).sum(dim=-1).mean().item()
                pair_kls.append(0.5 * (float(kl_pq) + float(kl_qp)))
        out["pairwise_kl_val_mean"] = (
            sum(pair_kls) / len(pair_kls) if pair_kls else 0.0
        )

        return out

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_specialists_checkpoint(self, step: int, suffix: Optional[str] = None) -> None:
        ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        for i, spec in enumerate(self.specialists):
            tag = suffix or f"step_{step}"
            path = os.path.join(ckpt_dir, f"specialist_{i}_{tag}.pt")
            torch.save(
                {
                    "transformer_state_dict": spec.transformer.state_dict(),
                    "step": step,
                    "specialist_index": i,
                },
                path,
            )


# ---------------------------------------------------------------------------
# Top-level entry: train_multi_with_consistency
# ---------------------------------------------------------------------------


def train_multi_with_consistency(hparams: Namespace) -> str:
    """End-to-end multi-specialist training with consistency regularisation.

    Returns the experiment directory path.
    """
    n_models: int = hparams.n_models
    experiment_dir: str = os.path.join(hparams.logdir, hparams.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    cfg_path = save_run_config(experiment_dir, hparams)
    env_path = save_environment_metadata(experiment_dir)
    print(f"  Saved run config: {cfg_path}")
    print(f"  Saved environment metadata: {env_path}")

    seed = hparams.random_seed if hparams.random_seed != -1 else 42
    torch.manual_seed(seed)

    device = _resolve_device(hparams)
    print(f"  Device: {device}")

    print(f"\n{'='*60}")
    print(f"Consistency-regularised multi-specialist run: {n_models} specialists")
    print(f"  operator={hparams.math_operator}  train_pct={hparams.train_data_pct}%")
    print(f"  consistency_loss={hparams.consistency_loss}  "
          f"lambda={hparams.consistency_lambda}  "
          f"warmup={hparams.consistency_warmup_steps}")
    print(f"  consistency_domain={hparams.consistency_domain}")
    print(f"  total_steps={hparams.consistency_steps}")
    print(f"{'='*60}")

    # Build dataset and shards once
    full_train_ds, val_ds = ArithmeticDataset.splits(
        train_pct=hparams.train_data_pct,
        operator=hparams.math_operator,
        operand_length=hparams.operand_length,
        data_dir=hparams.datadir,
    )
    print(f"Full training set: {len(full_train_ds)} equations  "
          f"|  Validation set: {len(val_ds)} equations")

    sharding = getattr(hparams, "sharding", "disjoint")
    shards = build_specialist_shards(full_train_ds, n_models, sharding, seed=seed)
    print(f"Sharding strategy: {sharding}")
    for i, shard in enumerate(shards):
        print(f"  Shard {i}: {len(shard)} equations")

    # Build specialists
    specialists: List[TrainableTransformer] = []
    init_dir = os.path.join(experiment_dir, "init")
    os.makedirs(init_dir, exist_ok=True)
    for i in range(n_models):
        spec_hparams = copy.deepcopy(hparams)
        # Important: each specialist gets a different init seed to avoid
        # all four collapsing to the same trajectory by coincidence.
        torch.manual_seed(seed + i)
        spec = TrainableTransformer(
            spec_hparams,
            train_dataset=shards[i],
            val_dataset=val_ds,
        ).float()
        specialists.append(spec)
        torch.save(
            {"transformer_state_dict": spec.transformer.state_dict(),
             "specialist_index": i},
            os.path.join(init_dir, f"specialist_{i}_init.pt"),
        )
    # Reset RNG to a deterministic state for the trainer
    torch.manual_seed(seed)

    trainer = ConsistencyTrainer(
        specialists=specialists,
        shards=shards,
        full_train_ds=full_train_ds,
        val_ds=val_ds,
        device=device,
        log_dir=experiment_dir,
        consistency_loss=hparams.consistency_loss,
        consistency_lambda=hparams.consistency_lambda,
        consistency_warmup_steps=hparams.consistency_warmup_steps,
        consistency_domain=hparams.consistency_domain,
        consistency_batch_size=hparams.consistency_batch_size,
        shard_batch_size=hparams.shard_batch_size,
        total_steps=hparams.consistency_steps,
        eval_every=hparams.eval_every,
        checkpoint_every=hparams.checkpoint_every,
        log_every=hparams.log_every,
    )

    summary = trainer.fit()

    # Persist summary
    results_path = os.path.join(experiment_dir, "consistency_results.json")
    results = {
        "experiment_dir": os.path.abspath(experiment_dir),
        "run_config_path": os.path.abspath(cfg_path),
        "environment_path": os.path.abspath(env_path),
        "hparams_subset": {
            "n_models": n_models,
            "sharding": sharding,
            "train_data_pct": hparams.train_data_pct,
            "math_operator": hparams.math_operator,
            "max_lr": hparams.max_lr,
            "weight_decay": hparams.weight_decay,
            "anneal_lr": getattr(hparams, "anneal_lr", False),
            "warmup_steps": getattr(hparams, "warmup_steps", None),
            "consistency_loss": hparams.consistency_loss,
            "consistency_lambda": hparams.consistency_lambda,
            "consistency_warmup_steps": hparams.consistency_warmup_steps,
            "consistency_domain": hparams.consistency_domain,
            "consistency_batch_size": hparams.consistency_batch_size,
            "shard_batch_size": hparams.shard_batch_size,
            "consistency_steps": hparams.consistency_steps,
        },
        "best_val_acc_ensemble": summary["best_val_acc_ensemble"],
        "best_val_acc_ensemble_step": summary["best_val_acc_ensemble_step"],
        "best_val_acc_merged": summary["best_val_acc_merged"],
        "best_val_acc_merged_step": summary["best_val_acc_merged_step"],
        "final_eval": _json_safe(summary["final_eval"]),
        "elapsed_sec": summary["elapsed_sec"],
        "train_metrics_csv": os.path.abspath(summary["train_metrics_csv"]),
        "eval_metrics_csv": os.path.abspath(summary["eval_metrics_csv"]),
        "checkpoint_dir": os.path.abspath(os.path.join(experiment_dir, "checkpoints")),
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {results_path}")
    print(f"  best ensemble val acc = {summary['best_val_acc_ensemble']:.2f}% "
          f"@ step {summary['best_val_acc_ensemble_step']}")
    print(f"  best merged   val acc = {summary['best_val_acc_merged']:.2f}% "
          f"@ step {summary['best_val_acc_merged_step']}")
    print(f"  elapsed = {summary['elapsed_sec']:.1f}s")

    return experiment_dir


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def add_consistency_args(parser: ArgumentParser) -> ArgumentParser:
    """Extend an existing parser (built by ``add_multi_args``) with the
    consistency-trainer flags. Imported and called from
    ``scripts/train_multi.py``."""
    parser.add_argument(
        "--use_consistency",
        action="store_true",
        default=False,
        help="Run the consistency-regularised multi-specialist trainer "
             "(supersedes the legacy distillation pipeline).",
    )
    parser.add_argument(
        "--consistency_loss",
        type=str,
        default="mse_logits",
        choices=["none", "mse_logits", "kl_softmax"],
        help="Pairwise consistency loss between specialists, evaluated at "
             "the answer-prediction position.",
    )
    parser.add_argument(
        "--consistency_lambda",
        type=float,
        default=0.0,
        help="Weight of the consistency loss in the per-specialist objective. "
             "0 disables consistency (multi-specialist baseline).",
    )
    parser.add_argument(
        "--consistency_warmup_steps",
        type=int,
        default=0,
        help="If > 0, lambda(t) ramps linearly from 0 to consistency_lambda "
             "over the first N steps. Use to delay the consistency pressure "
             "so specialists do not memorise their shards before alignment "
             "kicks in (see plan §5 mode 2).",
    )
    parser.add_argument(
        "--consistency_domain",
        type=str,
        default="full_grid",
        choices=["full_grid", "train_inputs_only"],
        help="Which inputs the consistency loss is enforced on. 'full_grid' = "
             "train+val inputs (transductive); 'train_inputs_only' = train "
             "inputs only (no val inputs touched at training time).",
    )
    parser.add_argument(
        "--consistency_batch_size",
        type=int,
        default=256,
        help="Batch size for the unsup consistency forward pass.",
    )
    parser.add_argument(
        "--shard_batch_size",
        type=int,
        default=256,
        help="Per-specialist batch size sampled from its own shard each step.",
    )
    parser.add_argument(
        "--consistency_steps",
        type=int,
        default=25000,
        help="Total per-specialist gradient steps in the lockstep trainer.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=500,
        help="Run full evaluation (per-spec val, ensemble, merged, diagnostics) "
             "every N steps.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5000,
        help="Save per-specialist checkpoints every N steps. 0 disables "
             "intermediate checkpoints (final checkpoint is always written).",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Append a row to consistency_metrics.csv every N steps.",
    )
    return parser
