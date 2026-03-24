#!/usr/bin/env python
"""
Multi-model grokking experiment.

Three-phase pipeline:
  1. Split the training data into N disjoint shards and train one specialist
     transformer per shard until it overfits (train_accuracy >= threshold).
  2. Average the specialist weight tensors into a single merged model.
  3. Fine-tune the merged model on the *full* training data and track grokking.
     Optionally, also run a random-init baseline on the full data for comparison.
  4. Distill knowledge from specialists into a student model.

Usage (from repo root):
    python scripts/train_multi.py --n_models 4 --train_data_pct 50 \
        --specialist_steps 50000 --final_steps 100000
"""

import copy
import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger

import pytorch_lightning as pl

_LIGHTNING_2 = (
    getattr(pl, "__version__", "0").split(".")[0].isdigit()
    and int(pl.__version__.split(".")[0]) >= 2
)

from grok.data import ArithmeticDataset
from grok.training import DEFAULT_LOG_DIR, TrainableTransformer, add_args
from grok.transformer import Transformer


# ---------------------------------------------------------------------------
# Early stopping callback
# ---------------------------------------------------------------------------

class EarlyStopOnOverfit(Callback):
    """
    Stop a specialist model once its training accuracy crosses a threshold.

    Reads `train_accuracy` from `trainer.callback_metrics`, which is populated
    by TrainableTransformer's on_train_epoch_end logging.

    NOTE: train_accuracy is logged in [0, 100] (percentage), so the threshold
    must also be in that range (e.g. 99.0 means 99%, not 0.99).
    """

    def __init__(self, threshold: float = 99.0) -> None:
        super().__init__()
        self.threshold = threshold

    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:
        # Prefer full_train_acc (computed on the entire shard every validation
        # epoch) because train_accuracy is only written to callback_metrics at
        # exponentially-spaced epochs and may lag.
        acc = trainer.callback_metrics.get("full_train_acc", None)
        if acc is None:
            acc = trainer.callback_metrics.get("train_accuracy", None)
        if acc is not None and float(acc) >= self.threshold:
            trainer.should_stop = True


# ---------------------------------------------------------------------------
# Weight averaging
# ---------------------------------------------------------------------------

def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Element-wise arithmetic mean of a list of state dicts.

    All tensors are cast to float32 for the average and then cast back to the
    dtype of the first state dict, so integer buffers (e.g. step counters) are
    handled safely.
    """
    ref = state_dicts[0]
    avg: Dict[str, torch.Tensor] = {}
    for key in ref:
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        avg[key] = stacked.mean(dim=0).to(ref[key].dtype)
    return avg


# ---------------------------------------------------------------------------
# Knowledge Distillation
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """
    Trains a student model using knowledge distillation from multiple teacher models.
    
    Distillation loss = alpha * KLDiv(student_soft, teacher_soft) + (1-alpha) * CE(student, labels)
    
    Teacher predictions are averaged across all specialists.
    """
    
    def __init__(
        self,
        teacher_models: List[TrainableTransformer],
        student_model: TrainableTransformer,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = "cuda",
    ):
        self.teachers = [t.eval() for t in teacher_models]
        self.student = student_model.train()
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
    @torch.no_grad()
    def get_teacher_logits(self, batch: Dict) -> torch.Tensor:
        """Get averaged logits from all teachers."""
        all_logits = []
        for teacher in self.teachers:
            logits, _, _ = teacher(batch["text"].to(self.device))
            all_logits.append(logits.to(self.device))
        return torch.stack(all_logits).mean(dim=0)
    
    def distillation_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss using KL divergence with temperature scaling.
        """
        T = self.temperature
        
        # Move teacher logits to same device as student
        teacher_logits = teacher_logits.to(student_logits.device)
        
        # Get answer positions (right-hand side tokens)
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        
        # KL divergence averaged over tokens
        kl_div = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # Scale by T^2 as per Hinton et al.
        return kl_div * (T * T)
    
    def hard_loss(
        self, 
        student_logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard cross-entropy loss on hard labels.
        """
        return F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            targets.reshape(-1).to(self.device)
        )
    
    def train_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Single distillation training step.
        Returns (loss, metrics_dict)
        """
        text = batch["text"].to(self.device)
        targets = batch["target"].to(self.device)
        
        student_logits, _, _ = self.student(text)
        teacher_logits = self.get_teacher_logits({"text": text})
        
        # Combined loss
        soft_loss = self.distillation_loss(student_logits, teacher_logits)
        hard_loss = self.hard_loss(student_logits, targets)
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        # Accuracy
        with torch.no_grad():
            student_pred = student_logits.argmax(dim=-1)
            accuracy = (student_pred == targets).float().mean() * 100
            teacher_pred = teacher_logits.argmax(dim=-1)
            teacher_acc = (teacher_pred == targets).float().mean() * 100
        
        metrics = {
            "loss": loss.item(),
            "soft_loss": soft_loss.item(),
            "hard_loss": hard_loss.item(),
            "student_acc": accuracy.item(),
            "teacher_acc": teacher_acc.item(),
        }
        return loss, metrics


def distill_from_specialists(
    specialist_models: List[TrainableTransformer],
    student_model: TrainableTransformer,
    train_dataset,
    val_dataset,
    log_dir: str,
    distill_steps: int = 10000,
    temperature: float = 2.0,
    alpha: float = 0.5,
    device: str = "cuda",
) -> Tuple[TrainableTransformer, Dict]:
    """
    Distill knowledge from specialist models into a student model.
    
    Returns the trained student and training metrics.
    """
    print(f"\n{'='*60}")
    print(f"Distillation: T={temperature}, alpha={alpha}")
    print(f"{'='*60}")
    
    distill_trainer = DistillationTrainer(
        teacher_models=specialist_models,
        student_model=student_model,
        temperature=temperature,
        alpha=alpha,
        device=device,
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=1e-3,
        weight_decay=1.0,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=distill_steps, eta_min=1e-5
    )
    
    metrics = {
        "loss": [], "soft_loss": [], "hard_loss": [],
        "student_acc": [], "teacher_acc": [],
        "val_acc": [], "steps": []
    }
    
    step = 0
    student_model = student_model.train()
    
    while step < distill_steps:
        for batch in train_loader:
            if step >= distill_steps:
                break
                
            optimizer.zero_grad()
            loss, m = distill_trainer.train_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            metrics["loss"].append(m["loss"])
            metrics["soft_loss"].append(m["soft_loss"])
            metrics["hard_loss"].append(m["hard_loss"])
            metrics["student_acc"].append(m["student_acc"])
            metrics["teacher_acc"].append(m["teacher_acc"])
            metrics["steps"].append(step)
            
            step += 1
            
            if step % 500 == 0:
                # Validate
                student_model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        logits, _, _ = student_model(val_batch["text"].to(device))
                        preds = logits.argmax(dim=-1)
                        targets = val_batch["target"].to(device)
                        correct += (preds == targets).sum().item()
                        total += targets.numel()
                val_acc = correct / total * 100 if total > 0 else 0
                metrics["val_acc"].append(val_acc)
                student_model.train()
                
                print(f"  Step {step}/{distill_steps}: "
                      f"train_acc={m['student_acc']:.2f}%, "
                      f"val_acc={val_acc:.2f}%, "
                      f"loss={m['loss']:.4f}")
    
    # Final validation
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for val_batch in val_loader:
            logits, _, _ = student_model(val_batch["text"].to(device))
            preds = logits.argmax(dim=-1)
            targets = val_batch["target"].to(device)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    final_val_acc = correct / total * 100 if total > 0 else 0
    print(f"  Final distillation val_acc: {final_val_acc:.2f}%")
    
    return student_model, metrics


# ---------------------------------------------------------------------------
# Trainer builder
# ---------------------------------------------------------------------------

def _build_trainer(
    hparams: Namespace,
    log_dir: str,
    max_steps: int,
    min_steps: int = 1,
    extra_callbacks: Optional[List[Callback]] = None,
    check_val_every_n_epoch: int = 25,
) -> Trainer:
    logger = CSVLogger(log_dir)
    trainer_args = {
        "max_steps": max_steps,
        "min_steps": min_steps,
        "max_epochs": int(1e8),
        # Run validation once every N training epochs rather than every batch.
        # val_check_interval=1 (per batch) caused Lightning to invoke the full
        # validation loop after every single training step, creating large
        # Python overhead even when validation_step returned early {}.
        # check_val_every_n_epoch avoids the "must be <= training batches"
        # constraint and reduces validation overhead by ~25x.
        "check_val_every_n_epoch": check_val_every_n_epoch,
        "logger": logger,
        "log_every_n_steps": 1,
        "callbacks": extra_callbacks or [],
    }
    if torch.cuda.is_available() and hparams.gpu >= 0:
        if _LIGHTNING_2:
            trainer_args["accelerator"] = "gpu"
            trainer_args["devices"] = [hparams.gpu]
        else:
            trainer_args["gpus"] = [hparams.gpu]
    else:
        if _LIGHTNING_2:
            trainer_args["accelerator"] = "cpu"
        else:
            trainer_args["gpus"] = None
    return Trainer(**trainer_args)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def train_multi(hparams: Namespace) -> str:
    """
    Run the full multi-model grokking experiment and return the experiment
    log directory path.

    Phases:
      1. Generate full train/val split, then partition train into N shards.
      2. Train each specialist on its shard for specialist_steps steps.
         If equal_compute=True, specialist_steps = final_steps // n_models,
         so total specialist compute equals one baseline run (fair comparison).
         If overfit_stop=True, also add early stopping via EarlyStopOnOverfit.
      3. Average the N transformer weight tensors.
      4. Fine-tune the merged model on the full training data.
      5. (Optional) Train a random-init baseline on the full data.
    """
    n_models: int = hparams.n_models
    final_steps: int = hparams.final_steps
    overfit_threshold: float = hparams.overfit_threshold
    experiment_dir: str = os.path.join(hparams.logdir, hparams.experiment_name)

    # Equal-compute mode: each specialist runs for final_steps // n_models
    # so total specialist gradient steps == one baseline run.
    if getattr(hparams, "equal_compute", False):
        specialist_steps = final_steps // n_models
        print(
            f"equal_compute=True: specialist_steps = {final_steps} // {n_models} "
            f"= {specialist_steps}"
        )
    else:
        specialist_steps = hparams.specialist_steps

    seed = hparams.random_seed if hparams.random_seed != -1 else 42

    # -----------------------------------------------------------------------
    # Phase 0: Build the full dataset once so all models share the same split
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Multi-model grokking: {n_models} specialists")
    print(f"Operator: {hparams.math_operator}  |  train_data_pct: {hparams.train_data_pct}%")
    print(f"{'='*60}")

    full_train_ds, val_ds = ArithmeticDataset.splits(
        train_pct=hparams.train_data_pct,
        operator=hparams.math_operator,
        operand_length=hparams.operand_length,
        data_dir=hparams.datadir,
    )
    print(
        f"Full training set: {len(full_train_ds)} equations  |  "
        f"Validation set: {len(val_ds)} equations"
    )

    # -----------------------------------------------------------------------
    # Phase 1: Train specialist models on disjoint data shards
    # -----------------------------------------------------------------------
    shards = ArithmeticDataset.split_n_ways(full_train_ds, n_models, seed=seed)
    for i, shard in enumerate(shards):
        print(f"  Shard {i}: {len(shard)} equations")

    specialist_state_dicts: List[Dict[str, torch.Tensor]] = []

    for i, shard in enumerate(shards):
        print(f"\n{'='*60}")
        print(f"Phase 1 — Specialist {i + 1}/{n_models}  (shard size: {len(shard)})")
        print(f"{'='*60}")

        spec_hparams = copy.deepcopy(hparams)
        spec_hparams.logdir = os.path.join(experiment_dir, f"specialist_{i}")
        ckpt_dir = os.path.join(spec_hparams.logdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        spec_hparams.checkpoint_path = ckpt_dir

        model = TrainableTransformer(
            spec_hparams,
            train_dataset=shard,
            val_dataset=val_ds,
        ).float()

        torch.save(model, os.path.join(ckpt_dir, "init.pt"))

        # Early stopping is optional.  In equal-compute mode it is usually
        # disabled so every specialist uses its full budget and has a chance to
        # grok its own shard, not just memorise it.
        callbacks = []
        if getattr(hparams, "overfit_stop", False):
            callbacks.append(EarlyStopOnOverfit(threshold=overfit_threshold))

        trainer = _build_trainer(
            hparams,
            spec_hparams.logdir,
            max_steps=specialist_steps,
            min_steps=1,
            extra_callbacks=callbacks,
        )
        trainer.fit(model)

        final_train_acc = float(
            trainer.callback_metrics.get("train_accuracy", float("nan"))
        )
        final_val_acc = float(
            trainer.callback_metrics.get("val_accuracy", float("nan"))
        )
        print(
            f"  Specialist {i} done — "
            f"train_acc={final_train_acc:.4f}  val_acc={final_val_acc:.4f}"
        )

        specialist_state_dicts.append(copy.deepcopy(model.transformer.state_dict()))

    # -----------------------------------------------------------------------
    # Phase 2: Merge (average) specialist weights
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 2 — Averaging weights from {n_models} specialists")
    print(f"{'='*60}")

    merged_weights = average_state_dicts(specialist_state_dicts)

    merge_path = os.path.join(experiment_dir, "merged_weights.pt")
    torch.save(merged_weights, merge_path)
    print(f"  Merged weights saved to {merge_path}")

    # -----------------------------------------------------------------------
    # Phase 3: Fine-tune merged model on the full training data
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 3 — Fine-tuning merged model on full training data")
    print(f"{'='*60}")

    merged_hparams = copy.deepcopy(hparams)
    merged_hparams.logdir = os.path.join(experiment_dir, "merged")
    ckpt_dir = os.path.join(merged_hparams.logdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    merged_hparams.checkpoint_path = ckpt_dir

    merged_model = TrainableTransformer(
        merged_hparams,
        train_dataset=full_train_ds,
        val_dataset=val_ds,
    ).float()

    merged_model.transformer.load_state_dict(merged_weights)
    torch.save(merged_model, os.path.join(ckpt_dir, "init.pt"))

    trainer = _build_trainer(
        hparams,
        merged_hparams.logdir,
        max_steps=final_steps,
        min_steps=final_steps,
    )
    trainer.fit(merged_model)

    final_val_acc = float(
        trainer.callback_metrics.get("val_accuracy", float("nan"))
    )
    print(f"  Merged model final val_acc={final_val_acc:.4f}")

    # -----------------------------------------------------------------------
    # Phase 4 (optional): Baseline — random-init model on full data
    # -----------------------------------------------------------------------
    if hparams.run_baseline:
        print(f"\n{'='*60}")
        print("Phase 4 — Baseline: random-init model on full training data")
        print(f"{'='*60}")

        baseline_hparams = copy.deepcopy(hparams)
        baseline_hparams.logdir = os.path.join(experiment_dir, "baseline")
        ckpt_dir = os.path.join(baseline_hparams.logdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        baseline_hparams.checkpoint_path = ckpt_dir

        baseline_model = TrainableTransformer(
            baseline_hparams,
            train_dataset=full_train_ds,
            val_dataset=val_ds,
        ).float()

        torch.save(baseline_model, os.path.join(ckpt_dir, "init.pt"))

        trainer = _build_trainer(
            hparams,
            baseline_hparams.logdir,
            max_steps=final_steps,
            min_steps=final_steps,
        )
        trainer.fit(baseline_model)

        final_val_acc = float(
            trainer.callback_metrics.get("val_accuracy", float("nan"))
        )
        print(f"  Baseline model final val_acc={final_val_acc:.4f}")

    print(f"\n{'='*60}")
    print(f"Experiment complete. Logs in: {experiment_dir}")
    print(f"{'='*60}")

    return experiment_dir


def train_multi_with_distillation(hparams: Namespace) -> str:
    """
    Run multi-model grokking experiment with BOTH weight averaging AND distillation.
    
    This allows direct comparison of the two merging approaches.
    """
    import json
    import numpy as np
    
    n_models: int = hparams.n_models
    final_steps: int = hparams.final_steps
    overfit_threshold: float = hparams.overfit_threshold
    experiment_dir: str = os.path.join(hparams.logdir, hparams.experiment_name)

    seed = hparams.random_seed if hparams.random_seed != -1 else 42

    print(f"\n{'='*60}")
    print(f"Multi-model grokking with DISTILLATION: {n_models} specialists")
    print(f"Operator: {hparams.math_operator}  |  train_data_pct: {hparams.train_data_pct}%")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Phase 0: Build the full dataset
    # -----------------------------------------------------------------------
    full_train_ds, val_ds = ArithmeticDataset.splits(
        train_pct=hparams.train_data_pct,
        operator=hparams.math_operator,
        operand_length=hparams.operand_length,
        data_dir=hparams.datadir,
    )
    print(f"Full training set: {len(full_train_ds)} equations")
    print(f"Validation set: {len(val_ds)} equations")

    # -----------------------------------------------------------------------
    # Phase 1: Train specialist models
    # -----------------------------------------------------------------------
    shards = ArithmeticDataset.split_n_ways(full_train_ds, n_models, seed=seed)
    
    specialist_models: List[TrainableTransformer] = []
    
    for i, shard in enumerate(shards):
        print(f"\n{'='*60}")
        print(f"Phase 1 — Specialist {i + 1}/{n_models}  (shard size: {len(shard)})")
        print(f"{'='*60}")

        spec_hparams = copy.deepcopy(hparams)
        spec_hparams.logdir = os.path.join(experiment_dir, f"specialist_{i}")
        ckpt_dir = os.path.join(spec_hparams.logdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        spec_hparams.checkpoint_path = ckpt_dir
        if not hasattr(spec_hparams, 'random_seed'):
            spec_hparams.random_seed = seed

        model = TrainableTransformer(
            spec_hparams,
            train_dataset=shard,
            val_dataset=val_ds,
        ).float()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        callbacks = []
        if getattr(hparams, "overfit_stop", False):
            callbacks.append(EarlyStopOnOverfit(threshold=overfit_threshold))

        specialist_steps = getattr(hparams, "specialist_steps", 50000)
        if getattr(hparams, "equal_compute", False):
            specialist_steps = final_steps // n_models

        trainer = _build_trainer(
            hparams,
            spec_hparams.logdir,
            max_steps=specialist_steps,
            min_steps=1,
            extra_callbacks=callbacks,
        )
        trainer.fit(model)

        specialist_models.append(model)
        print(f"  Specialist {i} done")

    # -----------------------------------------------------------------------
    # Phase 2: Weight Averaging
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 2 — Weight Averaging")
    print(f"{'='*60}")

    specialist_state_dicts = [copy.deepcopy(m.transformer.state_dict()) for m in specialist_models]
    merged_weights = average_state_dicts(specialist_state_dicts)
    
    # Create merged model for fine-tuning
    merged_hparams = copy.deepcopy(hparams)
    merged_hparams.logdir = os.path.join(experiment_dir, "merged_average")
    merged_hparams.checkpoint_path = os.path.join(merged_hparams.logdir, "checkpoints")
    merged_model = TrainableTransformer(
        merged_hparams,
        train_dataset=full_train_ds,
        val_dataset=val_ds,
    ).float().to(device)
    merged_model.transformer.load_state_dict(merged_weights)

    # -----------------------------------------------------------------------
    # Phase 3: Fine-tune Merged (Average) Model
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 3 — Fine-tuning Weight-Averaged Model")
    print(f"{'='*60}")

    trainer = _build_trainer(
        hparams,
        merged_hparams.logdir,
        max_steps=final_steps,
        min_steps=final_steps,
    )
    trainer.fit(merged_model)
    
    # -----------------------------------------------------------------------
    # Phase 4: Distillation
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 4 — Knowledge Distillation")
    print(f"{'='*60}")

    # Create fresh student model
    distill_hparams = copy.deepcopy(hparams)
    distill_hparams.logdir = os.path.join(experiment_dir, "distilled")
    distill_model = TrainableTransformer(
        distill_hparams,
        train_dataset=full_train_ds,
        val_dataset=val_ds,
    ).float().to(device)

    distill_steps = getattr(hparams, "distill_steps", final_steps // 2)
    temperature = getattr(hparams, "distill_temperature", 2.0)
    alpha = getattr(hparams, "distill_alpha", 0.5)

    distilled_model, distill_metrics = distill_from_specialists(
        specialist_models=specialist_models,
        student_model=distill_model,
        train_dataset=full_train_ds,
        val_dataset=val_ds,
        log_dir=distill_hparams.logdir,
        distill_steps=distill_steps,
        temperature=temperature,
        alpha=alpha,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Phase 4 (optional): Baseline — random-init model on full data
    # -----------------------------------------------------------------------
    if hparams.run_baseline:
        print(f"\n{'='*60}")
        print(f"Phase 4 — Baseline: random-init model on full training data")
        print(f"{'='*60}")

        baseline_hparams = copy.deepcopy(hparams)
        baseline_hparams.logdir = os.path.join(experiment_dir, "baseline")
        baseline_hparams.checkpoint_path = os.path.join(baseline_hparams.logdir, "checkpoints")
        baseline_model = TrainableTransformer(
            baseline_hparams,
            train_dataset=full_train_ds,
            val_dataset=val_ds,
        ).float().to(device)

        trainer = _build_trainer(
            hparams,
            baseline_hparams.logdir,
            max_steps=final_steps,
            min_steps=final_steps,
        )
        trainer.fit(baseline_model)

    # -----------------------------------------------------------------------
    # Save comparison results
    # -----------------------------------------------------------------------
    results_path = os.path.join(experiment_dir, "comparison_results.json")
    results = {
        "n_models": n_models,
        "specialist_steps": specialist_steps,
        "final_steps": final_steps,
        "distill_steps": distill_steps,
        "temperature": temperature,
        "alpha": alpha,
        "distill_final_metrics": {
            "final_loss": float(np.mean(distill_metrics["loss"][-100:])),
            "final_student_acc": float(np.mean(distill_metrics["student_acc"][-100:])),
        },
        "experiment_dir": experiment_dir,
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")

    return experiment_dir


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def add_multi_args(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    """
    Extends the base training argument parser with multi-model experiment flags.
    """
    parser = add_args(parser)

    parser.add_argument(
        "--n_models",
        type=int,
        default=4,
        help="Number of specialist models (= number of data shards)",
    )
    parser.add_argument(
        "--specialist_steps",
        type=int,
        default=50000,
        help="Maximum gradient steps for each specialist model",
    )
    parser.add_argument(
        "--final_steps",
        type=int,
        default=100000,
        help="Gradient steps for the merged model and baseline",
    )
    parser.add_argument(
        "--overfit_threshold",
        type=float,
        default=99.0,
        help=(
            "Train accuracy (in %%, range 0-100) at which specialist training "
            "is stopped early. Must be in [0,100] because train_accuracy is "
            "logged as a percentage."
        ),
    )
    parser.add_argument(
        "--merge_strategy",
        type=str,
        default="average",
        choices=["average"],
        help="Strategy for combining specialist weights (currently only 'average')",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="multi_experiment",
        help="Subdirectory name under --logdir for this experiment",
    )
    parser.add_argument(
        "--run_baseline",
        dest="run_baseline",
        action="store_true",
        default=True,
        help="Also train a random-init baseline on the full data for comparison",
    )
    parser.add_argument(
        "--no_baseline",
        dest="run_baseline",
        action="store_false",
        help="Skip the baseline training run",
    )
    parser.add_argument(
        "--equal_compute",
        action="store_true",
        default=False,
        help=(
            "Auto-set specialist_steps = final_steps // n_models so the total "
            "number of specialist gradient steps equals one baseline run. "
            "This makes the specialist phase compute-fair vs. the baseline."
        ),
    )
    parser.add_argument(
        "--overfit_stop",
        action="store_true",
        default=False,
        help=(
            "Enable early stopping for specialists: stop each specialist once "
            "full_train_acc >= overfit_threshold. Disabled by default so "
            "specialists can use their full compute budget and potentially grok."
        ),
    )
    parser.add_argument(
        "--distill_steps",
        type=int,
        default=50000,
        help="Number of distillation training steps",
    )
    parser.add_argument(
        "--distill_temperature",
        type=float,
        default=2.0,
        help="Temperature for knowledge distillation",
    )
    parser.add_argument(
        "--distill_alpha",
        type=float,
        default=0.5,
        help="Weight for soft labels in distillation loss (1-alpha for hard labels)",
    )
    parser.add_argument(
        "--use_distillation",
        action="store_true",
        default=False,
        help="Run distillation experiment in addition to weight averaging",
    )

    return parser
