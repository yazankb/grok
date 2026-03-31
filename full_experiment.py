#!/usr/bin/env python
"""
Full experiment with distillation and weight averaging for grokking.
Combines all phases: specialists, weight averaging, distillation, and baseline.
"""

import os
import sys
import copy
import json
import argparse
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl

from grok.data import ArithmeticDataset
from grok.training import TrainableTransformer, add_args
from grok.transformer import Transformer


def add_multi_args(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    parser = add_args(parser)
    parser.add_argument("--n_models", type=int, default=4, help="Number of specialist models")
    parser.add_argument("--specialist_steps", type=int, default=50000, help="Steps per specialist")
    parser.add_argument("--final_steps", type=int, default=100000, help="Steps for merged/baseline")
    parser.add_argument("--distill_steps", type=int, default=50000, help="Distillation steps")
    parser.add_argument("--distill_temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--distill_alpha", type=float, default=0.5, help="Distillation alpha")
    parser.add_argument("--experiment_name", type=str, default="full_experiment", help="Experiment name")
    parser.add_argument("--run_baseline", dest="run_baseline", action="store_true", default=True)
    parser.add_argument("--no_baseline", dest="run_baseline", action="store_false")
    return parser


class EarlyStopOnOverfit(Callback):
    def __init__(self, threshold: float = 99.0) -> None:
        super().__init__()
        self.threshold = threshold

    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:
        acc = trainer.callback_metrics.get("full_train_acc", None)
        if acc is None:
            acc = trainer.callback_metrics.get("train_accuracy", None)
        if acc is not None and float(acc) >= self.threshold:
            trainer.should_stop = True


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    ref = state_dicts[0]
    avg: Dict[str, torch.Tensor] = {}
    for key in ref:
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        avg[key] = stacked.mean(dim=0).to(ref[key].dtype)
    return avg


class DistillationTrainer:
    def __init__(self, teacher_models: List[TrainableTransformer], student_model: TrainableTransformer,
                 temperature: float = 2.0, alpha: float = 0.5, device: str = "cuda"):
        self.teachers = [t.eval() for t in teacher_models]
        self.student = student_model.train()
        self.temperature = temperature
        self.alpha = alpha
        self.device = device

    @torch.no_grad()
    def get_teacher_logits(self, batch: Dict) -> torch.Tensor:
        all_logits = []
        for teacher in self.teachers:
            logits, _, _ = teacher(batch["text"].to(self.device))
            all_logits.append(logits.to(self.device))
        return torch.stack(all_logits).mean(dim=0)

    def distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        teacher_logits = teacher_logits.to(student_logits.device)
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        kl_div = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        return kl_div * (T * T)

    def hard_loss(self, student_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)),
                               targets.reshape(-1).to(self.device))

    def train_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        text = batch["text"].to(self.device)
        targets = batch["target"].to(self.device)
        student_logits, _, _ = self.student(text)
        teacher_logits = self.get_teacher_logits({"text": text})
        soft_loss = self.distillation_loss(student_logits, teacher_logits)
        hard_loss = self.hard_loss(student_logits, targets)
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        with torch.no_grad():
            student_pred = student_logits.argmax(dim=-1)
            accuracy = (student_pred == targets).float().mean() * 100
            teacher_pred = teacher_logits.argmax(dim=-1)
            teacher_acc = (teacher_pred == targets).float().mean() * 100
        metrics = {"loss": loss.item(), "soft_loss": soft_loss.item(), "hard_loss": hard_loss.item(),
                   "student_acc": accuracy.item(), "teacher_acc": teacher_acc.item()}
        return loss, metrics


def distill_from_specialists(specialist_models: List[TrainableTransformer], student_model: TrainableTransformer,
                              train_dataset, val_dataset, distill_steps: int = 10000,
                              temperature: float = 2.0, alpha: float = 0.5, device: str = "cuda"):
    print(f"\n{'='*60}")
    print(f"Distillation: T={temperature}, alpha={alpha}")
    print(f"{'='*60}")
    distill_trainer = DistillationTrainer(specialist_models, student_model, temperature, alpha, device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3, weight_decay=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=distill_steps, eta_min=1e-5)
    metrics = {"loss": [], "soft_loss": [], "hard_loss": [], "student_acc": [], "teacher_acc": [],
               "val_acc": [], "steps": []}
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
                print(f"  Step {step}/{distill_steps}: train_acc={m['student_acc']:.2f}%, val_acc={val_acc:.2f}%")
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


def _build_trainer(hparams: Namespace, log_dir: str, max_steps: int, min_steps: int = 1,
                   extra_callbacks: Optional[List[Callback]] = None, check_val_every_n_epoch: int = 25) -> Trainer:
    logger = CSVLogger(log_dir)
    trainer_args = {"max_steps": max_steps, "min_steps": min_steps, "max_epochs": int(1e8),
                    "check_val_every_n_epoch": check_val_every_n_epoch, "logger": logger,
                    "log_every_n_steps": 1, "callbacks": extra_callbacks or []}
    if torch.cuda.is_available() and hparams.gpu >= 0:
        trainer_args["accelerator"] = "gpu"
        trainer_args["devices"] = [hparams.gpu]
    else:
        trainer_args["accelerator"] = "cpu"
    return Trainer(**trainer_args)


def run_full_experiment(hparams: Namespace) -> str:
    import numpy as np
    n_models = hparams.n_models
    final_steps = hparams.final_steps
    experiment_dir = os.path.join(hparams.logdir, hparams.experiment_name)
    seed = hparams.random_seed if hparams.random_seed != -1 else 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Full experiment: {n_models} specialists | {hparams.math_operator} | {hparams.train_data_pct}% data")
    print(f"{'='*60}")

    full_train_ds, val_ds = ArithmeticDataset.splits(
        train_pct=hparams.train_data_pct, operator=hparams.math_operator,
        operand_length=hparams.operand_length, data_dir=hparams.datadir)
    print(f"Full training set: {len(full_train_ds)} equations | Validation: {len(val_ds)} equations")

    shards = ArithmeticDataset.split_n_ways(full_train_ds, n_models, seed=seed)
    for i, shard in enumerate(shards):
        print(f"  Shard {i}: {len(shard)} equations")

    specialist_models: List[TrainableTransformer] = []
    specialist_steps = hparams.specialist_steps

    for i, shard in enumerate(shards):
        print(f"\n{'='*60}")
        print(f"Phase 1 — Specialist {i + 1}/{n_models}")
        print(f"{'='*60}")
        print(f"  Starting specialist {i} training for {specialist_steps} steps...")
        spec_hparams = copy.deepcopy(hparams)
        spec_hparams.logdir = os.path.join(experiment_dir, f"specialist_{i}")
        ckpt_dir = os.path.join(spec_hparams.logdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        spec_hparams.checkpoint_path = ckpt_dir
        model = TrainableTransformer(spec_hparams, train_dataset=shard, val_dataset=val_ds).float().to(device)
        trainer = _build_trainer(hparams, spec_hparams.logdir, max_steps=specialist_steps, min_steps=1)
        trainer.fit(model)
        specialist_models.append(model)
        print(f"  Finished specialist {i} training")

    print(f"\n{'='*60}")
    print("Phase 2 — Weight Averaging")
    print(f"{'='*60}")
    specialist_state_dicts = [copy.deepcopy(m.transformer.state_dict()) for m in specialist_models]
    merged_weights = average_state_dicts(specialist_state_dicts)

    merged_hparams = copy.deepcopy(hparams)
    merged_hparams.logdir = os.path.join(experiment_dir, "merged_average")
    merged_hparams.checkpoint_path = os.path.join(merged_hparams.logdir, "checkpoints")
    merged_model = TrainableTransformer(merged_hparams, train_dataset=full_train_ds, val_dataset=val_ds).float().to(device)
    merged_model.transformer.load_state_dict(merged_weights)

    print(f"\n{'='*60}")
    print("Phase 3 — Fine-tuning Weight-Averaged Model")
    print(f"{'='*60}")
    trainer = _build_trainer(hparams, merged_hparams.logdir, max_steps=final_steps, min_steps=final_steps)
    trainer.fit(merged_model)

    print(f"\n{'='*60}")
    print("Phase 4 — Knowledge Distillation")
    print(f"{'='*60}")
    distill_hparams = copy.deepcopy(hparams)
    distill_hparams.logdir = os.path.join(experiment_dir, "distilled")
    distill_model = TrainableTransformer(distill_hparams, train_dataset=full_train_ds, val_dataset=val_ds).float().to(device)
    distilled_model, distill_metrics = distill_from_specialists(
        specialist_models, distill_model, full_train_ds, val_ds,
        hparams.distill_steps, hparams.distill_temperature, hparams.distill_alpha, device)

    if hparams.run_baseline:
        print(f"\n{'='*60}")
        print("Phase 5 — Baseline: random-init model")
        print(f"{'='*60}")
        baseline_hparams = copy.deepcopy(hparams)
        baseline_hparams.logdir = os.path.join(experiment_dir, "baseline")
        baseline_hparams.checkpoint_path = os.path.join(baseline_hparams.logdir, "checkpoints")
        baseline_model = TrainableTransformer(baseline_hparams, train_dataset=full_train_ds, val_dataset=val_ds).float().to(device)
        trainer = _build_trainer(hparams, baseline_hparams.logdir, max_steps=final_steps, min_steps=final_steps)
        trainer.fit(baseline_model)

    results_path = os.path.join(experiment_dir, "comparison_results.json")
    results = {"n_models": n_models, "specialist_steps": specialist_steps, "final_steps": final_steps,
               "distill_steps": hparams.distill_steps, "temperature": hparams.distill_temperature,
               "alpha": hparams.distill_alpha,
               "distill_final_metrics": {"final_loss": float(np.mean(distill_metrics["loss"][-100:])),
                                         "final_student_acc": float(np.mean(distill_metrics["student_acc"][-100:]))},
               "experiment_dir": experiment_dir}
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment complete! Results: {results_path}")
    print(f"{'='*60}")
    return experiment_dir


if __name__ == "__main__":
    parser = add_multi_args()
    parser.set_defaults(logdir=os.path.join(os.getcwd(), "logs"), datadir=os.path.join(os.getcwd(), "data"),
                        math_operator="/", train_data_pct=50, n_models=4, specialist_steps=50000,
                        final_steps=100000, distill_steps=50000, distill_temperature=2.0, distill_alpha=0.5,
                        experiment_name="full_experiment", random_seed=42, gpu=0)
    hparams = parser.parse_args()
    hparams.datadir = os.path.normpath(hparams.datadir)
    hparams.logdir = os.path.normpath(hparams.logdir)
    print(hparams)
    run_full_experiment(hparams)