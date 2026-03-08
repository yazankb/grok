#!/usr/bin/env python
"""
Multi-model grokking experiment.

Three-phase pipeline:
  1. Split the training data into N disjoint shards and train one specialist
     transformer per shard until it overfits (train_accuracy >= threshold).
  2. Average the specialist weight tensors into a single merged model.
  3. Fine-tune the merged model on the *full* training data and track grokking.
     Optionally, also run a random-init baseline on the full data for comparison.

Usage (from repo root):
    python scripts/train_multi.py --n_models 4 --train_data_pct 50 \
        --specialist_steps 50000 --final_steps 100000
"""

import copy
import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional

import torch
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

    return parser
