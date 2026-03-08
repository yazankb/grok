#!/usr/bin/env python
"""
Run only merged and baseline phases from an existing multi-model experiment.

This is useful when specialists are already trained and you want to quickly cap
merged/baseline runs to a smaller step budget.
"""

import copy
import os
from argparse import ArgumentParser

import torch

from grok.data import ArithmeticDataset
from grok.multi_training import _build_trainer
from grok.training import TrainableTransformer, add_args


def main() -> None:
    parser = add_args(ArgumentParser())
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="multi_fixed",
        help="Experiment directory under --logdir containing merged_weights.pt",
    )
    parser.add_argument(
        "--merged_weights_path",
        type=str,
        default="",
        help="Optional explicit path to merged_weights.pt",
    )
    parser.add_argument(
        "--final_steps",
        type=int,
        default=5000,
        help="Step budget for merged and baseline runs",
    )
    parser.add_argument(
        "--run_baseline",
        action="store_true",
        default=True,
        help="Run baseline model in addition to merged model",
    )
    parser.add_argument(
        "--no_baseline",
        dest="run_baseline",
        action="store_false",
        help="Skip baseline model run",
    )
    args = parser.parse_args()

    experiment_dir = os.path.join(args.logdir, args.experiment_name)
    merged_weights_path = (
        args.merged_weights_path
        if args.merged_weights_path
        else os.path.join(experiment_dir, "merged_weights.pt")
    )
    if not os.path.exists(merged_weights_path):
        raise FileNotFoundError(
            f"Could not find merged weights: {merged_weights_path}"
        )

    full_train_ds, val_ds = ArithmeticDataset.splits(
        train_pct=args.train_data_pct,
        operator=args.math_operator,
        operand_length=args.operand_length,
        data_dir=args.datadir,
    )

    print(f"Using merged weights: {merged_weights_path}")
    print(
        f"Train set: {len(full_train_ds)} equations | "
        f"Val set: {len(val_ds)} equations"
    )
    print(f"final_steps={args.final_steps}")

    # Merged model run
    merged_hparams = copy.deepcopy(args)
    merged_hparams.logdir = os.path.join(experiment_dir, "merged_5k")
    ckpt_dir = os.path.join(merged_hparams.logdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    merged_hparams.checkpoint_path = ckpt_dir

    merged_model = TrainableTransformer(
        merged_hparams,
        train_dataset=full_train_ds,
        val_dataset=val_ds,
    ).float()
    merged_weights = torch.load(merged_weights_path, map_location="cpu")
    merged_model.transformer.load_state_dict(merged_weights)
    torch.save(merged_model, os.path.join(ckpt_dir, "init.pt"))

    trainer = _build_trainer(
        args,
        merged_hparams.logdir,
        max_steps=args.final_steps,
        min_steps=args.final_steps,
    )
    trainer.fit(merged_model)

    merged_final_val_acc = float(
        trainer.callback_metrics.get("val_accuracy", float("nan"))
    )
    print(f"Merged final val_acc={merged_final_val_acc:.4f}")

    # Baseline run
    if args.run_baseline:
        baseline_hparams = copy.deepcopy(args)
        baseline_hparams.logdir = os.path.join(experiment_dir, "baseline_5k")
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
            args,
            baseline_hparams.logdir,
            max_steps=args.final_steps,
            min_steps=args.final_steps,
        )
        trainer.fit(baseline_model)

        baseline_final_val_acc = float(
            trainer.callback_metrics.get("val_accuracy", float("nan"))
        )
        print(f"Baseline final val_acc={baseline_final_val_acc:.4f}")

    print(f"Done. Outputs under: {experiment_dir}")


if __name__ == "__main__":
    main()
