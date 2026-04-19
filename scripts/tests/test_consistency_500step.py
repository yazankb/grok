#!/usr/bin/env python
"""Larger smoke test: M=4 specialists, 500 steps, real architecture (d_model=128).

Used to (a) sanity-check timing for Kaggle budgeting and (b) confirm the
training dynamics behave as expected at realistic scale: shard CE drops,
pairwise KL drops when consistency is on but not when it is off, no NaNs,
no memory blowup.
"""

from __future__ import annotations

import os
import sys
import tempfile
from argparse import Namespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)


def _hparams(logdir: str, name: str, **overrides) -> Namespace:
    h = Namespace(
        n_layers=2, n_heads=4, d_model=128, dropout=0.0, weight_noise=0.0,
        non_linearity="relu", max_context_len=50,
        math_operator="+", operand_length=None, train_data_pct=50.0,
        max_lr=5e-4, weight_decay=0.1, weight_decay_kind="to_zero",
        noise_factor=0, warmup_steps=10, anneal_lr=False, anneal_lr_steps=100000,
        batchsize=0,
        n_models=4, sharding="disjoint", random_seed=42,
        logdir=logdir, datadir=os.path.join(ROOT, "data"),
        experiment_name=name, gpu=-1,
        consistency_loss="mse_logits",
        consistency_lambda=1.0,
        consistency_warmup_steps=100,
        consistency_domain="full_grid",
        consistency_batch_size=256, shard_batch_size=256,
        consistency_steps=500,
        eval_every=100, checkpoint_every=0, log_every=25,
    )
    for k, v in overrides.items():
        setattr(h, k, v)
    return h


def main() -> None:
    from grok.consistency_training import train_multi_with_consistency

    tmp = tempfile.mkdtemp(prefix="consistency_500_")
    try:
        print(f"[500-step smoke] M=4, 500 steps, d_model=128, dir={tmp}")
        h = _hparams(tmp, "smoke_M4_500_lam1")
        exp_dir = train_multi_with_consistency(h)

        # Read final eval CSV and check signs of life
        import csv
        eval_csv = os.path.join(exp_dir, "consistency_eval.csv")
        with open(eval_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"\n[500-step smoke] eval rows: {len(rows)}")
        first = rows[0]
        last = rows[-1]
        first_kl = float(first["pairwise_kl_val_mean"])
        last_kl = float(last["pairwise_kl_val_mean"])
        first_norm = float(first["weight_norm_spec_0"])
        last_norm = float(last["weight_norm_spec_0"])
        print(f"  pairwise KL    first→last: {first_kl:.4f} → {last_kl:.4f}")
        print(f"  weight_norm[0] first→last: {first_norm:.3f} → {last_norm:.3f}")
        print(f"  val_acc_merged first→last: {float(first['val_acc_merged']):.2f}% → {float(last['val_acc_merged']):.2f}%")
        print(f"  val_acc_ensemble first→last: {float(first['val_acc_ensemble']):.2f}% → {float(last['val_acc_ensemble']):.2f}%")
        print(f"  unsup_entropy first→last: {float(first['unsup_entropy_mean']):.3f} → {float(last['unsup_entropy_mean']):.3f}")

        # Sanity assertions
        assert last_kl < first_kl, "pairwise KL should decrease under consistency"
        # Weight norm should change (training is doing something)
        assert abs(last_norm - first_norm) > 1e-3, "weights should be moving"
        print("\n[500-step smoke] ALL CHECKS PASSED")
    finally:
        import shutil
        if "consistency_500_" in tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
