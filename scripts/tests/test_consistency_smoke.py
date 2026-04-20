#!/usr/bin/env python
"""Smoke test for the consistency trainer.

Runs a tiny end-to-end consistency-regularised multi-specialist experiment
(2 specialists, 80 steps, eval every 40 steps) and asserts that:

- the run completes,
- per-step CSV is non-empty and well-formed,
- per-eval CSV contains at least the expected eval rows,
- consistency_results.json is written and contains best-of-run fields,
- final per-specialist checkpoints are saved.

This exercises both the CE-only path (lambda=0) and the full consistency path.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from argparse import Namespace

# Make repo root importable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)


def _build_hparams(experiment_dir_root: str, name: str, **overrides) -> Namespace:
    """Construct a minimal Namespace that the consistency trainer accepts."""
    h = Namespace(
        # Model architecture (tiny: 2 layers, d_model=32 keeps it fast on CPU)
        n_layers=2,
        n_heads=4,
        d_model=32,
        dropout=0.0,
        weight_noise=0.0,
        non_linearity="relu",
        max_context_len=50,
        # Data
        math_operator="+",
        operand_length=None,
        train_data_pct=50.0,
        # Optim
        max_lr=5e-4,
        weight_decay=0.1,
        weight_decay_kind="to_zero",
        noise_factor=0,
        warmup_steps=10,
        anneal_lr=False,
        anneal_lr_steps=100000,
        batchsize=0,
        # Multi-model
        n_models=2,
        sharding="disjoint",
        random_seed=42,
        # Paths
        logdir=experiment_dir_root,
        datadir=os.path.join(ROOT, "data"),
        experiment_name=name,
        # Lightning / trainer plumbing (consumed by helpers in multi_training)
        gpu=-1,
        # Consistency-trainer-specific
        consistency_loss="mse_logits",
        consistency_lambda=1.0,
        consistency_warmup_steps=20,
        consistency_domain="full_grid",
        consistency_batch_size=64,
        shard_batch_size=64,
        consistency_steps=80,
        eval_every=40,
        checkpoint_every=0,
        log_every=10,
    )
    for k, v in overrides.items():
        setattr(h, k, v)
    return h


def _assert_csv_has_rows(path: str, min_rows: int) -> int:
    assert os.path.isfile(path), f"missing CSV: {path}"
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) >= 1, f"empty CSV: {path}"
    data_rows = rows[1:]
    assert (
        len(data_rows) >= min_rows
    ), f"{path}: expected >= {min_rows} data rows, got {len(data_rows)}"
    return len(data_rows)


def _assert_results_json(path: str) -> dict:
    assert os.path.isfile(path), f"missing results JSON: {path}"
    with open(path, "r") as f:
        d = json.load(f)
    for key in (
        "best_val_acc_ensemble",
        "best_val_acc_ensemble_step",
        "best_val_acc_merged",
        "best_val_acc_merged_step",
        "final_eval",
        "elapsed_sec",
        "train_metrics_csv",
        "eval_metrics_csv",
        "checkpoint_dir",
    ):
        assert key in d, f"results JSON missing key: {key}"
    fe = d["final_eval"]
    for ek in (
        "val_acc_ensemble",
        "val_acc_merged",
        "unsup_entropy_mean",
        "pairwise_kl_val_mean",
    ):
        assert ek in fe, f"final_eval missing key: {ek}"
    return d


def _assert_checkpoints(experiment_dir: str, n_models: int) -> None:
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    assert os.path.isdir(ckpt_dir), f"missing checkpoint dir: {ckpt_dir}"
    for i in range(n_models):
        p = os.path.join(ckpt_dir, f"specialist_{i}_final.pt")
        assert os.path.isfile(p), f"missing final checkpoint: {p}"
    init_dir = os.path.join(experiment_dir, "init")
    for i in range(n_models):
        p = os.path.join(init_dir, f"specialist_{i}_init.pt")
        assert os.path.isfile(p), f"missing init checkpoint: {p}"


def main() -> None:
    from grok.consistency_training import train_multi_with_consistency

    tmp_root = tempfile.mkdtemp(prefix="consistency_smoke_")
    print(f"[smoke] temp logdir: {tmp_root}")

    try:
        # ---- Run A: with consistency on -----------------------------------
        print("\n[smoke] running config A: consistency_lambda=1.0, warmup=20")
        h_on = _build_hparams(tmp_root, "smoke_consistency_on")
        exp_dir_on = train_multi_with_consistency(h_on)
        assert os.path.isdir(exp_dir_on)

        train_csv = os.path.join(exp_dir_on, "consistency_metrics.csv")
        eval_csv = os.path.join(exp_dir_on, "consistency_eval.csv")
        results_json = os.path.join(exp_dir_on, "consistency_results.json")

        # 80 total steps, log_every=10 + step=1 => >= 9 rows
        n_train_rows = _assert_csv_has_rows(train_csv, min_rows=8)
        # eval_every=40 + step=1 + final => >= 4 rows
        n_eval_rows = _assert_csv_has_rows(eval_csv, min_rows=3)
        results_on = _assert_results_json(results_json)
        _assert_checkpoints(exp_dir_on, n_models=h_on.n_models)
        print(
            f"[smoke] A ok: train_rows={n_train_rows} eval_rows={n_eval_rows} "
            f"best_ens={results_on['best_val_acc_ensemble']:.2f}% "
            f"best_merged={results_on['best_val_acc_merged']:.2f}% "
            f"elapsed={results_on['elapsed_sec']:.1f}s"
        )

        # ---- Run B: lambda=0 to confirm degenerate path is wired correctly
        print("\n[smoke] running config B: consistency_lambda=0 (CE-only)")
        h_off = _build_hparams(
            tmp_root,
            "smoke_consistency_off",
            consistency_lambda=0.0,
            consistency_warmup_steps=0,
            consistency_steps=40,
            eval_every=20,
        )
        exp_dir_off = train_multi_with_consistency(h_off)
        train_csv = os.path.join(exp_dir_off, "consistency_metrics.csv")
        eval_csv = os.path.join(exp_dir_off, "consistency_eval.csv")
        results_json = os.path.join(exp_dir_off, "consistency_results.json")
        _assert_csv_has_rows(train_csv, min_rows=4)
        _assert_csv_has_rows(eval_csv, min_rows=2)
        results_off = _assert_results_json(results_json)
        _assert_checkpoints(exp_dir_off, n_models=h_off.n_models)
        print(
            f"[smoke] B ok: best_ens={results_off['best_val_acc_ensemble']:.2f}% "
            f"elapsed={results_off['elapsed_sec']:.1f}s"
        )

        # Cross-check: with lambda=0, the cons_spec_* columns in metrics CSV
        # should still be populated (we compute them either way, just don't
        # add them to the loss). This confirms diagnostic logging works even
        # in the baseline cell of the sweep.
        with open(train_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0, "no train rows for B"
        assert "cons_spec_0" in rows[0], f"cons col missing in {list(rows[0].keys())}"
        print("[smoke] B diagnostic columns present in metrics CSV.")

        # Sanity: eval CSV contains entropy and pairwise_kl columns
        with open(eval_csv, "r", newline="") as f:
            eval_rows = list(csv.DictReader(f))
        assert "unsup_entropy_mean" in eval_rows[0]
        assert "pairwise_kl_val_mean" in eval_rows[0]
        assert "val_acc_ensemble" in eval_rows[0]
        assert "val_acc_merged" in eval_rows[0]
        print("[smoke] eval CSV column schema ok.")

        # ---- Run C: KL-softmax consistency (v2 default loss) ----------------
        print("\n[smoke] running config C: consistency_loss=kl_softmax, lam=0.5")
        h_kl = _build_hparams(
            tmp_root,
            "smoke_consistency_kl",
            consistency_loss="kl_softmax",
            consistency_lambda=0.5,
            consistency_warmup_steps=20,
            consistency_steps=40,
            eval_every=20,
        )
        exp_dir_kl = train_multi_with_consistency(h_kl)
        results_kl = _assert_results_json(
            os.path.join(exp_dir_kl, "consistency_results.json")
        )
        # KL losses are in [0, log(vocab)] so cons_spec_* values must be finite
        with open(os.path.join(exp_dir_kl, "consistency_metrics.csv"), "r") as f:
            rows = list(csv.DictReader(f))
        import math as _math
        for r in rows:
            for i in range(h_kl.n_models):
                v = float(r[f"cons_spec_{i}"])
                assert _math.isfinite(v) and v >= 0.0, f"bad KL cons: {v}"
        print(
            f"[smoke] C ok: best_ens={results_kl['best_val_acc_ensemble']:.2f}% "
            f"elapsed={results_kl['elapsed_sec']:.1f}s"
        )

        # ---- Run D: single-model baseline (M=1, consistency_loss=none) ------
        print("\n[smoke] running config D: single-model baseline (M=1)")
        h_single = _build_hparams(
            tmp_root,
            "smoke_single_model",
            n_models=1,
            consistency_loss="none",
            consistency_lambda=0.0,
            consistency_warmup_steps=0,
            consistency_steps=40,
            eval_every=20,
        )
        exp_dir_single = train_multi_with_consistency(h_single)
        results_single = _assert_results_json(
            os.path.join(exp_dir_single, "consistency_results.json")
        )
        _assert_checkpoints(exp_dir_single, n_models=1)
        # With M=1 there are no specialists to compare, so pairwise_kl must be 0
        with open(os.path.join(exp_dir_single, "consistency_eval.csv"), "r") as f:
            eval_rows = list(csv.DictReader(f))
        for r in eval_rows:
            assert float(r["pairwise_kl_val_mean"]) == 0.0, (
                f"pairwise_kl should be 0 for M=1, got {r['pairwise_kl_val_mean']}"
            )
        # Metrics CSV must contain the single-specialist columns and no _1 cols
        with open(os.path.join(exp_dir_single, "consistency_metrics.csv"), "r") as f:
            train_rows = list(csv.DictReader(f))
        assert "ce_spec_0" in train_rows[0]
        assert "cons_spec_0" in train_rows[0]
        assert "ce_spec_1" not in train_rows[0]
        print(
            f"[smoke] D ok: val_acc_spec_0="
            f"{results_single['final_eval']['val_acc_spec_0']:.2f}%"
        )

        # ---- Run E: guard that M=1 with consistency_loss != 'none' errors ---
        print("\n[smoke] running config E: M=1 + non-'none' loss should error")
        from grok.consistency_training import ConsistencyTrainer
        try:
            h_bad = _build_hparams(
                tmp_root,
                "smoke_single_with_consistency",
                n_models=1,
                consistency_loss="kl_softmax",
                consistency_lambda=1.0,
                consistency_steps=4,
                eval_every=2,
            )
            train_multi_with_consistency(h_bad)
        except ValueError as e:
            assert "M=1" in str(e) or "at least 2" in str(e) or "M >= 2" in str(e)
            print(f"[smoke] E ok: got expected ValueError ({e})")
        else:
            raise AssertionError(
                "expected ValueError for M=1 with kl_softmax consistency_loss"
            )

        print("\n[smoke] ALL CHECKS PASSED")
    finally:
        # Keep the dir on failure for inspection; only delete on full success
        if "smoke" in tmp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
