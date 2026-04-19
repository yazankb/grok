#!/usr/bin/env python
"""Kaggle-friendly sweep driver for the consistency-regularised experiment.

Runs a configurable list of cells (each cell = one full training run) and
writes a top-level ``sweep_summary.json`` indexing all per-run results. Each
cell creates its own subdirectory under ``--logdir/<sweep_name>/<cell_name>``.

Defaults match the pilot sweep in
``reports/consistency_regularized_grokking_plan.md`` §8: 4 cells covering
{lambda in {0, 1.0}} x {warmup in {0, 1000}} on disjoint shards with the
full-grid consistency domain. This is the "before committing the full sweep,
run this and look at it" set.

Example (Kaggle, T4 GPU):

    python scripts/run_consistency_sweep.py \
        --sweep_name pilot_v1 \
        --logdir /kaggle/working/consistency_runs \
        --consistency_steps 25000 \
        --gpu 0

Custom cells via JSON file::

    python scripts/run_consistency_sweep.py \
        --cells_json scripts/sweep_cells_lambda_ablation.json \
        --logdir /kaggle/working/consistency_runs

Each cell in the JSON list is a dict whose keys override the corresponding
hparam fields (for example ``{"name": "lam_10_warm_500", "consistency_lambda": 10.0,
"consistency_warmup_steps": 500}``).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from argparse import Namespace
from typing import Any, Dict, List, Optional


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # Sweep metadata
    p.add_argument("--sweep_name", type=str, default="pilot_v1",
                   help="Top-level subdirectory name for this sweep.")
    p.add_argument("--cells_json", type=str, default=None,
                   help="Optional path to a JSON list of cell-overrides "
                        "dicts. If unset, the default pilot cells (4) run.")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only run the first N cells in the list.")
    # Shared hparams (apply to every cell unless overridden in cells_json)
    p.add_argument("--logdir", type=str,
                   default=os.environ.get("GROK_LOGDIR", "consistency_runs"))
    p.add_argument("--datadir", type=str, default="data")
    p.add_argument("--gpu", type=int, default=-1)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--math_operator", type=str, default="+")
    p.add_argument("--train_data_pct", type=float, default=50.0)
    p.add_argument("--operand_length", type=int, default=None)
    p.add_argument("--n_models", type=int, default=4)
    p.add_argument("--sharding", type=str, default="disjoint",
                   choices=["disjoint", "bag"])
    p.add_argument("--max_lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--weight_decay_kind", type=str, default="to_zero")
    p.add_argument("--anneal_lr", action="store_true", default=False)
    p.add_argument("--anneal_lr_steps", type=int, default=100000)
    p.add_argument("--warmup_steps", type=int, default=10)
    # Architecture
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--non_linearity", type=str, default="relu")
    p.add_argument("--max_context_len", type=int, default=50)
    p.add_argument("--weight_noise", type=float, default=0.0)
    # Trainer
    p.add_argument("--consistency_steps", type=int, default=25000)
    p.add_argument("--shard_batch_size", type=int, default=256)
    p.add_argument("--consistency_batch_size", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--checkpoint_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=50)
    return p


# Pilot sweep: 4 runs that, together, distinguish all four §5 failure modes
# from a real positive result. Cheap enough to run end-to-end in a few hours.
DEFAULT_PILOT_CELLS: List[Dict[str, Any]] = [
    {
        "name": "lam0_baseline",
        "description": "No consistency. Multi-specialist baseline.",
        "consistency_loss": "none",
        "consistency_lambda": 0.0,
        "consistency_warmup_steps": 0,
        "consistency_domain": "full_grid",
    },
    {
        "name": "lam1_constant",
        "description": "Constant lambda=1.0, no warmup. Tests whether early "
                       "consistency is overwhelmed by hard CE (failure mode 2).",
        "consistency_loss": "mse_logits",
        "consistency_lambda": 1.0,
        "consistency_warmup_steps": 0,
        "consistency_domain": "full_grid",
    },
    {
        "name": "lam1_warmup1k",
        "description": "lambda=1.0 with 1000-step warmup. Headline cell.",
        "consistency_loss": "mse_logits",
        "consistency_lambda": 1.0,
        "consistency_warmup_steps": 1000,
        "consistency_domain": "full_grid",
    },
    {
        "name": "lam10_warmup1k",
        "description": "lambda=10.0 with 1000-step warmup. Probes whether a "
                       "stronger consistency pressure prevents memorization "
                       "and forces structured solutions earlier.",
        "consistency_loss": "mse_logits",
        "consistency_lambda": 10.0,
        "consistency_warmup_steps": 1000,
        "consistency_domain": "full_grid",
    },
]


def _build_cell_hparams(base: argparse.Namespace, cell: Dict[str, Any]) -> Namespace:
    """Compose the per-run Namespace from base CLI args + per-cell overrides."""
    h = Namespace(
        # Architecture (from base, may be overridden per-cell)
        n_layers=base.n_layers,
        n_heads=base.n_heads,
        d_model=base.d_model,
        dropout=base.dropout,
        weight_noise=base.weight_noise,
        non_linearity=base.non_linearity,
        max_context_len=base.max_context_len,
        # Data
        math_operator=base.math_operator,
        operand_length=base.operand_length,
        train_data_pct=base.train_data_pct,
        # Optim
        max_lr=base.max_lr,
        weight_decay=base.weight_decay,
        weight_decay_kind=base.weight_decay_kind,
        noise_factor=0,
        warmup_steps=base.warmup_steps,
        anneal_lr=base.anneal_lr,
        anneal_lr_steps=base.anneal_lr_steps,
        batchsize=0,
        # Multi
        n_models=base.n_models,
        sharding=base.sharding,
        random_seed=base.random_seed,
        # Paths (per-cell experiment_name set below)
        logdir=os.path.abspath(os.path.join(base.logdir, base.sweep_name)),
        datadir=os.path.abspath(base.datadir),
        gpu=base.gpu,
        # Defaults for consistency flags (cell overrides take precedence)
        consistency_loss="mse_logits",
        consistency_lambda=0.0,
        consistency_warmup_steps=0,
        consistency_domain="full_grid",
        consistency_batch_size=base.consistency_batch_size,
        shard_batch_size=base.shard_batch_size,
        consistency_steps=base.consistency_steps,
        eval_every=base.eval_every,
        checkpoint_every=base.checkpoint_every,
        log_every=base.log_every,
    )
    name = cell.get("name")
    if not name:
        raise ValueError(f"Cell missing 'name': {cell!r}")
    h.experiment_name = name
    for k, v in cell.items():
        if k in ("name", "description"):
            continue
        setattr(h, k, v)
    return h


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    # Make repo importable when run from arbitrary cwd (e.g. /kaggle/working)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from grok.consistency_training import train_multi_with_consistency  # noqa: E402

    # Resolve cells
    if args.cells_json:
        with open(args.cells_json, "r") as f:
            cells = json.load(f)
        if not isinstance(cells, list):
            raise ValueError("cells_json must contain a JSON list of dicts.")
    else:
        cells = DEFAULT_PILOT_CELLS

    if args.limit is not None:
        cells = cells[: args.limit]

    sweep_dir = os.path.abspath(os.path.join(args.logdir, args.sweep_name))
    os.makedirs(sweep_dir, exist_ok=True)
    sweep_summary_path = os.path.join(sweep_dir, "sweep_summary.json")

    summary: Dict[str, Any] = {
        "sweep_name": args.sweep_name,
        "sweep_dir": sweep_dir,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shared_args": vars(args),
        "cells": [],
    }
    with open(sweep_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#'*70}")
    print(f"# Sweep: {args.sweep_name}  ({len(cells)} cell(s))")
    print(f"# Output dir: {sweep_dir}")
    print(f"# Summary file: {sweep_summary_path}")
    print(f"{'#'*70}\n")

    for idx, cell in enumerate(cells):
        print(f"\n{'#'*70}")
        print(f"# [{idx+1}/{len(cells)}] cell: {cell.get('name')}")
        if cell.get("description"):
            print(f"# {cell['description']}")
        print(f"{'#'*70}\n")
        cell_hparams = _build_cell_hparams(args, cell)
        cell_record: Dict[str, Any] = {
            "index": idx,
            "name": cell.get("name"),
            "description": cell.get("description"),
            "overrides": {k: v for k, v in cell.items()
                          if k not in ("name", "description")},
            "experiment_dir": os.path.join(sweep_dir, cell["name"]),
        }
        t0 = time.time()
        try:
            exp_dir = train_multi_with_consistency(cell_hparams)
            cell_record["experiment_dir"] = os.path.abspath(exp_dir)
            results_path = os.path.join(exp_dir, "consistency_results.json")
            if os.path.isfile(results_path):
                with open(results_path, "r") as fh:
                    cell_record["results"] = json.load(fh)
            cell_record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            cell_record["status"] = "failed"
            cell_record["error"] = str(exc)
            cell_record["traceback"] = traceback.format_exc()
            print(f"  [sweep] cell FAILED: {exc}")
        cell_record["elapsed_sec"] = time.time() - t0
        summary["cells"].append(cell_record)
        # Persist after every cell so partial progress survives crashes
        with open(sweep_summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(sweep_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#'*70}")
    print(f"# Sweep complete. Summary: {sweep_summary_path}")
    print(f"{'#'*70}\n")
    # Friendly table
    print(f"{'cell':<25}{'status':<10}{'best_ens':>10}{'best_merged':>14}{'elapsed_s':>12}")
    print("-" * 71)
    for c in summary["cells"]:
        ens = c.get("results", {}).get("best_val_acc_ensemble", float("nan"))
        mrg = c.get("results", {}).get("best_val_acc_merged", float("nan"))
        elapsed = c.get("elapsed_sec", float("nan"))
        try:
            ens_s = f"{ens:.2f}"
        except (TypeError, ValueError):
            ens_s = str(ens)
        try:
            mrg_s = f"{mrg:.2f}"
        except (TypeError, ValueError):
            mrg_s = str(mrg)
        try:
            el_s = f"{elapsed:.1f}"
        except (TypeError, ValueError):
            el_s = str(elapsed)
        print(f"{c['name']:<25}{c['status']:<10}{ens_s:>10}{mrg_s:>14}{el_s:>12}")


if __name__ == "__main__":
    main()
