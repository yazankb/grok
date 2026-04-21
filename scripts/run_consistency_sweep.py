#!/usr/bin/env python
"""Kaggle-friendly sweep driver for the consistency-regularised experiment.

Runs a configurable list of cells (each cell = one full training run) and
writes a top-level ``sweep_summary.json`` indexing all per-run results. Each
cell creates its own subdirectory under ``--logdir/<sweep_name>/<cell_name>``.

Default pilot is v4 (see ``reports/consistency_regularized_grokking_plan.md``
§8 revision). Based on the v2 and v3 results:

- Single-model baseline on the 50% slice is architecturally capped at ~12%
  val acc (confirmed at 100k steps, not budget-limited).
- ``train_inputs_only`` KL-softmax consistency with 5k warmup is robust:
  4/4 v3 seeds gave 21-45% ensemble val acc at 25k (mean 36%), vs 12%
  single-model and 1.4% no-consistency.
- The v3 long multi cell (seed 42, lam=0.1, 100k steps) kept climbing
  throughout: 21% (25k) -> 31% (50k) -> 34.5% (75k) -> 38% (100k).
- At seed 42 and 25k steps, ensemble val acc is monotonic in lambda over
  [0.03, 1.0]: 16% / 21% / 26% / 27% for lam in {0.03, 0.1, 0.3, 1.0}.

v4 is our final sweep. The open question is whether the
best-seed x best-lambda x full-budget configuration can cross the
grokking phase transition (~100% val acc), or whether it asymptotes
below. We also replicate at a second seed to confirm robustness.

    [3 cells, all at 100k steps, all at ``train_inputs_only``]
    1. ``seed44_lam1_100k`` - best-known seed x largest-known-good lambda
       x full budget. Primary bet on grokking.
    2. ``seed43_lam1_100k`` - cross-seed replication of (1). Tests
       whether any grokking we see is seed-44-specific.
    3. ``seed44_lam01_100k`` - lambda ablation at the winning seed.
       Directly compares against (1) to quantify the lambda contribution
       at the 100k budget (v3's lambda sweep was only at 25k, seed 42).

Example (Kaggle, T4 GPU)::

    python scripts/run_consistency_sweep.py \
        --sweep_name pilot_v4 \
        --logdir /kaggle/working/consistency_runs \
        --consistency_steps 25000 \
        --gpu 0

Each cell sets its own ``consistency_steps`` (100k in v4) so the CLI
``--consistency_steps`` value only applies to cells that don't override.

Custom cells via JSON file::

    python scripts/run_consistency_sweep.py \
        --cells_json scripts/sweep_cells_custom.json \
        --logdir /kaggle/working/consistency_runs

Each cell in the JSON list is a dict whose keys override the corresponding
hparam fields. A cell with ``"kind": "single_model"`` runs a single model on
the current ``--train_data_pct`` slice; any other ``"kind"`` (or missing key)
runs the multi-specialist consistency trainer. Example::

    {"name": "lam_10_warm_500", "consistency_loss": "kl_softmax",
     "consistency_lambda": 10.0, "consistency_warmup_steps": 500}
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
    p.add_argument("--sweep_name", type=str, default="pilot_v4",
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
    p.add_argument("--infonce_temperature", type=float, default=0.1,
                   help="Temperature for InfoNCE contrastive loss.")
    p.add_argument("--shard_batch_size", type=int, default=256)
    p.add_argument("--consistency_batch_size", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--checkpoint_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=50)
    return p


# Pilot sweep v4: 3 runs, final experiment. All cells use 100k steps and
# the train_inputs_only KL-softmax consistency domain with 5k warmup. The
# cells differ only in (random_seed, lambda). See the module docstring
# for the rationale and the v1-v3 history in the plan §8 revision.
_V4_BASE: Dict[str, Any] = {
    "consistency_loss": "kl_softmax",
    "consistency_warmup_steps": 5000,
    "consistency_domain": "train_inputs_only",
    "consistency_steps": 100000,
}

DEFAULT_PILOT_CELLS: List[Dict[str, Any]] = [
    {
        "name": "seed44_lam1_100k",
        "description": "Primary bet. Best-known seed from v3 (seed 44, "
                       "which gave 45% ensemble val acc at 25k with "
                       "lam=0.1) combined with the largest-known-good "
                       "lambda (lam=1.0, monotonic winner of v3's lambda "
                       "ablation) at the full 100k budget. If this cell "
                       "does not cross the grokking phase transition "
                       "(~100% val acc), nothing at this scale will.",
        "random_seed": 44,
        "consistency_lambda": 1.0,
        **_V4_BASE,
    },
    {
        "name": "seed43_lam1_100k",
        "description": "Cross-seed replication of cell 1. Seed 43 gave "
                       "39% ensemble val acc at 25k with lam=0.1 (second "
                       "best in v3). If cells 1 and 2 both grok, the "
                       "effect is robust across seeds at the best lambda.",
        "random_seed": 43,
        "consistency_lambda": 1.0,
        **_V4_BASE,
    },
    {
        "name": "seed44_lam01_100k",
        "description": "Lambda ablation at the winning seed. Compare "
                       "against cell 1 (seed44_lam1_100k) to quantify "
                       "how much of any v4 gain comes from lambda vs "
                       "seed. v3's lambda sweep was only at 25k steps "
                       "and seed 42, so this is the first 100k-budget "
                       "direct comparison.",
        "random_seed": 44,
        "consistency_lambda": 0.1,
        **_V4_BASE,
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

    kind = cell.get("kind", "consistency")
    if kind not in ("consistency", "single_model"):
        raise ValueError(
            f"Cell {name!r} has unknown kind={kind!r}. "
            "Allowed: 'consistency' (default), 'single_model'."
        )
    h.cell_kind = kind

    for k, v in cell.items():
        if k in ("name", "description", "kind"):
            continue
        setattr(h, k, v)

    # For a single-model baseline, force M=1 and disable consistency so the
    # existing trainer takes the baseline code path (no unsup forward, no
    # agreement term). The "shard" is then the full permuted train set.
    if kind == "single_model":
        h.n_models = 1
        h.sharding = "disjoint"
        h.consistency_loss = "none"
        h.consistency_lambda = 0.0
        h.consistency_warmup_steps = 0
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
