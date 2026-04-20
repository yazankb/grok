#!/usr/bin/env python
"""Kaggle-friendly sweep driver for the consistency-regularised experiment.

Runs a configurable list of cells (each cell = one full training run) and
writes a top-level ``sweep_summary.json`` indexing all per-run results. Each
cell creates its own subdirectory under ``--logdir/<sweep_name>/<cell_name>``.

Default pilot is v2 (see ``reports/consistency_regularized_grokking_plan.md``
§8 revision): 5 cells covering

    1. ``single_model_50pct``     - single model trained on the 50%-shard (M=1)
                                   baseline; calibrates what grokking looks
                                   like without multi-specialist dynamics.
    2. ``multi_lam0``              - multi-specialist, consistency off.
                                   Calibrates ensemble/merged acc baseline.
    3. ``multi_kl_lam01_warm5k``   - KL-softmax, lam=0.1, 5k-step warmup.
                                   Headline consistency cell.
    4. ``multi_kl_lam1_warm5k``    - KL-softmax, lam=1.0, 5k-step warmup.
                                   Probes whether stronger consistency helps
                                   or triggers agreement collapse.
    5. ``multi_kl_lam01_warm5k_trainonly``
                                   - same as #3 but consistency is enforced
                                   on ``train_inputs_only`` rather than the
                                   full grid. Controls for transductive leak.

v2 fixes two bugs identified in the v1 pilot (see the updated plan):
- MSE-on-logits was not scale-matched to CE; we switch to KL-on-softmax which
  has comparable magnitudes across training.
- 1k-step warmup was too short on a 25k-step budget; models collapsed to flat
  outputs before they had a chance to memorise. We use 5k-step warmup.
- The missing single-model-on-50% baseline is added so we can interpret the
  multi-specialist cells quantitatively.

Example (Kaggle, T4 GPU)::

    python scripts/run_consistency_sweep.py \
        --sweep_name pilot_v2 \
        --logdir /kaggle/working/consistency_runs \
        --consistency_steps 25000 \
        --gpu 0

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
    p.add_argument("--sweep_name", type=str, default="pilot_v2",
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


# Pilot sweep v2: 5 runs. Cells 1 and 2 calibrate baselines; cells 3-5
# perturb lambda, strength-of-consistency, and consistency domain on an
# otherwise identical setup. KL-softmax (not MSE-on-logits) and 5k warmup
# (not 1k) come from the v1-pilot post-mortem in the plan §8 revision.
DEFAULT_PILOT_CELLS: List[Dict[str, Any]] = [
    {
        "name": "single_model_50pct",
        "description": "Single model on the 50% train slice. Baseline that "
                       "calibrates what grokking looks like without any "
                       "multi-specialist or consistency dynamics.",
        "kind": "single_model",
        # The remaining consistency fields are ignored for kind=single_model
        # (consistency_loss is forced to 'none' and n_models to 1), but we
        # set them explicitly so the run config is unambiguous.
        "consistency_loss": "none",
        "consistency_lambda": 0.0,
        "consistency_warmup_steps": 0,
        "consistency_domain": "full_grid",
    },
    {
        "name": "multi_lam0",
        "description": "Multi-specialist (M=4, disjoint shards), no "
                       "consistency. Calibrates ensemble/merged val acc with "
                       "only CE on shards.",
        "consistency_loss": "none",
        "consistency_lambda": 0.0,
        "consistency_warmup_steps": 0,
        "consistency_domain": "full_grid",
    },
    {
        "name": "multi_kl_lam01_warm5k",
        "description": "KL-softmax consistency, lam=0.1, 5k-step warmup on "
                       "25k total. Headline candidate: weak consistency "
                       "after memorisation has a chance to start.",
        "consistency_loss": "kl_softmax",
        "consistency_lambda": 0.1,
        "consistency_warmup_steps": 5000,
        "consistency_domain": "full_grid",
    },
    {
        "name": "multi_kl_lam1_warm5k",
        "description": "KL-softmax consistency, lam=1.0, 5k-step warmup. "
                       "Bounds the strong-consistency regime; will likely "
                       "show agreement collapse if one exists at this scale.",
        "consistency_loss": "kl_softmax",
        "consistency_lambda": 1.0,
        "consistency_warmup_steps": 5000,
        "consistency_domain": "full_grid",
    },
    {
        "name": "multi_kl_lam01_warm5k_trainonly",
        "description": "Same as multi_kl_lam01_warm5k but consistency is "
                       "enforced on train inputs only (no val-input leak). "
                       "Controls for transductive contamination.",
        "consistency_loss": "kl_softmax",
        "consistency_lambda": 0.1,
        "consistency_warmup_steps": 5000,
        "consistency_domain": "train_inputs_only",
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
