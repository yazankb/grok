#!/usr/bin/env python
"""Kaggle-friendly sweep driver for the consistency-regularised experiment.

Runs a configurable list of cells (each cell = one full training run) and
writes a top-level ``sweep_summary.json`` indexing all per-run results. Each
cell creates its own subdirectory under ``--logdir/<sweep_name>/<cell_name>``.

Default pilot is v3 (see ``reports/consistency_regularized_grokking_plan.md``
§8 revision). Based on the v2 results the headline finding was that
``train_inputs_only`` KL-softmax consistency at lambda=0.1 (21% ensemble val
acc, best specialist 29%) substantially beats both the single-model-on-50%
baseline (12%) and the multi-specialist-no-consistency baseline (1.4%) at
25k steps. v3 is three micro-sweeps on that winning configuration:

    [seed robustness, 4 cells]
    1. ``seed42_trainonly_lam01``  - reruns the pilot_v2 winner (seed 42).
    2. ``seed43_trainonly_lam01``  - same config, seed 43.
    3. ``seed44_trainonly_lam01``  - same config, seed 44.
    4. ``seed45_trainonly_lam01``  - same config, seed 45. Together these
       decide whether the 21% result is a real effect or an init lottery
       driven by whichever specialist seed landed in a lucky basin.

    [longer runs, 2 cells at 100k steps]
    5. ``long_single_50pct_100k``  - single-model baseline at 100k steps.
                                   Checks whether the 12% plateau at 25k
                                   is budget-limited (still climbing) or
                                   architectural (would need a different
                                   recipe to grok at all). 100k is ~4x
                                   the grokking-transition budget reported
                                   in the original grokking paper.
    6. ``long_trainonly_lam01_100k`` - winning config at 100k steps.
                                   Checks whether val acc keeps climbing
                                   past 25k or plateaus.

    [lambda ablation on train_inputs_only, 3 cells]
    7. ``trainonly_lam003``        - same config as winner but lam=0.03.
    8. ``trainonly_lam03``         - lam=0.3.
    9. ``trainonly_lam10``         - lam=1.0. (lam=0.1 is covered by cell 1.)

Example (Kaggle, T4 GPU)::

    python scripts/run_consistency_sweep.py \
        --sweep_name pilot_v3 \
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
    p.add_argument("--sweep_name", type=str, default="pilot_v3",
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


# Pilot sweep v3: 9 runs, three micro-sweeps on the pilot_v2 winner
# (multi_kl_lam01_warm5k_trainonly: KL-softmax, lam=0.1, 5k warmup,
# train_inputs_only). See the module docstring for the rationale and the
# "winner" reference in reports/consistency_regularized_grokking_plan.md.
_WINNER_BASE: Dict[str, Any] = {
    "consistency_loss": "kl_softmax",
    "consistency_lambda": 0.1,
    "consistency_warmup_steps": 5000,
    "consistency_domain": "train_inputs_only",
}

DEFAULT_PILOT_CELLS: List[Dict[str, Any]] = [
    # --- 1-4: seed robustness on the winning config -----------------------
    {
        "name": "seed42_trainonly_lam01",
        "description": "Seed 42 (pilot_v2 winner). Establishes the anchor "
                       "for the 3 additional-seed reruns in cells 2-4.",
        "random_seed": 42,
        **_WINNER_BASE,
    },
    {
        "name": "seed43_trainonly_lam01",
        "description": "Seed 43 of the winning config. Different shard "
                       "permutation and different specialist inits.",
        "random_seed": 43,
        **_WINNER_BASE,
    },
    {
        "name": "seed44_trainonly_lam01",
        "description": "Seed 44 of the winning config.",
        "random_seed": 44,
        **_WINNER_BASE,
    },
    {
        "name": "seed45_trainonly_lam01",
        "description": "Seed 45 of the winning config. If all four seeds "
                       "produce 15-25% ensemble val acc with a ~25-30% "
                       "breakout specialist, the co-training story is real.",
        "random_seed": 45,
        **_WINNER_BASE,
    },
    # --- 5-6: longer runs (100k steps) -----------------------------------
    {
        "name": "long_single_50pct_100k",
        "description": "Single-model on the 50% slice, 100k steps. Tells us "
                       "whether single-model-12%-at-25k is budget-limited "
                       "(keeps climbing) or architectural (plateaued). 100k "
                       "is ~4x the budget at which the original grokking "
                       "paper reports its transition on modular arithmetic.",
        "kind": "single_model",
        "consistency_loss": "none",
        "consistency_lambda": 0.0,
        "consistency_warmup_steps": 0,
        "consistency_domain": "full_grid",
        "consistency_steps": 100000,
    },
    {
        "name": "long_trainonly_lam01_100k",
        "description": "Winning config at 100k steps. Tells us whether the "
                       "21%-at-25k result keeps climbing or plateaus.",
        "consistency_steps": 100000,
        **_WINNER_BASE,
    },
    # --- 7-9: lambda ablation on train_inputs_only ------------------------
    {
        "name": "trainonly_lam003",
        "description": "Winning config but lam=0.03. Probes the weak-lambda "
                       "tail to see whether less consistency also works.",
        **{**_WINNER_BASE, "consistency_lambda": 0.03},
    },
    {
        "name": "trainonly_lam03",
        "description": "Winning config but lam=0.3. Probes the midpoint.",
        **{**_WINNER_BASE, "consistency_lambda": 0.3},
    },
    {
        "name": "trainonly_lam10",
        "description": "Winning config but lam=1.0. Strong-consistency bound "
                       "on train_inputs_only. Compare against the lam=1.0 "
                       "full-grid cell from pilot_v2 to isolate the domain "
                       "effect at high lambda.",
        **{**_WINNER_BASE, "consistency_lambda": 1.0},
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
