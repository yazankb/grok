#!/usr/bin/env python
"""
Entry point for the multi-model grokking experiment.

Example (from repo root):

    python scripts/train_multi.py \
        --n_models 4 \
        --train_data_pct 50 \
        --specialist_steps 50000 \
        --final_steps 100000 \
        --overfit_threshold 0.99 \
        --experiment_name multi_exp_1

With distillation:
    python scripts/train_multi.py \
        --n_models 4 \
        --train_data_pct 50 \
        --specialist_steps 10000 \
        --final_steps 50000 \
        --use_distillation \
        --distill_steps 25000 \
        --experiment_name distill_exp_1

All base training flags (--math_operator, --n_layers, --d_model, etc.) are
also accepted and forwarded to each model.
"""

import os

from grok.multi_training import add_multi_args, train_multi, train_multi_with_distillation

# Use cwd as project root to avoid Unicode path issues on Windows
_project_root = os.getcwd()

parser = add_multi_args()
parser.set_defaults(
    logdir=os.environ.get("GROK_LOGDIR", os.path.join(_project_root, "logs")),
    datadir=os.path.join(_project_root, "data"),
)
hparams = parser.parse_args()

# Resolve paths to absolute so nested log subdirs work regardless of cwd
hparams.datadir = (
    os.path.normpath(os.path.join(_project_root, hparams.datadir))
    if not os.path.isabs(hparams.datadir)
    else os.path.normpath(hparams.datadir)
)
hparams.logdir = (
    os.path.normpath(os.path.join(_project_root, hparams.logdir))
    if not os.path.isabs(hparams.logdir)
    else os.path.normpath(hparams.logdir)
)

print(hparams)

if getattr(hparams, "use_distillation", False):
    print(train_multi_with_distillation(hparams))
else:
    print(train_multi(hparams))
