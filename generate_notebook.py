#!/usr/bin/env python
"""Generate simplified Kaggle notebook using this repo."""
import json

def md(text):
    return {"cell_type": "markdown", "source": text.splitlines(), "metadata": {}, "execution_count": None, "outputs": []}

def code(text):
    return {"cell_type": "code", "source": text.splitlines(), "metadata": {}, "execution_count": None, "outputs": []}

cells = []

cells.append(md(
    "# Grokking: Weight Averaging & Distillation Experiment\n"
    "\n"
    "Reproduces the grokking experiment with:\n"
    "- **Baseline**: single model trained on full data\n"
    "- **4 Specialists**: each trained on 25% of the data\n"
    "- **Weight Averaged**: average specialist weights, fine-tune on full data\n"
    "- **Distilled**: knowledge distillation from specialists\n"
))

cells.append(code(
    "!pip install -q pytorch_lightning mod sympy blobfile numpy pandas matplotlib"
))

cells.append(md("## Clone Repo"))

cells.append(code(
    "import os, sys\n"
    "\n"
    "# Clone this repo - already has validation fix + split_n_ways + distillation metrics\n"
    "REPO_DIR = '/kaggle/working/grok'\n"
    "if not os.path.exists(REPO_DIR):\n"
    "    !git clone https://github.com/yazankb/grok.git {REPO_DIR}\n"
    "\n"
    "sys.path.insert(0, REPO_DIR)\n"
    "os.chdir(REPO_DIR)\n"
    "print(f'Repo at: {REPO_DIR}')"
))

cells.append(md("## Imports & GPU"))

cells.append(code(
    "import torch\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "from grok.multi_training import train_multi_with_distillation, add_multi_args\n"
    "\n"
    "GPU_ID = 0 if torch.cuda.is_available() else -1\n"
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "print(f'Device: {DEVICE}')"
))

cells.append(md("## Configuration\n\n"
    "For quick test: FINAL_STEPS = 2000"))

cells.append(code(
    "# Experiment config\n"
    "FINAL_STEPS = 250\n"
    "N_MODELS = 4\n"
    "SPECIALIST_STEPS = FINAL_STEPS // N_MODELS\n"
    "RANDOM_SEED = 42\n"
    "\n"
    "LOGDIR = '/kaggle/working/logs'\n"
    "DATADIR = os.path.join(REPO_DIR, 'data')\n"
    "EXPERIMENT_NAME = 'grokking_experiment'\n"
    "\n"
    "print(f'Final steps: {FINAL_STEPS}')\n"
    "print(f'Specialist steps: {SPECIALIST_STEPS} x {N_MODELS} = {SPECIALIST_STEPS * N_MODELS} total')"
))

cells.append(md("## Build Hyperparameters"))

cells.append(code(
    "parser = add_multi_args()\n"
    "hparams = parser.parse_args([])\n"
    "\n"
    "hparams.logdir = LOGDIR\n"
    "hparams.datadir = DATADIR\n"
    "hparams.math_operator = '/'\n"
    "hparams.train_data_pct = 50\n"
    "hparams.n_models = N_MODELS\n"
    "hparams.specialist_steps = SPECIALIST_STEPS\n"
    "hparams.final_steps = FINAL_STEPS\n"
    "hparams.distill_steps = FINAL_STEPS\n"
    "hparams.distill_temperature = 2.0\n"
    "hparams.distill_alpha = 0.5\n"
    "hparams.experiment_name = EXPERIMENT_NAME\n"
    "hparams.run_baseline = True\n"
    "hparams.random_seed = RANDOM_SEED\n"
    "hparams.gpu = GPU_ID\n"
    "hparams.weight_decay = 1\n"
    "hparams.max_lr = 1e-3\n"
    "hparams.warmup_steps = 10\n"
    "hparams.n_layers = 2\n"
    "hparams.n_heads = 4\n"
    "hparams.d_model = 128\n"
    "\n"
    "print(f'Model: {hparams.n_layers}L-{hparams.n_heads}H-{hparams.d_model}D')\n"
    "print(f'Operator: {hparams.math_operator} mod 97 | Train: {hparams.train_data_pct}%')"
))

cells.append(md("## Run Experiment\n\n"
    "This runs: 4 specialists -> weight average -> fine-tune -> distill -> baseline"))

cells.append(code(
    "experiment_dir = train_multi_with_distillation(hparams)\n"
    "print(f'Results: {experiment_dir}')"
))

cells.append(md("## Load Metrics"))

cells.append(code(
    "import os\n"
    "\n"
    "def find_csv(base_dir, subdir):\n"
    "    path = os.path.join(base_dir, subdir, 'lightning_logs', 'version_0', 'metrics.csv')\n"
    "    return path if os.path.exists(path) else None\n"
    "\n"
    "metrics = {}\n"
    "for i in range(N_MODELS):\n"
    "    p = find_csv(experiment_dir, f'specialist_{i}')\n"
    "    if p: metrics[f'specialist_{i}'] = p\n"
    "\n"
    "for name in ['merged_average', 'baseline']:\n"
    "    p = find_csv(experiment_dir, name)\n"
    "    if p: metrics[name] = p\n"
    "\n"
    "# Distillation has custom CSV\n"
    "dc = os.path.join(experiment_dir, 'distilled', 'distill_metrics.csv')\n"
    "if os.path.exists(dc): metrics['distilled'] = dc\n"
    "\n"
    "dvc = os.path.join(experiment_dir, 'distilled', 'distill_val_metrics.csv')\n"
    "if os.path.exists(dvc): metrics['distilled_val'] = dvc\n"
    "\n"
    "print('Found:', list(metrics.keys()))\n"
    "\n"
    "# Load and inspect data\n"
    "data = {}\n"
    "for k, v in metrics.items():\n"
    "    df = pd.read_csv(v)\n"
    "    # Drop rows where step or target metric is NaN\n"
    "    if 'val_accuracy' in df.columns:\n"
    "        df = df.dropna(subset=['val_accuracy', 'step'])\n"
    "    elif 'student_acc' in df.columns:\n"
    "        df = df.dropna(subset=['student_acc', 'step'])\n"
    "    elif 'full_train_acc' in df.columns:\n"
    "        df = df.dropna(subset=['full_train_acc', 'step'])\n"
    "    data[k] = df\n"
    "    print(f'{k}: {len(df)} rows, cols: {list(df.columns)[:10]}')"
))

cells.append(md("## Plot: Accuracy Curves"))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "\n"
    "colors = {'baseline': '#2196F3', 'merged_average': '#4CAF50', 'distilled': '#9C27B0'}\n"
    "\n"
    "# Validation Accuracy\n"
    "ax = axes[0]\n"
    "for i in range(N_MODELS):\n"
    "    name = f'specialist_{i}'\n"
    "    if name in data and len(data[name]) > 0 and 'val_accuracy' in data[name].columns:\n"
    "        data[name].plot(x='step', y='val_accuracy', ax=ax, alpha=0.3, color='gray', legend=False)\n"
    "\n"
    "for name, color in colors.items():\n"
    "    if name in data and len(data[name]) > 0 and 'val_accuracy' in data[name].columns:\n"
    "        data[name].plot(x='step', y='val_accuracy', ax=ax, label=name, color=color, legend=True)\n"
    "\n"
    "if 'distilled_val' in data and len(data['distilled_val']) > 0:\n"
    "    data['distilled_val'].plot(x='step', y='val_accuracy', ax=ax, label='distilled', color=colors['distilled'], legend=True)\n"
    "\n"
    "ax.set_title('Validation Accuracy')\n"
    "ax.set_xlabel('Step')\n"
    "ax.set_ylabel('Accuracy %')\n"
    "ax.set_ylim(0, 105)\n"
    "ax.grid(True, alpha=0.3)\n"
    "\n"
    "# Training Accuracy\n"
    "ax = axes[1]\n"
    "for i in range(N_MODELS):\n"
    "    name = f'specialist_{i}'\n"
    "    if name in data and len(data[name]) > 0:\n"
    "        col = 'full_train_acc' if 'full_train_acc' in data[name].columns else 'train_accuracy'\n"
    "        if col in data[name].columns:\n"
    "            data[name].plot(x='step', y=col, ax=ax, alpha=0.3, color='gray', legend=False)\n"
    "\n"
    "for name, color in colors.items():\n"
    "    if name in data and len(data[name]) > 0:\n"
    "        col = 'full_train_acc' if 'full_train_acc' in data[name].columns else 'train_accuracy'\n"
    "        if col in data[name].columns:\n"
    "            data[name].plot(x='step', y=col, ax=ax, label=name, color=color, legend=True)\n"
    "\n"
    "if 'distilled' in data and len(data['distilled']) > 0 and 'student_acc' in data['distilled'].columns:\n"
    "    data['distilled'].plot(x='step', y='student_acc', ax=ax, label='distilled', color=colors['distilled'], legend=True)\n"
    "\n"
    "ax.set_title('Training Accuracy')\n"
    "ax.set_xlabel('Step')\n"
    "ax.set_ylabel('Accuracy %')\n"
    "ax.set_ylim(0, 105)\n"
    "ax.grid(True, alpha=0.3)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig('/kaggle/working/accuracy.png', dpi=150)\n"
    "plt.show()"
))

cells.append(md("## Plot: Dashboard"))

cells.append(code(
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n"
    "\n"
    "# Val Accuracy\n"
    "ax = axes[0, 0]\n"
    "for name, color in colors.items():\n"
    "    if name in data and len(data[name]) > 0 and 'val_accuracy' in data[name].columns:\n"
    "        data[name].plot(x='step', y='val_accuracy', ax=ax, label=name, color=color)\n"
    "if 'distilled_val' in data and len(data['distilled_val']) > 0:\n"
    "    data['distilled_val'].plot(x='step', y='val_accuracy', ax=ax, label='distilled', color=colors['distilled'])\n"
    "ax.set_title('Validation Accuracy')\n"
    "ax.set_ylim(0, 105)\n"
    "ax.grid(True, alpha=0.3)\n"
    "\n"
    "# Val Loss (log)\n"
    "ax = axes[0, 1]\n"
    "for name, color in colors.items():\n"
    "    if name in data and len(data[name]) > 0 and 'val_loss' in data[name].columns:\n"
    "        data[name].plot(x='step', y='val_loss', ax=ax, label=name, color=color)\n"
    "ax.set_title('Validation Loss')\n"
    "ax.set_yscale('log')\n"
    "ax.grid(True, alpha=0.3)\n"
    "\n"
    "# Train Accuracy\n"
    "ax = axes[1, 0]\n"
    "for name, color in colors.items():\n"
    "    if name in data and len(data[name]) > 0:\n"
    "        col = 'full_train_acc' if 'full_train_acc' in data[name].columns else 'train_accuracy'\n"
    "        if col in data[name].columns:\n"
    "            data[name].plot(x='step', y=col, ax=ax, label=name, color=color)\n"
    "if 'distilled' in data and len(data['distilled']) > 0 and 'student_acc' in data['distilled'].columns:\n"
    "    data['distilled'].plot(x='step', y='student_acc', ax=ax, label='distilled', color=colors['distilled'])\n"
    "ax.set_title('Training Accuracy')\n"
    "ax.set_ylim(0, 105)\n"
    "ax.grid(True, alpha=0.3)\n"
    "\n"
    "# Final bar chart\n"
    "ax = axes[1, 1]\n"
    "final_acc = []\n"
    "labels = []\n"
    "bar_colors = []\n"
    "for name, label, bar_color in [('baseline', 'Baseline', colors['baseline']), ('merged_average', 'Weight Avg', colors['merged_average']), ('distilled_val', 'Distilled', colors['distilled'])]:\n"
    "    if name in data and len(data[name]) > 0 and 'val_accuracy' in data[name].columns:\n"
    "        acc = float(data[name]['val_accuracy'].iloc[-1])\n"
    "        final_acc.append(acc)\n"
    "        labels.append(label)\n"
    "        bar_colors.append(bar_color)\n"
    "if final_acc:\n"
    "    bars = ax.bar(labels, final_acc, color=bar_colors)\n"
    "    ax.set_ylabel('Accuracy %')\n"
    "    ax.set_title('Final Val Accuracy')\n"
    "    for bar, acc in zip(bars, final_acc):\n"
    "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%', ha='center')\n"
    "    ax.set_ylim(0, max(final_acc) * 1.2 if final_acc else 100)\n"
    "    ax.grid(True, alpha=0.3, axis='y')\n"
    "\n"
    "plt.suptitle(f'Grokking: {hparams.math_operator} mod 97 | {N_MODELS} specialists | {FINAL_STEPS:,} steps')\n"
    "plt.tight_layout()\n"
    "plt.savefig('/kaggle/working/dashboard.png', dpi=150)\n"
    "plt.show()\n"
    "\n"
    "print('Plots saved to /kaggle/working/')"
))

cells.append(md("## Results"))

cells.append(code(
    "print('='*50)\n"
    "print('FINAL RESULTS')\n"
    "print('='*50)\n"
    "for name, label in [('baseline', 'Baseline'), ('merged_average', 'Weight Avg'), ('distilled_val', 'Distilled')]:\n"
    "    if name in data and len(data[name]) > 0 and 'val_accuracy' in data[name].columns:\n"
    "        acc = float(data[name]['val_accuracy'].iloc[-1])\n"
    "        print(f'{label:15s}: {acc:.1f}%')"
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('kaggle_grokking.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

with open('kaggle_grokking.ipynb', 'r') as f:
    loaded = json.load(f)
print(f'Notebook: {len(loaded["cells"])} cells, valid JSON')
