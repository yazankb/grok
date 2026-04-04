#!/usr/bin/env python
"""Generate the Kaggle grokking notebook with proper JSON encoding."""
import json

def md(text):
    return {"cell_type": "markdown", "source": text.splitlines(), "metadata": {}, "execution_count": None, "outputs": []}

def code(text):
    return {"cell_type": "code", "source": text.splitlines(), "metadata": {}, "execution_count": None, "outputs": []}

cells = []

cells.append(md(
    "# Grokking: Weight Averaging & Distillation Experiment\n"
    "\n"
    "Reproduces the grokking experiment (Power et al., 2022) on modular division (mod 97), then compares:\n"
    "- **Baseline**: single model trained on full data\n"
    "- **4 Specialists**: each trained on 25% of the data\n"
    "- **Weight Averaged**: average specialist weights, fine-tune on full data\n"
    "- **Distilled**: knowledge distillation from specialists into a fresh student\n"
    "\n"
    "All methods use the **same total compute budget** for fair comparison."
))

cells.append(code(
    "!pip install -q pytorch_lightning mod sympy blobfile"
))

cells.append(md("## Clone Repo"))

cells.append(code(
    "import os, sys\n"
    "\n"
    "REPO_DIR = '/kaggle/working/grok'\n"
    "if not os.path.exists(REPO_DIR):\n"
    "    !git clone https://github.com/openai/grok {REPO_DIR}\n"
    "\n"
    "sys.path.insert(0, REPO_DIR)\n"
    "os.chdir(REPO_DIR)\n"
    "print(f'Repo at: {REPO_DIR}')"
))

cells.append(md(
    "## Apply Fixes\n"
    "\n"
    "The original code never logs val_loss/val_accuracy/full_train_acc in Lightning 2.x. "
    "Also adds `split_n_ways` (data splitting) which is missing from the original repo."
))

# This is the critical fix cell - use raw string to avoid escaping issues
fix_lines = [
    "import re",
    "",
    "TRAINING_FILE = os.path.join(REPO_DIR, 'grok', 'training.py')",
    "with open(TRAINING_FILE, 'r') as f:",
    "    code = f.read()",
    "",
    "# Fix 1: validation_step - append {} to _val_step_outputs on skip",
    "old_val_step = (",
    "    '        if self.next_epoch_to_eval < self.current_epoch:\\n'",
    "    '            self.next_epoch_to_eval = self.current_epoch\\n'",
    "    '        if self.current_epoch != self.next_epoch_to_eval:\\n'",
    "    '            return {}'",
    ")",
    "new_val_step = (",
    "    '        if self.next_epoch_to_eval < self.current_epoch:\\n'",
    "    '            self.next_epoch_to_eval = self.current_epoch\\n'",
    "    '        if self.current_epoch != self.next_epoch_to_eval:\\n'",
    "    '            self._val_step_outputs.append({})\\n'",
    "    '            return {}'",
    ")",
    "assert old_val_step in code, 'validation_step pattern not found'",
    "code = code.replace(old_val_step, new_val_step)",
    "print('Fixed validation_step')",
    "",
    "# Fix 2: on_validation_epoch_end - remove stale next_epoch_to_eval update",
    "old_val_end = (",
    "    '        if validation_is_real:\\n'",
    "    '            self.next_epoch_to_eval = max(\\n'",
    "    '                int(1.02 * self.next_epoch_to_eval), self.next_epoch_to_eval + 1\\n'",
    "    '            )'",
    ")",
    "assert old_val_end in code, 'on_validation_epoch_end header not found'",
    "code = code.replace(old_val_end, '        if validation_is_real:')",
    "print('Fixed on_validation_epoch_end')",
    "",
    "# Fix 3: checkpoint_path fallback",
    "old_ckpt = (",
    "    '            self.trainer.save_checkpoint(\\n'",
    "    '                os.path.join(\\n'",
    "    '                    self.hparams.checkpoint_path,\\n'",
    "    '                    \"epoch_\" + str(self.current_epoch) + \".ckpt\",\\n'",
    "    '                )\\n'",
    "    '            )'",
    ")",
    "new_ckpt = (",
    "    '            ckpt_path = getattr(self.hparams, \"checkpoint_path\", None) or os.path.join(\\n'",
    "    '                self.hparams.logdir, \"checkpoints\"\\n'",
    "    '            )\\n'",
    "    '            os.makedirs(ckpt_path, exist_ok=True)\\n'",
    "    '            self.trainer.save_checkpoint(\\n'",
    "    '                os.path.join(\\n'",
    "    '                    ckpt_path,\\n'",
    "    '                    \"epoch_\" + str(self.current_epoch) + \".ckpt\",\\n'",
    "    '                )\\n'",
    "    '            )'",
    ")",
    "assert old_ckpt in code, 'checkpoint_path pattern not found'",
    "code = code.replace(old_ckpt, new_ckpt)",
    "print('Fixed checkpoint_path')",
    "",
    "with open(TRAINING_FILE, 'w') as f:",
    "    f.write(code)",
    "",
    "# Add split_n_ways to data.py (not in original OpenAI repo)",
    "DATA_FILE = os.path.join(REPO_DIR, 'grok', 'data.py')",
    "with open(DATA_FILE, 'r') as f:",
    "    data_code = f.read()",
    "",
    "if 'split_n_ways' not in data_code:",
    "    split_n_ways_code = (",
    "        '\\n'",
    "        '    @classmethod\\n'",
    "        '    def split_n_ways(\\n'",
    "        '        cls,\\n'",
    "        '        dataset: \"ArithmeticDataset\",\\n'",
    "        '        n: int,\\n'",
    "        '        seed: int = 42,\\n'",
    "        '    ) -> List[\"ArithmeticDataset\"]:\\n'",
    "        '        \"\"\"Partition an existing dataset into n disjoint shards.\"\"\"\\n'",
    "        '        gen = torch.Generator()\\n'",
    "        '        gen.manual_seed(seed)\\n'",
    "        '        perm = torch.randperm(len(dataset.data), generator=gen)\\n'",
    "        '        chunks = torch.chunk(perm, n)\\n'",
    "        '        result = []\\n'",
    "        '        for chunk in chunks:\\n'",
    "        '            shard = cls.__new__(cls)\\n'",
    "        '            shard.tokenizer = dataset.tokenizer\\n'",
    "        '            shard.name = dataset.name\\n'",
    "        '            shard.train = True\\n'",
    "        '            shard.data = dataset.data[chunk]\\n'",
    "        '            result.append(shard)\\n'",
    "        '        return result\\n'",
    "    )",
    "    marker = '    @classmethod\\n    def _make_lists'",
    "    idx = data_code.find(marker)",
    "    if idx != -1:",
    "        data_code = data_code[:idx] + split_n_ways_code + '\\n\\n' + data_code[idx:]",
    "    with open(DATA_FILE, 'w') as f:",
    "        f.write(data_code)",
    "    print('Added split_n_ways to data.py')",
    "else:",
    "    print('split_n_ways already in data.py')",
    "",
    "# Fix multi_training.py imports",
    "MULTI_FILE = os.path.join(REPO_DIR, 'grok', 'multi_training.py')",
    "with open(MULTI_FILE, 'r') as f:",
    "    multi_code = f.read()",
    "if 'from grok.data import ArithmeticDataset' in multi_code and 'ArithmeticIterator' not in multi_code:",
    "    multi_code = multi_code.replace(",
    "        'from grok.data import ArithmeticDataset',",
    "        'from grok.data import ArithmeticDataset, ArithmeticIterator'",
    "    )",
    "    with open(MULTI_FILE, 'w') as f:",
    "        f.write(multi_code)",
    "    print('Added ArithmeticIterator import')",
    "else:",
    "    print('multi_training.py imports already correct')",
    "",
    "print('All fixes applied!')",
]

cells.append({"cell_type": "code", "source": fix_lines, "metadata": {}, "execution_count": None, "outputs": []})

cells.append(md("## Imports & GPU Detection"))

cells.append(code(
    "import torch\n"
    "import copy\n"
    "import json\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "from argparse import Namespace\n"
    "\n"
    "from grok.multi_training import (\n"
    "    train_multi_with_distillation,\n"
    "    add_multi_args,\n"
    ")\n"
    "\n"
    "GPU_ID = 0 if torch.cuda.is_available() else -1\n"
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "print(f'Device: {DEVICE}')\n"
    "print(f'GPU ID: {GPU_ID}')"
))

cells.append(md(
    "## Experiment Configuration\n"
    "\n"
    "Paper-exact hyperparameters for modular division (mod 97).\n"
    "\n"
    "**Compute budget:** 100k total steps.\n"
    "- 4 specialists x 25k steps each = 100k total specialist compute\n"
    "- Merged model: 100k steps (fine-tune from averaged weights)\n"
    "- Distillation student: 100k steps (train from scratch)\n"
    "- Baseline: 100k steps (train from scratch)\n"
    "\n"
    "For a quick test, reduce `FINAL_STEPS` to 1000-5000."
))

cells.append(code(
    "# --- Experiment parameters ---\n"
    "FINAL_STEPS = 100000        # Total compute budget (reduce for quick test: e.g. 2000)\n"
    "N_MODELS = 4                # Number of specialist models\n"
    "SPECIALIST_STEPS = FINAL_STEPS // N_MODELS  # Equal compute per specialist\n"
    "DISTILL_STEPS = FINAL_STEPS # Same budget as baseline\n"
    "RANDOM_SEED = 42\n"
    "\n"
    "# --- Paths ---\n"
    "LOGDIR = '/kaggle/working/logs'\n"
    "DATADIR = os.path.join(REPO_DIR, 'data')\n"
    "EXPERIMENT_NAME = 'grokking_experiment'\n"
    "\n"
    "print(f'Final steps (baseline/merged/distilled): {FINAL_STEPS}')\n"
    "print(f'Specialist steps: {SPECIALIST_STEPS} each x {N_MODELS} = {SPECIALIST_STEPS * N_MODELS} total')\n"
    "print(f'Distill steps: {DISTILL_STEPS}')\n"
    "print(f'Logdir: {LOGDIR}')"
))

cells.append(md("## Build Hyperparameters"))

cells.append(code(
    "parser = add_multi_args()\n"
    "hparams = parser.parse_args([])  # empty args, use defaults below\n"
    "\n"
    "# Override with our config\n"
    "hparams.logdir = LOGDIR\n"
    "hparams.datadir = DATADIR\n"
    "hparams.math_operator = '/'\n"
    "hparams.train_data_pct = 50\n"
    "hparams.n_models = N_MODELS\n"
    "hparams.specialist_steps = SPECIALIST_STEPS\n"
    "hparams.final_steps = FINAL_STEPS\n"
    "hparams.distill_steps = DISTILL_STEPS\n"
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
    "hparams.dropout = 0.0\n"
    "hparams.max_context_len = 50\n"
    "hparams.non_linearity = 'relu'\n"
    "hparams.batchsize = 0\n"
    "hparams.weight_noise = 0.0\n"
    "hparams.noise_factor = 0\n"
    "hparams.weight_decay_kind = 'to_zero'\n"
    "hparams.anneal_lr = False\n"
    "hparams.anneal_lr_steps = 100000\n"
    "hparams.save_activations = False\n"
    "hparams.save_outputs = False\n"
    "hparams.operand_length = None\n"
    "\n"
    "print(hparams)\n"
    "print(f'Model: {hparams.n_layers}L-{hparams.n_heads}H-{hparams.d_model}D (~455K params)')\n"
    "print(f'Operator: {hparams.math_operator} mod 97 | Train: {hparams.train_data_pct}%')"
))

cells.append(md(
    "## Run Full Experiment\n"
    "\n"
    "Phases:\n"
    "1. Train 4 specialists on disjoint data shards\n"
    "2. Average specialist weights\n"
    "3. Fine-tune merged model on full data\n"
    "4. Distill knowledge from specialists into a fresh student\n"
    "5. Train baseline (random init) on full data"
))

cells.append(code(
    "experiment_dir = train_multi_with_distillation(hparams)\n"
    "print(f'Experiment dir: {experiment_dir}')"
))

cells.append(md("## Discover All Metrics Files"))

cells.append(code(
    "def find_metrics_csv(experiment_dir, subdir):\n"
    "    base = os.path.join(experiment_dir, subdir)\n"
    "    if not os.path.exists(base):\n"
    "        return None\n"
    "    lightning_dir = os.path.join(base, 'lightning_logs')\n"
    "    if not os.path.exists(lightning_dir):\n"
    "        return None\n"
    "    versions = sorted([d for d in os.listdir(lightning_dir) if d.startswith('version_')])\n"
    "    for v in reversed(versions):\n"
    "        csv_path = os.path.join(lightning_dir, v, 'metrics.csv')\n"
    "        if os.path.exists(csv_path):\n"
    "            return csv_path\n"
    "    return None\n"
    "\n"
    "metrics_paths = {}\n"
    "for i in range(N_MODELS):\n"
    "    p = find_metrics_csv(experiment_dir, f'specialist_{i}')\n"
    "    if p:\n"
    "        metrics_paths[f'specialist_{i}'] = p\n"
    "\n"
    "for name in ['merged_average', 'baseline']:\n"
    "    p = find_metrics_csv(experiment_dir, name)\n"
    "    if p:\n"
    "        metrics_paths[name] = p\n"
    "\n"
    "distill_csv = os.path.join(experiment_dir, 'distilled', 'distill_metrics.csv')\n"
    "if os.path.exists(distill_csv):\n"
    "    metrics_paths['distilled'] = distill_csv\n"
    "\n"
    "distill_val_csv = os.path.join(experiment_dir, 'distilled', 'distill_val_metrics.csv')\n"
    "if os.path.exists(distill_val_csv):\n"
    "    metrics_paths['distilled_val'] = distill_val_csv\n"
    "\n"
    "print('Found metrics:')\n"
    "for name, path in metrics_paths.items():\n"
    "    print(f'  {name}: {path}')"
))

cells.append(md("## Load & Inspect All Metrics"))

cells.append(code(
    "all_data = {}\n"
    "for name, path in metrics_paths.items():\n"
    "    df = pd.read_csv(path)\n"
    "    all_data[name] = df\n"
    "    non_empty = {col: df[col].notna().sum() for col in df.columns}\n"
    "    print(f'{name}: {len(df)} rows')\n"
    "    for col in ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'full_train_acc', 'full_train_loss', 'student_acc']:\n"
    "        if col in non_empty:\n"
    "            print(f'  {col}: {non_empty[col]} values')"
))

cells.append(md("## Plot: Accuracy Curves (Grokking Curve)"))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
    "\n"
    "colors = {'baseline': '#2196F3', 'merged_average': '#4CAF50', 'distilled': '#9C27B0'}\n"
    "labels = {'baseline': 'Baseline', 'merged_average': 'Weight Averaged', 'distilled': 'Distilled'}\n"
    "\n"
    "# Left: Validation Accuracy\n"
    "ax = axes[0]\n"
    "for i in range(N_MODELS):\n"
    "    name = f'specialist_{i}'\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        if 'val_accuracy' in df.columns:\n"
    "            d = df.dropna(subset=['val_accuracy'])\n"
    "            ax.plot(d['step'], d['val_accuracy'], alpha=0.3, linewidth=0.8, color='gray')\n"
    "\n"
    "for name in ['baseline', 'merged_average']:\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        if 'val_accuracy' in df.columns:\n"
    "            d = df.dropna(subset=['val_accuracy'])\n"
    "            ax.plot(d['step'], d['val_accuracy'], label=labels[name], linewidth=2, color=colors[name])\n"
    "\n"
    "if 'distilled_val' in all_data:\n"
    "    df = all_data['distilled_val']\n"
    "    if 'val_accuracy' in df.columns:\n"
    "        ax.plot(df['step'], df['val_accuracy'], label='Distilled', linewidth=2, color=colors['distilled'])\n"
    "\n"
    "ax.set_title('Validation Accuracy', fontsize=14)\n"
    "ax.set_xlabel('Step')\n"
    "ax.set_ylabel('Accuracy (%)')\n"
    "ax.legend(loc='lower right')\n"
    "ax.grid(True, alpha=0.3)\n"
    "ax.set_ylim(0, 105)\n"
    "\n"
    "# Right: Training Accuracy\n"
    "ax = axes[1]\n"
    "for i in range(N_MODELS):\n"
    "    name = f'specialist_{i}'\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        col = 'full_train_acc' if 'full_train_acc' in df.columns else 'train_accuracy'\n"
    "        if col in df.columns:\n"
    "            d = df.dropna(subset=[col])\n"
    "            ax.plot(d['step'], d[col], alpha=0.3, linewidth=0.8, color='gray')\n"
    "\n"
    "for name in ['baseline', 'merged_average']:\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        col = 'full_train_acc' if 'full_train_acc' in df.columns else 'train_accuracy'\n"
    "        if col in df.columns:\n"
    "            d = df.dropna(subset=[col])\n"
    "            ax.plot(d['step'], d[col], label=labels[name], linewidth=2, color=colors[name])\n"
    "\n"
    "if 'distilled' in all_data:\n"
    "    df = all_data['distilled']\n"
    "    if 'student_acc' in df.columns:\n"
    "        ax.plot(df['step'], df['student_acc'], label='Distilled', linewidth=2, color=colors['distilled'], alpha=0.7)\n"
    "\n"
    "ax.set_title('Training Accuracy', fontsize=14)\n"
    "ax.set_xlabel('Step')\n"
    "ax.set_ylabel('Accuracy (%)')\n"
    "ax.legend(loc='lower right')\n"
    "ax.grid(True, alpha=0.3)\n"
    "ax.set_ylim(0, 105)\n"
    "\n"
    "fig.suptitle('Grokking: Division mod 97 - Accuracy', fontsize=16, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig('/kaggle/working/accuracy_curves.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: /kaggle/working/accuracy_curves.png')"
))

cells.append(md("## Plot: Loss Curves"))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
    "\n"
    "# Left: Validation Loss\n"
    "ax = axes[0]\n"
    "for i in range(N_MODELS):\n"
    "    name = f'specialist_{i}'\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        if 'val_loss' in df.columns:\n"
    "            d = df.dropna(subset=['val_loss'])\n"
    "            ax.plot(d['step'], d['val_loss'], alpha=0.3, linewidth=0.8, color='gray')\n"
    "\n"
    "for name in ['baseline', 'merged_average']:\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        if 'val_loss' in df.columns:\n"
    "            d = df.dropna(subset=['val_loss'])\n"
    "            ax.plot(d['step'], d['val_loss'], label=labels[name], linewidth=2, color=colors[name])\n"
    "\n"
    "ax.set_title('Validation Loss', fontsize=14)\n"
    "ax.set_xlabel('Step')\n"
    "ax.set_ylabel('Loss')\n"
    "ax.legend()\n"
    "ax.grid(True, alpha=0.3)\n"
    "ax.set_yscale('log')\n"
    "\n"
    "# Right: Training Loss\n"
    "ax = axes[1]\n"
    "for i in range(N_MODELS):\n"
    "    name = f'specialist_{i}'\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        col = 'full_train_loss' if 'full_train_loss' in df.columns else 'train_loss'\n"
    "        if col in df.columns:\n"
    "            d = df.dropna(subset=[col])\n"
    "            ax.plot(d['step'], d[col], alpha=0.3, linewidth=0.8, color='gray')\n"
    "\n"
    "for name in ['baseline', 'merged_average']:\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        col = 'full_train_loss' if 'full_train_loss' in df.columns else 'train_loss'\n"
    "        if col in df.columns:\n"
    "            d = df.dropna(subset=[col])\n"
    "            ax.plot(d['step'], d[col], label=labels[name], linewidth=2, color=colors[name])\n"
    "\n"
    "if 'distilled' in all_data:\n"
    "    df = all_data['distilled']\n"
    "    if 'loss' in df.columns:\n"
    "        window = max(1, len(df) // 200)\n"
    "        smoothed = df['loss'].rolling(window=window, center=True).mean()\n"
    "        ax.plot(df['step'], smoothed, label='Distilled', linewidth=2, color=colors['distilled'])\n"
    "\n"
    "ax.set_title('Training Loss', fontsize=14)\n"
    "ax.set_xlabel('Step')\n"
    "ax.set_ylabel('Loss')\n"
    "ax.legend()\n"
    "ax.grid(True, alpha=0.3)\n"
    "ax.set_yscale('log')\n"
    "\n"
    "fig.suptitle('Grokking: Division mod 97 - Loss', fontsize=16, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig('/kaggle/working/loss_curves.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: /kaggle/working/loss_curves.png')"
))

cells.append(md("## Plot: Combined Summary Dashboard"))

cells.append(code(
    "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n"
    "\n"
    "# Top Left: Val Accuracy\n"
    "ax = axes[0, 0]\n"
    "for i in range(N_MODELS):\n"
    "    name = f'specialist_{i}'\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        if 'val_accuracy' in df.columns:\n"
    "            d = df.dropna(subset=['val_accuracy'])\n"
    "            ax.plot(d['step'], d['val_accuracy'], alpha=0.3, linewidth=0.8, color='gray')\n"
    "for name in ['baseline', 'merged_average']:\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        if 'val_accuracy' in df.columns:\n"
    "            d = df.dropna(subset=['val_accuracy'])\n"
    "            ax.plot(d['step'], d['val_accuracy'], label=labels[name], linewidth=2, color=colors[name])\n"
    "if 'distilled_val' in all_data:\n"
    "    df = all_data['distilled_val']\n"
    "    ax.plot(df['step'], df['val_accuracy'], label='Distilled', linewidth=2, color=colors['distilled'])\n"
    "ax.set_title('Validation Accuracy', fontsize=13)\n"
    "ax.set_xlabel('Step')\n"
    "ax.set_ylabel('Accuracy (%)')\n"
    "ax.legend(loc='lower right')\n"
    "ax.grid(True, alpha=0.3)\n"
    "ax.set_ylim(0, 105)\n"
    "\n"
    "# Top Right: Val Loss (log)\n"
    "ax = axes[0, 1]\n"
    "for i in range(N_MODELS):\n"
    "    name = f'specialist_{i}'\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        if 'val_loss' in df.columns:\n"
    "            d = df.dropna(subset=['val_loss'])\n"
    "            ax.plot(d['step'], d['val_loss'], alpha=0.3, linewidth=0.8, color='gray')\n"
    "for name in ['baseline', 'merged_average']:\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        if 'val_loss' in df.columns:\n"
    "            d = df.dropna(subset=['val_loss'])\n"
    "            ax.plot(d['step'], d['val_loss'], label=labels[name], linewidth=2, color=colors[name])\n"
    "ax.set_title('Validation Loss (log scale)', fontsize=13)\n"
    "ax.set_xlabel('Step')\n"
    "ax.set_ylabel('Loss')\n"
    "ax.set_yscale('log')\n"
    "ax.legend()\n"
    "ax.grid(True, alpha=0.3)\n"
    "\n"
    "# Bottom Left: Train Accuracy\n"
    "ax = axes[1, 0]\n"
    "for i in range(N_MODELS):\n"
    "    name = f'specialist_{i}'\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        col = 'full_train_acc' if 'full_train_acc' in df.columns else 'train_accuracy'\n"
    "        if col in df.columns:\n"
    "            d = df.dropna(subset=[col])\n"
    "            ax.plot(d['step'], d[col], alpha=0.3, linewidth=0.8, color='gray')\n"
    "for name in ['baseline', 'merged_average']:\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        col = 'full_train_acc' if 'full_train_acc' in df.columns else 'train_accuracy'\n"
    "        if col in df.columns:\n"
    "            d = df.dropna(subset=[col])\n"
    "            ax.plot(d['step'], d[col], label=labels[name], linewidth=2, color=colors[name])\n"
    "if 'distilled' in all_data:\n"
    "    df = all_data['distilled']\n"
    "    if 'student_acc' in df.columns:\n"
    "        ax.plot(df['step'], df['student_acc'], label='Distilled', linewidth=2, color=colors['distilled'], alpha=0.7)\n"
    "ax.set_title('Training Accuracy', fontsize=13)\n"
    "ax.set_xlabel('Step')\n"
    "ax.set_ylabel('Accuracy (%)')\n"
    "ax.legend(loc='lower right')\n"
    "ax.grid(True, alpha=0.3)\n"
    "ax.set_ylim(0, 105)\n"
    "\n"
    "# Bottom Right: Final Accuracy Bar Chart\n"
    "ax = axes[1, 1]\n"
    "method_names = []\n"
    "final_accs = []\n"
    "bar_colors = []\n"
    "\n"
    "for name, label_text in [('baseline', 'Baseline'), ('merged_average', 'Weight Avg'), ('distilled', 'Distilled')]:\n"
    "    if name == 'distilled' and 'distilled_val' in all_data:\n"
    "        df = all_data['distilled_val']\n"
    "        if 'val_accuracy' in df.columns and len(df) > 0:\n"
    "            method_names.append(label_text)\n"
    "            final_accs.append(float(df['val_accuracy'].iloc[-1]))\n"
    "            bar_colors.append(colors.get(name, 'gray'))\n"
    "    elif name in all_data:\n"
    "        df = all_data[name]\n"
    "        if 'val_accuracy' in df.columns:\n"
    "            d = df.dropna(subset=['val_accuracy'])\n"
    "            if len(d) > 0:\n"
    "                method_names.append(label_text)\n"
    "                final_accs.append(float(d['val_accuracy'].iloc[-1]))\n"
    "                bar_colors.append(colors.get(name, 'gray'))\n"
    "\n"
    "if method_names:\n"
    "    bars = ax.bar(method_names, final_accs, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)\n"
    "    for bar, acc in zip(bars, final_accs):\n"
    "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,\n"
    "                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')\n"
    "    ax.set_ylim(0, max(final_accs) * 1.3 if final_accs else 100)\n"
    "ax.set_title('Final Validation Accuracy', fontsize=13)\n"
    "ax.set_ylabel('Accuracy (%)')\n"
    "ax.grid(True, alpha=0.3, axis='y')\n"
    "\n"
    "fig.suptitle(\n"
    "    f'Grokking Experiment: {hparams.math_operator} mod 97 | '\n"
    "    f'{N_MODELS} specialists | {FINAL_STEPS:,} steps budget',\n"
    "    fontsize=16, fontweight='bold'\n"
    ")\n"
    "plt.tight_layout()\n"
    "plt.savefig('/kaggle/working/grokking_dashboard.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: /kaggle/working/grokking_dashboard.png')"
))

cells.append(md("## Final Results Summary"))

cells.append(code(
    "print('=' * 60)\n"
    "print('FINAL RESULTS')\n"
    "print('=' * 60)\n"
    "\n"
    "for name, label_text in [('baseline', 'Baseline'), ('merged_average', 'Weight Averaged'), ('distilled', 'Distilled')]:\n"
    "    if name == 'distilled' and 'distilled_val' in all_data:\n"
    "        df = all_data['distilled_val']\n"
    "        if 'val_accuracy' in df.columns and len(df) > 0:\n"
    "            print(f'  {label_text:20s}: val_acc = {float(df[\"val_accuracy\"].iloc[-1]):.2f}%')\n"
    "    elif name in all_data:\n"
    "        df = all_data[name]\n"
    "        if 'val_accuracy' in df.columns:\n"
    "            d = df.dropna(subset=['val_accuracy'])\n"
    "            if len(d) > 0:\n"
    "                print(f'  {label_text:20s}: val_acc = {float(d[\"val_accuracy\"].iloc[-1]):.2f}%')\n"
    "\n"
    "print(f'Specialists:')\n"
    "for i in range(N_MODELS):\n"
    "    name = f'specialist_{i}'\n"
    "    if name in all_data:\n"
    "        df = all_data[name]\n"
    "        if 'val_accuracy' in df.columns:\n"
    "            d = df.dropna(subset=['val_accuracy'])\n"
    "            if len(d) > 0:\n"
    "                print(f'    specialist_{i}: val_acc = {float(d[\"val_accuracy\"].iloc[-1]):.2f}%')\n"
    "\n"
    "print(f'Plots saved to /kaggle/working/')\n"
    "print(f'  - grokking_dashboard.png (combined)')\n"
    "print(f'  - accuracy_curves.png')\n"
    "print(f'  - loss_curves.png')"
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('kaggle_grokking.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

# Verify
with open('kaggle_grokking.ipynb', 'r') as f:
    loaded = json.load(f)
print(f'Notebook: {len(loaded["cells"])} cells, valid JSON')

# Verify each code cell compiles
for i, cell in enumerate(loaded['cells']):
    if cell['cell_type'] == 'code':
        src = '\n'.join(cell['source'])
        try:
            compile(src, f'<cell {i}>', 'exec')
        except SyntaxError as e:
            print(f'  Cell {i+1} SYNTAX ERROR: {e}')
            print(f'    Source: {src[:200]}')

print('All code cells compile OK')
