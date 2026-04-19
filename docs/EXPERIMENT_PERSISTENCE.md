# Experiment outputs and persistence

This document describes **what gets written to disk** when you run the multi-model pipelines in `grok/multi_training.py`. Existing behavior (Lightning logs, `init.pt`, `merged_weights.pt`, distillation CSVs, legacy `comparison_results.json` fields) is **unchanged**; the entries below are **additive**.

## Layout

All paths are relative to your experiment directory:

`{logdir}/{experiment_name}/`  

(e.g. `logs/bagging_ema_distill/` with default `--logdir logs`.)

---

## Shared (both `train_multi` and `train_multi_with_distillation`)

### `{experiment}/artifacts/run_config.json`

Full CLI / `Namespace` hyperparameters as JSON-serializable data (reproducibility).

### `{experiment}/artifacts/environment.json`

- `torch_version`, `pytorch_lightning_version`
- `cuda_available`, `mps_available`
- `git_commit` (if `git rev-parse HEAD` succeeds from the process cwd; else `null`)

**Nothing here replaces** your manual notes or conda env exports; it is a lightweight fingerprint.

---

## `scripts/train_multi.py` without `--use_distillation` (`train_multi`)

### Checkpoints (new files)

| Path | Contents |
|------|-----------|
| `specialist_{i}/artifacts/specialist_final.pt` | `transformer_state_dict` + `hparams` snapshot after that specialist finishes |
| `merged/artifacts/merged_final.pt` | Same, after merged-model fine-tuning |
| `baseline/artifacts/baseline_final.pt` | Same, after baseline run (if `--run_baseline`) |

Existing files **`checkpoints/init.pt`**, **`merged_weights.pt`**, and Lightning **`epoch_*.ckpt`** (power-of-2 epochs) are **untouched**.

### `{experiment}/artifacts/train_multi_summary.json`

Pointers to checkpoints, Lightning `metrics.csv` paths (when present), and best-effort **`val_accuracy` max** from each phase’s CSV (`merged` / `baseline`).

---

## `scripts/train_multi.py` with `--use_distillation` (`train_multi_with_distillation`)

### Checkpoints (new files)

| Path | Contents |
|------|-----------|
| `specialist_{i}/artifacts/specialist_final.pt` | After each specialist |
| `merged_average/artifacts/merged_final.pt` | After weight-averaged model fine-tuning |
| `distilled/artifacts/student_final.pt` | After distillation |
| `baseline/artifacts/baseline_final.pt` | After baseline (if enabled) |

**Loading example** (PyTorch):

```python
import torch
from grok.training import TrainableTransformer
# build model with same hparams, then:
ckpt = torch.load(".../student_final.pt", map_location="cpu")
model.transformer.load_state_dict(ckpt["transformer_state_dict"])
```

### `distill_from_specialists` metrics (additive keys)

The returned `distill_metrics` dict still includes `loss`, `soft_loss`, … and now also:

- `final_val_acc` — full-val RHS accuracy after the last distill epoch  
- `best_val_acc` — max over sampled `val_acc` during distill (or final if none sampled)

Existing **`distill_metrics.csv`** and **`distill_val_metrics.csv`** are still written as before.

### `comparison_results.json` (extended, backward compatible)

Original keys are preserved: `n_models`, `specialist_steps`, `final_steps`, `distill_steps`, `temperature`, `alpha`, `distill_final_metrics`, `experiment_dir`.

**Additional keys:**

| Key | Meaning |
|-----|---------|
| `run_config_path` | Absolute path to `artifacts/run_config.json` |
| `environment_path` | Absolute path to `artifacts/environment.json` |
| `teacher_aggregation` | `"probs"` (default) for `Q = (1/M) Σ softmax(z_i/T)` or `"logits"` for `Q = softmax((1/M) Σ z_i/T)`. Selected via `--teacher_aggregation`. The `"logits"` mode reproduces the milestone-era distillation target shape (pre-commit `9ce29b8`). See `reports/distillation_repro_attempt3.md` for context on why this matters. |
| `persisted_checkpoints` | Absolute paths: `specialists[]`, `merged_average`, `distilled_student`, `baseline` (or `null` if skipped) |
| `lightning_metrics_csv_paths` | Paths to Lightning `metrics.csv` per phase + distill CSVs (`distilled`, `distilled_val`) |
| `trainer_callback_metrics` | Snapshot of `trainer.callback_metrics` after **merged** and **baseline** (baseline `null` if `--no_baseline`) |
| `best_val_from_lightning_csv` | Best-effort max `val_accuracy` (+ step/column) from merged/baseline Lightning CSVs |
| `distillation_extended` | `final_val_accuracy`, `best_val_accuracy`, paths to distill CSVs |

> **Caveat — distillation student optimizer is not yet recorded.** The
> distillation student currently builds its optimizer via
> `student_model.configure_optimizers()`, which inherits `weight_decay`,
> `max_lr`, `weight_decay_kind`, `anneal_lr*` and friends from the run
> hparams (and therefore from `run_config.json`). Older code at commit
> `09befb8` instead hardcoded `AdamW(lr=1e-3, wd=1.0) +
> CosineAnnealingLR(eta_min=1e-5)` for the student only, ignoring those
> hparams. Until a `--distill_optimizer {hparams,milestone}` flag is
> wired in (see `reports/distillation_repro_attempt3.md` §4–§5), the
> per-phase optimizer recipe is implicit and cross-version comparisons of
> distilled-student curves require knowing which commit the run came
> from.

### `{experiment}/artifacts/paths_index.json`

Short index: `comparison_results_json`, `run_config_json`, `environment_json`, `checkpoints`, `metrics_csv`.

---

## What is **not** duplicated

- **Lightning** `lightning_logs/version_*/metrics.csv` — still the source of truth for per-step training curves.  
- **`checkpoints/init.pt`** per phase — still saved as before.  
- **Distillation CSVs** — same filenames and columns as previously.  
- **`merged_weights.pt`** — still written only in `train_multi` (non-distill path).

---

## Smoke test

A minimal run that exercises persistence:

```bash
PYTHONPATH=. python scripts/train_multi.py \
  --n_models 2 --train_data_pct 50 \
  --specialist_steps 30 --final_steps 30 \
  --use_distillation --distill_steps 30 \
  --ema_decay 0.99 --experiment_name persist_smoke --no_baseline
```

Verify: `comparison_results.json`, `artifacts/run_config.json`, `artifacts/environment.json`, `artifacts/paths_index.json`, and `*/artifacts/*.pt` exist under `logs/persist_smoke/`.
