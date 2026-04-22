# Multi-specialist consistency regularization for grokking on modular arithmetic

This repository hosts the code and experimental artefacts for our final
project on the [Grokking](https://arxiv.org/abs/2201.02177) setup
(Power et al., 2022). The final write-up is
[`reports/final_report.pdf`](reports/final_report.pdf).

The short version: a four-specialist ensemble with train-inputs-only
KL-softmax consistency regularization groks modular addition at 50%
train data (100% ensemble validation accuracy at 100k steps), given
`max_lr=1e-3`. See §10 of the report for the key isolation control
(`lr_control/v4_winner_lr1e3`). The long version is how we ended up
with a misleading 48-49% "architectural ceiling" first.

## Repository layout

| path | what |
|---|---|
| `grok/` | Model, data pipeline, multi-specialist + consistency training (`grok/consistency_training.py`) |
| `scripts/run_consistency_sweep.py` | Sweep driver used by every consistency pilot |
| `scripts/run_consistency_sweep.py --help` | All CLI flags |
| `notebooks/kaggle_consistency_pilot.ipynb` | Kaggle runner for pilots v2-v4 (at older commits) and the lr control (at `HEAD`) |
| `notebooks/kaggle_infonce_pilot.ipynb` | Kaggle runner for pilot v5 (InfoNCE) |
| `notebooks/kaggle_loss_control.ipynb` | Kaggle runner for the loss control (§10) |
| `reports/final_report.pdf` | The final write-up |
| `reports/final_report.tex` | LaTeX source |
| `reports/final_figures/` | Figures and the scripts that produce them |
| `reports/milestone.pdf` | The pre-pivot milestone we inherited and disproved |
| `reports/distillation_repro_*.md` | Three post-mortems on the milestone reproduction (ignored by git, see below) |
| `reports/consistency_regularized_grokking_plan.md` | Full experimental plan for the consistency pivot |

## Environment

Python 3.10+ with `pip install -e .`. GPU strongly recommended
(PyTorch with CUDA); all pilots were run on a single Kaggle T4.

On Kaggle the notebooks install the repo, pin a commit, and run one of
the sweeps. Locally you can do the same:

```bash
conda create -n grok python=3.10 -y
conda activate grok
pip install -e .
# GPU build of torch for your CUDA (example: cu118)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Reproducing the final finding (fastest path)

The cleanest single comparison in the whole project is the **lr
control** (§10 of the report). One sweep cell, ~4h on a T4, grokks to
100%:

```bash
python scripts/run_consistency_sweep.py \
    --sweep_name lr_control \
    --logdir lr_control \
    --gpu 0 \
    --n_models 4 \
    --sharding disjoint \
    --train_data_pct 50 \
    --max_lr 1e-3 \
    --weight_decay 0.1 \
    --shard_batch_size 256 \
    --consistency_batch_size 256 \
    --consistency_steps 100000 \
    --eval_every 500 \
    --log_every 50 \
    --checkpoint_every 0 \
    --cells_json '[{"name":"v4_winner_lr1e3","random_seed":44,"consistency_loss":"kl_softmax","consistency_lambda":1.0,"consistency_warmup_steps":5000,"consistency_domain":"train_inputs_only","consistency_steps":100000}]'
```

For the InfoNCE-recipe loss control (§10, second row of Table 4),
same budget, use `notebooks/kaggle_loss_control.ipynb` (it carries the
right hparams + commit pin) or pass
`--consistency_loss kl_softmax --n_models 2 --max_lr 1e-3 --shard_batch_size 512 --consistency_batch_size 32`
and a single cell with `consistency_lambda=5`, `consistency_warmup_steps=0`.

## Reproducing the historical pilots

Each pilot was run from the Kaggle notebook pinned to a specific
commit. The code evolved across the sweeps (e.g. InfoNCE support was
added after v4), so to reproduce an older pilot exactly you must
check out the commit that pilot ran at. The table below lists the
commit, the sweep name, and the notebook.

| pilot | section | commit | sweep name | notebook |
|---|---|---|---|---|
| v1   | §5  | (pre-`e48e235`)            | `pilot_v1`       | n/a (superseded) |
| v2   | §6  | `c84898e`                  | `pilot_v2`       | `notebooks/kaggle_consistency_pilot.ipynb` |
| v3   | §7  | `23d4c21`                  | `pilot_v3`       | `notebooks/kaggle_consistency_pilot.ipynb` |
| v4   | §8  | `6a85fc0` (pins `e1e74e6`) | `pilot_v4`       | `notebooks/kaggle_consistency_pilot.ipynb` |
| v5 (InfoNCE)   | §9  | `7b74aff`     | `infonce_pilot_v3`         | `notebooks/kaggle_infonce_pilot.ipynb` |
| lr control     | §10 | `7b74aff`     | `lr_control`               | `notebooks/kaggle_consistency_pilot.ipynb` |
| loss control   | §10 | `7b74aff`     | `loss_control_kl`          | `notebooks/kaggle_loss_control.ipynb` |

To check out any of these pinned commits and run from that exact
snapshot:

```bash
git checkout <commit>        # e.g. 23d4c21 for pilot v3
pip install -e .              # re-install in case grok/ changed
# then open the relevant notebook and run, or invoke the sweep CLI
```

The Kaggle notebooks already include the `git checkout <commit>` step
inline (cell 2 of each notebook sets `GROK_COMMIT`). Running them on
Kaggle as-is reproduces the historical pilot exactly.

Returning to the current `main` afterwards:

```bash
git checkout main
pip install -e .
```

## Sweep outputs

Pilot output folders (`pilot_v*/`, `infonce_pilot_v3/`, `lr_control/`,
`loss_control_kl/`) and their zipped Kaggle archives are **not
committed** -- they are large (per-step CSVs + optional checkpoints)
and can be regenerated by rerunning the sweeps above. Figures in
`reports/final_figures/` are derived from these folders by
`make_figs.py`, `make_infonce_figs.py`, and `make_controls_figs.py`.

If you want to regenerate every figure from cached sweep outputs, put
the unzipped sweep folders at the repo root and run:

```bash
python reports/final_figures/make_figs.py
python reports/final_figures/make_infonce_figs.py
python reports/final_figures/make_controls_figs.py
```

## Rebuilding the report

```bash
cd reports
pdflatex -interaction=nonstopmode final_report.tex
pdflatex -interaction=nonstopmode final_report.tex  # second pass for refs
```

Requires a TeX Live with `microtype`, `lmodern`, `enumitem`,
`booktabs`, `hyperref`, `subcaption` (all in `texlive-latex-extra`).

## Legacy / pre-pivot code

The milestone code path (knowledge distillation from specialists) is
in the repo history before the consistency pivot -- see commits
prior to `e48e235`. The milestone itself is `reports/milestone.pdf`
and was disproved in `reports/distillation_repro_attempt3.md`; those
post-mortems live locally but are gitignored (they contain a lot of
log dumps). See §2 of the final report for the summary.

Earlier scripts that are still around but no longer relevant:
`scripts/train.py`, `scripts/train_multi.py`, and the Windows
helper shells (`scripts/run_*.ps1`) were used for the pre-pivot
baselines. They are left in place so that older commits remain
runnable.
