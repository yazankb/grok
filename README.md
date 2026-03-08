# OpenAI Grok Curve Experiments

## Paper

This is the code for the paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) by Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra

## Installation and Training

```bash
conda activate grok
pip install -e .
python scripts/train.py
```

If `conda` isn’t available in your terminal (e.g. Cursor): run ` . .\init_conda.ps1` once, then the commands above.

## Running on GPU

Training uses the GPU when PyTorch has CUDA support and you pass a non-negative GPU index (default is `--gpu 0`).

1. **Install PyTorch with CUDA** (required for GPU). Use the official wheel index for your CUDA version:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   Use `cu118` for CUDA 11.8 or `cu126` for CUDA 12.6. Without this, `torch.cuda.is_available()` is false and training runs on CPU.

2. **Run training on GPU:**

   ```bash
   python scripts/train.py --gpu 0
   ```

   On Windows with Anaconda (if `python` is not in PATH):

   ```powershell
   & "C:\anaconda3\python.exe" scripts/train.py --gpu 0
   ```

   You should see `Using GPU (cuda:0)` and Lightning will report `GPU available: True`. Use `--gpu -1` or a negative value to force CPU.

3. **Optional:** `scripts/run_train_gpu.ps1` checks that CUDA is available, then runs training with `--gpu 0`.

## Multi-model baseline

To run the multi-model grokking experiment (specialists on shards, weight averaging, merged vs baseline):

```bash
python scripts/train_multi.py --train_data_pct 50 --equal_compute --final_steps 20000 --experiment_name multi_fixed
```

Or use `scripts/run_multi_fixed.ps1` (Windows). After training, run `scripts/run_merged_baseline.py` for the 5k merged/baseline comparison; `scripts/plot_multi_fixed_5k.py` plots the curves. The write-up is in `reports/YazanKbaili_AliSalloum.pdf`.

## Requirements

- **torchvision**: Must be at least 0.13 so that `VGG16_Weights` is available (used by `torchmetrics` via PyTorch Lightning). Older versions (e.g. 0.2.0) cause:

  ```
  ImportError: cannot import name 'VGG16_Weights' from 'torchvision.models'
  ```

  If you see this, upgrade: `pip install "torchvision>=0.13"`.
