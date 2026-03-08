# Run training on GPU. Ensure PyTorch with CUDA is installed first:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$ErrorActionPreference = "Stop"
$python = "C:\anaconda3\python.exe"
$root = Split-Path $PSScriptRoot -Parent
if (Test-Path "c:\anaconda3\python.exe") { $python = "c:\anaconda3\python.exe" }
Set-Location $root
& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available. Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118'"
Write-Host "CUDA available. Starting training on GPU..."
& $python scripts/train.py --gpu 0
