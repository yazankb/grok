# Run once per terminal:  . .\init_conda.ps1   then:  conda activate grok
$condaExe = $null
foreach ($base in @("C:\anaconda3", "$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda", "$env:USERPROFILE\anaconda")) {
    $c = Join-Path $base "Scripts\conda.exe"
    if (Test-Path $c) { $condaExe = $c; break }
}
if (-not $condaExe) {
    Write-Host "Conda not found in user folder. Set CONDA_EXE or edit init_conda.ps1 with your conda path."
    return
}
& $condaExe "shell.powershell" "hook" | Out-String | Invoke-Expression
Write-Host "Conda loaded. Run: conda activate grok"
