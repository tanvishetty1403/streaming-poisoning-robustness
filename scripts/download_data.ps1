$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force -Path data\raw | Out-Null

if (-not (Test-Path "$HOME\.kaggle\kaggle.json")) { throw "Missing $HOME\.kaggle\kaggle.json (see README)" }

Write-Host "Downloading PaySim..."
kaggle datasets download -d ealaxi/paysim1 -p data\raw --unzip
$ps = Get-ChildItem data\raw\PS_*.csv | Select-Object -First 1
Move-Item -Force $ps.FullName data\raw\paysim.csv

Write-Host "Downloading ULB credit card fraud (cross-check)..."
kaggle datasets download -d mlg-ulb/creditcardfraud -p data\raw --unzip

Get-ChildItem data\raw