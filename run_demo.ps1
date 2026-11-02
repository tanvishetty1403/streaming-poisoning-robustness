# One command for the whole demo (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File run_demo.ps1          full demo
#   powershell -ExecutionPolicy Bypass -File run_demo.ps1 quick    smoke test
#   powershell -ExecutionPolicy Bypass -File run_demo.ps1 kafka    also Kafka benchmark (Docker)
param([string]$Mode = "full")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path data/raw/paysim.csv)) { powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1 }

if ($Mode -eq "quick") { python scripts/sweep.py --fresh --quick } else { python scripts/sweep.py --fresh }
python scripts/plots.py
python scripts/fairness.py --attack targeted_flip --budget 0.5 --timing burst

if ($Mode -eq "kafka") {
  docker compose -f docker/docker-compose.yml up -d
  Start-Sleep -Seconds 20
  python -m src.pipeline.kafka_stream --dataset paysim --attack targeted_flip --budget 0.5 --timing burst --defense on --check
  python scripts/benchmark.py --partitions 1,2,3
  docker compose -f docker/docker-compose.yml down
}
Write-Host "Done. See results/: metrics.csv summary.csv fig_*.png fairness.csv audit_log.csv benchmark.csv"
