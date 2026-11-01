$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1
docker compose -f docker\docker-compose.yml up -d
Start-Sleep -Seconds 15
python -m src.pipeline.run_offline --dataset paysim --attack none --defense off
python -m src.pipeline.kafka_stream --dataset paysim --attack none --defense off --check
# python -m src.attacks.sweep ...   (P2, added later)
docker compose -f docker\docker-compose.yml down