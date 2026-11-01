#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash scripts/download_data.sh
docker compose -f docker/docker-compose.yml up -d
sleep 15
python -m src.pipeline.run_offline --dataset paysim --attack none --defense off
python -m src.pipeline.kafka_stream --dataset paysim --attack none --defense off --check
# python -m src.attacks.sweep ...   (P2, added later)
docker compose -f docker/docker-compose.yml down