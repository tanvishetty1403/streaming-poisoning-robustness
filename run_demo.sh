#!/usr/bin/env bash
# One command for the whole demo (Mac/Linux/Git-Bash).
#   bash run_demo.sh            full demo grid + figures + fairness  (no Kafka)
#   bash run_demo.sh quick      tiny grid, ~2 min, for a smoke test
#   bash run_demo.sh kafka      also run the Kafka benchmark (needs Docker)
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-full}"

[ -f data/raw/paysim.csv ] || bash scripts/download_data.sh

if [ "$MODE" = "quick" ]; then
  python scripts/sweep.py --fresh --quick
else
  python scripts/sweep.py --fresh
fi
python scripts/plots.py
python scripts/fairness.py --attack targeted_flip --budget 0.5 --timing burst

if [ "$MODE" = "kafka" ]; then
  docker compose -f docker/docker-compose.yml up -d
  sleep 20
  python -m src.pipeline.kafka_stream --dataset paysim --attack targeted_flip --budget 0.5 --timing burst --defense on --check
  python scripts/benchmark.py --partitions 1,2,3
  docker compose -f docker/docker-compose.yml down
fi
echo "Done. See results/: metrics.csv summary.csv fig_*.png fairness.csv audit_log.csv benchmark.csv"
