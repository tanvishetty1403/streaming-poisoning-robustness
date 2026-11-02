#!/usr/bin/env bash
# Detector ablation at budget 0.2 over the attack seeds (resumable). Pieces land in results/abl/<variant>_s<seed>.csv.
#   bash scripts/run_ablation.sh      then      python scripts/collect_results.py
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/abl
run() {  # name, then tune_defense args
  local name=$1; shift
  for s in ${SEEDS:-0 1 2}; do
    out=results/abl/${name}_s${s}.csv
    [ -s "$out" ] && continue
    python scripts/tune_defense.py --budgets 0.2 --seeds $s "$@" --out "$out.tmp" > /dev/null && mv "$out.tmp" "$out"
    echo "done $name seed $s"
  done
}
run robust_z --detectors robust_z
run label_loss --detectors label_loss
run mahalanobis --detectors mahalanobis
run robust_z+label_loss --detectors robust_z label_loss
run all_noalarm --set drift_alarm=false
