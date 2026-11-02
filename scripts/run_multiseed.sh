#!/usr/bin/env bash
# Defense detection metrics for every attack x seed, one small CSV each (resumable: finished pieces are skipped).
#   bash scripts/run_multiseed.sh [extra tune_defense args, e.g. --set drift_alarm=false]
# then: python scripts/collect_results.py
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/ms
for atk in random_flip targeted_flip feature_shift feature_scale feature_noise backdoor; do
  for s in ${SEEDS:-0 1 2}; do
    out=results/ms/${atk}_s${s}.csv
    [ -s "$out" ] && continue
    python scripts/tune_defense.py --attacks $atk --seeds $s --subgroups "$@" --out "$out.tmp" > /dev/null && mv "$out.tmp" "$out"
    echo "done $atk seed $s"
  done
done
