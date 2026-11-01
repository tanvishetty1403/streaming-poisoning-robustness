#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

command -v kaggle >/dev/null || { echo "Run: pip install -r requirements.txt"; exit 1; }
[ -f "$HOME/.kaggle/kaggle.json" ] || { echo "Missing ~/.kaggle/kaggle.json (see README)"; exit 1; }

echo "Downloading PaySim..."
kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip
mv -f data/raw/PS_*.csv data/raw/paysim.csv

echo "Downloading ULB credit card fraud (cross-check)..."
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

echo "Done:"; ls -lh data/raw