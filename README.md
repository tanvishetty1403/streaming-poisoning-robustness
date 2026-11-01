# Poisoning and Robustness in Streaming Classifiers

Simulates label-flip, feature-poison and backdoor attacks on a streaming fraud
classifier, and defends with statistical detection and quarantine.

## Setup (Python 3.11)

Windows (PowerShell):
```powershell
git clone https://github.com/tanvishetty1403/streaming-poisoning-robustness.git
cd streaming-poisoning-robustness
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Mac / Linux:
```bash
git clone https://github.com/tanvishetty1403/streaming-poisoning-robustness.git
cd streaming-poisoning-robustness
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data
1. Kaggle > Settings > Create New Token. Save `kaggle.json` to `~/.kaggle/kaggle.json`
   (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
2. Download PaySim (main) and ULB (cross-check) into `data/raw/`:
   - Mac / Linux: `bash scripts/download_data.sh`
   - Windows: `powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1`

## Run the offline loop
```
python -m src.pipeline.run_offline --dataset paysim --attack none --defense off
python -m src.pipeline.run_offline --dataset paysim --attack flip_targeted --budget 0.5 --timing burst --defense on
```
Results are appended to `results/metrics.csv`.

## Tests
```
python -m pytest -q
```

## Layout
- `contracts/CONTRACTS.md`: interfaces every module must follow
- `config/default.yaml`: all settings
- `src/pipeline/`: replay, learner, metrics, offline loop, Kafka (P1)
- `src/attacks/`: attacks, sweep runner (P2)
- `src/defense/`: detectors, quarantine, audit log (P3)
- `docker/`: Kafka via Docker Compose
- `tests/`: unit tests

## Team rules
No contract change without all three agreeing. One PR per feature, one reviewer.