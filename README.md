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
python -m src.pipeline.run_offline --dataset paysim --attack random_flip --budget 0.2 --timing uniform --defense on
```
Attacks: `random_flip`, `targeted_flip`, `feature_shift`, `feature_scale`, `feature_noise`, `backdoor`;
timings: `uniform`, `burst`, `low_and_slow`. Results are appended to `results/metrics.csv`.

## Run the full experiment (3 seeds, defense off and on)
Needs `data/attacked_sample/stream_clean.csv.gz` and `test_clean.csv.gz` (`scripts/make_attacked_sample.py`), or the raw data.
```
python scripts/extend_attacked_samples.py --seeds 0 1 2            # attacked files for seeds 0 to 2
python -m src.attacks.sweep --clean-dir data/attacked_sample       # results/metrics.csv (resumable)
bash scripts/run_multiseed.sh                                      # detection, clean-record loss, subgroups
bash scripts/run_ablation.sh                                       # detector ablation
python scripts/collect_results.py                                  # results/summary/*.csv
python scripts/build_notebook.py                                   # notebooks/defense_results.ipynb
```

## Defense results
`notebooks/defense_results.ipynb` (PR-AUC with and without the defense, detector ablation, clean-record loss by group),
`docs/model_data_card.md` (model and data card), `docs/report_defense_section.md` (the defense section of the report).

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