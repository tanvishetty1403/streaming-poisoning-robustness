#!/usr/bin/env python
"""Collect the raw tuning / utility / ablation outputs into three small tidy files under results/summary/.

Run after:
  python scripts/tune_defense.py --subgroups --out results/final_default.csv
  python scripts/eval_utility.py --budgets 0.05 --out results/utility_b005.csv
  python scripts/eval_utility.py --budgets 0.2  --out results/utility_b02.csv
  ablation: python scripts/tune_defense.py --budgets 0.2 --detectors <names> --out results/abl_<names joined by +>.csv
            (and --set drift_alarm=false for the no-alarm row, saved as results/abl_all_noalarm.csv)
The summary files are committed (results/*.csv is otherwise gitignored) so the results notebook can be
re-run without the 600 MB attacked samples.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "summary"
OUT.mkdir(parents=True, exist_ok=True)

det = pd.read_csv(R / "final_default.csv")
keep = ["file", "attack", "budget", "timing", "seed", "n_poisoned", "n_flagged", "precision", "recall",
        "clean_loss", "clean_fraud_loss", "time_to_detect"] + [c for c in det.columns if c.startswith("loss_")]
det[keep].to_csv(OUT / "detection_seed0.csv", index=False)

util = pd.concat([pd.read_csv(R / "utility_b005.csv"), pd.read_csv(R / "utility_b02.csv")], ignore_index=True)
util.to_csv(OUT / "utility_seed0.csv", index=False)

names = {"robust_z": "robust_z only", "label_loss": "label_loss only", "mahalanobis": "mahalanobis only",
         "robust_z+label_loss": "robust_z + label_loss", "all": "all three + alarm", "all_noalarm": "all three, no alarm"}
rows = []
for k, label in names.items():
    d = pd.read_csv(R / f"abl_{k}.csv")
    d = d[["file", "attack", "budget", "timing", "recall", "precision", "clean_loss", "clean_fraud_loss"]].assign(variant=label)
    rows.append(d)
pd.concat(rows, ignore_index=True).to_csv(OUT / "ablation_seed0.csv", index=False)
print("wrote", sorted(p.name for p in OUT.glob("*.csv")))
