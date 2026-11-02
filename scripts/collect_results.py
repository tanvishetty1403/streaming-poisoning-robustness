#!/usr/bin/env python
"""Collect the raw multi-seed outputs into three small tidy files under results/summary/.

Run after (each step is resumable):
  python -m src.attacks.sweep --clean-dir data/attacked_sample     # results/metrics.csv (contract metrics, defense off/on)
  bash scripts/run_multiseed.sh                                    # results/ms/*.csv   (detection, clean-record loss, subgroups)
  bash scripts/run_ablation.sh                                     # results/abl/*.csv  (detector ablation, budget 0.2)
The summary files are committed (results/*.csv is otherwise gitignored) so the results notebook can be
re-run without the 600 MB attacked samples.

  utility.csv    one row per (attack, budget, timing, seed, defense): final PR-AUC / recall / F1 on the clean test set
  detection.csv  one row per attacked file: precision, recall, clean_loss, clean_fraud_loss, time_to_detect, subgroup loss
  ablation.csv   one row per (variant, attacked file) at budget 0.2
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "summary"
OUT.mkdir(parents=True, exist_ok=True)


def cat(pattern):
    files = sorted(R.glob(pattern))
    if not files:
        raise SystemExit(f"nothing matches results/{pattern}: run the step listed in the docstring first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# ---- utility: last window of each run in the contract metrics file ----
m = pd.read_csv(R / "metrics.csv")
last = m.sort_values("window").groupby(["dataset", "attack", "budget", "timing", "defense", "seed"], as_index=False).tail(1)
keep = ["attack", "budget", "timing", "seed", "defense", "pr_auc", "recall", "f1", "backdoor_hit_rate", "time_to_detect"]
last[keep].sort_values(keep[:5]).to_csv(OUT / "utility.csv", index=False)

# ---- detection ----
det = cat("ms/*.csv")
cols = ["file", "attack", "budget", "timing", "seed", "n_poisoned", "n_flagged", "precision", "recall", "clean_loss",
        "clean_fraud_loss", "time_to_detect"] + [c for c in det.columns if c.startswith("loss_type=") or c.startswith("loss_amount_bucket=")]
det[cols].to_csv(OUT / "detection.csv", index=False)

# ---- ablation (budget 0.2) ----
names = {"robust_z": "robust_z only", "label_loss": "label_loss only", "mahalanobis": "mahalanobis only",
         "robust_z+label_loss": "robust_z + label_loss", "all_noalarm": "all three, no alarm"}
rows = [cat(f"abl/{k}_s*.csv").assign(variant=label) for k, label in names.items()]
rows.append(det[det["budget"] == 0.2].assign(variant="all three + alarm"))
ab = pd.concat(rows, ignore_index=True)[["file", "attack", "budget", "timing", "seed", "recall", "precision", "clean_loss", "clean_fraud_loss", "variant"]]
ab.to_csv(OUT / "ablation.csv", index=False)
print({p.name: len(pd.read_csv(p)) for p in sorted(OUT.glob("*.csv"))})
