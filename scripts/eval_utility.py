#!/usr/bin/env python
"""Does the defense protect the learner? Final PR-AUC / recall on the clean test set for
(a) the clean stream, (b) the attacked stream with no defense, (c) the attacked stream through defend.

  python scripts/eval_utility.py --attacks random_flip --budgets 0.2
Offline only: attacked files carry is_poisoned, which is stripped before defend. Uses P1's OnlineLearner,
replay windows and eval_clean, so numbers match the offline loop for the same files.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.attacks import load_config  # noqa: E402
from src.defense.pipeline import load_defense_cfg  # noqa: E402
from src.pipeline.data import featurize  # noqa: E402
from src.pipeline.learner import OnlineLearner  # noqa: E402
from src.pipeline.metrics import eval_clean  # noqa: E402
from tune_defense import parse_name, run_file  # noqa: E402


def train_eval(df, keep, ws, test, lcfg, seed=0):
    """Replay windows, fit the learner on rows where keep is True, evaluate once on the clean test set."""
    df = df.reset_index(drop=True)
    win = ((df["step"] - df["step"].min()) // ws).to_numpy()
    learner = OnlineLearner(lcfg["fraud_weight"], seed)
    for k in np.unique(win):
        idx = np.flatnonzero((win == k) & keep)
        if len(idx):
            w = df.iloc[idx]
            learner.partial_fit(featurize(w, "paysim"), w["isFraud"].to_numpy())
    return eval_clean(learner, featurize(test, "paysim"), test["isFraud"].to_numpy(), lcfg["threshold"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "data" / "attacked_sample"))
    ap.add_argument("--attacks", nargs="+")
    ap.add_argument("--budgets", nargs="+", type=float)
    ap.add_argument("--timings", nargs="+")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="override defense: keys, e.g. alarm_tighten=2")
    ap.add_argument("--out", default=str(ROOT / "results" / "utility.csv"))
    a = ap.parse_args()
    config = load_config()
    ws, lcfg, dcfg = config["datasets"]["paysim"]["window_steps"], config["learner"], load_defense_cfg()
    import yaml
    for kv in a.set:
        k, v = kv.split("=", 1)
        dcfg[k] = yaml.safe_load(v)
    d = Path(a.dir)
    test = pd.read_csv(d / "test_clean.csv.gz")
    clean = pd.read_csv(d / "stream_clean.csv.gz")
    ref = train_eval(clean, np.ones(len(clean), dtype=bool), ws, test, lcfg)
    print(f"clean stream, no attack: pr_auc={ref['pr_auc']:.3f} recall={ref['recall']:.3f}", flush=True)
    rows = []
    for f in sorted(d.glob("*_s0.csv.gz")):
        m = parse_name(f.name.replace(".csv.gz", ""))
        if (a.attacks and m["attack"] not in a.attacks) or (a.budgets and float(m["budget"]) not in a.budgets) \
                or (a.timings and m["timing"] not in a.timings):
            continue
        df = pd.read_csv(f)
        undef = train_eval(df, np.ones(len(df), dtype=bool), ws, test, lcfg)
        sdf, flagged, _ = run_file(df, dcfg, ws)
        defended = train_eval(sdf, ~flagged, ws, test, lcfg)
        rows.append({"file": f.name, **m, "pr_auc_clean": ref["pr_auc"], "pr_auc_no_defense": undef["pr_auc"],
                     "pr_auc_defended": defended["pr_auc"], "recall_clean": ref["recall"],
                     "recall_no_defense": undef["recall"], "recall_defended": defended["recall"]})
        r = rows[-1]
        print(f"{f.name:40s} pr_auc  clean {ref['pr_auc']:.3f} | none {r['pr_auc_no_defense']:.3f} | defended {r['pr_auc_defended']:.3f}"
              f"   recall  none {r['recall_no_defense']:.2f} | defended {r['recall_defended']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
