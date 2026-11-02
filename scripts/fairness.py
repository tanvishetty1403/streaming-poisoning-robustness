"""Audit log + fairness check.

Runs one attacked, defended run window by window, saves every quarantine decision
(results/audit_log.csv) and reports, per subgroup, how many CLEAN records the defense
quarantined (results/fairness.csv). Ground truth is read here, in the evaluator, only.

    python scripts/fairness.py --attack targeted_flip --budget 0.5 --timing burst
"""
import argparse
import os
import sys
import uuid

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline.core import StreamRun, prepare_run                          # noqa: E402
from src.pipeline.replay import replay_windows                               # noqa: E402


def amount_bucket(amount):
    return pd.cut(amount, [-1, 1_000, 10_000, 100_000, 1_000_000, np.inf],
                  labels=["<1k", "1k-10k", "10k-100k", "100k-1M", ">1M"]).astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="paysim")
    ap.add_argument("--attack", default="targeted_flip")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--timing", default="burst")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="config/default.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    poisoned, test, dcfg = prepare_run(a.dataset, a.attack, a.budget, a.timing, a.seed, cfg)
    run = StreamRun(a.dataset, test, cfg, a.attack, a.budget, a.timing, "on", a.seed, uuid.uuid4().hex[:8])
    for k, win in replay_windows(poisoned, dcfg["window_steps"]):
        run.step(k, win)

    log = pd.DataFrame(run.state["quarantine"].log)
    os.makedirs("results", exist_ok=True)
    if log.empty:
        print("Defense quarantined nothing; no audit log written.")
        return
    info = poisoned[["record_id", "step", "type", "amount", "isFraud", "is_poisoned"]] \
        if "type" in poisoned.columns else poisoned[["record_id", "step", "Amount", "isFraud", "is_poisoned"]]
    audit = log.merge(info, on="record_id", how="left")
    audit.to_csv("results/audit_log.csv", index=False)

    df = poisoned.copy()
    amt = df["amount"] if "amount" in df.columns else df["Amount"]
    df["amount_bucket"] = amount_bucket(amt)
    df["held"] = df["record_id"].isin(log.loc[log["event"] == "held", "record_id"])
    clean = df[~df["is_poisoned"]]
    # records in the trusted warm-up window(s) are never judged; exclude them so rates are not diluted
    first = clean["step"].min() + dcfg["window_steps"] * cfg["defense"]["trusted_windows"]
    clean = clean[clean["step"] >= first]

    groups = ["type", "amount_bucket"] if "type" in clean.columns else ["amount_bucket"]
    rows = []
    for g in groups:
        for val, sub in clean.groupby(g):
            rows.append({"group": g, "value": val, "clean_seen": len(sub),
                         "clean_quarantined": int(sub["held"].sum()),
                         "clean_loss_rate": float(sub["held"].mean()),
                         "fraud_seen": int((sub["isFraud"] == 1).sum()),
                         "true_fraud_quarantined": int(sub.loc[sub["isFraud"] == 1, "held"].sum())})
    fair = pd.DataFrame(rows)
    fair.to_csv("results/fairness.csv", index=False)

    overall = clean["held"].mean()
    print(f"Attack={a.attack} budget={a.budget} timing={a.timing}")
    print(f"Quarantined: {int(df['held'].sum())} records; clean records lost: {overall:.2%}")
    print(fair.to_string(index=False))
    for g in groups:
        s = fair[fair["group"] == g]
        s = s[s["clean_seen"] >= 50]
        if len(s) > 1:
            print(f"Largest gap across {g}: {s['clean_loss_rate'].max() - s['clean_loss_rate'].min():.2%} "
                  f"({s.loc[s['clean_loss_rate'].idxmax(), 'value']} vs {s.loc[s['clean_loss_rate'].idxmin(), 'value']})")
    print("\nAudit log sample (results/audit_log.csv):")
    print(audit.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
