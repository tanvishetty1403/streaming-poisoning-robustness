import argparse
import os
import uuid

import numpy as np
import pandas as pd
import yaml

from src.attacks.stub import no_attack
from src.defense.stub import no_defense
from src.pipeline.data import load_dataset, split_stream_test, featurize
from src.pipeline.learner import OnlineLearner
from src.pipeline.metrics import eval_clean, eval_detection
from src.pipeline.replay import replay_windows

COLUMNS = ["run_id", "dataset", "attack", "budget", "timing", "defense", "seed", "window",
           "step_end", "n_seen", "n_quarantined", "pr_auc", "recall", "f1",
           "det_precision", "det_recall", "det_fpr", "time_to_detect", "backdoor_hit_rate"]

ATTACKS = ["none", "random_flip", "targeted_flip", "feature_shift", "feature_scale",
           "feature_noise", "backdoor"]


def apply_attack_by_name(stream, attack, budget, timing, seed, dataset):
    if attack == "none":
        return no_attack(stream)
    from src.attacks.api import apply_attack        # P2's entry point (per contract)
    return apply_attack(stream, attack_name=attack, budget=budget, timing=timing,
                        seed=seed, dataset=dataset)


def get_defense(name):
    if name == "off":
        return no_defense
    from src.defense.pipeline import defend         # P3's entry point (per contract)
    return defend


def run_offline(dataset, attack="none", budget=0.0, timing="uniform", defense="off",
                seed=0, cfg=None, data=None):
    dcfg = cfg["datasets"][dataset]
    lcfg = cfg["learner"]
    thr = lcfg["threshold"]
    df = data if data is not None else load_dataset(dataset, dcfg, seed)
    stream, test = split_stream_test(df, dcfg["test_from_step"])   # attacks only ever see the stream

    poisoned, mask = apply_attack_by_name(stream, attack, budget, timing, seed, dataset)
    poisoned = poisoned.copy()
    poisoned["is_poisoned"] = mask

    X_test, y_test = featurize(test, dataset), test["isFraud"].to_numpy()
    X_trig = None
    if attack == "backdoor":
        from src.attacks.api import make_triggered_test
        trig = make_triggered_test(test[test["isFraud"] == 1], dataset)
        X_trig = featurize(trig, dataset)

    learner = OnlineLearner(lcfg["fraud_weight"], seed)
    defend = get_defense(defense)
    state, run_id, rows, npois, n_seen = {}, uuid.uuid4().hex[:8], [], [], 0

    for k, win in replay_windows(poisoned, dcfg["window_steps"]):
        visible = win.drop(columns=["is_poisoned"])           # defenses never see ground truth
        accepted, quarantined, state = defend(visible, state)
        acc = win[win["record_id"].isin(accepted["record_id"])]
        learner.partial_fit(featurize(acc, dataset), acc["isFraud"].to_numpy())
        n_seen += len(win)
        npois.append(int(win["is_poisoned"].sum()))
        row = {"run_id": run_id, "dataset": dataset, "attack": attack, "budget": budget,
               "timing": timing, "defense": defense, "seed": seed, "window": k,
               "step_end": int(win["step"].max()), "n_seen": n_seen,
               "n_quarantined": len(quarantined)}
        row.update(eval_clean(learner, X_test, y_test, thr))
        row.update(eval_detection(win, quarantined) if defense != "off"
                   else {"det_precision": np.nan, "det_recall": np.nan, "det_fpr": np.nan})
        row["backdoor_hit_rate"] = (float((learner.predict_proba(X_trig) < thr).mean())
                                    if X_trig is not None and len(X_trig) else np.nan)
        rows.append(row)

    # time_to_detect: windows from first poisoned window to first window that quarantines poison
    ttd = np.nan
    pw = next((r["window"] for r, n in zip(rows, npois) if n > 0), None)
    if pw is not None and defense != "off":
        dw = next((r["window"] for r in rows if r["window"] >= pw and r["det_recall"] > 0), None)
        if dw is not None:
            ttd = dw - pw
    for r in rows:
        r["time_to_detect"] = ttd
    return pd.DataFrame(rows, columns=COLUMNS)


def append_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="paysim")
    ap.add_argument("--attack", default="none", choices=ATTACKS)
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--timing", default="uniform")
    ap.add_argument("--defense", default="off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="config/default.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = run_offline(a.dataset, a.attack, a.budget, a.timing, a.defense, a.seed, cfg)
    append_csv(out, cfg["results_csv"])
    print(out.tail(3).to_string(index=False))